"""Command line entry points.

Examples
--------
Phase 0 audit + leakage check for every track::

    python -m darc_nli.run audit --data-root data/repo

Train one experiment (English, CE only)::

    python -m darc_nli.run train --track 1 --experiment baseline --epochs 3

Train the full 4x4 matrix::

    python -m darc_nli.run train --track all --experiment all --epochs 3

Write a submission file with the best checkpoint of a run::

    python -m darc_nli.run predict --checkpoint outputs/checkpoints/english/baseline/best \
        --input data/repo/final_data/dev/dico_nli_dev_track1_participant_labeled.csv \
        --output submissions/dev_track1.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import EXPERIMENT_NAMES, TRACK_NAMES, ExperimentConfig
from .data import (
    Instance,
    NliCollator,
    NliDataset,
    Split,
    audit_split,
    leakage_report,
    load_split,
    print_audit,
    split_paths,
    store_json,
)
from .labels import ID2LABEL, LABEL2ID
from .metrics import format_table, summary_row, symmetric_average, write_predictions
from .model import DarcNliModel, build_tokenizer
from .trainer import Trainer, pick_device


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_tracks(value: str) -> list[int]:
    if value == "all":
        return [1, 2, 3, 4]
    track = int(value)
    if track not in TRACK_NAMES:
        raise SystemExit(f"track must be one of 1-4 or 'all', got {value!r}")
    return [track]


def _parse_experiments(value: str) -> list[str]:
    if value == "all":
        return list(EXPERIMENT_NAMES)
    names = [name.strip() for name in value.split(",") if name.strip()]
    for name in names:
        if name not in EXPERIMENT_NAMES:
            raise SystemExit(f"unknown experiment {name!r}; choose from {EXPERIMENT_NAMES}")
    return names


def _recommend_max_length(cfg: ExperimentConfig, tokenizer: Any) -> int:
    if cfg.max_length:
        return cfg.max_length
    train = load_split(cfg.data_root, cfg.track, "train")
    lengths = [
        len(tokenizer(inst.text1, inst.text2, truncation=False)["input_ids"])
        for inst in train.instances
    ]
    lengths.sort()
    p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
    # 64 is a safe floor for this corpus (the technical document's audit shows
    # p99 <= 90 characters per pair); never blindly go to 512.
    return int(min(512, max(64, p99 + 8)))


def _cleanup(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def cmd_audit(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig(track=1, data_root=Path(args.data_root))
    tokenizer = None
    if args.tokenizer:
        tokenizer = build_tokenizer(args.tokenizer)
        print(f"tokenizer: {args.tokenizer}")

    report: dict[str, Any] = {}
    for track in _parse_tracks(args.track):
        audits = {}
        for split in ("train", "dev"):
            info = audit_split(cfg.data_root, track, split, tokenizer=tokenizer)
            audits[split] = info
            print_audit(info)
        leakage = leakage_report(
            load_split(cfg.data_root, track, "train"),
            load_split(cfg.data_root, track, "dev"),
        )
        print(f"\n=== LEAKAGE track {track} ===")
        for key, value in leakage.items():
            print(f"  {key:<24} {value}")
        if not leakage["split_leak_free"]:
            print("  WARNING: split overlap detected")
        report[str(track)] = {"splits": audits, "leakage": leakage}

    out = Path(args.out or cfg.output_dir / "audit_report.json")
    store_json(out, report)
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def run_one(track: int, experiment: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = ExperimentConfig(
        track=track,
        experiment=experiment,
        model_name=args.model_name,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        head_lr=args.head_lr,
        scheduler=args.scheduler,
        label_smoothing=args.label_smoothing,
        layerwise_lr_decay=args.layerwise_lr_decay,
        freeze_layers=args.freeze_layers,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        max_length=args.max_length,
        fp16=not args.no_fp16,
        max_train_steps=args.max_train_steps,
        max_train_samples=args.max_train_samples,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        select_metric=args.select_metric,
        pair_consistency=args.pair_consistency,
        projection_dim=args.projection_dim,
    )
    if args.lambda_supcon is not None:
        cfg.loss.lambda_supcon = args.lambda_supcon
    if args.lambda_dc is not None:
        cfg.loss.lambda_dc = args.lambda_dc
    if args.temperature is not None:
        cfg.loss.temperature = args.temperature

    device = pick_device(args.device)
    train_split = load_split(cfg.data_root, track, "train")
    dev_split = load_split(cfg.data_root, track, "dev")

    max_length = _recommend_max_length(cfg, build_tokenizer(cfg.model_name))

    trainer = Trainer(
        cfg,
        train_split,
        dev_split,
        device=device,
        max_length=max_length if not args.max_length else args.max_length,
        patience=args.patience,
    )
    summary = trainer.train()
    best = trainer.best_score
    report_path = cfg.results_dir / "best_scores.json"
    if report_path.is_file():
        summary["best_scores"] = json.loads(report_path.read_text(encoding="utf-8"))

    _cleanup(trainer, train_split, dev_split)
    return summary


def cmd_train(args: argparse.Namespace) -> None:
    tracks = _parse_tracks(args.track)
    experiments = _parse_experiments(args.experiment)
    results = []
    for track in tracks:
        for experiment in experiments:
            summary = run_one(track, experiment, args)
            results.append(summary)
    print("\nfinished:")
    for result in results:
        print(
            f"  track{result['track']}/{result['experiment']}: "
            f"best_epoch={result['best_epoch']} best_{result['select_metric']}={result['best_score']:.4f}"
        )
    if args.summarize:
        cmd_summarize(argparse.Namespace(output_dir=args.output_dir))


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def _read_text_pairs(path: Path) -> list[Instance]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for required in ("instance_id", "text1", "text2"):
        if rows and required not in rows[0]:
            raise SystemExit(f"{path} has no {required!r} column")
    instances = []
    for row in rows:
        label = (row.get("label") or "EQUIVALENCE").strip()
        instances.append(
            Instance(
                instance_id=row["instance_id"],
                pair_id=row.get("pair_id", ""),
                reverse_pair_id=None,
                text1=row["text1"],
                text2=row["text2"],
                text1_lang=row.get("text1_lang", ""),
                text2_lang=row.get("text2_lang", ""),
                label_id=LABEL2ID.get(label, 0),
            )
        )
    return instances


@torch.no_grad()
def cmd_predict(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    device = pick_device(args.device)

    tokenizer = build_tokenizer(str(checkpoint))
    model = DarcNliModel.load_checkpoint(checkpoint).to(device)
    model.eval()

    instances = _read_text_pairs(Path(args.input))
    split = Split(track=int(metadata["track"]), split="predict", instances=instances)
    collator = NliCollator(tokenizer, int(metadata["max_length"]), with_labels=False)
    loader = DataLoader(
        NliDataset(split),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    ids: list[str] = []
    chunks: list["np.ndarray"] = []
    for batch in loader:
        logits, _ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        ids.extend(batch["instance_ids"])

    import numpy as np

    probs = np.concatenate(chunks, axis=0)
    if metadata.get("pair_consistency"):
        probs, repaired = symmetric_average(ids, probs)
        print(f"pair consistency applied to {repaired} reversible pairs")

    labels = [ID2LABEL[int(i)] for i in probs.argmax(axis=1)]
    write_predictions(Path(args.output), ids, labels)
    print(f"wrote {len(ids)} predictions to {args.output}")


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def cmd_summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    results_root = output_dir / "results"
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*/*/best_scores.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": payload["experiment"],
                "track": TRACK_NAMES.get(int(payload["track"]), str(payload["track"])),
                **{key: payload.get(key) for key in ("weighted_f1", "soft_cons", "hard_cons", "macro_f1")},
            }
        )

    if not rows:
        print(f"no results under {results_root}")
        return

    order = {name: i for i, name in enumerate(EXPERIMENT_NAMES)}
    rows.sort(key=lambda row: (row["track"], order.get(row["experiment"], 99)))
    table = format_table(
        rows,
        ["track", "experiment", "weighted_f1", "soft_cons", "hard_cons", "macro_f1"],
    )
    print("\n" + table)

    header = ["track", "experiment", "weighted_f1", "soft_cons", "hard_cons", "macro_f1"]
    out_csv = output_dir / "summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})

    markdown = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        markdown.append("| " + " | ".join(str(row.get(key, "")) for key in header) + " |")
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"\nwrote {out_csv} and {output_dir / 'summary.md'}")


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="darc_nli")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="phase 0: inspect data, reverse pairs and leakage")
    audit.add_argument("--data-root", default="data/repo")
    audit.add_argument("--track", default="all")
    audit.add_argument("--tokenizer", default="xlm-roberta-base")
    audit.add_argument("--out", default=None)
    audit.set_defaults(func=cmd_audit)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default="data/repo")
    common.add_argument("--output-dir", default="outputs")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--epochs", type=int, default=5)
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--grad-accum", type=int, default=1)
    common.add_argument("--lr", type=float, default=2e-5)
    common.add_argument("--head-lr", type=float, default=0.0, help="0 = same as --lr")
    common.add_argument("--scheduler", default="linear", choices=["linear", "cosine"])
    common.add_argument("--label-smoothing", type=float, default=0.0)
    common.add_argument("--layerwise-lr-decay", type=float, default=0.0)
    common.add_argument("--freeze-layers", type=int, default=0)
    common.add_argument("--model-name", default="xlm-roberta-base")
    common.add_argument("--weight-decay", type=float, default=0.01)
    common.add_argument("--warmup-ratio", type=float, default=0.06)
    common.add_argument("--dropout", type=float, default=0.1)
    common.add_argument("--max-length", type=int, default=0, help="0 = decided by the data audit")
    common.add_argument("--projection-dim", type=int, default=256)
    common.add_argument("--eval-batch-size", type=int, default=64)
    common.add_argument("--num-workers", type=int, default=2)
    common.add_argument("--patience", type=int, default=2, help="0 disables early stopping")
    common.add_argument("--select-metric", default="weighted_f1",
                        choices=["weighted_f1", "soft_cons", "hard_cons"])
    common.add_argument("--pair-consistency", action="store_true",
                        help="average both directions of each reversible pair before argmax")
    common.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    common.add_argument("--no-fp16", action="store_true")
    common.add_argument("--max-train-steps", type=int, default=0)
    common.add_argument("--max-train-samples", type=int, default=0)
    common.add_argument("--lambda-supcon", type=float, default=None)
    common.add_argument("--lambda-dc", type=float, default=None)
    common.add_argument("--temperature", type=float, default=None)

    train = sub.add_parser("train", parents=[common], help="train one or more experiments")
    train.add_argument("--track", default="all")
    train.add_argument("--experiment", default="all")
    train.add_argument("--summarize", action="store_true")
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="write a submission file from a checkpoint")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--batch-size", type=int, default=64)
    predict.add_argument("--device", default=None)
    predict.set_defaults(func=cmd_predict)

    summarize = sub.add_parser("summarize", help="build the results table")
    summarize.add_argument("--output-dir", default="outputs")
    summarize.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
