"""Training / evaluation loop for DARC-NLI v1."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .data import NliCollator, NliDataset, Split, split_paths, store_json
from .labels import ID2LABEL, LABELS, NUM_LABELS
from .losses import DarcLoss, build_dc_pairs
from .metrics import per_language_report, score_predictions, summary_row, write_predictions
from .model import DarcNliModel, build_tokenizer
from .sampler import PairAwareClassBalancedSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        train_split: Split,
        dev_split: Split,
        *,
        device: torch.device,
        max_length: int,
        patience: int = 2,
    ) -> None:
        self.cfg = cfg
        self.train_split = train_split
        self.dev_split = dev_split
        self.device = device
        self.max_length = max_length
        self.patience = patience

        set_seed(cfg.seed)

        self.tokenizer = build_tokenizer(cfg.model_name)
        self.model = DarcNliModel(
            cfg.model_name,
            projection_dim=cfg.projection_dim,
            dropout=cfg.dropout,
        ).to(device)

        self.collator = NliCollator(self.tokenizer, max_length)
        self.train_dataset = NliDataset(train_split)
        self.dev_dataset = NliDataset(dev_split)

        self.train_loader = self._build_train_loader()
        self.dev_loader = self._build_eval_loader(dev_split)

        self.loss_fn = DarcLoss(
            lambda_supcon=cfg.loss.lambda_supcon,
            lambda_dc=cfg.loss.lambda_dc,
            temperature=cfg.loss.temperature,
            label_smoothing=cfg.label_smoothing,
        ).to(device)

        self.frozen_parameters = self._freeze_lower_layers()
        self.optimizer = self._build_optimizer()
        total_steps = max(
            1,
            math.ceil(len(self.train_loader) * cfg.epochs / max(1, cfg.grad_accum)),
        )
        if cfg.max_train_steps:
            total_steps = min(total_steps, cfg.max_train_steps)
        self.total_steps = total_steps
        warmup = int(total_steps * cfg.warmup_ratio)
        if cfg.scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps,
            )
        elif cfg.scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"unknown scheduler {cfg.scheduler!r}")

        self.amp = bool(cfg.fp16 and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.history: list[dict[str, Any]] = []
        self.best_score = -1.0
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    # ------------------------------------------------------------------ #
    # loaders
    # ------------------------------------------------------------------ #
    def _build_train_loader(self) -> DataLoader:
        cfg = self.cfg
        if cfg.max_train_samples:
            # Slice whole units: cutting inside a reversible pair would either
            # break DC or create an artificial singleton.
            selected: list[int] = []
            for unit in self.train_split.units:
                if selected and len(selected) + len(unit) > cfg.max_train_samples:
                    break
                selected.extend(unit)
            self.train_split.instances = [self.train_split.instances[i] for i in selected]
            self.train_split.build_units()

        sampler = PairAwareClassBalancedSampler(
            units=self.train_split.units,
            instance_labels=[i.label_id for i in self.train_split.instances],
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.collator,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

    def _build_eval_loader(self, split: Split) -> DataLoader:
        return DataLoader(
            NliDataset(split),
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _encoder_lr_scale(self, name: str) -> float:
        """Layer-wise lr multiplier (0.95^depth is the usual choice)."""
        decay = self.cfg.layerwise_lr_decay
        if decay <= 0:
            return 1.0
        layers = getattr(getattr(self.model.encoder, "encoder", None), "layer", None)
        depth = len(layers) if layers is not None else 0
        if name.startswith("encoder.encoder.layer."):
            index = int(name.split(".")[3])
            # the last layer keeps lr, the first one gets decay^(depth-1)
            return decay ** (depth - index)
        if name.startswith("encoder."):
            return decay ** (depth + 1)  # embeddings
        return 1.0  # classification / projection heads

    def _freeze_lower_layers(self) -> int:
        count = int(self.cfg.freeze_layers)
        if count <= 0:
            return 0
        frozen = 0
        for name, param in self.model.named_parameters():
            if name.startswith("encoder.embeddings."):
                param.requires_grad = False
                frozen += 1
                continue
            if name.startswith("encoder.encoder.layer."):
                index = int(name.split(".")[3])
                if index < count:
                    param.requires_grad = False
                    frozen += 1
        return frozen

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        head_lr = cfg.head_lr if cfg.head_lr > 0 else cfg.lr

        buckets: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith(("classifier.", "projection."))
            lr = head_lr if is_head else cfg.lr * self._encoder_lr_scale(name)
            wd = 0.0 if any(key in name for key in no_decay) else cfg.weight_decay
            buckets.setdefault((round(lr, 12), wd), []).append(param)

        groups = [
            {"params": params, "lr": lr, "weight_decay": wd}
            for (lr, wd), params in buckets.items()
        ]
        return torch.optim.AdamW(groups, lr=cfg.lr)

    def _optimizer_step(self) -> None:
        """Clip, step and advance the schedule; skip the schedule on inf/nan.

        ``GradScaler`` silently skips ``optimizer.step()`` when it finds
        non-finite gradients. Advancing the LR schedule anyway would desync it
        from the real number of updates, so it is advanced only when the
        optimizer actually moved (detected by the un-changed scale factor).
        """
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() >= scale_before:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        store_json(cfg.run_dir / "config.json", cfg.to_dict())

        print(
            f"[{cfg.track_name}/{cfg.run_name}] train={len(self.train_split)} "
            f"dev={len(self.dev_split)} batches/epoch={len(self.train_loader)} "
            f"device={self.device.type} amp={self.amp} max_length={self.max_length}"
        )

        for epoch in range(1, cfg.epochs + 1):
            started = time.time()
            train_stats = self.train_epoch(epoch)
            eval_result = self.evaluate(self.dev_split, tag=f"epoch{epoch}")
            elapsed = time.time() - started
            scores = summary_row(eval_result["report"])
            record = {
                "epoch": epoch,
                "seconds": round(elapsed, 1),
                "train": train_stats,
                "dev": scores,
            }
            self.history.append(record)
            print(
                f"  epoch {epoch}: "
                + " ".join(f"{key}={value}" for key, value in scores.items())
                + f" | ce={train_stats['ce']:.4f} scl={train_stats['supcon']:.4f} "
                f"dc={train_stats['dc']:.4f}"
            )
            store_json(cfg.run_dir / "training_log.json", self.history)

            metric_value = float(scores[cfg.select_metric])
            if metric_value > self.best_score + 1e-5:
                self.best_score = metric_value
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_best(eval_result, epoch)
            else:
                self.epochs_without_improvement += 1

            if self.patience and self.epochs_without_improvement >= self.patience:
                print(f"  early stopping at epoch {epoch} (best epoch {self.best_epoch})")
                break

        summary = {
            "track": cfg.track,
            "experiment": cfg.run_name,
            "select_metric": cfg.select_metric,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "history": self.history,
        }
        store_json(cfg.run_dir / "summary.json", summary)
        return summary

    def train_epoch(self, epoch: int) -> dict[str, float]:
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        totals: dict[str, float] = defaultdict(float)
        seen = 0
        pair_batches = 0
        step = 0
        optimizer_steps = 0

        started = time.time()
        for batch_index, batch in enumerate(self.train_loader, start=1):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            pair_a, pair_b = build_dc_pairs(batch["instance_ids"], batch["reverse_pair_ids"])
            pair_a = pair_a.to(self.device)
            pair_b = pair_b.to(self.device)
            if pair_a.numel():
                pair_batches += 1

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits, projections = self.model(input_ids, attention_mask)
                logits = logits.float()
                projections = projections.float()
                loss_out = self.loss_fn(logits, projections, labels, pair_a, pair_b)

            scaled = loss_out.total / cfg.grad_accum
            self.scaler.scale(scaled).backward()

            batch_size = labels.size(0)
            totals["ce"] += float(loss_out.ce) * batch_size
            totals["supcon"] += float(loss_out.supcon) * batch_size
            totals["dc"] += float(loss_out.dc) * batch_size
            totals["total"] += float(loss_out.total.detach()) * batch_size
            seen += batch_size
            step += 1

            if batch_index % cfg.grad_accum == 0:
                self._optimizer_step()
                optimizer_steps += 1
                if cfg.max_train_steps and optimizer_steps >= cfg.max_train_steps:
                    break

            if cfg.log_every and batch_index % cfg.log_every == 0:
                rate = seen / max(1e-6, time.time() - started)
                print(
                    f"    step {batch_index}/{len(self.train_loader)} "
                    f"loss={totals['total'] / seen:.4f} "
                    f"ce={totals['ce'] / seen:.4f} "
                    f"scl={totals['supcon'] / seen:.4f} "
                    f"dc={totals['dc'] / seen:.4f} "
                    f"({rate:.1f} inst/s)"
                )

        # flush a trailing partial accumulation window
        if step % cfg.grad_accum != 0 and not (cfg.max_train_steps and optimizer_steps >= cfg.max_train_steps):
            self._optimizer_step()
            optimizer_steps += 1

        stats = {key: value / max(1, seen) for key, value in totals.items()}
        stats["instances"] = seen
        stats["batches_with_reverse_pairs"] = pair_batches
        stats["batches"] = step
        stats["optimizer_steps"] = optimizer_steps
        return stats

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, split: Split, *, tag: str) -> dict[str, Any]:
        self.model.eval()
        all_ids: list[str] = []
        all_langs: dict[str, str] = {}
        all_probs: list[np.ndarray] = []

        for batch in self._build_eval_loader(split):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_ids.extend(batch["instance_ids"])
            all_langs.update(dict(zip(batch["instance_ids"], batch["lang_pairs"])))

        probs = np.concatenate(all_probs, axis=0)

        cfg = self.cfg
        reference_path = split_paths(cfg.data_root, split.track, split.split)["reference"]

        variants: dict[str, np.ndarray] = {"raw": probs}
        repaired = 0
        if cfg.pair_consistency:
            variants["symmetric"], repaired = symmetric_average(all_ids, probs)

        scored: dict[str, Any] = {}
        for name, matrix in variants.items():
            labels = [ID2LABEL[int(i)] for i in matrix.argmax(axis=1)]
            predictions_path = cfg.results_dir / f"{split.split}_predictions_{tag}_{name}.csv"
            write_predictions(predictions_path, all_ids, labels)
            score_dir = cfg.results_dir / f"{split.split}_scores_{tag}_{name}"
            report = score_predictions(cfg.data_root, reference_path, predictions_path, score_dir)
            languages = per_language_report(
                cfg.data_root, reference_path, predictions_path, all_langs
            )
            store_json(
                score_dir / "detailed_report.json",
                {"report": report, "per_language": languages},
            )
            scored[name] = {
                "report": report,
                "per_language": languages,
                "predictions_path": str(predictions_path),
            }

        chosen = "symmetric" if cfg.pair_consistency else "raw"
        best = scored[chosen]
        return {
            "report": best["report"],
            "predictions_path": best["predictions_path"],
            "per_language": best["per_language"],
            "variants": {name: summary_row(item["report"]) for name, item in scored.items()},
            "repaired_pairs": repaired,
            "probabilities": probs,
            "instance_ids": all_ids,
        }
        }

    def _save_best(self, eval_result: dict[str, Any], epoch: int) -> None:
        cfg = self.cfg
        best_dir = cfg.run_dir / "best"
        metadata = {
            "track": cfg.track,
            "track_name": cfg.track_name,
            "experiment": cfg.run_name,
            "epoch": epoch,
            "seed": cfg.seed,
            "model_name": cfg.model_name,
            "max_length": self.max_length,
            "lambda_supcon": cfg.loss.lambda_supcon,
            "lambda_dc": cfg.loss.lambda_dc,
            "temperature": cfg.loss.temperature,
            "pair_consistency": cfg.pair_consistency,
            "repaired_pairs": eval_result.get("repaired_pairs", 0),
            "dev_variants": eval_result.get("variants", {}),
            "label2id": {label: idx for idx, label in enumerate(LABELS)},
            "dev_scores": summary_row(eval_result["report"]),
            "dev_per_language": eval_result["per_language"],
            "training_history": self.history,
        }
        self.model.save_checkpoint(best_dir, tokenizer=self.tokenizer, metadata=metadata)

        best_predictions = cfg.results_dir / "dev_predictions_best.csv"
        best_predictions.write_text(
            Path(eval_result["predictions_path"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        store_json(cfg.results_dir / "best_scores.json", meta_summary(metadata))
        print(f"    saved new best checkpoint (epoch {epoch}, {cfg.select_metric}={self.best_score:.4f})")


def meta_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": metadata["track"],
        "experiment": metadata["experiment"],
        "epoch": metadata["epoch"],
        "seed": metadata["seed"],
        **metadata["dev_scores"],
    }


def load_run_summary(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
