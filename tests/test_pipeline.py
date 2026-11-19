"""End-to-end smoke test on the real CSVs with a tiny randomly initialised model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from darc_nli.config import ExperimentConfig  # noqa: E402
from darc_nli.data import audit_split, leakage_report, load_split  # noqa: E402
from darc_nli.trainer import Trainer  # noqa: E402


def find_repo() -> Path:
    candidates = [
        ROOT / "repo",
        ROOT / "data" / "repo",
        ROOT.parent / "repo",
    ]
    for candidate in candidates:
        if (candidate / "final_data" / "train").is_dir():
            return candidate
    pytest.skip("organizer repository data not available")


def test_audit_and_leakage():
    repo = find_repo()
    info = audit_split(repo, 1, "train")
    assert info["labels"]["FORWARD_ENTAILMENT"] > 0
    assert info["reverse_pairs"]["reversible_units"] > 0
    train = load_split(repo, 1, "train")
    dev = load_split(repo, 1, "dev")
    report = leakage_report(train, dev)
    assert report["split_leak_free"] is True
    assert report["pair_id_overlap"] == 0
    assert report["instance_id_overlap"] == 0
    # exact text repetition across different source pairs is a diagnostic, not
    # a split violation; it must never create contradictory labels silently
    assert report["text_overlap_label_conflicts"] == 0
    # every reversible unit holds a valid reversal pair
    for unit in train.units:
        if len(unit) == 2:
            a, b = (train.instances[i] for i in unit)
            assert b.reverse_pair_id == a.instance_id


@pytest.mark.parametrize("experiment", ["baseline", "ce_dc"])
def test_one_epoch_end_to_end(tmp_path, experiment):
    repo = find_repo()
    from transformers import AutoTokenizer, XLMRobertaConfig

    tiny = tmp_path / "tiny-xlmr"
    tiny.mkdir()
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer.save_pretrained(tiny)
    from transformers import XLMRobertaModel

    config = XLMRobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=514,
    )
    XLMRobertaModel(config).save_pretrained(tiny)

    cfg = ExperimentConfig(
        track=1,
        experiment=experiment,
        model_name=str(tiny),
        data_root=repo,
        output_dir=tmp_path / "outputs",
        epochs=1,
        batch_size=8,
        eval_batch_size=16,
        num_workers=0,
        max_train_samples=160,
        max_length=64,
        fp16=False,
        log_every=0,
        projection_dim=16,
    )
    train = load_split(repo, 1, "train")
    dev = load_split(repo, 1, "dev")
    trainer = Trainer(cfg, train, dev, device=torch.device("cpu"), max_length=64)
    summary = trainer.train()

    assert summary["best_epoch"] == 1
    assert (cfg.run_dir / "best" / "model.pt").is_file()
    assert (cfg.run_dir / "best" / "metadata.json").is_file()
    assert (cfg.results_dir / "dev_predictions_best.csv").is_file()
    scores = json.loads((cfg.results_dir / "best_scores.json").read_text(encoding="utf-8"))
    assert 0.0 <= scores["weighted_f1"] <= 1.0
    assert 0.0 <= scores["soft_cons"] <= 1.0
    assert 0.0 <= scores["hard_cons"] <= 1.0

    rows = (cfg.results_dir / "dev_predictions_best.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "instance_id,label"
    assert len(rows) == len(dev) + 1
