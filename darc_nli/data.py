"""Data loading, phase-0 audit, reverse-pair indexing and torch Datasets.

The real files released by the organizers are the single source of truth: this
module never assumes a column name or an id format beyond what it reads from
the CSVs (see ``audit_split``), and it validates every assumption it relies on.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .labels import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    NEGATIVE,
    NUM_LABELS,
    is_reversible,
    reverse_label,
)

TRACK_CODES: dict[int, str] = {1: "en", 2: "es", 3: "eu", 4: "mixed"}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    instance_id: str
    pair_id: str
    reverse_pair_id: str | None
    text1: str
    text2: str
    text1_lang: str
    text2_lang: str
    label_id: int

    @property
    def label(self) -> str:
        return ID2LABEL[self.label_id]

    @property
    def lang_pair(self) -> str:
        return f"{self.text1_lang}-{self.text2_lang}"

    @property
    def reversible(self) -> bool:
        return is_reversible(self.label_id)


@dataclass
class Split:
    track: int
    split: str
    instances: list[Instance]
    units: list[tuple[int, ...]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.instances)

    @property
    def ids(self) -> list[str]:
        return [inst.instance_id for inst in self.instances]

    def label_counts(self) -> Counter:
        return Counter(inst.label for inst in self.instances)

    def build_units(self) -> list[tuple[int, ...]]:
        """Group instances into reversible units ``(a, rev_a)`` and singletons.

        A unit is a tuple of dataset indices. Reversible units contain exactly
        the two members of an organizer-provided ``reverse_pair_id`` link; every
        other instance (including ``NEGATIVE_OTHER``) is a singleton unit.
        """
        index_of = {inst.instance_id: i for i, inst in enumerate(self.instances)}
        units: list[tuple[int, ...]] = []
        used: set[int] = set()
        for i, inst in enumerate(self.instances):
            if i in used:
                continue
            rev = inst.reverse_pair_id
            if rev and rev in index_of and rev != inst.instance_id:
                j = index_of[rev]
                units.append((min(i, j), max(i, j)))
                used.update({i, j})
            else:
                units.append((i,))
                used.add(i)
        self.units = units
        return units


# --------------------------------------------------------------------------- #
# Paths and reading
# --------------------------------------------------------------------------- #
def split_paths(data_root: Path, track: int, split: str) -> dict[str, Path]:
    base = Path(data_root) / "final_data" / split
    stem = f"dico_nli_{split}_track{track}"
    return {
        "labeled": base / f"{stem}_participant_labeled.csv",
        "reference": base / f"{stem}_reference.csv",
        "template": base / f"{stem}_submission_template.csv",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def load_split(data_root: Path, track: int, split: str) -> Split:
    """Load one track split and attach organizer-provided reverse-pair links.

    Labels and texts come from ``*_participant_labeled.csv`` (the file meant for
    training). ``reverse_pair_id`` only exists in ``*_reference.csv``, so the two
    files are joined on ``instance_id`` and every join is validated.
    """
    paths = split_paths(data_root, track, split)
    for key in ("labeled", "reference"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"missing {split} file: {paths[key]}")

    labeled = _read_csv(paths["labeled"])
    reference = _read_csv(paths["reference"])
    ref_by_id = {r["instance_id"]: r for r in reference}

    missing = [r["instance_id"] for r in labeled if r["instance_id"] not in ref_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} labeled instances are absent from the reference file, "
            f"e.g. {missing[:3]}"
        )

    instances: list[Instance] = []
    seen: set[str] = set()
    for row in labeled:
        iid = row["instance_id"]
        if iid in seen:
            raise ValueError(f"duplicate instance_id in {paths['labeled']}: {iid}")
        seen.add(iid)
        if row["label"] not in LABEL2ID:
            raise ValueError(f"unknown label {row['label']!r} for {iid}")
        ref = ref_by_id[iid]
        if ref["label"] != row["label"]:
            raise ValueError(
                f"label mismatch between labeled and reference file for {iid}: "
                f"{row['label']} vs {ref['label']}"
            )
        rev = ref.get("reverse_pair_id") or None
        instances.append(
            Instance(
                instance_id=iid,
                pair_id=row.get("pair_id", ref.get("pair_id", "")),
                reverse_pair_id=rev,
                text1=row["text1"],
                text2=row["text2"],
                text1_lang=row.get("text1_lang", "") or ref.get("text1_lang", ""),
                text2_lang=row.get("text2_lang", "") or ref.get("text2_lang", ""),
                label_id=LABEL2ID[row["label"]],
            )
        )

    split_obj = Split(track=track, split=split, instances=instances)
    split_obj.build_units()
    _validate_units(split_obj)
    return split_obj


def _validate_units(split: Split) -> None:
    """Reverse links must be reciprocal and label-consistent."""
    index_of = {inst.instance_id: i for i, inst in enumerate(split.instances)}
    for inst in split.instances:
        rev = inst.reverse_pair_id
        if not rev:
            continue
        if rev not in index_of:
            raise ValueError(f"{inst.instance_id}: reverse id {rev} not in this split")
        other = split.instances[index_of[rev]]
        if other.reverse_pair_id != inst.instance_id:
            raise ValueError(
                f"reverse link is not reciprocal: {inst.instance_id} -> {rev} -> "
                f"{other.reverse_pair_id}"
            )
        expected = reverse_label(inst.label)
        if expected is None or other.label != expected:
            raise ValueError(
                f"reversal label mismatch for {inst.instance_id}: "
                f"{inst.label} -> expected {expected}, found {other.label}"
            )


# --------------------------------------------------------------------------- #
# Phase 0: audit
# --------------------------------------------------------------------------- #
def _percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def audit_split(
    data_root: Path,
    track: int,
    split: str,
    *,
    tokenizer: Any | None = None,
    max_length: int = 512,
) -> dict[str, Any]:
    paths = split_paths(data_root, track, split)
    raw = {name: _read_csv(path) for name, path in paths.items()}
    labeled = raw["labeled"]
    reference = raw["reference"]
    template = raw["template"]

    data = load_split(data_root, track, split)

    total = len(labeled)
    nulls = {
        col: sum(1 for row in labeled if not row.get(col))
        for col in labeled[0].keys()
    }
    dup_ids = total - len({row["instance_id"] for row in labeled})
    dup_texts = total - len({(row["text1"], row["text2"]) for row in labeled})
    empty_texts = sum(
        1 for row in labeled if not row["text1"].strip() or not row["text2"].strip()
    )

    labels = Counter(inst.label for inst in data.instances)
    lang_pairs = Counter(inst.lang_pair for inst in data.instances)

    rev_links = [inst for inst in data.instances if inst.reverse_pair_id]
    rev_units = [u for u in data.units if len(u) == 2]
    self_links = sum(1 for i in rev_links if i.reverse_pair_id == i.instance_id)

    char_lens = [len(inst.text1) + len(inst.text2) for inst in data.instances]
    token_lens: list[int] = []
    if tokenizer is not None:
        token_lens = [
            len(tokenizer(inst.text1, inst.text2, truncation=False)["input_ids"])
            for inst in data.instances[: min(len(data.instances), 5000)]
        ]

    recommended = max(48, int(_percentile(token_lens, 0.99) + 8) if token_lens else 128)
    recommended = min(recommended, max_length)

    return {
        "track": track,
        "split": split,
        "files": {
            "labeled": {"path": str(paths["labeled"]), "rows": len(labeled), "columns": list(labeled[0])},
            "reference": {
                "path": str(paths["reference"]),
                "rows": len(reference),
                "columns": list(reference[0]),
            },
            "submission_template": {
                "path": str(paths["template"]),
                "rows": len(template),
                "columns": list(template[0]),
            },
        },
        "id_sets": {
            "labeled_equals_reference": {r["instance_id"] for r in labeled}
            == {r["instance_id"] for r in reference},
            "instance_ids": total,
        },
        "quality": {
            "empty_cells": nulls,
            "duplicate_instance_ids": dup_ids,
            "duplicate_text_pairs": dup_texts,
            "empty_texts": empty_texts,
        },
        "labels": {label: labels.get(label, 0) for label in LABELS},
        "language_pairs": dict(sorted(lang_pairs.items())),
        "reverse_pairs": {
            "linked_instances": len(rev_links),
            "reversible_units": len(rev_units),
            "singleton_units": len(data.units) - len(rev_units),
            "self_links": self_links,
            "label_mapping_ok": True,  # enforced by _validate_units()
        },
        "lengths": {
            "chars_total": {
                "mean": round(sum(char_lens) / len(char_lens), 2),
                "p50": _percentile(char_lens, 0.50),
                "p95": _percentile(char_lens, 0.95),
                "p99": _percentile(char_lens, 0.99),
                "max": max(char_lens),
            },
            "tokens": (
                {
                    "mean": round(sum(token_lens) / len(token_lens), 2),
                    "p95": _percentile(token_lens, 0.95),
                    "p99": _percentile(token_lens, 0.99),
                    "max": max(token_lens),
                }
                if token_lens
                else "pass a tokenizer to measure token lengths"
            ),
            "recommended_max_length": recommended,
        },
    }


def leakage_report(train: Split, dev: Split) -> dict[str, Any]:
    """Check that no source pair or direction crosses the train/dev boundary.

    ``pair_id``/``instance_id`` overlap is the official leakage criterion: the
    organizers guarantee it is zero and we never rebuild the split. Exact text
    repetition across different source pairs can still occur (PhrasIS reuses
    phrases), so it is reported separately as a diagnostic, together with any
    label conflict it creates.
    """
    train_ids = set(train.ids)
    dev_ids = set(dev.ids)
    train_pairs = {inst.pair_id for inst in train.instances}
    dev_pairs = {inst.pair_id for inst in dev.instances}

    def norm(text: str) -> str:
        return " ".join(text.lower().split())

    train_texts = {(norm(i.text1), norm(i.text2)): i.label for i in train.instances}
    dev_texts = {(norm(i.text1), norm(i.text2)) for i in dev.instances}
    train_texts_rev = {(b, a): v for (a, b), v in train_texts.items()}

    overlap_instances = 0
    label_conflicts = 0
    examples: list[dict[str, str]] = []
    for inst in dev.instances:
        key = (norm(inst.text1), norm(inst.text2))
        for table, direction in ((train_texts, "same"), (train_texts_rev, "reversed")):
            if key in table:
                overlap_instances += 1
                if table[key] != inst.label:
                    label_conflicts += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "instance_id": inst.instance_id,
                            "text1": inst.text1,
                            "text2": inst.text2,
                            "dev_label": inst.label,
                            "train_label": table[key],
                            "direction": direction,
                        }
                    )
                break

    report = {
        "instance_id_overlap": len(train_ids & dev_ids),
        "pair_id_overlap": len(train_pairs & dev_pairs),
        "text_pair_overlap": len(dev_texts & set(train_texts)),
        "reversed_text_overlap": len(dev_texts & set(train_texts_rev)),
        "text_overlap_instances": overlap_instances,
        "text_overlap_label_conflicts": label_conflicts,
        "text_overlap_examples": examples,
        "train_pair_ids": len(train_pairs),
        "dev_pair_ids": len(dev_pairs),
    }
    report["split_leak_free"] = (
        report["instance_id_overlap"] == 0 and report["pair_id_overlap"] == 0
    )
    return report


def print_audit(audit: dict[str, Any], leakage: dict[str, Any] | None = None) -> None:
    print(f"\n=== AUDIT track {audit['track']} / {audit['split']} ===")
    for name, info in audit["files"].items():
        print(f"  {name:<20} rows={info['rows']:<6} columns={info['columns']}")
    print(f"  labels              {audit['labels']}")
    print(f"  language pairs      {audit['language_pairs']}")
    print(f"  reverse pairs       {audit['reverse_pairs']}")
    print(f"  duplicate texts     {audit['quality']['duplicate_text_pairs']}")
    print(f"  lengths (chars)     {audit['lengths']['chars_total']}")
    print(f"  max_length advice   {audit['lengths']['recommended_max_length']}")
    if leakage is not None:
        print(
            f"  split leak-free     {leakage['split_leak_free']} "
            f"(instance_id overlap={leakage['instance_id_overlap']}, "
            f"pair_id overlap={leakage['pair_id_overlap']})"
        )
        print(
            f"  text duplication    {leakage['text_overlap_instances']} dev instances, "
            f"{leakage['text_overlap_label_conflicts']} label conflicts (diagnostic only)"
        )
        if not leakage["split_leak_free"]:
            print("  WARNING: official split overlap detected - do not train before fixing it")


# --------------------------------------------------------------------------- #
# Torch dataset / collator
# --------------------------------------------------------------------------- #
class NliDataset(Dataset):
    """Returns raw texts plus ids; tokenization happens in the collator."""

    def __init__(self, split: Split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        inst = self.split.instances[index]
        return {
            "instance_id": inst.instance_id,
            "reverse_pair_id": inst.reverse_pair_id or "",
            "text1": inst.text1,
            "text2": inst.text2,
            "lang_pair": inst.lang_pair,
            "label": inst.label_id,
        }


class NliCollator:
    def __init__(self, tokenizer: Any, max_length: int, *, with_labels: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [f["text1"] for f in features],
            [f["text2"] for f in features],
            truncation="longest_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "instance_ids": [f["instance_id"] for f in features],
            "reverse_pair_ids": [f["reverse_pair_id"] for f in features],
            "lang_pairs": [f["lang_pair"] for f in features],
        }
        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded["token_type_ids"]
        if self.with_labels:
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch


# --------------------------------------------------------------------------- #
# Batching helpers shared by train and evaluation
# --------------------------------------------------------------------------- #
def store_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def describe_split(split: Split) -> str:
    counts = split.label_counts()
    return f"track{split.track} {split.split}: {len(split)} instances, " + ", ".join(
        f"{label}={counts.get(label, 0)}" for label in LABELS
    )
