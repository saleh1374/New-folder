"""Thin, faithful wrapper around the official DiCo-NLI scorer.

All headline numbers (``weighted_f1``, ``soft_cons``, ``hard_cons``) and the
per-language diagnostics are computed with ``evaluation_functions/`` from the
organizer repository. Nothing is re-implemented here, so local numbers match
CodaBench.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .labels import LABELS


def _scorer_module(data_root: Path) -> Any:
    root = Path(data_root)
    candidates = [root, root.parent]
    for candidate in candidates:
        if (candidate / "evaluation_functions" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return importlib.import_module("evaluation_functions.scorer")
    raise FileNotFoundError(
        f"evaluation_functions/ not found under {root} or {root.parent}; "
        "clone the organizer repository first"
    )


def write_predictions(path: Path, instance_ids: Iterable[str], labels: Iterable[str]) -> Path:
    """Write the official two-column submission format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(instance_ids)
    names = list(labels)
    if len(ids) != len(names):
        raise ValueError("instance_ids and labels must have the same length")
    for label in names:
        if label not in LABELS:
            raise ValueError(f"invalid label {label!r}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "label"])
        writer.writerows(zip(ids, names))
    return path


def score_predictions(
    data_root: Path,
    reference_path: Path,
    predictions_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the official scorer and return its report as a plain dict."""
    scorer = _scorer_module(data_root)
    report = scorer.score_files(
        reference_path=reference_path,
        prediction_path=predictions_path,
        output_dir=output_dir,
    )
    return _report_to_dict(report)


def _report_to_dict(report: Any) -> dict[str, Any]:
    payload = report.to_dict()
    payload["weighted_f1"] = float(payload["weighted_f1"])
    payload["soft_cons"] = float(payload["soft_cons"])
    payload["hard_cons"] = float(payload["hard_cons"])
    payload["macro_f1"] = float(payload["classification"]["macro_f1"])
    payload["accuracy"] = float(payload["classification"]["accuracy"])
    return payload


def per_language_report(
    data_root: Path,
    reference_path: Path,
    predictions_path: Path,
    lang_by_id: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Weighted/macro F1, accuracy and confusion matrix per language combination.

    Uses the organizer's ``compute_classification_report`` on filtered gold
    subsets, so the definition of F1 is identical to the official one.
    """
    _scorer_module(data_root)  # ensure sys.path
    io = importlib.import_module("evaluation_functions.io")
    metrics = importlib.import_module("evaluation_functions.metrics")
    validation = importlib.import_module("evaluation_functions.validation")

    gold_by_id = validation.validate_reference_records(io.read_reference_file(reference_path))
    predictions = validation.validate_prediction_records(
        io.read_prediction_file(predictions_path), gold_by_id
    )

    groups: dict[str, list[str]] = {}
    for instance_id in gold_by_id:
        groups.setdefault(lang_by_id.get(instance_id, "unknown"), []).append(instance_id)

    result: dict[str, dict[str, Any]] = {}
    for group, ids in sorted(groups.items()):
        subset = {i: gold_by_id[i] for i in ids}
        report = metrics.compute_classification_report(subset, predictions)
        result[group] = {
            "instances": len(ids),
            "weighted_f1": float(report.weighted_f1),
            "macro_f1": float(report.macro_f1),
            "accuracy": float(report.accuracy),
            "confusion_matrix": report.confusion_matrix,
        }
    return result


def summary_row(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "weighted_f1": round(float(report["weighted_f1"]), 4),
        "soft_cons": round(float(report["soft_cons"]), 4),
        "hard_cons": round(float(report["hard_cons"]), 4),
        "macro_f1": round(float(report["macro_f1"]), 4),
        "accuracy": round(float(report["accuracy"]), 4),
    }


REVERSAL = (0, 2, 1, 3)  # EQUIVALENCE, FORWARD, BACKWARD, NEGATIVE_OTHER


def base_instance_id(instance_id: str) -> str:
    """Strip the ``__original`` / ``__flipped`` suffix of a DiCo-NLI id."""
    for suffix in ("__original", "__flipped"):
        if instance_id.endswith(suffix):
            return instance_id[: -len(suffix)]
    return instance_id


def symmetric_average(
    instance_ids: list[str],
    probs: "np.ndarray",
) -> tuple["np.ndarray", int]:
    """Average each reversible pair with its reversed counterpart.

    ``p(A->B)`` and ``p(B->A)`` describe the same source pair, so the
    direction-flipped distribution of one is a second, independent estimate of
    the other. Only 2-member groups are touched: NEGATIVE_OTHER rows have no
    counterpart (their ``reverse_pair_id`` is empty) and stay as they are.

    Returns the adjusted probabilities and the number of repaired pairs.
    """
    import numpy as np

    groups: dict[str, list[int]] = {}
    for index, instance_id in enumerate(instance_ids):
        groups.setdefault(base_instance_id(instance_id), []).append(index)

    adjusted = np.array(probs, dtype="float64", copy=True)
    perm = list(REVERSAL)
    repaired = 0
    for indices in groups.values():
        if len(indices) != 2:
            continue
        a, b = indices
        adjusted[a] = 0.5 * (probs[a] + probs[b][perm])
        adjusted[b] = 0.5 * (probs[b] + probs[a][perm])
        repaired += 1
    adjusted = np.clip(adjusted, 1e-12, None)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, repaired


def format_table(rows: list[Mapping[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no results yet)"
    widths = {column: max(len(str(column)), max(len(str(row.get(column, ""))) for row in rows)) for column in columns}
    header = " | ".join(str(c).ljust(widths[c]) for c in columns)
    rule = "-+-".join("-" * widths[c] for c in columns)
    lines = [header, rule]
    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)
