"""Label space of DiCo-NLI (DARC-NLI implementation).

The label order is fixed across the whole project because the Directional
Consistency loss depends on it (see ``REVERSE_PERMUTATION``). Never reorder.
"""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "EQUIVALENCE",
    "FORWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT",
    "NEGATIVE_OTHER",
)

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

EQUIVALENCE = LABEL2ID["EQUIVALENCE"]
FORWARD = LABEL2ID["FORWARD_ENTAILMENT"]
BACKWARD = LABEL2ID["BACKWARD_ENTAILMENT"]
NEGATIVE = LABEL2ID["NEGATIVE_OTHER"]

NUM_LABELS = len(LABELS)

#: Labels that have a deterministic reversal mapping.
REVERSIBLE_IDS: tuple[int, ...] = (EQUIVALENCE, FORWARD, BACKWARD)
REVERSIBLE_LABELS: tuple[str, ...] = tuple(ID2LABEL[i] for i in REVERSIBLE_IDS)

#: Reversal operator over the *three* reversible classes, expressed as a
#: permutation of the contiguous class order [EQUIVALENCE, FORWARD, BACKWARD].
#: EQUIVALENCE -> EQUIVALENCE, FORWARD -> BACKWARD, BACKWARD -> FORWARD.
REVERSE_PERMUTATION: tuple[int, int, int] = (0, 2, 1)

#: Full-label reversal map sampled by the official scorer. ``None`` means the
#: label has no deterministic reversal (NEGATIVE_OTHER).
_REVERSE_LABEL: dict[str, str | None] = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
    "NEGATIVE_OTHER": None,
}


def reverse_label(label: str) -> str | None:
    """Return the deterministic reversed label, or ``None`` if undefined."""
    return _REVERSE_LABEL[label]


def is_reversible(label_id: int) -> bool:
    return ID2LABEL[label_id] in ("EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")


def check_label_maps() -> None:
    """Fail fast if the label space or the reversal operator is inconsistent."""
    assert len(LABELS) == 4
    assert set(LABEL2ID) == set(ID2LABEL.values())
    for label in LABELS:
        rev = reverse_label(label)
        if rev is not None:
            assert reverse_label(rev) == label, f"reversal is not an involution for {label}"
    # The 3-class permutation must agree with the string-level map.
    for pos, label_id in enumerate(REVERSIBLE_IDS):
        rev = reverse_label(ID2LABEL[label_id])
        assert rev is not None
        assert REVERSE_PERMUTATION[pos] == REVERSIBLE_LABELS.index(rev)
