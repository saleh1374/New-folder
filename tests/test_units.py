"""Unit tests for the parts that are easy to get subtly wrong.

Run from the project root::

    python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from darc_nli.labels import (  # noqa: E402
    BACKWARD,
    EQUIVALENCE,
    FORWARD,
    LABELS,
    NEGATIVE,
    REVERSE_PERMUTATION,
    check_label_maps,
    reverse_label,
)
from darc_nli.losses import (  # noqa: E402
    DirectionalConsistencyLoss,
    SupConLoss,
    build_dc_pairs,
    class_weights_from_counts,
)
from darc_nli.sampler import PairAwareClassBalancedSampler  # noqa: E402


def test_label_space_and_reversal():
    check_label_maps()
    assert LABELS == (
        "EQUIVALENCE",
        "FORWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT",
        "NEGATIVE_OTHER",
    )
    assert reverse_label("EQUIVALENCE") == "EQUIVALENCE"
    assert reverse_label("FORWARD_ENTAILMENT") == "BACKWARD_ENTAILMENT"
    assert reverse_label("BACKWARD_ENTAILMENT") == "FORWARD_ENTAILMENT"
    assert reverse_label("NEGATIVE_OTHER") is None
    assert REVERSE_PERMUTATION == (0, 2, 1)


def test_symmetric_average_is_identity_for_consistent_pairs():
    import numpy as np

    ids = ["x__original", "x__flipped", "n1"]
    probs = np.array([
        [0.05, 0.90, 0.03, 0.02],   # x -> y : FORWARD
        [0.05, 0.03, 0.90, 0.02],   # y -> x : BACKWARD (consistent)
        [0.10, 0.10, 0.10, 0.70],   # NEGATIVE_OTHER singleton
    ])
    adjusted, repaired = symmetric_average(ids, probs)
    assert repaired == 1
    assert np.allclose(adjusted, probs, atol=1e-9)


def test_symmetric_average_repairs_a_flipped_prediction():
    import numpy as np

    ids = ["x__original", "x__flipped"]
    probs = np.array([
        [0.05, 0.90, 0.03, 0.02],
        [0.05, 0.90, 0.03, 0.02],   # same direction twice -> one must be wrong
    ])
    adjusted, _ = symmetric_average(ids, probs)
    assert abs(adjusted.sum(axis=1) - 1).max() < 1e-9
    # the two directions now disagree by construction
    assert adjusted[0].argmax() != adjusted[1].argmax()


def test_supcon_no_positive_anchor_is_zero_not_nan():
    loss_fn = SupConLoss(temperature=0.1)
    features = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(features, labels)
    assert torch.isfinite(loss).all()
    assert float(loss) == 0.0
    loss.backward()  # must stay connected to the graph


def test_supcon_smaller_for_clustered_same_class():
    loss_fn = SupConLoss(temperature=0.1)
    labels = torch.tensor([0, 0, 1, 1])
    clustered = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )
    mixed = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    assert float(loss_fn(clustered, labels)) < float(loss_fn(mixed, labels))


def _logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.log(probs.clamp_min(1e-9))


def test_dc_is_zero_for_exact_reversal():
    loss_fn = DirectionalConsistencyLoss()
    # A -> B : [E, F, B, N] = [.05, .90, .03, .02]
    p_ab = torch.tensor([[0.05, 0.90, 0.03, 0.02]])
    # B -> A must be [E, B, F, N] = [.05, .03, .90, .02]
    p_ba = torch.tensor([[0.05, 0.03, 0.90, 0.02]])
    logits = _logits_from_probs(torch.cat([p_ab, p_ba], dim=0))
    loss = loss_fn(logits, torch.tensor([0]), torch.tensor([1]))
    assert float(loss) < 1e-5


def test_dc_penalises_forward_twice():
    loss_fn = DirectionalConsistencyLoss()
    probs = torch.tensor(
        [[0.05, 0.90, 0.03, 0.02], [0.05, 0.90, 0.03, 0.02]]
    )
    logits = _logits_from_probs(probs)
    loss = loss_fn(logits, torch.tensor([0]), torch.tensor([1]))
    # JSD is bounded by ln 2
    assert 0.5 < float(loss) <= math.log(2) + 1e-6


def test_dc_ignores_negative_other_in_renormalisation():
    loss_fn = DirectionalConsistencyLoss()
    p_ab = torch.tensor([[0.01, 0.45, 0.01, 0.53]])
    p_ba = torch.tensor([[0.01, 0.01, 0.45, 0.53]])
    logits = _logits_from_probs(torch.cat([p_ab, p_ba], dim=0))
    assert float(loss_fn(logits, torch.tensor([0]), torch.tensor([1]))) < 1e-5


def test_dc_empty_pairs_is_zero_with_grad():
    loss_fn = DirectionalConsistencyLoss()
    logits = torch.randn(3, 4, requires_grad=True)
    loss = loss_fn(logits, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
    assert float(loss) == 0.0
    loss.backward()


def test_build_dc_pairs_unique_and_ordered():
    ids = ["b", "a", "c"]
    reverse = ["a", "b", ""]
    left, right = build_dc_pairs(ids, reverse)
    assert left.tolist() == [0]
    assert right.tolist() == [1]
    # order does not matter: the smaller index is always first
    ids2 = ["a", "b"]
    left2, right2 = build_dc_pairs(ids2, ["b", "a"])
    assert ids2[left2[0]] == "a" and ids2[right2[0]] == "b"
    # pair with missing counterpart is skipped
    assert build_dc_pairs(["a"], ["zzz"])[0].numel() == 0


def test_class_weights_are_inverse_frequency():
    weights = class_weights_from_counts({0: 10, 1: 10, 2: 10, 3: 10})
    assert torch.allclose(weights, torch.ones(4))
    weights = class_weights_from_counts({0: 20, 1: 10, 2: 10, 3: 10})
    assert float(weights[0]) < float(weights[1])


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #
def _toy_units():
    # 6 reversible F/B units, 4 equivalence units, 4 NEGATIVE_OTHER singletons
    units = []
    labels = []
    for _ in range(6):
        i = len(labels)
        units.append((i, i + 1))
        labels.extend([FORWARD, BACKWARD])
    for _ in range(4):
        i = len(labels)
        units.append((i, i + 1))
        labels.extend([EQUIVALENCE, EQUIVALENCE])
    for _ in range(4):
        i = len(labels)
        units.append((i,))
        labels.append(NEGATIVE)
    return units, labels


def test_sampler_keeps_reverse_pairs_together_and_covers_classes():
    units, labels = _toy_units()
    sampler = PairAwareClassBalancedSampler(units, labels, batch_size=8, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    for batch in batches:
        assert sorted(batch) == sorted(set(batch)), "duplicate index in a batch"
        present = {labels[i] for i in batch}
        assert len(present) >= 3, f"batch misses classes: {present}"
        for unit in units:
            members = set(unit) & set(batch)
            assert len(members) in (0, len(unit)), "reversible unit was split across batches"


def test_sampler_rejects_unlinked_reversible_instance():
    units = [(0,), (1,)]
    labels = [FORWARD, NEGATIVE]
    with pytest.raises(ValueError):
        PairAwareClassBalancedSampler(units, labels, batch_size=4)
