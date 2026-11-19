"""The three objectives of DARC-NLI v1.

    L_total = L_CE + lambda_scl * L_SupCon + lambda_dc * L_DC

Numerical rules from the technical document that are enforced here:

* SupCon anchors without any positive in the batch are dropped instead of
  producing NaN; a batch with no valid anchor contributes exactly zero.
* Only the three reversible classes are compared by DC. Their probabilities are
  re-normalised, NEGATIVE_OTHER never enters the DC loss.
* DC uses Jensen-Shannon (symmetric, bounded) instead of plain KL.
* Every 2-D pair inside a batch is compared at most once (the trainer passes
  unique pairs ordered by instance id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import REVERSE_PERMUTATION

EPS = 1e-8


def zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    """A zero that still carries gradients (so ``loss.backward()`` is safe)."""
    return reference.sum() * 0.0


class SupConLoss(nn.Module):
    """Class-aware supervised contrastive loss over L2-normalised projections."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must be [batch, dim]")
        if features.size(0) < 2:
            return zero_like_loss(features)

        device = features.device
        z = F.normalize(features, dim=1, eps=EPS)
        batch = z.size(0)

        similarity = torch.matmul(z, z.t()) / self.temperature
        self_mask = torch.eye(batch, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(self_mask, float("-inf"))
        # log-sum-exp stabilisation: subtracting the row max keeps exp() finite.
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(similarity) * (~self_mask)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + EPS)

        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()) & (~self_mask)
        positives_per_anchor = positive_mask.sum(dim=1)
        valid = positives_per_anchor > 0
        if not bool(valid.any()):
            return zero_like_loss(features)

        # masked_fill avoids 0 * -inf = NaN for non-positive entries.
        positive_log_prob = log_prob.masked_fill(~positive_mask, 0.0)
        mean_log_prob = positive_log_prob.sum(dim=1)[valid] / positives_per_anchor[valid]
        return -mean_log_prob.mean()


class DirectionalConsistencyLoss(nn.Module):
    """Jensen-Shannon consistency between a pair and its reversed counterpart."""

    def forward(
        self,
        logits: torch.Tensor,
        pair_a: torch.Tensor,
        pair_b: torch.Tensor,
    ) -> torch.Tensor:
        if pair_a.numel() == 0:
            return zero_like_loss(logits)
        if pair_a.shape != pair_b.shape:
            raise ValueError("pair_a and pair_b must have the same shape")

        probabilities = F.softmax(logits, dim=-1)
        reversible = probabilities[:, :3]
        reversible = reversible / reversible.sum(dim=1, keepdim=True).clamp_min(EPS)

        p_a = reversible[pair_a]
        p_b = reversible[pair_b]
        # Reverse the distribution of member A: [E, F, B] -> [E, B, F].
        p_a_reversed = p_a[:, list(REVERSE_PERMUTATION)]

        middle = 0.5 * (p_a_reversed + p_b)
        jsd = 0.5 * _kl(p_a_reversed, middle) + 0.5 * _kl(p_b, middle)
        return jsd.mean()


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    return (p * (p.log() - q.log())).sum(dim=-1)


@dataclass
class LossOutput:
    total: torch.Tensor
    ce: torch.Tensor
    supcon: torch.Tensor
    dc: torch.Tensor


class DarcLoss(nn.Module):
    """Weighted combination of CE, SupCon and DC."""

    def __init__(
        self,
        *,
        lambda_supcon: float = 0.0,
        lambda_dc: float = 0.0,
        temperature: float = 0.1,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_supcon = float(lambda_supcon)
        self.lambda_dc = float(lambda_dc)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.supcon = SupConLoss(temperature=temperature)
        self.dc = DirectionalConsistencyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        projections: torch.Tensor,
        labels: torch.Tensor,
        pair_a: torch.Tensor,
        pair_b: torch.Tensor,
    ) -> LossOutput:
        weight = self.class_weights
        ce = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

        supcon = (
            self.supcon(projections, labels)
            if self.lambda_supcon > 0
            else zero_like_loss(logits)
        )
        dc = (
            self.dc(logits, pair_a, pair_b)
            if self.lambda_dc > 0
            else zero_like_loss(logits)
        )
        total = ce + self.lambda_supcon * supcon + self.lambda_dc * dc
        return LossOutput(total=total, ce=ce.detach(), supcon=supcon.detach(), dc=dc.detach())


def class_weights_from_counts(counts: Mapping[int, int], num_labels: int = 4) -> torch.Tensor:
    """Inverse-frequency weights (disabled by default, kept for ablations)."""
    total = sum(counts.values())
    present = [counts.get(i, 0) for i in range(num_labels)]
    weights = [total / (num_labels * c) if c > 0 else 0.0 for c in present]
    return torch.tensor(weights, dtype=torch.float32)


def build_dc_pairs(instance_ids: list[str], reverse_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs whose *both* members are in the current batch.

    ``reverse_ids`` holds the organizer's ``reverse_pair_id`` per instance.
    A pair is emitted once, with the lexicographically smaller instance first,
    so no pair is counted twice even if duplicate links exist.
    """
    position = {instance_id: i for i, instance_id in enumerate(instance_ids)}
    left: list[int] = []
    right: list[int] = []
    seen: set[tuple[int, int]] = set()
    for i, (instance_id, reverse_id) in enumerate(zip(instance_ids, reverse_ids)):
        if not reverse_id:
            continue
        j = position.get(reverse_id)
        if j is None or j == i:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:  # both directions are in the batch: count the pair once
            continue
        seen.add(key)
        left.append(key[0])
        right.append(key[1])
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)
