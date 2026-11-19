"""Batching that serves both SupCon and Directional Consistency.

Two requirements must hold for (almost) every training batch:

1. several instances of every class, so supervised contrastive loss has
   positives that are not the anchor itself;
2. the two members of every reversible pair travel together, so the DC loss can
   compare their distributions. Pairs are never invented: units come from the
   organizer-provided ``reverse_pair_id`` links only.

Classes are balanced proportionally to the training distribution: each
reversible F/B unit contributes one F and one B instance, an equivalence unit
contributes two E instances and a negative unit contributes one N instance, so
the quota is computed on the unit level from the desired instance-level share.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from .labels import BACKWARD, EQUIVALENCE, FORWARD, NEGATIVE


class PairAwareClassBalancedSampler(Sampler[list[int]]):
    def __init__(
        self,
        units: Sequence[Sequence[int]],
        instance_labels: Sequence[int],
        batch_size: int,
        *,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if not units:
            raise ValueError("sampler needs at least one unit")
        self.units = [tuple(int(i) for i in unit) for unit in units]
        self.instance_labels = list(instance_labels)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = shuffle
        self._epoch = 0

        buckets: dict[str, list[int]] = {"eq": [], "fb": [], "neg": []}
        for idx, unit in enumerate(self.units):
            labels = {self.instance_labels[i] for i in unit}
            if labels == {EQUIVALENCE}:
                buckets["eq"].append(idx)
            elif labels & {FORWARD, BACKWARD}:
                if len(unit) != 2:
                    raise ValueError(
                        "a reversible F/B instance is not linked to its reverse; "
                        "DC would silently lose it"
                    )
                buckets["fb"].append(idx)
            else:
                buckets["neg"].append(idx)
        if not any(buckets.values()):
            raise ValueError("no units at all")

        available = {name: bucket for name, bucket in buckets.items() if bucket}

        counts = Counter(self.instance_labels)
        total = sum(counts.values())
        share = {label: counts.get(label, 0) / total for label in (EQUIVALENCE, FORWARD, BACKWARD, NEGATIVE)}

        quota = {
            # two E instances per equivalence unit
            "eq": max(1, round(share[EQUIVALENCE] * self.batch_size / 2)),
            # one F and one B instance per reversible unit
            "fb": max(1, round((share[FORWARD] + share[BACKWARD]) * self.batch_size / 2)),
            "neg": max(1, round(share[NEGATIVE] * self.batch_size)),
        }
        quota = {
            name: min(value, len(available[name]))
            for name, value in quota.items()
            if name in available
        }

        self.buckets = available
        self.quota = quota
        instances_per_batch = 2 * quota.get("eq", 0) + 2 * quota.get("fb", 0) + quota.get("neg", 0)
        self.steps_per_epoch = max(1, math.ceil(len(self.instance_labels) / instances_per_batch))

        self._queues: dict[str, list[int]] = {name: [] for name in self.buckets}
        self._position: dict[str, int] = {name: 0 for name in buckets}

    # ------------------------------------------------------------------ #
    def _refill(self, rng: random.Random) -> None:
        for name, bucket in self.buckets.items():
            order = list(bucket)
            rng.shuffle(order)
            self._queues[name] = order + self._queues[name][self._position[name] :]
            self._position[name] = 0

    def _take(self, name: str, rng: random.Random) -> int:
        if self._position[name] >= len(self._queues[name]):
            bucket = list(self.buckets[name])
            rng.shuffle(bucket)
            self._queues[name].extend(bucket)
        unit = self._queues[name][self._position[name]]
        self._position[name] += 1
        return unit

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        for name in self._queues:
            if not self._queues[name]:
                self._queues[name] = list(self.buckets[name])
                rng.shuffle(self._queues[name])
            self._position[name] = 0

        groups = [name for name in self.quota]
        for _ in range(self.steps_per_epoch):
            chosen: list[int] = []
            seen: set[int] = set()
            if self.shuffle:
                rng.shuffle(groups)
            for name in groups:
                wanted = self.quota[name]
                tries = 0
                picked = 0
                while picked < wanted and tries < wanted * 4:
                    tries += 1
                    unit_index = self._take(name, rng)
                    if unit_index in seen:
                        continue
                    seen.add(unit_index)
                    chosen.append(unit_index)
                    picked += 1
            if self.shuffle:
                rng.shuffle(chosen)
            batch = [i for unit_index in chosen for i in self.units[unit_index]]
            if batch:
                yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch
