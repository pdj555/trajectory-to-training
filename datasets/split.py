"""Train/val split, stratified by task type.

Stratification matters because task types have different difficulty and
different base rates. A naive random split can put all of one task type
in train and none in val.
"""

from __future__ import annotations

import random
from collections import defaultdict


def stratified_split(rows: list[dict],
                     val_fraction: float = 0.2,
                     seed: int = 42,
                     key: str = "task_type") -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get(key, "unknown"))].append(row)

    train, val = [], []
    for _, group in sorted(by_key.items()):
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
