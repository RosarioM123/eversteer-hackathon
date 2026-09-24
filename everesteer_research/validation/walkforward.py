"""
WalkForwardSplitter: generates chronological, exped-based folds.

Rules enforced here (per the research harness requirements):
  - Never shuffle. Order is always by exped/time.
  - Embargo/gap between train and validation to respect target horizon.
  - Preprocessing must be fit ONLY on training data (enforced by convention
    in ExperimentRunner, not here - this class only produces index splits).
  - The most recent block of expeds is carved out as an untouched final
    holdout, returned separately from the research folds.

This is deliberately generic: it operates on a list/array of exped
identifiers, not on NYC-specific column names. The caller supplies the
exped column once the real schema is known.
"""

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Any
import math


@dataclass
class Fold:
    fold_index: int
    train_expeds: List[Any]
    validation_expeds: List[Any]


class WalkForwardSplitter:
    def __init__(self, n_folds: int, embargo: int, min_train_expeds: int,
                 holdout_fraction: float):
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        if not (0 <= holdout_fraction < 1):
            raise ValueError("holdout_fraction must be in [0, 1)")
        self.n_folds = n_folds
        self.embargo = embargo
        self.min_train_expeds = min_train_expeds
        self.holdout_fraction = holdout_fraction

    def split(self, ordered_unique_expeds: Sequence) -> Tuple[List["Fold"], List]:
        """
        ordered_unique_expeds: a chronologically sorted, de-duplicated sequence
        of exped identifiers (e.g. from `sorted(df[exped_col].unique())`,
        sorted by the actual chronological key, NOT assumed to be sortable
        by value alone unless confirmed).

        Returns (research_folds, final_holdout_expeds).
        The final_holdout_expeds must not be touched until a candidate has
        already been selected using only research_folds.
        """
        n = len(ordered_unique_expeds)
        if n < self.min_train_expeds + self.embargo + 1:
            raise ValueError(
                f"Not enough expeds ({n}) to satisfy min_train_expeds="
                f"{self.min_train_expeds} + embargo={self.embargo} + >=1 validation exped."
            )

        n_holdout = max(1, math.floor(n * self.holdout_fraction)) if self.holdout_fraction > 0 else 0
        research_expeds = list(ordered_unique_expeds[: n - n_holdout]) if n_holdout else list(ordered_unique_expeds)
        final_holdout_expeds = list(ordered_unique_expeds[n - n_holdout:]) if n_holdout else []

        n_research = len(research_expeds)
        usable_start = self.min_train_expeds
        usable_len = n_research - usable_start - self.embargo
        if usable_len <= 0:
            raise ValueError(
                "Not enough research (non-holdout) expeds left to build even one fold. "
                "Reduce holdout_fraction, min_train_expeds, or embargo."
            )

        fold_size = max(1, usable_len // self.n_folds)
        folds: List[Fold] = []
        for i in range(self.n_folds):
            val_start = usable_start + self.embargo + i * fold_size
            val_end = val_start + fold_size if i < self.n_folds - 1 else n_research
            if val_start >= n_research:
                break
            train_end = val_start - self.embargo
            train_expeds = research_expeds[:train_end]
            validation_expeds = research_expeds[val_start:val_end]
            if not train_expeds or not validation_expeds:
                continue
            folds.append(Fold(fold_index=i, train_expeds=train_expeds,
                               validation_expeds=validation_expeds))

        if not folds:
            raise ValueError("Fold generation produced zero usable folds - check fold_config.")

        return folds, final_holdout_expeds
