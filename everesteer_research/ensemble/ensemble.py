"""
EnsembleBuilder: combine predictions from multiple portfolio models.

Rule enforced in code, not just convention: learned weights can ONLY be
fit when the caller explicitly passes train_only=True AND the data passed
is explicitly not the final holdout. There is no code path that fits
weights on holdout data.
"""

from typing import Dict, List, Sequence
import statistics


class HoldoutProtectionError(Exception):
    pass


class EnsembleBuilder:
    def equal_weight(self, predictions: Dict[str, Sequence[float]]) -> List[float]:
        """Simple average across all provided model predictions. This is
        always the first ensemble tried, per the research plan."""
        if not predictions:
            raise ValueError("no predictions supplied")
        names = list(predictions.keys())
        n = len(predictions[names[0]])
        for name in names:
            if len(predictions[name]) != n:
                raise ValueError(f"prediction length mismatch for model '{name}'")
        return [
            statistics.mean(predictions[name][i] for name in names)
            for i in range(n)
        ]

    def learned_weight(
        self,
        predictions: Dict[str, Sequence[float]],
        targets: Sequence[float],
        train_only: bool,
        is_final_holdout: bool = False,
    ) -> Dict[str, float]:
        """
        Fits ensemble weights via simple constrained regression (weights
        summing to 1). ONLY runs if train_only=True and is_final_holdout=False.

        This is a placeholder for the actual weight-fitting logic (e.g.
        non-negative least squares) - the important part right now is the
        guard rail, not the optimizer, since we have no real predictions
        or targets yet.
        """
        if is_final_holdout:
            raise HoldoutProtectionError(
                "Refusing to fit ensemble weights on final holdout data. "
                "Learned weights must be fit only on research/training folds."
            )
        if not train_only:
            raise HoldoutProtectionError(
                "learned_weight() requires train_only=True to be passed explicitly. "
                "This is a deliberate guard, not a default-on convenience."
            )
        raise NotImplementedError(
            "TODO: implement actual weight optimization (e.g. NNLS) once "
            "real model predictions and targets are available."
        )

    def prediction_correlation_summary(
        self, predictions: Dict[str, Sequence[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Thin wrapper so ensemble decisions can reference the same
        correlation logic used in ModelPortfolio."""
        from metrics.scoring import prediction_correlation
        import itertools

        names = list(predictions.keys())
        matrix = {n: {} for n in names}
        for a, b in itertools.combinations(names, 2):
            c = prediction_correlation(predictions[a], predictions[b])
            matrix[a][b] = c
            matrix[b][a] = c
        for n in names:
            matrix[n][n] = 1.0
        return matrix
