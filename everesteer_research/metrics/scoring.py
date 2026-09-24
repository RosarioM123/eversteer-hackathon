"""
Scoring metric interface.

CRITICAL: None of the formulas below are implemented. Every function raises
NotImplementedError until the exact NYC formulas have been confirmed against
official event documentation. Do NOT fill these in from memory of a
different tournament's formulas - your original instructions explicitly
warn against assuming NYC uses the same scoring as Everesteer's general
docs.

Fill these in only after the EVENT SPEC checklist item "current CORR/AIMC/
NCORR weights" and "scoring metrics" have been confirmed.
"""

from dataclasses import dataclass, field
from typing import Sequence, Optional, Dict, List
import statistics


def corr(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """TODO: confirm against NYC event specification.
    Is this Pearson, Spearman, or a rank-based variant? Confirm before implementing."""
    raise NotImplementedError("TODO: confirm CORR formula against NYC event specification")


def aimc(predictions: Sequence[float], targets: Sequence[float],
         benchmark: Sequence[float]) -> float:
    """TODO: confirm against NYC event specification.
    Likely measures incremental/orthogonal contribution beyond a benchmark -
    exact definition must be confirmed, not assumed from other tournaments."""
    raise NotImplementedError("TODO: confirm AIMC formula against NYC event specification")


def ncorr(predictions: Sequence[float], targets: Sequence[float],
          core_features: Optional[Sequence[Sequence[float]]] = None) -> float:
    """TODO: confirm against NYC event specification.
    Likely a neutralized correlation (post feature-exposure neutralization) -
    exact neutralization procedure must be confirmed."""
    raise NotImplementedError("TODO: confirm NCORR formula against NYC event specification")


def blended_score(component_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """TODO: confirm against NYC event specification.
    Requires the CURRENT metric_weights from EventSpec - do not hardcode
    weights guessed from another event or from historical documentation."""
    raise NotImplementedError("TODO: confirm blend weights against NYC event specification")


def prediction_correlation(pred_a: Sequence[float], pred_b: Sequence[float]) -> float:
    """This one is NOT event-specific - it's just correlation between two
    prediction vectors, used for model-diversity analysis. Safe to implement
    once we pick a correlation convention (default: Pearson)."""
    if len(pred_a) != len(pred_b):
        raise ValueError("prediction vectors must be the same length")
    if len(pred_a) < 2:
        raise ValueError("need at least 2 observations to compute correlation")
    try:
        return statistics.correlation(pred_a, pred_b)  # Python 3.10+
    except AttributeError:
        # Fallback manual Pearson correlation for older Python
        n = len(pred_a)
        mean_a = sum(pred_a) / n
        mean_b = sum(pred_b) / n
        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(pred_a, pred_b))
        std_a = math_sqrt(sum((a - mean_a) ** 2 for a in pred_a))
        std_b = math_sqrt(sum((b - mean_b) ** 2 for b in pred_b))
        if std_a == 0 or std_b == 0:
            return 0.0
        return cov / (std_a * std_b)


def math_sqrt(x: float) -> float:
    return x ** 0.5


@dataclass
class FoldMetrics:
    fold_index: int
    corr: Optional[float] = None
    aimc: Optional[float] = None
    ncorr: Optional[float] = None
    blended: Optional[float] = None
    benchmark_corr: Optional[float] = None
    n_observations: Optional[int] = None


@dataclass
class MetricsResult:
    """Aggregated result across all folds for one experiment."""
    per_fold: List[FoldMetrics] = field(default_factory=list)

    def mean_blended(self) -> Optional[float]:
        vals = [f.blended for f in self.per_fold if f.blended is not None]
        return statistics.mean(vals) if vals else None

    def std_blended(self) -> Optional[float]:
        vals = [f.blended for f in self.per_fold if f.blended is not None]
        return statistics.pstdev(vals) if len(vals) > 1 else None

    def worst_fold_blended(self) -> Optional[float]:
        vals = [f.blended for f in self.per_fold if f.blended is not None]
        return min(vals) if vals else None

    def mean_corr(self) -> Optional[float]:
        vals = [f.corr for f in self.per_fold if f.corr is not None]
        return statistics.mean(vals) if vals else None
