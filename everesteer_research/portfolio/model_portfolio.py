"""
ModelPortfolio: living table of KEEP/PROMISING-tier models.

Purpose: identify models that are GOOD + DIFFERENT, not just rank by CORR.
This is intentionally separate from the ledger (which logs every experiment,
including rejects) - the portfolio only tracks candidates worth comparing
for ensembling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import itertools


@dataclass
class PortfolioEntry:
    model_name: str
    experiment_id: str
    corr: Optional[float]
    aimc: Optional[float]
    ncorr: Optional[float]
    blended: Optional[float]
    benchmark_corr: Optional[float]
    stability: Optional[float]      # e.g. -std_fold_score, or worst_fold_score; convention TBD
    predictions: Optional[List[float]] = None   # kept only in-memory for corr matrix, not persisted lightly


class ModelPortfolio:
    def __init__(self):
        self._entries: Dict[str, PortfolioEntry] = {}

    def add_or_update(self, entry: PortfolioEntry) -> None:
        self._entries[entry.model_name] = entry

    def table(self) -> List[Dict]:
        """Returns rows suitable for printing/exporting as the comparison table:
        Model | CORR | AIMC | NCORR | Blended | Benchmark Corr | Stability
        """
        rows = []
        for e in self._entries.values():
            rows.append({
                "model": e.model_name,
                "CORR": e.corr,
                "AIMC": e.aimc,
                "NCORR": e.ncorr,
                "blended": e.blended,
                "benchmark_corr": e.benchmark_corr,
                "stability": e.stability,
            })
        return rows

    def prediction_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Pairwise prediction correlation between every model currently in
        the portfolio that has predictions attached. Requires
        metrics.scoring.prediction_correlation to be available."""
        from metrics.scoring import prediction_correlation

        names = [n for n, e in self._entries.items() if e.predictions is not None]
        matrix: Dict[str, Dict[str, float]] = {n: {} for n in names}
        for a, b in itertools.combinations(names, 2):
            corr_ab = prediction_correlation(
                self._entries[a].predictions, self._entries[b].predictions
            )
            matrix[a][b] = corr_ab
            matrix[b][a] = corr_ab
        for n in names:
            matrix[n][n] = 1.0
        return matrix

    def find_diverse_candidates(self, correlation_threshold: float = 0.85) -> List[str]:
        """Return model names whose predictions are NOT highly correlated
        with any other model already in the portfolio - i.e. candidates
        that plausibly carry different information, per the 'good + different'
        principle."""
        matrix = self.prediction_correlation_matrix()
        diverse = []
        for name, row in matrix.items():
            others = {k: v for k, v in row.items() if k != name}
            if not others or max(others.values()) < correlation_threshold:
                diverse.append(name)
        return diverse
