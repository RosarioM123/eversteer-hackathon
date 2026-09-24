"""
ExperimentConfig: the single object that fully defines one experiment.

Design goal: changing one field (e.g. `window`) and re-running should be
enough to produce a complete, logged result. No experiment-specific code
should need to be written by hand once branches/*.py has registered the
building blocks (feature sets, model families, windows).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid


@dataclass
class FoldConfig:
    """Controls how walk-forward folds are constructed."""
    n_folds: int = 5                          # TODO: tune once exped count is known
    embargo: int = 1                          # gap (in expeds) between train and validation, >= target horizon
    min_train_expeds: int = 10                # minimum history required before first validation fold
    holdout_fraction: float = 0.15            # fraction of most-recent expeds reserved as untouched final holdout


@dataclass
class ExperimentConfig:
    # --- identity ---
    name: str
    hypothesis: str                            # required: force explicit articulation of WHY before running
    branch: str                                # one of: "signal_discovery", "model_families",
                                                # "orthogonal_signal", "time_regime", "ensemble_portfolio"

    # --- modeling ---
    model: str                                 # key into models.registry.MODEL_REGISTRY
    model_params: Dict[str, Any] = field(default_factory=dict)
    features: str = "all_raw"                  # key into a feature-set registry (built once schema is known)
    target: str = "PRIMARY_TARGET_PLACEHOLDER"  # TODO: replace with EventSpec.primary_target_column
    preprocessing: List[str] = field(default_factory=list)  # e.g. ["impute_missing_sentinel", "zscore"]

    # --- temporal ---
    window: str = "expanding"                  # "expanding" | "rolling_N" | "recency_weighted"
    fold_config: FoldConfig = field(default_factory=FoldConfig)

    # --- reproducibility ---
    seed: int = 42

    # --- bookkeeping (auto-filled by runner, not the user) ---
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    data_version: Optional[str] = None          # TODO: set by caller once data provenance is known

    def validate(self) -> List[str]:
        """Return a list of problems with this config. Empty list = OK to run."""
        problems = []
        if not self.hypothesis or len(self.hypothesis.strip()) < 10:
            problems.append("hypothesis must be explicitly stated (no blank/lazy hypotheses).")
        allowed_branches = {
            "signal_discovery", "model_families", "orthogonal_signal",
            "time_regime", "ensemble_portfolio",
        }
        if self.branch not in allowed_branches:
            problems.append(f"branch '{self.branch}' not in {allowed_branches}")
        if self.target == "PRIMARY_TARGET_PLACEHOLDER":
            problems.append(
                "target is still the placeholder value - set it from the confirmed EventSpec "
                "before running against real data."
            )
        return problems
