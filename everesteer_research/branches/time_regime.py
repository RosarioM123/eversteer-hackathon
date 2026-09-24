"""
Branch D: Time / Regime.

Covers: expanding vs rolling windows, recency weighting, regime-conditioned
behavior. Window choice must NOT be selected using the final holdout - see
FoldConfig.holdout_fraction and the ExperimentRunner's enforcement.
"""

from config.experiment_config import ExperimentConfig

BRANCH_NAME = "time_regime"


def registered_experiments() -> list:
    return [
        ExperimentConfig(
            name="temporal_stability_check",
            hypothesis=(
                "Feature-target relationships may not be static; per-feature "
                "correlation computed in rolling time blocks will reveal drift "
                "if it exists."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="all_raw",
            window="expanding",
        ),
        ExperimentConfig(
            name="expanding_window_baseline",
            hypothesis="Reference point: train on all available history.",
            branch=BRANCH_NAME,
            model="ridge",
            features="all_raw",
            window="expanding",
        ),
        ExperimentConfig(
            name="rolling_window_60",
            hypothesis=(
                "If relationships are unstable (per temporal_stability_check), "
                "training only on the most recent ~60 expeds may generalize "
                "better than using all history."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="all_raw",
            window="rolling_60",   # window size is illustrative - tune only on research folds
        ),
        ExperimentConfig(
            name="recency_weighted_training",
            hypothesis=(
                "Down-weighting older expeds (rather than dropping them entirely) "
                "may balance stability and recency better than a hard rolling cutoff."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="all_raw",
            window="recency_weighted",
        ),
    ]
