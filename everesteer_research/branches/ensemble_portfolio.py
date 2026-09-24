"""
Branch E: Ensemble / Model Portfolio.

Covers: equal-weight ensembling first, prediction-correlation-based
diversity search, and (only later, with evidence) learned weights fit on
research folds only - never the final holdout.

This branch depends on outputs from branches A-D (it needs candidate
models to already exist in the ModelPortfolio) - so it is naturally the
last branch to actually execute, even though all branches can be
registered in parallel now.
"""

from config.experiment_config import ExperimentConfig

BRANCH_NAME = "ensemble_portfolio"


def registered_experiments() -> list:
    return [
        ExperimentConfig(
            name="equal_weight_ensemble_top3",
            hypothesis=(
                "A simple average of 2-3 genuinely diverse, individually-solid "
                "models outperforms any single model on out-of-sample folds, "
                "particularly on worst-fold performance."
            ),
            branch=BRANCH_NAME,
            model="ridge",  # placeholder - this experiment actually consumes multiple
                            # portfolio models via EnsembleBuilder.equal_weight(),
                            # not a single 'model' in the usual sense
            features="all_raw",
        ),
        ExperimentConfig(
            name="learned_weight_ensemble",
            hypothesis=(
                "Once equal-weight ensembling is validated, learned (non-equal) "
                "weights fit on research folds may further improve robustness. "
                "This must NEVER be fit on the final holdout."
            ),
            branch=BRANCH_NAME,
            model="ridge",  # placeholder, see EnsembleBuilder.learned_weight()
            features="all_raw",
        ),
    ]
