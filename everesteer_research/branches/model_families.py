"""
Branch B: Model Families.

Covers: neutral baseline, regularized linear, shallow boosting, strong
boosting, and a structurally different model family - the baseline ladder.
"""

from config.experiment_config import ExperimentConfig

BRANCH_NAME = "model_families"


def registered_experiments() -> list:
    return [
        ExperimentConfig(
            name="B0_neutral_baseline",
            hypothesis="Establishes the floor every real model must beat.",
            branch=BRANCH_NAME,
            model="neutral_baseline",
            features="all_raw",
        ),
        ExperimentConfig(
            name="B2_regularized_linear",
            hypothesis=(
                "Regularization improves out-of-sample stability over plain "
                "linear regression, especially if features are correlated."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="all_raw",
        ),
        ExperimentConfig(
            name="B3_shallow_gbm_nonlinearity_probe",
            hypothesis=(
                "A shallow GBM tells us whether nonlinear interactions exist "
                "at all, without the overfitting risk of a deep model."
            ),
            branch=BRANCH_NAME,
            model="shallow_gbm",
            features="all_raw",
        ),
        ExperimentConfig(
            name="B4_full_gbm",
            hypothesis=(
                "A well-tuned GBM captures more nonlinear signal than shallow "
                "variants, if such signal genuinely exists (per B3's result)."
            ),
            branch=BRANCH_NAME,
            model="full_gbm",
            features="all_raw",
        ),
        ExperimentConfig(
            name="B5_structurally_different_model",
            hypothesis=(
                "A structurally different model family (random forest) may "
                "capture different signal than GBM/linear, valuable for "
                "diversity even if standalone score is lower."
            ),
            branch=BRANCH_NAME,
            model="random_forest",
            features="all_raw",
        ),
    ]
