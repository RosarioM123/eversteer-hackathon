"""
Branch A: Signal Discovery.

Covers: univariate features, feature clusters, breadth/consensus,
disagreement, nonlinearities, missingness, temporal stability.

This module only REGISTERS experiment configs - it does not run them.
The actual feature-set implementations (e.g. "cluster_breadth_v1") must be
built once the real feature catalogue is known; until then, feature-set
keys here are placeholders that the runner will reject if used against
real data without a matching implementation.

Each experiment must carry an explicit hypothesis - this is enforced by
ExperimentConfig.validate().
"""

from config.experiment_config import ExperimentConfig

BRANCH_NAME = "signal_discovery"


def registered_experiments() -> list:
    """Returns a list of ExperimentConfig objects for this branch.
    Feature-set / target values are placeholders until EventSpec is filled in.
    """
    return [
        ExperimentConfig(
            name="univariate_top_feature_probe",
            hypothesis=(
                "One or a small number of features may dominate the signal; "
                "identifying this early prevents over-crediting complex models."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="single_feature_placeholder",   # TODO: iterate per-feature once catalogue known
        ),
        ExperimentConfig(
            name="cluster_breadth_v1",
            hypothesis=(
                "Count/proportion of features in high vs low quantiles per row "
                "carries information beyond any individual feature (consensus)."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="cluster_breadth_v1",            # TODO: build from feature-cluster mapping
        ),
        ExperimentConfig(
            name="cluster_disagreement_v1",
            hypothesis=(
                "Rows where features disagree (mixed high/low) may be predictable "
                "differently than rows where features agree."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="cluster_disagreement_v1",       # TODO: build from feature-cluster mapping
        ),
        ExperimentConfig(
            name="missingness_signal_probe",
            hypothesis=(
                "Missingness count/pattern may correlate with the target, "
                "provided this information is legitimately available at prediction time."
            ),
            branch=BRANCH_NAME,
            model="ridge",
            features="missingness_summary_v1",        # TODO: build once missing-value convention confirmed
        ),
    ]
