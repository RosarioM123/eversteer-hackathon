"""
Branch C: Orthogonal Signal.

Covers: benchmark correlation, prediction correlation, AIMC, NCORR,
feature exposure - the search for signal that predicts the target but
isn't already captured by the benchmark.

These experiments cannot actually compute AIMC/NCORR until
metrics/scoring.py has real formulas (currently NotImplementedError) and
EventSpec.benchmark_column is confirmed. They are registered now so the
research plan is ready to execute the moment those are available.
"""

from config.experiment_config import ExperimentConfig

BRANCH_NAME = "orthogonal_signal"


def registered_experiments() -> list:
    return [
        ExperimentConfig(
            name="benchmark_correlation_audit",
            hypothesis=(
                "Measuring each candidate model's correlation with the benchmark "
                "tells us which candidates are just reconstructing it vs. adding "
                "genuinely new information."
            ),
            branch=BRANCH_NAME,
            model="ridge",   # placeholder - actually run across ALL portfolio models, not just one
            features="all_raw",
        ),
        ExperimentConfig(
            name="orthogonal_plus_benchmark_blend",
            hypothesis=(
                "Combining a benchmark-like model with an orthogonal-signal model "
                "improves the actual blended score beyond either alone."
            ),
            branch=BRANCH_NAME,
            model="ridge",   # placeholder - this is really an ensemble experiment, registered here
            features="all_raw",  # for narrative grouping under 'orthogonal signal'
        ),
    ]
