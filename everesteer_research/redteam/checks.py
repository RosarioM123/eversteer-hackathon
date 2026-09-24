"""
RedTeamChecks: hooks for adversarial review of any apparently strong result.

Each check returns a (passed: bool, note: str) tuple. Checks that need real
data / real models to run will raise NotImplementedError until wired up -
they are listed here so no check is silently skipped or forgotten, per the
requirement that "every apparently strong model must survive adversarial
review."

None of these checks fabricate a pass. If a check can't be run yet (e.g. no
data), it must return status "NOT_RUN", never a fake "PASS".
"""

from typing import Callable, Dict, Tuple, Any, Optional


class RedTeamChecks:
    def __init__(self):
        self.registry: Dict[str, Callable[..., Tuple[bool, str]]] = {
            "temporal_leakage": self.check_temporal_leakage,
            "feature_leakage": self.check_feature_leakage,
            "target_leakage": self.check_target_leakage,
            "preprocessing_leakage": self.check_preprocessing_leakage,
            "id_leakage": self.check_id_leakage,
            "structural_column_leakage": self.check_structural_column_leakage,
            "stale_data_leakage": self.check_stale_data_leakage,
            "excessive_hp_search": self.check_excessive_hyperparameter_search,
            "multiple_testing_risk": self.check_multiple_testing_risk,
            "seed_sensitivity": self.check_seed_sensitivity,
            "window_sensitivity": self.check_window_sensitivity,
            "feature_removal_sensitivity": self.check_feature_removal_sensitivity,
        }

    def run_all(self, context: Dict[str, Any]) -> Dict[str, str]:
        """context carries whatever each check needs (fold data, configs,
        prior experiment counts, etc.). Missing context => NOT_RUN, not a
        fabricated pass."""
        results = {}
        for name, fn in self.registry.items():
            try:
                passed, note = fn(context)
                results[name] = f"{'PASS' if passed else 'FAIL'}: {note}"
            except NotImplementedError as e:
                results[name] = f"NOT_RUN: {e}"
        return results

    # --- Individual checks ---
    # Each is a stub with a clear contract. Implement the body once real
    # data/model objects are available; do not guess a result.

    def check_temporal_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify no validation-fold exped precedes (or overlaps within the
        embargo of) any training-fold exped."""
        raise NotImplementedError("requires real fold exped lists to check ordering")

    def check_feature_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify no feature encodes information only knowable after the
        target's realization time."""
        raise NotImplementedError("requires confirmed feature availability timing from EventSpec")

    def check_target_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify the target itself (or a transform of it) isn't present
        among the features."""
        raise NotImplementedError("requires real feature/target column list")

    def check_preprocessing_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify scalers/imputers were fit only on training folds, not on
        validation or the full dataset."""
        raise NotImplementedError("requires access to the actual preprocessing pipeline object")

    def check_id_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify IDs are not being used as if they carried ordinal/semantic
        meaning they don't have (per 'IDs are opaque')."""
        raise NotImplementedError("requires real ID column and feature importance output")

    def check_structural_column_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify structural columns (like exped) aren't used directly as
        model features without justification."""
        raise NotImplementedError("requires real feature list to check against exped/id columns")

    def check_stale_data_leakage(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify the data version used for a live-round prediction matches
        the current live split, not a cached/stale copy."""
        raise NotImplementedError("requires real data version metadata - not applicable pre-data")

    def check_excessive_hyperparameter_search(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Flag if a very large number of hyperparameter combinations were
        tried for a single hypothesis (search-noise risk)."""
        n_trials = ctx.get("n_hyperparameter_trials")
        if n_trials is None:
            raise NotImplementedError("context must supply n_hyperparameter_trials")
        threshold = ctx.get("hp_trial_threshold", 25)
        passed = n_trials <= threshold
        return passed, f"{n_trials} trials vs threshold {threshold}"

    def check_multiple_testing_risk(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Flag if a hypothesis is being promoted after only one test."""
        n_tests = ctx.get("n_tests_logged")
        if n_tests is None:
            raise NotImplementedError("context must supply n_tests_logged from the ResearchLedger")
        passed = n_tests >= 2
        return passed, f"{n_tests} independent test(s) logged (need >= 2 for CONFIRMED)"

    def check_seed_sensitivity(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Compare scores across multiple random seeds; large variance is a
        red flag."""
        raise NotImplementedError("requires re-running the experiment across >=3 seeds")

    def check_window_sensitivity(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Compare scores across nearby training-window choices; a result
        that collapses with a small window change is fragile."""
        raise NotImplementedError("requires re-running the experiment across >=2 window configs")

    def check_feature_removal_sensitivity(self, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Remove the top-importance feature and re-check whether the result
        survives."""
        raise NotImplementedError("requires re-running the experiment with the top feature dropped")
