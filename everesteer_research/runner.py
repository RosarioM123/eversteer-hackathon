"""
ExperimentRunner: the single entry point that turns an ExperimentConfig
into a logged ResearchLedger row.

CRITICAL DESIGN RULE: this module never fabricates a result. If `data` is
None, or if the config's target/features are still placeholders, or if the
metric functions aren't implemented yet, `run()` raises an explicit error
rather than returning fake numbers. This is intentional and should not be
"fixed" by stubbing in fake data - that would defeat the entire purpose of
this harness.

No function in this file calls a network endpoint, downloads anything, or
submits anything.
"""

from dataclasses import dataclass
from typing import Any, Optional, Dict
import time

from config.experiment_config import ExperimentConfig
from validation.walkforward import WalkForwardSplitter
from ledger.research_ledger import ResearchLedger, LedgerEntry, now_iso
from redteam.checks import RedTeamChecks
from models.registry import build_model
from data.event_spec import EventSpec


class NoDataSuppliedError(RuntimeError):
    pass


class PlaceholderConfigError(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(self, event_spec: EventSpec, ledger_path: str = "research_ledger.json"):
        self.event_spec = event_spec
        self.ledger = ResearchLedger(ledger_path)
        self.redteam = RedTeamChecks()

    def run(self, config: ExperimentConfig, data: Optional[Any] = None) -> LedgerEntry:
        """
        data: intentionally untyped here - once the real schema is known this
        should become e.g. a pandas DataFrame with confirmed column names.

        This method deliberately refuses to run without real data, and
        refuses to run against placeholder config values, rather than
        silently producing meaningless numbers.
        """
        problems = config.validate()
        if problems:
            raise PlaceholderConfigError(
                f"Cannot run experiment '{config.name}': {problems}"
            )

        if data is None:
            raise NoDataSuppliedError(
                f"Cannot run experiment '{config.name}' without real data. "
                "This framework does not fabricate results. Supply the actual "
                "event dataset once available."
            )

        if not self.event_spec.ready_for_modeling():
            raise PlaceholderConfigError(
                "EventSpec is not ready for modeling. Missing fields: "
                f"{self.event_spec.missing_fields()}"
            )

        # From here on, this is where real fold generation / model fit /
        # metric computation would happen, using the CONFIRMED event_spec
        # and the real `data`. Left unimplemented on purpose - this is a
        # skeleton, not a place to guess at event's actual data shape.
        raise NotImplementedError(
            "Core fit/predict/score loop intentionally left unimplemented "
            "until real event data and confirmed metric formulas are available. "
            "See metrics/scoring.py and validation/walkforward.py for the "
            "pieces that need to be wired together here."
        )

    def log_manual_entry(self, entry: LedgerEntry) -> None:
        """Escape hatch for logging an experiment result that was computed
        outside this exact code path (e.g. during interactive exploration),
        as long as it's a REAL result, not a fabricated one. Caller is
        responsible for that guarantee."""
        self.ledger.append(entry)


def build_default_runner(ledger_path: str = "research_ledger.json") -> ExperimentRunner:
    """Convenience constructor using the module-level CURRENT_EVENT_SPEC.
    Will not be usable for real runs until that spec is filled in."""
    from data.event_spec import CURRENT_EVENT_SPEC
    return ExperimentRunner(event_spec=CURRENT_EVENT_SPEC, ledger_path=ledger_path)


if __name__ == "__main__":
    # Smoke test: demonstrates that the framework correctly REFUSES to
    # fabricate a result when no data / no confirmed spec is present.
    from branches import (
        signal_discovery, model_families, orthogonal_signal,
        time_regime, ensemble_portfolio,
    )

    all_experiments = (
        signal_discovery.registered_experiments()
        + model_families.registered_experiments()
        + orthogonal_signal.registered_experiments()
        + time_regime.registered_experiments()
        + ensemble_portfolio.registered_experiments()
    )
    print(f"Registered {len(all_experiments)} experiment configs across 5 branches.")

    runner = build_default_runner(ledger_path="/tmp/everesteer_smoke_test_ledger.json")

    example = all_experiments[0]
    try:
        runner.run(example, data=None)
    except PlaceholderConfigError as e:
        print(f"[expected] Refused to run - config still has placeholders: {e}")
    except NoDataSuppliedError as e:
        print(f"[expected] Refused to run without data: {e}")

    # Demonstrate the placeholder-target guard explicitly
    from config.experiment_config import ExperimentConfig
    bad_config = ExperimentConfig(
        name="placeholder_target_example",
        hypothesis="This should be rejected because target is still a placeholder.",
        branch="signal_discovery",
        model="ridge",
    )
    try:
        runner.run(bad_config, data={"fake": "data"})
    except PlaceholderConfigError as e:
        print(f"[expected] Refused to run with placeholder config: {e}")

    # Demonstrate the no-data guard on a config that HAS a real (non-placeholder) target
    real_target_config = ExperimentConfig(
        name="no_data_example",
        hypothesis="This should be rejected because no data was supplied, even though the config is valid.",
        branch="signal_discovery",
        model="ridge",
        target="some_confirmed_target_column",
    )
    try:
        runner.run(real_target_config, data=None)
    except NoDataSuppliedError as e:
        print(f"[expected] Refused to run without data: {e}")
