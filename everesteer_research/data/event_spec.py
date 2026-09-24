"""
EventSpec: container for every confirmed NYC event rule.

NOTHING in this file should be treated as real until it has been verified
against actual Everesteer NYC documentation/data. Every field starts as
None or "TODO" on purpose.

Do not let a model or experiment run against this until the relevant
fields have been filled in from confirmed sources.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class EventSpec:
    # --- Event structure ---
    event_name: Optional[str] = None                 # TODO: confirm
    lane: Optional[str] = None                        # TODO: confirm (distinct from general tournament?)
    current_round: Optional[int] = None                # TODO: confirm
    round_open_time: Optional[str] = None              # TODO: confirm (ISO timestamp)
    round_close_time: Optional[str] = None             # TODO: confirm
    num_rounds_total: Optional[int] = None             # TODO: confirm (do NOT assume London's 4)

    # --- Target ---
    primary_target_column: Optional[str] = None        # TODO: confirm exact column name
    target_horizon: Optional[int] = None                # TODO: confirm (in expeds/periods)
    target_encoding: Optional[str] = None               # TODO: confirm (continuous/binned/quintile/etc.)
    auxiliary_targets: List[str] = field(default_factory=list)  # TODO: confirm if any exist

    # --- Features ---
    feature_columns: List[str] = field(default_factory=list)     # TODO: confirm full catalogue
    feature_encoding: Optional[str] = None              # TODO: confirm (e.g. "cross-sectional quintile bins")
    missing_value_sentinel: Optional[Any] = None         # TODO: confirm (e.g. is -1 always "missing"?)
    feature_availability_notes: Optional[str] = None     # TODO: confirm whether features change over time

    # --- Data structure ---
    exped_column: Optional[str] = None                   # TODO: confirm structural grouping column name
    id_column: Optional[str] = None                      # TODO: confirm (opaque? meaningful?)
    train_file_path: Optional[str] = None                 # TODO: filled in once data is provided by user
    validation_file_path: Optional[str] = None            # TODO: filled in once data is provided by user
    live_file_path: Optional[str] = None                  # TODO: filled in once data is provided by user
    benchmark_column: Optional[str] = None                # TODO: confirm if a benchmark column/file exists

    # --- Scoring ---
    scoring_metrics: List[str] = field(default_factory=list)  # TODO: confirm exact metric names used by NYC
    metric_weights: Dict[str, float] = field(default_factory=dict)  # TODO: confirm current blend weights
    clipping_rules: Optional[str] = None                   # TODO: confirm (what gets clipped, at what value/pctile)
    rank_metric: Optional[str] = None                       # TODO: confirm if different from blended score

    # --- Submission mechanics (NOT used by this research codebase) ---
    submission_format_notes: Optional[str] = None           # TODO: confirm (documented only, not acted on here)
    submission_limits_notes: Optional[str] = None            # TODO: confirm
    model_artifact_requirements: Optional[str] = None         # TODO: confirm (pickle/ONNX/etc.)

    def confirmed_fields(self) -> Dict[str, Any]:
        """Return only the fields that have actually been filled in (non-None, non-empty)."""
        out = {}
        for k, v in self.__dict__.items():
            if v not in (None, [], {}, ""):
                out[k] = v
        return out

    def missing_fields(self) -> List[str]:
        """Return field names that are still unconfirmed placeholders."""
        return [k for k, v in self.__dict__.items() if v in (None, [], {}, "")]

    def ready_for_modeling(self) -> bool:
        """
        Minimal gate: do not consider this event spec usable for modeling
        until the essentials are confirmed. This is intentionally strict.
        """
        required = [
            self.primary_target_column,
            self.target_horizon,
            self.exped_column,
            self.feature_columns,
            self.scoring_metrics,
        ]
        return all(bool(x) for x in required)


# A module-level instance representing "current knowledge." Starts empty.
CURRENT_EVENT_SPEC = EventSpec()
