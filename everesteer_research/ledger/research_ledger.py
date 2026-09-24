"""
ResearchLedger: append-only experiment log.

Every experiment run through ExperimentRunner produces exactly one
LedgerEntry. Nothing is ever overwritten or deleted - if an experiment is
re-run, it gets a new experiment_id and a new row, so history is never lost
(per the "never lose a result" requirement).

Status flow (enforced by convention + validate_promotion, not silently
auto-applied):
    UNTESTED -> TESTED -> PROMISING -> CONFIRMED -> REJECTED

A result may only reach CONFIRMED if it has been tested more than once
(e.g. on a second, independent set of folds not used for discovery). This
module checks for that pattern but the caller is responsible for actually
running the second test.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import csv
import os
from datetime import datetime, timezone

VALID_STATUSES = ["UNTESTED", "TESTED", "PROMISING", "CONFIRMED", "REJECTED"]


@dataclass
class LedgerEntry:
    experiment_id: str
    experiment_name: str
    timestamp: str
    hypothesis: str
    branch: str
    data_version: Optional[str]
    target: str
    feature_set: str
    model: str
    hyperparameters: Dict[str, Any]
    training_window: str
    validation_window: str
    embargo: int
    CORR: Optional[float]
    AIMC: Optional[float]
    NCORR: Optional[float]
    blended_score: Optional[float]
    benchmark_corr: Optional[float]
    prediction_corr_to_existing_models: Optional[Dict[str, float]]
    mean_fold_score: Optional[float]
    std_fold_score: Optional[float]
    worst_fold_score: Optional[float]
    compute_time: Optional[float]
    status: str
    decision: str                     # "KEEP" | "PROMISING" | "REJECTED"
    decision_reason: str
    leakage_checks: Dict[str, str]    # check_name -> "PASS"/"FAIL"/"NOT_RUN"
    notes: str = ""

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status '{self.status}' not in {VALID_STATUSES}")
        if self.decision not in ("KEEP", "PROMISING", "REJECTED"):
            raise ValueError(f"decision must be KEEP/PROMISING/REJECTED, got '{self.decision}'")


class ResearchLedger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def append(self, entry: LedgerEntry) -> None:
        entries = self._load()
        entries.append(asdict(entry))
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    def _load(self) -> List[Dict[str, Any]]:
        with open(self.path, "r") as f:
            return json.load(f)

    def all_entries(self) -> List[Dict[str, Any]]:
        return self._load()

    def filter(self, **kwargs) -> List[Dict[str, Any]]:
        """e.g. ledger.filter(branch='signal_discovery', status='PROMISING')"""
        entries = self._load()
        for k, v in kwargs.items():
            entries = [e for e in entries if e.get(k) == v]
        return entries

    def count_tests_for_hypothesis(self, hypothesis: str) -> int:
        """Used to check whether a result has been independently re-tested
        before allowing promotion to CONFIRMED."""
        return len([e for e in self._load() if e.get("hypothesis") == hypothesis])

    def validate_promotion_to_confirmed(self, hypothesis: str) -> bool:
        """A hypothesis needs at least 2 logged tests before CONFIRMED is
        allowed - this operationalizes the discovery-vs-confirmatory rule."""
        return self.count_tests_for_hypothesis(hypothesis) >= 2

    def export_csv(self, csv_path: str) -> None:
        entries = self._load()
        if not entries:
            return
        fieldnames = list(entries[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                       for k, v in e.items()}
                writer.writerow(row)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
