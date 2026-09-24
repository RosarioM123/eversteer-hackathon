# Everesteer event Research Framework (Skeleton)

## Status: PRE-DATA. No event dataset, schema, or scoring rules have been confirmed yet.

This is a **dataset-agnostic, config-driven research harness**. It does not
contain any event-specific column names, target names, feature counts, scoring
weights, or submission rules. Every event-specific value lives in
`data/event_spec.py` and is currently `None` or a `"TODO"` string.

## Hard constraints honored by this codebase

1. No network calls. No API clients. No API keys read from env or anywhere else.
2. No download logic of any kind.
3. No submission logic of any kind — search the repo, there is no `submit()`.
4. No fabricated results. `ExperimentRunner.run()` raises `RuntimeError` if
   called without real `data` supplied by the caller. Nothing here invents
   numbers.
5. Metric formulas (CORR/AIMC/NCORR/blended) are **not implemented** — they
   raise `NotImplementedError` with a `TODO: confirm against event event
   specification` message until we've verified the real formulas.

## How this becomes usable the moment event data arrives

1. Fill in `data/event_spec.py` from the confirmed EVENT SPEC (see the
   checklist we built) — target name, horizon, feature list, missingness
   convention, scoring weights, clipping rules, etc.
2. Implement the real formulas in `metrics/scoring.py`.
3. Point `WalkForwardSplitter` at the real exped column and target horizon.
4. Register experiments in the five branch files under `branches/`.
5. Run experiments via `ExperimentConfig` + `ExperimentRunner.run()` — each
   one logs a row to `ledger/research_ledger.py`'s store automatically.

## Workflow this enforces

```
Hypothesis -> ExperimentConfig -> WalkForwardSplitter -> fit/predict
    -> metrics (CORR/AIMC/NCORR/blended) -> RedTeamChecks
    -> ResearchLedger row -> KEEP / PROMISING / REJECTED
    -> (if KEEP-tier) ModelPortfolio -> EnsembleBuilder
```

Not: "run N models, pick the highest score."

## Multiple-testing discipline

A `PROMISING` status is not enough to trust a result. Promotion to
`CONFIRMED` in the ledger requires a second run on folds that were not used
for discovery — this is enforced by convention (documented in
`ledger/research_ledger.py`), since enforcing it in code requires knowing
the real fold structure, which we don't have yet.
