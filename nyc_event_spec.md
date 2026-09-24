# NYC Event Spec — Everesteer Hedge Fund Hackathon (Sept 23, 2026)
Status: CONFIRMED from official Everesteer Docs (Everesteer_Docs.pdf, uploaded
2026-09-23) + platform UI panels. Remaining live values (weights, clip, horizon)
must be read from the API, never hardcoded.

## Event structure (confirmed)
- Lane: EVENT (closed, time-boxed). Event-scoped API key, base URL
  https://hackathon.everesteer.ai. MCP: https://hackathon.everesteer.ai/mcp
  (tools prefixed eiq_; install via /install-claude-mcp.sh on the event host).
- Phases: build → R1 → R2 → R3 → R4. R1 = 30 min. Ends ~22:06 EDT.
- Practice board: validation split, server-scored, display-only, open in every
  phase. NOT one of the scored rounds.
- Cumulative standings = sum of per-round clipped scores across R1–R4.
  Never skip a round: a missed round is a zero you cannot make up.
- On money events the WINNER is decided by final recorded stake balance,
  not the standings table.
- Per-round per-model stake allocation, changeable between rounds (user
  confirmed from dashboard). Drafts editable off-chain while window open;
  locked (immutable, on-chain) at window close. Current shape: window = round.
- max_slots models per round (read live). Only platform-granted money
  ($50 starting stake) can be staked.

## Data (confirmed)
- 178 features, int8, cross-sectional decile bins 0–9 within exped; -1 = missing
  (treat as NaN/category, NEVER ordinal). No units, no outliers.
- 15 targets, 5 levels (0.00/0.25/0.50/0.75/1.00), bell 5/20/49.5/20/5 per exped.
  ONLY primary_target (target_everest on this dataset — always re-read from
  schema) is scored. 14 aux targets: some near-duplicates (~0.84 corr);
  useful for per-target models + blending, train-time only.
- Row ids re-keyed daily: NO instrument time series. Pure cross-sectional.
  Copy ids verbatim; never renumber; id may arrive as column or index.
- Splits: train 6,522 expeds / 468,236 rows (labeled) → val 261 (blanked) →
  R1 261 / R2 260 / R3 261 / R4 261 (blanked, sealed). Universe 44 → 92.
  Each round's live split is a DISJOINT id namespace: re-download every round.
- Benchmark predictions downloadable: train_benchmark_models +
  validation_benchmark_models (live_benchmark_models WITHHELD during event).
  Designated benchmark column: v1_sherpa (float in [0,1]).
- NaN target in train = uncomputable forward return: drop those rows.

## Scoring (confirmed formulas; weights/clip LIVE from API)
- Computed PER EXPED, then averaged. Degenerate exped (no variance / too few
  ids) scores 0.0 for that exped.
- CORR: rank-gaussianize predictions only → center target (no rank transform)
  → signed 1.5 power on BOTH sides → Pearson. Tails matter more than middle.
  NOT Spearman.
- AIMC: rank-gaussianize predictions + v1_sherpa → remove scalar projection of
  predictions along benchmark → covariance of residual with centered target.
  Copying the benchmark scores ~0.
- NCORR: rank-gaussianize predictions → subtract projection onto frozen
  train-selected core feature set (+ intercept; server uses spectrally-anchored
  ridge when ill-conditioned) → CORR kernel on residual. Rewards signal that is
  not a linear function of core features.
- Round score = weighted blend of CORR + AIMC + NCORR, clipped per round
  (symmetric, wide: a single round can take an entire stake).
- Weights/clip/payout factor: GET /api/v1/scoring (no key needed by contract)
  or explain_scoring. NEVER hardcode; re-tuned more than once.
- Observed on practice board: Score = CORR + NCORR exactly on benchmark row
  (weights unknown until explain_scoring is called).
- Benchmark on validation: CORR -0.1096, NCORR -0.0462, Score -0.1558.
- Offline toolkit everestapi[scoring]: corr20/aimc20/ncorr are close
  approximations (corr20 powers predictions only; ncorr uses exact OLS no
  intercept; aimc20 matches server kernel). Use for relative comparisons.

## Evaluation protocol (confirmed)
- Split by EXPED, never by row. Gap >= target horizon between last fit exped
  and first hold-out exped (tournament horizon = 20 trading days → ~20 expeds;
  event dataset horizon: read from schema/metadata, default conservative 20).
- Labels of adjacent expeds overlap (forward-looking target) — the gap is
  what prevents leakage.
- Score per exped with the toolkit kernels, then average = honest round-score
  estimate. Never self-score on validation (targets blanked).

## Submissions (confirmed)
- create_model FIRST (server assigns public name; your label stays private).
- Event round lane: submit_event_predictions (multipart: id+prediction file,
  model_pkl REQUIRED, model_pkl_python_version "major.minor").
- Practice lane: submit_validation_diagnostics (same args). FREE uploads,
  never draws from cap. Round uploads draw from a PER-EVENT pool (not
  per-round, never replenishes). Budget: practice freely, spend round uploads
  deliberately (~1-2 models/round, keep reserve for R4).
- .pkl must be a cloudpickled callable named predict (use cloudpickle.dump,
  not pickle.dump); bare estimators refused. No torch opcodes.
- Predictions: exactly one per round id, finite floats in [0,1] (clip;
  rank-based scoring makes rescaling free). Wrong-lane submissions 202 then
  fail on zero id overlap — always re-check get_started before submitting.
- Byte-identical re-upload coalesces (no extra slot). Duplicate predictions
  across participants → 409.
- Per-agent row cap per board keeps best-scoring models.
- Final selection (set_final_selection): only if get_final_selection()
  .applicable is true; inert otherwise.

## Staking / payout (confirmed)
- Payout per round: clipped blend × payout factor → applied to snapshotted
  stake. Payout factor = 1 while round's total locked stake under threshold,
  shrinks as more stake commits; frozen at first settlement pass.
- Stake return bounded by stake_return_amplitude (fraction of stake).
- Clip symmetric: a bad round can take as much as a good round adds — up to
  the entire stake. Size accordingly.
- get_event_staking() → position; set_stake_allocation(model, amount, window).
- Empirical note from docs: lower-turnover models accumulate better over time.

## Still live (read from API, never assume)
- explain_scoring → weights {corr, aimc, ncorr}, clip, payout factor.
- get_started → lane, uploads_remaining, hosted_train_funded, open window,
  model_python_versions, event_staking block.
- get_dataset_schema → primary_target, feature/target encodings, horizon.
- get_final_selection().applicable → whether final window exists.

## 2026-09-23 data forensics (actual files)

Source files: `eiq_validation.parquet` (23,886 x 195), `eiq_train_benchmark_models.parquet` (468,236 x 2).

- Schema: `id` is the pandas **index** (unique), not a column. Columns: `exped`, `data_type`, 178 x `feature_*` (int8, values -1..9), 15 x `target_*` (float32, fully blanked in validation).
- Validation: 261 expeds `exped_6566`..`exped_6826`, 91-92 rows/exped. Targets 100% null as expected.
- Train benchmark: 6,522 expeds `exped_0003`..`exped_6524`, 468,236 rows (matches dashboard train exactly). `v1_sherpa` in [0.0109, 1.0], mean 0.507, std 0.289 — probability-like, roughly symmetric. No validation benchmark file (withheld, as docs warned).
- **Observed exped gap train->val: 6566-6524 = 42 expeds**, consistent with metadata `embargo_days_calendar: 60` (60 calendar days ~ 42 trading days). Use **42-exped embargo** for internal walk-forward splits.
- **Missingness regime shift**: validation -1 rate mean 0.9% (per-feature 0-5.6%) vs train UI showing 10.8%. Models must be robust to missingness differences; do not over-rely on -1 as signal.
- primary_target confirmed: `target_everest` (metadata platform_target_alias: graded column IS the primary target, no alias).
- MISSING: `eiq_train.parquet` (features + labeled targets) — required before any training. Requested from user 2026-09-23.

## 2026-09-23 local-agent forensics report (Claude Code, has full train file)

- Train confirmed: 468,236 rows, 6,522 expeds, matches dashboard. Local agent reports: target_everest bell 5.09/20.17/49.48/20.17/5.09, zero missing; v1_sherpa 2,461 unique values in [0,1], corr(v1_sherpa, target_everest)=0.123 on train.
- Feature missingness by era: earliest decile ~62.8% -> recent ~3-7%. Universe grew 44 -> 92 rows/exped. -> REQUIRED experiment: expanding vs rolling vs sparse-era-excluded.
- Aux targets: target_Tougroute corr 0.837 with target_everest (PROMISING, needs OOS lift test); two latent clusters corroborated by correlation + identical missingness sets (Anghomar/Ayachi; Gourza/Tiskiouine; Azurki/Tarhat; Igli/Timesguida).
- Naive univariate |corr| tops ~0.06-0.08 (with -1 as ordinal; will change with proper missingness handling).
- DISPUTED: local agent claims eiq_validation.parquet targets are NOT blanked. Direct measurement of uploaded file: 0/358,290 non-null target cells. Pending local recheck `df['target_everest'].notna().sum()`. Design assumes blind validation until resolved.
- Scoring weights/clip still unknown; kernels are doc-specified. Decision 2026-09-23: build baseline ladder under CORR as labeled provisional proxy; pull weights via eiq_explain_scoring in parallel.

## 2026-09-23 scoring docs confirmed (Everesteer_Docs.pdf, 206pp, via local agent)

- target_everest horizon CONFIRMED: 20-trading-day forward return, rank-based, 5 bins.
- Kernels (doc-exact): CORR = per-exped Pearson after rank-gaussianize preds + sign(x)|x|^1.5 on both sides (centered target), averaged. AIMC = covariance of centered target with preds orthogonalized vs v1_sherpa. NCORR = CORR after neutralizing vs frozen train-selected CORE FEATURE SET (undisclosed by design).
- round_score = weighted blend of CORR/AIMC/NCORR + per-round clip. Weights/clip are LIVE-ONLY (explain_scoring / GET /api/v1/scoring), re-tuned, never hardcoded. Not yet fetched.
- Offline deviations: server powers both sides for CORR (SDK only predictions); server NCORR = ridge-with-intercept (SDK = exact pinv, no intercept). Toolkit = relative comparisons only.
- Validation blank CONFIRMED (dispute resolved; agent recheck: 100% NaN, 23,886 rows). Server-scored dry run only.
- Predictions must be in [0,1] (clip). Model .pkl via cloudpickle.dump, callable, declare Python version.
- Local agent installed everestapi[scoring] (public PyPI, scoring submodule only) and wired metrics/scoring.py to corr20/aimc20/ncorr. B0 constant-pred sanity: 0.0/0.0 on CORR20/AIMC — harness plumbing verified.
- DECISION 2026-09-23: walk-forward embargo = 42 expeds (event-specific metadata 60 cal days + observed gap), NOT the generic docs' 20. Fold config: purged, gap 42.
- Selection doctrine until weights arrive: maximize CORR subject to AIMC>0 (not benchmark clone) and proxy-NCORR holding up. Undisclosed core set => favor diffuse/broad regularized models; distrust sparse linear models.
- Authorized: call eiq_explain_scoring (read-only metadata) to fetch live weights/clip.

## 2026-09-23 B0-B5 ladder running (local agent, real train data)

Spec: target_everest, 6,522 expeds, 42-exped embargo, corr20/aimc20 kernels, v1_sherpa orthogonality ref, 178 features (-1=missing).
Per-baseline report: CORR, AIMC, fold-by-fold, mean/std, worst fold, benchmark corr, pred-corr vs existing models, runtime, feature count, leakage/instability warnings. Results -> experiment ledger. No winner picked from aggregate score; interpretation pass (what/why/reliability/next question) happens when table lands here.
Flags raised before run: (1) -1 must be masked as missing in all baselines, imputation fit inside train folds only; (2) folds must span eras incl. sparse early era for temporal-stability read; (3) per-fold corr(preds, v1_sherpa) alongside AIMC for clone-risk detection.

## 2026-09-23 event mechanics from Everesteer_Docs.pdf (direct extraction)

- Event scoring uses the SAME weights as the Himalayas tournament ("on the same weights the Himalayas tournament pays on"). Numeric weights still live-only.
- Held-out final window: check get_diagnostics_leaderboard(window='final'); do not assume either way.
- cadence object (get_started/get_status): branch on cadence.open_window; live_round is null for event keys BY DESIGN.
- Upload pool: per-event (not per-round); practice uploads free; uploads_remaining in 202 response; failed/cancelled runs free a slot; 409 = upload limit. Batch flow: up to 25 items/call.
- Staking fields (micro-USDC ints): staking_armed, phase, stake_window_open, draft_window, max_slots, min_stake_micro, principal_micro, settled_micro, net_micro, forwarder_balance_micro.
- Model .pkl REQUIRED on both event lanes. Round lane: POST /api/v1/event/predictions/upload (multipart). Practice: POST /api/v1/diagnostics/upload (multipart). create_model first (404 otherwise); platform never auto-creates.
- Wrong-lane upload: accepted (202), fails minutes later with zero id overlap; wastes submission + minutes. Call get_started before every submit.
- Event host: https://hackathon.everesteer.ai. Live split 404s between rounds. Re-download live every round.
- Validation = practice board only, never counted toward event result.

## 2026-09-23 WEIGHTS REVERSE-ENGINEERED from live practice board (all 15 rows)

Score = CORR + 2*AIMC + NCORR EXACTLY (max err < 2e-4, incl. benchmark row with AIMC=0).
Effective objective: maximize CORR + 2*AIMC + NCORR. AIMC doubly weighted.
Caveat: docs say weights re-tuned over time; confirm via eiq_explain_scoring when possible.
Leader to beat: participant_1df57bd4dc, 8 models, 0.1432/0.0114/0.0658 = 0.2318.
Best single-model AIMC in field: 0.0153 (participant_f3f4b120a6) -> benchmark-orthogonal signal is learnable.
Field: 5 positive, 10 negative -> hard task; bar is 0.2318, not 0.
