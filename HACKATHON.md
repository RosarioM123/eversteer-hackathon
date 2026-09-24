# Everesteer Quantitative Hedge Fund Hackathon

## What It Is

Everesteer is a prediction tournament for quantitative researchers and their autonomous agents. You train a model on an obfuscated financial dataset, submit predictions, and are scored out-of-sample against an answer key you never receive. The models that rank highest drive a real hedge fund.

The data is realistic and obfuscated: features are renamed and binned, instruments are identified by opaque ids, and the target is a rank-based forward return over a fixed horizon. You never see the live answer key, so you cannot overfit to it. You compete on prediction, not on data access.

## Event Structure

- **Format:** Closed, time-boxed competition — a sequence of 4 sealed rounds (R1–R4), each with its own live data split and scoring window.
- **Build phase:** Train models before rounds open. Practice board (validation split, server-scored) is display-only and open in every phase — it does not count toward the event.
- **Cumulative standings:** Sum of per-round clipped scores across all rounds. A missed round is a zero you cannot make up.
- **Winner:** On money events, the winner is decided by **final recorded stake balance**, not the standings table.

## The Data

- **178 features**, each rank-binned into 10 bins (0–9) within its own exped (trading day). `-1` marks unavailable — treat as missing, never as an ordinal value. No units, no outliers.
- **15 targets**, each a 5-level ordinal (0.00 / 0.25 / 0.50 / 0.75 / 1.00) in a bell distribution. Only **target_everest** is scored. The other 14 are auxiliary (some near-duplicates, useful for multi-target modeling).
- **Splits:** Train = 6,522 expeds / 468,236 rows (labeled). Validation = 261 expeds (targets blanked). R1–R4 = ~261 expeds each (targets blanked, sealed). Each round's live split is a disjoint id namespace — re-download every round.
- Row ids are re-keyed daily: no instrument time series. Pure cross-sectional prediction.
- **Benchmark:** `v1_sherpa` — the platform's reference model, downloadable per split. All AIMC scoring is measured against it.

## Scoring

Per round, per model:

```
Round Score = clip(CORR + 2·AIMC + NCORR, ±1)
```

- **CORR:** Rank correlation with target_everest. Computed per exped after rank-gaussianizing predictions and applying a signed 1.5 power transform on both sides — tails matter more than the middle. Not Spearman.
- **AIMC (2× weight):** Covariance of centered target with your predictions *after removing the component along v1_sherpa*. Copying the benchmark scores ~0. Rewards signal the benchmark doesn't have.
- **NCORR:** CORR after neutralizing predictions against a frozen train-selected core feature set. Rewards signal that isn't a linear function of core features.

Computed per exped, then averaged. Degenerate expeds score 0.

## Staking

- Stake USDC against your own predictions, per model per round. Drafts are editable while the round window is open; locked on-chain at close.
- **Payout:** `stake × 0.75 × tanh(score / 0.75)` — one round can move a stake by at most ~75%. Stake size changes payout, never leaderboard rank.
- Only the platform-granted starting stake ($50) can be staked. Winnings compound into future rounds.

## Key Strategic Lessons

1. **Practice ≠ live.** The practice board is inversely correlated with sealed-round performance. What wins on practice (e.g., negated models, +0.17) can lose badly on live (−0.38). The benchmark itself was #1 on live money while scoring negative on validation.
2. **AIMC is the edge.** At 2× weight, orthogonal signal beyond the benchmark is the highest-leverage term. But it must still predict the target — low benchmark correlation alone isn't enough.
3. **Consistency compounds.** Lower-turnover, stable models accumulate better over rounds than boom-or-bust bets. A bad round can take as much as a good round adds.
4. **Never skip a round.** Cumulative standings sum all four rounds; a zero can't be recovered.
