# Road to 10th Place — goatengine ("The Goat")

## P&L Summary
| | Amount |
|---|---|
| Starting stake | $50.00 |
| Current balance | $66.26 |
| Net profit | **+$16.26 (+32.5%)** |
| Leaderboard | **10th place** |

## Round-by-Round

### Round 1 — The Regime Flip Lesson
- **Staked:** $16.00 → **Result:** $0.68 (loss)
- Models: fermat-bayesian-liquidity ($7.00 → +$0.38), godel-unhedged-edge ($3.00 → +$0.12), wald-sparse-arbitrage ($6.00 → +$0.19)
- **Key lesson:** Negated E1 scored +0.1720 on practice but **-0.3808 live**. Practice and live are inverted. Switched to training-direction (raw) for all live rounds.

### Round 2 — Recovery
- **Staked:** $45.61 on euler-unhedged-coupon → **Payout:** +$9.75 (+21.4%)
- Submitted raw E1 (training direction) to treynor-frugal-basis.

### Round 3 — Both Models Green
- **Staked:** $60.43 → **Payout:** $5.83 (+9.6%)
- euler-unhedged-coupon: $30.00 → +$2.50 (+8.3%)
- treynor-frugal-basis: $30.43 → +$3.33 (+10.9%)
- Treynor (E1 raw) outperformed Euler.

### Round 4 — Cloning the Benchmark
- **Status:** In progress
- **Drafts:** euler $32.50, treynor $33.76
- **Key insight:** Everesteer Benchmark (v1_sherpa) is **#1 on live** ($88.40, +$38.40) but **-0.1096 CORR on practice** — confirming the practice↔live inversion.
- Built a **97.83% R² LightGBM clone** of v1_sherpa (`pareto-overfit-likelihood`) to track the #1 strategy.

## Models
- **E1** = 0.5×Ridge + 0.5×LightGBM — primary live model (treynor-frugal-basis)
- **Sherpa Clone** — 97.83% R² LightGBM replicating v1_sherpa (pareto-overfit-likelihood)
- Rejected: pure Ridge (-0.1446), pure LightGBM (-0.1686), E1 70/30 (-0.1672) — all worse than E1 on practice.

## The Core Lesson
**Practice leaderboard is inversely correlated with live performance.** What wins on practice (negation, +0.17) loses on live (-0.38). The benchmark itself proves it: #1 live, negative on practice. For live rounds, always use training-direction (raw).
