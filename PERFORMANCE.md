# Road to 10th Place — goatengine ("The Goat")

## Final Result — Event Complete
| | Amount |
|---|---|
| Starting stake | $50.00 |
| Final balance | **$73.93** |
| Net profit | **+$23.93 (+48%)** |
| Leaderboard | **10th of 36** |
| Final model | euler-unhedged-coupon (Score 0.1802) |

## Round-by-Round

### Round 1 — The Regime Flip Lesson
- **Staked:** $16.00 → **Result:** +$0.68 (+4%)
- Models: fermat-bayesian-liquidity ($7.00 → +$0.38), godel-unhedged-edge ($3.00 → +$0.12), wald-sparse-arbitrage ($6.00 → +$0.19)
- **Key lesson:** Negated E1 scored +0.1720 on practice but **-0.3808 live**. Practice and live are inverted. Switched to training-direction (raw) for all live rounds.

### Round 2 — Recovery
- **Staked:** $45.61 on euler-unhedged-coupon → **Result:** +$9.75 (+21%)
- Euler scored +0.358 in R1 (#4 of 201 models).

### Round 3 — Both Models Green
- **Staked:** $60.43 → **Result:** +$5.83 (+10%)
- euler-unhedged-coupon: $30.00 → +$2.50
- treynor-frugal-basis (E1 raw): $30.43 → +$3.33
- Round rank: 10th of 32

### Round 4 — Cloning the Benchmark
- **Staked:** $66.26 on boltzmann-robust-rebound (sherpa clone) → **Result:** +$7.67 (+12%)
- **Final balance:** $73.93
- The clone tracked v1_sherpa (0.992 correlation). Benchmark finished #1 at $97.28.

## Models
- **E1** = 0.5×Ridge + 0.5×LightGBM — primary live model (treynor-frugal-basis)
- **Euler** = raw training-direction LightGBM — best sealed-round scorer (+0.358 R1)
- **Sherpa Clone** — 97.83% R² LightGBM replicating v1_sherpa (boltzmann-robust-rebound / pareto-overfit-likelihood)
- Rejected: pure Ridge (-0.1446), pure LightGBM (-0.1686), E1 70/30 (-0.1672) — all worse than E1 on practice.

## The Core Lesson
**Practice leaderboard is inversely correlated with live performance.** What wins on practice (negation, +0.17) loses on live (-0.38). The benchmark itself proves it: #1 live, negative on practice. For live rounds, always use training-direction (raw).
