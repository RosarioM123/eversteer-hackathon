# Everesteer Quantitative Hedge Fund Hackathon

Live quant-trading competition: grow a $50 stake across four sealed live rounds. Final result: **10th of 36, $50.00 to $73.93 (+48%)**, with all four rounds profitable.

## Final result

| | |
|---|---|
| Starting stake | $50.00 |
| Final balance | $73.93 |
| Net profit | +$23.93 (+48%) |
| Leaderboard | 10th of 36 |
| Final model | euler-unhedged-coupon (Score 0.1802) |

## Round by round

| Round | Staked | P&L | What ran |
|---|---|---|---|
| R1 | $16.00 | +$0.68 | Three-model split. Negated E1 scored +0.1720 on practice but -0.3808 live. |
| R2 | $45.61 | +$9.75 | euler-unhedged-coupon, raw training direction |
| R3 | $60.43 | +$5.83 | euler-unhedged-coupon + treynor-frugal-basis (raw E1) |
| R4 | $66.26 | +$7.67 | boltzmann-robust-rebound, a 97.83% R² LightGBM clone of the v1_sherpa benchmark |

## How the research ran

Data: 178 rank-binned features over 6,522 time-ordered periods, six splits (labeled train, validation, four sealed live rounds with blanked targets). Models: Ridge, LightGBM, and ensembles, with embargoed validation and leakage checks. Protocol: discover, test, reject, confirm, combine, validate, submit.

The durable lesson: practice scores invert on live rounds (the same model scored +0.17 on practice and -0.38 live), so live rounds always run training-direction signals. Full writeup in [playbook.md](playbook.md).

## Docs

- [HACKATHON.md](HACKATHON.md) — how the event works: data, scoring, staking, rounds
- [PERFORMANCE.md](PERFORMANCE.md) — road to 10th place: round-by-round trades and P&L
- [playbook.md](playbook.md) — durable live-round lessons: verified scoring, practice/live inversion, speed-over-research mode

## Repo layout

- `*.parquet` — training and live round data
- `final/` — models, predictions, and submission artifacts
- `*.py` — model building scripts
