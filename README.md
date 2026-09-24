# Everesteer Quantitative Hedge Fund Hackathon

Quant research for the Everesteer Quantitative Hedge Fund Hackathon competition.

## Docs
- [HACKATHON.md](HACKATHON.md) — how the event works: data, scoring, staking, rounds
- [PERFORMANCE.md](PERFORMANCE.md) — road to 10th place: round-by-round trades and P&L

## Structure
- `*.parquet` - Training and live round data
- `final/` - Models, predictions, and submission artifacts
- `*.py` - Model building scripts

## Models
- **E1**: 0.5*Ridge + 0.5*LightGBM (primary live model)
- **Euler**: raw training-direction LightGBM (best sealed-round scorer)
- **Sherpa Clone**: LightGBM trained to replicate v1_sherpa benchmark (97.83% R²)

## Rounds
- R1: Negated E1 (-0.3808 live, regime flip lesson)
- R2: Raw E1 (+$9.75)
- R3: Raw E1 (+$5.83)
- R4: Sherpa clone (+$7.67)
