# Everesteer NYC Hackathon

Quant research for the Everesteer NYC hackathon competition.

## Structure
- `*.parquet` - Training and live round data
- `final/` - Models, predictions, and submission artifacts
- `*.py` - Model building scripts

## Models
- **E1**: 0.5*Ridge + 0.5*LightGBM (primary live model)
- **Sherpa Clone**: Ridge/LightGBM trained to replicate v1_sherpa benchmark (97.83% R²)

## Rounds
- R1: Negated E1 (-0.3808 live, regime flip lesson)
- R2: Raw E1 (training direction)
- R3: Raw E1
- R4: Sherpa clone / E1 raw
