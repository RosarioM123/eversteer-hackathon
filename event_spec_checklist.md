# Event Spec Checklist
Fill this in during the first 30 minutes. No modeling until it's done.

## Target
- Target column name:
- Target horizon (periods ahead):
- Target encoding (continuous / binned / quintile / rank):
- Auxiliary targets (if any):

## Data structure
- Exped / time column name:
- ID column name (opaque or meaningful?):
- # rows / # features / # expeds:
- Train file:
- Live/prediction file:

## Features
- Feature columns (count + naming pattern):
- Feature encoding (raw / quintile bins / z-scored):
- Missing value convention (sentinel? e.g. -1):
- Do features change availability over time?

## Scoring (get EXACT formulas)
- CORR definition (Pearson / Spearman / rank-based?):
- AIMC definition:
- NCORR definition (neutralization procedure):
- Blend weights:
- Clipping rules:
- Rank metric (if different from blended):

## Benchmark
- Benchmark column or file:
- What is it (prior model? simple heuristic?):

## Submission
- Submission format (columns, order):
- Deadline per round:
- # live rounds tonight:
- Staking mechanics (how do you choose stake per model?):

## Leakage sanity checks
- Are all features knowable at prediction time? (ask if unsure)
- Embargo needed between train and validation (>= horizon): YES
