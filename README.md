# Simplified-EPV-Evaluation

This repository contains a simplified implementation of the framework for evaluating the instantaneous expected value (EPV) of soccer possessions, based on the PFF data. The goal is to provide a straightforward approach to analyzing soccer possession data and estimating its value.

## Overview

The provided implementation streamlines the process of calculating EPV for soccer possessions using data from PFF. This approach simplifies the methodology discussed in the paper "A Framework for the Fine-Grained Evaluation of the Instantaneous Expected Value of Soccer Possessions."

## Features

- Simplified EPV calculation
- Based on PFF data
- Easy-to-understand code and implementation

## Installation

To use this repository, clone it to your local machine:

```bash
    git clone https://github.com/dtchaves/Simplified-EPV-Evaluation.git
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

If you are using the repository virtual environment, activate it before running the training or test commands:

```bash
source .venv/bin/activate
```

**Important:** This project requires `pyarrow>=10.0.0` for Parquet data support. It is included in `requirements.txt`.

## Data Format

The project uses **Parquet files** as the primary data format. During loading, pass tracking files are enriched with `pass_outcome_type` from the raw event JSON and cached artifacts are written under `data/processed/cache/`.

## Training and Testing Models

The framework consists of four neural network models, each targeting a specific EPV aspect:

### 1. Pass Success Probability (PP)
Predicts whether a pass will be successful using spatial features.

**Train:**
```bash
cd src/Pass/Pass_sucess_probability
python -m trainer
```

**Test:**
```bash
cd src/Pass/Pass_sucess_probability
python -m test
```

### 2. Pass Selection Probability (PS)
Estimates the distribution of pass destinations on the field.

**Train:**
```bash
cd src/Pass/Pass_selection_probability
python -m trainer
```

**Test:**
```bash
cd src/Pass/Pass_selection_probability
python -m test
```

### 3. Pass EPV (Successful) - PE-Success
Estimates expected value for completed passes (pass_outcome_type = "C").

**Train:**
```bash
cd src/Pass/Pass_epv_success
python main.py train
```

**Test:**
```bash
cd src/Pass/Pass_epv_success
python main.py test
```

**Train + Test:**
```bash
cd src/Pass/Pass_epv_success
python main.py train_test
```

### 4. Pass EPV (Missed) - PE-Missed
Estimates expected value for missed passes (pass_outcome_type in {D, B, O, S, G, I}).

**Train:**
```bash
cd src/Pass/Pass_epv_missed
python main.py train
```

**Test:**
```bash
cd src/Pass/Pass_epv_missed
python main.py test
```

**Train + Test:**
```bash
cd src/Pass/Pass_epv_missed
python main.py train_test
```

## Pre-Training Validation

Before running full training, validate data compatibility using the smoke test:

```bash
cd src/Pass
python smoke_test.py
```

This verifies:
- Parquet files are discoverable and readable
- All dataloaders load non-zero samples
- Tensor shapes are correct (13 or 16 channels, 68×104 spatial)
- Reward labeling functions correctly

## Cache Behavior

- By default, Pass dataloaders use a small shared cache policy: canonical enriched pass files are cached under `data/processed/cache/canonical/`, while large tensor-shard caches are skipped.
- The canonical cache is reusable across PP, PS, PE-Success, PE-Missed, and Pass audit scripts because it stores source-level pass rows after PFF triplet canonicalization and outcome enrichment.
- Match split manifests under `data/processed/cache/splits/` are tiny and remain useful for consistent train/validation/test assignment.
- PP and PS can optionally share a large tensor cache family under `data/processed/cache/pp_ps/` with artifact stem `pass_outcome_tensors`.
- PE-Success and PE-Missed can optionally write large EPV tensor shards under `data/processed/cache/epv/` with separate success/missed artifact stems.
- EPV tensor cache entries are keyed by the PP checkpoint fingerprint, so retraining or replacing the PP model invalidates dependent EPV caches automatically.
- Set `PFF_DISABLE_CACHE=1` to disable all caches. Set `PFF_CACHE_ENABLED=1` to enable all caches, including the large tensor-shard caches.

## Memory-Friendly Match Sampling

- Pass dataloaders and the BallDrive, Shot, baseline xG, and ActionSelection match-triplet builders sample a balanced fraction of each season before loading source matches.
- The default `PFF_SEASON_SAMPLE_RATIO` is `0.5`, so half of each discovered season is loaded by default.
- Set `PFF_SEASON_SAMPLE_RATIO=1` to use all matches, or set another value such as `0.25` for a smaller run.
- The Python parameter is `season_sample_ratio`; an explicit dataloader/config value overrides the environment default.

## Complete Training Sequence

To train and test all four models in dependency order (PP → PS → PE-Success → PE-Missed):

```bash
cd src/Pass

# Run smoke tests first
python smoke_test.py

# Train and test Pass Success Probability
cd Pass_sucess_probability && python -m trainer && python -m test && cd ..

# Train and test Pass Selection Probability
cd Pass_selection_probability && python -m trainer && python -m test && cd ..

# Train and test PE-Success
cd Pass_epv_success && python main.py train_test && cd ..

# Train and test PE-Missed
cd Pass_epv_missed && python main.py train_test && cd ..

echo "All models trained and tested successfully!"
```

Train PP before either EPV model. The EPV dataloaders include a PP-derived input surface and default to the checkpoint path `results/models/Pass_success_probability.pt`.

## Output Artifacts

All trained models and evaluation outputs are saved in the `results/` directory:

```
results/
├── models/
│   ├── Pass_success_probability.pt
│   ├── Pass_selection_probability.pt
│   ├── pass_epv_success/
│   │   └── Pass_epv_success.pt
│   └── pass_epv_missed/
│       └── Pass_epv_missed.pt
├── loss/
│   ├── Pass_success_probability_loss_curves.png
│   └── Pass_selection_probability_loss_curves.png
├── metrics/
│   ├── Pass_success_probability_metrics.png
│   ├── Pass_selection_probability_metrics.png
│   ├── Pass_epv_success_metrics.png
│   ├── Pass_epv_success_metrics.csv
│   ├── Pass_epv_missed_metrics.png
│   └── Pass_epv_missed_metrics.csv
└── heatmaps/
    ├── Pass_success_probability_*.png
    ├── Pass_selection_probability_*.png
    ├── Pass_epv_success_*.png
    └── Pass_epv_missed_*.png
```

## Architecture Notes

- **PP & PS Models:** Classification networks predicting pass success (binary) and pass destination heatmap
- **PE-Success & PE-Missed Models:** Regression networks estimating expected value conditioned on pass outcome
- All models use SoccerMap architecture with 13 input channels (spatial features) for PP/PS and 16 channels for PE models
- Spatial resolution: 68×104 pixels (field discretization)

## References

1. Fernández, J., Bornn, L., & Cervone, D. (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning, 110(6), 1389--1427. Springer. DOI: 10.1007/s10994-021-05989-6 [https://arxiv.org/abs/2011.09426]

2. Fernández, J., & Bornn, L. (2020). SoccerMap: A Deep Learning Architecture for Visually-Interpretable Analysis in Soccer. [https://arxiv.org/abs/2010.10202]