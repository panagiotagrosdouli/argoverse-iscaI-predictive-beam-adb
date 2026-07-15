# Argoverse ISCAI: Predictive Beam and ADB

Research prototype for **uncertainty-aware trajectory-to-beam and illumination control** using the Argoverse 2 Motion Forecasting Dataset.

## Objective

The project extends a PC-FMCW vehicular ISCAI pipeline with:

1. real-world actor trajectories from Argoverse 2,
2. short-horizon trajectory prediction,
3. uncertainty-aware adaptive Top-K beam selection,
4. predictive adaptive driving beam (ADB) shadow zones.

The original PC-FMCW/DPSK, range-Doppler and CA-CFAR subsystem is treated as Part A. This repository implements Part B.

## Five-day scope

- Argoverse 2 scenario loading and preprocessing
- Ego-centric coordinate conversion
- Constant-position and constant-velocity baselines
- Kalman prediction with covariance
- ADE, FDE and angular-error metrics
- 16-beam geometric codebook
- Adaptive Top-K selection
- Predictive ADB angular shadow zones
- Reproducible figures and CSV metrics

A GRU predictor is optional after the deterministic pipeline is stable.

## Repository layout

```text
configs/              experiment configuration
data/                 local dataset instructions; raw data are ignored
scripts/              runnable command-line programs
src/iscaI/            reusable Python package
tests/                unit tests
outputs/              generated figures and metrics
```

## Dataset layout

Download the **Argoverse 2 Motion Forecasting Dataset** separately. Do not commit dataset files.

Expected layout:

```text
data/raw/train/<scenario_id>/scenario_<scenario_id>.parquet
data/raw/val/<scenario_id>/scenario_<scenario_id>.parquet
data/raw/test/<scenario_id>/scenario_<scenario_id>.parquet
```

The loader also accepts a direct path to one scenario parquet file.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## First run

Inspect one scenario and create a trajectory plot:

```bash
python scripts/inspect_scenario.py \
  --scenario data/raw/train/<scenario_id>/scenario_<scenario_id>.parquet \
  --output outputs/figures/scenario.png
```

Run the first end-to-end baseline:

```bash
python scripts/run_baseline_demo.py \
  --scenario data/raw/train/<scenario_id>/scenario_<scenario_id>.parquet \
  --output-dir outputs/demo
```

The demo performs:

```text
AV2 trajectory -> ego-centric coordinates -> constant-velocity prediction
-> future angle -> beam index -> predictive ADB shadow interval
```

## Scientific evaluation

Trajectory metrics:

- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Mean angular error

Beam metrics:

- Top-1 accuracy
- Top-K coverage
- average number of probed beams
- overhead reduction relative to exhaustive search

ADB metrics:

- angular interval IoU
- shadow-zone violation rate
- over-masking width

## Important limitation

The initial beam labels are geometry-derived from actor azimuth and not measured optical channel labels. This must be stated explicitly in the report. DeepSense or ray-traced validation is future work.

## License and data

Code in this repository is for academic research. Argoverse 2 data remain subject to their original dataset terms and are not redistributed here.
