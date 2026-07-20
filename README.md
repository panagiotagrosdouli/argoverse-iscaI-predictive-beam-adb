<h1 align="center">Argoverse ISCAI: Predictive Beam Management and ADB</h1>
<h3 align="center">Uncertainty-aware trajectory prediction for vehicular beam selection and adaptive illumination</h3>

<p align="center">
  <strong>Argoverse 2 • Motion Prediction • Uncertainty Propagation • Adaptive Top-K • Predictive ADB</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"></a>
  <a href="https://www.argoverse.org/av2.html"><img src="https://img.shields.io/badge/Dataset-Argoverse%202-orange" alt="Argoverse 2"></a>
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-purple" alt="Research Prototype">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"></a>
</p>

<p align="center">
  <a href="README_GR.md">Greek documentation</a>
</p>

> A research proof-of-concept that uses real traffic trajectories to study whether short-horizon motion prediction and uncertainty can reduce directional beam-search overhead and create safer predictive Adaptive Driving Beam (ADB) shadow zones.

<p align="center">
  <img src="docs/image2.png" width="100%" alt="Predictive beam management and ADB graphical abstract">
</p>

---

## At a glance

### Research question

> Can predicted actor motion and its uncertainty reduce directional beam-search overhead while maintaining beam coverage and predictive illumination safety?

### What I implemented

- Argoverse 2 trajectory loading and ego-centric preprocessing.
- Constant-velocity and Kalman trajectory predictors.
- Cartesian-to-angular uncertainty propagation.
- A geometry-derived directional beam codebook.
- Probabilistic adaptive Top-K beam selection.
- Predictive ADB angular shadow zones.
- An inference-time risk estimator that changes beam and ADB conservatism.
- Batch experiments, bootstrap confidence intervals, failure analysis, and worst-case ranking.

### What this prototype demonstrates

The project demonstrates a complete control-level pipeline from real annotated vehicle trajectories to uncertainty-aware beam and illumination decisions. It does **not** yet claim measured optical-channel performance or a full photometric ADB implementation.

---

## Research context

This repository was developed as a possible extension of a vehicular integrated sensing, communication, and illumination architecture based on a PC-FMCW laser headlamp.

The original sensing and communication chain can be viewed as:

```text
PC-FMCW / DPSK transmission
            ↓
Range-Doppler processing
            ↓
Detection and tracking
            ↓
Communication and illumination control
```

This project investigates the higher-level predictive extension:

```text
Argoverse 2 actor trajectories
              ↓
Short-horizon motion prediction
              ↓
Angular uncertainty propagation
              ↓
Adaptive Top-K directional beam selection
              ↓
Predictive uncertainty-aware ADB shadow zones
```

The current experiments use Argoverse annotations in place of detections produced by a PC-FMCW sensing front end. This isolates the motion-prediction and control problem before future end-to-end integration.

---

## Main methodological contribution

A single predicted position is often too optimistic for directional beam control. This prototype therefore propagates position uncertainty into angular space and uses the resulting probability distribution for two decisions:

1. **Beam management:** select the smallest set of candidate beams whose combined probability reaches a requested coverage target.
2. **Predictive ADB:** create a future angular shadow interval whose width reflects the predicted uncertainty.

The risk-aware extension changes the requested confidence level according to motion and geometry conditions available at inference time.

```text
recent actor history
        ↓
trajectory predictor
        ↓
future position + covariance
        ↓
angular mean + angular uncertainty
        ↓
┌──────────────────────┬────────────────────────┐
│ Adaptive Top-K beams │ Predictive ADB interval│
└──────────────────────┴────────────────────────┘
```

---

## Current experimental results

The following results were obtained on the first 100 Argoverse 2 training scenarios processed by the batch pipeline.

### Trajectory prediction

| Predictor | ADE ↓ | FDE ↓ | Mean angular error ↓ |
|---|---:|---:|---:|
| Constant velocity | **1.910 m** | **5.005 m** | **3.931°** |
| Kalman constant velocity | 2.337 m | 5.602 m | 5.080° |

Constant velocity produced lower ADE in **73%** of scenarios and lower FDE in **69%**. The paired mean ADE difference, `Kalman − CV`, was **0.427 m**, with a 95% bootstrap confidence interval of **[0.112, 0.744] m**.

This result is useful rather than contradictory: filtering does not automatically improve forecasting when the assumed constant-velocity process model does not match manoeuvring traffic.

### Beam and predictive ADB control

| Metric | Result |
|---|---:|
| Top-1 beam accuracy | 86.35% |
| Adaptive Top-K coverage | 96.03% |
| Average selected beams | 2.27 / 16 |
| Beam-search overhead reduction | 85.80% |
| Predictive ADB angular coverage | 95.23% |

These are **geometry-derived control metrics**. They are not measured optical beam-power, SNR, or BER results. Validation with measured or ray-traced channel labels remains future work.

---

## Why risk-aware control?

Average performance can hide rare but safety-relevant failures. In the 100-scenario experiment:

- 35 scenarios contained at least one Top-1 beam failure,
- 11 scenarios had adaptive Top-K coverage below 100%,
- 12 scenarios had predictive ADB coverage below 100%,
- several worst cases showed large angular errors despite strong median performance.

The inference-time risk score combines normalized indicators for:

- predicted angular uncertainty,
- recent acceleration,
- recent turn rate,
- close-range angular sensitivity,
- proximity to the beam-codebook field-of-view boundary.

The score is written as

\[
R = w_\sigma r_\sigma + w_a r_a + w_\omega r_\omega + w_d r_d + w_f r_f.
\]

It changes both the requested beam probability coverage and the ADB confidence multiplier:

| Risk score | Beam probability target | ADB confidence scale |
|---:|---:|---:|
| `< 0.25` | 90.0% | 1.64σ |
| `0.25–0.50` | 95.0% | 1.96σ |
| `0.50–0.75` | 98.0% | 2.33σ |
| `≥ 0.75` | 99.5% | 2.58σ |

The intended behaviour is simple: spend additional beam probes and create wider protective shadow zones only when the estimated risk is high.

---

## Scientific evaluation

### Trajectory prediction

- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Mean angular prediction error

### Beam management

- Top-1 beam accuracy
- Adaptive Top-K coverage
- Average number of selected beams
- Search-overhead reduction relative to exhaustive search
- Coverage-overhead trade-off

### Predictive ADB

- Angular shadow coverage
- Shadow-zone violation rate
- Angular interval width and over-masking

### Statistical analysis

- Paired predictor comparisons
- Bootstrap 95% confidence intervals
- Failure counts and tail analysis
- Worst-case scenario ranking

---

## Repository structure

```text
.
├── configs/                       Experiment configuration
├── data/                          Dataset instructions; raw AV2 data are ignored
├── docs/
│   ├── image2.png                 Research infographic
│   └── graphical_abstract.svg     Vector graphical abstract
├── scripts/
│   ├── inspect_scenario.py        Visualize one AV2 scenario
│   ├── run_baseline_demo.py       End-to-end single-scenario demo
│   ├── run_batch_baselines.py     Batch baseline evaluation
│   ├── analyze_batch_results.py   Bootstrap statistics and worst cases
│   └── run_risk_aware_batch.py    Fixed vs risk-aware comparison
├── src/iscai/
│   ├── adb.py                     Predictive angular shadow zones
│   ├── beam.py                    Beam codebook and adaptive Top-K
│   ├── data.py                    AV2 loading and actor-track extraction
│   ├── evaluation.py              Trajectory and angular metrics
│   ├── geometry.py                Ego-centric coordinate transformations
│   ├── prediction.py              CV and Kalman predictors
│   └── risk.py                    Inference-time risk estimator
├── tests/                         Unit tests
└── outputs/                       Generated figures, CSVs, and JSON metrics
```

---

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/panagiotagrosdouli/argoverse-iscaI-predictive-beam-adb.git
cd argoverse-iscaI-predictive-beam-adb

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Point the project to an existing Argoverse 2 installation:

```bash
export AV2_ROOT=/home/<user>/Datasets/Argoverse2
```

Expected dataset layout:

```text
$AV2_ROOT/
├── train/<scenario_id>/scenario_<scenario_id>.parquet
├── val/<scenario_id>/scenario_<scenario_id>.parquet
└── test/<scenario_id>/scenario_<scenario_id>.parquet
```

Argoverse data are not included in this repository and must not be committed.

---

## Quick start

### Inspect one scenario

```bash
SCENARIO=$(find "$AV2_ROOT/train" -name "scenario_*.parquet" | head -n 1)

python scripts/inspect_scenario.py \
  --scenario "$SCENARIO" \
  --output outputs/figures/scenario.png
```

### Run one end-to-end demonstration

```bash
python scripts/run_baseline_demo.py \
  --scenario "$SCENARIO" \
  --output-dir outputs/demo
```

Generated files:

```text
outputs/demo/trajectory_prediction.png
outputs/demo/metrics.json
```

### Run the 100-scenario baseline experiment

```bash
python scripts/run_batch_baselines.py \
  --dataset-root "$AV2_ROOT" \
  --split train \
  --max-scenarios 100 \
  --output-dir outputs/batch_100
```

### Analyze confidence intervals and worst cases

```bash
python scripts/analyze_batch_results.py \
  --csv outputs/batch_100/per_scenario_metrics.csv \
  --output-dir outputs/batch_100/analysis
```

### Compare fixed and risk-aware control

```bash
python scripts/run_risk_aware_batch.py \
  --dataset-root "$AV2_ROOT" \
  --split train \
  --max-scenarios 100 \
  --output-dir outputs/risk_aware_100
```

---

## Methodological limitations

1. Beam labels are derived from relative actor azimuth and a geometric codebook.
2. The experiments do not use measured optical beam-power vectors.
3. Actor trajectories come from dataset annotations rather than a PC-FMCW detection and tracking front end.
4. The ADB evaluation is angular and does not include a complete photometric or SAE J3069 simulation.
5. The risk policy is interpretable but hand-calibrated.
6. Results are based on a limited initial batch of 100 training scenarios and require broader validation.

These limitations separate what has been demonstrated from what remains a research direction.

---

## Research roadmap

- [x] Real-world Argoverse 2 trajectory pipeline
- [x] Constant-velocity and Kalman baselines
- [x] Uncertainty propagation to beam probabilities
- [x] Predictive ADB angular control
- [x] Batch evaluation and bootstrap analysis
- [x] Worst-case scenario analysis
- [x] Inference-time risk estimator
- [x] Risk-adaptive Top-K and ADB confidence
- [ ] Calibrate the risk policy on validation data
- [ ] Add pedestrian/cyclist class-aware safety margins
- [ ] Compare GRU or Transformer trajectory predictors
- [ ] Validate with DeepSense, measured, or ray-traced beam labels
- [ ] Integrate a PC-FMCW detection and tracking front end
- [ ] Add full optical and photometric ADB evaluation

---

## Research status

This is an active academic research prototype. Its purpose is to test mechanisms, quantify trade-offs, and identify failure modes—not to claim a validated production system. Reproducibility and a clear distinction between measured, annotated, and geometry-derived quantities are design priorities.

## Citation

A `CITATION.cff` file is included for software citation. A paper-style BibTeX entry can be added when an accompanying manuscript is finalized.

## License and data terms

The source code is released under the MIT License. Argoverse 2 data are not redistributed and remain governed by the original dataset license and terms.
