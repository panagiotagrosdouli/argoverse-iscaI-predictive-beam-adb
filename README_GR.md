# Argoverse ISCAI: Πρόβλεψη Τροχιάς, Adaptive Beam Management και Predictive ADB

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Argoverse%202-orange)](https://www.argoverse.org/av2.html)
[![Research](https://img.shields.io/badge/Κατάσταση-Research%20Prototype-purple)](#κατάσταση-της-έρευνας)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Από συνθετικές τροχιές σε πραγματικές κυκλοφοριακές σκηνές:** βραχυπρόθεσμη πρόβλεψη κίνησης, διάδοση αβεβαιότητας, adaptive Top-K επιλογή δεσμών και predictive Adaptive Driving Beam για οχήματα ISCAI.

[English documentation](README.md)

---

## Περιγραφή

Το repository υλοποιεί το **Μέρος Β** μιας ερευνητικής εργασίας για vehicular PC-FMCW laser-headlamp ISCAI. Το αρχικό σύστημα του Μέρους Α περιλαμβάνει PC-FMCW/DPSK μετάδοση, Range-Doppler επεξεργασία, CA-CFAR ανίχνευση και έλεγχο φωτισμού. Η παρούσα επέκταση εξετάζει πραγματικές τροχιές από το Argoverse 2 και υλοποιεί:

```text
Πραγματικές τροχιές Argoverse 2
              ↓
Βραχυπρόθεσμη πρόβλεψη κίνησης
              ↓
Διάδοση αβεβαιότητας από θέση σε γωνία
              ↓
Adaptive Top-K directional beam probing
              ↓
Predictive uncertainty-aware ADB shadow zones
```

Το βασικό ερευνητικό ερώτημα είναι:

> Μπορεί η πρόβλεψη της μελλοντικής κίνησης και της αβεβαιότητάς της να μειώσει το beam-search overhead, διατηρώντας ταυτόχρονα υψηλή κάλυψη beam και ασφαλή predictive illumination control;

---

## Κύριες συνεισφορές

- Αξιολόγηση σε πραγματικές σκηνές του **Argoverse 2 Motion Forecasting Dataset**.
- Ego-centric μετασχηματισμός τροχιών στα 10 Hz.
- Constant Velocity και Kalman baselines.
- Μετρικές ADE, FDE και angular error.
- Geometry-derived directional beam codebook.
- Πιθανοτική adaptive Top-K επιλογή δεσμών.
- Predictive ADB angular shadow intervals με uncertainty margins.
- Batch αξιολόγηση, bootstrap confidence intervals και worst-case analysis.
- **Inference-time risk estimator** χωρίς χρήση μελλοντικού ground truth.
- Risk-adaptive beam probability coverage και ADB confidence policy.

---

## Τρέχοντα πειραματικά αποτελέσματα

Τα παρακάτω αποτελέσματα προέρχονται από τα πρώτα 100 Argoverse 2 training scenarios που επεξεργάστηκε το batch pipeline.

### Πρόβλεψη τροχιάς

| Predictor | ADE ↓ | FDE ↓ | Μέσο angular error ↓ |
|---|---:|---:|---:|
| Constant Velocity | **1.910 m** | **5.005 m** | **3.931°** |
| Kalman Constant Velocity | 2.337 m | 5.602 m | 5.080° |

Το Constant Velocity είχε χαμηλότερο ADE στο **73%** των scenarios και χαμηλότερο FDE στο **69%**. Η μέση paired διαφορά ADE, `Kalman − CV`, ήταν **0.427 m**, με 95% bootstrap confidence interval **[0.112, 0.744] m**.

### Beam management και predictive ADB

| Μετρική | Αποτέλεσμα |
|---|---:|
| Top-1 beam accuracy | 86.35% |
| Adaptive Top-K coverage | 96.03% |
| Μέσος αριθμός επιλεγμένων beams | 2.27 / 16 |
| Μείωση beam-search overhead | 85.80% |
| Predictive ADB angular coverage | 95.23% |

Οι beam labels είναι **geometry-derived** από τη σχετική γωνία actor–ego. Δεν είναι πραγματικές optical beam-power measurements. Η εξωτερική επικύρωση με DeepSense ή ray tracing αποτελεί επόμενο βήμα.

---

## Γιατί χρειαζόμαστε risk-aware control

Οι μέσες τιμές αποκρύπτουν σπάνιες αλλά κρίσιμες αποτυχίες. Στα 100 scenarios παρατηρήθηκαν:

- 35 scenarios με τουλάχιστον μία Top-1 beam αποτυχία,
- 11 scenarios με adaptive Top-K coverage μικρότερο από 100%,
- 12 scenarios με predictive ADB coverage μικρότερο από 100%,
- λίγες ακραίες σκηνές με πολύ μεγάλο angular error, παρότι η median επίδοση είναι πολύ καλή.

Η νέα risk-aware μέθοδος χρησιμοποιεί μόνο πληροφορία διαθέσιμη κατά το inference:

\[
R = w_\sigma r_\sigma + w_a r_a + w_\omega r_\omega + w_d r_d + w_f r_f,
\]

όπου οι κανονικοποιημένοι όροι αντιστοιχούν σε:

- προβλεπόμενη angular uncertainty,
- πρόσφατη επιτάχυνση,
- πρόσφατο turn rate,
- ευαισθησία λόγω μικρής απόστασης,
- εγγύτητα στα όρια του beam field of view.

Η εκτιμώμενη επικινδυνότητα αλλάζει δυναμικά το coverage target και το ADB confidence margin:

| Risk score | Beam probability target | ADB confidence scale |
|---:|---:|---:|
| `< 0.25` | 90.0% | 1.64σ |
| `0.25–0.50` | 95.0% | 1.96σ |
| `0.50–0.75` | 98.0% | 2.33σ |
| `≥ 0.75` | 99.5% | 2.58σ |

Έτσι το σύστημα ξοδεύει περισσότερα beam probes μόνο όταν η εκτιμώμενη δυσκολία της σκηνής είναι υψηλή.

---

## Δομή repository

```text
.
├── configs/                       Ρυθμίσεις πειραμάτων
├── data/                          Οδηγίες dataset· τα raw δεδομένα αγνοούνται
├── scripts/
│   ├── inspect_scenario.py        Οπτικοποίηση ενός AV2 scenario
│   ├── run_baseline_demo.py       End-to-end demo ενός scenario
│   ├── run_batch_baselines.py     Batch baseline evaluation
│   ├── analyze_batch_results.py   Bootstrap και worst-case analysis
│   └── run_risk_aware_batch.py    Fixed έναντι risk-aware control
├── src/iscai/
│   ├── adb.py                     Predictive angular shadow zones
│   ├── beam.py                    Beam codebook και adaptive Top-K
│   ├── data.py                    AV2 loader και actor tracks
│   ├── evaluation.py              Trajectory/angular metrics
│   ├── geometry.py                Ego-centric μετασχηματισμοί
│   ├── prediction.py              CV και Kalman predictors
│   └── risk.py                    Inference-time risk estimator
├── tests/                         Unit tests
└── outputs/                       Figures, CSV και JSON αποτελέσματα
```

---

## Εγκατάσταση

Προτείνεται Python 3.10 ή 3.11.

```bash
git clone https://github.com/panagiotagrosdouli/argoverse-iscaI-predictive-beam-adb.git
cd argoverse-iscaI-predictive-beam-adb

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Σε WSL/Linux, δήλωσε το υπάρχον Argoverse 2 dataset:

```bash
export AV2_ROOT=/home/<user>/Datasets/Argoverse2
```

Αναμενόμενη δομή:

```text
$AV2_ROOT/
├── train/<scenario_id>/scenario_<scenario_id>.parquet
├── val/<scenario_id>/scenario_<scenario_id>.parquet
└── test/<scenario_id>/scenario_<scenario_id>.parquet
```

Τα δεδομένα του Argoverse δεν πρέπει να γίνουν commit στο GitHub.

---

## Γρήγορη εκτέλεση

### Οπτικοποίηση ενός scenario

```bash
SCENARIO=$(find "$AV2_ROOT/train" -name "scenario_*.parquet" | head -n 1)

python scripts/inspect_scenario.py \
  --scenario "$SCENARIO" \
  --output outputs/figures/scenario.png
```

### End-to-end demo

```bash
python scripts/run_baseline_demo.py \
  --scenario "$SCENARIO" \
  --output-dir outputs/demo
```

Παράγονται:

```text
outputs/demo/trajectory_prediction.png
outputs/demo/metrics.json
```

### Batch πείραμα 100 scenarios

```bash
python scripts/run_batch_baselines.py \
  --dataset-root "$AV2_ROOT" \
  --split train \
  --max-scenarios 100 \
  --output-dir outputs/batch_100
```

### Στατιστική ανάλυση

```bash
python scripts/analyze_batch_results.py \
  --csv outputs/batch_100/per_scenario_metrics.csv \
  --output-dir outputs/batch_100/analysis
```

### Fixed έναντι risk-aware control

```bash
python scripts/run_risk_aware_batch.py \
  --dataset-root "$AV2_ROOT" \
  --split train \
  --max-scenarios 100 \
  --output-dir outputs/risk_aware_100
```

Το τελικό JSON αναφέρει coverage gains, πρόσθετα beam probes και τη μεταβολή του overhead.

---

## Επιστημονικές μετρικές

### Trajectory prediction

- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Mean angular error

### Beam management

- Top-1 beam accuracy
- Adaptive Top-K coverage
- Average selected beams
- Overhead reduction έναντι exhaustive search
- Coverage–overhead trade-off

### Predictive ADB

- Angular shadow coverage
- Shadow-zone violation rate
- Angular interval width
- Over-masking

### Στατιστική αξιολόγηση

- Paired predictor comparisons
- Bootstrap 95% confidence intervals
- Worst-case ranking
- Failure counts και tail analysis

---

## Περιορισμοί

1. Τα beam labels παράγονται γεωμετρικά από το actor azimuth.
2. Δεν χρησιμοποιούνται ακόμη πραγματικά optical beam-power vectors.
3. Οι τροχιές προέρχονται από dataset annotations και όχι από detections του PC-FMCW sensing subsystem.
4. Το ADB evaluation είναι angular και όχι πλήρης φωτομετρική προσομοίωση SAE J3069.
5. Η risk policy είναι ερμηνεύσιμη και hand-calibrated· η learned ή conformal calibration αποτελεί μελλοντική εργασία.

Οι περιορισμοί αναφέρονται ρητά ώστε να διαχωρίζονται τα αποδεδειγμένα αποτελέσματα από τις μελλοντικές επεκτάσεις.

---

## Ερευνητικό roadmap

- [x] Real-world AV2 trajectory pipeline
- [x] Constant Velocity και Kalman baselines
- [x] Uncertainty propagation σε beam probabilities
- [x] Predictive ADB angular control
- [x] Batch evaluation και bootstrap analysis
- [x] Worst-case scenario analysis
- [x] Inference-time risk estimator
- [x] Risk-adaptive Top-K και ADB confidence
- [ ] Calibration της risk policy στο validation split
- [ ] GRU ή Transformer trajectory predictor
- [ ] Class-aware margins για pedestrians/cyclists
- [ ] DeepSense ή ray-traced beam validation
- [ ] Πλήρης optical/photometric ADB αξιολόγηση

---

## Κατάσταση της έρευνας

Το project είναι ενεργό academic research prototype. Οι αριθμοί πρέπει να ερμηνεύονται με βάση τις παραπάνω παραδοχές. Βασικοί στόχοι του repository είναι η αναπαραγωγιμότητα, η διαφανής αξιολόγηση και ο σαφής διαχωρισμός μεταξύ measured και geometry-derived μεγεθών.

---

## Citation

Το repository περιλαμβάνει `CITATION.cff` για citation του λογισμικού. BibTeX citation για τη συνοδευτική εργασία θα προστεθεί όταν ολοκληρωθεί το manuscript.

---

## Άδεια και δεδομένα

Ο κώδικας διατίθεται με MIT License. Τα δεδομένα Argoverse 2 δεν αναδιανέμονται και παραμένουν υπό τους αρχικούς όρους χρήσης του dataset.