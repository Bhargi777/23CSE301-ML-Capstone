# 23CSE301 — Machine Learning Capstone Project

> **B.Tech Computer Science and Engineering — III Year**  
> **Academic Year: 2026–27**

---

## Project Overview

This repository contains the implementation of the **23CSE301 Machine Learning Capstone Project**.

The project covers three major machine learning tracks:

- **Regression**
- **Classification**
- **Clustering**

Multiple machine learning algorithms are implemented, evaluated, compared, and interpreted using appropriate performance metrics and visualization techniques.

The project is developed collaboratively using **Git and GitHub**, with each team member working on a separate branch.

---

# Team Members

| Member | Name | Roll Number | Part | Branch |
|---|---|---|---|---|
| 1 | Abhigna | CB.SC.U4CSE24227 | Regression EDA, cleaning, preprocessing | `Abhigna-Regression` |
| 2 | Bhargava | CB.SC.U4CSE24268 | Regression models, tuning, CV, diagnostics | `Bhargava-Regression` |
| 3 | Vikranth | CB.SC.U4CSE24244 | Classification preprocessing, models, evaluation | `Vikranth-Classification` |

### Branching Strategy

Each member works independently on their assigned branch.

```text
main
│
├── Member 1
│   └── Abhigna-Regression
│
├── Member 2
│   └── Bhargava-Regression
│
└── Member 3
    └── Vikranth-Classification
```

---

## Datasets

1. **Regression — Air Quality**
   Predict pollutant concentration from sensor and environmental measurements.
   9,358 instances | 15 features | hourly air-quality sensor readings
   https://archive.ics.uci.edu/dataset/360/air+quality

2. **Classification — Adult / Census Income**
   Predict whether annual income exceeds $50K. Binary classification.
   48,842 instances | 14 features | numerical + categorical
   https://archive.ics.uci.edu/dataset/2/adult

3. **Clustering — Gas Sensor Array Drift at Different Concentrations**
   Identify natural clusters in gas-sensor measurements and check whether clusters
   correspond to different gases/concentration patterns.
   13,910 measurements | 16 chemical sensor features
   https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+at+different+concentrations

---

## Data Preparation and Train/Test Data

### Regression (Air Quality, target `CO(GT)`)

- Source: `data/raw/airquality/AirQualityUCI.csv` (`;` separator, `,` decimal). Train: `data/processed/regression_train.csv`
  (6,139 rows). Test: `data/processed/regression_test.csv` (1,535 rows). Stratified 80/20 split of 7,674 rows.
- Cleaning: 114 empty trailing rows and 2 empty columns dropped; `-200` sentinel converted to NaN; 1,683 rows with a
  missing `CO(GT)` dropped; `NMHC(GT)` (~90% missing) dropped; duplicates checked (none).
- Features: `hour`, `day`, `month` parsed from Date/Time, plus `T_RH_Interaction`.
- Median imputation, IQR outlier capping and standardization are fit on the training split only (no leakage).
- Models: all 10 regressors are compared on R2, RMSE and MAE; the top 2 are tuned with `GridSearchCV` and checked with
  5-fold CV (best after tuning: Gradient Boosting, test R2 0.947). The remaining algorithms are tuned in an extra
  section, with linear coefficients, polynomial degrees, tree feature importance and the effect of scaling on KNN.
- Limitations: random split of hourly time-series data and reference-analyser features (`C6H6(GT)`, `NOx(GT)`,
  `NO2(GT)`) make scores optimistic; see the notebook for details.

### Classification (Adult, target `income`)

- Data: the two UCI files (`adult_train.csv`, `adult_test.csv`) are pooled (48,842 rows), cleaned, then split once with a
  stratified 80/20 split (`random_state=42`): 36,140 train / 9,035 test rows, same class ratio in both.
- Cleaning: 3,620 rows with `?` dropped, 47 duplicates dropped (45,175 rows left), whitespace stripped, trailing period
  on the test labels removed.
- Outliers and features: `log1p` on the skewed `capital-gain` / `capital-loss`; engineered `net_capital` and `overtime`;
  `fnlwgt` (sampling weight) and the redundant `education` dropped.
- Encoding: one-hot for nominal columns and `StandardScaler` for numeric ones, both fit on train only.
- Imbalance: ~75% `<=50K` / 25% `>50K`. `class_weight='balanced'` is used where supported, and the majority-class baseline
  (75% accuracy, weighted F1 0.646) is reported next to accuracy, precision, recall, weighted/macro F1, balanced accuracy
  and ROC-AUC.
- 5 classifiers (Part A): Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC; KNN, Decision Tree and SVC
  are tuned, the top 2 are cross-validated (best: KNN, weighted F1 0.836, ROC-AUC 0.897).

---

## Repository Structure

```text
data/
  raw/            # original datasets (airquality, adult, gassensor)
  processed/      # cleaned/engineered train-test splits
notebooks/        # member1/2/3 notebooks, run in that order
models/
  regression/     # fitted regression models + diagnostic plots
  classification/ # fitted classification models + confusion matrices
results/
  regression/     # metrics tables, comparison plots
  classification/ # metrics tables, comparison plots
  clustering/     # clustering evaluation (Review 2)
```

---

## Cloning this repo

Model artifacts are versioned with **Git LFS**. Run `git lfs install` before cloning,
or the files under `models/` will come down as text pointers instead of the actual
binaries.
