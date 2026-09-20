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
  missing `CO(GT)` dropped; `NMHC(GT)` (~90% missing) dropped; duplicates removed.
- Features: `hour`, `day`, `month` parsed from Date/Time, plus `T_RH_Interaction`.
- Median imputation, IQR outlier capping and standardization are fit on the training split only (no leakage).
- Limitations: random split of hourly time-series data and reference-analyser features (`C6H6(GT)`, `NOx(GT)`,
  `NO2(GT)`) make scores optimistic; see the notebook for details.

### Classification (Adult, target `income`)

- Train: `data/raw/adult/adult_train.csv` (32,561 raw rows, 30,139 after cleaning). Test: `adult_test.csv`
  (16,281 raw rows, 15,060 after cleaning). This is the official UCI split.
- Cleaning: `?` rows dropped, whitespace stripped, trailing period on the test labels removed, duplicate train rows
  removed, `fnlwgt` and redundant `education` dropped.
- Encoding: one-hot for nominal columns (fit on train only, unseen categories ignored); numeric columns standardized.
- Imbalance: ~75% `<=50K` / 25% `>50K`. `class_weight='balanced'` is used where supported, and macro F1, balanced
  accuracy and the majority-class baseline (~75% accuracy) are reported next to accuracy and weighted F1.
- 10 classifiers: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC, Random Forest, Extra Trees,
  Gradient Boosting, AdaBoost, MLP.

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
