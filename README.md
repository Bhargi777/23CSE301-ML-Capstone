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

## Review 2 — Member 1 (Bhargava)

Scope: two Classification Part B algorithms (Random Forest, AdaBoost) and clustering deliverables B2 (metrics,
Elbow curve, Dendrogram) and B3 (PCA and t-SNE plots). Notebooks live in `review2/notebooks/`; run them from that
folder top to bottom.

### Classification Part B (Adult) — `member1_classification_partB.ipynb`

- `review2/notebooks/adult_prep.py` reproduces the Review 1 pipeline, so Part B uses the same 36,140 / 9,035
  stratified split and train-only encoder/scaler as Part A.
- Both models tuned with stratified 5-fold `GridSearchCV` on weighted F1; 5-fold CV reported for both.

| Model | Best params | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1` | 0.833 | 0.836 | 0.897 |
| AdaBoost | `n_estimators=500`, `learning_rate=1.0` | 0.814 | 0.823 | 0.910 |

- Tuning raised RF accuracy by 0.011 and weighted F1 by 0.005, but lowered ROC-AUC by 0.020. AdaBoost improved on
  every metric (weighted F1 +0.005, ROC-AUC +0.007).
- Outputs: `review2/results/classification/partB_member1_*.csv|png`, `rf_*.png`, `adaboost_*.png`;
  models in `review2/models/classification/random_forest.pkl`, `adaboost.pkl`.

### Clustering B2 + B3 (Gas Sensor) — `member1_clustering_evaluation.ipynb`

- Features: 128 sensor features, signed `log1p` and `StandardScaler`; gas label and concentration removed before
  fitting and used only to interpret the clusters afterwards.
- k = 4 from the K-Means Elbow curve; the Ward dendrogram also has a natural cut at 4 clusters.

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI vs gas |
|---|---|---|---|---|
| K-Means (k=4) | 0.267 | 1.253 | 6,674 | 0.196 |
| Agglomerative, Ward (k=4) | 0.202 | 1.378 | 5,390 | 0.121 |

- Linkage comparison: single and average linkage put 13,907 of 13,910 points in one cluster, so Ward is used.
- PCA (first 2 components = 77% of variance) and t-SNE plots for both algorithms, next to the true gas.
- Outputs: `review2/results/clustering/` (metrics CSVs, elbow, dendrogram, linkage, PCA, t-SNE, cluster-vs-gas plots).

### Environment and running

```bash
pip install -r review2/requirements.txt
cd review2/notebooks
jupyter nbconvert --to notebook --execute --inplace member1_classification_partB.ipynb   # ~4 min
jupyter nbconvert --to notebook --execute --inplace member1_clustering_evaluation.ipynb  # ~2 min
```

AI assistance: generative AI tools were used for code scaffolding in these notebooks, as allowed by guideline 7.5.

---

## Review 2 — Team results

### Work split

| Member | Classification Part B | Clustering |
|---|---|---|
| 1 | Random Forest, AdaBoost (`member1_classification_partB.ipynb`) | B2 metrics, Elbow, Dendrogram, linkage comparison; B3 PCA + t-SNE (`member1_clustering_evaluation.ipynb`) |
| 2 | Bagging, MLP (`classification_member2.ipynb`, incl. step-by-step Adult preprocessing) | B1 K-Means (`clustering_member2.ipynb`) |
| 3 | Gradient Boosting (`member3_gradient_boosting.ipynb`); A2 10-model table + A3 final model (`member3_classification_A2_A3_corrected.ipynb`) | B1 Agglomerative (`member3_agglomerative_clustering.ipynb`) |

All classifiers use the same Adult split from `review2/notebooks/adult_prep.py` (36,140 train / 9,035 test,
stratified, encoder and scaler fitted on train only) and balanced class weights.

### Classification — all ten algorithms (test set, sorted by weighted F1)

| Rank | Model | Part | Accuracy | Precision (w) | Recall (w) | F1 (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | Bagging | B | 0.846 | 0.855 | 0.846 | 0.849 | 0.914 |
| 2 | **Gradient Boosting** | B | 0.837 | 0.867 | 0.837 | 0.845 | **0.926** |
| 3 | KNN | A | 0.839 | 0.835 | 0.839 | 0.836 | 0.897 |
| 4 | Random Forest | B | 0.833 | 0.841 | 0.833 | 0.836 | 0.897 |
| 5 | AdaBoost | B | 0.814 | 0.851 | 0.814 | 0.823 | 0.910 |
| 6 | Decision Tree | A | 0.809 | 0.849 | 0.809 | 0.819 | 0.894 |
| 7 | SVC | A | 0.807 | 0.849 | 0.807 | 0.817 | 0.895 |
| 8 | MLP | B | 0.803 | 0.853 | 0.803 | 0.814 | 0.908 |
| 9 | Logistic Regression | A | 0.800 | 0.842 | 0.800 | 0.810 | 0.894 |
| 10 | Gaussian Naive Bayes | A | 0.586 | 0.811 | 0.586 | 0.605 | 0.834 |

**Final model: Gradient Boosting** (`learning_rate=0.05, max_depth=5, n_estimators=300`). Best ROC-AUC (0.926, CV
0.927 ± 0.004), balanced accuracy (0.844) and macro F1 (0.804) of all ten; catches 86% of `>50K` earners. Bagging has
slightly higher accuracy / weighted F1 but finds ~215 fewer high earners. Full justification in the A3 notebook.

### Clustering — Gas Sensor (k = 4)

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI vs gas |
|---|---|---|---|---|
| K-Means | 0.267 | 1.253 | 6,674 | 0.196 |
| Agglomerative (Ward) | 0.202 | 1.378 | 5,390 | 0.121 |

### Run order (from `review2/notebooks/`)

1. Classification: `member1_classification_partB.ipynb`, `classification_member2.ipynb`,
   `member3_gradient_boosting.ipynb`, then `member3_classification_A2_A3_corrected.ipynb` (reads the others' result CSVs).
2. Clustering: `clustering_member2.ipynb`, `member3_agglomerative_clustering.ipynb`, then
   `member1_clustering_evaluation.ipynb` (loads both label files).

---

## Streamlit app (bonus D8)

`review2/app/app.py` is a web GUI over the trained models:

- **Income prediction:** enter a person's census details; the inputs go through the same `adult_prep` pipeline as the
  notebooks (encoder and scaler fitted on the training split only) and the chosen Part B model returns the predicted
  class and P(>50K). Default model: Gradient Boosting (the A3 final model); all five Part B models can be compared.
- **Gas sensor cluster:** pick a measurement from the dataset or upload a CSV with `feature_1 ... feature_128`; it is
  signed-log transformed, scaled with the saved clustering scaler and assigned to a K-Means cluster, shown on the
  PCA map with the cluster's gas make-up. (Agglomerative clustering cannot assign new points, so K-Means is used.)
- **Results:** the 10-classifier table and the clustering metrics.

Run locally (from the repository root, after `git lfs pull` so the `.pkl` models are real files):

```bash
pip install -r review2/app/requirements.txt
streamlit run review2/app/app.py
```

`review2/app/requirements.txt` pins `scikit-learn==1.9.0`, the version the models were saved with. To deploy on
Streamlit Community Cloud, point a new app at this repository with main file `review2/app/app.py`.

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
