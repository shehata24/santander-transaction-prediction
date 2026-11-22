# Santander Customer Transaction Prediction

Binary classification on Kaggle's
[Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)
competition: predict which customers will make a specific transaction in the
future, regardless of the amount of money involved.

> **Status:** 🚧 Work in progress. The sections below describe the problem, the data
> and the targets I'm working toward. Results will be filled in as the project moves along.

---

## 1. The problem

Santander's data science team wants to know **whether a customer will make a
particular transaction** (yes/no). The data has the same structure as the real data
they use for this, but every feature is anonymized, so nothing about the columns
says what they mean.

- **Task type:** supervised binary classification (tabular data)
- **Target:** `target` ∈ {0, 1}, where 1 means the customer made the transaction
- **Output:** for each customer in the test set, the **probability** that `target = 1`
- **Competition dates:** February to April 2019 (about 8,800 teams, $65,000 prize pool)

## 2. The dataset

| File | Rows | Columns | Contents |
|---|---|---|---|
| `train.csv` | 200,000 | 202 | `ID_code`, `target`, `var_0` … `var_199` |
| `test.csv` | 200,000 | 201 | `ID_code`, `var_0` … `var_199` (no target) |
| `sample_submission.csv` | 200,000 | 2 | `ID_code`, `target` (submission template) |

### What the columns are

- **`ID_code`**: a row identifier such as `train_0` or `test_0`. It is not a feature.
- **`var_0` … `var_199`**: 200 anonymized, continuous numeric features. Their names
  and meanings are hidden, so domain knowledge can't help. Anything useful has to
  come from the data itself.
- **`target`**: the binary label (training set only).

### Properties worth knowing before modelling

| Property | What it means for the project |
|---|---|
| **Class imbalance**: only about **10%** of the training rows are positive (~20k of 200k) | Accuracy is useless here (always predicting 0 scores ~90%). Use stratified CV and a ranking metric. |
| **No missing values** | No imputation needed. |
| **The features are almost uncorrelated** with each other | Each feature carries its own signal. Feature interactions matter less than in most tabular data, which is why even Naive Bayes is a strong baseline. |
| **Train and test distributions match closely** for each feature | Plain stratified K-fold CV should track the leaderboard well. |
| **About half of `test.csv` is synthetic**: public kernels showed that ~100k test rows contain no value that is unique in its column, so they were likely generated from real rows | Only real rows are scored. This matters if you build features from the combined train and test data (see §4). |
| **How often a value repeats inside its column is predictive**, not just the value itself | The "magic" behind the top solutions was count/frequency features per variable (see §4). |

## 3. Evaluation

Submissions are scored with **ROC AUC** (area under the ROC curve) between the
predicted probability and the true target.

- AUC only depends on **ranking**: it is the probability that a randomly chosen
  positive gets a higher score than a randomly chosen negative. You don't need
  calibrated probabilities, just a good ordering.
- 0.5 = random guessing, 1.0 = perfect ranking.
- AUC doesn't depend on the class threshold, which suits the 10% positive rate.

**Submission format:**

```csv
ID_code,target
test_0,0.0731
test_1,0.2210
...
```

## 4. Expected outcomes

### Score targets

These are the rough AUC levels the community reached with each approach. They are
the milestones this project will try to reproduce and explain.

| Milestone | Approach | Approx. AUC |
|---|---|---|
| 0 | Random / constant prediction | 0.500 |
| 1 | Logistic regression on standardized raw features | ~0.86 |
| 2 | Gaussian Naive Bayes | ~0.89 |
| 3 | Tuned gradient boosting (LightGBM/XGBoost) on raw features | ~0.90 |
| 4 | Add per-feature **value-count / frequency** features, computed on train plus the *real* test rows | ~0.91–0.92 |
| 5 | Treat each of the 200 features as independent (per-feature models or shared-weight NN), then ensemble | ~0.92+ |
| — | 1st place on the private leaderboard (reference) | ~0.926 |

The jump from milestone 3 to milestone 5 is the interesting part of this
competition. The best models didn't come from more tuning. They came from
understanding **how the data was generated**.

### Deliverables

1. **EDA notebook**: class balance, per-feature distributions (train vs test),
   correlation matrix, unique-value counts, identifying the synthetic test rows.
2. **Baselines**: logistic regression and Naive Bayes, scored with stratified 5-fold CV AUC.
3. **Gradient boosting model**: LightGBM with a reproducible CV setup and
   out-of-fold predictions.
4. **Feature engineering**: frequency/count features and a measured, cross-validated
   effect on AUC.
5. **Final model + Kaggle submission**, with public and private leaderboard scores
   recorded here.
6. **Write-up**: what worked, what didn't, and why. Includes ROC curves and feature
   importance plots in `reports/figures/`.

### Skills this project shows

- Working with a heavily imbalanced binary target
- Choosing and understanding a ranking metric (ROC AUC)
- A cross-validation setup that matches the leaderboard
- Gradient-boosted trees on tabular data
- Feature engineering based on how the data was generated, plus leakage awareness
  (synthetic test rows)

## 5. Results

| Model | CV AUC | Public LB | Private LB |
|---|---|---|---|
| _to be filled in_ | | | |

## 6. Project structure

```
santander-transaction-prediction/
├── data/              # raw Kaggle CSVs (git-ignored, see below)
├── notebooks/         # EDA and experiments
├── src/               # reusable code (features, training, CV)
├── reports/figures/   # plots for the write-up
├── requirements.txt
└── README.md
```

## 7. Getting the data

The data can't be redistributed, so it isn't in this repo. To download it:

1. Accept the competition rules on the
   [Kaggle competition page](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/rules).
2. Set up the Kaggle API (`pip install kaggle`, and put an API token in `~/.kaggle/`).
3. Run:

```bash
kaggle competitions download -c santander-customer-transaction-prediction -p data/
unzip data/santander-customer-transaction-prediction.zip -d data/
```

## 8. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Acknowledgements

The dataset was provided by Banco Santander through Kaggle. It is used here under
the competition's rules for non-commercial, educational purposes.
