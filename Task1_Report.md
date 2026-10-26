# Task 1 Report: Customer Churn Classification

## Problem and data
Binary classification: predict whether a telecom customer churns. Telco Customer Churn dataset (IBM): 7043 customers, 19 features
(demographics, services, contract, billing). Churn rate is 26.5% (imbalanced). EDA shows churn is concentrated among short-tenure
and month-to-month-contract customers.

## Method
- Preprocessing: `TotalCharges` blanks converted to NaN and median-imputed; numeric features standardised; categoricals one-hot encoded (all inside a Pipeline, so no leakage).
- Split: 80/20 stratified train/test (5634 / 1409), random_state=42.
- Models compared: Logistic Regression and Random Forest, both with `class_weight='balanced'`.
- Tuning and validation: GridSearchCV with stratified 5-fold CV (scoring ROC-AUC). Best parameters: {'Logistic Regression': {'clf__C': 10}, 'Random Forest': {'clf__max_depth': 6, 'clf__min_samples_leaf': 5, 'clf__n_estimators': 200}}.

## Results
Cross-validation (training set):

| Model | CV Accuracy | CV F1 | CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7481 | 0.6275 | 0.8462 |
| Random Forest | 0.7503 | 0.6277 | 0.8466 |

Held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7395 | 0.5060 | 0.7834 | 0.6149 | 0.8405 |
| Random Forest | 0.7488 | 0.5174 | 0.7941 | 0.6266 | 0.8419 |

## Model selection
**Random Forest** was selected (highest test ROC-AUC: 0.8419 vs 0.8405 for Logistic Regression).
The two models are very close, so the simpler and more interpretable Logistic Regression is a reasonable alternative if explainability matters most.
Because `class_weight='balanced'` was used, recall on churners is high (0.79) at the cost of precision (0.52):
the model flags most real churners but also some loyal customers. For a retention campaign (cheap offers) this trade-off is usually acceptable;
the decision threshold can be tuned to the real cost of a false alarm versus a lost customer.

## Limitations
Single dataset and single split; no probability calibration; no cost-sensitive threshold tuning. The saved model is audited for bias and explained in Task 3.
