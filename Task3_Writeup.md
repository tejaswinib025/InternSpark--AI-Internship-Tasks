# Task 3 Write-up: Explainability, Bias Checks and Mitigation

## Model audited
RandomForestClassifier churn classifier from Task 1 (test ROC-AUC 0.842).

## Explainability
- **Permutation importance** (drop in ROC-AUC): top features = Contract, tenure, InternetService, TotalCharges, OnlineSecurity.
- **SHAP** (mean |SHAP|, global): top drivers = Contract, tenure, OnlineSecurity, InternetService, TechSupport, TotalCharges, PaymentMethod.
- These make business sense: contract type and tenure dominate (month-to-month and short-tenure customers churn most, as the EDA in Task 1 also showed), followed by service-related features such as internet service type, online security and tech support. No sensitive attribute is among the top drivers.
- **Local explanation**: for test customer #0 SHAP and LIME were compared; their top-10 feature lists overlap on 8/10 features. LIME is a local linear surrogate, SHAP is based on Shapley values, so small differences in ranking are expected; agreement on the main drivers increases confidence.

## Bias checks (sensitive attributes: gender, SeniorCitizen)
Largest gap before mitigation: **DPD (SeniorCitizen)** = 0.281. Full comparison (DPD = demographic parity difference, EOD = equal-opportunity difference / TPR gap):

| Metric | Before | After reweighing |
|---|---|---|
| DPD (gender) | 0.0110 | 0.0150 |
| EOD (gender) | 0.0457 | 0.0723 |
| DPD (SeniorCitizen) | 0.2810 | 0.2698 |
| EOD (SeniorCitizen) | 0.1269 | 0.1029 |
| Accuracy | 0.7488 | 0.7424 |
| F1 | 0.6266 | 0.6199 |

Gaps that shrank after reweighing: DPD (SeniorCitizen), EOD (SeniorCitizen). Gaps that grew slightly: DPD (gender), EOD (gender) (small in absolute terms and within the noise expected for a 1,409-row test set, but they show reweighing is not a guaranteed fix and must be re-audited). Note that part of any SeniorCitizen gap reflects a genuine difference in churn base rates
(seniors churn more in the data), so a selection-rate gap is not automatically unfair; TPR/EOD shows whether the model finds real churners equally well in each group.
Accuracy changed from 0.749 to 0.742 and F1 from 0.627 to 0.620, i.e. the accuracy/fairness trade-off is small.

## Mitigation recommendations
1. **Reweighing** (done above) or other pre-processing; re-audit after every retrain.
2. **Per-group decision thresholds** to equalise TPR, only where legally and ethically acceptable.
3. **Check proxy features** (e.g. partner/dependents, tenure, plan type) that may encode age or gender even if the sensitive column is removed.
4. **More representative data** and larger samples for small groups (senior citizens are ~16% of customers), so metrics have narrow confidence intervals.
5. **Human review** for high-impact actions; churn offers are low-stakes, but the same model must not be reused for credit or service denial.
6. **Governance**: publish a model card (intended use, metrics per group, limitations), monitor drift and group metrics in production, and allow customers to contest decisions.
