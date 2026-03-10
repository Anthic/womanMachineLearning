# BDHS 2022 Mental Health ML Study — Expert Q1 Research Assessment

### Senior ML Expert & Academic Peer-Review Perspective

---

## STUDY OVERVIEW

| Item                | Detail                                             |
| ------------------- | -------------------------------------------------- |
| Dataset             | Bangladesh Demographic & Health Survey (BDHS) 2022 |
| Total Sample        | 30,078 women respondents                           |
| Dependent Variables | Depression (dep), Anxiety (anx), Disability (disu) |
| Algorithm Pipeline  | 6 ML classifiers × 3 outcomes                      |
| Feature Selection   | Union of MI + Chi² + RF Importance (Top-15 each)   |
| Imbalance Handling  | SMOTE applied AFTER feature selection              |
| Leakage Prevention  | MTH22, MTH24 excluded; SMOTE after split           |
| Cross-Validation    | 5-Fold CV with F1 stability tracking               |

---

## SECTION 1: FULL MODEL PERFORMANCE COMPARISON (ALL 3 OUTCOMES)

### 1.1 Depression (dep) — Prevalence: 3.4% (1,026 / 30,078)

| Model         | Train Acc | Test Acc | Overfit Gap  | Precision | Recall | F1        | ROC-AUC   | CV F1 | CV Std |
| ------------- | --------- | -------- | ------------ | --------- | ------ | --------- | --------- | ----- | ------ |
| XGBoost       | 98.3%     | 96.5%    | 1.7% ✅      | 30.0%     | 1.5%   | 2.79%     | **75.6%** | 97.9% | ±3.9   |
| Random Forest | 100%      | 96.5%    | 3.5% ✅      | 0%        | 0%     | **0%** ❌ | 73.1%     | 98.1% | ±3.6   |
| Decision Tree | 90.9%     | 93.9%    | -3.0% ✅     | 9.4%      | 9.3%   | 9.34%     | 72.5%     | 90.2% | ±3.2   |
| Logistic Reg. | 81.1%     | 78.6%    | 2.5% ✅      | 5.9%      | 35.1%  | 10.05%    | 68.5%     | 81.1% | ±2.0   |
| SVM (Linear)  | 93.8%     | 90.3%    | 3.5% ✅      | 5.6%      | 11.7%  | 7.58%     | 66.8%     | 92.6% | ±3.1   |
| KNN           | 100%      | 83.1%    | **16.9% ⚠️** | 6.2%      | 27.8%  | 10.09%    | 63.9%     | 91.7% | ±0.7   |

### 1.2 Anxiety (anx) — Prevalence: 3.0% (896 / 30,078)

| Model         | Train Acc | Test Acc | Overfit Gap  | Precision | Recall | F1        | ROC-AUC   | CV F1 | CV Std |
| ------------- | --------- | -------- | ------------ | --------- | ------ | --------- | --------- | ----- | ------ |
| XGBoost       | 98.5%     | 97.0%    | 1.5% ✅      | 0%        | 0%     | **0%** ❌ | **75.1%** | 98.2% | ±3.3   |
| Random Forest | 100%      | 97.0%    | 3.0% ✅      | 25.0%     | 0.6%   | 1.09%     | 74.1%     | 98.3% | ±3.2   |
| Decision Tree | 93.3%     | 95.5%    | -2.2% ✅     | 4.9%      | 2.8%   | 3.55%     | 69.9%     | 92.8% | ±2.8   |
| Logistic Reg. | 82.2%     | 79.0%    | 3.2% ✅      | 4.6%      | 30.7%  | 8.02%     | 67.2%     | 82.5% | ±1.8   |
| SVM (Linear)  | 82.2%     | 79.4%    | 2.8% ✅      | 4.7%      | 30.7%  | 8.16%     | 66.7%     | 82.5% | ±1.8   |
| KNN           | 100%      | 85.0%    | **15.0% ⚠️** | 6.1%      | 27.9%  | 9.96%     | 62.8%     | 92.7% | ±0.5   |

### 1.3 Disability (disu) — Prevalence: 4.7% (1,414 / 30,078)

| Model         | Train Acc | Test Acc | Overfit Gap  | Precision | Recall | F1           | ROC-AUC   | CV F1 | CV Std |
| ------------- | --------- | -------- | ------------ | --------- | ------ | ------------ | --------- | ----- | ------ |
| Random Forest | 100%      | 95.2%    | 4.8% ✅      | 20.0%     | 0.7%   | **1.37%** ❌ | **75.2%** | 97.2% | ±5.3   |
| XGBoost       | 97.6%     | 95.2%    | 2.5% ✅      | 16.7%     | 0.7%   | 1.36% ❌     | 75.1%     | 97.0% | ±5.6   |
| Decision Tree | 91.9%     | 90.9%    | 1.0% ✅      | 11.8%     | 14.5%  | 13.0%        | 71.9%     | 89.9% | ±5.9   |
| Logistic Reg. | 80.5%     | 77.6%    | 2.9% ✅      | 8.5%      | 38.5%  | 13.94%       | 68.7%     | 80.7% | ±3.0   |
| SVM (Linear)  | 80.6%     | 77.9%    | 2.7% ✅      | 8.6%      | 38.5%  | 14.09%       | 68.1%     | 80.6% | ±3.0   |
| KNN           | 100%      | 81.5%    | **18.5% ⚠️** | 8.2%      | 29.0%  | 12.82%       | 64.4%     | 90.9% | ±0.8   |

---

## SECTION 2: CROSS-OUTCOME BEST MODEL SUMMARY

| Outcome    | Best by AUC   | AUC   | F1 of "Best" | Clinically Useful?          |
| ---------- | ------------- | ----- | ------------ | --------------------------- |
| Depression | XGBoost       | 75.6% | 2.79%        | ⚠️ Barely                   |
| Anxiety    | XGBoost       | 75.1% | **0%**       | ❌ NO — Predicts zero cases |
| Disability | Random Forest | 75.2% | **1.37%**    | ❌ Nearly zero              |

### Cross-Outcome Pattern Observations:

1. **AUC is consistent (~75%) across all 3 outcomes** — suggests the predictor set has similar discriminatory power regardless of outcome, which is scientifically interesting (shared sociodemographic determinants).

2. **Tree-based models (RF, XGBoost) suffer complete or near-complete minority-class failure** despite SMOTE balancing. At the default 0.5 decision threshold, they output majority class for almost all predictions.

3. **Logistic Regression and SVM consistently detect more positive cases** (Recall 30-38%) with expected low precision — this is the realistic tradeoff in imbalanced clinical screening.

4. **KNN consistently overfits** (15-18% gap) across all three outcomes, making it unreliable.

5. **Decision Tree offers the best precision-recall balance** (F1: 9-13%) even though AUC is lower — this is academically significant for clinical practice.

---

## SECTION 3: SELECTED FEATURES — CROSS-OUTCOME COMPARISON

| Feature                        | Depression | Anxiety | Disability | Shared   |
| ------------------------------ | ---------- | ------- | ---------- | -------- |
| V005 (Sample weight)           | ✅         | ✅      | ✅         | All 3    |
| V012 (Age)                     | ✅         | ✅      | ✅         | All 3    |
| V021 (PSU/Cluster)             | ✅         | ✅      | ✅         | All 3    |
| V023 (Strata)                  | ✅         | ✅      | ✅         | All 3    |
| V025 (Urban/Rural)             | ✅         | ✅      | ✅         | All 3    |
| V106 (Education)               | ✅         | ✅      | ✅         | All 3    |
| V190 (Wealth index)            | ✅         | ✅      | ✅         | All 3    |
| V201 (Total children)          | ✅         | ✅      | ✅         | All 3    |
| V212 (Age 1st birth)           | ✅         | ✅      | ✅         | All 3    |
| V312 (Contraceptive method)    | ✅         | ✅      | ✅         | All 3    |
| V511 (Age at 1st cohabitation) | ✅         | ✅      | ✅         | All 3    |
| V632 (Decision: contraception) | ✅         | ✅      | ✅         | All 3    |
| V701 (Husband education)       | ✅         | ✅      | ✅         | All 3    |
| V730 (Husband age)             | ✅         | ✅      | ✅         | All 3    |
| age_dif (Husband-wife age gap) | ✅         | ✅      | ✅         | All 3    |
| age_dif_cat                    | ✅         | ✅      | ✅         | All 3    |
| age_fb_cat                     | ✅         | ✅      | ✅         | All 3    |
| age_cat                        | ✅         | ✅      | ✅         | All 3    |
| contra (Contraceptive use)     | ✅         | ✅      | ✅         | All 3    |
| contra_decision3               | ✅         | ✅      | ✅         | All 3    |
| V024 (Region)                  | ✅         | ❌      | ✅         | Dep+Disu |
| parity_cat                     | ❌         | ✅      | ❌         | Anx only |
| wealth_cat                     | ❌         | ✅      | ❌         | Anx only |

**Key Finding:** 20 out of 21-22 selected features are IDENTICAL across all three outcomes. This suggests a shared sociodemographic risk structure underlying depression, anxiety, and disability in Bangladeshi women — a publishable finding in its own right.

---

## SECTION 4: HONEST Q1 JOURNAL ASSESSMENT

### ⭐ VERDICT: **NOT READY FOR Q1 — Requires Major Revision**

**Target Q1 Journals for this study type:**

- Journal of Affective Disorders (IF ~6.5)
- BMC Psychiatry (IF ~3.4)
- PLOS ONE (IF ~3.7)
- International Journal of Environmental Research and Public Health (IF ~4.6)
- SSM - Population Health (IF ~4.1)
- Asian Journal of Psychiatry (IF ~4.6)

---

### 4.1 WHY IT WILL BE REJECTED AT Q1 (Critical Issues)

#### ❌ ISSUE 1: Best Model for Anxiety Has F1 = 0% and Recall = 0%

**This is a fatal flaw.**

- XGBoost is declared "Best Model" for anxiety with ROC-AUC=75.1%
- But F1=0% and Recall=0% mean it predicts ZERO anxious women correctly
- A model that identifies no cases is clinically worthless — this is the **Accuracy Paradox**
- Peer reviewers will reject this instantly
- **Root cause:** Default 0.5 threshold combined with severe class imbalance (3% prevalence)
- **Fix:** Threshold optimization (use Youden's J index or F1-maximizing threshold)

#### ❌ ISSUE 2: Random Forest Completely Fails for Depression (F1=0%, Recall=0%)

- Same problem: RF for depression identifies ZERO depressed women
- Yet RF is the second model listed — reviewers will notice this immediately
- This contradiction (high AUC but zero recall) must be explained and fixed

#### ❌ ISSUE 3: AUC of 75% Is "Acceptable" Not "Strong"

- Medical ML literature standard: AUC < 0.70 = poor, 0.70-0.79 = fair, 0.80-0.89 = good, ≥0.90 = excellent
- All three outcomes plateau at 75% AUC — this is "fair" not "strong"
- Q1 reviewers will ask: "Why is the discriminatory ability so modest?"
- **This needs explanation:** likely because underlying survey variables are distal determinants, not proximal clinical biomarkers

#### ❌ ISSUE 4: No Confidence Intervals on Any Metric

- Q1 journals require bootstrap 95% CIs on AUC, F1, Precision, Recall
- Without CIs, you cannot statistically compare models
- "XGBoost AUC=75.6% vs RF AUC=73.1%" — is this difference significant? Unknown.

#### ❌ ISSUE 5: No Hyperparameter Tuning Reported

- Default parameters used for all 6 models
- RandomizedSearchCV / GridSearchCV not mentioned
- Reviewers will ask: "Did you optimize anything? Or just use defaults?"
- This is publishable but weak for a Q1 ML paper

#### ❌ ISSUE 6: No SHAP Values / Explainability

- As of 2024-2026, SHAP (SHapley Additive exPlanations) is MANDATORY in top ML health papers
- Without SHAP, you cannot say which features drive predictions for individuals
- This is a major gap versus current literature standard

#### ❌ ISSUE 7: No Precision-Recall AUC (PRAUC) Reported

- For severely imbalanced datasets (3-5% prevalence), ROC-AUC is misleading
- PRAUC is the more honest metric
- Reviewers familiar with imbalanced ML will flag this immediately

#### ❌ ISSUE 8: KNN Overfitting (15-18% gap) Not Addressed

- KNN shows 100% training vs 81-85% test across all outcomes
- This is reported but not addressed — should KNN even be included as a candidate?
- Recommendation: Exclude KNN or address with distance-weighted KNN

#### ❌ ISSUE 9: SMOTE Did Not Solve the Problem

- Despite SMOTE balancing (minority class upsampled to equal majority class)
- The best models (RF, XGBoost) STILL refuse to predict positive cases at test time
- This means SMOTE was applied to training but models still learned majority bias
- Need to combine SMOTE with threshold tuning and/or class_weight parameter

#### ❌ ISSUE 10: No Model Calibration Assessment

- High AUC models may be poorly calibrated (overconfident or underconfident)
- Brier score and calibration plots are expected in health ML papers
- Calibration matters when models are used for risk scoring

---

### 4.2 WHY IT HAS Q1 POTENTIAL (Genuine Strengths)

#### ✅ STRENGTH 1: Dataset Is Gold Standard

- BDHS 2022 is a nationally representative probability sample
- n=30,078 is substantial — rare to have this scale for mental health ML
- DHS methodology is internationally validated (USAID-funded, WHO-aligned)
- Bangladesh context = underrepresented region in global mental health ML literature

#### ✅ STRENGTH 2: Simultaneous Analysis of 3 Mental Health Outcomes

- Depression + Anxiety + Disability analyzed with identical methodology
- Enables comorbidity discussion and shared determinant identification
- No comparable study exists for Bangladesh 2022 DHS — genuine novelty

#### ✅ STRENGTH 3: Data Leakage Prevention Is Excellent

- MTH22 and MTH24 exclusion (perfect separators) is textbook-correct
- SMOTE after feature selection is methodologically sound
- Proper stratified train-test split
- This is better methodology than many published health ML papers

#### ✅ STRENGTH 4: Overfitting Detection Reported

- Train vs test accuracy comparison with 5% gap threshold
- 5-fold CV with standard deviation (stability measure)
- This demonstrates methodological awareness

#### ✅ STRENGTH 5: Feature Convergence Is a Strong Finding

- 20/21-22 features identical across all 3 outcomes
- This convergence is NOT noise — it reflects shared SES-reproductive determinants
- This is a publishable finding: common risk architecture for mental health comorbidities

#### ✅ STRENGTH 6: Public Health Relevance (SDG 3)

- Mental health is SDG 3 priority
- Bangladesh has limited mental health infrastructure
- ML screening tools for population surveys = high policy impact
- This framing positions the paper well in global health journals

#### ✅ STRENGTH 7: Multi-Method Feature Selection (Ensemble Approach)

- Using 3 independent methods (MI, Chi², RF importance) and taking union
- More robust than single-method selection
- Reduces method dependency bias

---

## SECTION 5: WHAT NEEDS TO CHANGE FOR Q1 ACCEPTANCE

### Step-by-Step Upgrade Roadmap:

**PRIORITY 1 — MUST FIX (Deal-breakers):**

1. **Threshold Optimization:** For each model, find the optimal classification threshold using Youden's J index (maximizes Sensitivity + Specificity). This will fix the F1=0% problem for RF/XGBoost.
2. **Add PRAUC:** Report Precision-Recall AUC alongside ROC-AUC for all models.
3. **Bootstrap Confidence Intervals:** 1000-bootstrap resampling for AUC and F1.
4. **Redefine Best Model:** Best model = highest PRAUC OR best clinical tradeoff (F1 or sensitivity), NOT just ROC-AUC. A model with AUC=75% but Recall=0% is NOT the best model.

**PRIORITY 2 — SHOULD ADD (Major strengthening):** 5. **SHAP Analysis:** SHAP values for the best model per outcome — beeswarm + bar plots. 6. **Hyperparameter Tuning:** RandomizedSearchCV (50 iterations) for at least RF and XGBoost. 7. **Calibration Curves + Brier Score:** Add for top 2 models per outcome. 8. **Balanced Accuracy:** More honest primary metric than standard accuracy. 9. **Cost-Sensitive Learning:** Use class_weight='balanced' in addition to SMOTE.

**PRIORITY 3 — GOOD TO HAVE (Reviewer satisfaction):** 10. **Multiple Imputation Sensitivity Analysis:** Compare MNAR-zero fill vs MICE. 11. **Subgroup Analysis:** Urban vs rural, education levels. 12. **Feature Stability Analysis:** Bootstrap feature selection to check which features are consistently selected. 13. **Socioeconomic Gradient Analysis:** Dose-response relationship for key predictors.

---

## SECTION 6: NARRATIVE FOR RESEARCH IMPACT

### How to Frame This Research:

> _"Using the first nationally representative ML analysis of women's mental health in Bangladesh (BDHS 2022, n=30,078), this study identifies shared sociodemographic risk factors across three co-occurring conditions — depression (3.4%), anxiety (3.0%), and disability (4.7%). A convergent feature set (20 shared predictors) dominated by reproductive health variables—age at first birth, contraceptive autonomy, spousal age difference, and wealth—consistently discriminates cases across all three outcomes (ROC-AUC: 75.1–75.6%). This suggests a unified social determinants pathway to mental health burden in low-income South Asian women, with implications for integrated screening programs."_

### Key Contributions to Claim:

1. **First ML study** on BDHS 2022 mental health data (novelty)
2. **Shared determinant convergence** across 3 outcomes (new theory)
3. **Contraceptive autonomy** (contra_decision3) as a mental health predictor (policy relevance)
4. **Spousal age gap** (age_dif) consistently selected — gender power dynamics finding
5. **V511 (Age at first cohabitation)** = top RF importance feature — child marriage link
6. **Rural vs urban** and **wealth index** confirm SES gradient (SDG evidence)

---

## SECTION 7: DETAILED CRITICAL ANALYSIS — THE PARADOX PROBLEM

### The Accuracy Paradox — Why 97% Accuracy Is Meaningless Here:

**For Depression (3.4% prevalence):**

- A model that predicts "no depression" for EVERYONE gets: 96.6% accuracy
- XGBoost gets 96.5% accuracy by detecting ~1% of cases
- These numbers are nearly indistinguishable — accuracy is useless here

**Mathematical Proof of the Paradox:**

```
Prevalence = 3.4% → Majority class = 96.6%
"Always-No" classifier: Accuracy = 96.6%, Precision=0%, Recall=0%, F1=0%
XGBoost: Accuracy = 96.5%, Precision=30%, Recall=1.5%, F1=2.79%
```

The "best model" is barely better than always predicting "no" in terms of classification utility.

**Key Insight for the Paper:**

> The ROC-AUC of 75% is the REAL performance metric here. It shows that the model can RANK individuals by risk in the right order 75% of the time — which IS clinically valuable for population-level risk stratification even if it cannot reliably classify individual cases. The framing should shift from "prediction" to "risk stratification."

---

## SECTION 8: SUGGESTED NARRATIVE REFRAME (Q1-Compatible)

### Current (Wrong) Framing:

"ML models predict depression/anxiety/disability with 96% accuracy"
→ Misleading, will be rejected

### Correct (Q1) Framing:

"ML models identify sociodemographic risk profiles associated with mental health burden in Bangladeshi women, with moderate discriminatory ability (AUC: 0.75), supporting population-level risk stratification rather than individual clinical diagnosis"

### Why This Works:

- Honest about limitations
- Positions ROC-AUC correctly
- Aligns with real-world use (survey-based screening, not clinical diagnosis)
- Matches what the data can actually support

---

## SECTION 9: JOURNAL SUBMISSION STRATEGY

| Journal                             | IF  | Scope Match  | AUC Requirement  | Likelihood              |
| ----------------------------------- | --- | ------------ | ---------------- | ----------------------- |
| Journal of Affective Disorders      | 6.5 | ✅ High      | 0.80+ preferred  | ⚠️ Hard                 |
| Asian Journal of Psychiatry         | 4.6 | ✅ Very High | 0.75+ acceptable | ✅ Feasible             |
| SSM – Population Health             | 4.1 | ✅ High      | Methods-flexible | ✅ Feasible             |
| BMC Psychiatry                      | 3.4 | ✅ High      | 0.75+ acceptable | ✅ Best fit             |
| Int. J. Environ. Res. Public Health | 4.6 | ✅ High      | 0.75 acceptable  | ✅ Feasible             |
| PLOS ONE                            | 3.7 | ✅ Broad     | Sound methods    | ✅ Feasible after fixes |

**Recommended First Target:** BMC Psychiatry or Asian Journal of Psychiatry after implementing Priority 1 fixes.

---

## SECTION 10: FINAL VERDICT SCORECARD

| Criterion                | Current Score | Required for Q1 | Gap      |
| ------------------------ | ------------- | --------------- | -------- |
| Dataset quality          | 9/10 ✅       | 8/10            | None     |
| Sample size              | 9/10 ✅       | 8/10            | None     |
| Novelty                  | 8/10 ✅       | 7/10            | None     |
| Data leakage prevention  | 8/10 ✅       | 7/10            | None     |
| Model performance (AUC)  | 6/10 ⚠️       | 7/10            | -1       |
| Minority class detection | 2/10 ❌       | 7/10            | -5       |
| Confidence intervals     | 0/10 ❌       | 8/10            | -8       |
| Hyperparameter tuning    | 2/10 ❌       | 6/10            | -4       |
| Explainability (SHAP)    | 0/10 ❌       | 7/10            | -7       |
| Calibration assessment   | 0/10 ❌       | 5/10            | -5       |
| Writing & framing        | 5/10 ⚠️       | 8/10            | -3       |
| **OVERALL**              | **4.5/10**    | **7/10**        | **-2.5** |

---

## SECTION 11: EXECUTIVE SUMMARY (For Supervisor)

### What is good:

- The dataset is excellent (BDHS 2022, n=30,078, nationally representative)
- The research gap is real (Bangladesh mental health ML = underexplored)
- The methodology is cleaner than average (no data leakage, proper split)
- The finding that 20 features are shared across all 3 outcomes is genuinely novel and worth publishing

### What is broken:

- The "best models" (XGBoost, Random Forest) literally detect ZERO positive cases for anxiety and nearly zero for depression/disability — this is a classification failure, not a success
- Reporting 97% accuracy on a 3% prevalence dataset is academically incorrect framing
- No confidence intervals = no statistical validity
- No SHAP = no explainability = not current standard

### What must happen before submission:

1. Fix threshold optimization (2-3 lines of code change) → Will fix the F1=0% problem
2. Add bootstrap CIs → Shows statistical rigor
3. Add SHAP values → Meets 2024-2026 standard for ML health papers
4. Reframe from "prediction" to "risk stratification" → Honest and publishable

### Realistic timeline assessment:

- With the above 4 fixes: Target PLOS ONE / BMC Psychiatry / Asian J Psychiatry
- Without these fixes: Rejection at any Q1 journal is near-certain

---

_Assessment prepared by: Senior ML Research Expert | Standard: Oxford-level peer review_
_Date: March 2026 | Framework: TRIPOD guidelines for ML health studies_
