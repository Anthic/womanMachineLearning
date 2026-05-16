# Data Audit Before Final ML Fix

## Dataset
- File: `clean_mental_health_data.csv`
- Shape: `11,523` rows x `24` columns
- Columns: `CASEID, V021, V023, Division, Residence, Respondent_Education, V171A, V312, V511, V632, V701, V730, age_at_First_Birth, Respondent_Age, Parity, Spousal_Age_Gap, Wealth_Index, Internet_Use, Contraceptive_Use, contra_decision_maker, Weight, Depression, Anxiety, Mental_Health`

## Approved Feature Rule Check

Only these 11 renamed predictors should be used:
- `Division`
- `Residence`
- `Respondent_Education`
- `age_at_First_Birth`
- `Respondent_Age`
- `Parity`
- `Spousal_Age_Gap`
- `Wealth_Index`
- `Internet_Use`
- `Contraceptive_Use`
- `contra_decision_maker`

Excluded variables are present but must not be predictors:
- `CASEID`: present
- `V021`: present
- `V023`: present
- `Weight`: present
- `V171A`: present
- `V312`: present
- `V511`: present
- `V632`: present
- `V701`: present
- `V730`: present
- `Depression`: present
- `Anxiety`: present
- `Mental_Health`: present

Feature/excluded overlap: `[]`

## Target Distribution and Baseline Accuracy

| Outcome | Raw values | Binary mapping used | Negative | Positive | Positive % | Majority accuracy baseline |
|---|---|---|---:|---:|---:|---:|
| `Depression` | `{0.0: 10706, 1.0: 817}` | `0 = no`, `1 = yes` | 10,706 | 817 | 7.09% | 92.91% |
| `Anxiety` | `{0.0: 10811, 1.0: 712}` | `0 = no`, `1 = yes` | 10,811 | 712 | 6.18% | 93.82% |
| `Mental_Health` | `{0.0: 10399, 1.0: 412, 2.0: 307, 3.0: 405}` | `0 = no`, `>0 = any problem` | 10,399 | 1,124 | 9.75% | 90.25% |

## Missing Values in Approved Features and Targets

| Column | Missing count | Missing % |
|---|---:|---:|
| `Division` | 0 | 0.00% |
| `Residence` | 0 | 0.00% |
| `Respondent_Education` | 0 | 0.00% |
| `age_at_First_Birth` | 0 | 0.00% |
| `Respondent_Age` | 0 | 0.00% |
| `Parity` | 0 | 0.00% |
| `Spousal_Age_Gap` | 0 | 0.00% |
| `Wealth_Index` | 0 | 0.00% |
| `Internet_Use` | 0 | 0.00% |
| `Contraceptive_Use` | 0 | 0.00% |
| `contra_decision_maker` | 0 | 0.00% |
| `Depression` | 0 | 0.00% |
| `Anxiety` | 0 | 0.00% |
| `Mental_Health` | 0 | 0.00% |

## Feature Coding Audit

| Feature | Non-missing | Unique values | Min | Max | Likely type | Values preview |
|---|---:|---:|---:|---:|---|---|
| `Division` | 11,523 | 8 | 1.00 | 8.00 | categorical/ordinal code | `1, 2, 3, 4, 5, 6, 7, 8` |
| `Residence` | 11,523 | 2 | 1.00 | 2.00 | categorical/ordinal code | `1, 2` |
| `Respondent_Education` | 11,523 | 4 | 0.00 | 3.00 | categorical/ordinal code | `0, 1, 2, 3` |
| `age_at_First_Birth` | 11,523 | 2 | 1.00 | 2.00 | categorical/ordinal code | `1, 2` |
| `Respondent_Age` | 11,523 | 3 | 1.00 | 3.00 | categorical/ordinal code | `1, 2, 3` |
| `Parity` | 11,523 | 3 | 1.00 | 3.00 | categorical/ordinal code | `1, 2, 3` |
| `Spousal_Age_Gap` | 11,523 | 4 | 1.00 | 4.00 | categorical/ordinal code | `1, 2, 3, 4` |
| `Wealth_Index` | 11,523 | 3 | 1.00 | 3.00 | categorical/ordinal code | `1, 2, 3` |
| `Internet_Use` | 11,523 | 2 | 0.00 | 1.00 | categorical/ordinal code | `0, 1` |
| `Contraceptive_Use` | 11,523 | 2 | 0.00 | 1.00 | categorical/ordinal code | `0, 1` |
| `contra_decision_maker` | 11,523 | 3 | 0.00 | 2.00 | categorical/ordinal code | `0, 1, 2` |

## Raw Duplicate Relationship Check

| Raw variable | Approved variable | Pearson corr | Same nonmissing rows | Note |
|---|---|---:|---:|---|
| `V171A` | `Internet_Use` | 0.984 | 11,523 | duplicate/derived relationship likely |
| `V312` | `Contraceptive_Use` | 0.009 | 11,523 | relationship weak or recoded nonlinearly |
| `V632` | `contra_decision_maker` | 0.755 | 11,523 | duplicate/derived relationship likely |

## Simple Feature Signal Screening

Mutual information and Chi-square were calculated on approved features only after median imputation. These are screening signals, not final model performance.

### Depression
| Rank | Feature | MI | Chi2 |
|---:|---|---:|---:|
| 1 | `Contraceptive_Use` | 0.00357 | 0.623 |
| 2 | `Parity` | 0.00315 | 10.792 |
| 3 | `age_at_First_Birth` | 0.00240 | 1.013 |
| 4 | `contra_decision_maker` | 0.00219 | 0.045 |
| 5 | `Respondent_Age` | 0.00162 | 6.570 |
| 6 | `Division` | 0.00133 | 21.279 |
| 7 | `Wealth_Index` | 0.00105 | 2.742 |
| 8 | `Respondent_Education` | 0.00085 | 9.141 |
| 9 | `Residence` | 0.00000 | 0.727 |
| 10 | `Spousal_Age_Gap` | 0.00000 | 0.548 |
| 11 | `Internet_Use` | 0.00000 | 5.749 |

### Anxiety
| Rank | Feature | MI | Chi2 |
|---:|---|---:|---:|
| 1 | `Contraceptive_Use` | 0.00768 | 1.408 |
| 2 | `Division` | 0.00322 | 2.691 |
| 3 | `Parity` | 0.00267 | 24.247 |
| 4 | `age_at_First_Birth` | 0.00230 | 0.263 |
| 5 | `Residence` | 0.00223 | 0.393 |
| 6 | `Wealth_Index` | 0.00210 | 0.057 |
| 7 | `Respondent_Age` | 0.00171 | 10.510 |
| 8 | `contra_decision_maker` | 0.00072 | 0.131 |
| 9 | `Spousal_Age_Gap` | 0.00058 | 2.069 |
| 10 | `Respondent_Education` | 0.00000 | 10.268 |
| 11 | `Internet_Use` | 0.00000 | 3.417 |

### Mental_Health
| Rank | Feature | MI | Chi2 |
|---:|---|---:|---:|
| 1 | `age_at_First_Birth` | 0.00554 | 1.158 |
| 2 | `Contraceptive_Use` | 0.00542 | 1.539 |
| 3 | `Division` | 0.00430 | 13.310 |
| 4 | `Parity` | 0.00388 | 22.163 |
| 5 | `contra_decision_maker` | 0.00368 | 0.053 |
| 6 | `Respondent_Age` | 0.00364 | 11.895 |
| 7 | `Wealth_Index` | 0.00090 | 1.140 |
| 8 | `Respondent_Education` | 0.00016 | 13.617 |
| 9 | `Residence` | 0.00000 | 0.377 |
| 10 | `Spousal_Age_Gap` | 0.00000 | 1.879 |
| 11 | `Internet_Use` | 0.00000 | 5.952 |

## Quick Baseline Logistic Check

This check uses one-hot encoding and class_weight balanced. It is only for diagnosis, not final publication results.

| Outcome | ROC-AUC | Accuracy | Positive prevalence | Comment |
|---|---:|---:|---:|---|
| `Depression` | 54.50% | 58.79% | 7.09% | weak signal |
| `Anxiety` | 59.63% | 59.26% | 6.18% | weak signal |
| `Mental_Health` | 59.35% | 58.18% | 9.75% | weak signal |

## Problems Found

1. Severe class imbalance: positive class is only 6-10%, so accuracy is misleading.
2. Approved predictors are mostly coded categories/ordinal variables. Plain numeric SMOTE can create artificial category values and should be avoided.
3. Current result files and current script are not fully synchronized. Some result summaries mention SMOTEN/one-hot/threshold tuning, while current script still shows numeric SMOTE/median-imputation code.
4. Mental_Health is stored as 0/1/2/3. The binary mapping `>0 = any problem` needs explicit confirmation from Mam.
5. Feature signal appears weak. Even one-hot logistic baseline gives ROC-AUC around the mid-50s, meaning the approved 11 variables alone may not predict outcomes strongly.
6. If threshold tuning is reported, threshold must be selected using validation/CV, not optimized directly on final test set.

## Recommended Fix Plan

1. Make one final reproducible script and regenerate all outcomes from that exact script.
2. Treat approved predictors as categorical/ordinal safely: most-frequent imputation + one-hot encoding, or carefully justified ordinal handling.
3. Prefer class_weight and threshold tuning first; use SMOTEN only if oversampling is necessary and documented.
4. Add majority baseline, balanced accuracy, PR-AUC, ROC-AUC, recall, specificity, F1, MCC, and confusion matrices.
5. Use validation-safe threshold tuning: train/validation/test split or CV threshold selection, then final untouched test results.
6. Confirm Mental_Health mapping before final analysis.
7. If performance remains weak, report honestly as limited predictive value of the approved 11 socio-demographic/reproductive variables, not as a coding failure.
