# 🔍 Dataset & Requirement Mismatch Analysis
## New Dataset: `clean_mental_health_data.csv` vs. Previous Scripts

---

## ✅ What the New Dataset Has

**Columns in `clean_mental_health_data.csv` (Header row 1):**

| Column | Description |
|--------|-------------|
| `CASEID` | Case identifier (should be dropped) |
| `V021` | **Primary Sampling Unit (PSU)** ⚠️ |
| `V023` | **Strata** ⚠️ |
| `Division` | Division (renamed ✅) |
| `Residence` | Urban/Rural (renamed ✅) |
| `Respondent_Education` | Education level (renamed ✅) |
| `V171A` | Internet frequency (NOT renamed) |
| `V312` | Contraceptive method (NOT renamed) |
| `V511` | Age at first cohabitation (NOT renamed) |
| `V632` | Contraception decision maker (NOT renamed) |
| `V701` | Husband's education (NOT renamed) |
| `V730` | Husband's age (NOT renamed) |
| `age_at_First_Birth` | Age at first birth (renamed ✅) |
| `Respondent_Age` | Age (renamed ✅) |
| `Parity` | Number of children (renamed ✅) |
| `Spousal_Age_Gap` | Husband-wife age difference (renamed ✅) |
| `Wealth_Index` | Wealth index (renamed ✅) |
| `Internet_Use` | Internet use (renamed ✅) |
| `Contraceptive_Use` | Contraceptive use (renamed ✅) |
| `contra_decision_maker` | Contraceptive decision maker (renamed ✅) |
| `Weight` | **Survey weight** ⚠️ |
| `Depression` | Depression outcome |
| `Anxiety` | Anxiety outcome |
| `Mental_Health` | Combined mental health outcome |

---

## ❌ Problems Found (Mismatches with Mam's Requirements)

### Problem 1: V021, V023, Weight Used as Independent Variables (MAM'S COMPLAINT ✅)
> _"Primary Sampling Unit, Weight, Strata ei 3ta ke independent variable hisebe drchilen"_

**In the OLD scripts (`disu_analysis.py`, `anxiety_analysis.py`, etc.):**
- The old dataset `BDHS2022_MH_ML_ready_1.csv` contained `V021` (PSU), `V023` (Strata), and survey weights
- These were NOT properly excluded → used as features in ML models
- **This is a serious methodological error!** PSU, Strata, and Weight are **survey design variables**, NOT predictors of mental health

**In the NEW dataset `clean_mental_health_data.csv`:**
- `V021` (PSU) = still present → **must be dropped**
- `V023` (Strata) = still present → **must be dropped**
- `Weight` = still present → **must be dropped**

---

### Problem 2: Same Variable Included Twice — Categorized AND Raw (MAM'S COMPLAINT ✅)
> _"Eki variable 2 bar kore nisen — category kora and chara"_

**In the OLD dataset/scripts**, variables like:
- `age_dif` (raw age difference) AND `age_dif_cat` (categorized version) — both included
- `internet_use` (raw) AND `Internet_Use` (possibly recoded) — duplicates
- `parity_cat` (categorized parity) AND parity/V201 (raw) — duplicates
- `contra_decision3` AND `V632` (contraception decision) — same variable twice

**In the NEW dataset `clean_mental_health_data.csv`:**
- `V171A` AND `Internet_Use` — possibly the SAME internet variable (V171A = internet frequency, Internet_Use = binary recoded) ⚠️ **Check needed**
- `V632` AND `contra_decision_maker` — SAME variable (V632 is the raw code, contra_decision_maker is renamed) ⚠️ **Double inclusion**
- `V312` AND `Contraceptive_Use` — possibly same (V312 = method, Contraceptive_Use = binary) ⚠️ **Check needed**
- `V511` — age at first cohabitation (not renamed, raw code still included alongside `age_at_First_Birth`)
- `V701`, `V730` — husband's education and age (raw codes, not renamed, still in data)

---

### Problem 3: Only Renamed Variables Should Be Used as Features
> _"J variable gula ami rename korsi sudu tader ke niye feature selection korben"_

**Variables that were RENAMED (use ONLY these as features):**

| New Name | Original | Status |
|----------|----------|--------|
| `Division` | Geographic division | ✅ Use |
| `Residence` | Urban/Rural | ✅ Use |
| `Respondent_Education` | Education | ✅ Use |
| `age_at_First_Birth` | Age at 1st birth | ✅ Use |
| `Respondent_Age` | Age | ✅ Use |
| `Parity` | Number of children | ✅ Use |
| `Spousal_Age_Gap` | Age gap with husband | ✅ Use |
| `Wealth_Index` | Wealth index | ✅ Use |
| `Internet_Use` | Internet use (binary) | ✅ Use |
| `Contraceptive_Use` | Contraceptive use (binary) | ✅ Use |
| `contra_decision_maker` | Who decides contraception | ✅ Use |

**Variables to DROP (NOT renamed = should not be used):**

| Column | Reason |
|--------|--------|
| `CASEID` | Survey identifier — no predictive value |
| `V021` | Primary Sampling Unit — survey design variable |
| `V023` | Strata — survey design variable |
| `Weight` | Survey weight — design variable, not predictor |
| `V171A` | Raw code; Internet_Use already covers this |
| `V312` | Raw code; Contraceptive_Use already covers this |
| `V511` | Raw code (age first cohabitation) — not renamed |
| `V632` | Raw code; contra_decision_maker already covers this |
| `V701` | Raw code (husband's education) — not renamed |
| `V730` | Raw code (husband's age) — not renamed |

---

### Problem 4: OLD Scripts Use Wrong Data Path
**Old scripts point to:** `e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv`  
**New dataset is at:** `s:\Anthic kumar singh\womanMachineLearning\clean_mental_health_data.csv`

---

### Problem 5: Target Variable Names Changed
| Old Dataset | New Dataset |
|------------|------------|
| `dep` | `Depression` |
| `anx` | `Anxiety` |
| `disu` | `Mental_Health` (combined?) |

> [!NOTE]
> The new dataset has `Mental_Health` (0/2 values visible in data) which seems to be the composite/combined outcome. `Depression` and `Anxiety` are binary (0/1).

---

## 📋 Summary: What New Script Must Fix

1. **Drop `V021`, `V023`, `Weight`** — survey design variables, NOT predictors
2. **Drop all raw-code columns** (`V171A`, `V312`, `V511`, `V632`, `V701`, `V730`) — already represented by renamed versions
3. **Use ONLY renamed variables** as feature candidates
4. **New target variable names**: `Depression`, `Anxiety`, `Mental_Health`
5. **Update data path** to new CSV
6. **Hyperparameter tuning** required (GridSearchCV or RandomizedSearchCV)
7. **Show best parameter values** for each model
8. **Confusion matrix for all models**
9. **All metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, MCC
10. **Cross-validation** (5-fold)
11. **SHAP analysis** on best model

---

## 🔑 Correct Feature Set (11 renamed variables)

```python
RENAMED_FEATURES = [
    'Division',              # Geographic division
    'Residence',             # Urban/Rural
    'Respondent_Education',  # Education level
    'age_at_First_Birth',    # Age at first birth
    'Respondent_Age',        # Respondent's age
    'Parity',                # Number of children
    'Spousal_Age_Gap',       # Age gap with husband
    'Wealth_Index',          # Wealth index
    'Internet_Use',          # Internet use (binary)
    'Contraceptive_Use',     # Contraceptive use (binary)
    'contra_decision_maker', # Who decides contraception
]

# Variables to EXCLUDE:
EXCLUDE_COLS = [
    'CASEID',   # Identifier
    'V021',     # PSU (survey design)
    'V023',     # Strata (survey design)
    'Weight',   # Survey weight (design)
    'V171A',    # Duplicate of Internet_Use
    'V312',     # Duplicate of Contraceptive_Use
    'V511',     # Raw code, not renamed
    'V632',     # Duplicate of contra_decision_maker
    'V701',     # Raw code, not renamed
    'V730',     # Raw code, not renamed
    # Targets (exclude when predicting others):
    'Depression', 'Anxiety', 'Mental_Health'
]
```

---

> [!IMPORTANT]
> The new complete analysis script will be built for **3 outcomes**: `Depression`, `Anxiety`, and `Mental_Health` — each as a separate analysis with: missing data check → feature selection (only renamed vars) → SMOTE → hyperparameter tuning → confusion matrices → all metrics → cross-validation → SHAP analysis.
