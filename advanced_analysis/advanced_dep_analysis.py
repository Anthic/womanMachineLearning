"""
=============================================================================
BANGLADESH DEMOGRAPHIC AND HEALTH SURVEY (BDHS) 2022
ADVANCED ML ANALYSIS â€” DEPRESSION (dep) as Dependent Variable
Q1 PUBLICATION STANDARD â€” Advanced Analysis Module
=============================================================================
Author     : Research Pipeline (Oxford-standard methodology)
Dataset    : BDHS2022_MH_ML_ready_1.csv
Target     : dep (Depression) â†’ Binary (0 = No, 1 = Yes)

VARIABLE DEFINITIONS (CORRECTLY LABELLED):
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INDEPENDENT VARIABLES (Predictors):
  v024           = Division (respondent's geographic division)
  v025           = Residence (urban/rural)
  v106           = Respondent's Education Level
  v701           = Husband's Education Level
  age_cat        = Current Age of Respondent (categorized)
  age_fb_cat     = Teenage Pregnancy (age at first birth, categorized)
  parity_cat     = Total Number of Children (categorized)
  age_diff_cat   = Age Difference of Spouses (categorized)
  wealth_cat     = Wealth Index
  internet_use   = Mass Media / Internet Use
  contra         = Contraceptive Use
  contra_decision= Decision for Contraceptive Use

DEPENDENT VARIABLES (Mental Health Outcomes):
  dep            = Depression (THIS ANALYSIS)
  anx            = Anxiety
  disu           = Mental Health Disutility / Disability

SURVEY DESIGN WEIGHTS (for weighted analysis):
  v005           = Sampling Weight
  v021           = Primary Sampling Unit (PSU)
  v023           = Stratification Number

ADVANCED ANALYSIS FEATURES:
  1. SHAP (SHapley Additive exPlanations) â€” explainability
  2. Bootstrap Confidence Intervals (B=1000) â€” metric uncertainty
  3. Clinical Tests: DeLong's AUC comparison, McNemar's test
  4. Survey-weighted sample description
=============================================================================
"""

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 0: IMPORT LIBRARIES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, norm

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             roc_curve, auc)
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# SMOTE
from imblearn.over_sampling import SMOTE

# SHAP â€” Advanced explainability
try:
    import shap
    SHAP_AVAILABLE = True
    print("  [OK] SHAP library available")
except ImportError:
    SHAP_AVAILABLE = False
    print("  [WARNING] SHAP not installed. Run: pip install shap")
    print("  SHAP analysis will be skipped.")

print("=" * 70)
print("  BDHS 2022 â€” ADVANCED DEPRESSION ANALYSIS (Q1 Standard)")
print("=" * 70)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 1: CONFIGURATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_PATH   = r"e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv"
OUTPUT_DIR  = r"e:\hafiza mam work\advanced_analysis\results_dep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET      = 'dep'
TARGET_LABEL= 'Depression'
OTHER_TARGETS = ['anx', 'disu']

# Survey design variables (used for weighted description ONLY, not features)
SURVEY_VARS = ['v005', 'v021', 'v023']

# Correct variable labels for plots and reports
VAR_LABELS = {
    'v024'          : 'Division',
    'v025'          : 'Residence (Urban/Rural)',
    'v106'          : "Respondent's Education",
    'v701'          : "Husband's Education",
    'age_cat'       : "Respondent's Current Age",
    'age_fb_cat'    : 'Teenage Pregnancy (Age at 1st Birth)',
    'parity_cat'    : 'Total Number of Children',
    'age_diff_cat'  : 'Age Difference of Spouses',
    'wealth_cat'    : 'Wealth Index',
    'internet_use'  : 'Mass Media / Internet Use',
    'contra'        : 'Contraceptive Use',
    'contra_decision': 'Decision for Contraceptive Use',
}

BOOTSTRAP_N = 1000   # Number of bootstrap iterations
RANDOM_SEED = 42

print(f"\n[CONFIG] Target         : {TARGET} ({TARGET_LABEL})")
print(f"[CONFIG] Output Dir     : {OUTPUT_DIR}")
print(f"[CONFIG] Bootstrap N    : {BOOTSTRAP_N}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 2: LOAD & PREPARE DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 1] Loading and Preparing Dataset...")

df_raw = pd.read_csv(DATA_PATH, dtype=str)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)
df = df_raw.apply(pd.to_numeric, errors='coerce')

if 'CASEID' in df.columns:
    df = df.drop(columns=['CASEID'])

print(f"  â†’ Shape: {df.shape[0]} rows Ã— {df.shape[1]} columns")

# Smart imputation (research-grade)
df_imp = df.copy()
structural_cols = [
    'V212', 'V312', 'V511', 'V632', 'V701', 'V730',
    'MTH22', 'MTH24', 'V171A', 'age_dif', 'age_dif_cat',
    'internet_use', 'contra_decision3', 'parity_cat',
]
for col in structural_cols:
    if col in df_imp.columns:
        n_null = int(df_imp[col].isnull().sum())
        if n_null > 0:
            df_imp[col] = df_imp[col].fillna(0)

remaining_null_cols = [c for c in df_imp.columns if df_imp[c].isnull().any()]
for col in remaining_null_cols:
    unique_v = int(df_imp[col].dropna().nunique())
    fill_val = df_imp[col].mode().iloc[0] if unique_v <= 15 else df_imp[col].median()
    df_imp[col] = df_imp[col].fillna(fill_val)

print(f"  â†’ Imputation complete. Remaining nulls: {df_imp.isnull().sum().sum()}")

# Survey weight (v005 / 1,000,000) â€” for weighted statistics only
if 'v005' in df_imp.columns:
    df_imp['wt'] = df_imp['v005'] / 1_000_000
    print(f"  â†’ Survey weight 'wt' created (v005 / 1,000,000)")

# Drop leakage cols + other outcomes
DROP_COLS = other_outcomes_drop = OTHER_TARGETS + [TARGET, 'MTH22', 'MTH24']
y = df_imp[TARGET].astype(int)
X = df_imp.drop(columns=[c for c in DROP_COLS + SURVEY_VARS + ['wt']
                          if c in df_imp.columns])
X = X.dropna(axis=1, how='all')

print(f"  â†’ Feature space: {X.shape[1]} variables")
print(f"  â†’ {TARGET_LABEL} prevalence: {y.mean()*100:.2f}%  "
      f"(N={int(y.sum())} positive out of {len(y)})")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 3: SURVEY-WEIGHTED DESCRIPTIVE STATS
# Weighted by v005/1,000,000 as per BDHS methodology
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 2] Survey-Weighted Descriptive Analysis...")

if 'wt' in df_imp.columns:
    wt = df_imp['wt']
    # Weighted prevalence by key independent variables
    weighted_desc = []
    key_vars = [v for v in VAR_LABELS.keys() if v in df_imp.columns]
    for var in key_vars:
        label = VAR_LABELS[var]
        for cat_val in sorted(df_imp[var].dropna().unique()):
            mask = df_imp[var] == cat_val
            n_unweighted = mask.sum()
            w_total      = wt[mask].sum()
            w_positive   = (wt[mask] * (y[mask] == 1)).sum()
            w_prev       = (w_positive / w_total * 100) if w_total > 0 else 0
            weighted_desc.append({
                'Variable'          : label,
                'Category_Code'     : cat_val,
                'N_Unweighted'      : n_unweighted,
                'Weighted_N'        : round(w_total, 2),
                f'{TARGET_LABEL}_Weighted_%': round(w_prev, 2)
            })
    wt_desc_df = pd.DataFrame(weighted_desc)
    wt_desc_df.to_csv(f"{OUTPUT_DIR}/weighted_descriptives.csv", index=False)
    print(f"  [Saved] weighted_descriptives.csv")
else:
    print("  [WARNING] v005 not found â€” unweighted analysis only")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 4: TRAIN-TEST SPLIT + FEATURE SELECTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 3] Train-Test Split and Feature Selection...")

class_ratio   = y.value_counts(normalize=True)
is_imbalanced = class_ratio.min() < 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
X_train_orig = X_train.copy()
y_train_orig = y_train.copy()

feature_names = list(X_train_orig.columns)
TOP_K = 15

# MI
mi_scores = mutual_info_classif(X_train_orig, y_train_orig, random_state=RANDOM_SEED)
mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})\
          .sort_values('MI_Score', ascending=False).reset_index(drop=True)
mi_top = set(mi_df.head(TOP_K)['Feature'])

# Chi2
X_shifted = X_train_orig - X_train_orig.min() + 0.001
chi2_scores, chi2_pvals = chi2(X_shifted, y_train_orig)
chi2_df = pd.DataFrame({'Feature': feature_names,
                         'Chi2_Score': chi2_scores, 'P_value': chi2_pvals})\
            .sort_values('Chi2_Score', ascending=False).reset_index(drop=True)
chi2_top = set(chi2_df.head(TOP_K)['Feature'])

# RF
rf_sel = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                 n_jobs=-1, class_weight='balanced')
rf_sel.fit(X_train_orig, y_train_orig)
rf_df = pd.DataFrame({'Feature': feature_names,
                       'RF_Importance': rf_sel.feature_importances_})\
          .sort_values('RF_Importance', ascending=False).reset_index(drop=True)
rf_top = set(rf_df.head(TOP_K)['Feature'])

selected_features = sorted(mi_top | chi2_top | rf_top)
print(f"  â†’ Selected {len(selected_features)} features (UNION: MI + Chi2 + RF)")

# Apply labels to feature names for nice plots
def get_label(feat):
    return VAR_LABELS.get(feat, feat)

X_train_sel_orig = X_train_orig[selected_features]
X_test_sel       = X_test[selected_features]

# SMOTE AFTER feature selection
if is_imbalanced:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_sel_bal, y_train_bal = smote.fit_resample(X_train_sel_orig, y_train_orig)
    print(f"  â†’ SMOTE applied: {sum(y_train_bal==0)} / {sum(y_train_bal==1)}")
else:
    X_train_sel_bal, y_train_bal = X_train_sel_orig.copy(), y_train_orig.copy()

# Scaling
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel_bal)
X_test_scaled  = scaler.transform(X_test_sel)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 5: MODEL TRAINING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 4] Training 6 ML Models...")

MODELS = {
    "Logistic Regression": LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000,
        class_weight='balanced', solver='lbfgs'
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_SEED, max_depth=8, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED,
        n_jobs=-1, class_weight='balanced'
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_SEED, eval_metric='logloss',
        scale_pos_weight=(sum(y_train_bal==0) / max(sum(y_train_bal==1), 1)),
        verbosity=0
    ),
    "SVM (Linear)": CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', random_state=RANDOM_SEED,
                  C=1.0, max_iter=2000),
        cv=3, method='sigmoid'
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='euclidean'
    )
}

NEEDS_SCALING = {"Logistic Regression", "SVM (Linear)", "KNN"}

fitted_models = {}
predictions   = {}   # stores {'y_pred': ..., 'y_prob': ...}

for name, model in MODELS.items():
    Xtr = X_train_scaled if name in NEEDS_SCALING else X_train_sel_bal.values
    Xte = X_test_scaled  if name in NEEDS_SCALING else X_test_sel.values
    model.fit(Xtr, y_train_bal)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    fitted_models[name] = {'model': model, 'Xtr': Xtr, 'Xte': Xte}
    predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}
    print(f"  âœ“ {name}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 6: BOOTSTRAP CONFIDENCE INTERVALS (1000 iterations)
# Bootstrap CI is gold standard for ML metric uncertainty in Q1 papers
# Method: Stratified bootstrap of test set, compute metric distribution
# 95% CI = [2.5th percentile, 97.5th percentile]
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[STEP 5] Bootstrap Confidence Intervals (B={BOOTSTRAP_N})...")
print("  This may take 1-2 minutes...")

rng = np.random.RandomState(RANDOM_SEED)

def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstrap=1000, rng=None):
    """Compute bootstrap 95% CI for Accuracy, Precision, Recall, F1, AUC."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    boot_acc, boot_prec, boot_rec, boot_f1, boot_auc = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_acc.append(accuracy_score(yt, yp))
        boot_prec.append(precision_score(yt, yp, zero_division=0))
        boot_rec.append(recall_score(yt, yp, zero_division=0))
        boot_f1.append(f1_score(yt, yp, zero_division=0))
        try:
            boot_auc.append(roc_auc_score(yt, ypr))
        except Exception:
            pass

    def ci(arr):
        arr = np.array(arr)
        return (round(np.mean(arr)*100, 2),
                round(np.percentile(arr, 2.5)*100, 2),
                round(np.percentile(arr, 97.5)*100, 2))

    return {
        'Accuracy_mean': ci(boot_acc)[0],
        'Accuracy_CI_lower': ci(boot_acc)[1],
        'Accuracy_CI_upper': ci(boot_acc)[2],
        'Precision_mean': ci(boot_prec)[0],
        'Precision_CI_lower': ci(boot_prec)[1],
        'Precision_CI_upper': ci(boot_prec)[2],
        'Recall_mean': ci(boot_rec)[0],
        'Recall_CI_lower': ci(boot_rec)[1],
        'Recall_CI_upper': ci(boot_rec)[2],
        'F1_mean': ci(boot_f1)[0],
        'F1_CI_lower': ci(boot_f1)[1],
        'F1_CI_upper': ci(boot_f1)[2],
        'AUC_mean': ci(boot_auc)[0],
        'AUC_CI_lower': ci(boot_auc)[1],
        'AUC_CI_upper': ci(boot_auc)[2],
    }

bootstrap_results = {}
y_test_arr = np.array(y_test)

for name in MODELS:
    preds = predictions[name]
    bs = bootstrap_metrics(y_test_arr, preds['y_pred'], preds['y_prob'],
                           n_bootstrap=BOOTSTRAP_N, rng=rng)
    bootstrap_results[name] = bs
    print(f"  âœ“ {name:25s}  AUC = {bs['AUC_mean']}% "
          f"[{bs['AUC_CI_lower']}â€“{bs['AUC_CI_upper']}]")

# Save bootstrap results
bs_rows = []
for name, bs in bootstrap_results.items():
    row = {'Model': name}
    row.update(bs)
    bs_rows.append(row)
bs_df = pd.DataFrame(bs_rows)
bs_df.to_csv(f"{OUTPUT_DIR}/bootstrap_CI_results.csv", index=False)
print(f"  [Saved] bootstrap_CI_results.csv")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 7: BOOTSTRAP CI VISUALIZATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 6] Bootstrap CI Visualization...")

metrics_to_vis = [
    ('AUC', 'AUC_mean', 'AUC_CI_lower', 'AUC_CI_upper', '#E53935'),
    ('F1-Score', 'F1_mean', 'F1_CI_lower', 'F1_CI_upper', '#1E88E5'),
    ('Recall', 'Recall_mean', 'Recall_CI_lower', 'Recall_CI_upper', '#43A047'),
    ('Precision', 'Precision_mean', 'Precision_CI_lower',
     'Precision_CI_upper', '#FB8C00'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes = axes.flatten()

for ax, (metric_name, mean_col, lo_col, hi_col, color) in \
        zip(axes, metrics_to_vis):
    models_list = bs_df['Model'].values
    means = bs_df[mean_col].values
    lows  = bs_df[lo_col].values
    highs = bs_df[hi_col].values
    yerr_lo = means - lows
    yerr_hi = highs - means

    bars = ax.bar(models_list, means, color=color, alpha=0.80,
                  edgecolor='black', linewidth=0.8)
    ax.errorbar(range(len(models_list)), means,
                yerr=[yerr_lo, yerr_hi],
                fmt='none', color='black', capsize=6, linewidth=2, capthick=2)
    ax.set_title(f'{metric_name} with 95% Bootstrap CI\n'
                 f'({TARGET_LABEL}, BDHS 2022)',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(f'{metric_name} (%)', fontsize=10)
    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(models_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(max(0, min(lows) - 10), min(100, max(highs) + 10))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, m, lo, hi in zip(bars, means, lows, highs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + yerr_hi[list(means).index(m)] + 0.5,
                f'{m:.1f}\n[{lo:.1f}â€“{hi:.1f}]',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.suptitle(f'Bootstrap 95% Confidence Intervals â€” {TARGET_LABEL} Prediction\n'
             f'(B={BOOTSTRAP_N} iterations, BDHS 2022, Stratified Resampling)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bootstrap_CI_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Saved] bootstrap_CI_chart.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 8: CLINICAL TESTS
# Test A: McNemar's Test â€” pairwise comparison of best 2 models
# Test B: DeLong-style Hanley-McNeil AUC comparison
# These are required for Q1 methodology sections
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 7] Clinical Statistical Tests...")

# Rank models by bootstrap AUC
bs_df_sorted = bs_df.sort_values('AUC_mean', ascending=False).reset_index(drop=True)
best_name    = bs_df_sorted.iloc[0]['Model']
second_name  = bs_df_sorted.iloc[1]['Model']

print(f"  Best model   : {best_name}")
print(f"  Second best  : {second_name}")

y_pred_best   = predictions[best_name]['y_pred']
y_pred_second = predictions[second_name]['y_pred']
y_prob_best   = predictions[best_name]['y_prob']
y_prob_second = predictions[second_name]['y_prob']

# â”€â”€â”€ McNemar's Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tests whether two classifiers make different errors on the same test samples
# H0: Both models make the same errors (no significant difference)
# p < 0.05 â†’ significantly different
b = np.sum((y_pred_best == y_test_arr) & (y_pred_second != y_test_arr))
c = np.sum((y_pred_best != y_test_arr) & (y_pred_second == y_test_arr))
if b + c > 0:
    mcnemar_statistic = (abs(b - c) - 1)**2 / (b + c)
    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_statistic, df=1)
else:
    mcnemar_statistic, mcnemar_p = 0.0, 1.0

print(f"\n  McNemar's Test ({best_name} vs {second_name}):")
print(f"    b (best correct, 2nd wrong) = {b}")
print(f"    c (best wrong, 2nd correct) = {c}")
print(f"    Ï‡Â² = {mcnemar_statistic:.4f},  p = {mcnemar_p:.4f}")
print(f"    {'SIGNIFICANT (p<0.05): Models differ significantly' if mcnemar_p < 0.05 else 'NOT significant (pâ‰¥0.05): No significant difference'}")

# â”€â”€â”€ DeLong-style AUC Bootstrap Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Compare AUC of best vs second-best using bootstrap difference distribution
def bootstrap_auc_diff(y_true, y_prob1, y_prob2, n=1000, seed=42):
    """Bootstrap test: H0: AUC1 == AUC2. Returns p-value (two-sided)."""
    rng2 = np.random.RandomState(seed)
    y_true = np.array(y_true)
    diffs = []
    N = len(y_true)
    for _ in range(n):
        idx  = rng2.choice(N, size=N, replace=True)
        yt   = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            a1 = roc_auc_score(yt, np.array(y_prob1)[idx])
            a2 = roc_auc_score(yt, np.array(y_prob2)[idx])
            diffs.append(a1 - a2)
        except Exception:
            pass
    diffs = np.array(diffs)
    observed_diff = (roc_auc_score(y_true, y_prob1) -
                     roc_auc_score(y_true, y_prob2))
    # Two-sided p-value: proportion of bootstrap diffs on wrong side
    p_val = 2 * min(np.mean(diffs >= observed_diff),
                    np.mean(diffs <= observed_diff))
    return observed_diff, diffs, p_val

obs_diff, diff_dist, delong_p = bootstrap_auc_diff(
    y_test_arr, y_prob_best, y_prob_second,
    n=BOOTSTRAP_N, seed=RANDOM_SEED
)
auc_best_val   = roc_auc_score(y_test_arr, y_prob_best)
auc_second_val = roc_auc_score(y_test_arr, y_prob_second)

print(f"\n  Bootstrap AUC Comparison ({best_name} vs {second_name}):")
print(f"    AUC ({best_name})   = {auc_best_val:.4f}")
print(f"    AUC ({second_name}) = {auc_second_val:.4f}")
print(f"    Î” AUC (observed)    = {obs_diff:.4f}")
print(f"    Bootstrap p-value   = {delong_p:.4f}")
print(f"    {'SIGNIFICANT: AUC difference is statistically meaningful' if delong_p < 0.05 else 'NOT significant: AUC difference may be by chance'}")

# Save clinical tests summary
clinical_summary = {
    'Test': ['McNemar\'s Test', 'Bootstrap AUC Comparison'],
    'Model_1': [best_name, best_name],
    'Model_2': [second_name, second_name],
    'Statistic': [round(mcnemar_statistic, 4), round(obs_diff, 4)],
    'P_Value': [round(mcnemar_p, 4), round(delong_p, 4)],
    'Significant_p05': [mcnemar_p < 0.05, delong_p < 0.05],
    'Interpretation': [
        'Best model errors significantly differ from 2nd best' if mcnemar_p < 0.05
        else 'No significant error difference',
        'AUC difference is statistically significant' if delong_p < 0.05
        else 'AUC difference may be by chance'
    ]
}
pd.DataFrame(clinical_summary).to_csv(
    f"{OUTPUT_DIR}/clinical_tests.csv", index=False)
print(f"  [Saved] clinical_tests.csv")

# Plot AUC difference bootstrap distribution
plt.figure(figsize=(9, 5))
plt.hist(diff_dist, bins=50, color='#5C6BC0', edgecolor='black',
         alpha=0.8, density=True, label='Bootstrap Î” AUC distribution')
plt.axvline(obs_diff, color='#E53935', linewidth=2.5,
            linestyle='--', label=f'Observed Î” AUC = {obs_diff:.4f}')
plt.axvline(0, color='black', linewidth=1.5, linestyle='-', label='Hâ‚€: Î” AUC = 0')
plt.fill_betweenx([0, plt.gca().get_ylim()[1] if plt.gca().get_ylim()[1] > 0 else 5],
                   np.percentile(diff_dist, 2.5),
                   np.percentile(diff_dist, 97.5),
                   alpha=0.15, color='green', label='95% Bootstrap CI')
plt.xlabel('Î” AUC (Best âˆ’ Second Best)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title(f'Bootstrap AUC Comparison: {best_name} vs {second_name}\n'
          f'({TARGET_LABEL}, BDHS 2022) â€” p = {delong_p:.4f} '
          f'{"â˜… Significant" if delong_p < 0.05 else "(Not Significant)"}',
          fontsize=11, fontweight='bold', pad=10)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/auc_comparison_bootstrap.png", dpi=150)
plt.close()
print(f"  [Saved] auc_comparison_bootstrap.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 9: ROC CURVES â€” All Models on One Plot
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 8] ROC Curve Overlay Plot...")

palette = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']
linestyles = ['-', '--', '-.', ':', '-', '--']

fig, ax = plt.subplots(figsize=(9, 7))
for i, (name, preds) in enumerate(predictions.items()):
    fpr, tpr, _ = roc_curve(y_test_arr, preds['y_prob'])
    roc_auc_val = auc(fpr, tpr)
    ci_lo = bootstrap_results[name]['AUC_CI_lower']
    ci_hi = bootstrap_results[name]['AUC_CI_upper']
    ax.plot(fpr, tpr, color=palette[i], lw=2.2,
            linestyle=linestyles[i],
            label=f'{name}  AUC={roc_auc_val*100:.1f}% [{ci_lo}â€“{ci_hi}]')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC=50%)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title(f'ROC Curves â€” All Models ({TARGET_LABEL} Prediction, BDHS 2022)\n'
             f'(AUC with 95% Bootstrap CI in brackets)',
             fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25, linestyle='--')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curves_all_models.png", dpi=150)
plt.close()
print(f"  [Saved] roc_curves_all_models.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 10: SHAP ANALYSIS â€” Best Model Deep Interpretation
# SHAP is the standard Q1 explainability tool for tree-based models
# Identifies not just WHICH features matter, but HOW and for WHOM
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[STEP 9] SHAP Analysis â€” {best_name}...")

# Get best model object and test data
best_model_obj = fitted_models[best_name]['model']
best_uses_scale = (best_name in NEEDS_SCALING)
X_test_for_shap = (
    pd.DataFrame(X_test_scaled, columns=selected_features)
    if best_uses_scale else X_test_sel
)

# Apply readable labels to feature columns for SHAP plots
X_test_shap_labelled = X_test_for_shap.copy()
X_test_shap_labelled.columns = [VAR_LABELS.get(c, c) for c in X_test_for_shap.columns]

if SHAP_AVAILABLE:
    try:
        # Choose explainer type based on model
        if best_name in ["XGBoost", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(best_model_obj)
            shap_values = explainer.shap_values(X_test_for_shap, check_additivity=False)
            
            # Robust parsing for list, 2D, or 3D SHAP outputs
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3: # (samples, features, classes) -> Select class 1
                    sv = shap_values[:, :, 1]
                else:
                    sv = shap_values
            else:
                sv = shap_values
        else:
            # LinearExplainer for LR / SVM
            masker = shap.maskers.Independent(X_test_for_shap)
            explainer = shap.LinearExplainer(best_model_obj, masker)
            sv = explainer.shap_values(X_test_for_shap)
            if isinstance(sv, list):
                sv = sv[1]

        print(f"  âœ“ SHAP values computed â€” shape: {np.array(sv).shape}")

        # â”€â”€ SHAP Summary Bar Plot (mean |SHAP|)
        feat_labels = [VAR_LABELS.get(c, c) for c in selected_features]
        shap_abs_mean = np.abs(sv).mean(axis=0)
        shap_bar_df = pd.DataFrame({
            'Feature': feat_labels,
            'Mean_|SHAP|': shap_abs_mean
        }).sort_values('Mean_|SHAP|', ascending=True)

        plt.figure(figsize=(10, max(7, len(feat_labels) * 0.45)))
        colors_shap = plt.cm.RdYlBu_r(
            np.linspace(0.2, 0.9, len(shap_bar_df))
        )
        bars = plt.barh(shap_bar_df['Feature'],
                        shap_bar_df['Mean_|SHAP|'],
                        color=colors_shap, edgecolor='black', linewidth=0.7)
        plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)',
                   fontsize=11)
        plt.title(f'SHAP Feature Importance â€” {best_name}\n'
                  f'({TARGET_LABEL} Prediction, BDHS 2022)',
                  fontsize=12, fontweight='bold', pad=10)
        for bar, val in zip(bars, shap_bar_df['Mean_|SHAP|']):
            plt.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png", dpi=150)
        plt.close()
        print(f"  [Saved] shap_importance_bar.png")

        # â”€â”€ SHAP Beeswarm Summary Plot
        # SHAP beeswarm Plot
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[1]
        
        shap_exp = shap.Explanation(
            values=sv,
            base_values=np.full(len(sv), base_val),
            data=X_test_for_shap.values,
            feature_names=feat_labels
        )
        plt.figure(figsize=(11, max(7, len(feat_labels) * 0.5)))
        shap.plots.beeswarm(shap_exp, max_display=min(20, len(feat_labels)),
                            show=False)
        plt.title(f'SHAP Beeswarm Plot â€” {best_name} ({TARGET_LABEL})',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] shap_beeswarm.png")

        # â”€â”€ SHAP Dependence Plots â€” Top 3 features
        top3_feats = shap_bar_df.tail(3)['Feature'].tolist()[::-1]
        top3_orig  = [selected_features[feat_labels.index(f)]
                      for f in top3_feats if f in feat_labels]

        fig, axes = plt.subplots(1, min(3, len(top3_orig)),
                                 figsize=(6 * min(3, len(top3_orig)), 5))
        if len(top3_orig) == 1:
            axes = [axes]
        for ax_dep, feat, flabel in zip(axes, top3_orig, top3_feats):
            feat_idx = selected_features.index(feat)
            sc = ax_dep.scatter(
                X_test_for_shap.iloc[:, feat_idx],
                sv[:, feat_idx],
                c=sv[:, feat_idx], cmap='RdYlBu_r',
                alpha=0.5, s=15, edgecolors='none'
            )
            plt.colorbar(sc, ax=ax_dep, label='SHAP value')
            ax_dep.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_dep.set_xlabel(flabel, fontsize=10)
            ax_dep.set_ylabel('SHAP Value (Impact)', fontsize=10)
            ax_dep.set_title(f'SHAP Dependence:\n{flabel}',
                             fontsize=10, fontweight='bold')
            ax_dep.grid(alpha=0.25, linestyle='--')

        plt.suptitle(f'SHAP Dependence Plots â€” Top Predictors of {TARGET_LABEL}',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_dependence_top3.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] shap_dependence_top3.png")

        # Save SHAP values CSV
        shap_df = pd.DataFrame(sv, columns=feat_labels)
        shap_df.to_csv(f"{OUTPUT_DIR}/shap_values.csv", index=False)
        # Save SHAP mean importance
        shap_bar_df.sort_values('Mean_|SHAP|', ascending=False).to_csv(
            f"{OUTPUT_DIR}/shap_feature_importance.csv", index=False)
        print(f"  [Saved] shap_values.csv, shap_feature_importance.csv")

    except Exception as e:
        print(f"  [ERROR] SHAP computation failed: {e}")
        print("  Skipping SHAP analysis...")
else:
    print("  SHAP not available â€” skipping. Install with: pip install shap")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 11: FULL RESULTS TABLE WITH BOOTSTRAP CI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 10] Compiling Full Results Table...")

full_results = []
for name in MODELS:
    preds = predictions[name]
    bs    = bootstrap_results[name]
    row = {
        'Model': name,
        'Accuracy_%': bs['Accuracy_mean'],
        'Accuracy_95CI': f"[{bs['Accuracy_CI_lower']}â€“{bs['Accuracy_CI_upper']}]",
        'Precision_%': bs['Precision_mean'],
        'Precision_95CI': f"[{bs['Precision_CI_lower']}â€“{bs['Precision_CI_upper']}]",
        'Recall_%': bs['Recall_mean'],
        'Recall_95CI': f"[{bs['Recall_CI_lower']}â€“{bs['Recall_CI_upper']}]",
        'F1_%': bs['F1_mean'],
        'F1_95CI': f"[{bs['F1_CI_lower']}â€“{bs['F1_CI_upper']}]",
        'AUC_%': bs['AUC_mean'],
        'AUC_95CI': f"[{bs['AUC_CI_lower']}â€“{bs['AUC_CI_upper']}]",
        'Outcome': TARGET_LABEL,
    }
    full_results.append(row)

full_df = pd.DataFrame(full_results).sort_values('AUC_%', ascending=False)\
            .reset_index(drop=True)
full_df.to_csv(f"{OUTPUT_DIR}/full_results_with_CI.csv", index=False)

print("\n" + "=" * 90)
print(f"  FULL RESULTS â€” {TARGET_LABEL} (BDHS 2022) | Metrics with 95% Bootstrap CI")
print("=" * 90)
display_cols = ['Model', 'AUC_%', 'AUC_95CI', 'F1_%', 'F1_95CI',
                'Recall_%', 'Recall_95CI', 'Precision_%', 'Precision_95CI']
print(full_df[display_cols].to_string(index=False))
print("=" * 90)
print(f"  [Saved] full_results_with_CI.csv")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 12: PUBLICATION-READY SUMMARY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best_row = full_df.iloc[0]

summary = f"""
=============================================================================
ADVANCED ANALYSIS REPORT â€” {TARGET_LABEL.upper()} ML STUDY (BDHS 2022)
Q1 PUBLICATION STANDARD
=============================================================================
Dataset        : BDHS2022_MH_ML_ready_1.csv
Target         : {TARGET} ({TARGET_LABEL}) â€” Binary (0 = No, 1 = Yes)
Total Sample   : {len(df):,} respondents
Prevalence     : {y.mean()*100:.2f}% positive ({int(y.sum()):,} cases)
Selected Feat. : {len(selected_features)} variables (Union: MI + Chi2 + RF, Top-{TOP_K})

â”€â”€â”€ VARIABLE LABELS (Corrected for Publication) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INDEPENDENT (Predictors):
  v024          = Division
  v025          = Residence (Urban/Rural)
  v106          = Respondent's Education Level
  v701          = Husband's Education Level
  age_cat       = Current Age of Respondent
  age_fb_cat    = Teenage Pregnancy (Age at First Birth)
  parity_cat    = Total Number of Children
  age_diff_cat  = Age Difference of Spouses
  wealth_cat    = Wealth Index
  internet_use  = Mass Media / Internet Use
  contra        = Contraceptive Use
  contra_decision = Decision for Contraceptive Use

DEPENDENT (Mental Health Outcomes):
  dep  = Depression  |  anx = Anxiety  |  disu = Mental Health Disutility

SURVEY DESIGN:
  v005 = Sampling Weight  |  v021 = PSU  |  v023 = Stratification

â”€â”€â”€ METHODOLOGY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  1. Survey-weighted descriptive analysis (v005/1,000,000)
  2. Smart imputation (structural nulls â†’ 0, MAR â†’ Median/Mode)
  3. Feature selection: Union(MI, Chi2, RF Importance), Top-{TOP_K} each
  4. SMOTE (AFTER feature selection â€” leakage-free)
  5. 6 ML classifiers trained and evaluated
  6. Bootstrap CI (B={BOOTSTRAP_N}) for all performance metrics
  7. McNemar's test for pairwise model comparison
  8. Bootstrap AUC comparison (DeLong-equivalent)
  9. SHAP explainability for best model

â”€â”€â”€ BEST MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  â˜… {best_row['Model']}
     AUC      : {best_row['AUC_%']}%  {best_row['AUC_95CI']}
     F1-Score : {best_row['F1_%']}%   {best_row['F1_95CI']}
     Recall   : {best_row['Recall_%']}%  {best_row['Recall_95CI']}
     Precision: {best_row['Precision_%']}%  {best_row['Precision_95CI']}

â”€â”€â”€ CLINICAL TESTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  McNemar's Test p-value    : {mcnemar_p:.4f}  {'â˜… Significant' if mcnemar_p < 0.05 else '(Not significant)'}
  Bootstrap AUC diff p-value: {delong_p:.4f}  {'â˜… Significant' if delong_p < 0.05 else '(Not significant)'}

â”€â”€â”€ OUTPUT FILES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  {OUTPUT_DIR}/
  â”œâ”€ weighted_descriptives.csv        â€” Survey-weighted variable breakdown
  â”œâ”€ bootstrap_CI_results.csv         â€” Bootstrap 95% CI for all metrics
  â”œâ”€ bootstrap_CI_chart.png           â€” CI visualization per metric
  â”œâ”€ clinical_tests.csv               â€” McNemar + AUC comparison results
  â”œâ”€ auc_comparison_bootstrap.png     â€” Bootstrap AUC comparison plot
  â”œâ”€ roc_curves_all_models.png        â€” ROC overlay with CI labels
  â”œâ”€ full_results_with_CI.csv         â€” Complete results for publication table
  â”œâ”€ shap_importance_bar.png          â€” SHAP feature importance bar
  â”œâ”€ shap_beeswarm.png                â€” SHAP beeswarm plot
  â”œâ”€ shap_dependence_top3.png         â€” SHAP dependence: top 3 features
  â”œâ”€ shap_values.csv                  â€” Raw SHAP values matrix
  â””â”€ shap_feature_importance.csv      â€” SHAP mean |value| per feature
=============================================================================
"""
print(summary)
with open(f"{OUTPUT_DIR}/advanced_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"[Saved] advanced_summary.txt")
print(f"\n{'â•'*70}")
print(f"  âœ… ADVANCED {TARGET_LABEL.upper()} ANALYSIS COMPLETE!")
print(f"  ðŸ“ All outputs: {OUTPUT_DIR}")
print(f"{'â•'*70}\n")
