"""
=============================================================================
BANGLADESH DEMOGRAPHIC AND HEALTH SURVEY (BDHS) 2022
ADVANCED ML ANALYSIS â€” ANXIETY (anx) as Dependent Variable
Q1 PUBLICATION STANDARD â€” Advanced Analysis Module
=============================================================================
Author     : Research Pipeline (Oxford-standard methodology)
Dataset    : BDHS2022_MH_ML_ready_1.csv
Target     : anx (Anxiety) â†’ Binary (0 = No, 1 = Yes)

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
  dep            = Depression
  anx            = Anxiety (THIS ANALYSIS)
  disu           = Mental Health Disutility / Disability

SURVEY DESIGN WEIGHTS (for weighted analysis):
  v005           = Sampling Weight
  v021           = Primary Sampling Unit (PSU)
  v023           = Stratification Number

ADVANCED ANALYSIS FEATURES:
  1. SHAP (SHapley Additive exPlanations) â€” explainability
  2. Bootstrap Confidence Intervals (B=1000) â€” metric uncertainty
  3. Clinical Tests: DeLong-style AUC comparison, McNemar's test
  4. Survey-weighted descriptive analysis
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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
from imblearn.over_sampling import SMOTE

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  [WARNING] SHAP not installed. Run: pip install shap")

print("=" * 70)
print("  BDHS 2022 â€” ADVANCED ANXIETY ANALYSIS (Q1 Standard)")
print("=" * 70)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIGURATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_PATH     = r"e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv"
OUTPUT_DIR    = r"e:\hafiza mam work\advanced_analysis\results_anx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET        = 'anx'
TARGET_LABEL  = 'Anxiety'
OTHER_TARGETS = ['dep', 'disu']
SURVEY_VARS   = ['v005', 'v021', 'v023']
BOOTSTRAP_N   = 1000
RANDOM_SEED   = 42

VAR_LABELS = {
    'v024'           : 'Division',
    'v025'           : 'Residence (Urban/Rural)',
    'v106'           : "Respondent's Education",
    'v701'           : "Husband's Education",
    'age_cat'        : "Respondent's Current Age",
    'age_fb_cat'     : 'Teenage Pregnancy (Age at 1st Birth)',
    'parity_cat'     : 'Total Number of Children',
    'age_diff_cat'   : 'Age Difference of Spouses',
    'wealth_cat'     : 'Wealth Index',
    'internet_use'   : 'Mass Media / Internet Use',
    'contra'         : 'Contraceptive Use',
    'contra_decision': 'Decision for Contraceptive Use',
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LOAD & PREPARE DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 1] Loading and Preparing Dataset...")

df_raw = pd.read_csv(DATA_PATH, dtype=str)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)
df = df_raw.apply(pd.to_numeric, errors='coerce')
if 'CASEID' in df.columns:
    df = df.drop(columns=['CASEID'])

df_imp = df.copy()
structural_cols = [
    'V212','V312','V511','V632','V701','V730',
    'MTH22','MTH24','V171A','age_dif','age_dif_cat',
    'internet_use','contra_decision3','parity_cat',
]
for col in structural_cols:
    if col in df_imp.columns and df_imp[col].isnull().sum() > 0:
        df_imp[col] = df_imp[col].fillna(0)
for col in [c for c in df_imp.columns if df_imp[c].isnull().any()]:
    uv = df_imp[col].dropna().nunique()
    df_imp[col] = df_imp[col].fillna(
        df_imp[col].mode().iloc[0] if uv <= 15 else df_imp[col].median())

if 'v005' in df_imp.columns:
    df_imp['wt'] = df_imp['v005'] / 1_000_000

y = df_imp[TARGET].astype(int)
DROP_COLS = OTHER_TARGETS + [TARGET, 'MTH22', 'MTH24']
X = df_imp.drop(columns=[c for c in DROP_COLS + SURVEY_VARS + ['wt']
                          if c in df_imp.columns])
X = X.dropna(axis=1, how='all')
print(f"  â†’ Shape: {len(df):,} rows, {X.shape[1]} features")
print(f"  â†’ {TARGET_LABEL} prevalence: {y.mean()*100:.2f}%")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SURVEY-WEIGHTED DESCRIPTIVE STATS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 2] Survey-Weighted Descriptive Analysis...")

if 'wt' in df_imp.columns:
    wt = df_imp['wt']
    weighted_desc = []
    for var in [v for v in VAR_LABELS if v in df_imp.columns]:
        label = VAR_LABELS[var]
        for cat_val in sorted(df_imp[var].dropna().unique()):
            mask = df_imp[var] == cat_val
            w_total   = wt[mask].sum()
            w_positive= (wt[mask] * (y[mask] == 1)).sum()
            weighted_desc.append({
                'Variable': label,
                'Category_Code': cat_val,
                'N_Unweighted': int(mask.sum()),
                'Weighted_N': round(w_total, 2),
                f'{TARGET_LABEL}_Weighted_%': round(
                    w_positive/w_total*100 if w_total > 0 else 0, 2)
            })
    pd.DataFrame(weighted_desc).to_csv(
        f"{OUTPUT_DIR}/weighted_descriptives.csv", index=False)
    print(f"  [Saved] weighted_descriptives.csv")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FEATURE SELECTION + SMOTE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 3] Feature Selection (Union: MI + Chi2 + RF)...")

class_ratio   = y.value_counts(normalize=True)
is_imbalanced = class_ratio.min() < 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
X_train_orig, y_train_orig = X_train.copy(), y_train.copy()

feature_names = list(X_train_orig.columns)
TOP_K = 15

mi_scores = mutual_info_classif(X_train_orig, y_train_orig, random_state=RANDOM_SEED)
mi_df  = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})\
           .sort_values('MI_Score', ascending=False).reset_index(drop=True)
mi_top = set(mi_df.head(TOP_K)['Feature'])

X_shifted = X_train_orig - X_train_orig.min() + 0.001
chi2_scores, chi2_pvals = chi2(X_shifted, y_train_orig)
chi2_df  = pd.DataFrame({'Feature': feature_names,
                          'Chi2_Score': chi2_scores, 'P_value': chi2_pvals})\
             .sort_values('Chi2_Score', ascending=False).reset_index(drop=True)
chi2_top = set(chi2_df.head(TOP_K)['Feature'])

rf_sel = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                 n_jobs=-1, class_weight='balanced')
rf_sel.fit(X_train_orig, y_train_orig)
rf_df  = pd.DataFrame({'Feature': feature_names,
                        'RF_Importance': rf_sel.feature_importances_})\
           .sort_values('RF_Importance', ascending=False).reset_index(drop=True)
rf_top = set(rf_df.head(TOP_K)['Feature'])

selected_features = sorted(mi_top | chi2_top | rf_top)
print(f"  â†’ {len(selected_features)} features selected (UNION)")

X_train_sel_orig = X_train_orig[selected_features]
X_test_sel       = X_test[selected_features]

if is_imbalanced:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_sel_bal, y_train_bal = smote.fit_resample(X_train_sel_orig, y_train_orig)
    print(f"  â†’ SMOTE: {sum(y_train_bal==0)} / {sum(y_train_bal==1)}")
else:
    X_train_sel_bal, y_train_bal = X_train_sel_orig.copy(), y_train_orig.copy()

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel_bal)
X_test_scaled  = scaler.transform(X_test_sel)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MODEL TRAINING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 4] Training 6 ML Models...")

MODELS = {
    "Logistic Regression": LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000,
        class_weight='balanced', solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_SEED, max_depth=8, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED,
        n_jobs=-1, class_weight='balanced'),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_SEED, eval_metric='logloss',
        scale_pos_weight=(sum(y_train_bal==0)/max(sum(y_train_bal==1),1)),
        verbosity=0),
    "SVM (Linear)": CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', random_state=RANDOM_SEED,
                  C=1.0, max_iter=2000),
        cv=3, method='sigmoid'),
    "KNN": KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='euclidean')
}

NEEDS_SCALING = {"Logistic Regression", "SVM (Linear)", "KNN"}
fitted_models  = {}
predictions    = {}

for name, model in MODELS.items():
    Xtr = X_train_scaled if name in NEEDS_SCALING else X_train_sel_bal.values
    Xte = X_test_scaled  if name in NEEDS_SCALING else X_test_sel.values
    model.fit(Xtr, y_train_bal)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    fitted_models[name] = {'model': model, 'Xtr': Xtr, 'Xte': Xte}
    predictions[name]   = {'y_pred': y_pred, 'y_prob': y_prob}
    print(f"  âœ“ {name}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# BOOTSTRAP CONFIDENCE INTERVALS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[STEP 5] Bootstrap CI (B={BOOTSTRAP_N})...")

rng = np.random.RandomState(RANDOM_SEED)
y_test_arr = np.array(y_test)

def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstrap=1000, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    n = len(y_true)
    boot = {k: [] for k in ['acc','prec','rec','f1','auc']}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt,yp,ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        boot['acc'].append(accuracy_score(yt, yp))
        boot['prec'].append(precision_score(yt, yp, zero_division=0))
        boot['rec'].append(recall_score(yt, yp, zero_division=0))
        boot['f1'].append(f1_score(yt, yp, zero_division=0))
        try: boot['auc'].append(roc_auc_score(yt, ypr))
        except: pass
    def ci(arr):
        arr = np.array(arr)
        return (round(np.mean(arr)*100,2),
                round(np.percentile(arr,2.5)*100,2),
                round(np.percentile(arr,97.5)*100,2))
    r = {}
    for metric, key in [('Accuracy','acc'),('Precision','prec'),
                        ('Recall','rec'),('F1','f1'),('AUC','auc')]:
        m,l,h = ci(boot[key])
        r[f'{metric}_mean'] = m
        r[f'{metric}_CI_lower'] = l
        r[f'{metric}_CI_upper'] = h
    return r

bootstrap_results = {}
for name in MODELS:
    preds = predictions[name]
    bs = bootstrap_metrics(y_test_arr, preds['y_pred'], preds['y_prob'],
                           n_bootstrap=BOOTSTRAP_N, rng=rng)
    bootstrap_results[name] = bs
    print(f"  âœ“ {name:25s}  AUC={bs['AUC_mean']}% "
          f"[{bs['AUC_CI_lower']}â€“{bs['AUC_CI_upper']}]")

bs_rows = [{'Model': n, **v} for n, v in bootstrap_results.items()]
bs_df   = pd.DataFrame(bs_rows)
bs_df.to_csv(f"{OUTPUT_DIR}/bootstrap_CI_results.csv", index=False)

# Bootstrap CI chart
metrics_to_vis = [
    ('AUC',       'AUC_mean',       'AUC_CI_lower',       'AUC_CI_upper',       '#E53935'),
    ('F1-Score',  'F1_mean',        'F1_CI_lower',         'F1_CI_upper',        '#1E88E5'),
    ('Recall',    'Recall_mean',    'Recall_CI_lower',     'Recall_CI_upper',     '#43A047'),
    ('Precision', 'Precision_mean', 'Precision_CI_lower',  'Precision_CI_upper',  '#FB8C00'),
]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes = axes.flatten()
for ax, (mn, mc, lc, hc, col) in zip(axes, metrics_to_vis):
    models_list = bs_df['Model'].values
    means = bs_df[mc].values; lows = bs_df[lc].values; highs = bs_df[hc].values
    bars = ax.bar(models_list, means, color=col, alpha=0.80, edgecolor='black', linewidth=0.8)
    ax.errorbar(range(len(models_list)), means,
                yerr=[means-lows, highs-means],
                fmt='none', color='black', capsize=6, linewidth=2, capthick=2)
    ax.set_title(f'{mn} with 95% Bootstrap CI\n({TARGET_LABEL}, BDHS 2022)',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(f'{mn} (%)', fontsize=10)
    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(models_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(max(0, min(lows)-10), min(100, max(highs)+10))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, m, lo, hi in zip(bars, means, lows, highs):
        ax.text(bar.get_x()+bar.get_width()/2,
                hi+0.5, f'{m:.1f}\n[{lo:.1f}â€“{hi:.1f}]',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
plt.suptitle(f'Bootstrap 95% CI â€” {TARGET_LABEL} Prediction (B={BOOTSTRAP_N}, BDHS 2022)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bootstrap_CI_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Saved] bootstrap_CI_chart.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLINICAL TESTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 6] Clinical Statistical Tests...")

bs_df_sorted = bs_df.sort_values('AUC_mean', ascending=False).reset_index(drop=True)
best_name    = bs_df_sorted.iloc[0]['Model']
second_name  = bs_df_sorted.iloc[1]['Model']

y_pred_best   = predictions[best_name]['y_pred']
y_pred_second = predictions[second_name]['y_pred']
y_prob_best   = predictions[best_name]['y_prob']
y_prob_second = predictions[second_name]['y_prob']

# McNemar's Test
b = np.sum((y_pred_best == y_test_arr) & (y_pred_second != y_test_arr))
c = np.sum((y_pred_best != y_test_arr) & (y_pred_second == y_test_arr))
if b + c > 0:
    mcnemar_stat = (abs(b-c)-1)**2 / (b+c)
    mcnemar_p    = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
else:
    mcnemar_stat, mcnemar_p = 0.0, 1.0

print(f"  McNemar p = {mcnemar_p:.4f}  "
      f"({'Significant' if mcnemar_p<0.05 else 'Not significant'})")

# Bootstrap AUC comparison
def bootstrap_auc_diff(y_true, y_prob1, y_prob2, n=1000, seed=42):
    rng2 = np.random.RandomState(seed)
    y_true = np.array(y_true)
    diffs = []
    N = len(y_true)
    for _ in range(n):
        idx = rng2.choice(N, size=N, replace=True)
        yt  = y_true[idx]
        if len(np.unique(yt)) < 2: continue
        try:
            a1 = roc_auc_score(yt, np.array(y_prob1)[idx])
            a2 = roc_auc_score(yt, np.array(y_prob2)[idx])
            diffs.append(a1-a2)
        except: pass
    diffs = np.array(diffs)
    obs_diff = (roc_auc_score(y_true, y_prob1) - roc_auc_score(y_true, y_prob2))
    p_val = 2*min(np.mean(diffs>=obs_diff), np.mean(diffs<=obs_diff))
    return obs_diff, diffs, p_val

obs_diff, diff_dist, delong_p = bootstrap_auc_diff(
    y_test_arr, y_prob_best, y_prob_second, n=BOOTSTRAP_N, seed=RANDOM_SEED)
print(f"  Bootstrap AUC diff p = {delong_p:.4f}  "
      f"({'Significant' if delong_p<0.05 else 'Not significant'})")

pd.DataFrame({
    'Test': ["McNemar's Test", 'Bootstrap AUC Comparison'],
    'Model_1': [best_name, best_name],
    'Model_2': [second_name, second_name],
    'Statistic': [round(mcnemar_stat,4), round(obs_diff,4)],
    'P_Value': [round(mcnemar_p,4), round(delong_p,4)],
    'Significant_p05': [mcnemar_p<0.05, delong_p<0.05],
}).to_csv(f"{OUTPUT_DIR}/clinical_tests.csv", index=False)
print(f"  [Saved] clinical_tests.csv")

# AUC diff plot
plt.figure(figsize=(9,5))
plt.hist(diff_dist, bins=50, color='#5C6BC0', edgecolor='black', alpha=0.8,
         density=True, label='Bootstrap Î” AUC')
plt.axvline(obs_diff, color='#E53935', lw=2.5, linestyle='--',
            label=f'Observed Î” AUC = {obs_diff:.4f}')
plt.axvline(0, color='black', lw=1.5, label='Hâ‚€: Î” AUC = 0')
plt.xlabel('Î” AUC (Best âˆ’ Second Best)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title(f'Bootstrap AUC Comparison: {best_name} vs {second_name}\n'
          f'({TARGET_LABEL}, BDHS 2022) â€” p = {delong_p:.4f}',
          fontsize=11, fontweight='bold', pad=10)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/auc_comparison_bootstrap.png", dpi=150)
plt.close()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ROC CURVES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 7] ROC Curve Overlay...")
palette    = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1']
linestyles = ['-','--','-.',':', '-','--']
fig, ax = plt.subplots(figsize=(9,7))
for i,(name,preds) in enumerate(predictions.items()):
    fpr, tpr, _ = roc_curve(y_test_arr, preds['y_prob'])
    roc_val = auc(fpr, tpr)
    bs = bootstrap_results[name]
    ax.plot(fpr, tpr, color=palette[i], lw=2.2, linestyle=linestyles[i],
            label=f"{name}  AUC={roc_val*100:.1f}% [{bs['AUC_CI_lower']}â€“{bs['AUC_CI_upper']}]")
ax.plot([0,1],[0,1],'k--',lw=1,label='Random (AUC=50%)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves â€” {TARGET_LABEL} Prediction (BDHS 2022)\nAUC with 95% Bootstrap CI',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curves_all_models.png", dpi=150)
plt.close()
print(f"  [Saved] roc_curves_all_models.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SHAP ANALYSIS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[STEP 8] SHAP Analysis â€” {best_name}...")
feat_labels = [VAR_LABELS.get(c,c) for c in selected_features]
best_uses_scale = (best_name in NEEDS_SCALING)
X_test_for_shap = (
    pd.DataFrame(X_test_scaled, columns=selected_features)
    if best_uses_scale else X_test_sel)
best_model_obj = fitted_models[best_name]['model']

if SHAP_AVAILABLE:
    try:
        if best_name in ["XGBoost","Random Forest","Decision Tree"]:
            explainer  = shap.TreeExplainer(best_model_obj)
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
            masker    = shap.maskers.Independent(X_test_for_shap)
            explainer = shap.LinearExplainer(best_model_obj, masker)
            sv_raw    = explainer.shap_values(X_test_for_shap)
            sv        = sv_raw[1] if isinstance(sv_raw, list) else sv_raw

        shap_abs_mean = np.abs(sv).mean(axis=0)
        shap_bar_df   = pd.DataFrame({'Feature': feat_labels,
                                       'Mean_|SHAP|': shap_abs_mean})\
                          .sort_values('Mean_|SHAP|', ascending=True)

        # SHAP bar
        plt.figure(figsize=(10, max(7, len(feat_labels)*0.45)))
        colors_shap = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(shap_bar_df)))
        bars = plt.barh(shap_bar_df['Feature'], shap_bar_df['Mean_|SHAP|'],
                        color=colors_shap, edgecolor='black', linewidth=0.7)
        plt.xlabel('Mean |SHAP Value|', fontsize=11)
        plt.title(f'SHAP Feature Importance â€” {best_name}\n({TARGET_LABEL}, BDHS 2022)',
                  fontsize=12, fontweight='bold')
        for bar, val in zip(bars, shap_bar_df['Mean_|SHAP|']):
            plt.text(val+0.0005, bar.get_y()+bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png", dpi=150)
        plt.close()

        # SHAP beeswarm
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[1]
        
        shap_exp = shap.Explanation(
            values=sv, base_values=np.full(len(sv), base_val),
            data=X_test_for_shap.values, feature_names=feat_labels)
        plt.figure(figsize=(11, max(7, len(feat_labels)*0.5)))
        shap.plots.beeswarm(shap_exp, max_display=min(20, len(feat_labels)), show=False)
        plt.title(f'SHAP Beeswarm â€” {best_name} ({TARGET_LABEL})',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
        plt.close()

        # SHAP dependence top 3
        top3_feats = shap_bar_df.tail(3)['Feature'].tolist()[::-1]
        top3_orig  = [selected_features[feat_labels.index(f)]
                      for f in top3_feats if f in feat_labels]
        fig, axes_dep = plt.subplots(1, min(3, len(top3_orig)),
                                     figsize=(6*min(3,len(top3_orig)), 5))
        if len(top3_orig) == 1: axes_dep = [axes_dep]
        for ax_d, feat, flabel in zip(axes_dep, top3_orig, top3_feats):
            fidx = selected_features.index(feat)
            sc = ax_d.scatter(X_test_for_shap.iloc[:,fidx], sv[:,fidx],
                              c=sv[:,fidx], cmap='RdYlBu_r', alpha=0.5, s=15)
            plt.colorbar(sc, ax=ax_d, label='SHAP value')
            ax_d.axhline(0, color='black', lw=0.8, linestyle='--')
            ax_d.set_xlabel(flabel, fontsize=10)
            ax_d.set_ylabel('SHAP Value', fontsize=10)
            ax_d.set_title(f'Dependence: {flabel}', fontsize=10, fontweight='bold')
        plt.suptitle(f'SHAP Dependence â€” Top {TARGET_LABEL} Predictors (BDHS 2022)',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_dependence_top3.png", dpi=150, bbox_inches='tight')
        plt.close()

        pd.DataFrame(sv, columns=feat_labels).to_csv(
            f"{OUTPUT_DIR}/shap_values.csv", index=False)
        shap_bar_df.sort_values('Mean_|SHAP|', ascending=False).to_csv(
            f"{OUTPUT_DIR}/shap_feature_importance.csv", index=False)
        print(f"  [Saved] SHAP plots and data files")
    except Exception as e:
        print(f"  [ERROR] SHAP: {e}")
else:
    print("  SHAP not available â€” install with: pip install shap")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FULL RESULTS TABLE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 9] Saving Full Results...")
full_results = []
for name in MODELS:
    bs = bootstrap_results[name]
    full_results.append({
        'Model': name,
        'AUC_%': bs['AUC_mean'],
        'AUC_95CI': f"[{bs['AUC_CI_lower']}â€“{bs['AUC_CI_upper']}]",
        'F1_%': bs['F1_mean'],
        'F1_95CI': f"[{bs['F1_CI_lower']}â€“{bs['F1_CI_upper']}]",
        'Recall_%': bs['Recall_mean'],
        'Recall_95CI': f"[{bs['Recall_CI_lower']}â€“{bs['Recall_CI_upper']}]",
        'Precision_%': bs['Precision_mean'],
        'Precision_95CI': f"[{bs['Precision_CI_lower']}â€“{bs['Precision_CI_upper']}]",
        'Outcome': TARGET_LABEL,
    })
full_df = pd.DataFrame(full_results).sort_values('AUC_%', ascending=False)\
            .reset_index(drop=True)
full_df.to_csv(f"{OUTPUT_DIR}/full_results_with_CI.csv", index=False)

best_row = full_df.iloc[0]
print("\n" + "="*80)
print(f"  BEST MODEL â€” {TARGET_LABEL}: {best_row['Model']}")
print(f"  AUC:  {best_row['AUC_%']}%  {best_row['AUC_95CI']}")
print(f"  F1:   {best_row['F1_%']}%   {best_row['F1_95CI']}")
print(f"  McNemar p = {mcnemar_p:.4f}  |  Bootstrap AUC diff p = {delong_p:.4f}")
print("="*80)

print(f"\n{'â•'*70}")
print(f"  âœ… ADVANCED {TARGET_LABEL.upper()} ANALYSIS COMPLETE!")
print(f"  ðŸ“ All outputs: {OUTPUT_DIR}")
print(f"{'â•'*70}\n")
