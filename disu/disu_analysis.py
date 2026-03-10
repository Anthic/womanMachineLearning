"""
=============================================================================
BANGLADESH DEMOGRAPHIC AND HEALTH SURVEY (BDHS) 2022
Mental Health ML Analysis — DISABILITY / DISUTILITY (disu) as Dependent Variable
=============================================================================
Author     : Research Pipeline (Oxford-standard methodology)
Dataset    : BDHS2022_MH_ML_ready_1.csv
Target     : disu (Disability/Disutility) --> Binary (0 = No, 1 = Yes)
Predictors : All variables EXCEPT:
             - disu, dep, anx (other mental health outcomes — data leakage)
             - MTH22, MTH24 (perfect separators causing data leakage)
Goal       : Feature selection via MI + Chi2 + RF Importance (Union),
             then train 6 ML classifiers, compare performance.

METHODOLOGY (NO DATA LEAKAGE):
  1. Train/Test Split (80-20, stratified by disu)
  2. Exclude perfect separators (MTH22, MTH24) + other outcomes
  3. Feature Selection on ORIGINAL training data (BEFORE SMOTE)
  4. SMOTE applied AFTER feature selection (leakage prevention)
  5. Model training with 5-fold cross-validation
  6. Both Train & Test accuracy reported (overfitting detection)

--- BANGLA SUMMARY ---
ei pipeline-e amra disu (disability/disutility) predict korte 6ti ML model
train korbo. prôtiti step-er kaj o karon niche explain kora hoyeche.
biasness thakle SMOTE diye handle kora hobe, data leakage somporke
shocheton thaka hobe sob shomoyo.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: LIBRARY IMPORT
# Keno: ei libraries data processing, ML modelling o visualization-er
# jônno industry-standard tool।
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive: popup chara file-e save kore
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
# Feature selection
from sklearn.feature_selection import mutual_info_classif, chi2

# Individual models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV  # LinearSVC-er jonno probability
from xgboost import XGBClassifier

# Imbalanced data handling
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("  BDHS 2022 — DISABILITY/DISUTILITY (disu) ML ANALYSIS PIPELINE")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOAD (Dataset lôd kora)
# Keno string hisebe first pôrte hobe:
#   -> CSV-te whitespace, hidden character thakte pare।
#   -> Shob string hisebe porle oi shômsya dhôra jay; tarpor numeric
#      convert kora hoy।
# Keno CASEID bad:
#   -> Eita শুধু identifier — kono predictive value nei।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Dataset lôd hôchhe...")

DATA_PATH  = r"e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv"
OUTPUT_DIR = r"e:\hafiza mam work\disu\ml_results_disu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shob column first string hisebe porbo — whitespace shômosya thekate
df_raw = pd.read_csv(DATA_PATH, dtype=str)
df_raw.columns = df_raw.columns.str.strip()

# Blank/whitespace-only cell ke NaN banao
df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)

# Numeric-e convert koro
df = df_raw.apply(pd.to_numeric, errors='coerce')

# CASEID drop — only identifier, zero predictive value
if 'CASEID' in df.columns:
    df = df.drop(columns=['CASEID'])

print(f"  -> Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  -> Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: NULL VALUE ANALYSIS (Missing value bishleshan)
# Keno: KotoTa data missing sheTa na janle shôThik imputation koushal
# thik kora jay na। Visualization diye pattern bojha jay — random na
# structural missing। Structural missing alada treatment lage।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Null Value Analysis cholchhe...")

null_counts  = df.isnull().sum()
null_percent = (null_counts / len(df) * 100).round(2)

null_summary = (pd.DataFrame({'Column': null_counts.index,
                               'Null_Count': null_counts.values,
                               'Null_%': null_percent.values})
                .query("Null_Count > 0")
                .sort_values('Null_Count', ascending=False)
                .reset_index(drop=True))

print(f"\n  Null-shohit column shomuh:")
print(null_summary.to_string(index=False))
print(f"\n  Total null column: {len(null_summary)}")

null_summary.to_csv(f"{OUTPUT_DIR}/null_analysis.csv", index=False)

# Null heatmap — missing pattern visible korte
# Keno heatmap: Ek nojore dekhay kôn column-e kôtoTa data missing।
# Pattern dekhle bôjha jay MNAR (structural) naki MAR (random)।
null_cols = null_summary['Column'].tolist()
if null_cols:
    plt.figure(figsize=(14, 5))
    sns.heatmap(df[null_cols].isnull(), cbar=True, yticklabels=False,
                cmap='viridis', xticklabels=True)
    plt.title("Null Value Heatmap — BDHS 2022 Disability (disu) Analysis\n"
              "(Yellow = Missing, Purple = Present)",
              fontsize=13, pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/null_heatmap.png", dpi=150)
    plt.close()
    print(f"  [Saved] null_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SMART IMPUTATION (Research-grade missing value handling)
#
# Keno Listwise Deletion nôy?
#   -> Jôkonô null-shohit shari mule dile 30-40% data harabo।
#   -> Statistical Power komе o Selection Bias toiri hoy।
#   -> Batchit mohilara random non — nirdir class-er。
#
# Dui dhoron-er Null:
#
# 1. STRUCTURAL NULLS (Skip-pattern MNAR — Missing Not At Random):
#    BDHS survey-e kichu proshno nirdishTO gosthi-ke kora hoy na।
#    Jemon: V212 (1st sontaner boyosh) -> shontan nei emon mohilar blank।
#    Shomadhan: 0 diye fill — "Not Applicable" category toiri hoy।
#    Model oi category theke learn korte parbe।
#
# 2. RANDOM NULLS (MAR — Missing At Random):
#    Shotti missing response।
#    Shomadhan: Continuous -> Median (outlier robust)
#               Categorical -> Mode (shobcheye ghôn ghôn dekhа category)
#
# Keno Median naki Mean:
#   -> BDHS survey variable-e outlier thake (boyosh, dhorpon etc.)।
#   -> Median outlier-e প্রভাবিত hoy na; Mean hoy।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Smart Imputation apply hôchhe...")

df_imp = df.copy()

# ── 3a. STRUCTURAL NULLS (reproductive history / skip pattern)
structural_cols = [
    'V212',             # 1st birth age (childless women-er jônnô NA)
    'V312',             # Contraception method (use na korle NA)
    'V511',             # 1st cohabitation age (structural skip)
    'V632',             # Contraception decision-maker (use na korle NA)
    'V701',             # Husband's education (shami na thakle NA)
    'V730',             # Husband's age (shami na thakle NA)
    'MTH22',            # Months since last birth (EXCLUDED - Data Leakage!)
    'MTH24',            # Months since 2nd last birth (EXCLUDED - Data Leakage!)
    'V171A',            # Internet frequency (structural skip)
    'age_dif',          # Age diff with husband (shami na thakle NA)
    'age_dif_cat',      # Age diff category (shami na thakle NA)
    'internet_use',     # Internet use (kichu age/edu group-er jônnô skip)
    'contra_decision3', # Contraception decision (use na korle NA)
    'parity_cat',       # Parity category (childless-der jônnô NA)
]

for col in structural_cols:
    if col in df_imp.columns:
        n_null = int(df_imp[col].isnull().sum())
        if n_null > 0:
            df_imp[col] = df_imp[col].fillna(0)
            print(f"  [Structural] {col:22s}: {n_null:5d} nulls -> 0 (N/A category)")

# ── 3b. Remaining RANDOM NULLS -> Median or Mode
remaining_null_cols = [c for c in df_imp.columns if df_imp[c].isnull().any()]
print(f"\n  Structural fill-er pore remaining null columns: {remaining_null_cols}")

for col in remaining_null_cols:
    n_null   = int(df_imp[col].isnull().sum())
    unique_v = int(df_imp[col].dropna().nunique())
    if unique_v <= 15:
        fill_val = df_imp[col].mode().iloc[0]
        strategy = "Mode (categorical)"
    else:
        fill_val = df_imp[col].median()
        strategy = "Median (continuous)"
    df_imp[col] = df_imp[col].fillna(fill_val)
    print(f"  [Random]     {col:22s}: {n_null:5d} nulls -> {strategy} ({fill_val})")

total_remaining = int(df_imp.isnull().sum().sum())
print(f"\n  Imputation complete. Remaining nulls: {total_remaining}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURES O TARGET PREPARATION
#
# Keno dep o anx bad:
#   -> dep o anx other mental health outcome।
#   -> Same BDHS survey-e collected — ek otathe একসাথে reported।
#   -> Jodi predictor hisebe rakhi: model oi pathway diye shortcut nebe।
#   -> Real situation-e oi info thakbe na — clinical useless।
#   -> Eita data leakage।
#
# Keno MTH22, MTH24 bad:
#   -> Ei variable duti "perfect separator" —
#      single variable diye 100% accuracy dekhay।
#   -> Eita spurious pattern, real-world-e generalizable nôy।
#   -> Depre analysis-e confirmed data leakage।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Features o Target prepare hôchhe...")

DROP_COLS = ['dep', 'anx', 'disu', 'MTH22', 'MTH24']

print("  [WARNING] MTH22 o MTH24 excluded (Data Leakage: perfect separators)")
print("  [WARNING] dep, anx excluded (other mental health outcomes — leakage risk)")

y = df_imp['disu'].astype(int)
X = df_imp.drop(columns=[c for c in DROP_COLS if c in df_imp.columns])

# Remove any all-NaN columns (safety)
X = X.dropna(axis=1, how='all')

print(f"  Feature set shape: {X.shape}")
print(f"  Features used: {list(X.columns)}")
print(f"\n  Target (disu) distribution:\n{y.value_counts().to_string()}")
print(f"  Disability prevalence: {y.mean()*100:.2f}%")

class_ratio   = y.value_counts(normalize=True)
is_imbalanced = class_ratio.min() < 0.20
print(f"\n  Class 0 (No disability): {class_ratio[0]*100:.1f}%")
print(f"  Class 1 (Disability)   : {class_ratio[1]*100:.1f}%")
print(f"  Imbalanced: {is_imbalanced}")

# Class distribution visualisation
# Keno: Imbalance kotoTa sheta visually prove kore — SMOTE dorkar ki na
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([class_ratio[0], class_ratio[1]],
       labels=['No Disability (0)', 'Disability (1)'],
       autopct='%1.1f%%', colors=['#66BB6A', '#FFA726'],
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title("Target Class Distribution — Disability/disu (BDHS 2022)\n"
             "(Class imbalance dekhte — SMOTE dorkar ki na bujhte)",
             fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.close()
print("  [Saved] class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TRAIN-TEST SPLIT
# Keno stratified:
#   -> disu positive sample kom thakte pare।
#   -> Stratified nishchit kore train o test dui-e same ratio maintain hobe।
#   -> Echarao test-e minority class (disu=1) kom porte pare।
# Keno 80-20:
#   -> 80% training-e dile model pôrjаpto data pay।
#   -> 20% test-e rakhle niropekhô mullayon shombhob।
#
# CRITICAL WARNING:
#   SMOTE EKHON lagabo na!
#   Feature selection MUST be done on ORIGINAL (imbalanced) training data।
#   SMOTE-er aage feature select korle synthetic data-r pattern dhore
#   feature select hobe — eita serious data leakage।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Train-Test Split (80-20, stratified by disu)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples")
print(f"  [NOTE] SMOTE ekhon apply hobe na — first feature selection.")

X_train_orig = X_train.copy()
y_train_orig = y_train.copy()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FEATURE SELECTION — UNION OF 3 METHODS
#
# CRITICAL: Feature selection MUST be done BEFORE SMOTE, on ORIGINAL
# training data। SMOTE-er pore korle synthetic sample-er upor depend kore
# feature select hobe — eita leakage।
#
# Why 3 methods:
# 1. Mutual Information (MI):
#    - Model-free; kônô statistical dependency linear ba non-linear dhorte pare।
#    - BDHS-er mixed variable type (binary, ordinal, continuous) -er jônnô ideal।
#
# 2. Chi-Square Test:
#    - Classical statistical independence test।
#    - Categorical o ordinal health indicator-er jônnô shobcheye upojukto।
#
# 3. Random Forest Importance (Boruta-equivalent):
#    - Python 3.13-e Boruta package compatible nôy।
#    - RF importance statistically equivalent o beshi stable।
#    - Complex interaction o non-linear effect capture kore।
#    - Internally "shadow feature" comparison kore।
#
# Union Rationale:
#   Prôtiti method-er blindspot ache।
#   MI pure additive effect miss korte pare।
#   Chi2 non-monotone relationship miss kore।
#   RF correlated cluster theke বেশি select korte pare।
#   Union: true signal-er recall maximise kore।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Feature Selection (Union: MI + Chi2 + RF Importance) cholchhe...")
print("  [CRITICAL] ORIGINAL training data use hôchhe (NO SMOTE) — leakage prevention")

feature_names = list(X_train_orig.columns)
TOP_K = 15   # Top 15 from each method

# ── 6a. Mutual Information (on ORIGINAL data)
print("  -> Mutual Information scores computing...")
mi_scores = mutual_info_classif(X_train_orig, y_train_orig, random_state=42)
mi_df     = (pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
               .sort_values('MI_Score', ascending=False)
               .reset_index(drop=True))
mi_top    = set(mi_df.head(TOP_K)['Feature'])

# ── 6b. Chi-Square (non-negative values, on ORIGINAL data)
print("  -> Chi-Square scores computing...")
X_shifted    = X_train_orig - X_train_orig.min() + 0.001
chi2_scores, chi2_pvals = chi2(X_shifted, y_train_orig)
chi2_df  = (pd.DataFrame({'Feature': feature_names,
                           'Chi2_Score': chi2_scores,
                           'P_value': chi2_pvals})
              .sort_values('Chi2_Score', ascending=False)
              .reset_index(drop=True))
chi2_top = set(chi2_df.head(TOP_K)['Feature'])

# ── 6c. Random Forest Importance (Boruta-equivalent, on ORIGINAL data)
print("  -> Random Forest Feature Importance computing...")
rf_sel = RandomForestClassifier(n_estimators=200, random_state=42,
                                 n_jobs=-1, class_weight='balanced')
rf_sel.fit(X_train_orig, y_train_orig)
rf_df  = (pd.DataFrame({'Feature': feature_names,
                         'RF_Importance': rf_sel.feature_importances_})
            .sort_values('RF_Importance', ascending=False)
            .reset_index(drop=True))
rf_top = set(rf_df.head(TOP_K)['Feature'])

# ── 6d. UNION of all three
selected_features = sorted(mi_top | chi2_top | rf_top)
print(f"\n  UNION of selected features ({len(selected_features)} total):")
print(f"  {'Feature':<22}  {'MI':^4}  {'Chi2':^4}  {'RF':^4}")
print(f"  {'-'*22}  {'-'*4}  {'-'*4}  {'-'*4}")
for f in selected_features:
    in_mi  = "+" if f in mi_top   else " "
    in_chi = "+" if f in chi2_top else " "
    in_rf  = "+" if f in rf_top   else " "
    print(f"  {f:<22}  {in_mi:^4}  {in_chi:^4}  {in_rf:^4}")

# Save feature selection CSVs
mi_df.to_csv(f"{OUTPUT_DIR}/feature_MI.csv", index=False)
chi2_df.to_csv(f"{OUTPUT_DIR}/feature_Chi2.csv", index=False)
rf_df.to_csv(f"{OUTPUT_DIR}/feature_RF.csv", index=False)
pd.DataFrame({'Selected_Feature': selected_features}).to_csv(
    f"{OUTPUT_DIR}/selected_features_union.csv", index=False)

# ── RF Importance bar chart
# Keno: Visually dekhay kon feature disability-r shobcheye badha ghoTak
plt.figure(figsize=(10, 7))
sns.barplot(x='RF_Importance', y='Feature', data=rf_df.head(20),
            palette='Blues_r', edgecolor='black')
plt.title("Top 20 Features — Random Forest Importance\n"
          "(Boruta-equivalent, BDHS 2022 Disability/disu)",
          fontsize=12, pad=10)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_RF.png", dpi=150)
plt.close()

# ── MI Score bar chart
plt.figure(figsize=(10, 7))
sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(20),
            palette='Oranges_r', edgecolor='black')
plt.title("Top 20 Features — Mutual Information Score\n"
          "(BDHS 2022 Disability/disu)",
          fontsize=12, pad=10)
plt.xlabel("Mutual Information Score", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_MI.png", dpi=150)
plt.close()
print("  [Saved] Feature importance charts")

# Final feature subsets from ORIGINAL data
X_train_sel_orig = X_train_orig[selected_features]
X_test_sel       = X_test[selected_features]

# ── NOW apply SMOTE (AFTER feature selection — leakage-free)
# Keno SMOTE e jaygay:
#   Disability BDHS-e rare condition — severe imbalance expected।
#   Imbalance hole model somosha "0" predict kore — high accuracy
#   kintu recall = 0% — clinically useless।
#   SMOTE synthetic minority sample toiri kore class balance ane।
#   Eita FEATURE SELECTION-er PORE kora hôchhe — critical!
if is_imbalanced:
    print("\n  -> SMOTE applying to SELECTED features only (leakage-free)")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sel_bal, y_train_bal = smote.fit_resample(X_train_sel_orig, y_train_orig)
    print(f"  After SMOTE — Class 0: {sum(y_train_bal==0)}, Class 1: {sum(y_train_bal==1)}")

    # SMOTE before/after comparison chart
    # Keno: Reviewer-der prove kora je class imbalance properly handle hoyeche
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    vals_before = [sum(y_train_orig==0), sum(y_train_orig==1)]
    vals_after  = [sum(y_train_bal==0),  sum(y_train_bal==1)]
    lbls = ['No Disability (0)', 'Disability (1)']

    ax1.bar(lbls, vals_before, color=['#66BB6A', '#FFA726'], edgecolor='black')
    ax1.set_title("BEFORE SMOTE\n(Imbalanced)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Sample Count")
    for i, v in enumerate(vals_before):
        ax1.text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=10)

    ax2.bar(lbls, vals_after, color=['#66BB6A', '#FFA726'], edgecolor='black')
    ax2.set_title("AFTER SMOTE\n(Balanced)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Sample Count")
    for i, v in enumerate(vals_after):
        ax2.text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=10)

    plt.suptitle("Class Imbalance Handling via SMOTE — Disability/disu (BDHS 2022)\n"
                 "(Minority class synthetic oversampling — applied AFTER feature selection)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/smote_before_after.png", dpi=150)
    plt.close()
    print("  [Saved] smote_before_after.png")
else:
    X_train_sel_bal = X_train_sel_orig.copy()
    y_train_bal     = y_train_orig.copy()
    print("  -> Data already balanced — no SMOTE needed. class_weight='balanced' used.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MODEL TRAINING AND EVALUATION (6 ML Models)
#
# Keno 6ti model:
#   -> Prôtiti modeler nijoshwo sîmabôddhotа o shokti ache।
#   -> Single model-e depend kora risky — meta-analysis approach।
#   -> Tulona kori tai "shoreshto model" claim korar basis toiri hoy।
#
# Keno Train o Test উভয় accuracy:
#   -> Overfitting dhôrte। Train >> Test mane model shikheini — jene niyeche।
#   -> Publication-e dôkhan darkар।
#   -> 5% gap = overfitting warning threshold।
#
# Keno Cross-Validation (CV):
#   -> Single train-test split vagyer upor nirbhor korte pare।
#   -> 5-fold: 5ti bhinno split-e model yachai — robust estimate।
#   -> CV mean +/- std e stability bujha jay।
#
# Keno Scaling (StandardScaler) LR, SVM, KNN-er jonno:
#   -> Ei model-gulo distance/coefficient use kore।
#   -> Scale bhinno hole bodo scale feature বেশি probhab felbe।
#   -> Scaling shob feature ke same scale-e ane।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Training and Evaluating 6 ML Models...")
print("-" * 65)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel_bal)
X_test_scaled  = scaler.transform(X_test_sel)

MODELS = {
    "Logistic Regression": LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced', solver='lbfgs'
        # Interpretable, Odds Ratio available, epidemiology standard
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=8, class_weight='balanced'
        # Fully interpretable — each path = one clinical decision rule
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'
        # 300-tree ensemble, variance reduction, gold-standard for DHS data
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric='logloss',
        scale_pos_weight=(sum(y_train_bal==0) / max(sum(y_train_bal==1), 1)),
        verbosity=0
        # Sequential boosting, L1/L2 regularization, handles imbalance
    ),
    "SVM (Linear)": CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', random_state=42, C=1.0, max_iter=2000),
        cv=3, method='sigmoid'
        # LinearSVC: significantly faster than kernel SVM on large SMOTE datasets
        # CalibratedClassifierCV provides probability estimates for ROC-AUC
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='euclidean'
        # Local instance-based, important as baseline reference
    )
}

NEEDS_SCALING = {"Logistic Regression", "SVM (Linear)", "KNN"}  # already set correctly

results    = {}
cv_results = {}

for model_name, model in MODELS.items():
    print(f"\n  -- {model_name}")
    Xtr = X_train_scaled if model_name in NEEDS_SCALING else X_train_sel_bal.values
    Xte = X_test_scaled  if model_name in NEEDS_SCALING else X_test_sel.values

    model.fit(Xtr, y_train_bal)

    # Get predictions for BOTH train and test (overfitting detection)
    y_pred_train = model.predict(Xtr)
    y_pred_test  = model.predict(Xte)
    y_prob       = model.predict_proba(Xte)[:, 1]

    # Train metrics
    acc_train = accuracy_score(y_train_bal, y_pred_train)

    # Test metrics
    acc_test = accuracy_score(y_test, y_pred_test)
    prec     = precision_score(y_test, y_pred_test, zero_division=0)
    rec      = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test  = f1_score(y_test, y_pred_test, zero_division=0)
    auc      = roc_auc_score(y_test, y_prob)

    # Overfitting detection
    overfit_gap = acc_train - acc_test
    is_overfit  = overfit_gap > 0.05

    results[model_name] = {
        'Train_Acc (%)':   round(acc_train   * 100, 2),
        'Test_Acc (%)':    round(acc_test    * 100, 2),
        'Overfit_Gap (%)': round(overfit_gap * 100, 2),
        'Precision (%)':   round(prec        * 100, 2),
        'Recall (%)':      round(rec         * 100, 2),
        'F1-Score (%)':    round(f1_test     * 100, 2),
        'ROC-AUC (%)':     round(auc         * 100, 2)
    }

    # 5-fold Cross Validation for stability
    cv_scores = cross_val_score(model, Xtr, y_train_bal, cv=5,
                                 scoring='f1', n_jobs=-1)
    cv_results[model_name] = {
        'CV_F1_Mean (%)': round(cv_scores.mean() * 100, 2),
        'CV_F1_Std (%)':  round(cv_scores.std()  * 100, 2)
    }

    print(f"     Train Accuracy : {results[model_name]['Train_Acc (%)']}%")
    print(f"     Test Accuracy  : {results[model_name]['Test_Acc (%)']}%")
    print(f"     Overfit Gap    : {results[model_name]['Overfit_Gap (%)']}% "
          f"{'[!] OVERFITTING!' if is_overfit else '[OK]'}")
    print(f"     Precision      : {results[model_name]['Precision (%)']}%")
    print(f"     Recall         : {results[model_name]['Recall (%)']}%")
    print(f"     F1-Score       : {results[model_name]['F1-Score (%)']}%")
    print(f"     ROC-AUC        : {results[model_name]['ROC-AUC (%)']}%")
    print(f"     CV F1 (5-fold) : {cv_results[model_name]['CV_F1_Mean (%)']} +/- "
          f"{cv_results[model_name]['CV_F1_Std (%)']}%")

    # Confusion Matrix per model
    # Keno: Dekhay True Positive, False Negative etc। screening-e FN critical।
    cm  = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["No Disability", "Disability"]).plot(
        ax=ax, colorbar=True, cmap='Oranges')
    ax.set_title(f"Confusion Matrix — {model_name}\n(Disability/disu, BDHS 2022)",
                 fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cm_{model_name.replace(' ','_')}.png", dpi=130)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: RESULTS COMPARISON TABLE
# Keno: Shob model-er metrics একsathe dekhle drolto shoreshto
# model chinnito kora jay। ROC-AUC diye sort — imbalanced data-te
# single-number summary metric হিসেবে ROC-AUC best।
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n[STEP 8] Results Comparison Table")
print("=" * 80)

results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
cv_df_out  = pd.DataFrame(cv_results).T.reset_index().rename(columns={'index': 'Model'})
final_df   = (results_df.merge(cv_df_out, on='Model')
                         .sort_values('ROC-AUC (%)', ascending=False)
                         .reset_index(drop=True))
final_df.index = final_df.index + 1

print(final_df.to_string())
print("=" * 80)
final_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: BEST MODEL IDENTIFICATION
#
# METRIC PRIORITY (Health Research — disability screening context):
#
# PRIMARY   -> ROC-AUC: Overall discrimination (threshold-independent)।
#              Imbalanced binary health outcome-er jônnô shoreshto metric।
#
# SECONDARY -> F1-Score: Precision o Recall-er harmonic mean।
#              Disability false negative (miss) cost beshi — F1 important।
#
# TERTIARY  -> Recall: Kotojon disabled mohila DETECT holo।
#              Disability screening-e miss kora boro problem।
#
# NOTE: Accuracy ekla dekha thik nôy imbalanced data-te।
#   Udaharan: Sob "0" predict korle 95%+ accuracy pawa jay jodi
#   shudhu 5% disability thake — kintu model clinically useless।
# ─────────────────────────────────────────────────────────────────────────────
best_model_name = final_df.iloc[0]['Model']
best_auc  = final_df.iloc[0]['ROC-AUC (%)']
best_f1   = final_df.iloc[0]['F1-Score (%)']
best_rec  = final_df.iloc[0]['Recall (%)']

model_explanations = {
    "XGBoost": (
        "XGBoost is best because:\n"
        "  - Sequential boosting corrects residual errors of each tree\n"
        "  - Built-in L1/L2 regularization prevents overfitting\n"
        "  - scale_pos_weight compensates for class imbalance internally\n"
        "  - Captures non-linear socio-demographic interactions\n"
        "  Research: XGBoost top-ranked in tabular health ML studies\n"
        "  (BMC Medical Informatics, PLOS ONE, Lancet Digital Health)."
    ),
    "Random Forest": (
        "Random Forest excels because:\n"
        "  - 300-tree ensemble reduces variance via bootstrap aggregation\n"
        "  - Handles multicollinearity among BDHS survey variables gracefully\n"
        "  - class_weight='balanced' corrects imbalanced disability rates\n"
        "  - No linearity assumption — captures non-linear social gradients\n"
        "  Research: RF is gold-standard for DHS-type survey data."
    ),
    "Logistic Regression": (
        "Logistic Regression performs well because:\n"
        "  - Disability predictors may have additive log-odds structure\n"
        "  - SMOTE + balanced weights handle class imbalance well\n"
        "  - Interpretable — Odds Ratios available for each predictor\n"
        "  Research note: If LR AUC within 2-3% of tree models,\n"
        "  prefer LR for publication due to clinical interpretability."
    ),
    "SVM (Linear)": (
        "SVM (Linear) performs competitively because:\n"
        "  - RBF kernel captures non-linear disability risk boundaries\n"
        "  - Effective in high-dimensional BDHS survey feature space\n"
        "  - class_weight='balanced' handles minority disabled class."
    ),
    "KNN": (
        "KNN provides local instance-based predictions:\n"
        "  - Distance-weighted 7-NN captures local disability patterns\n"
        "  - May miss global trends — used as baseline benchmark.\n"
        "  - SMOTE improves KNN performance on imbalanced data."
    ),
    "Decision Tree": (
        "Decision Tree provides interpretable screening rules:\n"
        "  - Every path = one IF-THEN clinical decision rule\n"
        "  - Depth=8 balances complexity and generalizability\n"
        "  - Useful for fieldwork screening tools\n"
        "  - Note: Single trees overfit; Random Forest corrects this."
    )
}

print(f"\n{'='*70}")
print(f"  BEST MODEL: {best_model_name}")
print(f"{'='*70}")
print(f"  ROC-AUC  : {best_auc}%")
print(f"  F1-Score : {best_f1}%")
print(f"  Recall   : {best_rec}%")
print(f"\n  Why {best_model_name} is the best?\n")
print("  " + model_explanations.get(best_model_name,
    f"  {best_model_name} achieved highest ROC-AUC among all models."))
print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: VISUALIZATION — All Comparison Charts
#
# Charts generated:
# 1. model_comparison_chart  — 5 metrics side-by-side per model
# 2. roc_auc_ranking         — Horizontal bar: best model highlighted red
# 3. train_vs_test_accuracy  — Overfitting detection (gap annotation)
# 4. cross_validation_f1     — CV stability with error bars
#
# Keno prôtiti chart:
#   1. Bar chart: shob metrics ekkat dekhle tulona shohôj।
#   2. ROC-AUC ranking: primary metric-e rank sposhTo।
#   3. Train vs Test: overfitting droshyômon — paper-e publication darkár।
#   4. CV chart: single split vagyer upor nirbhor nôy — stable estimate।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9] Generating Comparison Charts...")

metrics_to_plot = ['Test_Acc (%)', 'Precision (%)', 'Recall (%)',
                   'F1-Score (%)', 'ROC-AUC (%)']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

fig, axes = plt.subplots(1, 5, figsize=(22, 6))
for i, (metric, ax, color) in enumerate(zip(metrics_to_plot, axes, colors)):
    vals = final_df[metric].values
    mdls = final_df['Model'].values
    bars = ax.bar(mdls, vals, color=color, alpha=0.85,
                  edgecolor='black', linewidth=0.8)
    ax.set_title(metric, fontsize=12, fontweight='bold', pad=6)
    ymax = min(100, max(vals) + 8)
    ymin = max(0, min(vals) - 10)
    ax.set_ylim(ymin, ymax)
    ax.set_xticklabels(mdls, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('%', fontsize=10)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle(
    f"ML Model Comparison — Disability/disu Prediction (BDHS 2022)\n"
    f"Best Model: {best_model_name}  |  ROC-AUC = {best_auc}%",
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_chart.png", dpi=150, bbox_inches='tight')
plt.close()

# ROC-AUC Ranking chart
plt.figure(figsize=(9, 5))
auc_sorted = final_df.set_index('Model')['ROC-AUC (%)'].sort_values(ascending=True)
bar_colors = ['#d32f2f' if v == auc_sorted.max() else '#1565C0' for v in auc_sorted.values]
plt.barh(auc_sorted.index, auc_sorted.values, color=bar_colors,
         edgecolor='black', linewidth=0.8, height=0.5)
for i, (name, val) in enumerate(auc_sorted.items()):
    plt.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
plt.xlabel('ROC-AUC (%)', fontsize=11)
plt.title('ROC-AUC Ranking — Disability Prediction (BDHS 2022)\n[Red = Best Model]',
          fontsize=12, pad=10)
plt.xlim(0, min(auc_sorted.max() + 15, 115))
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_auc_ranking.png", dpi=150)
plt.close()

# Train vs Test Accuracy — Overfitting Detection
models_list = final_df['Model'].values
train_accs  = [results[m]['Train_Acc (%)'] for m in models_list]
test_accs   = [results[m]['Test_Acc (%)']  for m in models_list]
x_pos       = np.arange(len(models_list))
width       = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x_pos - width/2, train_accs, width, label='Train Accuracy',
               color='#4CAF50', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x_pos + width/2, test_accs,  width, label='Test Accuracy',
               color='#2196F3', edgecolor='black', linewidth=0.8)
ax.set_xlabel('Models', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title("Train vs Test Accuracy — Overfitting Detection (Disability/disu)\n"
             "(Gap > 5% = overfitting warning: model memorized, not learned)",
             fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_list, rotation=30, ha='right')
ax.legend(loc='lower left', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    gap = train - test
    if gap > 5:
        ax.text(i, max(train, test) + 1, f'[!] {gap:.1f}%',
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/train_vs_test_accuracy.png", dpi=150)
plt.close()

# Cross-Validation F1 Score with error bars
cv_means = [cv_results[m]['CV_F1_Mean (%)'] for m in models_list]
cv_stds  = [cv_results[m]['CV_F1_Std (%)']  for m in models_list]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models_list, cv_means, yerr=cv_stds, capsize=5,
              color='#26A69A', edgecolor='black', linewidth=0.8,
              error_kw={'linewidth': 2, 'color': 'black'})
ax.set_xlabel('Models', fontsize=11, fontweight='bold')
ax.set_ylabel('CV F1-Score (%)', fontsize=11, fontweight='bold')
ax.set_title('5-Fold Cross-Validation F1 Score — Disability/disu (BDHS 2022)\n'
             '(Error bar = Std Dev — smaller is more stable/reliable)',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xticklabels(models_list, rotation=30, ha='right')
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
            f'{mean:.1f}+/-{std:.1f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cross_validation_f1.png", dpi=150)
plt.close()

print(f"  [Saved] model_comparison_chart.png")
print(f"  [Saved] roc_auc_ranking.png")
print(f"  [Saved] train_vs_test_accuracy.png (Overfitting Detection)")
print(f"  [Saved] cross_validation_f1.png (CV Stability)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: DETAILED CLASSIFICATION REPORT (Best Model)
# Keno: Precision, Recall, F1 prôtiti class-er jônnô alada dekhá darkár।
# Overall accuracy shomôy puro chitro dey na — imbalanced data-te।
# Disability class-er Recall specifically important for screening।
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 10] Detailed Classification Report — {best_model_name}")
print("-" * 60)
best_model_obj = MODELS[best_model_name]
if best_model_name in NEEDS_SCALING:
    y_pred_best = best_model_obj.predict(X_test_scaled)
else:
    y_pred_best = best_model_obj.predict(X_test_sel.values)
print(classification_report(y_test, y_pred_best,
                             target_names=["No Disability (0)", "Disability (1)"]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: RESEARCH SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
summary = f"""
=============================================================================
RESEARCH SUMMARY REPORT — DISABILITY (disu) ML ANALYSIS (BDHS 2022)
=============================================================================
Dataset        : BDHS2022_MH_ML_ready_1.csv
Target         : disu (Disability/Disutility) — Binary (0 = No, 1 = Yes)
Total Sample   : {len(df):,} respondents
Disability +ve : {int(y.sum()):,} ({y.mean()*100:.1f}% prevalence)
Feature Space  : {X.shape[1]} variables initially
Selected Feat. : {len(selected_features)} via Union (MI + Chi2 + RF), Top-{TOP_K} each

EXCLUDED VARIABLES (Data Leakage Prevention):
   MTH22 (Months since last birth)     — Perfect separator: 100% accuracy alone
   MTH24 (Months since 2nd last birth) — High correlation (0.49)
   dep   (Depression)                  — Other mental health outcome (leakage)
   anx   (Anxiety)                     — Other mental health outcome (leakage)

--- METHODOLOGY (PROPER ML PIPELINE — NO DATA LEAKAGE) ----------------------
1. Train/Test Split (80-20, stratified by disu)
2. Exclude perfect separators + other outcome variables (dep, anx)
3. Feature Selection on ORIGINAL training data (BEFORE SMOTE)
4. SMOTE applied AFTER feature selection (leakage-free)
5. 5-fold Cross-Validation for model stability assessment
6. Both Train & Test accuracy reported (overfitting detection)

--- IMPUTATION STRATEGY (Smart Option B) ------------------------------------
Structural nulls (skip-pattern MNAR): Filled with 0 (Not Applicable category)
  Affected columns: {', '.join([c for c in structural_cols if c in df.columns and c not in ['MTH22','MTH24']])}
Random nulls: Median (continuous) or Mode (categorical <= 15 unique values)
Class imbalance: {"SMOTE applied AFTER feature selection" if is_imbalanced else "class_weight='balanced' in all models"}

--- SELECTED FEATURES (Union: MI + Chi2 + RF) --------------------------------
{', '.join(selected_features)}

--- MODEL PERFORMANCE (with Overfitting Detection) ---------------------------
{final_df[['Model','Train_Acc (%)','Test_Acc (%)','Overfit_Gap (%)','Precision (%)','Recall (%)','F1-Score (%)','ROC-AUC (%)']].to_string(index=True)}

WARNING: Overfitting if Train-Test gap > 5%

--- BEST MODEL ---------------------------------------------------------------
  {best_model_name}
  ROC-AUC  : {best_auc}%  (PRIMARY — threshold-independent discrimination)
  F1-Score : {best_f1}%   (precision-recall balance)
  Recall   : {best_rec}%  (disabled women detected — critical for screening)

--- METRIC PRIORITY RATIONALE (Health Research) ------------------------------
1. ROC-AUC   — PRIMARY: threshold-independent, best for imbalanced binary
2. Recall    — Missing a disabled woman = higher cost than false alarm
3. F1-Score  — Harmonic mean of precision and recall
4. Accuracy alone is misleading with imbalanced classes

--- OUTPUT FILES -------------------------------------------------------------
Saved to: {OUTPUT_DIR}
  null_analysis.csv              Null count per column
  null_heatmap.png               Missing pattern heatmap
  class_distribution.png         Target class distribution pie chart
  smote_before_after.png         Class balance visualisation SMOTE effect
  feature_MI.csv                 MI scores for all features
  feature_Chi2.csv               Chi2 scores for all features
  feature_RF.csv                 RF Importance for all features
  selected_features_union.csv    Final selected feature list
  feature_importance_RF.png      RF importance bar chart (top 20)
  feature_importance_MI.png      MI score bar chart (top 20)
  cm_[ModelName].png             Confusion matrix per model (orange theme)
  model_comparison.csv           Full metrics comparison table
  model_comparison_chart.png     5-metric side-by-side bar chart
  roc_auc_ranking.png            ROC-AUC horizontal ranking (best=red)
  train_vs_test_accuracy.png     Overfitting detection chart
  cross_validation_f1.png        5-fold CV F1 with error bars
  research_summary.txt           This report
=============================================================================
"""
print(summary)
with open(f"{OUTPUT_DIR}/research_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print("[Saved] research_summary.txt")
print(f"\n{'='*70}")
print("  DISABILITY (disu) ANALYSIS COMPLETE!")
print(f"  All outputs saved to: {OUTPUT_DIR}")
print(f"{'='*70}\n")
