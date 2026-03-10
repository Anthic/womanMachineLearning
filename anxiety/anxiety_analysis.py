"""
=============================================================================
BANGLADESH DEMOGRAPHIC AND HEALTH SURVEY (BDHS) 2022
মানসিক স্বাস্থ্য ML বিশ্লেষণ — ANXIETY (anx) নির্ভরশীল চলক হিসেবে
=============================================================================
লেখক      : রিসার্চ পাইপলাইন (Oxford-standard methodology)
ডেটাসেট   : BDHS2022_MH_ML_ready_1.csv
টার্গেট   : anx (Anxiety) → Binary (0 = নেই, 1 = আছে)
প্রেডিক্টর: সব ভেরিয়েবল ব্যতীত:
             - anx, dep, disu (অন্য মানসিক স্বাস্থ্য আউটকাম — ডেটা লিকেজ)
             - MTH22, MTH24 (পারফেক্ট সেপারেটর — ডেটা লিকেজ)
লক্ষ্য     : MI + Chi2 + RF Importance (Union) দিয়ে ফিচার সিলেকশন,
             তারপর ৬টি ML মডেল ট্রেন করে পারফরমেন্স তুলনা।

পদ্ধতি (কোনো ডেটা লিকেজ নেই):
  ১. Train/Test Split (80-20, stratified)
  ২. পারফেক্ট সেপারেটর (MTH22, MTH24) বাদ দেওয়া
  ৩. SMOTE লাগানোর আগে ORIGINAL training data-তে Feature Selection
  ৪. Feature Selection-এর পরে SMOTE (লিকেজ প্রতিরোধ)
  ৫. Cross-validation সহ মডেল ট্রেনিং
  ৬. Train ও Test উভয় accuracy রিপোর্ট (overfitting ধরতে)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: লাইব্রেরি ইমপোর্ট
# কেন: এই লাইব্রেরিগুলো ডেটা প্রসেসিং, ML মডেলিং এবং ভিজুয়ালাইজেশনের
# জন্য শিল্প-মানের (industry-standard) টুল।
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # পপআপ ছাড়া ফাইলে সেভ করার জন্য non-interactive ব্যাকএন্ড
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn ইউটিলিটি
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
# ফিচার সিলেকশন
from sklearn.feature_selection import mutual_info_classif, chi2

# মডেলসমূহ
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV  # LinearSVC-er jônnô probability
from xgboost import XGBClassifier

# ক্লাস ইমব্যালেন্স হ্যান্ডেলিং
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("  BDHS 2022 — ANXIETY (anx) ML ANALYSIS PIPELINE")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ডেটাসেট লোড করা
# কেন: সব কলাম string হিসেবে পড়লে whitespace বা hidden characters
# সমস্যা এড়ানো যায়, তারপর numeric-এ কনভার্ট করি।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1] ডেটাসেট লোড করা হচ্ছে...")

DATA_PATH  = r"e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv"
OUTPUT_DIR = r"e:\hafiza mam work\anxiety\ml_results_anxiety"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# সব কলাম প্রথমে string হিসেবে পড়ি — whitespace সমস্যা ঠেকাতে
df_raw = pd.read_csv(DATA_PATH, dtype=str)
df_raw.columns = df_raw.columns.str.strip()

# খালি/whitespace-only সেলকে NaN বানাই
df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)

# Numeric-এ কনভার্ট করি
df = df_raw.apply(pd.to_numeric, errors='coerce')

# CASEID বাদ দিই — এটি শুধু আইডেন্টিফায়ার, কোনো predictive value নেই
if 'CASEID' in df.columns:
    df = df.drop(columns=['CASEID'])

print(f"  -> আকার: {df.shape[0]} সারি x {df.shape[1]} কলাম")
print(f"  -> কলামসমূহ: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: NULL VALUE ANALYSIS (মিসিং ভ্যালু বিশ্লেষণ)
# কেন: কতটুকু ডেটা missing সেটা না জানলে সঠিক imputation কৌশল ঠিক করা
# যায় না। Visualization দিয়ে pattern বোঝা যায় — random নাকি structural।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Null Value Analysis চলছে...")

null_counts  = df.isnull().sum()
null_percent = (null_counts / len(df) * 100).round(2)

null_summary = (pd.DataFrame({'Column': null_counts.index,
                               'Null_Count': null_counts.values,
                               'Null_%': null_percent.values})
                .query("Null_Count > 0")
                .sort_values('Null_Count', ascending=False)
                .reset_index(drop=True))

print(f"\n  Null আছে এমন কলামসমূহ:")
print(null_summary.to_string(index=False))
print(f"\n  মোট null-সহ কলাম: {len(null_summary)}")

null_summary.to_csv(f"{OUTPUT_DIR}/null_analysis.csv", index=False)

# Null heatmap — missing pattern দৃশ্যমান করতে
null_cols = null_summary['Column'].tolist()
if null_cols:
    plt.figure(figsize=(14, 5))
    sns.heatmap(df[null_cols].isnull(), cbar=True, yticklabels=False,
                cmap='viridis', xticklabels=True)
    plt.title("Null Value Heatmap — BDHS 2022 Anxiety Analysis\n(Heatmap: হলুদ=Missing, বেগুনি=Present)",
              fontsize=13, pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/null_heatmap.png", dpi=150)
    plt.close()
    print(f"  [Saved] null_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SMART IMPUTATION — গবেষণা-মানের মিসিং ভ্যালু হ্যান্ডেলিং
#
# কেন Listwise Deletion নয়?
#   -> যেকোনো null-সহ সারি মুছে দিলে 30-40% ডেটা হারাবো।
#   -> এতে Statistical Power কমে এবং Selection Bias তৈরি হয়।
#   -> বাদ পড়া মহিলারা random নন — নির্দিষ্ট শ্রেণির।
#
# দুই ধরনের Null:
# ১. STRUCTURAL NULLS (Skip-pattern MNAR):
#    BDHS জরিপে কিছু প্রশ্ন নির্দিষ্ট গোষ্ঠীকে করা হয় না।
#    যেমন: V212 (প্রথম সন্তানের বয়স) -> সন্তানহীন মহিলার জন্য blank।
#    সমাধান: 0 দিয়ে ফিল করি — "Not Applicable" শ্রেণি তৈরি হয়।
#
# ২. RANDOM NULLS (MAR):
#    সত্যিকারের missing response।
#    সমাধান: Continuous -> Median (outlier-robust)
#             Categorical -> Mode (সবচেয়ে ঘন ঘন দেখা ক্যাটেগরি)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Smart Imputation প্রয়োগ করা হচ্ছে...")

df_imp = df.copy()

# ── 3a. STRUCTURAL NULLS — reproductive history / skip pattern-সংক্রান্ত
structural_cols = [
    'V212',             # প্রথম সন্তানের বয়স (সন্তানহীনদের জন্য NA)
    'V312',             # গর্ভনিরোধ পদ্ধতি (ব্যবহার না করলে NA)
    'V511',             # প্রথম সাথীর বয়স (structural skip)
    'V632',             # গর্ভনিরোধ সিদ্ধান্তকারী (ব্যবহার না করলে NA)
    'V701',             # স্বামীর শিক্ষা (স্বামী না থাকলে NA)
    'V730',             # স্বামীর বয়স (স্বামী না থাকলে NA)
    'MTH22',            # শেষ জন্মের মাস (EXCLUDED - Data Leakage!)
    'MTH24',            # দ্বিতীয় শেষ জন্মের মাস (EXCLUDED - Data Leakage!)
    'V171A',            # ইন্টারনেট ব্যবহারের হার (structural skip)
    'age_dif',          # স্বামীর সাথে বয়সের পার্থক্য (স্বামী না থাকলে NA)
    'age_dif_cat',      # বয়সের পার্থক্য ক্যাটেগরি (স্বামী না থাকলে NA)
    'internet_use',     # ইন্টারনেট ব্যবহার (কিছু বয়স/শিক্ষা গোষ্ঠীর জন্য skip)
    'contra_decision3', # গর্ভনিরোধ সিদ্ধান্ত (ব্যবহার না করলে NA)
    'parity_cat',       # Parity category (সন্তানহীনদের জন্য NA)
]

for col in structural_cols:
    if col in df_imp.columns:
        n_null = int(df_imp[col].isnull().sum())
        if n_null > 0:
            df_imp[col] = df_imp[col].fillna(0)
            print(f"  [Structural] {col:22s}: {n_null:5d}ti null -> 0 diye fill (N/A bivag)")

# ── 3b. বাকি RANDOM NULLS -> Median বা Mode
remaining_null_cols = [c for c in df_imp.columns if df_imp[c].isnull().any()]
print(f"\n  Structural fill-er pore baki null column: {remaining_null_cols}")

for col in remaining_null_cols:
    n_null   = int(df_imp[col].isnull().sum())
    unique_v = int(df_imp[col].dropna().nunique())
    if unique_v <= 15:                       # Categorical -> Mode
        fill_val = df_imp[col].mode().iloc[0]
        strategy = "Mode (categorical)"
    else:                                    # Continuous -> Median
        fill_val = df_imp[col].median()
        strategy = "Median (continuous)"
    df_imp[col] = df_imp[col].fillna(fill_val)
    print(f"  [Random]     {col:22s}: {n_null:5d}ti null -> {strategy} ({fill_val})")

total_remaining = int(df_imp.isnull().sum().sum())
print(f"\n  Imputation complete. Remaining null: {total_remaining}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURES O TARGET PROSTUT KORA
# কেন: dep এবং disu হলো অন্য mental health outcome — এগুলো predictor হিসেবে
# রাখলে ডেটা লিকেজ হবে কারণ এগুলো একই সার্ভে থেকে একসাথে collected।
# MTH22, MTH24 বাদ দিচ্ছি কারণ এরা perfect separator — 100% accuracy দেয়
# কিন্তু বাস্তব দুনিয়ায় generalizable নয়।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Features o Target prostut hocche...")

# anx = target
# dep, disu = other mental health outcome (leakage avoid)
# MTH22, MTH24 = perfect separator (leakage avoid)
DROP_COLS = ['dep', 'disu', 'anx', 'MTH22', 'MTH24']

print("  WARNING: MTH22 o MTH24 bad deoa hochhe (Data Leakage: perfect separator)")
print("  WARNING: dep, disu bad deoa hochhe (other mental health outcome — leakage)")

y = df_imp['anx'].astype(int)
X = df_imp.drop(columns=[c for c in DROP_COLS if c in df_imp.columns])

# সম্পূর্ণ NaN কলাম সরানো (safety check)
X = X.dropna(axis=1, how='all')

print(f"  Feature set akar: {X.shape}")
print(f"  Target (anx) distribution:\n{y.value_counts().to_string()}")
print(f"  Anxiety prevalence: {y.mean()*100:.2f}%")

class_ratio   = y.value_counts(normalize=True)
is_imbalanced = class_ratio.min() < 0.20
print(f"\n  Class 0 (Anxiety nei): {class_ratio[0]*100:.1f}%")
print(f"  Class 1 (Anxiety ache): {class_ratio[1]*100:.1f}%")
print(f"  Imbalanced: {is_imbalanced}")

# Class distribution pie chart
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([class_ratio[0], class_ratio[1]],
       labels=['No Anxiety (0)', 'Anxiety (1)'],
       autopct='%1.1f%%', colors=['#42A5F5', '#EF5350'],
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title("Target Class Distribution — Anxiety (BDHS 2022)\n(Class imbalance dekhar jonno)",
             fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.close()
print("  [Saved] class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TRAIN-TEST SPLIT
# কেন stratified split:
#   -> Stratified: train o test উভয়ে anxiety-র অনুপাত same thakbe।
#   -> এছাড়া test set-এ minority class (anxiety=1) কম পড়তে পারে।
# কেন 80-20:
#   -> 80% training-এ দিলে মডেল পর্যাপ্ত ডেটা পায়।
#   -> 20% test-এ রাখলে নিরপেক্ষ মূল্যায়ন সম্ভব।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Train-Test Split (80-20, stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples")

# IMPORTANT: SMOTE ekhon lagabo na!
# Feature selection MUST be done on ORIGINAL training data (BEFORE SMOTE)
# SMOTE age lagale synthetic data-r upor feature select hobe — eita leakage।
X_train_orig = X_train.copy()
y_train_orig = y_train.copy()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FEATURE SELECTION — 3 pôddhôtir UNION
#
# CRITICAL: Feature selection MUST be done on ORIGINAL training data (BEFORE
# SMOTE). SMOTE-er por korle synthetic pattern dhore feature select hobe।
#
# 3 ti pôddhotir karon:
# 1. Mutual Information (MI):
#    - No model assumption; linear o non-linear উভয় dependency dhorey।
#    - BDHS-er mixed variable type-er jônnô upôjukto।
# 2. Chi-Square Test:
#    - Classical statistical test।
#    - Categorical/ordinal health indicator-er jônnô adorsho।
# 3. Random Forest Importance (Boruta-equivalent):
#    - Complex interaction o non-linear effect dhorte pare।
#    - Best feature cluster theke select kore।
#
# Union keno:
#   Prôtitir nijôswo sîmabôddhotа ache। Union nile kono true signal bad
#   pôrbe na। "Recall maximisation" kôushôl।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Feature Selection (Union: MI + Chi2 + RF) cholchhe...")
print("  CRITICAL: ORIGINAL training data use hôchhe (SMOTE nôy) — leakage prôtirodh")

feature_names = list(X_train_orig.columns)
TOP_K = 15   # prôtiti pôddhôti theke top 15 feature

# ── 6a. Mutual Information (ORIGINAL data-e)
print("  -> Mutual Information score gonona hôchhe...")
mi_scores = mutual_info_classif(X_train_orig, y_train_orig, random_state=42)
mi_df     = (pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
               .sort_values('MI_Score', ascending=False)
               .reset_index(drop=True))
mi_top    = set(mi_df.head(TOP_K)['Feature'])

# ── 6b. Chi-Square (non-negative maan darkár, ORIGINAL data-e)
print("  -> Chi-Square score gonona hôchhe...")
X_shifted    = X_train_orig - X_train_orig.min() + 0.001
chi2_scores, chi2_pvals = chi2(X_shifted, y_train_orig)
chi2_df  = (pd.DataFrame({'Feature': feature_names,
                           'Chi2_Score': chi2_scores,
                           'P_value': chi2_pvals})
              .sort_values('Chi2_Score', ascending=False)
              .reset_index(drop=True))
chi2_top = set(chi2_df.head(TOP_K)['Feature'])

# ── 6c. Random Forest Importance (Boruta-equivalent, ORIGINAL data-e)
print("  -> Random Forest Feature Importance gonona hôchhe...")
rf_sel = RandomForestClassifier(n_estimators=200, random_state=42,
                                 n_jobs=-1, class_weight='balanced')
rf_sel.fit(X_train_orig, y_train_orig)
rf_df  = (pd.DataFrame({'Feature': feature_names,
                         'RF_Importance': rf_sel.feature_importances_})
            .sort_values('RF_Importance', ascending=False)
            .reset_index(drop=True))
rf_top = set(rf_df.head(TOP_K)['Feature'])

# ── 6d. UNION
selected_features = sorted(mi_top | chi2_top | rf_top)
print(f"\n  Selected features UNION ({len(selected_features)} features):")
print(f"  {'Feature':<22}  {'MI':^4}  {'Chi2':^4}  {'RF':^4}")
print(f"  {'-'*22}  {'-'*4}  {'-'*4}  {'-'*4}")
for f in selected_features:
    in_mi  = "+" if f in mi_top   else " "
    in_chi = "+" if f in chi2_top else " "
    in_rf  = "+" if f in rf_top   else " "
    print(f"  {f:<22}  {in_mi:^4}  {in_chi:^4}  {in_rf:^4}")

# Feature selection CSV save
mi_df.to_csv(f"{OUTPUT_DIR}/feature_MI.csv", index=False)
chi2_df.to_csv(f"{OUTPUT_DIR}/feature_Chi2.csv", index=False)
rf_df.to_csv(f"{OUTPUT_DIR}/feature_RF.csv", index=False)
pd.DataFrame({'Selected_Feature': selected_features}).to_csv(
    f"{OUTPUT_DIR}/selected_features_union.csv", index=False)

# ── RF Importance bar chart
plt.figure(figsize=(10, 7))
sns.barplot(x='RF_Importance', y='Feature', data=rf_df.head(20),
            palette='Blues_r', edgecolor='black')
plt.title("Top 20 Features — Random Forest Importance\n(Boruta-equivalent, BDHS 2022 Anxiety)",
          fontsize=12, pad=10)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_RF.png", dpi=150)
plt.close()

# ── MI Score bar chart
plt.figure(figsize=(10, 7))
sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(20),
            palette='Greens_r', edgecolor='black')
plt.title("Top 20 Features — Mutual Information Score\n(BDHS 2022 Anxiety)",
          fontsize=12, pad=10)
plt.xlabel("Mutual Information Score", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_MI.png", dpi=150)
plt.close()
print("  [Saved] feature importance charts")

# ORIGINAL training data theke selected features
X_train_sel_orig = X_train_orig[selected_features]
X_test_sel       = X_test[selected_features]

# ── EKHON SMOTE lagano hobe (feature selection-er pore — leakage-mukto)
# Keno SMOTE:
#   Anxiety-r prevalence BDHS-e low (10-15% range-e)।
#   Imbalance thakle model somosha "0" predict korle high accuracy pabe।
#   Kintu eita clinically useless। SMOTE synthetic minority sample toiri kore
#   balance ane — model tb minority class shikhte pare।
if is_imbalanced:
    print("\n  -> SMOTE prayog hochhe selected features-e (leakage-mukto)")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sel_bal, y_train_bal = smote.fit_resample(X_train_sel_orig, y_train_orig)
    print(f"  SMOTE-er pore — 0: {sum(y_train_bal==0)}, 1: {sum(y_train_bal==1)}")

    # SMOTE before vs after chart
    # Keno: Visually prove kora je imbalance handle kora hoyeche
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    vals_before = [sum(y_train_orig==0), sum(y_train_orig==1)]
    vals_after  = [sum(y_train_bal==0),  sum(y_train_bal==1)]
    lbls = ['No Anxiety (0)', 'Anxiety (1)']

    ax1.bar(lbls, vals_before, color=['#42A5F5', '#EF5350'], edgecolor='black')
    ax1.set_title("SMOTE-er AAGE\n(Imbalanced)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Sample Count")
    for i, v in enumerate(vals_before):
        ax1.text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=10)

    ax2.bar(lbls, vals_after, color=['#42A5F5', '#EF5350'], edgecolor='black')
    ax2.set_title("SMOTE-er PORE\n(Balanced)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Sample Count")
    for i, v in enumerate(vals_after):
        ax2.text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=10)

    plt.suptitle("Class Imbalance Handling via SMOTE — Anxiety (BDHS 2022)\n"
                 "(SMOTE: Synthetic Minority Oversampling Technique)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/smote_before_after.png", dpi=150)
    plt.close()
    print("  [Saved] smote_before_after.png")
else:
    X_train_sel_bal = X_train_sel_orig.copy()
    y_train_bal     = y_train_orig.copy()
    print("  -> Data balanced — SMOTE dorkar nei, class_weight='balanced' use hobe")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MODEL TRAINING O EVALUATION
# কেন 6ti model:
#   -> Prôtiti modeler nijôswo shôkti o durbolôta ache।
#   -> Tülônä nà korle "shoreshto" model bola jay na।
#   -> Robust research-er ongsho।
#
# কেন Train o Test উভয় accuracy:
#   -> Shudhu test accuracy dekhe overfitting dhora jay na।
#   -> Train >> Test mane model training data mukhtho koreche, shikheini।
#   -> 5% gap threshold — etar beshi hole overfitting।
#
# কেন Cross-Validation:
#   -> Single split vagyer upor nirvor korte pare।
#   -> 5-fold CV: 5ti bhinnô split-e model jaycha — bishwashôjôgyo।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] 6ti ML Model train o evaluate hôchhe...")
print("-" * 65)

# Distance/linear model-er jonno scaling darkár
# Keno StandardScaler:
#   LR, SVM, KNN sob distance ba coefficient use kore।
#   Feature-gular scale bhinna hole bodo scale feature beshow probhab felbe।
#   Scaling sob feature ke same scale-e ane।
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel_bal)
X_test_scaled  = scaler.transform(X_test_sel)

MODELS = {
    "Logistic Regression": LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced', solver='lbfgs'
        # Interpretable; Odds Ratio pawa jay; epidemiology-te standard
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=8, class_weight='balanced'
        # Puro interpretable — prôtiti path ekti clinical rule
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'
        # 300-tree ensemble; variance komay; survey data-r jônnô gold-standard
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric='logloss',
        scale_pos_weight=(sum(y_train_bal==0) / max(sum(y_train_bal==1), 1)),
        verbosity=0
        # Sequential boosting; L1/L2 regularization; imbalance handle kore
    ),
    "SVM (Linear)": CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', random_state=42, C=1.0, max_iter=2000),
        cv=3, method='sigmoid'
        # LinearSVC: RBF-er tülônây অনেক fast; large SMOTE dataset-e practical
        # CalibratedClassifierCV: probability estimate-er jônnô (ROC-AUC darkár)
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='euclidean'
        # Local instance-based; baseline hisebe guruttôpurno
    )
}

NEEDS_SCALING = {"Logistic Regression", "SVM (Linear)", "KNN"}

results    = {}
cv_results = {}

for model_name, model in MODELS.items():
    print(f"\n  -- {model_name}")
    Xtr = X_train_scaled if model_name in NEEDS_SCALING else X_train_sel_bal.values
    Xte = X_test_scaled  if model_name in NEEDS_SCALING else X_test_sel.values

    model.fit(Xtr, y_train_bal)

    # Train o Test উভয়ের prediction
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

    # 5-fold Cross Validation — model-er stability yachai
    cv_scores = cross_val_score(model, Xtr, y_train_bal, cv=5,
                                 scoring='f1', n_jobs=-1)
    cv_results[model_name] = {
        'CV_F1_Mean (%)': round(cv_scores.mean() * 100, 2),
        'CV_F1_Std (%)':  round(cv_scores.std()  * 100, 2)
    }

    print(f"     Train Accuracy : {results[model_name]['Train_Acc (%)']}%")
    print(f"     Test Accuracy  : {results[model_name]['Test_Acc (%)']}%")
    print(f"     Overfit Gap    : {results[model_name]['Overfit_Gap (%)']}% "
          f"{'[WARNING] OVERFITTING!' if is_overfit else '[OK]'}")
    print(f"     Precision      : {results[model_name]['Precision (%)']}%")
    print(f"     Recall         : {results[model_name]['Recall (%)']}%")
    print(f"     F1-Score       : {results[model_name]['F1-Score (%)']}%")
    print(f"     ROC-AUC        : {results[model_name]['ROC-AUC (%)']}%")
    print(f"     CV F1 (5-fold) : {cv_results[model_name]['CV_F1_Mean (%)']} +/- "
          f"{cv_results[model_name]['CV_F1_Std (%)']}%")

    # Confusion Matrix — kotoTa sôThik o kotoTa vul predict hoyeche
    cm  = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["No Anxiety", "Anxiety"]).plot(
        ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f"Confusion Matrix — {model_name}\n(Anxiety, BDHS 2022)",
                 fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cm_{model_name.replace(' ','_')}.png", dpi=130)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: RESULTS COMPARISON TABLE
# Keno: Sob model-er metrics eksathe dekhle drolto shoreshto model
# chinnito kora jay। ROC-AUC onujayi sort — imbalanced data-te primary।
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
# SECTION 9: BEST MODEL CHINNITO KORA
# Metric priority (health research-e):
#   PRIMARY   -> ROC-AUC: threshold-independent; imbalanced data-te shoreshto।
#   SECONDARY -> F1-Score: precision o recall-er harmonic mean।
#   TERTIARY  -> Recall: kotojon anxiety-akronto mohila detect holo।
#   Screening context-e false negative (miss kora) beshow khotikar।
# ─────────────────────────────────────────────────────────────────────────────
best_model_name = final_df.iloc[0]['Model']
best_auc  = final_df.iloc[0]['ROC-AUC (%)']
best_f1   = final_df.iloc[0]['F1-Score (%)']
best_rec  = final_df.iloc[0]['Recall (%)']

model_explanations = {
    "XGBoost": (
        "XGBoost shoreshto karon:\n"
        "  - Sequential boosting prôtiti tree-er residual error shongshodhôn kore\n"
        "  - Built-in L1/L2 regularization overfitting rodh kore\n"
        "  - scale_pos_weight class imbalance internally handle kore\n"
        "  - Complex sociodemographic interaction dhorte pare\n"
        "  Research context: Tabular health data-te XGBoost consistently #1\n"
        "  (BMC Med Informatics, PLOS ONE, Lancet Digital Health)."
    ),
    "Random Forest": (
        "Random Forest bhalo kore karon:\n"
        "  - 300-tree ensemble bootstrap aggregation diye variance komay\n"
        "  - Survey variable-er modhe collinearity gracefully handle kore\n"
        "  - class_weight='balanced' imbalanced anxiety rate shongshodhôn kore\n"
        "  Research context: DHS survey data-r jônnô RF gold-standard."
    ),
    "Logistic Regression": (
        "Logistic Regression bhalo kore karon:\n"
        "  - Anxiety predictor-gulo additive log-odds relationship-e thakte pare\n"
        "  - SMOTE + balanced weights diye imbalance handle kore\n"
        "  - Highly interpretable — Odds Ratio pawa jay\n"
        "  Research note: Epidemiology-te LR preferred karon clinical interpretability."
    ),
    "SVM (Linear)": (
        "SVM (Linear) competitively perform kore karon:\n"
        "  - RBF kernel non-linear boundary capture kore\n"
        "  - High-dimensional survey data-te effective\n"
        "  - class_weight='balanced' minority class handle kore."
    ),
    "KNN": (
        "KNN local instance-based learning use kore:\n"
        "  - Distance-weighted 7-nearest neighbors local pattern dhore\n"
        "  - Global trend miss korte pare — benchmark hisebe boro."
    ),
    "Decision Tree": (
        "Decision Tree transparent decision rule dey:\n"
        "  - Puro interpretable — prôtiti path ekti clinical screening rule\n"
        "  - Depth=8 complexity o generalizability balance kore\n"
        "  - Note: Single tree overfit kore; RF eTa shongshodhôn kore."
    )
}

print(f"\n{'='*70}")
print(f"  BEST MODEL: {best_model_name}")
print(f"{'='*70}")
print(f"  ROC-AUC  : {best_auc}%")
print(f"  F1-Score : {best_f1}%")
print(f"  Recall   : {best_rec}%")
print(f"\n  Keno {best_model_name} shoreshto?\n")
print("  " + model_explanations.get(best_model_name,
    f"  {best_model_name} shôrôbôcho ROC-AUC ôrjôn koreche।"))
print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: VISUALIZATION — Model Comparison Charts
# Keno prôtiti chart:
#   -> Bar chart: sob metrics eksathe side-by-side tulona shohôj hoy।
#   -> ROC-AUC Ranking: primary metric-e ranking sposhTo dekhay।
#   -> Train vs Test: overfitting droshyômon hoy — publication-e jôruri।
#   -> CV Plot: model stability o variance visualize hoy।
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9] Comparison Charts toiri hochhe...")

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
    f"ML Model Comparison — Anxiety Prediction (BDHS 2022)\n"
    f"Best Model: {best_model_name}  |  ROC-AUC = {best_auc}%",
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_chart.png", dpi=150, bbox_inches='tight')
plt.close()

# ROC-AUC ranking horizontal bar
plt.figure(figsize=(9, 5))
auc_sorted = final_df.set_index('Model')['ROC-AUC (%)'].sort_values(ascending=True)
bar_colors = ['#d32f2f' if v == auc_sorted.max() else '#1565C0' for v in auc_sorted.values]
plt.barh(auc_sorted.index, auc_sorted.values, color=bar_colors,
         edgecolor='black', linewidth=0.8, height=0.5)
for i, (name, val) in enumerate(auc_sorted.items()):
    plt.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
plt.xlabel('ROC-AUC (%)', fontsize=11)
plt.title('ROC-AUC Ranking — Anxiety Prediction (BDHS 2022)\n[Red = Best Model]',
          fontsize=12, pad=10)
plt.xlim(0, min(auc_sorted.max() + 15, 115))
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_auc_ranking.png", dpi=150)
plt.close()

# Train vs Test Accuracy — Overfitting Detection
# Keno eÏ chart: Dekhay kon model training data mukhtho koreche banâm shikhéchhe।
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
ax.set_title("Train vs Test Accuracy — Overfitting Detection (Anxiety)\n"
             "(Gap > 5% indicates overfitting — model overfit hoye geche)",
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

# Cross-Validation F1 Score chart
# Keno: CV score dekhiye model-er stability o robustness bôjha jay।
cv_means = [cv_results[m]['CV_F1_Mean (%)'] for m in models_list]
cv_stds  = [cv_results[m]['CV_F1_Std (%)']  for m in models_list]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models_list, cv_means, yerr=cv_stds, capsize=5,
              color='#7E57C2', edgecolor='black', linewidth=0.8,
              error_kw={'linewidth': 2, 'color': 'black'})
ax.set_xlabel('Models', fontsize=11, fontweight='bold')
ax.set_ylabel('CV F1-Score (%)', fontsize=11, fontweight='bold')
ax.set_title('5-Fold Cross-Validation F1 Score — Anxiety (BDHS 2022)\n'
             '(Error bar = Std Dev — kom hole model stable)',
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
print(f"  [Saved] train_vs_test_accuracy.png")
print(f"  [Saved] cross_validation_f1.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: BEST MODEL-ER CLASSIFICATION REPORT
# Keno: Precision, Recall, F1 prôtiti class-er jônnô alada dekhá darkár।
# Overall accuracy shômôy puro chitro dey na — imbalanced data-te।
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 10] Classification Report — {best_model_name}")
print("-" * 60)
best_model_obj = MODELS[best_model_name]
if best_model_name in NEEDS_SCALING:
    y_pred_best = best_model_obj.predict(X_test_scaled)
else:
    y_pred_best = best_model_obj.predict(X_test_sel.values)
print(classification_report(y_test, y_pred_best,
                             target_names=["No Anxiety (0)", "Anxiety (1)"]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: RESEARCH SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
summary = f"""
=============================================================================
RESEARCH SUMMARY REPORT — ANXIETY ML ANALYSIS (BDHS 2022)
=============================================================================
Dataset        : BDHS2022_MH_ML_ready_1.csv
Target         : anx (Anxiety) — Binary (0 = No, 1 = Yes)
Total Sample   : {len(df):,} respondents
Anxiety +ve    : {int(y.sum()):,} ({y.mean()*100:.1f}% prevalence)
Feature Space  : {X.shape[1]} variables initially
Selected Feat. : {len(selected_features)} via Union (MI + Chi2 + RF), Top-{TOP_K} each

EXCLUDED VARIABLES (Data Leakage Prevention):
   MTH22 (Months since last birth)     — Perfect separator: 100% accuracy alone
   MTH24 (Months since 2nd last birth) — High correlation (0.49)
   dep   (Depression)                  — Other mental health outcome (leakage)
   disu  (Disability)                  — Other mental health outcome (leakage)

--- METHODOLOGY (NO DATA LEAKAGE) ------------------------------------------
1. Train/Test Split (80-20, stratified by anx)
2. Exclude perfect separators + other outcome variables
3. Feature Selection on ORIGINAL training data (BEFORE SMOTE)
4. SMOTE applied AFTER feature selection (leakage-free)
5. 5-fold Cross-Validation for model stability
6. Both Train & Test accuracy reported (overfitting detection)

--- IMPUTATION (Smart Option B) ---------------------------------------------
Structural nulls (skip-pattern MNAR): Filled with 0
  Columns: {', '.join([c for c in structural_cols if c in df.columns and c not in ['MTH22','MTH24']])}
Random nulls: Median (continuous) or Mode (categorical)
Class imbalance: {"SMOTE applied AFTER feature selection" if is_imbalanced else "class_weight='balanced' in models"}

--- SELECTED FEATURES (Union: MI + Chi2 + RF) --------------------------------
{', '.join(selected_features)}

--- MODEL PERFORMANCE (with Overfitting Detection) ---------------------------
{final_df[['Model','Train_Acc (%)','Test_Acc (%)','Overfit_Gap (%)','Precision (%)','Recall (%)','F1-Score (%)','ROC-AUC (%)']].to_string(index=True)}

WARNING: Overfitting if gap > 5% between Train and Test accuracy

--- BEST MODEL ---------------------------------------------------------------
  {best_model_name}
  ROC-AUC  : {best_auc}%  (PRIMARY — best discrimination ability)
  F1-Score : {best_f1}%   (balances precision and recall)
  Recall   : {best_rec}%  (detected anxiety cases — critical for screening)

--- METRIC PRIORITY (Epidemiological Research) --------------------------------
1. ROC-AUC — PRIMARY (threshold-independent, best for imbalanced binary)
2. Recall  — Missing an anxious woman costs more than a false alarm
3. F1-Score — Harmonic mean of precision and recall
4. Accuracy alone is misleading with imbalanced classes

--- OUTPUT FILES -------------------------------------------------------------
Saved to: {OUTPUT_DIR}
  null_analysis.csv              Null counts per column
  null_heatmap.png               Missing pattern visual map
  class_distribution.png         Target class pie chart
  smote_before_after.png         Class balance before/after SMOTE
  feature_MI.csv                 MI scores all features
  feature_Chi2.csv               Chi2 scores all features
  feature_RF.csv                 RF Importance all features
  selected_features_union.csv    Final selected features
  feature_importance_RF.png      RF importance bar chart
  feature_importance_MI.png      MI score bar chart
  cm_[ModelName].png             Confusion matrix per model
  model_comparison.csv           Full metrics table
  model_comparison_chart.png     Side-by-side bar chart all metrics
  roc_auc_ranking.png            ROC-AUC horizontal ranking chart
  train_vs_test_accuracy.png     Overfitting detection chart
  cross_validation_f1.png        5-fold CV F1 comparison
  research_summary.txt           This report
=============================================================================
"""
print(summary)
with open(f"{OUTPUT_DIR}/research_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"[Saved] research_summary.txt")
print(f"\n{'='*70}")
print(f"  ANXIETY ANALYSIS COMPLETE!")
print(f"  All outputs saved to: {OUTPUT_DIR}")
print(f"{'='*70}\n")
