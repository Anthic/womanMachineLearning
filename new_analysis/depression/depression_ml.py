"""
Depression ML Analysis — BDHS 2022
Dataset : clean_mental_health_data.csv
Target  : Depression (0=No, 1=Yes)
Features: Only 11 renamed variables (no raw V-codes, no survey weights)
"""

import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             matthews_corrcoef)
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = r"s:\Anthic kumar singh\womanMachineLearning\clean_mental_health_data.csv"
OUT_DIR    = r"s:\Anthic kumar singh\womanMachineLearning\new_analysis\depression\results"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'Depression'

# ── Only these 11 renamed features (no raw V-codes, no design vars) ───────────
FEATURE_COLS = [
    'Division', 'Residence', 'Respondent_Education',
    'age_at_First_Birth', 'Respondent_Age', 'Parity',
    'Spousal_Age_Gap', 'Wealth_Index', 'Internet_Use',
    'Contraceptive_Use', 'contra_decision_maker',
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  DEPRESSION ML ANALYSIS — BDHS 2022")
print("=" * 65)
print("\n[STEP 1] Loading data...")

df_raw = pd.read_csv(DATA_PATH, dtype=str)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)
df = df_raw.apply(pd.to_numeric, errors='coerce')

print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: MISSING DATA CHECK (only on feature columns + target)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Missing data analysis...")

work_cols = FEATURE_COLS + [TARGET]
df_work   = df[work_cols].copy()

null_df = pd.DataFrame({
    'Column'      : df_work.columns,
    'Null_Count'  : df_work.isnull().sum().values,
    'Null_Percent': (df_work.isnull().mean() * 100).round(2).values
}).query("Null_Count > 0").reset_index(drop=True)

if null_df.empty:
    print("  [OK] No missing values in feature/target columns.")
else:
    print(null_df.to_string(index=False))
    null_df.to_csv(f"{OUT_DIR}/missing_data_report.csv", index=False)
    # Fill with median (all numeric)
    for col in FEATURE_COLS:
        if df_work[col].isnull().any():
            df_work[col].fillna(df_work[col].median(), inplace=True)

df_work.dropna(subset=[TARGET], inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FEATURES & TARGET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Preparing features and target...")

X = df_work[FEATURE_COLS].copy()
y = df_work[TARGET].astype(int)

print(f"  Features : {list(X.columns)}")
print(f"  Target distribution:\n{y.value_counts().to_string()}")
print(f"  Depression prevalence: {y.mean()*100:.2f}%")

# Class balance check
is_imbalanced = y.value_counts(normalize=True).min() < 0.20
print(f"  Imbalanced: {is_imbalanced}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN-TEST SPLIT (80-20, stratified)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Train-Test split (80-20, stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: FEATURE SELECTION — UNION (MI + Chi2 + RF) on ORIGINAL training data
# NOTE: Must be done BEFORE SMOTE to avoid data leakage
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Feature selection (MI + Chi2 + RF Union)...")
print("  [NOTE] On original training data BEFORE SMOTE — no leakage")

TOP_K    = min(8, len(FEATURE_COLS))   # top-8 from each method
feat_names = list(X_train.columns)

# Mutual Information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df     = pd.DataFrame({'Feature': feat_names, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
mi_top    = set(mi_df.head(TOP_K)['Feature'])

# Chi-Square (shift to non-negative)
X_shift     = X_train - X_train.min() + 0.001
chi2_sc, _  = chi2(X_shift, y_train)
chi2_df     = pd.DataFrame({'Feature': feat_names, 'Chi2_Score': chi2_sc}).sort_values('Chi2_Score', ascending=False)
chi2_top    = set(chi2_df.head(TOP_K)['Feature'])

# Random Forest Importance
rf_sel = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf_sel.fit(X_train, y_train)
rf_df  = pd.DataFrame({'Feature': feat_names, 'RF_Importance': rf_sel.feature_importances_}).sort_values('RF_Importance', ascending=False)
rf_top = set(rf_df.head(TOP_K)['Feature'])

# Union
selected = sorted(mi_top | chi2_top | rf_top)
print(f"  Selected features ({len(selected)}): {selected}")

# Save feature scores
mi_df.to_csv(f"{OUT_DIR}/feature_MI.csv", index=False)
chi2_df.to_csv(f"{OUT_DIR}/feature_Chi2.csv", index=False)
rf_df.to_csv(f"{OUT_DIR}/feature_RF.csv", index=False)
pd.DataFrame({'Selected_Feature': selected}).to_csv(f"{OUT_DIR}/selected_features.csv", index=False)

# MI bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x='MI_Score', y='Feature', data=mi_df, palette='Blues_r')
plt.title("Mutual Information Score — Depression\n(Feature importance for predicting Depression)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_MI_chart.png", dpi=150)
plt.close()

# Reduce to selected features
X_train_sel = X_train[selected].copy()
X_test_sel  = X_test[selected].copy()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SMOTE — AFTER feature selection (leakage-free)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] SMOTE oversampling (on selected features)...")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)
print(f"  Before SMOTE — 0: {sum(y_train==0)}, 1: {sum(y_train==1)}")
print(f"  After  SMOTE — 0: {sum(y_train_bal==0)}, 1: {sum(y_train_bal==1)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: HYPERPARAMETER TUNING (GridSearchCV, 5-fold CV on SMOTE training data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Hyperparameter tuning (GridSearchCV)...")

scaler         = StandardScaler()
Xtr_sc         = scaler.fit_transform(X_train_bal)
Xte_sc         = scaler.transform(X_test_sel)

param_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        {'C': [0.01, 0.1, 1, 10]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        {'max_depth': [4, 6, 8, 10], 'min_samples_split': [2, 5, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        {'n_estimators': [100, 200], 'max_depth': [6, 10, None]}
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 6]}
    ),
    "SVM": (
        CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, max_iter=2000), cv=3),
        {'estimator__C': [0.01, 0.1, 1, 10]}
    ),
    "KNN": (
        KNeighborsClassifier(),
        {'n_neighbors': [5, 7, 11], 'weights': ['uniform', 'distance']}
    ),
}

NEEDS_SCALE = {"Logistic Regression", "SVM", "KNN"}

best_models  = {}
best_params  = {}
tuning_log   = []

for name, (estimator, param_grid) in param_grids.items():
    print(f"  Tuning {name}...")
    Xtr = Xtr_sc if name in NEEDS_SCALE else X_train_bal.values
    gs  = GridSearchCV(estimator, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, refit=True)
    gs.fit(Xtr, y_train_bal)
    best_models[name] = gs.best_estimator_
    best_params[name] = gs.best_params_
    print(f"    Best params: {gs.best_params_}  |  CV AUC: {gs.best_score_:.4f}")
    tuning_log.append({'Model': name, 'Best_Params': str(gs.best_params_), 'CV_AUC': round(gs.best_score_, 4)})

pd.DataFrame(tuning_log).to_csv(f"{OUT_DIR}/hyperparameter_tuning_log.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: EVALUATE ALL MODELS — All metrics + Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 8] Evaluating all models...")

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

results    = {}
cv_results = {}
all_preds  = {}

for name, model in best_models.items():
    Xtr = Xtr_sc if name in NEEDS_SCALE else X_train_bal.values
    Xte = Xte_sc if name in NEEDS_SCALE else X_test_sel.values

    y_pred_train = model.predict(Xtr)
    y_pred_test  = model.predict(Xte)
    y_prob       = model.predict_proba(Xte)[:, 1]
    all_preds[name] = (y_pred_test, y_prob)

    acc_tr  = accuracy_score(y_train_bal, y_pred_train)
    acc_te  = accuracy_score(y_test, y_pred_test)
    prec    = precision_score(y_test, y_pred_test, zero_division=0)
    rec     = recall_score(y_test, y_pred_test, zero_division=0)
    f1      = f1_score(y_test, y_pred_test, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)
    spec    = specificity(y_test, y_pred_test)
    mcc     = matthews_corrcoef(y_test, y_pred_test)

    results[name] = {
        'Train_Acc': round(acc_tr*100, 2),
        'Test_Acc' : round(acc_te*100, 2),
        'Overfit'  : round((acc_tr - acc_te)*100, 2),
        'Precision': round(prec*100, 2),
        'Recall'   : round(rec*100, 2),
        'F1'       : round(f1*100, 2),
        'ROC_AUC'  : round(auc*100, 2),
        'Specificity': round(spec*100, 2),
        'MCC'      : round(mcc, 4),
    }

    # 5-fold CV
    cv_sc = cross_val_score(model, Xtr, y_train_bal, cv=5, scoring='f1', n_jobs=-1)
    cv_results[name] = {'CV_F1_Mean': round(cv_sc.mean()*100, 2), 'CV_F1_Std': round(cv_sc.std()*100, 2)}

    print(f"\n  {name}:")
    print(f"    Train Acc={results[name]['Train_Acc']}%  Test Acc={results[name]['Test_Acc']}%  Overfit={results[name]['Overfit']}%")
    print(f"    Precision={results[name]['Precision']}%  Recall={results[name]['Recall']}%  F1={results[name]['F1']}%")
    print(f"    ROC-AUC={results[name]['ROC_AUC']}%  Specificity={results[name]['Specificity']}%  MCC={results[name]['MCC']}")
    print(f"    CV F1={cv_results[name]['CV_F1_Mean']} ± {cv_results[name]['CV_F1_Std']}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Dep", "Depression"]).plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f"Confusion Matrix — {name}\n(Depression Prediction)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cm_{name.replace(' ','_')}.png", dpi=150)
    plt.close()
    print(f"    [Saved] Confusion Matrix")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: RESULTS TABLE & CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9] Saving results table and charts...")

res_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
cv_df  = pd.DataFrame(cv_results).T.reset_index().rename(columns={'index': 'Model'})
final  = res_df.merge(cv_df, on='Model').sort_values('ROC_AUC', ascending=False).reset_index(drop=True)
final.index += 1
print(final.to_string())
final.to_csv(f"{OUT_DIR}/model_results.csv", index=True)

# ROC-AUC Ranking chart
auc_s = final.set_index('Model')['ROC_AUC'].sort_values()
colors_bar = ['#d32f2f' if v == auc_s.max() else '#1565C0' for v in auc_s]
plt.figure(figsize=(9, 5))
plt.barh(auc_s.index, auc_s.values, color=colors_bar, edgecolor='k', height=0.5)
for i, (nm, val) in enumerate(auc_s.items()):
    plt.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
plt.xlabel('ROC-AUC (%)')
plt.title('ROC-AUC Ranking — Depression Prediction\n[Red = Best Model]')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_auc_ranking.png", dpi=150)
plt.close()

# Train vs Test (overfitting check)
mdls = final['Model'].values
x    = np.arange(len(mdls)); w = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, final['Train_Acc'].values, w, label='Train', color='#4CAF50', edgecolor='k')
ax.bar(x + w/2, final['Test_Acc'].values,  w, label='Test',  color='#2196F3', edgecolor='k')
ax.set_xticks(x); ax.set_xticklabels(mdls, rotation=30, ha='right')
ax.set_ylabel('Accuracy (%)'); ax.legend()
ax.set_title('Train vs Test Accuracy — Overfitting Detection (Depression)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/train_vs_test.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: CROSS-VALIDATION CHART
# ─────────────────────────────────────────────────────────────────────────────
cv_means = [cv_results[m]['CV_F1_Mean'] for m in mdls]
cv_stds  = [cv_results[m]['CV_F1_Std']  for m in mdls]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(mdls, cv_means, yerr=cv_stds, capsize=5, color='#26A69A', edgecolor='k')
ax.set_ylabel('CV F1-Score (%)')
ax.set_title('5-Fold Cross-Validation F1 — Depression (error bar = std dev)')
ax.set_xticklabels(mdls, rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cross_validation_f1.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: SHAP ANALYSIS — Best model (by ROC-AUC)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 11] SHAP Analysis — Best model...")

best_name  = final.iloc[0]['Model']
best_model = best_models[best_name]
Xte_for_shap = pd.DataFrame(Xte_sc if best_name in NEEDS_SCALE else X_test_sel.values,
                             columns=selected)
print(f"  Best model: {best_name}")

try:
    explainer   = shap.Explainer(best_model, Xte_for_shap)
    shap_values = explainer(Xte_for_shap)

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=11, show=False)
    plt.title(f"SHAP Beeswarm — {best_name} (Depression)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, max_display=11, show=False)
    plt.title(f"SHAP Mean |SHAP| — {best_name} (Depression)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Saved] SHAP charts")
except Exception as e:
    print(f"  [SHAP Note] {e} — trying TreeExplainer...")
    try:
        exp = shap.TreeExplainer(best_model)
        sv  = exp.shap_values(Xte_for_shap)
        if isinstance(sv, list): sv = sv[1]
        shap.summary_plot(sv, Xte_for_shap, show=False)
        plt.title(f"SHAP Summary — {best_name} (Depression)")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  [Saved] SHAP summary (TreeExplainer)")
    except Exception as e2:
        print(f"  [SHAP Skip] {e2}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12: DETAILED CLASSIFICATION REPORT (Best model)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 12] Detailed report — {best_name}")
y_pred_best = all_preds[best_name][0]
print(classification_report(y_test, y_pred_best, target_names=["No Depression", "Depression"]))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13: WRITE RESULTS SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────
bp_str = '\n'.join([f"  {k}: {v}" for k, v in best_params.items()])
summary = f"""
======================================================================
DEPRESSION ML ANALYSIS RESULTS SUMMARY — BDHS 2022
======================================================================
Dataset : clean_mental_health_data.csv  |  Target: Depression
Total N : {len(df_work):,}  |  Depression+ : {int(y.sum()):,} ({y.mean()*100:.1f}%)

FEATURE SELECTION (Union: MI + Chi2 + RF, Top-{TOP_K} each):
  Selected ({len(selected)}): {', '.join(selected)}
  Note: Only 11 renamed variables used. Excluded: CASEID, V021 (PSU),
        V023 (Strata), Weight, V171A, V312, V511, V632, V701, V730.

SMOTE: {'Applied after feature selection' if is_imbalanced else 'Not needed'}
  After SMOTE — Class 0: {sum(y_train_bal==0)}, Class 1: {sum(y_train_bal==1)}

HYPERPARAMETER TUNING (GridSearchCV, 5-fold CV, scoring=ROC-AUC):
{bp_str}

MODEL PERFORMANCE:
{final.to_string()}

BEST MODEL : {best_name}
  ROC-AUC  : {final.iloc[0]['ROC_AUC']}%
  F1-Score : {final.iloc[0]['F1']}%
  Recall   : {final.iloc[0]['Recall']}%
  Specificity: {final.iloc[0]['Specificity']}%
  MCC      : {final.iloc[0]['MCC']}

METRICS EXPLANATION:
  Accuracy    — Overall correct predictions
  Precision   — Of predicted +ve, how many are truly +ve
  Recall      — Of true +ve, how many detected (sensitivity)
  Specificity — Of true -ve, how many correctly rejected
  F1-Score    — Harmonic mean of Precision & Recall
  ROC-AUC     — Overall discrimination ability (best for imbalance)
  MCC         — Matthews Correlation Coefficient (balanced metric)
  Overfitting — Train-Test accuracy gap (>5% = warning)

OUTPUT FILES (saved to: {OUT_DIR}):
  missing_data_report.csv     — Null value analysis
  feature_MI.csv              — Mutual Information scores
  feature_Chi2.csv            — Chi-Square scores
  feature_RF.csv              — RF Importance scores
  selected_features.csv       — Final selected features
  feature_MI_chart.png        — MI feature importance bar chart
  hyperparameter_tuning_log.csv — Best params per model
  cm_[ModelName].png          — Confusion matrix per model
  model_results.csv           — Full metrics table
  roc_auc_ranking.png         — ROC-AUC ranking chart (best=red)
  train_vs_test.png           — Overfitting detection
  cross_validation_f1.png     — 5-fold CV F1 with error bars
  shap_beeswarm.png / shap_bar.png — SHAP explanation charts
======================================================================
"""
print(summary)
with open(f"{OUT_DIR}/results_summary.txt", 'w', encoding='utf-8', errors='replace') as f:
    f.write(summary)

print("\n[DONE] DEPRESSION ANALYSIS COMPLETE!")
print(f"  Results saved to: {OUT_DIR}")
