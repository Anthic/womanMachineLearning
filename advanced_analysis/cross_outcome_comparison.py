"""
=============================================================================
CROSS-OUTCOME COMPARISON â€” DEPRESSION vs ANXIETY vs MENTAL HEALTH (disu)
BDHS 2022 â€” Q1 PUBLICATION STANDARD
=============================================================================
This script:
  1. Reads bootstrap CI results from all 3 outcome analyses
  2. Creates unified comparison tables for publication
  3. Plots cross-outcome model performance comparison
  4. Generates a final summary for the research paper methods section

Run AFTER all three advanced analyses complete:
  - advanced_dep_analysis.py
  - advanced_anx_analysis.py
  - advanced_disu_analysis.py
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("  CROSS-OUTCOME COMPARISON â€” BDHS 2022 Mental Health ML")
print("=" * 70)

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = r"e:\hafiza mam work\advanced_analysis"
OUTPUT_DIR = r"e:\hafiza mam work\advanced_analysis\cross_outcome_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOMES = {
    'Depression (dep)': os.path.join(BASE_DIR, 'results_dep',
                                     'full_results_with_CI.csv'),
    'Anxiety (anx)'  : os.path.join(BASE_DIR, 'results_anx',
                                     'full_results_with_CI.csv'),
    'Mental Health (disu)': os.path.join(BASE_DIR, 'results_disu',
                                          'full_results_with_CI.csv'),
}

CLINICAL_PATHS = {
    'Depression (dep)': os.path.join(BASE_DIR, 'results_dep', 'clinical_tests.csv'),
    'Anxiety (anx)'  : os.path.join(BASE_DIR, 'results_anx', 'clinical_tests.csv'),
    'Mental Health (disu)': os.path.join(BASE_DIR, 'results_disu', 'clinical_tests.csv'),
}

SHAP_PATHS = {
    'Depression (dep)': os.path.join(BASE_DIR, 'results_dep',
                                     'shap_feature_importance.csv'),
    'Anxiety (anx)'  : os.path.join(BASE_DIR, 'results_anx',
                                     'shap_feature_importance.csv'),
    'Mental Health (disu)': os.path.join(BASE_DIR, 'results_disu',
                                          'shap_feature_importance.csv'),
}

# Variable labels for display
VAR_LABELS = {
    'v024'           : 'Division',
    'v025'           : 'Residence (Urban/Rural)',
    'v106'           : "Respondent's Education",
    'v701'           : "Husband's Education",
    'age_cat'        : "Current Age",
    'age_fb_cat'     : 'Teenage Pregnancy',
    'parity_cat'     : 'No. of Children',
    'age_diff_cat'   : 'Spousal Age Difference',
    'wealth_cat'     : 'Wealth Index',
    'internet_use'   : 'Mass Media/Internet',
    'contra'         : 'Contraceptive Use',
    'contra_decision': 'Contraceptive Decision',
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Load Results from All 3 Outcomes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 1] Loading results from all 3 outcome analyses...")

all_results  = []
missing_outs = []

for outcome_label, path in OUTCOMES.items():
    if os.path.exists(path):
        df_out = pd.read_csv(path)
        df_out['Outcome'] = outcome_label
        all_results.append(df_out)
        print(f"  âœ“ Loaded: {outcome_label}  ({len(df_out)} models)")
    else:
        print(f"  âœ— MISSING: {path}")
        print(f"    â†’ Run advanced_{outcome_label.split('(')[1].rstrip(')')}"\
              f"_analysis.py first!")
        missing_outs.append(outcome_label)

if not all_results:
    print("\n  ERROR: No results found. Run the 3 advanced analyses first!")
    raise SystemExit(1)

combined_df = pd.concat(all_results, ignore_index=True)
print(f"\n  Total records: {len(combined_df)} "
      f"(across {combined_df['Outcome'].nunique()} outcomes)")

combined_df.to_csv(f"{OUTPUT_DIR}/all_outcomes_combined.csv", index=False)
print(f"  [Saved] all_outcomes_combined.csv")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Best Model Per Outcome Summary Table
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 2] Best Model Per Outcome...")

best_per_outcome = []
for outcome_label in combined_df['Outcome'].unique():
    sub = combined_df[combined_df['Outcome'] == outcome_label]\
            .sort_values('AUC_%', ascending=False)
    best = sub.iloc[0]
    best_per_outcome.append({
        'Outcome'     : outcome_label,
        'Best_Model'  : best['Model'],
        'AUC_%'       : best['AUC_%'],
        'AUC_95CI'    : best['AUC_95CI'],
        'F1_%'        : best['F1_%'],
        'F1_95CI'     : best['F1_95CI'],
        'Recall_%'    : best['Recall_%'],
        'Recall_95CI' : best['Recall_95CI'],
        'Precision_%' : best['Precision_%'],
        'Precision_95CI': best['Precision_95CI'],
    })

best_df = pd.DataFrame(best_per_outcome)
best_df.to_csv(f"{OUTPUT_DIR}/best_model_per_outcome.csv", index=False)

print("\n" + "=" * 90)
print("  BEST MODEL PER OUTCOME (Q1 SUMMARY TABLE)")
print("=" * 90)
print(f"  {'Outcome':<25} {'Best Model':<22} "
      f"{'AUC%':>7} {'AUC 95CI':>20}  "
      f"{'F1%':>6} {'F1 95CI':>18}")
print("  " + "-"*88)
for _, row in best_df.iterrows():
    print(f"  {row['Outcome']:<25} {row['Best_Model']:<22} "
          f"{row['AUC_%']:>7.1f} {row['AUC_95CI']:>20}  "
          f"{row['F1_%']:>6.1f} {row['F1_95CI']:>18}")
print("=" * 90)
print(f"  [Saved] best_model_per_outcome.csv")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Cross-Outcome AUC Comparison Plot
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 3] Cross-Outcome Comparison Charts...")

outcome_colors = {
    'Depression (dep)'    : '#E53935',
    'Anxiety (anx)'       : '#1E88E5',
    'Mental Health (disu)': '#43A047',
}

all_models = combined_df['Model'].unique().tolist()

# â”€â”€ Chart 1: AUC comparison grouped by model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(figsize=(13, 7))
x      = np.arange(len(all_models))
n_out  = len(combined_df['Outcome'].unique())
width  = 0.8 / n_out
offsets= np.linspace(-(n_out-1)/2, (n_out-1)/2, n_out) * width

for i, (outcome, color) in enumerate(outcome_colors.items()):
    sub = combined_df[combined_df['Outcome'] == outcome].set_index('Model')
    means = []
    errs_lo, errs_hi = [], []
    for model in all_models:
        if model in sub.index:
            m = sub.loc[model, 'AUC_%']
            lo = sub.loc[model, 'AUC_CI_lower'] if 'AUC_CI_lower' in sub.columns else m
            hi = sub.loc[model, 'AUC_CI_upper'] if 'AUC_CI_upper' in sub.columns else m
            # Parse CI from string if numeric cols not available
            if 'AUC_CI_lower' not in sub.columns:
                ci_str = sub.loc[model, 'AUC_95CI']
                try:
                    parts = ci_str.strip('[]').split('â€“')
                    lo, hi = float(parts[0]), float(parts[1])
                except Exception:
                    lo, hi = m, m
        else:
            m, lo, hi = 0, 0, 0
        means.append(m)
        errs_lo.append(m - lo)
        errs_hi.append(hi - m)

    bars = ax.bar(x + offsets[i], means, width=width*0.9,
                  label=outcome, color=color, alpha=0.85,
                  edgecolor='black', linewidth=0.7)
    ax.errorbar(x + offsets[i], means,
                yerr=[errs_lo, errs_hi],
                fmt='none', color='black', capsize=4, linewidth=1.5)

ax.set_xlabel('ML Model', fontsize=12, fontweight='bold')
ax.set_ylabel('ROC-AUC (%)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Outcome AUC Comparison â€” BDHS 2022\n'
             '(Depression vs Anxiety vs Mental Health | with 95% Bootstrap CI)',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(all_models, rotation=30, ha='right', fontsize=10)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.set_ylim(max(0, min(combined_df['AUC_%'].min() - 8, 40)),
            min(100, combined_df['AUC_%'].max() + 10))
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cross_outcome_auc_comparison.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Saved] cross_outcome_auc_comparison.png")

# â”€â”€ Chart 2: Heatmap â€” AUC by Model Ã— Outcome â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pivot_auc = combined_df.pivot_table(
    index='Outcome', columns='Model', values='AUC_%')

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(pivot_auc, annot=True, fmt='.1f', cmap='RdYlGn',
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'},
            vmin=max(40, pivot_auc.min().min() - 5),
            vmax=min(100, pivot_auc.max().max() + 2),
            ax=ax, cbar_kws={'label': 'ROC-AUC (%)'})
ax.set_title('ROC-AUC Heatmap â€” All Models Ã— All Outcomes (BDHS 2022)\n'
             '(Green = Higher AUC, Red = Lower AUC)',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('ML Model', fontsize=11, fontweight='bold')
ax.set_ylabel('Mental Health Outcome', fontsize=11, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/auc_heatmap_outcomes_x_models.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Saved] auc_heatmap_outcomes_x_models.png")

# â”€â”€ Chart 3: F1 Score Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pivot_f1 = combined_df.pivot_table(
    index='Outcome', columns='Model', values='F1_%')

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(pivot_f1, annot=True, fmt='.1f', cmap='Blues',
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=ax, cbar_kws={'label': 'F1-Score (%)'})
ax.set_title('F1-Score Heatmap â€” All Models Ã— All Outcomes (BDHS 2022)',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('ML Model', fontsize=11, fontweight='bold')
ax.set_ylabel('Mental Health Outcome', fontsize=11, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/f1_heatmap_outcomes_x_models.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Saved] f1_heatmap_outcomes_x_models.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Best AUC per Outcome â€” Radar Chart
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 4] Best Model Radar Chart (AUC, F1, Recall, Precision)...")

metrics = ['AUC_%', 'F1_%', 'Recall_%', 'Precision_%']
metric_labels = ['AUC', 'F1-Score', 'Recall', 'Precision']

# Use best model per outcome for radar
fig = plt.figure(figsize=(10, 10))
ax_rad = fig.add_subplot(111, polar=True)

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

radar_colors = list(outcome_colors.values())
for idx, (_, row) in enumerate(best_df.iterrows()):
    values = [row[m] for m in metrics]
    values += values[:1]
    ax_rad.plot(angles, values, color=radar_colors[idx],
                linewidth=2.5, linestyle='-', label=row['Outcome'])
    ax_rad.fill(angles, values, color=radar_colors[idx], alpha=0.12)

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
ax_rad.set_ylim(0, 100)
ax_rad.set_yticks([20, 40, 60, 80, 100])
ax_rad.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax_rad.set_title('Best Model Performance â€” Radar Chart\n'
                 '(Depression vs Anxiety vs Mental Health, BDHS 2022)',
                 fontsize=13, fontweight='bold', pad=20)
ax_rad.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/radar_chart_best_models.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Saved] radar_chart_best_models.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Clinical Tests Summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 5] Cross-Outcome Clinical Tests Summary...")

clinical_all = []
for outcome_label, path in CLINICAL_PATHS.items():
    if os.path.exists(path):
        ct = pd.read_csv(path)
        ct['Outcome'] = outcome_label
        clinical_all.append(ct)
    else:
        print(f"  [MISSING] clinical_tests for {outcome_label}")

if clinical_all:
    clinical_df = pd.concat(clinical_all, ignore_index=True)
    clinical_df.to_csv(f"{OUTPUT_DIR}/clinical_tests_all_outcomes.csv", index=False)
    print(f"  [Saved] clinical_tests_all_outcomes.csv")

    print("\n  CLINICAL TESTS SUMMARY:")
    print(f"  {'Outcome':<25} {'Test':<30} {'p-value':>8}  {'Significant':>12}")
    print("  " + "-"*80)
    for _, row in clinical_df.iterrows():
        print(f"  {row['Outcome']:<25} {row['Test']:<30} "
              f"{row['P_Value']:>8.4f}  "
              f"{'â˜… Yes' if row['Significant_p05'] else 'No':>12}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: SHAP Feature Importance Cross-Outcome Comparison
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 6] Cross-Outcome SHAP Feature Importance...")

shap_all = {}
for outcome_label, path in SHAP_PATHS.items():
    if os.path.exists(path):
        sh = pd.read_csv(path)
        sh['Outcome'] = outcome_label
        shap_all[outcome_label] = sh
    else:
        print(f"  [MISSING] SHAP for {outcome_label}")

if shap_all:
    # Build a Feature Ã— Outcome matrix of mean |SHAP|
    all_feats = set()
    for sh in shap_all.values():
        all_feats.update(sh['Feature'].tolist())
    all_feats = sorted(all_feats)

    shap_matrix = pd.DataFrame(index=all_feats)
    for outcome_label, sh in shap_all.items():
        sh_dict = dict(zip(sh['Feature'], sh['Mean_|SHAP|']))
        shap_matrix[outcome_label] = [sh_dict.get(f, 0) for f in all_feats]

    # Normalize to rank (0-1 per outcome)
    shap_norm = shap_matrix.copy()
    for col in shap_norm.columns:
        mn, mx = shap_norm[col].min(), shap_norm[col].max()
        if mx > mn:
            shap_norm[col] = (shap_norm[col] - mn) / (mx - mn)

    # Sort by average importance across outcomes
    shap_norm['Avg'] = shap_norm.mean(axis=1)
    shap_norm = shap_norm.sort_values('Avg', ascending=False).drop(columns='Avg')
    shap_matrix = shap_matrix.loc[shap_norm.index]

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, max(8, len(all_feats) * 0.45)))
    sns.heatmap(shap_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 9, 'weight': 'bold'},
                ax=ax, cbar_kws={'label': 'Normalized SHAP Importance (0â€“1)'})
    ax.set_title('SHAP Feature Importance Across All Outcomes\n'
                 '(Normalized 0â€“1 per outcome | Higher = More Important)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Mental Health Outcome', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature / Predictor', fontsize=11, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_cross_outcome_heatmap.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] shap_cross_outcome_heatmap.png")

    shap_matrix.to_csv(f"{OUTPUT_DIR}/shap_cross_outcome_matrix.csv")
    shap_norm.to_csv(f"{OUTPUT_DIR}/shap_cross_outcome_normalized.csv")
    print(f"  [Saved] shap_cross_outcome_matrix.csv")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 7: Publication-Ready Table (LaTeX + CSV)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 7] Publication-Ready Summary Table...")

pub_table = combined_df[[
    'Outcome', 'Model', 'AUC_%', 'AUC_95CI', 'F1_%', 'F1_95CI',
    'Recall_%', 'Recall_95CI', 'Precision_%', 'Precision_95CI'
]].sort_values(['Outcome', 'AUC_%'], ascending=[True, False])

pub_table.to_csv(f"{OUTPUT_DIR}/publication_table_all_outcomes.csv", index=False)
print(f"  [Saved] publication_table_all_outcomes.csv")

# Also create formatted string table for paper
print("\n" + "=" * 100)
print("  PUBLICATION TABLE â€” ALL OUTCOMES Ã— ALL MODELS (BDHS 2022)")
print("  Metric: ROC-AUC % [95% Bootstrap CI]  |  F1 % [95% CI]")
print("=" * 100)
print(f"  {'Outcome':<22} {'Model':<22} "
      f"{'AUC%':>6}  {'AUC 95CI':>18}  "
      f"{'F1%':>5}  {'F1 95CI':>16}  "
      f"{'Rec%':>5}  {'Pre%':>5}")
print("  " + "-"*100)
for _, row in pub_table.iterrows():
    print(f"  {row['Outcome']:<22} {row['Model']:<22} "
          f"{row['AUC_%']:>6.1f}  {row['AUC_95CI']:>18}  "
          f"{row['F1_%']:>5.1f}  {row['F1_95CI']:>16}  "
          f"{row['Recall_%']:>5.1f}  {row['Precision_%']:>5.1f}")
print("=" * 100)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FINAL SUMMARY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
summary = f"""
=============================================================================
CROSS-OUTCOME COMPARISON SUMMARY â€” BDHS 2022 MENTAL HEALTH ML STUDY
Q1 PUBLICATION STANDARD
=============================================================================
Outcomes Analyzed  : Depression (dep) | Anxiety (anx) | Mental Health (disu)
{f"Missing Outcomes   : {', '.join(missing_outs)}" if missing_outs else "All 3 outcomes present âœ…"}
Bootstrap B        : 1000 iterations per outcome
Statistical Tests  : McNemar's Test + Bootstrap AUC Comparison

â”€â”€â”€ BEST MODELS PER OUTCOME â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
for _, row in best_df.iterrows():
    summary += (f"\n  {row['Outcome']:<25} â†’ {row['Best_Model']:<22}  "
                f"AUC={row['AUC_%']}%  {row['AUC_95CI']}")

summary += f"""

â”€â”€â”€ OUTPUT FILES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  {OUTPUT_DIR}/
  â”œâ”€ all_outcomes_combined.csv              â€” All models Ã— all outcomes
  â”œâ”€ best_model_per_outcome.csv             â€” Best model per outcome
  â”œâ”€ publication_table_all_outcomes.csv     â€” Publication-ready table
  â”œâ”€ clinical_tests_all_outcomes.csv        â€” McNemar + AUC tests
  â”œâ”€ cross_outcome_auc_comparison.png       â€” Grouped bar chart
  â”œâ”€ auc_heatmap_outcomes_x_models.png      â€” AUC heatmap
  â”œâ”€ f1_heatmap_outcomes_x_models.png       â€” F1 heatmap
  â”œâ”€ radar_chart_best_models.png            â€” Radar: best model per outcome
  â”œâ”€ shap_cross_outcome_heatmap.png         â€” SHAP cross-outcome heatmap
  â””â”€ shap_cross_outcome_matrix.csv          â€” Raw SHAP cross-outcome data

â”€â”€â”€ VARIABLE DEFINITIONS (for methods section) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INDEPENDENT VARIABLES:
  v024            = Division
  v025            = Residence (Urban/Rural)
  v106            = Respondent's Education Level
  v701            = Husband's Education Level
  age_cat         = Current Age of Respondent
  age_fb_cat      = Teenage Pregnancy (Age at First Birth)
  parity_cat      = Total Number of Children
  age_diff_cat    = Age Difference of Spouses
  wealth_cat      = Wealth Index
  internet_use    = Mass Media / Internet Use
  contra          = Contraceptive Use
  contra_decision = Decision for Contraceptive Use

SURVEY DESIGN (for weight adjustment only):
  v005            = Sampling Weight (divided by 1,000,000)
  v021            = Primary Sampling Unit (PSU)
  v023            = Stratification Number
=============================================================================
"""
print(summary)
with open(f"{OUTPUT_DIR}/cross_outcome_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"[Saved] cross_outcome_summary.txt")
print(f"\n{'â•'*70}")
print(f"  âœ… CROSS-OUTCOME COMPARISON COMPLETE!")
print(f"  ðŸ“ All outputs: {OUTPUT_DIR}")
print(f"{'â•'*70}\n")
