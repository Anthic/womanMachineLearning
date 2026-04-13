"""
=============================================================================
MASTER RUNNER â€” BDHS 2022 ADVANCED ML ANALYSIS (Q1 Standard)
=============================================================================
Runs all 4 scripts in sequence:
  1. advanced_dep_analysis.py   â†’ Depression
  2. advanced_anx_analysis.py   â†’ Anxiety
  3. advanced_disu_analysis.py  â†’ Mental Health (disu)
  4. cross_outcome_comparison.py â†’ Combined comparison

Usage:
  cd "e:\\hafiza mam work\\advanced_analysis"
  python run_all_advanced.py

Prerequisites:
  pip install pandas numpy scikit-learn xgboost imbalanced-learn shap
              matplotlib seaborn scipy pyreadstat
=============================================================================
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("Depression Analysis",      "advanced_dep_analysis.py"),
    ("Anxiety Analysis",         "advanced_anx_analysis.py"),
    ("Mental Health Analysis",   "advanced_disu_analysis.py"),
    ("Cross-Outcome Comparison", "cross_outcome_comparison.py"),
]

print("=" * 70)
print("  BDHS 2022 â€” ADVANCED ANALYSIS MASTER RUNNER")
print("  Q1 Publication Standard")
print("=" * 70)

total_start = time.time()
success_count = 0
failed = []

for title, script_name in SCRIPTS:
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'â”€'*70}")
    print(f"  â–¶  Running: {title}")
    print(f"     Script : {script_path}")
    print(f"{'â”€'*70}")

    if not os.path.exists(script_path):
        print(f"  âœ— Script not found: {script_path}")
        failed.append(title)
        continue

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,   # Show live output
            text=True,
            cwd=SCRIPT_DIR
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  âœ… {title} COMPLETE  ({elapsed:.1f}s)")
            success_count += 1
        else:
            print(f"\n  âœ— {title} FAILED (return code {result.returncode})")
            failed.append(title)
    except Exception as e:
        print(f"\n  âœ— Error running {title}: {e}")
        failed.append(title)

total_elapsed = time.time() - total_start
print(f"\n{'â•'*70}")
print(f"  MASTER RUNNER COMPLETE")
print(f"  Total time : {total_elapsed/60:.1f} minutes")
print(f"  Succeeded  : {success_count}/{len(SCRIPTS)}")
if failed:
    print(f"  Failed     : {', '.join(failed)}")
print(f"{'â•'*70}")

print("""
OUTPUT STRUCTURE:
  e:\\hafiza mam work\\advanced_analysis\\
  â”œâ”€â”€ results_dep\\           â† Depression outputs
  â”‚   â”œâ”€â”€ full_results_with_CI.csv
  â”‚   â”œâ”€â”€ bootstrap_CI_results.csv
  â”‚   â”œâ”€â”€ bootstrap_CI_chart.png
  â”‚   â”œâ”€â”€ clinical_tests.csv
  â”‚   â”œâ”€â”€ roc_curves_all_models.png
  â”‚   â”œâ”€â”€ shap_importance_bar.png
  â”‚   â”œâ”€â”€ shap_beeswarm.png
  â”‚   â””â”€â”€ shap_dependence_top3.png
  â”‚
  â”œâ”€â”€ results_anx\\           â† Anxiety outputs (same structure)
  â”œâ”€â”€ results_disu\\          â† Mental Health outputs (same structure)
  â”‚
  â””â”€â”€ cross_outcome_comparison\\
      â”œâ”€â”€ publication_table_all_outcomes.csv  â† MAIN PAPER TABLE
      â”œâ”€â”€ best_model_per_outcome.csv
      â”œâ”€â”€ cross_outcome_auc_comparison.png    â† KEY FIGURE
      â”œâ”€â”€ auc_heatmap_outcomes_x_models.png
      â”œâ”€â”€ f1_heatmap_outcomes_x_models.png
      â”œâ”€â”€ radar_chart_best_models.png
      â”œâ”€â”€ shap_cross_outcome_heatmap.png
      â””â”€â”€ clinical_tests_all_outcomes.csv
""")
