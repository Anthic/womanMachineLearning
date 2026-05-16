"""
Clean BDHS 2022 mental-health ML analysis.

This script fixes the mismatch issues documented in mismatch_analysis.md:
- uses only the 11 renamed feature variables;
- excludes survey design variables, IDs, raw V-code duplicates, and other targets;
- keeps each outcome in its own folder under new_analysis/<outcome>/results;
- performs train-only feature selection, leakage-safe SMOTE inside CV pipelines,
  hyperparameter tuning, full metrics, confusion matrices, and SHAP for the best model.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "clean_mental_health_data.csv"
ANALYSIS_ROOT = ROOT / "new_analysis"
RANDOM_STATE = 42

FEATURE_COLS = [
    "Division",
    "Residence",
    "Respondent_Education",
    "age_at_First_Birth",
    "Respondent_Age",
    "Parity",
    "Spousal_Age_Gap",
    "Wealth_Index",
    "Internet_Use",
    "Contraceptive_Use",
    "contra_decision_maker",
]

EXCLUDED_COLS = [
    "CASEID",
    "V021",
    "V023",
    "Weight",
    "V171A",
    "V312",
    "V511",
    "V632",
    "V701",
    "V730",
    "Depression",
    "Anxiety",
    "Mental_Health",
]

OUTCOMES = {
    "Depression": {
        "folder": "depression",
        "negative_label": "No Depression",
        "positive_label": "Depression",
        "mapping": "0 = no depression, 1 = depression",
    },
    "Anxiety": {
        "folder": "anxiety",
        "negative_label": "No Anxiety",
        "positive_label": "Anxiety",
        "mapping": "0 = no anxiety, 1 = anxiety",
    },
    "Mental_Health": {
        "folder": "mental_health",
        "negative_label": "No Mental Health Problem",
        "positive_label": "Any Mental Health Problem",
        "mapping": "0 = no problem, values > 0 = any depression/anxiety/combined problem",
    },
}


def make_preprocessor(selected_features: list[str], scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        [("num", numeric_pipeline, selected_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_grids(selected_features: list[str]) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    def pipe(estimator, *, scale: bool) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocess", make_preprocessor(selected_features, scale=scale)),
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                ("model", estimator),
            ]
        )

    grids: dict[str, tuple[Pipeline, dict[str, list]]] = {
        "Logistic Regression": (
            pipe(
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    solver="liblinear",
                ),
                scale=True,
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"), scale=False),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [5, 15],
            },
        ),
        "Random Forest": (
            pipe(
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                scale=False,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        "SVM Linear": (
            pipe(
                CalibratedClassifierCV(
                    LinearSVC(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                    cv=3,
                ),
                scale=True,
            ),
            {"model__estimator__C": [0.01, 0.1, 1.0]},
        ),
        "KNN": (
            pipe(KNeighborsClassifier(), scale=True),
            {
                "model__n_neighbors": [5, 9, 15],
                "model__weights": ["uniform", "distance"],
            },
        ),
    }

    if XGBClassifier is not None:
        grids["XGBoost"] = (
            pipe(
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                ),
                scale=False,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 4],
                "model__subsample": [0.8],
            },
        )
    return grids


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=str)
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df.apply(pd.to_numeric, errors="coerce")


def validate_columns(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLS + list(OUTCOMES))
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    overlap = sorted(set(FEATURE_COLS).intersection(EXCLUDED_COLS))
    if overlap:
        raise ValueError(f"Feature list contains excluded variables: {overlap}")


def target_series(df: pd.DataFrame, outcome: str) -> pd.Series:
    y_raw = df[outcome]
    if outcome == "Mental_Health":
        return (y_raw > 0).astype(float)
    return y_raw


def specificity_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) else 0.0


def save_feature_audit(df: pd.DataFrame, out_dir: Path, outcome: str) -> None:
    audit_rows = []
    for col in df.columns:
        if col in FEATURE_COLS:
            status = "USED_FEATURE"
            reason = "Renamed approved predictor"
        elif col == outcome:
            status = "TARGET"
            reason = "Current outcome"
        elif col in EXCLUDED_COLS:
            status = "EXCLUDED"
            reason = "ID/design variable/raw duplicate/other target"
        else:
            status = "UNUSED"
            reason = "Not in approved renamed feature list"
        audit_rows.append({"Column": col, "Status": status, "Reason": reason})
    pd.DataFrame(audit_rows).to_csv(out_dir / "feature_audit.csv", index=False)


def select_features(X_train: pd.DataFrame, y_train: pd.Series, out_dir: Path) -> list[str]:
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    top_k = min(8, len(FEATURE_COLS))

    mi_scores = mutual_info_classif(X_imp, y_train, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"Feature": X_imp.columns, "MI_Score": mi_scores}).sort_values(
        "MI_Score", ascending=False
    )

    X_shift = X_imp - X_imp.min() + 0.001
    chi_scores, _ = chi2(X_shift, y_train)
    chi_df = pd.DataFrame({"Feature": X_imp.columns, "Chi2_Score": chi_scores}).sort_values(
        "Chi2_Score", ascending=False
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_imp, y_train)
    rf_df = pd.DataFrame({"Feature": X_imp.columns, "RF_Importance": rf.feature_importances_}).sort_values(
        "RF_Importance", ascending=False
    )

    selected = sorted(
        set(mi_df.head(top_k)["Feature"])
        | set(chi_df.head(top_k)["Feature"])
        | set(rf_df.head(top_k)["Feature"])
    )

    mi_df.to_csv(out_dir / "feature_MI.csv", index=False)
    chi_df.to_csv(out_dir / "feature_Chi2.csv", index=False)
    rf_df.to_csv(out_dir / "feature_RF.csv", index=False)
    pd.DataFrame({"Selected_Feature": selected}).to_csv(out_dir / "selected_features.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=mi_df, x="MI_Score", y="Feature", color="#2A6F97")
    plt.title("Mutual Information Feature Scores")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_MI_chart.png", dpi=150)
    plt.close()

    return selected


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    out_dir: Path,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned_models: dict[str, Pipeline] = {}
    tuning_rows = []

    for name, (pipeline, grid) in model_grids(selected_features).items():
        print(f"    Tuning {name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train[selected_features], y_train)
        tuned_models[name] = search.best_estimator_
        tuning_rows.append(
            {
                "Model": name,
                "Best_Params": json.dumps(search.best_params_, sort_keys=True),
                "Best_CV_ROC_AUC": round(search.best_score_, 4),
            }
        )

    tuning_df = pd.DataFrame(tuning_rows).sort_values("Best_CV_ROC_AUC", ascending=False)
    tuning_df.to_csv(out_dir / "hyperparameter_tuning_log.csv", index=False)
    return tuned_models, tuning_df


def evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features: list[str],
    out_dir: Path,
    labels: tuple[str, str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    probabilities = {}

    for name, model in models.items():
        Xtr = X_train[selected_features]
        Xte = X_test[selected_features]
        y_pred_train = model.predict(Xtr)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        probabilities[name] = y_prob

        cv_f1 = cross_val_score(model, Xtr, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        rows.append(
            {
                "Model": name,
                "Train_Accuracy": round(accuracy_score(y_train, y_pred_train) * 100, 2),
                "Test_Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
                "F1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
                "ROC_AUC": round(roc_auc_score(y_test, y_prob) * 100, 2),
                "Specificity": round(specificity_score(y_test, y_pred) * 100, 2),
                "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
                "CV_F1_Mean": round(cv_f1.mean() * 100, 2),
                "CV_F1_Std": round(cv_f1.std() * 100, 2),
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            }
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=list(labels)).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"cm_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    results.to_csv(out_dir / "model_results.csv", index=False)

    plt.figure(figsize=(9, 5))
    auc_rank = results.sort_values("ROC_AUC")
    colors = ["#D62828" if v == auc_rank["ROC_AUC"].max() else "#2A6F97" for v in auc_rank["ROC_AUC"]]
    plt.barh(auc_rank["Model"], auc_rank["ROC_AUC"], color=colors)
    plt.xlabel("ROC-AUC (%)")
    plt.title("ROC-AUC Ranking")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_auc_ranking.png", dpi=150)
    plt.close()

    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, results["Train_Accuracy"], width, label="Train", color="#52796F")
    ax.bar(x + width / 2, results["Test_Accuracy"], width, label="Test", color="#84A98C")
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"], rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.set_title("Train vs Test Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "train_vs_test_accuracy.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results["Model"], results["CV_F1_Mean"], yerr=results["CV_F1_Std"], capsize=5, color="#3A86FF")
    ax.set_xticklabels(results["Model"], rotation=30, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_title("5-Fold Cross-Validation F1")
    plt.tight_layout()
    plt.savefig(out_dir / "cross_validation_f1.png", dpi=150)
    plt.close()

    return results, probabilities


def shap_analysis(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: list[str],
    out_dir: Path,
) -> None:
    for file_name in [
        "shap_bar.png",
        "shap_beeswarm.png",
        "shap_feature_importance.csv",
        "shap_skipped.txt",
    ]:
        path = out_dir / file_name
        if path.exists():
            path.unlink()

    X_explain_raw = X_test[selected_features].sample(n=min(300, len(X_test)), random_state=RANDOM_STATE)

    try:
        fitted_preprocessor = model.named_steps["preprocess"]
        fitted_model = model.named_steps["model"]
        X_explain_arr = fitted_preprocessor.transform(X_explain_raw)
        X_explain = pd.DataFrame(X_explain_arr, columns=selected_features)

        if model_name in {"Random Forest", "XGBoost", "Decision Tree"}:
            explainer = shap.TreeExplainer(fitted_model)
            raw_values = explainer.shap_values(X_explain)
            if isinstance(raw_values, list):
                values = raw_values[1]
            elif getattr(raw_values, "ndim", 0) == 3:
                values = raw_values[:, :, 1]
            else:
                values = raw_values
            shap_values = shap.Explanation(
                values=values,
                base_values=np.zeros(values.shape[0]),
                data=X_explain.values,
                feature_names=selected_features,
            )
        elif model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(fitted_model, X_explain)
            shap_values = explainer(X_explain)
        else:
            raise ValueError(f"Bounded SHAP is configured for tree and logistic models, not {model_name}.")

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": selected_features, "MeanAbsSHAP": mean_abs}).sort_values(
            "MeanAbsSHAP", ascending=False
        )
        shap_df.to_csv(out_dir / "shap_feature_importance.csv", index=False)

        plt.figure()
        shap.plots.bar(shap_values, max_display=len(selected_features), show=False)
        plt.title(f"SHAP Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=len(selected_features), show=False)
        plt.title(f"SHAP Beeswarm - {model_name}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        (out_dir / "shap_skipped.txt").write_text(
            f"SHAP could not be completed for {model_name}: {exc}\n",
            encoding="utf-8",
        )


def write_summary(
    out_dir: Path,
    outcome: str,
    config: dict[str, str],
    n_total: int,
    y: pd.Series,
    selected_features: list[str],
    tuning: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    best = results.iloc[0]
    text = f"""Clean ML analysis summary
Outcome: {outcome}
Target mapping: {config["mapping"]}
Total complete target rows: {n_total}
Positive class count: {int(y.sum())} ({y.mean() * 100:.2f}%)

Approved features used:
{", ".join(FEATURE_COLS)}

Selected features from train-only MI + Chi2 + RF union:
{", ".join(selected_features)}

Excluded from predictors:
{", ".join(EXCLUDED_COLS)}

Best model by test ROC-AUC:
{best["Model"]} | ROC-AUC={best["ROC_AUC"]}% | F1={best["F1"]}% | Recall={best["Recall"]}% | Specificity={best["Specificity"]}% | MCC={best["MCC"]}

Hyperparameter tuning:
{tuning.to_string(index=False)}

Model results:
{results.to_string(index=False)}

Method note:
Feature selection used only the training split. SMOTE was placed inside the imblearn Pipeline, so each GridSearchCV and cross-validation fold oversampled only its training portion. The held-out test set was never oversampled.
"""
    (out_dir / "results_summary.txt").write_text(text, encoding="utf-8")


def run_outcome(df: pd.DataFrame, outcome: str, config: dict[str, str]) -> None:
    print(f"\n=== {outcome} ===")
    out_dir = ANALYSIS_ROOT / config["folder"] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_feature_audit(df, out_dir, outcome)
    work = df[FEATURE_COLS].copy()
    y = target_series(df, outcome)
    keep = y.notna()
    X = work.loc[keep].copy()
    y = y.loc[keep].astype(int)

    missing = pd.DataFrame(
        {
            "Column": FEATURE_COLS + [outcome],
            "Null_Count": list(X.isna().sum()) + [int(df.loc[keep, outcome].isna().sum())],
            "Null_Percent": list((X.isna().mean() * 100).round(2)) + [0.0],
        }
    )
    missing.to_csv(out_dir / "missing_data_report.csv", index=False)

    distribution = y.value_counts().rename_axis("Class").reset_index(name="Count")
    distribution["Percent"] = (distribution["Count"] / len(y) * 100).round(2)
    distribution.to_csv(out_dir / "class_distribution.csv", index=False)
    print(f"  Rows: {len(y):,}; positives: {int(y.sum()):,} ({y.mean() * 100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("  Selecting features on training split only...")
    selected_features = select_features(X_train, y_train, out_dir)
    print(f"  Selected {len(selected_features)} features: {selected_features}")

    print("  Running leakage-safe hyperparameter tuning...")
    models, tuning = tune_models(X_train, y_train, selected_features, out_dir)

    print("  Evaluating tuned models on untouched test split...")
    labels = (config["negative_label"], config["positive_label"])
    results, _ = evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        selected_features,
        out_dir,
        labels,
    )
    print(results[["Model", "ROC_AUC", "F1", "Recall", "Specificity", "MCC"]].to_string(index=False))

    best_name = results.iloc[0]["Model"]
    print(f"  Running SHAP for best model: {best_name}")
    shap_analysis(models[best_name], best_name, X_train, X_test, selected_features, out_dir)

    write_summary(out_dir, outcome, config, len(y), y, selected_features, tuning, results)
    print(f"  Saved: {out_dir}")


def main() -> None:
    print("Loading clean dataset...")
    df = load_data()
    validate_columns(df)
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Using only approved features: {FEATURE_COLS}")

    for outcome, config in OUTCOMES.items():
        run_outcome(df, outcome, config)

    print("\nAll clean analyses complete.")


if __name__ == "__main__":
    main()
