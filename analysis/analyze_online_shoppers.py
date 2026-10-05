"""Reproducible analysis for Personal Portfolio Project Two.

Predict whether an e-commerce session ends in a purchase using the UCI Online
Shoppers Purchasing Intention dataset. Run from the repository root:

    python3 analysis/analyze_online_shoppers.py

The script writes publication-ready figures to assets/project-two and a compact
machine-readable results file to data/project2_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
ASSET_DIR = ROOT / "assets" / "project-two"
RESULTS_PATH = ROOT / "data" / "project2_summary.json"
RANDOM_STATE = 42

INK = "#13263a"
ACCENT = "#d46a3a"
SLATE = "#65717d"
PALE = "#dbe3e7"
PAPER = "#fbfaf7"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def style_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": PALE,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "text.color": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "grid.color": PALE,
            "grid.linewidth": 0.8,
        }
    )


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model, numeric_features: list[str] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("prepare", build_preprocessor(numeric_features or NUMERIC_FEATURES)),
            ("model", model),
        ]
    )


def evaluate(name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prediction = estimator.predict(X_test)
    probability = estimator.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
        "prediction": prediction,
        "probability": probability,
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
    }


def save_data_story(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    counts = df["Revenue"].value_counts().reindex([False, True])
    bars = axes[0].bar(["No purchase", "Purchase"], counts, color=[PALE, ACCENT], edgecolor=INK)
    axes[0].set_title("Purchases are the minority outcome", loc="left")
    axes[0].set_ylabel("Sessions")
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 180, f"{value:,}\n({value / len(df):.1%})", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylim(0, counts.max() * 1.18)

    medians = (
        df.assign(Outcome=np.where(df["Revenue"], "Purchase", "No purchase"))
        .groupby("Outcome")[["ProductRelated", "Administrative"]]
        .median()
        .reindex(["No purchase", "Purchase"])
    )
    x = np.arange(len(medians))
    width = 0.34
    axes[1].bar(x - width / 2, medians["ProductRelated"], width, label="Product pages", color=INK)
    axes[1].bar(x + width / 2, medians["Administrative"], width, label="Administrative pages", color=ACCENT)
    axes[1].set_xticks(x, medians.index)
    axes[1].set_ylabel("Median pages viewed")
    axes[1].set_title("Purchasing sessions show deeper browsing", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y")
    axes[1].set_axisbelow(True)
    fig.suptitle("What the session data reveals before modeling", x=0.08, y=1.03, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(ASSET_DIR / "data-story.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(results: list[dict]) -> None:
    metric_names = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    labels = ["Balanced accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"]
    names = [result["model"] for result in results]
    colors = [PALE, ACCENT, INK]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    x = np.arange(len(metric_names))
    width = 0.24
    for i, (result, color) in enumerate(zip(results, colors)):
        values = [result[metric] for metric in metric_names]
        bars = ax.bar(x + (i - 1) * width, values, width, label=result["model"], color=color, edgecolor=INK, linewidth=0.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.014, f"{value:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test-set score")
    ax.set_title("The tuned random forest gives the strongest overall ranking", loc="left", pad=18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "model-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_curves(results: list[dict], y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = [SLATE, ACCENT, INK]
    for result, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, result["probability"])
        precision, recall, _ = precision_recall_curve(y_test, result["probability"])
        axes[0].plot(fpr, tpr, color=color, linewidth=2.4, label=f"{result['model']} ({result['roc_auc']:.2f})")
        axes[1].plot(recall, precision, color=color, linewidth=2.4, label=f"{result['model']} ({result['average_precision']:.2f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALE, linewidth=2)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1].axhline(y_test.mean(), linestyle="--", color=PALE, linewidth=2, label=f"Purchase rate ({y_test.mean():.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    for ax in axes:
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Ranking quality matters more than raw accuracy", x=0.08, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(ASSET_DIR / "roc-pr-curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(14, 4.3))
    for result, ax in zip(results, axes):
        ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["No purchase", "Purchase"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=",d"
        )
        ax.set_title(result["model"], fontweight="bold")
    fig.suptitle("Where each model is right—and wrong", x=0.06, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(ASSET_DIR / "confusion-matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {"feature": X_test.columns, "importance": importance.importances_mean, "std": importance.importances_std}
    ).sort_values("importance", ascending=False)
    top = frame.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=ACCENT, edgecolor=INK, capsize=3)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Decrease in test PR AUC after shuffling")
    ax.set_title("Page value dominates, with exit behavior and depth adding signal", loc="left", pad=16)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "feature-importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return frame


def subgroup_metrics(result: dict, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    rows = []
    check = X_test[["VisitorType"]].copy()
    check["actual"] = y_test.to_numpy()
    check["predicted"] = result["prediction"]
    for group, values in check.groupby("VisitorType", observed=True):
        if values["actual"].sum() == 0:
            continue
        rows.append(
            {
                "visitor_type": str(group),
                "sessions": int(len(values)),
                "purchases": int(values["actual"].sum()),
                "recall": float(recall_score(values["actual"], values["predicted"], zero_division=0)),
                "precision": float(precision_score(values["actual"], values["predicted"], zero_division=0)),
            }
        )
    return rows


def main() -> None:
    style_plot()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if df.shape != (12330, 18):
        raise ValueError(f"Unexpected source shape: {df.shape}")
    if df.isna().any().any():
        raise ValueError("The source should contain no missing values.")

    X = df.drop(columns="Revenue")
    y = df["Revenue"].astype(bool)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline = make_pipeline(DummyClassifier(strategy="prior"))
    baseline.fit(X_train, y_train)

    logistic_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        {"model__C": [0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    logistic_search.fit(X_train, y_train)

    forest_search = GridSearchCV(
        make_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        {
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.5],
        },
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    forest_search.fit(X_train, y_train)

    results = [
        evaluate("Majority baseline", baseline, X_test, y_test),
        evaluate("Logistic regression", logistic_search.best_estimator_, X_test, y_test),
        evaluate("Random forest", forest_search.best_estimator_, X_test, y_test),
    ]

    behavior_features = [feature for feature in NUMERIC_FEATURES if feature != "PageValues"] + CATEGORICAL_FEATURES
    behavior_numeric = [feature for feature in NUMERIC_FEATURES if feature != "PageValues"]
    behavior_model = make_pipeline(
        clone(forest_search.best_estimator_.named_steps["model"]),
        numeric_features=behavior_numeric,
    )
    behavior_model.fit(X_train[behavior_features], y_train)
    behavior_result = evaluate("Random forest without PageValues", behavior_model, X_test[behavior_features], y_test)

    save_data_story(df)
    save_model_comparison(results)
    save_curves(results, y_test)
    save_confusion_matrices(results)
    importance = save_feature_importance(forest_search.best_estimator_, X_test, y_test)

    safe_results = []
    for result in results:
        safe_results.append(
            {
                key: (round(float(value), 4) if isinstance(value, (float, np.floating)) else value)
                for key, value in result.items()
                if key not in {"prediction", "probability"}
            }
        )

    summary = {
        "source": {
            "name": "UCI Online Shoppers Purchasing Intention Dataset",
            "doi": "10.24432/C5F88Q",
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "missing_values": int(df.isna().sum().sum()),
            "exact_duplicate_rows": int(df.duplicated().sum()),
        },
        "target": {
            "purchases": int(y.sum()),
            "non_purchases": int((~y).sum()),
            "purchase_rate": round(float(y.mean()), 4),
        },
        "exploratory_summary": {
            feature: {
                "median_no_purchase": round(float(df.loc[~y, feature].median()), 4),
                "median_purchase": round(float(df.loc[y, feature].median()), 4),
                "overall_95th_percentile": round(float(df[feature].quantile(0.95)), 4),
                "overall_maximum": round(float(df[feature].max()), 4),
            }
            for feature in [
                "Administrative",
                "ProductRelated",
                "ProductRelated_Duration",
                "ExitRates",
                "PageValues",
            ]
        },
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "strategy": "80/20 stratified holdout; 5-fold stratified CV on training data only",
        },
        "best_parameters": {
            "logistic_regression": logistic_search.best_params_,
            "random_forest": forest_search.best_params_,
        },
        "test_metrics": safe_results,
        "pagevalues_sensitivity": {
            key: round(float(value), 4)
            for key, value in behavior_result.items()
            if key in {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"}
        },
        "top_permutation_features": [
            {"feature": row.feature, "importance": round(float(row.importance), 4)}
            for row in importance.head(10).itertuples()
        ],
        "visitor_type_error_check": subgroup_metrics(results[-1], X_test, y_test),
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    for result in safe_results:
        print(result)
    print("Best logistic parameters:", logistic_search.best_params_)
    print("Best random-forest parameters:", forest_search.best_params_)
    print("Without PageValues:", summary["pagevalues_sensitivity"])
    print("Wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
