"""Leakage-aware model training and evaluation for incident triage."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "site",
    "incident_type",
    "shift",
    "badge_access",
    "cctv_alert",
    "visitor",
    "month",
    "day_of_week",
    "is_weekend",
]
CATEGORICAL_FEATURES = [
    "site",
    "incident_type",
    "shift",
    "badge_access",
    "cctv_alert",
    "visitor",
    "day_of_week",
]
NUMERIC_FEATURES = ["month", "is_weekend"]
TARGET = "high_risk_target"
LEAKAGE_EXCLUSIONS = ["severity", "outcome", "response_time_min", "incident_id"]


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by ordered rows: 60% train, 20% validation, 20% latest test."""
    ordered = df.sort_values(["date", "incident_id"]).reset_index(drop=True)
    train_end = int(len(ordered) * 0.60)
    validation_end = int(len(ordered) * 0.80)
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _models() -> dict[str, object]:
    return {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Prioritize recall while requiring at least 0.70 precision on validation data."""
    candidates = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        f1 = f1_score(y_true, predicted, zero_division=0)
        if precision >= 0.70:
            candidates.append((recall, f1, -threshold, threshold))
    if not candidates:
        return 0.50
    return float(max(candidates)[3])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    result = pd.DataFrame({"feature": names, "importance": values})
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def train_and_save(df: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = chronological_split(df)

    comparisons = []
    fitted = {}
    thresholds = {}
    for name, estimator in _models().items():
        pipeline = Pipeline([("preprocessor", _preprocessor()), ("model", estimator)])
        pipeline.fit(train[FEATURES], train[TARGET])
        validation_prob = pipeline.predict_proba(validation[FEATURES])[:, 1]
        threshold = 0.50 if name == "Dummy Baseline" else choose_threshold(validation[TARGET], validation_prob)
        test_prob = pipeline.predict_proba(test[FEATURES])[:, 1]
        metrics = evaluate(test[TARGET], test_prob, threshold)
        comparisons.append({"model": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        fitted[name] = pipeline
        thresholds[name] = threshold

    comparison = pd.DataFrame(comparisons).sort_values(["recall", "pr_auc", "precision"], ascending=False)
    eligible = comparison[comparison["model"] != "Dummy Baseline"]
    best_name = eligible.iloc[0]["model"]

    # Refit the selected specification on all pre-test history; preserve untouched latest 20% for evaluation.
    development = pd.concat([train, validation]).sort_values(["date", "incident_id"])
    final_model = Pipeline([("preprocessor", _preprocessor()), ("model", _models()[best_name])])
    final_model.fit(development[FEATURES], development[TARGET])
    final_prob = final_model.predict_proba(test[FEATURES])[:, 1]
    final_metrics = evaluate(test[TARGET], final_prob, thresholds[best_name])

    predictions = test[["incident_id", "date", "site", "incident_type", "severity", TARGET]].copy()
    predictions["predicted_probability"] = final_prob.round(6)
    predictions["predicted_high_risk"] = (final_prob >= thresholds[best_name]).astype(int)
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")

    importance = _feature_importance(final_model)
    metadata = {
        "selected_model": best_name,
        "target_definition": "1 = High or Critical severity; 0 = Low or Medium severity",
        "features": FEATURES,
        "leakage_exclusions": LEAKAGE_EXCLUSIONS,
        "split_strategy": "Chronological 60% train, 20% validation, 20% latest test; final model refit on first 80%.",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_date_range": [train["date"].min().strftime("%Y-%m-%d"), train["date"].max().strftime("%Y-%m-%d")],
        "validation_date_range": [validation["date"].min().strftime("%Y-%m-%d"), validation["date"].max().strftime("%Y-%m-%d")],
        "test_date_range": [test["date"].min().strftime("%Y-%m-%d"), test["date"].max().strftime("%Y-%m-%d")],
        "decision_threshold": round(float(thresholds[best_name]), 4),
        "threshold_policy": "Chosen on validation data to maximize recall subject to precision >= 0.70.",
        "test_metrics": final_metrics,
        "caution": (
            "Incident type strongly encodes the supplied severity labels. Performance is therefore an assessment-sample proof of concept, "
            "not evidence of production generalization. Validate prospectively on future operational data before deployment."
        ),
    }

    joblib.dump(final_model, output_dir / "incident_risk_model.joblib")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

