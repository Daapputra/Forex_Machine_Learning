"""
evaluator.py — Model Evaluation & Reporting (v2.0)
=====================================================
Generates per-class metrics, confusion matrices, feature importance,
and comparison tables. Now handles XGBWrapper and Ensemble transparently.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import config

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y_true: pd.Series,
    model_name: str,
) -> dict:
    """
    Evaluate a single model. All models (RF, XGBWrapper, Ensemble, LR)
    now return original labels (-1, 0, 1) from predict(), so no special handling.
    """
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)

    labels = sorted(y_true.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    f1_macro = np.mean(f1)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    for i, label in enumerate(labels):
        label_name = config.LABEL_NAMES.get(label, str(label))
        metrics["per_class"][label_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }

    logger.info(f"\n{model_name} -- Accuracy: {acc:.4f}, F1 Macro: {f1_macro:.4f}")
    for label_name, m in metrics["per_class"].items():
        logger.info(f"  {label_name}: P={m['precision']:.3f} R={m['recall']:.3f} "
                    f"F1={m['f1']:.3f} (n={m['support']})")

    return metrics


def evaluate_rule_based(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Evaluate rule-based baseline predictions."""
    return _compute_metrics(y_true, y_pred, "Rule-Based (RSI)")


def _compute_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    labels = sorted(y_true.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    f1_macro = np.mean(f1)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    for i, label in enumerate(labels):
        label_name = config.LABEL_NAMES.get(label, str(label))
        metrics["per_class"][label_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }

    logger.info(f"\n{model_name} -- Accuracy: {acc:.4f}, F1 Macro: {f1_macro:.4f}")
    for label_name, m in metrics["per_class"].items():
        logger.info(f"  {label_name}: P={m['precision']:.3f} R={m['recall']:.3f} "
                    f"F1={m['f1']:.3f} (n={m['support']})")

    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred,
    model_name: str,
    output_dir: str = None,
):
    """Plot and save confusion matrix."""
    output_dir = output_dir or config.OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    labels = sorted(y_true.unique())
    label_names = [config.LABEL_NAMES.get(l, str(l)) for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix -- {model_name}",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {filepath}")


def plot_feature_importance(
    model,
    feature_cols: list,
    model_name: str,
    top_n: int = 15,
    output_dir: str = None,
):
    """Plot top N feature importances."""
    output_dir = output_dir or config.OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} does not have feature_importances_. Skipping.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), top_importances[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance -- {model_name}")

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance: {filepath}")

    return list(zip(top_features, top_importances.tolist()))


def generate_comparison_table(all_metrics: Dict[str, dict]) -> pd.DataFrame:
    """Generate a comparison table of all models."""
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "F1 Macro": metrics["f1_macro"],
        }
        for label_name, m in metrics["per_class"].items():
            row[f"P_{label_name}"] = m["precision"]
            row[f"R_{label_name}"] = m["recall"]
            row[f"F1_{label_name}"] = m["f1"]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


def evaluate_all_models(
    trained_models: Dict[str, Any],
    y_val: pd.Series,
    y_test: pd.Series,
    feature_cols: list,
    rule_based_val_pred: pd.Series = None,
    rule_based_test_pred: pd.Series = None,
) -> Dict[str, dict]:
    """Evaluate all trained models on validation and test sets."""
    all_results = {}

    for model_name, model_info in trained_models.items():
        model = model_info["model"]
        X_val = model_info["X_val"]
        X_test = model_info["X_test"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")

        # Validation set
        val_metrics = evaluate_model(model, X_val, y_val, f"{model_name} (Val)")
        y_val_pred = model.predict(X_val)
        plot_confusion_matrix(y_val, y_val_pred, f"{model_name}_val")

        # Test set
        test_metrics = evaluate_model(model, X_test, y_test, f"{model_name} (Test)")
        y_test_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_test_pred, f"{model_name}_test")

        # Feature importance
        top_features = None
        if model_name != "logistic_regression":
            top_features = plot_feature_importance(model, feature_cols, model_name)

        all_results[model_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_features": top_features,
        }

    # Rule-based baseline
    if rule_based_val_pred is not None and rule_based_test_pred is not None:
        logger.info(f"\n{'='*60}")
        logger.info("Evaluating: Rule-Based Baseline (RSI)")
        logger.info(f"{'='*60}")

        val_metrics = evaluate_rule_based(y_val, rule_based_val_pred)
        test_metrics = evaluate_rule_based(y_test, rule_based_test_pred)

        all_results["rule_based_rsi"] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_features": None,
        }

    return all_results


def save_evaluation_report(
    all_results: Dict[str, dict],
    output_dir: str = None,
):
    """Save evaluation comparison as CSV."""
    output_dir = output_dir or config.OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    test_metrics = {
        name: res["test_metrics"] for name, res in all_results.items()
    }
    comparison_df = generate_comparison_table(test_metrics)

    csv_path = os.path.join(output_dir, "model_comparison_test.csv")
    comparison_df.to_csv(csv_path, float_format="%.4f")

    logger.info(f"\n{'='*60}")
    logger.info("MODEL COMPARISON (Test Set)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{comparison_df.to_string()}")
    logger.info(f"\nSaved to: {csv_path}")

    val_metrics = {
        name: res["val_metrics"] for name, res in all_results.items()
    }
    val_comparison_df = generate_comparison_table(val_metrics)
    val_csv_path = os.path.join(output_dir, "model_comparison_val.csv")
    val_comparison_df.to_csv(val_csv_path, float_format="%.4f")

    return comparison_df
