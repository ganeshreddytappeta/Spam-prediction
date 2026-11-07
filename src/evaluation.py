"""
Model Evaluation Module for MailGuard AI.

Calculates comprehensive classification metrics on held-out test data:
- Accuracy
- Precision (Spam class & Macro/Weighted)
- Recall (Spam class & Macro/Weighted)
- F1-Score (Spam class & Macro/Weighted)
- Confusion Matrix (TN, FP, FN, TP)
- Detailed Scikit-learn Classification Report
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: List[str],
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a trained classification pipeline on the held-out test set.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained preprocessing + TF-IDF + classifier pipeline.
    X_test : List[str]
        Raw test email strings (unseen during training/vectorization).
    y_test : np.ndarray
        True binary labels (0 = HAM, 1 = SPAM).
    model_name : str
        Human-readable name of the algorithm.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed metrics, confusion matrix values,
        and classification report.
    """
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    spam_precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    spam_recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    spam_f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    ham_precision = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
    ham_recall = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
    ham_f1 = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            roc_auc = None

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["HAM (0)", "SPAM (1)"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["HAM (0)", "SPAM (1)"],
         digits=4,
        zero_division=0,
    )

    # Composite selection score prioritizing Spam F1 (50%), Spam Precision (25%),
    # Spam Recall (15%), and Overall Accuracy (10%) so selection never relies on accuracy alone.
    composite_score = (
        0.50 * spam_f1
        + 0.25 * spam_precision
        + 0.15 * spam_recall
        + 0.10 * acc
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(spam_precision, 4),
        "recall": round(spam_recall, 4),
        "f1_score": round(spam_f1, 4),
        "ham_precision": round(ham_precision, 4),
        "ham_recall": round(ham_recall, 4),
        "ham_f1_score": round(ham_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc": roc_auc,
        "composite_score": round(float(composite_score), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def select_best_model(
    evaluations: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Select the best performing model based on Spam F1-score, Spam Precision,
    Spam Recall, and Overall Performance rather than accuracy alone.

    Returns
    -------
    Tuple[str, str]
        (best_model_name, detailed_selection_reason)
    """
    if not evaluations:
        raise ValueError("No model evaluations provided for selection.")

    ranked = sorted(
        evaluations.values(),
        key=lambda m: (
            m["f1_score"],
            m["composite_score"],
            m["precision"],
            m["recall"],
            m["accuracy"],
        ),
        reverse=True,
    )
    best = ranked[0]
    best_name = best["model_name"]

    if len(ranked) > 1:
        runner_up = ranked[1]
        reason = (
            f"{best_name} was selected as the production model because it achieved "
            f"the highest Spam F1-Score ({best['f1_score'] * 100:.2f}% vs "
            f"{runner_up['f1_score'] * 100:.2f}% for {runner_up['model_name']}), "
            f"with Spam Precision of {best['precision'] * 100:.2f}%, "
            f"Spam Recall of {best['recall'] * 100:.2f}%, "
            f"Overall Accuracy of {best['accuracy'] * 100:.2f}%, "
            f"only {best['false_positives']} False Positives (legitimate emails marked as spam), "
            f"and {best['false_negatives']} False Negatives (missed spam emails)."
        )
    else:
        reason = (
            f"{best_name} achieved Spam F1-Score of {best['f1_score'] * 100:.2f}% "
            f"and Accuracy of {best['accuracy'] * 100:.2f}%."
        )

    return best_name, reason
