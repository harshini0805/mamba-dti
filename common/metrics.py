"""
Metric computation for binary DTI classification.

Shared across all architectures.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(labels: list | np.ndarray, probs: list | np.ndarray) -> dict:
    """
    Compute all evaluation metrics for binary classification.

    Args:
        labels: List/array of ground-truth binary labels {0, 1}
        probs: List/array of predicted probabilities [0, 1]

    Returns:
        dict with keys:
          - accuracy: (TP + TN) / N
          - precision: TP / (TP + FP)
          - recall: TP / (TP + FN) [sensitivity]
          - specificity: TN / (TN + FP)
          - mcc: Matthews correlation coefficient
          - roc_auc: ROC-AUC (area under ROC curve)
          - pr_auc: PR-AUC (area under precision-recall curve)
    """
    # Binarize at 0.5 threshold
    preds = (np.array(probs) >= 0.5).astype(int)
    labels = np.array(labels, dtype=int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "mcc": matthews_corrcoef(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
    }
