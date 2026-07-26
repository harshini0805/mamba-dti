import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(labels, probabilities, threshold=0.5):

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities)

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {

        "accuracy": accuracy_score(labels, predictions),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "specificity": tn / (tn + fp + 1e-8),

        "mcc": matthews_corrcoef(
            labels,
            predictions,
        ),

        "roc_auc": roc_auc_score(
            labels,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            labels,
            probabilities,
        ),

    }


def find_best_threshold(labels, probabilities, metric="mcc"):
    """
    Sweep candidate decision thresholds and return the one that maximizes
    `metric` ("mcc" or "f1") on the given (labels, probabilities).

    IMPORTANT: call this on VALIDATION data only, then apply the returned
    threshold when evaluating the test set. Never fit the threshold on the
    test set itself — that would leak test information into evaluation.

    Returns
    -------
    best_threshold : float
    best_score     : float  (the metric's value at best_threshold)
    """

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities)

    if metric not in ("mcc", "f1"):
        raise ValueError(f"Unsupported metric for threshold search: {metric!r}")

    # Only probability values that actually occur in the data can change
    # which predictions flip, so sweeping the unique values is exact and
    # far cheaper than a fixed grid (e.g. np.linspace).
    candidate_thresholds = np.unique(probabilities)

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in candidate_thresholds:

        predictions = (probabilities >= threshold).astype(int)

        if metric == "mcc":
            score = matthews_corrcoef(labels, predictions)
        else:
            score = f1_score(labels, predictions, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score
