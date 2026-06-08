from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def build_knn(
    k: int,
    metric: str = "euclidean",
    weights: str = "uniform",
    n_jobs: int = 1,
) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        weights=weights,
        algorithm="brute",
        n_jobs=n_jobs,
    )


def predict_in_batches(model: KNeighborsClassifier, x: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    if batch_size <= 0:
        return model.predict(x)

    predictions = []
    for start in tqdm(range(0, len(x), batch_size), desc="Predict", leave=False, ascii=True):
        end = min(start + batch_size, len(x))
        predictions.append(model.predict(x[start:end]))
    return np.concatenate(predictions)


def tune_k_values(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    k_values: list[int],
    metric: str = "euclidean",
    weights: str = "uniform",
    batch_size: int = 1000,
    n_jobs: int = -1,
    logger=None,
) -> tuple[list[dict], int]:
    results: list[dict] = []
    best_k = k_values[0]
    best_accuracy = -1.0

    for k in k_values:
        if logger:
            logger.info("Start validation for K=%s", k)
        start_time = perf_counter()
        model = build_knn(k=k, metric=metric, weights=weights, n_jobs=n_jobs)
        model.fit(x_train, y_train)
        y_pred = predict_in_batches(model, x_val, batch_size=batch_size)
        elapsed = perf_counter() - start_time
        accuracy = accuracy_score(y_val, y_pred)
        result = {
            "k": k,
            "metric": metric,
            "weights": weights,
            "validation_accuracy": accuracy,
            "elapsed_seconds": elapsed,
        }
        results.append(result)

        if logger:
            logger.info("K=%s validation accuracy=%.4f elapsed=%.2fs", k, accuracy, elapsed)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return results, best_k


def evaluate_classifier(
    model: KNeighborsClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 1000,
) -> dict:
    y_pred = predict_in_batches(model, x_test, batch_size=batch_size)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report_text": classification_report(y_test, y_pred, digits=4, zero_division=0),
        "classification_report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }
