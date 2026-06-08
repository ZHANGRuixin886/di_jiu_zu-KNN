from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_sample_grid(images: np.ndarray, labels: np.ndarray, path: str | Path, max_images: int = 100) -> None:
    count = min(max_images, len(images))
    cols = 10
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx < count:
            ax.imshow(images[idx], cmap="gray")
            ax.set_title(str(labels[idx]), fontsize=8)

    fig.suptitle("MNIST samples")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_class_distribution(label_groups: dict[str, np.ndarray], path: str | Path) -> None:
    records = []
    for split_name, labels in label_groups.items():
        counts = np.bincount(labels.astype(int), minlength=10)
        for digit, count in enumerate(counts):
            records.append({"split": split_name, "digit": str(digit), "count": int(count)})

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot(index="digit", columns="split", values="count")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_k_accuracy_plot(results: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["validation_accuracy"], marker="o")
    ax.set_title("Validation accuracy by K")
    ax.set_xlabel("K")
    ax.set_ylabel("Validation accuracy")
    ax.set_xticks(results["k"].tolist())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(matrix: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks])
    ax.set_yticklabels([str(i) for i in ticks])
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
