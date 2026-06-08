from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_K_VALUES, DEFAULT_MNIST_ZIP, DEFAULT_OUTPUT_DIR
from .data import apply_limit, load_mnist_raw, preprocess_images, split_train_validation
from .model import build_knn, evaluate_classifier, tune_k_values
from .utils import ensure_dir, save_json, setup_logger
from .visualize import (
    save_class_distribution,
    save_confusion_matrix,
    save_k_accuracy_plot,
    save_sample_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate KNN on local MNIST data.")
    parser.add_argument("--data-zip", type=Path, default=DEFAULT_MNIST_ZIP, help="Path to mnist.zip")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help="K values to compare")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan"], help="Distance metric")
    parser.add_argument("--weights", default="uniform", choices=["uniform", "distance"], help="Voting weights")
    parser.add_argument("--validation-size", type=int, default=10000, help="Validation size from official train split")
    parser.add_argument("--train-limit", type=int, default=None, help="Use only first N train samples")
    parser.add_argument("--val-limit", type=int, default=None, help="Use only first N validation samples")
    parser.add_argument("--test-limit", type=int, default=None, help="Use only first N test samples")
    parser.add_argument("--quick", action="store_true", help="Use a smaller subset for quick debugging")
    parser.add_argument("--shuffle-train-val", action="store_true", help="Shuffle official train set before split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when shuffling")
    parser.add_argument("--batch-size", type=int, default=1000, help="Prediction batch size")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for scikit-learn KNN; use -1 for all cores")
    parser.add_argument("--no-final-refit", action="store_true", help="Do not refit on train+validation before testing")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figures")
    parser.add_argument("--save-model", action="store_true", help="Save final KNN model with joblib")
    return parser.parse_args()


def _apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    if args.train_limit is None:
        args.train_limit = 10000
    if args.val_limit is None:
        args.val_limit = 2000
    if args.test_limit is None:
        args.test_limit = 2000


def _write_text_report(
    path: Path,
    args: argparse.Namespace,
    run_id: str,
    train_count: int,
    val_count: int,
    test_count: int,
    best_k: int,
    validation_results: pd.DataFrame,
    test_accuracy: float,
    classification_report_text: str,
    total_seconds: float,
) -> None:
    lines = [
        "MNIST 手写数字识别 - KNN 实验报告",
        f"运行编号: {run_id}",
        "",
        "1. 数据处理",
        f"- 数据文件: {args.data_zip}",
        f"- 训练集样本数: {train_count}",
        f"- 验证集样本数: {val_count}",
        f"- 测试集样本数: {test_count}",
        "- 预处理方法: 将像素值从 0-255 归一化到 0-1，并将 28x28 图片展平成 784 维向量。",
        "",
        "2. 模型设计",
        "- 模型: KNeighborsClassifier",
        f"- 距离度量: {args.metric}",
        f"- 投票权重: {args.weights}",
        f"- 候选 K 值: {args.k_values}",
        f"- 验证集上选择的最优 K 值: {best_k}",
        "",
        "3. 验证集结果",
        validation_results.to_string(index=False),
        "",
        "4. 测试集结果",
        f"- 测试集准确率: {test_accuracy:.4f}",
        "",
        classification_report_text,
        "",
        f"总耗时秒数: {total_seconds:.2f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _apply_quick_defaults(args)
    if args.n_jobs == 1:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    run_id = datetime.now().strftime("knn_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(args.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures")
    models_dir = ensure_dir(output_dir / "models")

    logger = setup_logger(logs_dir / f"{run_id}.log")
    total_start = perf_counter()

    logger.info("Run ID: %s", run_id)
    logger.info("Load MNIST from %s", args.data_zip)
    raw_train_images, raw_train_labels, raw_test_images, raw_test_labels = load_mnist_raw(args.data_zip)

    train_images, y_train, val_images, y_val = split_train_validation(
        raw_train_images,
        raw_train_labels,
        validation_size=args.validation_size,
        shuffle=args.shuffle_train_val,
        seed=args.seed,
    )
    test_images, y_test = raw_test_images, raw_test_labels

    train_images, y_train = apply_limit(train_images, y_train, args.train_limit)
    val_images, y_val = apply_limit(val_images, y_val, args.val_limit)
    test_images, y_test = apply_limit(test_images, y_test, args.test_limit)

    logger.info("Data split: train=%s validation=%s test=%s", len(y_train), len(y_val), len(y_test))

    if not args.no_plots:
        logger.info("Save data visualization figures")
        save_sample_grid(train_images, y_train, figures_dir / f"{run_id}_samples.png")
        save_class_distribution(
            {"train": y_train, "validation": y_val, "test": y_test},
            figures_dir / f"{run_id}_class_distribution.png",
        )

    logger.info("Preprocess images: normalize and flatten")
    x_train = preprocess_images(train_images)
    x_val = preprocess_images(val_images)
    x_test = preprocess_images(test_images)

    logger.info("Tune K values: %s", args.k_values)
    validation_results, best_k = tune_k_values(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        k_values=args.k_values,
        metric=args.metric,
        weights=args.weights,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        logger=logger,
    )
    validation_df = pd.DataFrame(validation_results)
    validation_csv = reports_dir / f"{run_id}_validation_results.csv"
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8-sig")
    logger.info("Best K=%s", best_k)

    if not args.no_plots:
        save_k_accuracy_plot(validation_df, figures_dir / f"{run_id}_k_accuracy.png")

    if args.no_final_refit:
        x_final_train, y_final_train = x_train, y_train
        final_train_note = "train only"
    else:
        x_final_train = np.concatenate([x_train, x_val], axis=0)
        y_final_train = np.concatenate([y_train, y_val], axis=0)
        final_train_note = "train + validation"

    logger.info("Fit final model with K=%s on %s samples (%s)", best_k, len(y_final_train), final_train_note)
    final_model = build_knn(k=best_k, metric=args.metric, weights=args.weights, n_jobs=args.n_jobs)
    final_model.fit(x_final_train, y_final_train)

    logger.info("Evaluate final model on test set")
    evaluation = evaluate_classifier(final_model, x_test, y_test, batch_size=args.batch_size)
    test_accuracy = evaluation["accuracy"]
    y_pred = evaluation["y_pred"]
    confusion = evaluation["confusion_matrix"]
    logger.info("Test accuracy=%.4f", test_accuracy)

    predictions_df = pd.DataFrame({"y_true": y_test.astype(int), "y_pred": y_pred.astype(int)})
    predictions_df.to_csv(reports_dir / f"{run_id}_test_predictions.csv", index=False, encoding="utf-8-sig")

    if not args.no_plots:
        save_confusion_matrix(confusion, figures_dir / f"{run_id}_confusion_matrix.png")

    metadata = {
        "run_id": run_id,
        "data_zip": str(args.data_zip),
        "train_count": int(len(y_train)),
        "validation_count": int(len(y_val)),
        "test_count": int(len(y_test)),
        "k_values": args.k_values,
        "best_k": int(best_k),
        "metric": args.metric,
        "weights": args.weights,
        "test_accuracy": float(test_accuracy),
        "final_train_note": final_train_note,
        "quick": bool(args.quick),
    }
    save_json(metadata, reports_dir / f"{run_id}_summary.json")

    if args.save_model:
        model_path = models_dir / f"{run_id}_knn.joblib"
        joblib.dump({"model": final_model, "metadata": metadata}, model_path)
        logger.info("Saved model to %s", model_path)

    total_seconds = perf_counter() - total_start
    _write_text_report(
        reports_dir / f"{run_id}_report.txt",
        args=args,
        run_id=run_id,
        train_count=len(y_train),
        val_count=len(y_val),
        test_count=len(y_test),
        best_k=best_k,
        validation_results=validation_df,
        test_accuracy=test_accuracy,
        classification_report_text=evaluation["classification_report_text"],
        total_seconds=total_seconds,
    )
    logger.info("Finished in %.2fs", total_seconds)
