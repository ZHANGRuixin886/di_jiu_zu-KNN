from __future__ import annotations

import gzip
import struct
import zipfile
from pathlib import Path

import numpy as np


MNIST_MEMBERS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_zip_gzip_member(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        if member_name not in zf.namelist():
            raise FileNotFoundError(f"{member_name} not found in {zip_path}")
        with zf.open(member_name, "r") as compressed_file:
            return gzip.decompress(compressed_file.read())


def _parse_idx_images(raw: bytes) -> np.ndarray:
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Invalid image IDX magic number: {magic}")
    data = np.frombuffer(raw, dtype=np.uint8, offset=16)
    return data.reshape(count, rows, cols)


def _parse_idx_labels(raw: bytes) -> np.ndarray:
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Invalid label IDX magic number: {magic}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return labels.reshape(count)


def load_mnist_raw(zip_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST images and labels from the local zip file."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"MNIST zip file does not exist: {zip_path}")

    train_images = _parse_idx_images(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["train_images"]))
    train_labels = _parse_idx_labels(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["train_labels"]))
    test_images = _parse_idx_images(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["test_images"]))
    test_labels = _parse_idx_labels(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["test_labels"]))
    return train_images, train_labels, test_images, test_labels


def split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_size: int = 10000,
    shuffle: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split official MNIST train set into train and validation sets."""
    if validation_size <= 0 or validation_size >= len(images):
        raise ValueError("validation_size must be greater than 0 and smaller than train set size")

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(images))
        images = images[indices]
        labels = labels[indices]

    split_at = len(images) - validation_size
    return images[:split_at], labels[:split_at], images[split_at:], labels[split_at:]


def apply_limit(images: np.ndarray, labels: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    if limit is None:
        return images, labels
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    return images[:limit], labels[:limit]


def preprocess_images(images: np.ndarray, normalize: bool = True, flatten: bool = True) -> np.ndarray:
    """Normalize 0-255 pixels to 0-1 and flatten 28x28 images to 784 features."""
    features = images.astype(np.float32)
    if normalize:
        features = features / 255.0
    if flatten:
        features = features.reshape(features.shape[0], -1)
    return features

