from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MNIST_ZIP = PROJECT_ROOT / "mnist.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_K_VALUES = (1, 3, 5, 7, 9)
CLASS_NAMES = [str(i) for i in range(10)]

