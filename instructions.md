# MNIST KNN 项目代码说明文档

本文档面向 Python 初学者，解释 `C:\Users\Lenovo\Desktop\HDR` 项目中各个代码文件的作用、运行关系和关键代码原理。阅读顺序建议如下：

1. 先看“项目整体运行流程”
2. 再看 `src/knn_mnist` 下的核心模块
3. 最后看 `scripts` 启动脚本和 `app` 交互界面

## 一、项目整体运行流程

本项目完成的是 MNIST 手写数字识别中的 KNN 传统机器学习方法。完整流程可以理解为：

```text
mnist.zip
  -> 读取 IDX 格式图片和标签
  -> 划分训练集、验证集、测试集
  -> 像素归一化
  -> 28x28 图片展平成 784 维向量
  -> 使用不同 K 值训练和验证 KNN
  -> 选择验证集效果最好的 K
  -> 在测试集上评估
  -> 保存日志、报告、图表、模型
  -> Streamlit 界面加载模型进行交互演示
```

KNN 的核心思想是：如果一张新图片和训练集中某些图片最相似，那么它大概率属于这些相似图片中出现最多的类别。例如，一张新图片和训练集里最近的 5 张图片分别是 `7, 7, 7, 1, 9`，那么模型会预测为 `7`。

## 二、目录结构说明

```text
HDR/
  README.md
  requirements.txt
  environment.yml
  mnist.zip
  instructions.md
  scripts/
    train_knn.py
    run_streamlit.bat
  app/
    streamlit_app.py
  src/
    knn_mnist/
      __init__.py
      config.py
      data.py
      model.py
      train.py
      utils.py
      visualize.py
  outputs/
    logs/
    reports/
    figures/
    models/
```

主要代码都在 `src/knn_mnist` 中。`scripts` 是运行入口，`app` 是 Streamlit 交互界面，`outputs` 是程序运行后自动生成的结果。

## 三、`src/knn_mnist/__init__.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\__init__.py
```

代码内容很短：

```python
"""KNN implementation for the MNIST handwritten digit project."""

__version__ = "1.0.0"
```

逐行解释：

```python
"""KNN implementation for the MNIST handwritten digit project."""
```

这是模块说明文字，也叫 docstring。它不会影响程序逻辑，主要用于说明这个包是做什么的。

```python
__version__ = "1.0.0"
```

定义当前代码包的版本号。以后如果代码升级，可以把版本号改成 `1.1.0`、`2.0.0` 等。

这个文件还有一个隐含作用：只要目录里有 `__init__.py`，Python 就会把 `knn_mnist` 当作一个可以导入的包。因此其他文件可以写：

```python
from knn_mnist.data import load_mnist_raw
```

## 四、`src/knn_mnist/config.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\config.py
```

这个文件负责集中保存项目默认配置。

代码：

```python
from pathlib import Path
```

导入 `Path`。`Path` 是 Python 标准库里处理文件路径的工具，比直接拼接字符串更安全。例如：

```python
Path("outputs") / "figures"
```

会自动得到正确路径。

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

这行用于计算项目根目录。

分开理解：

```text
__file__                    当前文件 config.py 的路径
Path(__file__)              把字符串路径转换成 Path 对象
.resolve()                  转成绝对路径
.parents[2]                 向上找两级目录
```

因为 `config.py` 在：

```text
HDR/src/knn_mnist/config.py
```

向上两级后就是：

```text
HDR/
```

也就是项目根目录。

```python
DEFAULT_MNIST_ZIP = PROJECT_ROOT / "mnist.zip"
```

默认 MNIST 数据集压缩包路径。等价于：

```text
C:\Users\Lenovo\Desktop\HDR\mnist.zip
```

```python
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
```

默认输出目录。程序运行产生的日志、图表、报告、模型都会放到 `outputs` 下面。

```python
DEFAULT_K_VALUES = (1, 3, 5, 7, 9)
```

默认要尝试的 K 值。这里使用元组 `tuple`，表示一组固定值。

```python
CLASS_NAMES = [str(i) for i in range(10)]
```

生成数字类别名称：

```text
["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
```

这里用到了列表推导式。`range(10)` 会生成 0 到 9，`str(i)` 把数字转换成字符串。

## 五、`src/knn_mnist/data.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\data.py
```

这个文件负责数据读取和预处理，是“数据处理”部分的核心。

### 1. 导入模块

```python
from __future__ import annotations
```

让类型注解在较新 Python 写法下更兼容。初学者可以理解为：这行主要是为了让代码中的类型提示写法更方便。

```python
import gzip
import struct
import zipfile
from pathlib import Path
```

这几行导入 Python 标准库：

```text
gzip       用于解压 .gz 文件
struct     用于解析二进制文件头
zipfile    用于读取 .zip 压缩包
Path       用于处理路径
```

MNIST 数据在本项目中是：

```text
mnist.zip
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
```

所以要先打开 zip，再解压里面的 gzip，再解析 IDX 二进制格式。

```python
import numpy as np
```

导入 NumPy，并起别名为 `np`。NumPy 用来高效处理数组，例如图片矩阵、标签数组。

### 2. 定义 MNIST 文件名

```python
MNIST_MEMBERS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
```

这是一个字典 `dict`。左边是我们自己起的简短名字，右边是真实文件名。

例如：

```python
MNIST_MEMBERS["train_images"]
```

会得到：

```text
train-images-idx3-ubyte.gz
```

这样写的好处是：如果文件名需要修改，只改这里即可。

### 3. `_read_zip_gzip_member`

```python
def _read_zip_gzip_member(zip_path: Path, member_name: str) -> bytes:
```

定义一个函数，用来从 `mnist.zip` 中读取指定的 `.gz` 文件，并返回解压后的原始字节数据。

参数含义：

```text
zip_path      mnist.zip 的路径
member_name   zip 里面某个 gzip 文件的名字
```

返回值：

```text
bytes         二进制字节数据
```

函数名前面有下划线 `_`，表示这是内部辅助函数，一般不在外部直接调用。

```python
with zipfile.ZipFile(zip_path, "r") as zf:
```

以只读模式打开 zip 文件。`with` 的作用是：用完自动关闭文件，避免文件一直占用。

```python
if member_name not in zf.namelist():
    raise FileNotFoundError(f"{member_name} not found in {zip_path}")
```

检查 zip 里面有没有这个文件。如果没有，就主动报错。`raise` 表示抛出异常。

```python
with zf.open(member_name, "r") as compressed_file:
```

打开 zip 里面的某个 gzip 文件。

```python
return gzip.decompress(compressed_file.read())
```

先读取 gzip 压缩内容，再用 `gzip.decompress` 解压，最后返回原始二进制内容。

### 4. `_parse_idx_images`

```python
def _parse_idx_images(raw: bytes) -> np.ndarray:
```

定义函数，用来解析 MNIST 图片 IDX 文件。

```python
magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
```

MNIST 图片文件前 16 个字节是文件头，包含：

```text
magic   魔数，用于判断文件类型
count   图片数量
rows    图片高度
cols    图片宽度
```

`">IIII"` 的含义：

```text
>       大端字节序，MNIST 文件规定使用这种格式
I       unsigned int，无符号整数
IIII    连续读取 4 个无符号整数
```

```python
if magic != 2051:
    raise ValueError(f"Invalid image IDX magic number: {magic}")
```

图片文件的 magic number 应该是 `2051`。如果不是，说明文件格式不对。

```python
data = np.frombuffer(raw, dtype=np.uint8, offset=16)
```

从第 16 个字节之后开始读取图片像素数据。

```text
uint8 表示 0-255 的整数
offset=16 表示跳过文件头
```

```python
return data.reshape(count, rows, cols)
```

把一维像素数据重新变成三维数组：

```text
(图片数量, 图片高度, 图片宽度)
```

例如 MNIST 训练图片会变成：

```text
(60000, 28, 28)
```

### 5. `_parse_idx_labels`

```python
def _parse_idx_labels(raw: bytes) -> np.ndarray:
```

定义函数，用来解析 MNIST 标签 IDX 文件。

```python
magic, count = struct.unpack(">II", raw[:8])
```

标签文件前 8 个字节是文件头，包含：

```text
magic   魔数
count   标签数量
```

```python
if magic != 2049:
    raise ValueError(f"Invalid label IDX magic number: {magic}")
```

标签文件的 magic number 应该是 `2049`。

```python
labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
```

跳过 8 字节文件头后读取标签。

```python
return labels.reshape(count)
```

把标签整理成一维数组。例如：

```text
(60000,)
```

每个标签是 0 到 9 中的一个数字。

### 6. `load_mnist_raw`

```python
def load_mnist_raw(zip_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

这个函数是外部主要调用的数据加载函数。

参数：

```text
zip_path   mnist.zip 路径
```

返回 4 个数组：

```text
train_images
train_labels
test_images
test_labels
```

```python
zip_path = Path(zip_path)
```

把传进来的路径转换成 `Path` 对象。这样即使传入的是字符串，也能正常处理。

```python
if not zip_path.exists():
    raise FileNotFoundError(f"MNIST zip file does not exist: {zip_path}")
```

检查文件是否存在。如果不存在，直接报错。

```python
train_images = _parse_idx_images(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["train_images"]))
```

读取并解析训练图片。

执行顺序是：

```text
1. 找到 train-images-idx3-ubyte.gz
2. 从 zip 中读取
3. gzip 解压
4. 按 IDX 图片格式解析
```

下面三行同理：

```python
train_labels = _parse_idx_labels(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["train_labels"]))
test_images = _parse_idx_images(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["test_images"]))
test_labels = _parse_idx_labels(_read_zip_gzip_member(zip_path, MNIST_MEMBERS["test_labels"]))
```

分别读取训练标签、测试图片、测试标签。

```python
return train_images, train_labels, test_images, test_labels
```

返回这四组数据。

### 7. `split_train_validation`

```python
def split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_size: int = 10000,
    shuffle: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

这个函数把官方训练集分成训练集和验证集。

参数含义：

```text
images           官方 train 图片
labels           官方 train 标签
validation_size  验证集大小，默认 10000
shuffle          是否打乱数据
seed             随机种子，用于保证打乱结果可复现
```

```python
if validation_size <= 0 or validation_size >= len(images):
    raise ValueError("validation_size must be greater than 0 and smaller than train set size")
```

检查验证集大小是否合法。不能小于等于 0，也不能大于等于总样本数。

```python
if shuffle:
```

如果用户指定打乱数据，就进入下面的逻辑。

```python
rng = np.random.default_rng(seed)
indices = rng.permutation(len(images))
```

创建随机数生成器，并生成一个随机排列的下标数组。

```python
images = images[indices]
labels = labels[indices]
```

使用相同的随机下标同时打乱图片和标签，保证图片和标签仍然一一对应。

```python
split_at = len(images) - validation_size
```

计算切分位置。例如总共 60000，验证集 10000，则 `split_at=50000`。

```python
return images[:split_at], labels[:split_at], images[split_at:], labels[split_at:]
```

返回：

```text
训练图片
训练标签
验证图片
验证标签
```

### 8. `apply_limit`

```python
def apply_limit(images: np.ndarray, labels: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
```

这个函数用于限制样本数量，主要服务于快速实验。

```python
if limit is None:
    return images, labels
```

如果没有限制，就原样返回。

```python
if limit <= 0:
    raise ValueError("limit must be a positive integer")
```

限制数量必须是正整数。

```python
return images[:limit], labels[:limit]
```

只取前 `limit` 个样本和对应标签。

### 9. `preprocess_images`

```python
def preprocess_images(images: np.ndarray, normalize: bool = True, flatten: bool = True) -> np.ndarray:
```

这是 KNN 输入前的预处理函数。

```python
features = images.astype(np.float32)
```

把图片像素从整数类型转换成浮点数。这样才能做除法归一化。

```python
if normalize:
    features = features / 255.0
```

MNIST 原始像素范围是 0-255。除以 255 后变成 0-1。

这样做的原因：KNN 依靠距离计算，归一化可以让距离更稳定。

```python
if flatten:
    features = features.reshape(features.shape[0], -1)
```

把每张 `28x28` 图片展平成 `784` 维向量。

例如：

```text
(10000, 28, 28) -> (10000, 784)
```

`-1` 表示让 NumPy 自动计算这一维大小。

```python
return features
```

返回预处理后的特征矩阵。

## 六、`src/knn_mnist/model.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\model.py
```

这个文件负责 KNN 模型创建、调参和评估。

### 1. 导入模块

```python
from __future__ import annotations
```

用于增强类型注解兼容性。

```python
from time import perf_counter
```

导入高精度计时函数，用来统计某个 K 值实验耗时。

```python
import numpy as np
```

导入 NumPy，用于数组拼接和处理。

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

从 scikit-learn 导入三个评估指标工具：

```text
accuracy_score          计算准确率
classification_report   生成 precision、recall、f1-score 报告
confusion_matrix        生成混淆矩阵
```

```python
from sklearn.neighbors import KNeighborsClassifier
```

导入 KNN 分类器。

```python
from tqdm import tqdm
```

导入进度条工具，用于批量预测时显示进度。

### 2. `build_knn`

```python
def build_knn(
    k: int,
    metric: str = "euclidean",
    weights: str = "uniform",
    n_jobs: int = 1,
) -> KNeighborsClassifier:
```

定义创建 KNN 模型的函数。

参数含义：

```text
k        最近邻数量
metric   距离度量方式，默认欧氏距离
weights  投票权重，默认每个邻居权重相同
n_jobs   并行数量，1 表示单线程，-1 表示使用全部 CPU
```

```python
return KNeighborsClassifier(
```

直接返回一个 `KNeighborsClassifier` 对象。

```python
n_neighbors=k,
```

设置 K 值。例如 K=3 就找最近的 3 个训练样本投票。

```python
metric=metric,
```

设置距离计算方式。`euclidean` 是欧氏距离，也就是常见的直线距离。

```python
weights=weights,
```

设置投票权重。`uniform` 表示每个邻居一票。

```python
algorithm="brute",
```

使用暴力搜索。也就是预测时直接计算测试样本和所有训练样本的距离。MNIST 维度较高，暴力搜索简单可靠。

```python
n_jobs=n_jobs,
```

控制是否并行计算。

### 3. `predict_in_batches`

```python
def predict_in_batches(model: KNeighborsClassifier, x: np.ndarray, batch_size: int = 1000) -> np.ndarray:
```

定义批量预测函数。

为什么要批量预测：KNN 预测时要计算大量距离，如果一次把 10000 张测试图全部预测，可能很慢或占用内存较大。分批可以更稳定。

```python
if batch_size <= 0:
    return model.predict(x)
```

如果批大小小于等于 0，就不分批，直接预测全部。

```python
predictions = []
```

创建空列表，用来存放每一批的预测结果。

```python
for start in tqdm(range(0, len(x), batch_size), desc="Predict", leave=False, ascii=True):
```

循环生成每一批的起始位置。

例如有 2000 个样本，`batch_size=500`，则 `start` 依次是：

```text
0, 500, 1000, 1500
```

`tqdm` 会显示进度条。`ascii=True` 用普通字符显示，避免 Windows 控制台乱码。

```python
end = min(start + batch_size, len(x))
```

计算当前批次的结束位置。使用 `min` 是为了最后一批不会超过总长度。

```python
predictions.append(model.predict(x[start:end]))
```

预测当前批次，并把结果加入列表。

```python
return np.concatenate(predictions)
```

把多个批次的预测结果拼接成一个完整数组。

### 4. `tune_k_values`

这个函数用于比较多个 K 值。

```python
def tune_k_values(...):
```

主要输入：

```text
x_train, y_train   训练特征和标签
x_val, y_val       验证特征和标签
k_values           候选 K 值列表
metric, weights    KNN 参数
batch_size         批量预测大小
n_jobs             并行数量
logger             日志对象
```

```python
results: list[dict] = []
```

创建列表保存每个 K 的结果。每个结果是一个字典。

```python
best_k = k_values[0]
best_accuracy = -1.0
```

先假设第一个 K 是最佳 K，准确率初始化为 -1。

```python
for k in k_values:
```

依次尝试每个 K 值。

```python
if logger:
    logger.info("Start validation for K=%s", k)
```

如果传入了日志对象，就记录当前开始验证哪个 K。

```python
start_time = perf_counter()
```

记录开始时间。

```python
model = build_knn(k=k, metric=metric, weights=weights, n_jobs=n_jobs)
```

创建当前 K 值对应的 KNN 模型。

```python
model.fit(x_train, y_train)
```

训练模型。注意：KNN 的 `fit` 并不像神经网络那样反复迭代，它主要是把训练数据保存起来，方便预测时计算距离。

```python
y_pred = predict_in_batches(model, x_val, batch_size=batch_size)
```

在验证集上预测。

```python
elapsed = perf_counter() - start_time
```

计算当前 K 值实验耗时。

```python
accuracy = accuracy_score(y_val, y_pred)
```

计算验证集准确率。

```python
result = {
    "k": k,
    "metric": metric,
    "weights": weights,
    "validation_accuracy": accuracy,
    "elapsed_seconds": elapsed,
}
```

把当前 K 的结果整理成字典。

```python
results.append(result)
```

保存到结果列表中。

```python
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_k = k
```

如果当前 K 的准确率更高，就更新最佳 K。

```python
return results, best_k
```

返回所有实验结果和最佳 K。

### 5. `evaluate_classifier`

```python
def evaluate_classifier(
    model: KNeighborsClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 1000,
) -> dict:
```

这个函数用于在测试集上评估最终模型。

```python
y_pred = predict_in_batches(model, x_test, batch_size=batch_size)
```

对测试集进行预测。

```python
return {
    "accuracy": accuracy_score(y_test, y_pred),
    "classification_report_text": classification_report(y_test, y_pred, digits=4, zero_division=0),
    "classification_report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, y_pred),
    "y_pred": y_pred,
}
```

返回一个字典，里面包含：

```text
accuracy                    准确率
classification_report_text   文本形式分类报告
classification_report_dict   字典形式分类报告
confusion_matrix             混淆矩阵
y_pred                       每个测试样本的预测结果
```

`zero_division=0` 的作用是：如果某个类别没有预测样本，避免程序警告，把对应指标设为 0。

## 七、`src/knn_mnist/utils.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\utils.py
```

这个文件放通用辅助函数。

### 1. 导入模块

```python
from __future__ import annotations
import json
import logging
from pathlib import Path
```

含义：

```text
json       保存 JSON 文件
logging    记录日志
Path       处理路径
```

### 2. `ensure_dir`

```python
def ensure_dir(path: str | Path) -> Path:
```

定义函数，确保某个目录存在。

```python
path = Path(path)
```

把路径转换成 `Path` 对象。

```python
path.mkdir(parents=True, exist_ok=True)
```

创建目录。

参数解释：

```text
parents=True     如果上级目录不存在，也一起创建
exist_ok=True    如果目录已经存在，不报错
```

```python
return path
```

返回创建好的路径对象。

### 3. `setup_logger`

```python
def setup_logger(log_path: str | Path) -> logging.Logger:
```

定义日志配置函数。

```python
logger = logging.getLogger("knn_mnist")
```

创建名为 `knn_mnist` 的日志器。

```python
logger.setLevel(logging.INFO)
```

设置日志等级为 INFO。这样会记录普通运行信息。

```python
logger.handlers.clear()
```

清空已有 handler，避免重复输出日志。

```python
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
```

定义日志格式：

```text
时间 | 日志等级 | 日志内容
```

```python
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
```

创建控制台日志输出。运行程序时，你在终端看到的信息来自这里。

```python
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

创建文件日志输出。日志会保存到 `outputs/logs`。

```python
return logger
```

返回配置好的日志器。

### 4. `save_json`

```python
def save_json(data: dict, path: str | Path) -> None:
```

定义保存 JSON 文件的函数。

```python
with Path(path).open("w", encoding="utf-8") as f:
```

以写入模式打开文件，使用 UTF-8 编码。

```python
json.dump(data, f, ensure_ascii=False, indent=2)
```

把字典保存成 JSON。

参数解释：

```text
ensure_ascii=False   中文不转义，直接显示中文
indent=2             缩进 2 个空格，方便阅读
```

## 八、`src/knn_mnist/visualize.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\visualize.py
```

这个文件负责生成实验图表。

### 1. 导入和设置

```python
from __future__ import annotations
from pathlib import Path
```

导入类型注解兼容和路径工具。

```python
import matplotlib
matplotlib.use("Agg")
```

设置 Matplotlib 使用 `Agg` 后端。意思是：不弹出窗口，直接把图保存成图片文件。适合命令行训练脚本使用。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

分别导入：

```text
matplotlib.pyplot   画图
numpy               数组计算
pandas              表格数据处理
```

### 2. `save_sample_grid`

这个函数保存 MNIST 样本图。

```python
def save_sample_grid(images: np.ndarray, labels: np.ndarray, path: str | Path, max_images: int = 100) -> None:
```

参数：

```text
images      图片数组
labels      标签数组
path        保存路径
max_images  最多显示多少张
```

```python
count = min(max_images, len(images))
```

计算实际显示数量，不能超过数据总数。

```python
cols = 10
rows = int(np.ceil(count / cols))
```

每行显示 10 张图。根据总数计算需要几行。

```python
fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
```

创建一个图像窗口和多个子图。

```python
axes = np.array(axes).reshape(rows, cols)
```

把子图整理成二维数组，方便用行列索引访问。

```python
for idx in range(rows * cols):
```

遍历所有子图位置。

```python
ax = axes[idx // cols, idx % cols]
```

计算当前子图在第几行第几列。

```python
ax.axis("off")
```

关闭坐标轴，让图片更干净。

```python
if idx < count:
    ax.imshow(images[idx], cmap="gray")
    ax.set_title(str(labels[idx]), fontsize=8)
```

如果当前位置有图片，就显示灰度图，并把标签作为标题。

```python
fig.suptitle("MNIST samples")
fig.tight_layout()
fig.savefig(path, dpi=180)
plt.close(fig)
```

设置总标题，自动调整布局，保存图片，关闭图像对象。

### 3. `save_class_distribution`

这个函数保存类别分布图。

```python
records = []
```

创建空列表，用于保存每个类别的统计记录。

```python
for split_name, labels in label_groups.items():
```

遍历训练集、验证集、测试集。

```python
counts = np.bincount(labels.astype(int), minlength=10)
```

统计 0 到 9 每个数字出现多少次。

```python
for digit, count in enumerate(counts):
    records.append({"split": split_name, "digit": str(digit), "count": int(count)})
```

把统计结果保存成字典，例如：

```text
{"split": "train", "digit": "0", "count": 4932}
```

```python
df = pd.DataFrame(records)
```

把列表转换成 Pandas 表格。

```python
pivot = df.pivot(index="digit", columns="split", values="count")
```

把表格整理成适合画柱状图的形式。

```python
pivot.plot(kind="bar", ax=ax)
```

画柱状图。

后面的 `set_title`、`set_xlabel`、`set_ylabel`、`savefig` 都是设置标题、坐标轴名称和保存图片。

### 4. `save_k_accuracy_plot`

这个函数保存不同 K 值的验证集准确率折线图。

```python
ax.plot(results["k"], results["validation_accuracy"], marker="o")
```

横轴是 K 值，纵轴是验证集准确率，`marker="o"` 表示每个点画圆点。

```python
ax.set_xticks(results["k"].tolist())
```

让横轴只显示实际尝试过的 K 值。

```python
ax.grid(alpha=0.3)
```

添加浅色网格，方便读数。

### 5. `save_confusion_matrix`

这个函数保存混淆矩阵图。

```python
image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
```

把矩阵显示成蓝色热力图。

```python
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
```

添加颜色条。

```python
ticks = np.arange(matrix.shape[0])
```

生成类别刻度。MNIST 是 10 类，所以一般是 0 到 9。

```python
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
```

双层循环遍历矩阵中的每个格子。

```python
ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)
```

在每个格子里写上具体数字。例如真实是 7、预测为 7 的数量。

混淆矩阵含义：

```text
行     真实标签
列     预测标签
对角线 预测正确
非对角线 预测错误
```

## 九、`src/knn_mnist/train.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\src\knn_mnist\train.py
```

这个文件是训练和评估的总控制文件。`scripts/train_knn.py` 会调用这里的 `main()`。

### 1. 导入模块

```python
import argparse
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
```

这些都是标准库：

```text
argparse       解析命令行参数
os             设置环境变量
datetime       生成运行编号
Path           处理路径
perf_counter   统计运行耗时
```

```python
import joblib
import numpy as np
import pandas as pd
```

第三方库：

```text
joblib   保存和加载模型
numpy    数组操作
pandas   保存 CSV、处理表格
```

后面从本项目模块导入函数：

```python
from .config import DEFAULT_K_VALUES, DEFAULT_MNIST_ZIP, DEFAULT_OUTPUT_DIR
from .data import apply_limit, load_mnist_raw, preprocess_images, split_train_validation
from .model import build_knn, evaluate_classifier, tune_k_values
from .utils import ensure_dir, save_json, setup_logger
from .visualize import ...
```

开头的点 `.` 表示从当前包 `knn_mnist` 内部导入。

### 2. `parse_args`

这个函数读取命令行参数。

```python
parser = argparse.ArgumentParser(description="Train and evaluate KNN on local MNIST data.")
```

创建参数解析器。

例如用户运行：

```bash
python scripts/train_knn.py --quick --k-values 1 3 5
```

`argparse` 会帮程序知道用户传了哪些参数。

重要参数解释：

```python
parser.add_argument("--data-zip", type=Path, default=DEFAULT_MNIST_ZIP, help="Path to mnist.zip")
```

指定数据集路径，默认使用项目根目录下的 `mnist.zip`。

```python
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
```

指定输出目录，默认是 `outputs`。

```python
parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help="K values to compare")
```

指定要比较的 K 值。`nargs="+"` 表示可以接收一个或多个值。

例如：

```bash
--k-values 1 3 5 7 9
```

会变成列表：

```python
[1, 3, 5, 7, 9]
```

```python
parser.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan"], help="Distance metric")
```

指定距离度量。只能选 `euclidean` 或 `manhattan`。

```text
euclidean   欧氏距离
manhattan   曼哈顿距离
```

```python
parser.add_argument("--weights", default="uniform", choices=["uniform", "distance"], help="Voting weights")
```

指定投票权重。

```text
uniform    每个邻居权重相同
distance   距离越近权重越大
```

```python
parser.add_argument("--validation-size", type=int, default=10000, help="Validation size from official train split")
```

指定验证集数量，默认 10000。

```python
parser.add_argument("--train-limit", ...)
parser.add_argument("--val-limit", ...)
parser.add_argument("--test-limit", ...)
```

这三个参数用于限制数据量。主要用于快速测试。

```python
parser.add_argument("--quick", action="store_true", help="Use a smaller subset for quick debugging")
```

`--quick` 是开关参数。只要命令里写了它，就表示使用快速模式。

```python
parser.add_argument("--save-model", action="store_true", help="Save final KNN model with joblib")
```

如果加上 `--save-model`，程序会保存最终模型。

```python
return parser.parse_args()
```

解析并返回所有命令行参数。

### 3. `_apply_quick_defaults`

```python
def _apply_quick_defaults(args: argparse.Namespace) -> None:
```

这个函数用于处理快速模式。

```python
if not args.quick:
    return
```

如果没有开启 `--quick`，直接返回，不做任何事。

```python
if args.train_limit is None:
    args.train_limit = 10000
```

快速模式下，如果用户没有指定训练集限制，就默认取 10000 个训练样本。

```python
if args.val_limit is None:
    args.val_limit = 2000
if args.test_limit is None:
    args.test_limit = 2000
```

验证集和测试集默认各取 2000 个。

因此快速版和全量版区别就是样本数量不同，流程完全一样。

### 4. `_write_text_report`

这个函数负责写实验报告 txt。

```python
lines = [...]
```

把报告每一行先放到列表中。

里面记录：

```text
运行编号
数据文件路径
训练、验证、测试样本数量
预处理方法
模型参数
验证集结果
测试集准确率
分类报告
总耗时
```

```python
path.write_text("\n".join(lines), encoding="utf-8")
```

把列表用换行符拼成文本，并保存为 UTF-8 编码文件。

### 5. `main`

这是训练脚本最重要的函数。

```python
args = parse_args()
```

读取命令行参数。

```python
_apply_quick_defaults(args)
```

如果用户开启快速模式，就自动设置较小数据量。

```python
if args.n_jobs == 1:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
```

在单线程运行时设置环境变量，避免 Windows 下 joblib 误报 CPU 核心警告。

```python
run_id = datetime.now().strftime("knn_%Y%m%d_%H%M%S")
```

生成运行编号。例如：

```text
knn_20260527_204846
```

这样每次运行的输出文件不会重名。

```python
output_dir = ensure_dir(args.output_dir)
logs_dir = ensure_dir(output_dir / "logs")
reports_dir = ensure_dir(output_dir / "reports")
figures_dir = ensure_dir(output_dir / "figures")
models_dir = ensure_dir(output_dir / "models")
```

创建输出目录：

```text
outputs/logs
outputs/reports
outputs/figures
outputs/models
```

```python
logger = setup_logger(logs_dir / f"{run_id}.log")
```

创建日志器，并把日志保存到对应文件。

```python
total_start = perf_counter()
```

记录整个实验开始时间。

```python
raw_train_images, raw_train_labels, raw_test_images, raw_test_labels = load_mnist_raw(args.data_zip)
```

读取原始 MNIST 数据。

```python
train_images, y_train, val_images, y_val = split_train_validation(...)
```

把官方训练集分成训练集和验证集。

```python
test_images, y_test = raw_test_images, raw_test_labels
```

官方测试集直接作为测试集。

```python
train_images, y_train = apply_limit(train_images, y_train, args.train_limit)
val_images, y_val = apply_limit(val_images, y_val, args.val_limit)
test_images, y_test = apply_limit(test_images, y_test, args.test_limit)
```

如果使用快速模式或手动指定 limit，这里会裁剪数据数量。

```python
if not args.no_plots:
    save_sample_grid(...)
    save_class_distribution(...)
```

如果没有关闭画图，就保存样本图和类别分布图。

```python
x_train = preprocess_images(train_images)
x_val = preprocess_images(val_images)
x_test = preprocess_images(test_images)
```

将图片归一化并展平成 KNN 可输入的特征。

```python
validation_results, best_k = tune_k_values(...)
```

尝试多个 K 值，在验证集上选出最佳 K。

```python
validation_df = pd.DataFrame(validation_results)
validation_df.to_csv(validation_csv, index=False, encoding="utf-8-sig")
```

把验证结果保存成 CSV。

```python
if not args.no_plots:
    save_k_accuracy_plot(validation_df, ...)
```

保存 K 值准确率对比图。

```python
if args.no_final_refit:
    x_final_train, y_final_train = x_train, y_train
else:
    x_final_train = np.concatenate([x_train, x_val], axis=0)
    y_final_train = np.concatenate([y_train, y_val], axis=0)
```

默认情况下，选出最佳 K 后，会把训练集和验证集合并，再训练最终模型。这样最终模型能利用更多数据。

如果用户加了 `--no-final-refit`，就只用训练集训练最终模型。

```python
final_model = build_knn(...)
final_model.fit(x_final_train, y_final_train)
```

创建最终 KNN 模型并训练。

```python
evaluation = evaluate_classifier(final_model, x_test, y_test, batch_size=args.batch_size)
```

在测试集上评估最终模型。

```python
predictions_df = pd.DataFrame({"y_true": y_test.astype(int), "y_pred": y_pred.astype(int)})
predictions_df.to_csv(...)
```

保存每个测试样本的真实标签和预测标签。

```python
if not args.no_plots:
    save_confusion_matrix(...)
```

保存混淆矩阵图。

```python
metadata = {...}
save_json(metadata, ...)
```

保存本次实验摘要信息，例如 K 值、准确率、样本数。

```python
if args.save_model:
    joblib.dump({"model": final_model, "metadata": metadata}, model_path)
```

如果用户指定保存模型，就用 `joblib` 保存模型文件。

保存内容包括：

```text
model      KNN 模型对象
metadata   实验信息
```

```python
_write_text_report(...)
```

保存文本实验报告。

```python
logger.info("Finished in %.2fs", total_seconds)
```

在日志中记录总耗时。

## 十、`scripts/train_knn.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\scripts\train_knn.py
```

这是命令行训练入口。

```python
from pathlib import Path
import sys
```

导入路径工具和系统模块。

```python
ROOT = Path(__file__).resolve().parents[1]
```

获取项目根目录。这个文件在 `scripts` 目录中，向上一级就是项目根目录。

```python
SRC = ROOT / "src"
```

得到 `src` 目录路径。

```python
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

把 `src` 加入 Python 模块搜索路径。这样才能导入：

```python
from knn_mnist.train import main
```

如果不加这几行，直接运行脚本时 Python 可能找不到 `knn_mnist`。

```python
from knn_mnist.train import main
```

从核心训练模块导入 `main` 函数。

```python
if __name__ == "__main__":
    main()
```

这是 Python 常见写法。意思是：当这个文件被直接运行时，执行 `main()`。

例如：

```bash
python scripts/train_knn.py --quick
```

就会执行 `knn_mnist.train.main()`。

## 十一、`scripts/run_streamlit.bat`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\scripts\run_streamlit.bat
```

这是 Windows 批处理脚本，用来启动 Streamlit 界面。

```bat
@echo off
```

关闭命令回显，让终端输出更干净。

```bat
cd /d "%~dp0\.."
```

切换到项目根目录。

解释：

```text
%~dp0   当前 bat 文件所在目录
\..     向上一级
/d      允许跨盘符切换
```

因为脚本在 `scripts` 目录下，向上一级就是 `HDR` 项目根目录。

```bat
streamlit run app\streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

启动 Streamlit 应用。

参数说明：

```text
app\streamlit_app.py             Streamlit 界面文件
--server.port 8501               使用 8501 端口
--server.headless true            不自动弹出浏览器
--browser.gatherUsageStats false  不收集使用统计
```

运行后手动访问：

```text
http://localhost:8501
```

## 十二、`app/streamlit_app.py`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\app\streamlit_app.py
```

这个文件是交互式演示界面。它不负责训练模型，只负责加载已有模型，并提供上传图片识别、测试集评估、图表查看功能。

### 1. 导入模块

```python
import sys
from pathlib import Path
```

用于处理模块搜索路径和文件路径。

```python
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

导入第三方库：

```text
joblib        加载模型
matplotlib    画图
numpy         数组处理
pandas        表格显示
streamlit     Web 界面
PIL           图片读取和处理
sklearn       模型评估指标
```

### 2. 设置项目路径

```python
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
```

获取项目根目录和 `src` 目录。

```python
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

把 `src` 加入 Python 模块搜索路径。

```python
from knn_mnist.config import DEFAULT_MNIST_ZIP
from knn_mnist.data import load_mnist_raw, preprocess_images
```

复用项目已有代码：默认数据路径、MNIST 加载函数、图片预处理函数。

### 3. 定义输出路径

```python
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
```

定义模型目录和图表目录。Streamlit 会从这里读取已有模型和图表。

### 4. 设置 Streamlit 页面

```python
st.set_page_config(
    page_title="MNIST KNN 演示系统",
    page_icon=None,
    layout="wide",
)
```

设置网页标题、图标和宽屏布局。

```python
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
```

插入 CSS 样式。主要作用：

```text
背景使用浅灰
文字使用深色
指标卡片使用白底和边框
按钮、输入框、标签页尽量去掉圆角
整体风格简洁
```

`unsafe_allow_html=True` 表示允许 Streamlit 渲染 HTML/CSS。

### 5. `list_models`

```python
def list_models() -> list[Path]:
```

列出模型文件。

```python
if not MODEL_DIR.exists():
    return []
```

如果模型目录不存在，返回空列表。

```python
return sorted(MODEL_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
```

查找所有 `.joblib` 模型文件，并按修改时间从新到旧排序。这样界面默认显示最新模型。

### 6. `list_figures`

逻辑和 `list_models` 类似，只是查找的是：

```text
outputs/figures/*.png
```

用于展示已经生成的实验图表。

### 7. `load_model`

```python
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
```

这是带缓存的模型加载函数。

`@st.cache_resource` 是 Streamlit 装饰器，作用是：模型加载一次后缓存起来，避免每次页面刷新都重新读模型。

```python
payload = joblib.load(model_path)
```

读取 `.joblib` 模型文件。

```python
if isinstance(payload, dict) and "model" in payload:
    return payload["model"], payload.get("metadata", {})
```

训练脚本保存模型时使用了：

```python
{"model": final_model, "metadata": metadata}
```

所以这里从字典里取出模型和元信息。

```python
return payload, {}
```

如果模型文件不是字典格式，就直接把它当作模型返回。

### 8. `load_test_data`

```python
@st.cache_data(show_spinner=False)
def load_test_data(data_zip: str):
```

缓存测试集数据。`cache_data` 适合缓存数组、表格这类数据。

```python
_, _, test_images, test_labels = load_mnist_raw(Path(data_zip))
```

读取 MNIST 数据，但只保留测试图片和测试标签。前两个返回值用 `_` 表示不需要。

```python
return test_images, test_labels
```

返回测试集。

### 9. `preprocess_uploaded_image`

这个函数用于处理用户上传的图片，让它尽量接近 MNIST 格式。

```python
gray = ImageOps.grayscale(image)
```

把图片转成灰度图。MNIST 是灰度图，不需要彩色。

```python
if np.array(gray).mean() > 127:
    gray = ImageOps.invert(gray)
```

如果图片平均亮度大于 127，说明背景大概率是白色、数字是黑色。而 MNIST 通常是黑底白字，所以这里进行反色。

```python
arr = np.array(gray)
foreground = arr > 30
```

把图片转换成数组，并找出亮度大于 30 的前景像素。前景一般就是手写数字。

```python
if foreground.any():
```

如果找到了前景，就继续裁剪。

```python
ys, xs = np.where(foreground)
```

找出所有前景像素的行坐标 `ys` 和列坐标 `xs`。

```python
left, right = xs.min(), xs.max()
top, bottom = ys.min(), ys.max()
```

计算数字所在区域的边界。

```python
pad_x = max(2, int((right - left + 1) * 0.15))
pad_y = max(2, int((bottom - top + 1) * 0.15))
```

给裁剪区域增加一点边距，避免把数字边缘裁掉。

```python
left = max(0, left - pad_x)
right = min(arr.shape[1] - 1, right + pad_x)
top = max(0, top - pad_y)
bottom = min(arr.shape[0] - 1, bottom + pad_y)
```

确保裁剪范围不会超过图片边界。

```python
gray = gray.crop((left, top, right + 1, bottom + 1))
```

裁剪出数字区域。

```python
gray.thumbnail((20, 20), Image.Resampling.LANCZOS)
```

把数字缩放到最大 `20x20`。MNIST 原图是 `28x28`，通常数字主体不占满整张图，所以这里保留边缘空白。

```python
canvas = Image.new("L", (28, 28), 0)
```

创建黑色背景的 `28x28` 灰度画布。

```python
offset = ((28 - gray.width) // 2, (28 - gray.height) // 2)
```

计算居中粘贴位置。

```python
canvas.paste(gray, offset)
```

把缩放后的数字贴到画布中央。

```python
return np.array(canvas, dtype=np.uint8)
```

返回 `28x28` 的 NumPy 数组。

### 10. `predict_image`

```python
features = preprocess_images(image_array[np.newaxis, :, :])
```

把单张图片变成模型可输入格式。

解释：

```text
image_array              原本形状是 (28, 28)
image_array[np.newaxis]  变成 (1, 28, 28)
preprocess_images        归一化并展平成 (1, 784)
```

```python
prediction = int(model.predict(features)[0])
```

用模型预测，并取出第一个结果。

```python
if hasattr(model, "predict_proba"):
```

检查模型是否支持输出类别概率。

```python
probabilities = model.predict_proba(features)[0]
classes = [int(c) for c in model.classes_]
prob_df = pd.DataFrame({"数字": classes, "概率": probabilities})
```

如果支持，就生成概率表格。

```python
else:
    prob_df = pd.DataFrame({"数字": [prediction], "概率": [1.0]})
```

如果不支持概率，就只显示预测类别，概率设为 1。

```python
return prediction, prob_df
```

返回预测结果和概率表。

### 11. `predict_with_progress`

这个函数用于测试集评估时显示进度条。

```python
progress = st.progress(0.0)
```

创建 Streamlit 进度条。

```python
for start in range(0, total, batch_size):
```

按批次遍历测试集。

```python
predictions.append(model.predict(features[start:end]))
```

预测当前批次。

```python
progress.progress(end / total)
```

更新进度条百分比。

```python
progress.empty()
```

预测完成后移除进度条。

### 12. `make_confusion_figure`

这个函数和 `visualize.py` 中的混淆矩阵图类似，只是这里返回 `fig`，让 Streamlit 直接显示，而不是保存到文件。

```python
fig, ax = plt.subplots(figsize=(6.5, 5.8))
```

创建图像。

```python
ax.imshow(matrix, interpolation="nearest", cmap="Blues")
```

绘制矩阵热力图。

```python
ax.text(...)
```

在每个格子写入数字。

```python
return fig
```

返回图像对象给 Streamlit。

### 13. `make_probability_figure`

这个函数把单张图片的类别概率画成柱状图。

```python
ax.bar(prob_df["数字"].astype(str), prob_df["概率"], ...)
```

横轴是数字类别，纵轴是预测概率。

```python
ax.set_ylim(0, 1.0)
```

概率范围固定为 0 到 1。

### 14. `show_model_status`

```python
c1, c2, c3, c4 = st.columns(4)
```

把页面分成 4 列。

```python
c1.metric("模型文件", model_path.name)
c2.metric("K 值", ...)
c3.metric("距离度量", ...)
c4.metric("测试准确率", ...)
```

展示当前模型的关键信息。

`metadata.get("best_k", getattr(model, "n_neighbors", "未知"))` 的意思是：

```text
优先从 metadata 中取 best_k
如果没有，就从模型对象里取 n_neighbors
如果还没有，就显示“未知”
```

### 15. 页面主体逻辑

```python
st.title("MNIST 手写数字识别 KNN 演示系统")
```

显示页面标题。

```python
models = list_models()
if not models:
    st.error(...)
    st.stop()
```

如果找不到模型文件，显示错误并停止运行。因为没有模型就无法预测。

```python
selected_model_name = st.sidebar.selectbox("模型文件", model_names, index=0)
```

在侧边栏创建模型选择框。

```python
selected_model_path = next(path for path in models if path.name == selected_model_name)
```

根据用户选择的文件名找到对应路径。

```python
model, metadata = load_model(str(selected_model_path))
```

加载模型。

```python
data_zip_path = st.sidebar.text_input("MNIST 数据文件", value=str(DEFAULT_MNIST_ZIP))
```

在侧边栏显示数据集路径，用户也可以修改。

```python
tab_upload, tab_test, tab_figures = st.tabs(["上传图片识别", "测试集评估", "实验图表"])
```

创建三个标签页。

#### 上传图片识别标签页

```python
uploaded_file = st.file_uploader("上传手写数字图像", type=["png", "jpg", "jpeg", "bmp"])
```

创建文件上传控件，只允许上传常见图片格式。

```python
original_image = Image.open(uploaded_file)
processed_image = preprocess_uploaded_image(original_image)
```

读取原图，并转换成 MNIST 格式图。

```python
img_col_1.image(...)
img_col_2.image(...)
```

显示原始图片和预处理图片。

```python
prediction, probability_df = predict_image(model, processed_image)
st.metric("预测结果", str(prediction))
```

预测并显示结果。

```python
st.pyplot(make_probability_figure(probability_df), clear_figure=True)
st.dataframe(probability_df, ...)
```

显示概率柱状图和概率表。

#### 测试集评估标签页

```python
test_images, test_labels = load_test_data(data_zip_path)
```

读取 MNIST 测试集。

```python
sample_count = st.number_input(...)
batch_size = st.number_input(...)
```

让用户选择测试样本数量和批大小。

```python
if st.button("运行测试集评估"):
```

用户点击按钮后才开始评估，避免页面一打开就自动跑很慢的测试。

```python
features = preprocess_images(selected_images)
predictions = predict_with_progress(model, features, batch_size=int(batch_size))
```

预处理并批量预测。

```python
accuracy = accuracy_score(selected_labels, predictions)
matrix = confusion_matrix(...)
report_df = pd.DataFrame(classification_report(...)).transpose()
```

计算准确率、混淆矩阵和分类报告。

```python
st.download_button(...)
```

提供下载预测结果 CSV 的按钮。

#### 实验图表标签页

```python
figures = list_figures()
```

读取已有图表。

```python
selected_figure_name = st.selectbox("图表文件", figure_names)
```

让用户选择要看的图。

```python
st.image(str(selected_figure), caption=selected_figure.name, use_container_width=True)
```

显示选中的图表。

## 十三、`requirements.txt`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\requirements.txt
```

这个文件用于 pip 安装依赖。

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
tqdm>=4.66
streamlit>=1.32
pillow>=10.0
```

每一行表示一个第三方库：

```text
numpy          数组和矩阵计算
pandas         表格数据处理
scikit-learn   KNN 模型和评估指标
matplotlib     画图
joblib         保存和加载模型
tqdm           命令行进度条
streamlit      Web 交互界面
pillow         图片读取和处理
```

`>=` 表示版本不能低于指定版本。

## 十四、`environment.yml`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\environment.yml
```

这是 conda 环境配置文件。

```yaml
name: mnist-knn
```

创建的环境名叫 `mnist-knn`。

```yaml
channels:
  - conda-forge
```

指定从 `conda-forge` 这个频道安装包。

```yaml
dependencies:
  - python=3.10
  - numpy>=1.24
  ...
```

列出环境需要安装的 Python 版本和依赖库。

创建环境命令：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate mnist-knn
```

## 十五、`README.md`

文件路径：

```text
C:\Users\Lenovo\Desktop\HDR\README.md
```

这个文件是项目使用说明，主要告诉别人：

```text
项目是做什么的
目录结构是什么
怎么创建环境
怎么运行快速实验
怎么运行全量实验
输出文件在哪里
怎么启动 Streamlit 演示界面
```

它适合作为提交代码仓库时的首页说明。

## 十六、输出目录说明

这些目录不是手写代码，而是程序运行后生成结果。

### 1. `outputs/logs`

保存日志文件，例如：

```text
knn_20260527_204846.log
```

日志记录了：

```text
读取数据
数据划分
不同 K 值验证准确率
最佳 K 值
测试准确率
模型保存路径
总耗时
```

### 2. `outputs/reports`

保存报告和表格：

```text
*_report.txt                 实验文字报告
*_summary.json               实验摘要
*_validation_results.csv     不同 K 值结果
*_test_predictions.csv       测试集预测结果
```

这些文件可以用于技术报告和答辩 PPT。

### 3. `outputs/figures`

保存图表：

```text
*_samples.png                 MNIST 样本图
*_class_distribution.png      类别分布图
*_k_accuracy.png              K 值准确率对比图
*_confusion_matrix.png        混淆矩阵
```

Streamlit 界面的“实验图表”页也会读取这里的图片。

### 4. `outputs/models`

保存训练好的模型：

```text
*_knn.joblib
```

Streamlit 界面会自动读取这里的最新模型。

## 十七、运行命令和对应代码关系

### 1. 快速实验

```bash
python scripts/train_knn.py --quick --k-values 1 3 5 7 --save-model
```

运行后发生的事情：

```text
scripts/train_knn.py
  -> 调用 knn_mnist.train.main()
  -> 读取 mnist.zip
  -> 使用 10000 训练、2000 验证、2000 测试
  -> 比较 K=1,3,5,7
  -> 保存模型和输出结果
```

### 2. 全量实验

```bash
python scripts/train_knn.py --k-values 1 3 5 7 9 --save-model
```

使用完整数据：

```text
训练集 50000
验证集 10000
测试集 10000
```

### 3. Streamlit 界面

```bash
scripts\run_streamlit.bat
```

运行后发生的事情：

```text
启动 app/streamlit_app.py
读取 outputs/models 中的模型
读取 outputs/figures 中的图表
提供上传图片识别和测试集评估功能
```

浏览器访问：

```text
http://localhost:8501
```

## 十八、KNN 方法原理补充

KNN 没有复杂训练过程。它的 `fit` 主要是记住训练数据。

预测一张图片时：

```text
1. 把图片变成 784 维向量
2. 计算它和每个训练样本的距离
3. 找到最近的 K 个训练样本
4. 统计这 K 个样本中出现最多的标签
5. 输出这个标签
```

欧氏距离公式：

```text
distance = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (x784-y784)^2)
```

K 值影响：

```text
K 太小   容易受噪声影响
K 太大   分类边界过平滑，可能把细节忽略
```

所以项目中要比较多个 K 值，并在验证集上选择最优 K。

## 十九、为什么要划分训练集、验证集、测试集

```text
训练集   用来让模型记住样本
验证集   用来选择 K 值
测试集   用来评估最终模型效果
```

不能直接用测试集选择 K 值，否则测试集就参与了调参，最终结果会偏乐观。

## 二十、为什么要归一化和展平

### 1. 归一化

原始像素范围：

```text
0 到 255
```

归一化后：

```text
0 到 1
```

这样距离计算更稳定。

### 2. 展平

MNIST 原图：

```text
28 x 28
```

KNN 输入：

```text
784 维向量
```

所以要把二维图片展平。

例如：

```text
[[0, 0, 255],
 [0, 128, 0]]
```

展平后变成：

```text
[0, 0, 255, 0, 128, 0]
```

## 二十一、给 Python 小白的阅读建议

如果你刚开始学 Python，可以按这个顺序理解：

1. 先看 `scripts/train_knn.py`，理解程序入口
2. 再看 `train.py`，理解完整流程
3. 再看 `data.py`，理解数据怎么读
4. 再看 `model.py`，理解 KNN 怎么调参和评估
5. 再看 `visualize.py`，理解图表怎么生成
6. 最后看 `app/streamlit_app.py`，理解交互界面怎么复用前面的代码

不要一开始就从 `streamlit_app.py` 读起，因为它同时涉及界面、图片处理、模型预测、图表展示，内容更多。

## 二十二、常见修改位置

如果想修改 K 值：

```text
命令行里改 --k-values
或修改 config.py 里的 DEFAULT_K_VALUES
```

如果想改快速模式的数据量：

```text
修改 train.py 中 _apply_quick_defaults
```

如果想改输出目录：

```text
命令行传 --output-dir
或修改 config.py 中 DEFAULT_OUTPUT_DIR
```

如果想改 Streamlit 页面样式：

```text
修改 app/streamlit_app.py 中 st.markdown 的 CSS 部分
```

如果想改上传图片预处理方式：

```text
修改 app/streamlit_app.py 中 preprocess_uploaded_image
```

如果想改图表样式：

```text
训练脚本图表：修改 visualize.py
Streamlit 页面动态图表：修改 streamlit_app.py 中 make_confusion_figure 和 make_probability_figure
```

