# MNIST 手写数字识别 - KNN 方法

本目录实现课程项目中“传统机器学习方法（KNN）”部分，包含本地 MNIST 数据读取、预处理、K 值对比实验、测试集评估、日志记录和可视化输出。

## 项目结构

```text
HDR/
  mnist.zip
  README.md
  requirements.txt
  environment.yml
  scripts/
    train_knn.py
  app/
    streamlit_app.py
  src/
    knn_mnist/
      data.py
      model.py
      train.py
      utils.py
      visualize.py
  outputs/
    figures/
    logs/
    models/
    reports/
  Reference/
```

## 环境配置

使用 conda：

```bash
conda env create -f environment.yml
conda activate mnist-knn
```

或者使用已有 Python 环境：

```bash
pip install -r requirements.txt
```

## 快速运行

KNN 在全量 MNIST 上预测会比较慢，建议先用子集跑通流程：

```bash
python scripts/train_knn.py --quick --k-values 1 3 5 7 --save-model
```

## 全量实验

按项目计划中的划分方式运行：官方 train 集 60000 个样本中，前 50000 个作为训练集，后 10000 个作为验证集；官方 test 集 10000 个样本全部作为测试集。

```bash
python scripts/train_knn.py --k-values 1 3 5 7 9 --save-model
```

如果希望使用全部 CPU 核心加速距离计算，可以追加 `--n-jobs -1`。

## 输出内容

每次运行会在 `outputs/` 下生成：

- `logs/`：运行日志
- `reports/`：实验报告、K 值准确率表、预测结果
- `figures/`：样本图、类别分布图、K 值对比图、混淆矩阵
- `models/`：保存的 KNN 模型文件

## Streamlit 演示界面

先确保至少有一个模型文件：

```bash
python scripts/train_knn.py --quick --k-values 1 3 5 7 --save-model
```

启动交互式演示界面：

```bash
streamlit run app/streamlit_app.py
```

Windows 下也可以在激活 conda 环境后运行：

```bash
scripts\run_streamlit.bat
```

界面支持上传手写数字图片进行识别，也支持使用当前模型评估 MNIST 测试集，并浏览 `outputs/figures/` 中已经生成的实验图表。
