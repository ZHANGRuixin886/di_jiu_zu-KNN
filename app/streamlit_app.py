from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from knn_mnist.config import DEFAULT_MNIST_ZIP
from knn_mnist.data import load_mnist_raw, preprocess_images


OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"


st.set_page_config(
    page_title="MNIST KNN 演示系统",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --border-color: #c9d1d9;
        --text-color: #1f2328;
        --muted-color: #57606a;
        --background-color: #f6f8fa;
        --panel-color: #ffffff;
    }
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--text-color);
    }
    div[data-testid="stMetric"],
    div[data-testid="stExpander"],
    div[data-testid="stFileUploader"],
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        border-radius: 0 !important;
    }
    div[data-testid="stMetric"] {
        background: var(--panel-color);
        border: 1px solid var(--border-color);
        padding: 14px 16px;
    }
    .stButton > button,
    .stDownloadButton > button,
    .stSelectbox div,
    .stNumberInput input,
    .stSlider div,
    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .small-note {
        color: var(--muted-color);
        font-size: 0.92rem;
        margin-top: -0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_models() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_figures() -> list[Path]:
    if not FIGURE_DIR.exists():
        return []
    return sorted(FIGURE_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})
    return payload, {}


@st.cache_data(show_spinner=False)
def load_test_data(data_zip: str):
    _, _, test_images, test_labels = load_mnist_raw(Path(data_zip))
    return test_images, test_labels


def preprocess_uploaded_image(image: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    if np.array(gray).mean() > 127:
        gray = ImageOps.invert(gray)

    arr = np.array(gray)
    foreground = arr > 30
    if foreground.any():
        ys, xs = np.where(foreground)
        left, right = xs.min(), xs.max()
        top, bottom = ys.min(), ys.max()
        pad_x = max(2, int((right - left + 1) * 0.15))
        pad_y = max(2, int((bottom - top + 1) * 0.15))
        left = max(0, left - pad_x)
        right = min(arr.shape[1] - 1, right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(arr.shape[0] - 1, bottom + pad_y)
        gray = gray.crop((left, top, right + 1, bottom + 1))

    gray.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - gray.width) // 2, (28 - gray.height) // 2)
    canvas.paste(gray, offset)
    return np.array(canvas, dtype=np.uint8)


def predict_image(model, image_array: np.ndarray) -> tuple[int, pd.DataFrame]:
    features = preprocess_images(image_array[np.newaxis, :, :])
    prediction = int(model.predict(features)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [int(c) for c in model.classes_]
        prob_df = pd.DataFrame({"数字": classes, "概率": probabilities})
    else:
        prob_df = pd.DataFrame({"数字": [prediction], "概率": [1.0]})
    return prediction, prob_df


def predict_with_progress(model, features: np.ndarray, batch_size: int = 500) -> np.ndarray:
    predictions = []
    progress = st.progress(0.0)
    total = len(features)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        predictions.append(model.predict(features[start:end]))
        progress.progress(end / total)
    progress.empty()
    return np.concatenate(predictions)


def make_confusion_figure(matrix: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks])
    ax.set_yticklabels([str(i) for i in ticks])
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def make_probability_figure(prob_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(prob_df["数字"].astype(str), prob_df["概率"], color="#2f6f9f", edgecolor="#1f2328", linewidth=0.5)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def show_model_status(model, model_path: Path, metadata: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型文件", model_path.name)
    c2.metric("K 值", metadata.get("best_k", getattr(model, "n_neighbors", "未知")))
    c3.metric("距离度量", metadata.get("metric", getattr(model, "metric", "未知")))
    test_accuracy = metadata.get("test_accuracy")
    c4.metric("测试准确率", "未记录" if test_accuracy is None else f"{test_accuracy:.4f}")


st.title("MNIST 手写数字识别 KNN 演示系统")
st.markdown('<div class="small-note">传统机器学习方法演示界面</div>', unsafe_allow_html=True)

models = list_models()
if not models:
    st.error("未找到 KNN 模型文件。请先运行 python scripts/train_knn.py --quick --save-model")
    st.stop()

model_names = [p.name for p in models]
selected_model_name = st.sidebar.selectbox("模型文件", model_names, index=0)
selected_model_path = next(path for path in models if path.name == selected_model_name)
model, metadata = load_model(str(selected_model_path))

data_zip_path = st.sidebar.text_input("MNIST 数据文件", value=str(DEFAULT_MNIST_ZIP))
st.sidebar.markdown("---")
st.sidebar.write("当前项目目录")
st.sidebar.code(str(ROOT), language=None)

show_model_status(model, selected_model_path, metadata)

tab_upload, tab_test, tab_figures = st.tabs(["上传图片识别", "测试集评估", "实验图表"])

with tab_upload:
    left, right = st.columns([1, 1])
    with left:
        uploaded_file = st.file_uploader("上传手写数字图像", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            processed_image = preprocess_uploaded_image(original_image)

            img_col_1, img_col_2 = st.columns(2)
            img_col_1.image(original_image, caption="原始图像", use_container_width=True)
            img_col_2.image(processed_image, caption="MNIST 格式图像", width=180, clamp=True)

            prediction, probability_df = predict_image(model, processed_image)
            st.metric("预测结果", str(prediction))
        else:
            st.info("请选择一张手写数字图片。")

    with right:
        if uploaded_file is not None:
            st.subheader("类别概率")
            st.pyplot(make_probability_figure(probability_df), clear_figure=True)
            st.dataframe(probability_df, hide_index=True, use_container_width=True)
        else:
            st.subheader("模型信息")
            st.json(metadata or {"model": selected_model_path.name})

with tab_test:
    st.subheader("现有测试集")
    test_images, test_labels = load_test_data(data_zip_path)
    max_samples = len(test_labels)
    sample_count = st.number_input(
        "测试样本数量",
        min_value=100,
        max_value=max_samples,
        value=min(1000, max_samples),
        step=100,
    )
    batch_size = st.number_input("预测批大小", min_value=50, max_value=5000, value=500, step=50)

    if st.button("运行测试集评估"):
        selected_images = test_images[:sample_count]
        selected_labels = test_labels[:sample_count]
        features = preprocess_images(selected_images)
        predictions = predict_with_progress(model, features, batch_size=int(batch_size))
        accuracy = accuracy_score(selected_labels, predictions)
        matrix = confusion_matrix(selected_labels, predictions, labels=list(range(10)))
        report_df = pd.DataFrame(classification_report(
            selected_labels,
            predictions,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        )).transpose()

        c1, c2, c3 = st.columns(3)
        c1.metric("样本数量", str(sample_count))
        c2.metric("准确率", f"{accuracy:.4f}")
        c3.metric("错误数量", str(int((selected_labels != predictions).sum())))

        st.pyplot(make_confusion_figure(matrix), clear_figure=True)
        st.subheader("分类报告")
        st.dataframe(report_df, use_container_width=True)

        predictions_df = pd.DataFrame({
            "真实标签": selected_labels.astype(int),
            "预测标签": predictions.astype(int),
        })
        st.download_button(
            "下载预测结果",
            data=predictions_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="knn_test_predictions.csv",
            mime="text/csv",
        )

with tab_figures:
    st.subheader("已生成图表")
    figures = list_figures()
    if not figures:
        st.info("未找到图表文件。运行训练脚本后会生成图表。")
    else:
        figure_names = [figure.name for figure in figures]
        selected_figure_name = st.selectbox("图表文件", figure_names)
        selected_figure = next(path for path in figures if path.name == selected_figure_name)
        st.image(str(selected_figure), caption=selected_figure.name, use_container_width=True)
