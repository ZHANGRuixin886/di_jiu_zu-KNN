const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT_DIR = path.join(ROOT, "deliverables");
const FIG_DIR = path.join(ROOT, "outputs", "figures");
const OUT_FILE = path.join(OUT_DIR, "机器学习大作业PPT_KNN手写数字识别.pptx");

const FIG = {
  samples: path.join(FIG_DIR, "knn_20260527_230737_samples.png"),
  classDistribution: path.join(FIG_DIR, "knn_20260527_230737_class_distribution.png"),
  kAccuracy: path.join(FIG_DIR, "knn_20260527_230737_k_accuracy.png"),
  confusion: path.join(FIG_DIR, "knn_20260527_230737_confusion_matrix.png"),
};

for (const file of Object.values(FIG)) {
  if (!fs.existsSync(file)) {
    throw new Error(`Missing figure: ${file}`);
  }
}

fs.mkdirSync(OUT_DIR, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "MNIST KNN Project";
pptx.subject = "Machine Learning Course Project";
pptx.title = "基于 KNN 的 MNIST 手写数字识别";
pptx.company = "Course Project";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};
pptx.margin = 0;

const C = {
  ink: "17212B",
  muted: "5D6975",
  line: "CDD6E0",
  bg: "F7F9FC",
  white: "FFFFFF",
  accent: "2F6F9F",
  accent2: "B35C2E",
  soft: "EAF1F7",
  pale: "F0F4F8",
};

function addHeader(slide, kicker, title, page) {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.bg }, line: { color: C.bg } });
  slide.addText(kicker, { x: 0.65, y: 0.35, w: 2.7, h: 0.25, fontFace: "Microsoft YaHei", fontSize: 9, bold: true, color: C.accent, breakLine: false });
  slide.addText(title, { x: 0.65, y: 0.68, w: 10.9, h: 0.42, fontFace: "Microsoft YaHei", fontSize: 23, bold: true, color: C.ink, margin: 0 });
  slide.addShape(pptx.ShapeType.line, { x: 0.65, y: 1.22, w: 12.05, h: 0, line: { color: C.line, width: 0.8 } });
  slide.addText(String(page).padStart(2, "0"), { x: 12.18, y: 0.38, w: 0.5, h: 0.25, fontSize: 9, color: C.muted, align: "right", margin: 0 });
}

function addFooter(slide) {
  slide.addShape(pptx.ShapeType.line, { x: 0.65, y: 7.03, w: 12.05, h: 0, line: { color: C.line, width: 0.5 } });
  slide.addText("Machine Learning Course Project | MNIST KNN", { x: 0.65, y: 7.12, w: 5.6, h: 0.18, fontSize: 7.5, color: C.muted, margin: 0 });
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  for (const item of items) {
    runs.push({
      text: item,
      options: {
        bullet: { type: "ul" },
        breakLine: true,
      },
    });
  }
  slide.addText(runs, {
    x, y, w, h,
    fontFace: "Microsoft YaHei",
    fontSize: opts.fontSize || 15,
    color: opts.color || C.ink,
    fit: "shrink",
    margin: 0.05,
    paraSpaceAfterPt: 8,
    breakLine: false,
  });
}

function addMetric(slide, label, value, note, x, y, w) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.9, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  slide.addText(value, { x: x + 0.16, y: y + 0.14, w: w - 0.32, h: 0.27, fontSize: 21, bold: true, color: C.accent, margin: 0 });
  slide.addText(label, { x: x + 0.16, y: y + 0.48, w: w - 0.32, h: 0.17, fontSize: 9.5, bold: true, color: C.ink, margin: 0 });
  slide.addText(note, { x: x + 0.16, y: y + 0.68, w: w - 0.32, h: 0.15, fontSize: 7.5, color: C.muted, margin: 0 });
}

function addSectionLabel(slide, text, x, y, w = 1.6) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.24, fill: { color: C.soft }, line: { color: C.accent, width: 0.6 } });
  slide.addText(text, { x: x + 0.08, y: y + 0.045, w: w - 0.16, h: 0.12, fontSize: 7.5, bold: true, color: C.accent, margin: 0, valign: "mid" });
}

function addTextBox(slide, text, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: opts.fill || C.white }, line: { color: opts.line || C.line, width: 0.8 } });
  slide.addText(text, {
    x: x + 0.15,
    y: y + 0.14,
    w: w - 0.3,
    h: h - 0.22,
    fontFace: "Microsoft YaHei",
    fontSize: opts.fontSize || 13,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    fit: "shrink",
    margin: 0,
  });
}

function addSimpleTable(slide, rows, x, y, colWidths, rowHeight = 0.38) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  rows.forEach((row, r) => {
    let cx = x;
    const fill = r === 0 ? C.accent : (r % 2 === 0 ? C.pale : C.white);
    const color = r === 0 ? C.white : C.ink;
    row.forEach((cell, c) => {
      slide.addShape(pptx.ShapeType.rect, { x: cx, y: y + r * rowHeight, w: colWidths[c], h: rowHeight, fill: { color: fill }, line: { color: C.line, width: 0.6 } });
      slide.addText(String(cell), { x: cx + 0.05, y: y + r * rowHeight + 0.08, w: colWidths[c] - 0.1, h: rowHeight - 0.12, fontSize: r === 0 ? 9.2 : 9, bold: r === 0, color, align: c === 0 ? "center" : "center", margin: 0, fit: "shrink" });
      cx += colWidths[c];
    });
  });
  slide.addShape(pptx.ShapeType.rect, { x, y, w: totalW, h: rows.length * rowHeight, fill: { transparency: 100, color: C.white }, line: { color: C.line, width: 0.8 } });
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.bg }, line: { color: C.bg } });
  slide.addText("机器学习课程大作业", { x: 0.75, y: 0.65, w: 4.0, h: 0.28, fontSize: 11, bold: true, color: C.accent, margin: 0 });
  slide.addText("基于 KNN 的 MNIST 手写数字识别", { x: 0.75, y: 1.08, w: 9.6, h: 0.62, fontSize: 30, bold: true, color: C.ink, margin: 0, fit: "shrink" });
  slide.addText("从数据处理、K 值选择到测试集评估与交互式演示", { x: 0.77, y: 1.88, w: 7.4, h: 0.3, fontSize: 14, color: C.muted, margin: 0 });
  addMetric(slide, "测试集准确率", "97.05%", "10000 张测试图片", 0.78, 5.58, 2.2);
  addMetric(slide, "最优 K 值", "K=3", "验证集准确率 0.9720", 3.22, 5.58, 2.2);
  addMetric(slide, "算法", "KNN", "欧氏距离 + 多数投票", 5.66, 5.58, 2.35);
  slide.addImage({ path: FIG.samples, x: 8.35, y: 0.95, w: 4.15, h: 4.15 });
  addFooter(slide);
}

// Slide 2
{
  const slide = pptx.addSlide();
  addHeader(slide, "TASK", "手写数字识别是经典监督学习分类任务", 2);
  addBullets(slide, [
    "输入为 28x28 灰度手写数字图片，输出为 0-9 的数字类别。",
    "项目目标是使用传统机器学习算法完成分类，而不是依赖深度神经网络。",
    "完整流程包括数据读取、预处理、模型构建、参数选择、评估分析和交互展示。",
    "该任务适合检验对分类问题、距离度量、验证集调参和模型评价指标的理解。"
  ], 0.9, 1.65, 6.0, 3.0);
  addTextBox(slide, "机器学习任务定义\n\n样本：手写数字图片\n特征：784 维像素向量\n标签：数字 0-9\n模型：KNN 分类器", 7.35, 1.55, 4.55, 2.0, { fill: C.white, fontSize: 14 });
  addMetric(slide, "类别数量", "10", "数字 0-9", 7.35, 4.1, 1.65);
  addMetric(slide, "图片尺寸", "28x28", "灰度图像", 9.15, 4.1, 1.65);
  addMetric(slide, "任务类型", "分类", "监督学习", 10.95, 4.1, 1.65);
  addFooter(slide);
}

// Slide 3
{
  const slide = pptx.addSlide();
  addHeader(slide, "DATA", "MNIST 图像被转换为 KNN 可处理的数值特征", 3);
  slide.addImage({ path: FIG.classDistribution, x: 0.85, y: 1.52, w: 5.65, h: 3.1 });
  addSectionLabel(slide, "预处理流程", 7.0, 1.55, 1.5);
  const steps = [
    ["读取", "解析 mnist.zip 中的 IDX gzip 文件"],
    ["划分", "50000 训练 / 10000 验证 / 10000 测试"],
    ["归一化", "像素值从 0-255 转换到 0-1"],
    ["展平", "28x28 图片转换为 784 维向量"],
  ];
  steps.forEach((s, i) => {
    const y = 1.95 + i * 0.85;
    slide.addShape(pptx.ShapeType.rect, { x: 7.0, y, w: 1.15, h: 0.45, fill: { color: C.accent }, line: { color: C.accent } });
    slide.addText(s[0], { x: 7.0, y: y + 0.12, w: 1.15, h: 0.14, fontSize: 9, bold: true, color: C.white, align: "center", margin: 0 });
    slide.addShape(pptx.ShapeType.rect, { x: 8.25, y, w: 4.1, h: 0.45, fill: { color: C.white }, line: { color: C.line } });
    slide.addText(s[1], { x: 8.38, y: y + 0.12, w: 3.8, h: 0.16, fontSize: 10.5, color: C.ink, margin: 0 });
  });
  addFooter(slide);
}

// Slide 4
{
  const slide = pptx.addSlide();
  addHeader(slide, "ALGORITHM", "KNN 根据最近邻样本投票完成分类", 4);
  addTextBox(slide, "核心思想\n\n对于待识别样本，计算它与训练集中所有样本的距离，选出距离最近的 K 个样本，再根据邻居标签多数投票得到分类结果。", 0.85, 1.55, 5.1, 1.7, { fill: C.white, fontSize: 13.5 });
  addTextBox(slide, "欧氏距离\n\nd(x, y) = sqrt(Σ(xᵢ - yᵢ)²)\n\n距离越小，表示两张图片在像素特征上越相似。", 0.85, 3.55, 5.1, 1.55, { fill: C.white, fontSize: 13.5 });
  const flow = ["输入图片", "计算距离", "选 K 个邻居", "多数投票", "输出类别"];
  flow.forEach((text, i) => {
    const x = 6.35 + i * 1.25;
    slide.addShape(pptx.ShapeType.rect, { x, y: 2.25, w: 1.05, h: 0.62, fill: { color: i === 4 ? C.accent : C.white }, line: { color: C.line } });
    slide.addText(text, { x: x + 0.08, y: 2.45, w: 0.89, h: 0.12, fontSize: 8.2, bold: true, color: i === 4 ? C.white : C.ink, align: "center", margin: 0, fit: "shrink" });
    if (i < flow.length - 1) slide.addShape(pptx.ShapeType.line, { x: x + 1.05, y: 2.56, w: 0.2, h: 0, line: { color: C.accent, width: 1.2, beginArrowType: "none", endArrowType: "triangle" } });
  });
  addBullets(slide, [
    "K 过小：容易受噪声影响。",
    "K 过大：邻居范围变大，可能引入不相似样本。",
    "本项目通过验证集选择最优 K。"
  ], 6.55, 4.05, 5.2, 1.2, { fontSize: 13 });
  addFooter(slide);
}

// Slide 5
{
  const slide = pptx.addSlide();
  addHeader(slide, "IMPLEMENTATION", "项目采用模块化代码结构", 5);
  const rows = [
    ["文件", "功能"],
    ["data.py", "读取 MNIST、划分数据集、归一化和展平"],
    ["model.py", "构建 KNN、K 值调参、测试集评估"],
    ["train.py", "串联完整实验流程并保存结果"],
    ["visualize.py", "生成样本图、类别分布、K 值图和混淆矩阵"],
    ["streamlit_app.py", "交互式上传识别和测试集评估界面"],
  ];
  addSimpleTable(slide, rows, 0.85, 1.55, [2.25, 8.7], 0.54);
  addTextBox(slide, "结构特点\n\n数据、模型、训练、图表和演示界面分离，便于调试、复现实验和后续扩展。", 0.85, 5.2, 10.95, 0.92, { fill: C.white, fontSize: 13 });
  addFooter(slide);
}

// Slide 6
{
  const slide = pptx.addSlide();
  addHeader(slide, "EXPERIMENT", "通过验证集选择最优 K 值", 6);
  addMetric(slide, "训练集", "50000", "用于模型拟合", 0.9, 1.65, 2.1);
  addMetric(slide, "验证集", "10000", "用于选择 K", 3.25, 1.65, 2.1);
  addMetric(slide, "测试集", "10000", "用于最终评估", 5.6, 1.65, 2.1);
  addTextBox(slide, "实验参数\n\n候选 K 值：1、3、5、7、9\n距离度量：欧氏距离\n投票方式：普通多数投票\n最终训练：训练集 + 验证集", 8.1, 1.55, 3.8, 2.45, { fill: C.white, fontSize: 13.2 });
  addBullets(slide, [
    "训练集用于让 KNN 保存样本特征和标签。",
    "验证集用于比较不同 K 值，避免直接用测试集调参。",
    "测试集只用于最终效果评估，使结果更客观。"
  ], 1.0, 4.25, 10.0, 1.2, { fontSize: 14 });
  addFooter(slide);
}

// Slide 7
{
  const slide = pptx.addSlide();
  addHeader(slide, "VALIDATION", "K=3 在验证集上表现最好", 7);
  slide.addImage({ path: FIG.kAccuracy, x: 0.85, y: 1.48, w: 6.0, h: 3.85 });
  const rows = [
    ["K", "验证准确率"],
    ["1", "0.9712"],
    ["3", "0.9720"],
    ["5", "0.9718"],
    ["7", "0.9708"],
    ["9", "0.9705"],
  ];
  addSimpleTable(slide, rows, 7.45, 1.62, [1.25, 2.1], 0.48);
  addTextBox(slide, "结论\n\nK=3 的验证集准确率最高，因此作为最终模型参数。不同 K 值差距较小，说明 MNIST 类别结构较清晰。", 7.45, 4.85, 3.85, 1.1, { fill: C.white, fontSize: 12.5 });
  addFooter(slide);
}

// Slide 8
{
  const slide = pptx.addSlide();
  addHeader(slide, "TEST RESULT", "最终模型在测试集上达到 97.05% 准确率", 8);
  addMetric(slide, "最优参数", "K=3", "欧氏距离", 0.85, 1.45, 1.9);
  addMetric(slide, "测试准确率", "0.9705", "10000 张测试图", 2.95, 1.45, 1.9);
  addMetric(slide, "错误数量", "295", "错误率 2.95%", 5.05, 1.45, 1.9);
  slide.addImage({ path: FIG.confusion, x: 7.35, y: 1.32, w: 4.2, h: 4.2 });
  addBullets(slide, [
    "数字 0、1、6 的识别效果较好。",
    "数字 8 的召回率相对较低，为 0.9384。",
    "整体 macro F1-score 和 weighted F1-score 均约为 0.970。"
  ], 0.95, 3.05, 5.65, 1.55, { fontSize: 13.5 });
  addFooter(slide);
}

// Slide 9
{
  const slide = pptx.addSlide();
  addHeader(slide, "ERROR ANALYSIS", "误分类主要出现在形状相近的数字之间", 9);
  const rows = [
    ["真实", "预测", "次数"],
    ["7", "1", "21"],
    ["4", "9", "19"],
    ["8", "3", "16"],
    ["2", "7", "13"],
    ["3", "5", "13"],
    ["8", "5", "11"],
  ];
  addSimpleTable(slide, rows, 0.9, 1.55, [1.1, 1.1, 1.1], 0.47);
  addTextBox(slide, "错误原因分析\n\n手写数字存在较大个体差异。部分 7 写得较直时接近 1；4 和 9 在结构上相似；3、5、8 都包含弧线或封闭结构，因此容易混淆。", 4.55, 1.55, 3.8, 2.1, { fill: C.white, fontSize: 12.5 });
  addTextBox(slide, "交互式演示\n\nStreamlit 界面支持上传手写数字图片识别，也可以直接运行测试集评估并展示已有实验图表。", 8.75, 1.55, 3.35, 2.1, { fill: C.white, fontSize: 12.5 });
  addBullets(slide, [
    "KNN 主要根据像素距离判断相似性。",
    "它缺少对笔画结构和局部特征的主动提取能力。",
    "这解释了为什么形状相近的数字更容易被混淆。"
  ], 0.95, 4.75, 10.6, 1.0, { fontSize: 13.5 });
  addFooter(slide);
}

// Slide 10
{
  const slide = pptx.addSlide();
  addHeader(slide, "CONCLUSION", "KNN 简单直观，但预测计算量较大", 10);
  addTextBox(slide, "项目结论\n\nKNN 在 MNIST 手写数字识别任务上取得了较好效果。正式实验中，K=3 的模型在测试集上达到 97.05% 的准确率，说明传统机器学习方法也能完成较高质量的图像分类。", 0.9, 1.55, 5.3, 1.85, { fill: C.white, fontSize: 13 });
  addTextBox(slide, "学习体会\n\n数据预处理、验证集调参和错误分析同样重要。完整机器学习项目不只是模型能运行，还需要可复现实验、可视化结果和清晰的展示方式。", 6.65, 1.55, 5.3, 1.85, { fill: C.white, fontSize: 13 });
  addSectionLabel(slide, "后续改进方向", 0.9, 4.15, 1.8);
  addBullets(slide, [
    "尝试距离加权 KNN，增强近邻样本影响。",
    "使用 PCA 降维，降低预测计算量。",
    "与 SVM、决策树、随机森林等方法进行对比。",
    "继续优化上传图片预处理流程。"
  ], 1.0, 4.55, 10.3, 1.2, { fontSize: 13.5 });
  addFooter(slide);
}

pptx.writeFile({ fileName: OUT_FILE });
