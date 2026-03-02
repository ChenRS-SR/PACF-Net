# PACF-NET 可视化工具使用说明

本文档说明如何使用 `visualize.py` 和 `eval_matrix.py` 两个可视化脚本分析训练好的模型。

---

## 目录

- [环境准备](#环境准备)
- [eval_matrix.py - 混淆矩阵评估](#eval_matrixpy---混淆矩阵评估)
- [visualize.py - 特征可视化](#visualizepy---特征可视化)
- [输出文件说明](#输出文件说明)
- [常见问题](#常见问题)

---

## 环境准备

确保已安装以下依赖：

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python tqdm pandas pillow
```

所有可视化脚本需要在训练好的模型目录下运行，模型目录应包含：
- `best_model.pt` - 训练好的模型权重
- `scaler.pkl` - 参数标准化器（防止数据泄露）

---

## eval_matrix.py - 混淆矩阵评估

### 功能说明

评估模型在测试集上的分类性能，生成 **4 张高清混淆矩阵图**（每个任务一张），包含：
- 归一化的混淆矩阵（百分比显示）
- Accuracy / Precision / Recall / F1-Score
- 数量+百分比双信息显示

### 使用方法

```bash
# 基本用法
python eval_matrix.py --model_path saved_models/20260208_042805/best_model.pt

# 指定输出目录
python eval_matrix.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --output_dir results/experiment1

# 完整参数
python eval_matrix.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --test_csv D:\\data\\test.csv \
    --data_root D:\\data \
    --output_dir eval_results \
    --batch_size 32 \
    --num_workers 4
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | **必填** | 模型文件路径 `.pt` |
| `--test_csv` | `data/test.csv` | 测试集 CSV 文件路径 |
| `--data_root` | `data/` | 数据根目录 |
| `--output_dir` | `eval_results` | 输出目录 |
| `--batch_size` | 32 | 推理批大小 |
| `--num_workers` | 4 | 数据加载线程数 |

### 输出示例

```
======================================================================
评估结果汇总
======================================================================
Task                 Accuracy  Precision     Recall   F1-Score
----------------------------------------------------------------------
Flow Rate              0.3256     0.3124     0.3256     0.3189
Feed Rate              0.2153     0.1987     0.2153     0.2056
Z Offset               0.0897     0.0856     0.0897     0.0874
Hotend Temp            0.2678     0.2543     0.2678     0.2605
----------------------------------------------------------------------
Average                0.2246       ----       ----     0.2181
======================================================================
```

### 输出文件

```
eval_results/
├── cm_flow_rate.png          # 流量混淆矩阵
├── cm_feed_rate.png          # 速度混淆矩阵
├── cm_z_offset.png           # Z轴偏移混淆矩阵
├── cm_hot_end.png            # 温度混淆矩阵
└── cm_all_tasks.png          # 四合一综合图
```

---

## visualize.py - 特征可视化

### 功能说明

提供两种可视化功能：

1. **t-SNE 特征分布图** - 验证 MMD 对齐效果和特征可分离性
2. **Attention Map 热力图** - 可视化模型的空间注意力分布

### 使用方法

#### 模式 1: 仅生成 t-SNE 图

```bash
python visualize.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --mode tsne

# 限制样本数（加速）
python visualize.py \
    --model_path saved_models/.../best_model.pt \
    --mode tsne \
    --max_samples 500
```

#### 模式 2: 仅生成 Attention Map

```bash
# 随机选择 5 个样本（默认）
python visualize.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --mode attention

# 指定特定样本
python visualize.py \
    --model_path saved_models/.../best_model.pt \
    --mode attention \
    --sample_indices 10,25,100

# 随机选择更多样本
python visualize.py \
    --model_path saved_models/.../best_model.pt \
    --mode attention \
    --num_attention_samples 10
```

#### 模式 3: 生成全部可视化（推荐）

```bash
python visualize.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --mode all \
    --output_dir visualization
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | **必填** | 模型文件路径 |
| `--mode` | `all` | 可视化模式: `tsne` / `attention` / `all` |
| `--test_csv` | `data/test.csv` | 测试集 CSV 路径 |
| `--data_root` | `data/` | 数据根目录 |
| `--output_dir` | `visualization` | 输出目录 |
| `--max_samples` | 1000 | t-SNE 最大样本数 |
| `--sample_indices` | None | 指定 Attention 样本索引，如 `"10,25,100"` |
| `--num_attention_samples` | 5 | 随机选择 Attention 样本数 |
| `--batch_size` | 32 | 推理批大小 |
| `--num_workers` | 4 | 数据加载线程数 |

### t-SNE 图表解读

每张 t-SNE 图包含两个子图：

#### 左图：MMD 对齐前 (Before Alignment)
- 蓝色点：`RGB` 特征
- 红色点：`Local (IDS)` 特征
- **期望效果**：两种颜色混合在一起 → MMD 有效

#### 右图：融合后按类别着色 (After Fusion)
- 红色：`Low` 类别
- 绿色：`Normal` 类别
- 蓝色：`High` 类别
- **期望效果**：同类聚集、异类分离 → 任务可学习

### Attention Map 图表解读

每张 Attention Map 包含：

| 位置 | 内容 | 说明 |
|------|------|------|
| 第一行 | 原始输入图像 | IDS (448×448)、RGB (224×224)、Thermal (224×224) |
| 第二行 | Attention 权重条形图 | 147维 = 49(Local) + 49(RGB) + 49(Thermal)，颜色区分来源 |
| 第三行 | 空间 Attention 热力图 | 三个 7×7 热力图，显示空间关注区域 |

**颜色编码**：
- 珊瑚色 (coral) = Local (IDS) 特征
- 皇家蓝 (royalblue) = RGB 特征
- 金色 (gold) = Thermal 特征

---

## 输出文件说明

### eval_matrix.py 输出

| 文件名 | 尺寸 | 内容 |
|--------|------|------|
| `cm_flow_rate.png` | 8×6 英寸 | 流量任务混淆矩阵 |
| `cm_feed_rate.png` | 8×6 英寸 | 速度任务混淆矩阵 |
| `cm_z_offset.png` | 8×6 英寸 | Z轴偏移混淆矩阵 |
| `cm_hot_end.png` | 8×6 英寸 | 温度任务混淆矩阵 |
| `cm_all_tasks.png` | 16×12 英寸 | 四合一综合图 |

**分辨率**：所有图片 DPI=300，适合论文插入

### visualize.py 输出

```
visualization/
├── tsne_Flow_Rate.png           # 流量 t-SNE 分布
├── tsne_Feed_Rate.png           # 速度 t-SNE 分布
├── tsne_Z_Offset.png            # Z轴 t-SNE 分布
├── tsne_Hotend_Temp.png         # 温度 t-SNE 分布
├── tsne_all_tasks.png           # t-SNE 四合一
└── attention_maps/
    ├── attention_sample_10.png  # 样本 10 的 Attention Map
    ├── attention_sample_25.png  # 样本 25 的 Attention Map
    └── ...
```

---

## 常见问题

### Q1: 运行 t-SNE 时内存不足

**解决**：减少 `--max_samples` 参数
```bash
python visualize.py --model_path ... --mode tsne --max_samples 500
```

### Q2: 找不到 scaler.pkl

**原因**：模型目录下没有 scaler 文件

**解决**：确保使用 `train.py` 训练生成的模型目录，或手动指定 scaler 路径（需修改代码）

### Q3: Attention Map 全是一个颜色

**原因**：模型可能没有学到有效的注意力

**检查**：
1. 确认训练收敛（CE Loss < 1.0）
2. 检查 MMD Loss 是否正常下降
3. 增加训练 epoch 数

### Q4: t-SNE 图显示 RGB/Local 完全分离

**原因**：MMD 对齐效果不佳

**建议**：
1. 增加 `alpha_mmd` 权重（如从 2.0 提高到 5.0）
2. 延长 Warmup 阶段（从 5 epoch 到 10 epoch）
3. 检查数据预处理是否一致

---

## 完整工作流示例

```bash
# 1. 训练模型
python train.py --epochs 100 --mmd_weight 2.0

# 2. 评估模型（生成混淆矩阵）
python eval_matrix.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --output_dir results/eval

# 3. 可视化分析（t-SNE + Attention）
python visualize.py \
    --model_path saved_models/20260208_042805/best_model.pt \
    --mode all \
    --output_dir results/vis \
    --max_samples 800 \
    --num_attention_samples 8
```

---

## 引用

如果使用这些可视化工具，请引用：

```bibtex
@software{pacnet_visualization,
  title={PACF-NET Visualization Tools},
  author={Your Name},
  year={2024}
}
```
