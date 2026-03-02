# PACF-NET 消融实验 (Ablation Study) 指南

本文档说明如何进行消融实验以验证模型各组件的有效性。

---

## 消融实验设计

### 变体说明

| 变体 | 名称 | 说明 | 保存文件名 |
|------|------|------|-----------|
| **Full** | 完整模型 | 包含所有组件 (Attention + MMD + 全模态) | `model_full.pt` |
| **Variant A** | 无 MMD | 禁用 MMD 损失，验证特征对齐的必要性 | `model_no_mmd.pt` |
| **Variant B** | 拼接融合 | 用简单拼接替代交叉注意力机制 | `model_concat.pt` |
| **Variant C-1** | RGB+IDS (No Thermal) | 只使用可见光模态 (IDS + RGB)，屏蔽热成像 | `model_rgb_only.pt` |
| **Variant C-2** | IDS Only | 只使用 IDS 随轴相机模态 | `model_ids_only.pt` |

---

## 使用方法

### 启动训练并选择变体

```bash
cd pacnet_project
python train.py
```

运行后会显示交互式菜单：

```
======================================================================
请选择要训练的模型变体 (Ablation Study)
======================================================================
1. Full Model (完整模型)              -> model_full.pt
2. Variant A - No MMD (无MMD)         -> model_no_mmd.pt
3. Variant B - Concat Only (拼接)     -> model_concat.pt
4. Variant C-1 - RGB+IDS (无热成像)   -> model_rgb_only.pt
5. Variant C-2 - IDS Only (单模态)    -> model_ids_only.pt
======================================================================
请输入选项 (1-5): 
```

输入对应数字即可开始训练指定变体。

---

## 各变体详细说明

### 1. Full Model (完整模型)

```
输入: 1
```

- **结构**: 完整的 PACF-NET
- **融合**: Cross-Attention + 空间位置编码
- **对齐**: MMD 损失
- **模态**: IDS + RGB + Thermal
- **用途**: 基线模型，用于对比其他变体

---

### 2. Variant A - No MMD

```
输入: 2
```

- **修改**: `alpha_mmd = 0`
- **结构**: 完整模型，但禁用 MMD 损失
- **验证目标**: MMD 特征对齐是否提升跨域泛化能力
- **预期结果**: 验证集准确率可能下降，t-SNE 图中 RGB/IDS 分布分离

**实现方式**:
```python
# train.py 中自动设置
effective_mmd_weight = 0.0  # 强制为0
```

---

### 3. Variant B - Concat Only

```
输入: 3
```

- **修改**: 使用 `ConcatFusion` 替代 `CrossAttentionFusion`
- **融合方式**:
  ```python
  # 拼接所有特征
  combined = torch.cat([f_local, f_rgb, f_thermal, intent_feat], dim=1)
  # 投影降维
  fused = self.fusion_proj(combined)  # (B, 512)
  ```
- **验证目标**: 交叉注意力机制是否优于简单拼接
- **预期结果**: 准确率可能略低于完整模型

**结构对比**:

| 组件 | Full | Concat Only |
|------|------|-------------|
| 特征交互 | Cross-Attention (动态加权) | Concat + Linear (静态) |
| 参数量 | 较多 (Q/K/V 投影) | 较少 |
| 空间感知 | 是 (147 tokens) | 否 (全局池化) |

---

### 4. Variant C-1 - RGB+IDS (No Thermal)

```
输入: 4
```

- **修改**: 屏蔽 Thermal 模态（置零），保留 IDS + RGB
- **使用的模态**:
  - ✅ IDS (随轴相机)
  - ✅ Computer RGB (旁轴相机)
  - ❌ Fotric 热像 (置零)
  - ❌ Thermal 灰度 (置零)
- **验证目标**: 热成像模态的贡献（可见光 vs 可见光+热成像）
- **预期结果**: 温度相关任务性能下降，验证热成像的必要性

**实现方式**:
```python
# model.py _mask_modalities 方法
if self.variant == 'rgb_only':
    # 保留 IDS + RGB，只屏蔽 Thermal
    batch_data['fotric'] = torch.zeros_like(batch_data['fotric'])
    batch_data['thermal'] = torch.zeros_like(batch_data['thermal'])
```

---

### 5. Variant C-2 - IDS Only

```
输入: 5
```

- **修改**: 只保留 IDS 模态，其他置零
- **使用的模态**:
  - ✅ IDS (随轴相机)
  - ❌ Computer RGB (置零)
  - ❌ Fotric 热像 (置零)
  - ❌ Thermal 灰度 (置零)
- **验证目标**: 单模态性能上限
- **预期结果**: 性能最低，验证多模态融合的必要性

---

## 实验流程建议

### 推荐实验顺序

```bash
# 1. 首先训练完整模型作为基线
python train.py  # 选择 1

# 2. 训练变体A (无MMD) - 验证MMD有效性
python train.py  # 选择 2

# 3. 训练变体B (拼接) - 验证Attention有效性
python train.py  # 选择 3

# 4. 训练变体C-1 (RGB+IDS, No Thermal) - 验证热成像贡献
python train.py  # 选择 4

# 5. 训练变体C-2 (IDS Only) - 单模态基线
python train.py  # 选择 5
```

### 结果对比表

训练完成后，记录各变体的验证集准确率：

| 变体 | Flow Rate | Feed Rate | Z Offset | Hotend | Average |
|------|-----------|-----------|----------|--------|---------|
| Full | ??.??% | ??.??% | ??.??% | ??.??% | ??.??% |
| No MMD | ??.??% | ??.??% | ??.??% | ??.??% | ??.??% |
| Concat | ??.??% | ??.??% | ??.??% | ??.??% | ??.??% |
| RGB+IDS | ??.??% | ??.??% | ??.??% | ??.??% | ??.??% |
| IDS Only | ??.??% | ??.??% | ??.??% | ??.??% | ??.??% |

### 结果解读

| 对比 | 预期发现 |
|------|---------|
| Full vs No MMD | MMD 应提升跨域泛化，尤其 Feed/Z 任务 |
| Full vs Concat | Attention 应优于简单拼接，尤其是复杂样本 |
| Full vs RGB+IDS | 热成像应显著提升 Hotend 等温度相关任务 |
| Full vs IDS Only | 多模态融合应大幅优于单模态 |

---

## 输出文件结构

每个变体的输出保存在独立的目录中：

```
saved_models/
├── full_20260208_120000/
│   ├── model_full.pt
│   ├── scaler.pkl
│   └── train.log
├── no-mmd_20260208_130000/
│   ├── model_no_mmd.pt
│   ├── scaler.pkl
│   └── train.log
├── concat-only_20260208_140000/
│   ├── model_concat.pt
│   ├── scaler.pkl
│   └── train.log
├── rgb-only_20260208_150000/
│   ├── model_rgb_only.pt
│   ├── scaler.pkl
│   └── train.log
└── ids-only_20260208_160000/
    ├── model_ids_only.pt
    ├── scaler.pkl
    └── train.log
```

---

## 可视化分析

训练完成后，使用可视化脚本对比各变体：

```bash
# 完整模型 t-SNE
python visualize.py \
    --model_path saved_models/full_20260208_120000/model_full.pt \
    --output_dir ablation_results/full

# 无 MMD 变体 t-SNE (观察 RGB/IDS 分离)
python visualize.py \
    --model_path saved_models/no-mmd_20260208_130000/model_no_mmd.pt \
    --output_dir ablation_results/no_mmd

# 对比混淆矩阵
eval_matrix.py \
    --model_path saved_models/full_20260208_120000/model_full.pt \
    --output_dir ablation_results/full
eval_matrix.py \
    --model_path saved_models/no-mmd_20260208_130000/model_no_mmd.pt \
    --output_dir ablation_results/no_mmd
```

---

## 注意事项

### 1. MMD Warmup 在变体A中的处理

- **变体A (No MMD)**: Warmup 阶段被禁用，因为 `alpha_mmd = 0`
- **其他变体**: 正常进行 Warmup (epoch 1-5)

### 2. 学习率建议

对于单模态变体 (C-1, C-2)，模型容量减小，建议：

```bash
python train.py --lr 5e-5 --epochs 80
```

### 3. 早停策略

消融实验中，各变体收敛速度可能不同，建议增加耐心值：

```bash
python train.py --patience 25
```

### 4. 复现性

为确保实验可复现，建议固定随机种子：

```python
# 在 train.py main() 开头添加
torch.manual_seed(42)
np.random.seed(42)
```

---

## 扩展实验

### 自定义消融

如需添加更多变体，修改 `model.py` 和 `train.py`:

1. **在 `model.py` 中添加新的 `_mask_modalities` 条件**
2. **在 `train.py` 的 `select_variant()` 中添加选项**

示例：添加 "无 Thermal" 变体

```python
# model.py
elif self.variant == 'no_thermal':
    batch_data['fotric'] = torch.zeros_like(batch_data['fotric'])
    batch_data['thermal'] = torch.zeros_like(batch_data['thermal'])

# train.py
def select_variant():
    # ... 添加选项 6
    print("6. Variant C-3 - No Thermal       -> model_no_thermal.pt")
    # ...
    elif choice == '6':
        return 'no_thermal', 'model_no_thermal.pt', None
```

---

## 引用

如果使用了消融实验框架，请引用：

```bibtex
@software{pacnet_ablation,
  title={PACF-NET Ablation Study Framework},
  author={Your Name},
  year={2024}
}
```
