# PACF-NET: 多源异构数据融合的FDM打印缺陷诊断

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PACF-NET (Process-Aware Cross-modal Fusion Network) 是一种用于FDM (熔融沉积成型) 3D打印过程质量诊断的深度学习方法。

## 🌟 核心特性

- **多源异构数据融合**: 同时处理 IDS随轴相机、RGB旁轴相机、热成像三种模态
- **因果推理机制**: 基于工艺参数的交叉注意力融合
- **域适应对齐**: MMD损失实现RGB与IDS特征分布对齐
- **多任务学习**: 同时诊断流量、速度、Z轴偏移、温度四个工艺参数

## 📁 项目结构

```
pacnet_project/
├── Core Scripts (核心脚本)
│   ├── train.py              # 训练入口 (支持消融实验)
│   ├── model.py              # PACF-NET模型定义
│   ├── dataset.py            # 数据集加载
│   ├── eval_matrix.py        # 评估与混淆矩阵
│   ├── visualize.py          # 可视化 (t-SNE, Attention)
│   └── predict.py            # 实时预测/闭环验证
│
├── docs/                     # 文档
│   ├── ablation.md           # 消融实验指南
│   └── visualization.md      # 可视化说明
│
├── hardware/                 # 硬件驱动
│   ├── fotric_driver.py      # 热像仪驱动
│   ├── ids_websocket.py      # IDS相机采集
│   ├── coordinator.py        # XYZ坐标获取
│   └── vibration_sensor.py   # 振动传感器
│
└── [其他目录]
    ├── configs/              # 配置
    ├── saved_models/         # 模型保存
    └── logs/                 # 日志
```

详细结构说明见 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

## 🚀 快速开始

### 环境安装

```bash
# 创建环境
conda create -n pacfnet python=3.10
conda activate pacfnet

# 安装依赖
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm pillow
```

### 1. 训练模型

```bash
python train.py
```

按提示选择消融实验变体：
- 1: Full Model (完整模型)
- 2: Variant A - No MMD
- 3: Variant B - Concat Only
- 4: Variant C-1 - RGB+IDS (无热成像)
- 5: Variant C-2 - IDS Only (单模态)

### 2. 评估模型

```bash
python eval_matrix.py --model_path saved_models/xxx/model_full.pt
```

### 3. 可视化分析

```bash
# t-SNE特征分布 + Attention Map
python visualize.py --model_path saved_models/xxx/model_full.pt --mode all
```

## 🧪 消融实验

验证各组件有效性：

| 变体 | MMD | Attention | 热成像 | 预期对比 |
|------|-----|-----------|--------|----------|
| Full | ✅ | ✅ | ✅ | 基线 |
| A | ❌ | ✅ | ✅ | 验证MMD必要性 |
| B | ✅ | ❌(拼接) | ✅ | 验证Attention有效性 |
| C-1 | ✅ | ✅ | ❌ | 验证热成像贡献 |
| C-2 | ✅ | ✅ | ❌(仅IDS) | 单模态基线 |

详细说明见 [docs/ablation.md](./docs/ablation.md)

## 📊 可视化工具

### t-SNE 特征分布
- 验证MMD对齐效果（RGB vs IDS分布）
- 验证类别可分离性

### Attention Map
- 空间注意力热力图
- 多源特征融合权重可视化

详细说明见 [docs/visualization.md](./docs/visualization.md)

## 🔧 硬件采集

采集程序位于 `hardware/` 目录：

```bash
cd hardware
python ids_websocket.py  # 启动采集程序
```

支持：
- IDS uEye 工业相机
- Fotric 628CH 热像仪
- M114 运动控制器 (XYZ坐标)

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) | 项目结构详解 |
| [docs/ablation.md](./docs/ablation.md) | 消融实验完整指南 |
| [docs/visualization.md](./docs/visualization.md) | 可视化工具使用说明 |

## 📖 引用

```bibtex
@software{pacfnet2024,
  title={PACF-NET: Process-Aware Cross-modal Fusion Network for FDM Defect Diagnosis},
  author={Your Name},
  year={2024}
}
```

## 📧 联系

如有问题，请提交 Issue 或联系项目维护者。
