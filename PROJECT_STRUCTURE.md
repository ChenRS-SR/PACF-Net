# PACF-NET 项目结构说明

## 目录组织

```
pacnet_project/
├── README.md                      # 项目主说明文档
├── PROJECT_STRUCTURE.md           # 本文件：目录结构说明
│
├── Core (核心代码)
│   ├── train.py                   # 训练入口脚本
│   ├── model.py                   # PACF-NET 模型定义
│   ├── dataset.py                 # 数据集和数据加载
│   ├── eval_matrix.py             # 评估脚本（混淆矩阵）
│   ├── visualize.py               # 可视化脚本（t-SNE, Attention）
│   └── predict.py                 # 预测脚本（实时推理/闭环验证）
│
├── docs/                          # 文档目录
│   ├── ablation.md                # 消融实验指南
│   └── visualization.md           # 可视化工具使用说明
│
├── configs/                       # 配置文件目录
│   └── config.py                  # 配置参数（硬件采集程序使用）
│
├── hardware/                      # 硬件驱动目录
│   ├── fotric_driver.py           # Fotric 热像仪驱动 (原Fotric_628ch_enhanced.py)
│   ├── ids_websocket.py           # IDS 相机采集程序（最新）
│   ├── ids_driver_old.py          # IDS 相机旧版驱动 (原VB02_ids.py)
│   ├── coordinator.py             # XYZ坐标获取辅助 (原m114_coordinator.py)
│   └── vibration_sensor.py        # 振动传感器驱动 (原device_model.py)
│
├── checkpoints/                   # 训练检查点
├── saved_models/                  # 保存的训练模型
├── logs/                          # 训练日志
├── evaluation/                    # 评估结果
├── imgs/                          # 图片资源
├── ui/                            # UI界面相关
├── utils/                         # 工具函数
└── __pycache__/                   # Python缓存
```

---

## 核心代码说明

### train.py
- **功能**: 模型训练入口
- **特点**: 支持消融实验变体选择（交互式菜单）
- **输出**: 训练好的模型保存到 `saved_models/`

### model.py
- **功能**: PACF-NET 网络架构定义
- **关键类**:
  - `VisualBackbone`: 三塔ResNet18视觉骨干
  - `CrossAttentionFusion`: 交叉注意力融合（完整版）
  - `ConcatFusion`: 简单拼接融合（消融变体B）
  - `PACFNet`: 完整网络（支持多种变体）

### dataset.py
- **功能**: FDMDefectDataset 数据加载
- **特点**: 支持 ParameterScaler 防止数据泄露
- **模态**: IDS + RGB + Thermal (4通道)

### eval_matrix.py
- **功能**: 测试集评估，生成混淆矩阵
- **输出**: 4张高清混淆矩阵图 + 综合图
- **指标**: Accuracy, Precision, Recall, F1

### visualize.py
- **功能**: 特征可视化
- **支持**:
  - t-SNE 特征分布图（验证MMD效果）
  - Attention Map 热力图（空间注意力可视化）

### predict.py
- **功能**: 实时预测和闭环验证
- **模式**: 单张预测 / 批量预测 / 实时预测
- **用途**: 闭环调控系统集成

---

## 消融实验变体

| 变体 | 文件名 | 说明 |
|------|--------|------|
| Full | `model_full.pt` | 完整模型 |
| A - No MMD | `model_no_mmd.pt` | 禁用MMD损失 |
| B - Concat | `model_concat.pt` | 拼接替代注意力 |
| C-1 - RGB+IDS | `model_rgb_only.pt` | 无热成像 |
| C-2 - IDS Only | `model_ids_only.pt` | 单模态 |

---

## 快速开始

### 1. 训练模型
```bash
python train.py
# 按提示选择变体 (1-5)
```

### 2. 评估模型
```bash
python eval_matrix.py --model_path saved_models/.../model_full.pt
```

### 3. 可视化分析
```bash
python visualize.py --model_path saved_models/.../model_full.pt --mode all
```

### 4. 实时预测（闭环）
```bash
python predict.py --model_path saved_models/.../model_full.pt --mode realtime
```

---

## 硬件采集程序

### ids_websocket.py（最新采集程序）
- **功能**: IDS相机 + Fotric热像仪同步采集
- **特点**: WebSocket通信，实时传输
- **依赖**: 
  - `hardware.fotric_driver`
  - `hardware.coordinator` (XYZ坐标)
  - `configs.config`

### 文件迁移说明
旧文件 → 新位置:
- `Fotric_628ch_enhanced.py` → `hardware/fotric_driver.py`
- `VB02_ids.py` → `hardware/ids_driver_old.py`
- `VB02_ids_websocket.py` → `hardware/ids_websocket.py`
- `m114_coordinator.py` → `hardware/coordinator.py`
- `device_model.py` → `hardware/vibration_sensor.py`
