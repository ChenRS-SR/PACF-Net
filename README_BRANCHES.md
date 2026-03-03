# 分支说明

本项目有三个主要分支，分别用于不同目的：

## 分支结构

```
master (稳定采集版)
  │
  ├── 适用于：外部用户使用，仅需数据采集
  ├── 功能：三相机采集、标准化采集、坐标同步
  └── 特点：精简代码，仅包含采集必需的文件

develop (完整开发版)  
  │
  ├── 适用于：开发/研究人员，需要训练模型
  ├── 功能：采集 + 训练 + 测试 + 模型推理
  └── 特点：完整代码库，包含所有实验代码

collector (采集测试版)
     │
     ├── 适用于：采集功能开发和测试
     ├── 功能：最新的采集功能更新
     └── 特点：可能包含不稳定的新功能
```

## 分支详情

### `master` - 稳定采集版

**用途**: 给外部用户使用的稳定版本

**特点**:
- ✅ 经过测试的稳定采集功能
- ✅ 完善的配置文档
- ✅ 简化的依赖要求
- ✅ 无需深度学习环境

**使用**:
```bash
git clone -b master <repo-url>
```

---

### `develop` - 完整开发版

**用途**: 开发人员的完整工作版本

**特点**:
- 🔬 包含完整的深度学习模型
- 🔬 支持模型训练和评估
- 🔬 包含消融实验和可视化
- 🔬 支持闭环调控

**文件差异** (相比master额外包含):
```
train.py              # 模型训练
eval_matrix.py        # 评估矩阵
model.py              # 网络结构
predict.py            # 推理代码
dataset.py            # 数据加载
visualize.py          # 可视化
visualization/        # 可视化结果
```

**使用**:
```bash
git clone -b develop <repo-url>
```

---

### `collector` - 采集测试版

**用途**: 采集功能的前沿开发

**特点**:
- 🧪 最新的采集功能
- 🧪 实验性的配置选项
- 🧪 可能不稳定

**使用**:
```bash
git clone -b collector <repo-url>
```

## 版本关系

```
      collector (测试)
           │
           ▼ (功能稳定后合并)
      master (稳定采集版)
           │
           ▼ (添加训练代码)
      develop (完整开发版)
```

## 推送策略

### 个人使用（develop分支）
```bash
git checkout develop
# 修改代码
git add .
git commit -m "Your changes"
git push origin develop
```

### 采集版更新（master分支）
```bash
# 从collector测试版合并稳定功能
git checkout master
git merge collector
git push origin master
```

### 发布采集版给他人
```bash
# 用户只需执行
git clone -b master https://github.com/yourname/pacnet-collector.git
cd pacnet-collector
pip install -r requirements-collector.txt
python hardware/ids_websocket.py
```

## 配置差异

| 配置项 | master/collector | develop |
|--------|-----------------|---------|
| 相机配置 | `configs/collector_config.py` | 代码内配置 |
| OctoPrint | 配置文件设置 | 硬编码 |
| 模型推理 | ❌ 不支持 | ✅ 支持 |
| 闭环调控 | 部分支持 | 完整支持 |
| 训练功能 | ❌ 不支持 | ✅ 支持 |

## 依赖差异

**采集版** (`requirements-collector.txt`):
```
numpy, opencv-python, pillow, requests, scipy, tkinter
```

**完整版** (`requirements.txt`):
```
# 包含采集版所有依赖 +
torch, torchvision, timm, transformers, albumentations
```
