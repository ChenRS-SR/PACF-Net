# PAC-NET 采集系统配置指南

> 本文档用于配置采集系统，修改摄像头参数、端口设置等。

## 快速开始

### 1. 环境安装

```bash
# 创建conda环境
conda create -n pacnet_collector python=3.10
conda activate pacnet_collector

# 安装依赖
pip install -r requirements-collector.txt
```

### 2. 相机设备号检测

运行以下命令查看可用摄像头：

```bash
python check_cameras.py
```

或手动在Python中检查：

```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"设备 {i}: 可用")
        cap.release()
```

### 3. 修改配置

编辑 `configs/collector_config.py`：

```python
# 相机设备配置
CAMERAS = {
    "ids": {
        "enabled": True,
        "device_ids": [0, 2, 3],  # 修改为你的设备号
        "resolution": (1920, 1080),
    },
    "computer": {
        "enabled": True,
        "device_id": 1,  # 修改为你的旁轴相机设备号
        "resolution": (1920, 1080),
    },
    "fotric": {
        "enabled": True,
        "ip": "192.168.1.100",  # 修改为你的红外相机IP
        "port": 10080,
    }
}

# OctoPrint配置
OCTOPRINT = {
    "url": "http://127.0.0.1:5000",
    "api_key": "你的API_KEY"
}
```

### 4. 运行采集系统

```bash
python hardware/ids_websocket.py
```

---

## 详细配置说明

### 相机配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `enabled` | 是否启用该相机 | `True` / `False` |
| `device_id` | 摄像头设备号 | `0`, `1`, `2`... |
| `resolution` | 采集分辨率 | `(1920, 1080)` |
| `use_dshow` | Windows下使用DirectShow加速 | `True` |
| `buffer_size` | 缓冲区大小（越小延迟越低） | `1` |

### OctoPrint配置

1. 打开 OctoPrint 网页界面
2. 进入 **设置** → **API**
3. 复制 **Global API Key**
4. 填入配置文件

### 标准化采集参数

```python
# 高度区间配置 (起始高度, 结束高度, 速度%, 流量%, 描述)
HEIGHT_RANGES = [
    (0, 5, 100, 100, "默认参数区"),
    (5, 10, 50, 75, "低速低流量"),
    # ...
]
```

---

## 常见问题

### Q1: 摄像头打不开？

**检查步骤：**
1. 运行 `python check_cameras.py` 查看可用设备
2. 检查设备是否被其他程序占用
3. 尝试修改 `use_dshow` 为 `False`

### Q2: 坐标不更新？

**检查：**
1. OctoPrint URL是否正确
2. API Key是否有效
3. 打印机状态是否为Printing

### Q3: 红外相机连不上？

**检查：**
1. 相机IP地址是否正确
2. 网络是否连通（ping测试）
3. 相机是否已开机

---

## 分支说明

- `master`: 稳定版本（采集版）
- `collector`: 采集专用版（当前分支）
- `develop`: 开发版（包含train/test/model）
