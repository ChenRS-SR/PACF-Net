# PAC-NET 数据采集系统

> 用于3D打印质量监测的数据采集系统，支持三相机（随轴、旁轴可见光、红外）同步采集。

## 功能特性

- ✅ 三相机同步采集（IDS/随轴、旁轴可见光、Fotric红外）
- ✅ 实时坐标追踪（通过OctoPrint M114）
- ✅ 标准化采集模式（自动调节温度、速度、流量）
- ✅ 时间戳同步（图像与坐标时间对齐）
- ✅ 实时状态监控（坐标、温度、打印机状态）

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n pacnet_collector python=3.10
conda activate pacnet_collector

# 安装依赖
pip install -r requirements-collector.txt
```

### 2. 硬件检查

#### 检查摄像头

```bash
python check_cameras.py
```

根据输出修改 `configs/collector_config.py` 中的设备号。

#### 配置OctoPrint

1. 打开 OctoPrint 网页界面 → 设置 → API
2. 复制 Global API Key
3. 粘贴到 `configs/collector_config.py` 中的 `OCTOPRINT_CONFIG.api_key`

### 3. 运行采集系统

```bash
python hardware/ids_websocket.py
```

## 配置说明

所有配置集中在 `configs/collector_config.py`：

| 配置项 | 说明 | 修改位置 |
|--------|------|----------|
| 相机设备号 | 摄像头ID | `CAMERA_CONFIG.ids.device_ids` |
| OctoPrint地址 | 打印机连接地址 | `OCTOPRINT_CONFIG.url` |
| API Key | OctoPrint认证 | `OCTOPRINT_CONFIG.api_key` |
| 采集模式 | 普通/标准化 | `ACQUISITION_CONFIG.mode` |
| 红外相机IP | Fotric网络地址 | `CAMERA_CONFIG.fotric.ip` |

详细配置说明见 [CONFIG.md](CONFIG.md)

## 数据格式

采集数据保存为CSV，包含以下字段：

```
ids_image_path, computer_image_path, timestamp, 
img_timestamp, coord_timestamp, time_diff_ms,
current_x, current_y, current_z,
flow_rate, feed_rate, z_offset, target_hotend, actual_hotend, bed_temp, image_count, ...
```

## 目录结构

```
pacnet_project/
├── hardware/
│   └── ids_websocket.py      # 主采集程序
├── configs/
│   └── collector_config.py   # 配置文件
├── check_cameras.py          # 摄像头检测工具
├── CONFIG.md                 # 详细配置说明
└── requirements-collector.txt # 依赖文件
```

## 分支说明

- `collector` (当前分支): 采集专用版，仅包含数据采集功能
- `master`: 稳定采集版
- `develop`: 完整开发版（含train/test/model）

## 常见问题

**Q: 坐标不更新？**
A: 检查OctoPrint URL和API Key，确保打印机状态为"Printing"

**Q: 摄像头打不开？**
A: 运行 `check_cameras.py` 查看正确的设备号

**Q: 红外相机连不上？**
A: 检查相机IP地址和网络连通性

## License

MIT License
