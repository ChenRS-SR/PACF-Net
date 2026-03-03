# 采集版使用说明

## 与完整版的区别

此分支为**纯采集版**，已移除以下功能：

### ❌ 移除的功能
- 模型推理（predict）
- 闭环调控（CLOSE_LOOP）
- 自动参数调节
- 模型加载和初始化

### ✅ 保留的功能
- 三相机采集（IDS、旁轴、红外）
- 实时坐标追踪（M114）
- 标准化采集（自动调节温度/速度/流量）
- 时间戳同步
- 数据保存（CSV + 图像）

## 代码修改说明

在 `ids_websocket.py` 中，以下导入被注释：
```python
# from predict import predict_single, load_model
# from configs.config import config
```

以下功能被禁用：
- `CLOSE_LOOP` 恒为 `False`
- `INIT_MODEL` 相关代码被注释
- `auto_close_loop()` 函数被注释
- 闭环调控UI被隐藏

## 如需恢复闭环调控

请切换到 `develop` 分支使用完整功能。
