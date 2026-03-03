"""
PAC-NET 采集系统配置文件
修改以下参数以适配你的硬件环境
"""

# ==================== 相机配置 ====================

CAMERA_CONFIG = {
    # 随轴相机（IDS 或替代摄像头）
    "ids": {
        "enabled": True,           # 是否启用
        "device_ids": [0, 2, 3],   # 尝试的设备编号列表（按顺序尝试）
        "resolution": (1920, 1080), # 分辨率 (宽, 高)
        "use_dshow": True,         # 使用DirectShow后端（Windows下更快）
        "buffer_size": 1,          # 缓冲区大小
        "skip_brightness_check": True,  # 是否跳过亮度检测（加速但可能选到虚拟摄像头）
        "brightness_threshold": 6, # 亮度阈值
        "fps": 30,
    },
    
    # 旁轴可见光相机（电脑摄像头）
    "computer": {
        "enabled": True,
        "device_id": 1,            # 设备编号（通过check_cameras.py查看）
        "resolution": (1920, 1080),
        "fps": 30,
        "buffer_size": 1,          # 缓冲区大小（越小延迟越低）
        "use_dshow": True,         # 使用DirectShow后端（Windows下更快）
        "disable_autofocus": True, # 禁用自动对焦（大幅加快初始化）
        "disable_auto_exposure": True,
        "disable_auto_whitebalance": True,
        "warm_up_frames": 5,       # 预热帧数（清除缓冲区）
    },
    
    # 旁轴红外相机（Fotric）
    "fotric": {
        "enabled": True,
        "ip": "192.168.1.100",     # 相机IP地址（根据实际修改）
        "port": 10080,
        "username": "admin",
        "password": "admin",
        "simulation_mode": False,
        "high_resolution": True,
    }
}


# ==================== OctoPrint 配置 ====================

OCTOPRINT_CONFIG = {
    "url": "http://127.0.0.1:5000",  # OctoPrint地址
    "api_key": "UGjrS2T5n_48GF0YsWADx1EoTILjwn7ZkeWUfgGvW2Q",  # 替换为你的API Key
}


# ==================== 采集参数配置 ====================

ACQUISITION_CONFIG = {
    # 采集模式: "普通采集" 或 "标准化采集"
    "mode": "标准化采集",
    
    # 标准化采集参数
    "standardized": {
        # 9组实验的固定参数：温度(°C), Z偏移(mm)
        "experiments": {
            1: {"temp": 180, "z_offset": -0.15, "name": "180°C, Z-0.15"},
            2: {"temp": 180, "z_offset": 0.00, "name": "180°C, Z0.00"},
            3: {"temp": 180, "z_offset": 0.25, "name": "180°C, Z+0.25"},
            4: {"temp": 210, "z_offset": -0.15, "name": "210°C, Z-0.15"},
            5: {"temp": 210, "z_offset": 0.00, "name": "210°C, Z0.00"},
            6: {"temp": 210, "z_offset": 0.25, "name": "210°C, Z+0.25"},
            7: {"temp": 240, "z_offset": -0.15, "name": "240°C, Z-0.15"},
            8: {"temp": 240, "z_offset": 0.00, "name": "240°C, Z0.00"},
            9: {"temp": 240, "z_offset": 0.25, "name": "240°C, Z+0.25"},
        },
        
        # 各Z高度区间的速度和流量配置
        # 格式: (起始高度mm, 结束高度mm, 速度%, 流量%, 描述)
        "height_ranges": [
            (0, 5, 100, 100, "默认参数区"),        # 0-5mm: 默认参数
            (5, 10, 50, 75, "低速低流量"),
            (10, 15, 50, 100, "低速中流量"),
            (15, 20, 50, 125, "低速高流量"),
            (20, 25, 100, 75, "中速低流量"),
            (25, 30, 100, 100, "中速中流量"),
            (30, 35, 100, 125, "中速高流量"),
            (35, 40, 150, 75, "高速低流量"),
            (40, 45, 150, 100, "高速中流量"),
            (45, 50, 150, 125, "高速高流量"),
        ],
    },
    
    # 普通采集参数
    "normal": {
        "auto_change_params": False,  # 是否自动改变参数
        "change_interval": 120,       # 参数改变间隔（秒）
    },
}


# ==================== 记录控制配置 ====================

RECORDING_CONFIG = {
    # 保存路径配置
    "save_directory": "./data",
    
    # CSV文件配置
    "csv_filename": "print_data.csv",
    
    # 图像质量
    "image_quality": 95,
    
    # 时间同步配置
    "time_sync": {
        "max_time_diff_ms": 200,  # 最大允许的时间差（毫秒）
    },
}


# ==================== 调参配置（用于闭环调控） ====================

TUNING_CONFIG = {
    # 基础Z偏移（调平值）
    "primary_z_off": -2.55,
    
    # 参数调节范围
    "param_ranges": {
        "z_offset": {"min": -0.08, "max": 0.32, "step": 0.04},
        "flow_rate": {"min": 20, "max": 200, "step": 10},
        "feed_rate": {"min": 20, "max": 200, "step": 10},
        "hotend_temp": {"min": 120, "max": 280, "step": 10},
    },
}


# ==================== 标定配置 ====================

CALIBRATION_CONFIG = {
    # 标定点范围
    "points": {
        'X': [80, 130, 180],
        'Y': [20, 100, 180],
        'Z': [2, 36, 70]
    },
    
    # 移动后等待时间（秒）
    "movement_delay": 3.0,
    
    # 位置容差（mm）
    "position_tolerance": 0.5,
}
