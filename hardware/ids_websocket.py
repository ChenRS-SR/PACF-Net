import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
from scipy.io import savemat
import hardware.vibration_sensor as device_model
from PIL import Image, ImageTk
import queue  # 添加这一行
from datetime import datetime
import requests
import numpy as np
from collections import deque,Counter
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
from predict import predict_single, load_model
from configs.config import config
import struct
from hardware.fotric_driver import FotricEnhancedDevice
import websocket
import json

# M114 Coordinator for real-time position tracking
try:
    from hardware.coordinator import M114Coordinator
    print("[System] Initializing M114Coordinator...")
    m114_coord = M114Coordinator()
    print("[System] M114Coordinator initialized successfully")
except ImportError as e:
    print("[System WARNING] m114_coordinator module not found: {}".format(e))
    m114_coord = None
except Exception as e:
    print("[System WARNING] Failed to initialize M114Coordinator: {}".format(e))
    m114_coord = None

# 旁轴相机相关变量
computer_camera = None
camera_opened = False

# 红外相机（Fotric）相关变量
fotric_device = None
fotric_enabled = False
fotric_latest_frame = None
fotric_temp_min = 0.0
fotric_temp_max = 0.0
fotric_temp_avg = 0.0
fotric_lock = threading.Lock()

"""
    WTVB01-485示例 Example
"""

# region 常用寄存器地址对照表
"""

hex    dec      describe

0x00    0       保存/重启/恢复
0x04    4       串口波特率

0x1A    26      设备地址

0x3A    58      振动速度x
0x3B    59      振动速度y
0x3C    60      振动速度z

0x3D    61      振动角度x
0x3E    62      振动角度y
0x3F    63      振动角度z

0x40    64      温度

0x41    65      振动位移x
0x42    66      振动位移y
0x43    67      振动位移z

0x44    68      振动频率x

0x45    69      振动频率y
0x46    70      振动频率z

0x63    99      截止频率
0x64    100     截止频率
0x65    101     检测周期

"""
# endregion

# OctoPrint 配置
OCTOPRINT_URL = "http://127.0.0.1:5000"
API_KEY = "UGjrS2T5n_48GF0YsWADx1EoTILjwn7ZkeWUfgGvW2Q"

# ==================== 相机配置区域（用户可修改） ====================

# 相机设备编号配置（根据你的系统修改这些值）
CAMERA_CONFIG = {
    # 随轴相机（IDS 或替代摄像头）
    "ids": {
        "enabled": True,           # 是否启用IDS/随轴相机
        "alternative_device_ids": [0, 3, 4],  # IDS不可用时尝试的替代设备编号列表
        "skip_devices": [1],       # 要跳过的设备编号（如旁轴相机使用的设备）
        "resolution": (1920, 1080), # 目标分辨率 (宽, 高)
        "fallback_resolutions": [  # 如果目标分辨率不可用，尝试这些
            (1920, 1080),
            (1280, 720),
            (960, 540),
        ],
        "brightness_threshold": 15,  # 亮度阈值，低于此值认为是虚拟摄像头/未连接
        "std_threshold": 10,         # 标准差阈值，低于此值认为是虚拟摄像头
        "fps": 30,
    },
    # 旁轴可见光相机（电脑摄像头）
    "computer": {
        "enabled": True,
        "device_id": 1,            # 设备编号
        "resolution": (1920, 1080),
        "fps": 30,
        "buffer_size": 1,          # 缓冲区大小
        "disable_autofocus": True, # 禁用自动对焦
        "disable_auto_exposure": True,
        "disable_auto_whitebalance": True,
        "warm_up_frames": 8,       # 预热帧数
    },
    # 旁轴红外相机（Fotric）
    "fotric": {
        "enabled": True,
        "ip": "192.168.1.100",     # 相机IP地址
        "port": 10080,
        "username": "admin",
        "password": "admin",
        "simulation_mode": False,
        "high_resolution": True,
    }
}

# 如何查看设备编号：
# 1. Windows: 设备管理器 -> 图像设备，按连接顺序从0开始编号
# 2. 或者在Python中运行: python -c "import cv2; [print(f'{i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"

# ==================== 相机配置区域结束 ====================

# ------------------- WebSocket全局变量 -------------------
# OctoPrint 使用 SockJS 协议，需要正确的 URL 格式
import random
import string

def generate_sockjs_url():
    """生成 SockJS WebSocket URL"""
    server_id = random.randint(0, 999)
    session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{OCTOPRINT_URL.replace('http://', 'ws://')}/sockjs/{server_id}/{session_id}/websocket"

WS_URL = generate_sockjs_url()  # 动态生成 SockJS URL

ws = None
ws_thread = None
is_websocket_connected = False
current_x = 0.0
current_y = 0.0
current_z = 0.0

#全局参数
FLOW_RATE = 100
FEED_RATE = 100
Z_OFF = 0
CUR_Z_OFF = 0
TARGET_HOTEND = 200
PRIMARY_Z_OFF = -2.55

# 温度循环参数
HOTEND_TEMP = 200           # 当前热端温度
HOTEND_TEMP_DIRECTION = 1   # 温度方向: 1=增加, -1=减少
HOTEND_TEMP_STEP = 10       # 温度步长


# ==================== OctoPrint 服务管理 ====================
octoprint_process = None  # OctoPrint服务进程
octoprint_service_running = False  # 服务是否由本程序启动

import subprocess
import platform

def is_octoprint_running():
    """检测OctoPrint服务是否已运行"""
    try:
        response = requests.get(f"{OCTOPRINT_URL}/api/version", 
                               headers={"X-Api-Key": API_KEY}, 
                               timeout=2)
        return response.status_code == 200
    except:
        return False

def start_octoprint_service():
    """启动OctoPrint服务"""
    global octoprint_process, octoprint_service_running
    
    if is_octoprint_running():
        print("[OctoPrint] 服务已在运行")
        return True
    
    try:
        print("[OctoPrint] 正在启动服务...")
        
        # 检测操作系统
        system = platform.system()
        
        if system == "Windows":
            # Windows: 使用octoprint serve命令
            # 尝试检测conda环境并激活
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
            
            if conda_env:
                print(f"[OctoPrint] 检测到conda环境: {conda_env}")
                # 使用conda run直接运行
                octoprint_process = subprocess.Popen(
                    ["conda", "run", "-n", conda_env, "octoprint", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                # 直接运行octoprint
                octoprint_process = subprocess.Popen(
                    ["octoprint", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
        else:
            # Linux/Mac
            octoprint_process = subprocess.Popen(
                ["octoprint", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
        
        octoprint_service_running = True
        
        # 等待服务启动
        for i in range(30):  # 最多等待30秒
            time.sleep(1)
            if is_octoprint_running():
                print("[OctoPrint] 服务启动成功")
                return True
            if i % 5 == 0:
                print(f"[OctoPrint] 等待服务启动... ({i}s)")
        
        print("[OctoPrint] 服务启动超时，请检查配置")
        return False
        
    except Exception as e:
        print(f"[OctoPrint] 启动失败: {e}")
        print("[提示] 请确保已安装OctoPrint: pip install octoprint")
        return False

def stop_octoprint_service():
    """停止OctoPrint服务"""
    global octoprint_process, octoprint_service_running
    
    if not octoprint_service_running or octoprint_process is None:
        return
    
    try:
        print("[OctoPrint] 正在停止服务...")
        
        system = platform.system()
        
        if system == "Windows":
            # Windows: 发送CTRL+C然后终止
            octoprint_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # Linux/Mac: 发送SIGTERM
            os.killpg(os.getpgid(octoprint_process.pid), signal.SIGTERM)
        
        # 等待进程结束
        octoprint_process.wait(timeout=5)
        print("[OctoPrint] 服务已停止")
        
    except Exception as e:
        print(f"[OctoPrint] 停止服务时出错: {e}")
        # 强制终止
        try:
            octoprint_process.kill()
        except:
            pass
    finally:
        octoprint_process = None
        octoprint_service_running = False

# 导入signal模块（用于停止服务）
import signal

# ==================== OctoPrint 服务管理结束 ====================


#其他全局变量
IMAGE_COUNT = 0
INIT_MODEL = False
model = None

# ========== 标准化采集配置 ==========
# 9组实验的固定参数：温度(°C), Z偏移(mm)
STANDARDIZED_CONFIG = {
    1: {"temp": 180, "z_offset": -0.15, "name": "180°C, Z-0.15"},
    2: {"temp": 180, "z_offset": 0.00, "name": "180°C, Z0.00"},
    3: {"temp": 180, "z_offset": 0.25, "name": "180°C, Z+0.25"},
    4: {"temp": 210, "z_offset": -0.15, "name": "210°C, Z-0.15"},
    5: {"temp": 210, "z_offset": 0.00, "name": "210°C, Z0.00"},
    6: {"temp": 210, "z_offset": 0.25, "name": "210°C, Z+0.25"},
    7: {"temp": 240, "z_offset": -0.15, "name": "240°C, Z-0.15"},
    8: {"temp": 240, "z_offset": 0.00, "name": "240°C, Z0.00"},
    9: {"temp": 240, "z_offset": 0.25, "name": "240°C, Z+0.25"},
}

# 各Z高度区间的速度和流量配置
# 格式: (起始高度mm, 结束高度mm, 速度%, 流量%, 描述)
HEIGHT_RANGES = [
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
]

# 标准化采集当前状态
CURRENT_EXP_NUMBER = 1      # 当前实验编号
current_height_range_idx = 0  # 当前Z高度区间索引

# 记录控制标志
ACQUISITION_MODE = "普通采集"  # "普通采集" 或 "标准化采集"

PARAM_LOOP_LIST = [ [135,-2.6152913943521883,58,36],
                    [137,-2.6198246557918976,37,36],
                    [136,2.6182909509994445,153,51],
                    [132,-2.629735998167022,167,173],
                    [234,-2.6213575029293255,154,20],
                    [234,-2.627322289984435,155,157],
                    [230,-0.5859211743356925,40,159],
                    [139,-2.616035178378578,165,57],
                    [240,-0.7093743445720664,55,58],
                    [132,-2.62730371555484,43,159]]


def get_param_class(param_value,param_thresholds):
    """将连续值离散化为 0, 1, 2"""
    param_class = 1
    if param_value <= param_thresholds[0]:
        param_class = 0
    elif param_value > param_thresholds[1]:
        param_class = 2
    else:
        param_class = 1
    return param_class

#获取打印状态
def get_printer_status():
    url = f"{OCTOPRINT_URL}/api/printer"
    headers = {"X-Api-Key":API_KEY}
    response = requests.get(url,headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error:{response.status_code}")
        return None

def is_printer_ready_for_m114():
    """
    检查打印机是否准备好接收M114命令
    返回True表示可以发送M114，False表示打印机忙或状态异常
    
    注意：Printing状态也返回True，因为需要在打印过程中获取坐标
    """
    try:
        res_json = get_printer_status()
        if res_json is None:
            return False
        
        # 获取状态文本和标志
        state_text = res_json.get('state', {}).get('text', '').lower()
        flags = res_json.get('state', {}).get('flags', {})
        
        # 以下状态表示打印机忙，不应发送M114：
        busy_states = [
            'busy',           # 通用忙状态
            'processing',     # 处理中（如调平、预热）
            'heating',        # 加热中
            'pausing',        # 暂停中
            'resuming',       # 恢复中
            'cancelling',     # 取消中
            'homing',         # 归位中
        ]
        
        # 检查状态文本
        for busy in busy_states:
            if busy in state_text:
                return False
        
        # 检查标志位
        if flags.get('busy') or flags.get('heatingUp'):
            return False
        
        # "Operational"或"Printing"状态都可以发送M114
        return 'operational' in state_text or 'printing' in state_text
        
    except Exception as e:
        print(f"[状态检测错误] {e}")
        return False


def is_printer_actually_printing():
    """
    检查打印机是否真正在打印（用于判断是否记录数据）
    返回True表示正在打印，应该记录数据
    """
    try:
        res_json = get_printer_status()
        if res_json is None:
            return False
        
        state_text = res_json.get('state', {}).get('text', '').lower()
        flags = res_json.get('state', {}).get('flags', {})
        
        # 只有真正在打印或准备打印完成时才记录
        # 'Printing' 或 'Operational'（某些固件打印时也显示为Operational）
        is_printing = 'printing' in state_text
        is_operational = 'operational' in state_text
        
        return is_printing or is_operational
        
    except Exception as e:
        print(f"[打印状态检测错误] {e}")
        return False

#发送G代码
def send_gcode(command):
    url = f"{OCTOPRINT_URL}/api/printer/command"
    headers = {"X-Api-Key":API_KEY}
    data = {"command":command}
    response = requests.post(url,headers=headers,json=data)
    if response.status_code == 204:
        return True
    else:
        print(f"Error:{response.status_code}")
        return False

# 异步发送G代码（不阻塞主线程）
def send_gcode_async(command):
    """异步发送G代码，不阻塞主线程"""
    def _send():
        try:
            send_gcode(command)
        except Exception as e:
            print(f"[异步G代码] 发送失败: {command} - {e}")
    
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

# 生成下一个温度值（循环：200->210->...->250->240->...->150->160->...）
def get_next_hotend_temp():
    """
    生成下一个热端温度值，完整循环：200度→250度→150度→250度→150度...
    - 初始：200度，升温方向
    - 升温阶段：200→210→220→...→250度
    - 降温阶段：250→240→230→...→150度
    - 循环：重复升温和降温阶段
    每次调用增加或减少10度
    """
    global HOTEND_TEMP, HOTEND_TEMP_DIRECTION
    
    # 计算下一个温度
    HOTEND_TEMP += HOTEND_TEMP_DIRECTION * HOTEND_TEMP_STEP
    
    # 检查是否需要改变方向
    if HOTEND_TEMP >= 250:
        HOTEND_TEMP = 250
        HOTEND_TEMP_DIRECTION = -1
        print(f"[温度循环] 达到250度上限，切换为降温模式(→150度)")
    elif HOTEND_TEMP <= 150:
        HOTEND_TEMP = 150
        HOTEND_TEMP_DIRECTION = 1
        print(f"[温度循环] 达到150度下限，切换为升温模式(→250度)")
    
    print(f"[温度循环] 下一个目标: {HOTEND_TEMP}度")
    return HOTEND_TEMP

def get_computer_camera_frame():
    """获取旁轴相机的一帧图像"""
    global computer_camera
    try:
        if computer_camera is None:
            return None
        
        ret, frame = computer_camera.read()
        if ret and frame is not None and frame.size > 0:
            # 检查是否是黑屏（全黑或主要是黑色）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = cv2.mean(gray)[0]
            
            if brightness < 5:  # 如果画面太暗
                print(f"[警告] 摄像头画面太暗 (亮度: {brightness:.1f})")
                return frame
            
            return frame
        else:
            return None
    except Exception as e:
        print(f"获取旁轴相机图像失败: {e}")
        return None

def get_fotric_camera_frame():
    """获取Fotric红外相机的热像数据并转换为可视化图像"""
    global fotric_device, fotric_latest_frame, fotric_temp_min, fotric_temp_max, fotric_temp_avg
    
    if fotric_device is None or not fotric_device.is_connected:
        return None
    
    try:
        # 获取最新的热像帧
        thermal_data = fotric_device.get_thermal_data()
        if thermal_data is None:
            return None
        
        # 缓存温度统计信息
        with fotric_lock:
            fotric_temp_min = float(np.min(thermal_data))
            fotric_temp_max = float(np.max(thermal_data))
            fotric_temp_avg = float(np.mean(thermal_data))
            fotric_latest_frame = thermal_data.copy()
        
        # 归一化热像数据以便显示
        if fotric_temp_max > fotric_temp_min:
            normalized = ((thermal_data - fotric_temp_min) / (fotric_temp_max - fotric_temp_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_data, dtype=np.uint8)
        
        # 应用热力图颜色
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # 确保图像尺寸一致
        if colored.shape[0] != 360 or colored.shape[1] != 960:
            colored = cv2.resize(colored, (960, 360))
        
        return colored
        
    except Exception as e:
        print(f"获取Fotric图像失败: {e}")
        return None

def sharpen_image(img):
    # 读取图像
    
    
    # 高斯模糊（模拟去噪）
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    
    # 锐化（增强边缘）
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    # 对比度增强
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 保存结果
    
    return enhanced

def update_temperature_display():
    """更新红外相机温度显示面板"""
    global fotric_enabled, fotric_temp_min, fotric_temp_max, fotric_temp_avg
    
    try:
        if fotric_enabled and fotric_temp_max >= fotric_temp_min:
            # 格式化温度信息
            temp_text = f"最小: {fotric_temp_min:.1f}°C | 平均: {fotric_temp_avg:.1f}°C | 最大: {fotric_temp_max:.1f}°C"
            try:
                fotric_temp_info.config(text=temp_text)  # type: ignore
            except (NameError, AttributeError):
                pass  # UI还未初始化
    except Exception as e:
        pass  # 忽略任何错误

# ========== 时间戳同步相关全局变量 ==========
last_coord_timestamp = 0.0  # 上次坐标获取的时间戳
last_coord_time_diff = 0.0  # 上次图像与坐标的时间差(ms)

def update_m114_coordinates():
    """
    定期获取M114坐标
    每0.5秒执行一次，与图像采集频率同步
    记录坐标获取时间戳，用于事后时间匹配
    
    优化：在打印机忙（调平、预热等）时跳过查询，避免超时
    """
    global current_x, current_y, current_z, last_coord_timestamp
    
    # 检查打印机是否准备好接收M114命令
    if not is_printer_ready_for_m114():
        # 打印机忙，跳过本次查询
        return
    
    try:
        coords = m114_coord.wait_for_m114_response(timeout=0.8)  # 缩短超时时间
        if coords:
            old_x, old_y, old_z = current_x, current_y, current_z
            current_x = coords['X']
            current_y = coords['Y']
            current_z = coords['Z']
            
            # 记录坐标获取时间戳（关键）
            last_coord_timestamp = time.time()
            
            # 只在坐标改变时打印（减少日志噪音）
            if (abs(old_x - current_x) > 0.01 or 
                abs(old_y - current_y) > 0.01 or 
                abs(old_z - current_z) > 0.01):
                print("[M114] 坐标已更新: X={:.2f}, Y={:.2f}, Z={:.2f} (t={:.3f})".format(
                    current_x, current_y, current_z, last_coord_timestamp
                ))
            
            # 更新界面显示
            try:
                if coordinates_label.winfo_exists():
                    coordinates_label.config(
                        text="X: {:.2f}  Y: {:.2f}  Z: {:.2f}".format(current_x, current_y, current_z)
                    )
            except:
                pass
        else:
            # 超时但不打印警告（避免日志刷屏）
            pass
    except Exception as e:
        # 静默处理错误，不影响UI
        pass

def update_printer_status_display():
    """
    更新打印机状态显示面板
    包括：坐标、温度、打印参数、Z轴最终状态
    """
    global current_x, current_y, current_z, FLOW_RATE, FEED_RATE, Z_OFF, TARGET_HOTEND, PRIMARY_Z_OFF, CUR_Z_OFF
    
    try:
        # 更新XYZ坐标
        if 'x_label' in globals() and x_label.winfo_exists():
            x_label.config(text=f"{current_x:.2f}")
        if 'y_label' in globals() and y_label.winfo_exists():
            y_label.config(text=f"{current_y:.2f}")
        if 'z_label' in globals() and z_label.winfo_exists():
            z_label.config(text=f"{current_z:.2f}")
        
        # 更新尖端目标温度
        if 'hotend_target_label' in globals() and hotend_target_label.winfo_exists():
            hotend_target_label.config(text=f"{TARGET_HOTEND}°C")
        
        # 更新流量和速度
        if 'flow_status_label' in globals() and flow_status_label.winfo_exists():
            flow_status_label.config(text=f"{FLOW_RATE}%")
        if 'speed_status_label' in globals() and speed_status_label.winfo_exists():
            speed_status_label.config(text=f"{FEED_RATE}%")
        
        # 更新Z偏移和Z轴最终状态
        if 'z_offset_status_label' in globals() and z_offset_status_label.winfo_exists():
            z_offset_status_label.config(text=f"{Z_OFF:.2f}")
        
        # Z轴最终状态 = 初始Z补偿 + 当前Z偏移
        # PRIMARY_Z_OFF = -2.55 (初始调平值)
        # Z_OFF = 当前调整的偏移值
        z_final = PRIMARY_Z_OFF + Z_OFF
        if 'z_final_status_label' in globals() and z_final_status_label.winfo_exists():
            z_final_status_label.config(text=f"{z_final:.2f}")
        
        # 从OctoPrint获取实际温度和打印机状态
        try:
            res_json = get_printer_status()
            if res_json:
                # 尖端实际温度
                hotend_actual = res_json['temperature']['tool0']['actual']
                hotend_target_api = res_json['temperature']['tool0']['target']
                if 'hotend_actual_label' in globals() and hotend_actual_label.winfo_exists():
                    hotend_actual_label.config(text=f"{hotend_actual:.1f}°C")
                
                # 热床目标温度（从API获取）
                bed_target = res_json['temperature']['bed']['target']
                bed_actual = res_json['temperature']['bed']['actual']
                
                if 'bed_target_label' in globals() and bed_target_label.winfo_exists():
                    bed_target_label.config(text=f"{bed_target:.1f}°C")
                if 'bed_actual_label' in globals() and bed_actual_label.winfo_exists():
                    bed_actual_label.config(text=f"{bed_actual:.1f}°C")
                
                # 更新打印机状态文本和颜色
                if 'printer_state_label' in globals() and printer_state_label.winfo_exists():
                    state_text = res_json.get('state', {}).get('text', '未知')
                    flags = res_json.get('state', {}).get('flags', {})
                    
                    # 根据状态设置颜色
                    state_lower = state_text.lower()
                    if 'printing' in state_lower:
                        color = '#4CAF50'  # 绿色 - 打印中
                    elif 'paused' in state_lower or 'pausing' in state_lower:
                        color = '#FF9800'  # 橙色 - 暂停
                    elif 'operational' in state_lower or 'ready' in state_lower:
                        color = '#2196F3'  # 蓝色 - 就绪/空闲
                    elif 'heating' in state_lower or 'heatingup' in state_lower:
                        color = '#FF5722'  # 红色 - 加热中
                    elif 'busy' in state_lower or 'processing' in state_lower:
                        color = '#9C27B0'  # 紫色 - 忙碌
                    elif 'error' in state_lower or 'closed' in state_lower:
                        color = '#F44336'  # 深红色 - 错误
                    else:
                        color = '#666666'  # 灰色 - 其他
                    
                    printer_state_label.config(text=state_text, fg=color)
        except Exception as e:
            # API获取失败时不更新
            pass
        
        # 更新原来的coordinates_label（保持兼容）
        if 'coordinates_label' in globals() and coordinates_label.winfo_exists():
            coordinates_label.config(
                text=f"X: {current_x:.2f}  Y: {current_y:.2f}  Z: {current_z:.2f}"
            )
            
    except Exception as e:
        print(f"[打印机状态显示更新错误] {e}")

def update_frame():
    # 获取IDS相机图像（已自动旋转180度校正安装方向）
    ids_image_np = ids_process_img()
    
    # 获取旁轴相机图像
    computer_image_np = get_computer_camera_frame()
    
    # 获取Fotric红外相机图像
    fotric_image_np = get_fotric_camera_frame()
    
    # 显示尺寸配置（缩小以适应屏幕）
    # 旁轴和IDS相机: 16:9比例，480x270
    DISPLAY_WIDTH = 480
    DISPLAY_HEIGHT = 270
    
    # 红外相机: 4:3比例保持原始比例 (640x480按比例缩放)
    FOTRIC_WIDTH = 480
    FOTRIC_HEIGHT = 360  # 4:3比例
    
    # 转换IDS相机图像并显示（右上）
    ids_img = Image.fromarray(cv2.cvtColor(ids_image_np, cv2.COLOR_BGR2RGB))
    ids_img = ids_img.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS)
    ids_imgtk = ImageTk.PhotoImage(image=ids_img)
    ids_video_label.imgtk = ids_imgtk
    ids_video_label.configure(image=ids_imgtk)
    
    # 转换旁轴相机图像并显示（左上）
    if computer_image_np is not None:
        computer_img = Image.fromarray(cv2.cvtColor(computer_image_np, cv2.COLOR_BGR2RGB))
        computer_img = computer_img.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS)
        computer_imgtk = ImageTk.PhotoImage(image=computer_img)
        computer_video_label.imgtk = computer_imgtk
        computer_video_label.configure(image=computer_imgtk)
    
    # 转换Fotric红外相机图像并显示（下方，保持4:3比例）
    if fotric_image_np is not None:
        fotric_img = Image.fromarray(cv2.cvtColor(fotric_image_np, cv2.COLOR_BGR2RGB))
        fotric_img = fotric_img.resize((FOTRIC_WIDTH, FOTRIC_HEIGHT), Image.LANCZOS)
        fotric_imgtk = ImageTk.PhotoImage(image=fotric_img)
        fotric_video_label.imgtk = fotric_imgtk
        fotric_video_label.configure(image=fotric_imgtk)
    
    # 更新温度显示面板
    update_temperature_display()

    # 每30毫秒更新一次画面（降低CPU占用）
    ids_video_label.after(30, update_frame)

# ------------------- WebSocket消息处理 -------------------
def on_message(ws_app, message):
    global current_x, current_y, current_z
    
    # SockJS 协议的第一个消息是 'o' (open frame)
    if message == "o":
        return
    
    # SockJS 心跳帧 'h'
    if message.startswith('h'):
        return
    
    # SockJS 消息格式: a["..."] 其中 a 是数组帧
    if not message.startswith('a'):
        return
    
    try:
        # 解析 SockJS 数组帧: a[obj1, obj2, ...]
        frames = json.loads(message[1:])  # 去掉开头的 'a'
        for frame_obj in frames:
            try:
                # 如果是字符串，需要解析；如果已经是对象，直接使用
                if isinstance(frame_obj, str):
                    data = json.loads(frame_obj)
                else:
                    data = frame_obj
                
                # 转换整个消息为字符串以便搜索 M114 响应
                data_str = json.dumps(data) if not isinstance(data, str) else data
                
                # ========== 方案 1: 对象格式消息 ==========
                if isinstance(data, dict):
                    # 检查是否有 current 状态信息
                    if "current" in data and isinstance(data["current"], dict):
                        current_state = data["current"]
                        if "state" in current_state and isinstance(current_state["state"], dict):
                            state_info = current_state["state"]
                            # 更新坐标信息
                            if "position" in state_info and state_info["position"] is not None:
                                current_x = state_info["position"].get("x", 0.0)
                                current_y = state_info["position"].get("y", 0.0)
                                current_z = state_info["position"].get("z", 0.0)
                                
                                # 更新界面显示
                                if coordinates_label.winfo_exists():
                                    coordinates_label.config(
                                        text=f"X: {current_x:.2f}  Y: {current_y:.2f}  Z: {current_z:.2f}"
                                    )
                
                # ========== 方案 2: 数组格式消息 ==========
                elif isinstance(data, list) and len(data) > 0:
                    # 消息类型：current状态更新
                    if data[0] == "current":
                        payload = data[1] if len(data) > 1 else {}
                        if isinstance(payload, dict) and "position" in payload and payload["position"] is not None:
                            # 更新当前坐标
                            current_x = payload["position"].get("x", 0.0)
                            current_y = payload["position"].get("y", 0.0)
                            current_z = payload["position"].get("z", 0.0)
                            
                            # 更新界面显示
                            if coordinates_label.winfo_exists():
                                coordinates_label.config(
                                    text=f"X: {current_x:.2f}  Y: {current_y:.2f}  Z: {current_z:.2f}"
                                )
                    
                    # 消息类型：日志/命令响应（处理M114响应）
                    # M114 的响应格式：X:106.14 Y:117.45 Z:1.60 E:207.39 Count X:8491 Y:9396 Z:635
                    elif data[0] == "history" or data[0] == "logs" or data[0] == "exec":
                        if len(data) > 1:
                            log_data = data[1]
                            if isinstance(log_data, list):
                                # 处理日志列表
                                for entry in log_data:
                                    if isinstance(entry, str) and "X:" in entry and "Y:" in entry and "Z:" in entry:
                                        parse_m114_response(entry)
                            elif isinstance(log_data, str):
                                # 处理单个日志字符串
                                if "X:" in log_data and "Y:" in log_data and "Z:" in log_data:
                                    parse_m114_response(log_data)
                
                # ========== 方案 3: 纯字符串消息 ==========
                elif isinstance(data, str):
                    # 直接检查字符串中是否包含 M114 响应
                    if "X:" in data and "Y:" in data and "Z:" in data:
                        parse_m114_response(data)
                
                # ========== 方案 4: 搜索整个 JSON 中的坐标 ==========
                # 防御性处理：在整个消息中搜索坐标字符串
                if "X:" in data_str and "Y:" in data_str and "Z:" in data_str:
                    # 尝试从 JSON 字符串中直接提取
                    import re
                    match = re.search(r'X:([\d.]+)\s+Y:([\d.]+)\s+Z:([\d.]+)', data_str)
                    if match:
                        try:
                            x_val = float(match.group(1))
                            y_val = float(match.group(2))
                            z_val = float(match.group(3))
                            
                            current_x = x_val
                            current_y = y_val
                            current_z = z_val
                            
                            if coordinates_label.winfo_exists():
                                coordinates_label.config(
                                    text=f"X: {current_x:.2f}  Y: {current_y:.2f}  Z: {current_z:.2f}"
                                )
                        except:
                            pass
            
            except json.JSONDecodeError:
                pass
            except Exception as e:
                pass  # 静默处理单个帧错误
    except Exception as e:
        pass  # 静默处理 SockJS 帧解析错误


def parse_m114_response(text):
    """解析 M114 命令的坐标响应"""
    global current_x, current_y, current_z
    
    try:
        # M114 格式: X:106.14 Y:117.45 Z:1.60 E:207.39 Count X:8491 Y:9396 Z:635
        parts = text.split()
        for part in parts:
            if part.startswith("X:"):
                try:
                    current_x = float(part[2:])
                except:
                    pass
            elif part.startswith("Y:"):
                try:
                    current_y = float(part[2:])
                except:
                    pass
            elif part.startswith("Z:"):
                try:
                    current_z = float(part[2:])
                except:
                    pass
        
        # 更新界面显示
        if coordinates_label.winfo_exists():
            coordinates_label.config(
                text=f"X: {current_x:.2f}  Y: {current_y:.2f}  Z: {current_z:.2f}"
            )
    except:
        pass

def on_error(ws_app, error):
    global is_websocket_connected
    is_websocket_connected = False
    error_type = type(error).__name__
    print(f"[WebSocket错误] {error_type}: {error}")

def on_close(ws_app, close_status_code, close_msg):
    global is_websocket_connected
    is_websocket_connected = False
    print(f"[WebSocket连接关闭] 状态码: {close_status_code}, 消息: {close_msg}")

def on_open(ws_app):
    global is_websocket_connected
    is_websocket_connected = True
    print("✓ WebSocket连接成功")
    
    try:
        # OctoPrint SockJS 格式：直接发送 JSON 数组（在 a 帧中）
        # 订阅 current 状态来获取打印机坐标
        msg = json.dumps(["subscribe", "current"])
        ws_app.send(msg)
        print("[WebSocket] 已订阅 current 状态")
        
        # 也订阅 history
        msg = json.dumps(["subscribe", "history"])
        ws_app.send(msg)
        print("[WebSocket] 已订阅 history 消息")
    except Exception as e:
        print(f"[WebSocket] 订阅失败: {e}")
    
    # 实时获取坐标（定时发送M114并解析）
    def get_coordinates_via_m114():
        """通过定期发送 M114 命令获取打印机坐标"""
        fail_count = 0
        success_count = 0
        last_send_time = time.time()
        last_update_time = time.time()
        
        # 缓存最后一次的 M114 响应
        m114_buffer = deque(maxlen=100)
        
        while is_websocket_connected:
            try:
                current_time = time.time()
                
                # 每 3 秒发送一次 M114 命令（改为3秒，降低通信占用）
                if current_time - last_send_time >= 3.0:
                    try:
                        # 改为异步发送，不阻塞主线程
                        def _send_m114():
                            try:
                                response = requests.post(
                                    f"{OCTOPRINT_URL}/api/printer/command",
                                    headers={"X-Api-Key": API_KEY},
                                    json={"command": "M114"},
                                    timeout=2
                                )
                                if response.status_code == 204:
                                    nonlocal success_count, fail_count
                                    success_count += 1
                                    fail_count = 0
                                    if success_count % 20 == 0:
                                        print(f"[M114] 已发送 {success_count} 次命令")
                                else:
                                    fail_count += 1
                            except Exception as e:
                                fail_count += 1
                        
                        thread = threading.Thread(target=_send_m114, daemon=True)
                        thread.start()
                    except Exception as e:
                        fail_count += 1
                    
                    last_send_time = current_time
                
                # 定期检查缓存中的 M114 响应
                # 注意：M114 响应会通过 WebSocket 的 history 消息到达
                if current_time - last_update_time >= 2.0:
                    # 这里可以添加日志验证
                    last_update_time = current_time
                
                time.sleep(0.1)
            except Exception as e:
                fail_count += 1
                if fail_count % 20 == 0:
                    print(f"[坐标获取错误] ({fail_count}次): {e}")
                if fail_count > 100:
                    print("[坐标获取] 连续失败过多，停止")
                    break
                time.sleep(1)
    
    thread = threading.Thread(target=get_coordinates_via_m114, daemon=True)
    thread.start()

# ------------------- WebSocket连接管理 -------------------
def start_websocket():
    global ws, ws_thread, WS_URL
    try:
        # 启动WebSocket线程（添加重连机制）
        def ws_run_with_reconnect():
            global ws, WS_URL
            max_retries = 10  # 增加重试次数
            retry_count = 0
            backoff_delay = 2  # 初始退避延迟
            
            while retry_count < max_retries:
                try:
                    # 每次重连生成新的 SockJS URL
                    WS_URL = generate_sockjs_url()
                    print(f"[WebSocket] 正在连接到 {WS_URL}...")
                    
                    # 创建新的 WebSocket 对象
                    ws = websocket.WebSocketApp(
                        WS_URL,
                        header={"X-Api-Key": API_KEY},
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close
                    )
                    ws.on_open = on_open
                    
                    # 禁用 ping/pong，OctoPrint WebSocket 不需要心跳
                    ws.run_forever()
                    # 如果连接成功建立且运行，则重置重试计数
                    retry_count = 0
                    backoff_delay = 2
                except Exception as e:
                    retry_count += 1
                    print(f"[WebSocket] 连接异常（第{retry_count}/{max_retries}次）: {type(e).__name__}")
                    if retry_count < max_retries:
                        wait_time = min(backoff_delay * retry_count, 30)  # 最多等待30秒
                        print(f"[WebSocket] 等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print("[WebSocket] 达到最大重试次数，停止重连")
        
        ws_thread = threading.Thread(target=ws_run_with_reconnect, daemon=True)
        ws_thread.start()
        print("[WebSocket] 启动完成，启用心跳机制和自动重连")
    except Exception as e:
        print(f"WebSocket启动失败: {e}")

def stop_websocket():
    global is_websocket_connected
    is_websocket_connected = False
    if ws:
        ws.close()





def record_data(image_queue):
    global is_recording, is_paused,IMAGE_COUNT
    # 创建新的子文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(save_directory.get(), f"task_{timestamp}")
    os.makedirs(data_dir, exist_ok=True)
    
    sensor_data_csv = os.path.join(data_dir, "sensor_data.csv")
    #sensor_data_mat = os.path.join(data_dir, "sensor_data.mat")
    # 拿到设备模型
    # 注意关闭其他占用串口的软件，确保只有此程序占用串口
    device = device_model.DeviceModel("测试设备", "COM9", 115200, 0x50)
    # 开启设备
    device.openDevice()
    # 开启轮询
    device.startLoopRead()

    # 打开CSV文件
    with open(sensor_data_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["image_name","acx", "acy", "acz", "vx", "vy", "vz", "ax", "ay", "az", "t", "sx", "sy", "sz", "fx", "fy", "fz"])
                        #ac,v,a,t,s,f
                        #ac：振动加速度 v：振动速度 a：振动角度 t：温度 s：振动位移 f：振动频率
        sensor_data = []

        while is_recording:
            if not is_paused:
                # 读取传感器数据
                data = [device.get(str(i)) for i in range(58, 71)]
                
                # 读取传感器数据并格式化为三位小数
                if IMAGE_COUNT%25==0:
                    print(f"传感器数据: {data}")  # 调试信息10秒一次
                # 获取当前图片名称
                image_name = image_queue.get() if not image_queue.empty() else ""
                data.insert(0, image_name)
                writer.writerow(data)
                sensor_data.append(data)
                time.sleep(0.4)  # 持续读取数据的时间间隔
            else:
                print("传感器记录暂停")  # 调试信息
                time.sleep(1)

        # 保存MAT文件
        #savemat(sensor_data_mat, {"sensor_data": sensor_data})
    # 释放传感器资源
    device.stopLoopRead()
    device.closeDevice()
    print("传感器释放")  # 调试信息

def capture_images(capture_interval, image_queue):
    global is_recording, is_paused,PRINT_STATE,FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,IMAGE_COUNT,CLOSE_LOOP,PARAM_LOOP,INIT_MODEL,model
    global ACQUISITION_MODE, CURRENT_EXP_NUMBER, current_height_range_idx
    
    print("[线程] 图片捕获线程已启动")
    print(f"[线程] 采集模式: {ACQUISITION_MODE}")
    
    # ========== 标准化采集初始化 ==========
    if ACQUISITION_MODE == "标准化采集":
        exp_config = STANDARDIZED_CONFIG[CURRENT_EXP_NUMBER]
        print(f"[标准化采集] 实验编号: 第{CURRENT_EXP_NUMBER}组")
        print(f"[标准化采集] 配置: {exp_config['name']}")
        print(f"[标准化采集] 固定温度: {exp_config['temp']}°C")
        print(f"[标准化采集] Z偏移: {exp_config['z_offset']}mm")
        
        # 设置固定参数
        TARGET_HOTEND = exp_config['temp']
        Z_OFF = exp_config['z_offset']
        CUR_Z_OFF = 0  # 重置当前Z偏移
        
        # 发送初始参数到打印机
        change_param_auto(FLOW_RATE, FEED_RATE, Z_OFF, TARGET_HOTEND, init=True)
        print("[标准化采集] 初始参数已发送")
        
        # 初始化高度区间索引
        current_height_range_idx = 0
        
        # 标准化采集不使用自动修改参数（由高度区间控制）
        print("[标准化采集] 自动参数修改已禁用，按Z高度区间控制")
    # =====================================
    
    # 创建图片存储子文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = save_directory.get()
    
    # 规范化路径：处理中文字符和混合斜杠
    base_dir = os.path.normpath(base_dir)  # 统一为反斜杠
    
    image_dir = os.path.join(base_dir, f"task_{timestamp}/images")
    ids_image_dir = os.path.join(image_dir, "IDS_Camera")  # IDS相机图片目录
    computer_image_dir = os.path.join(image_dir, "Computer_Camera")  # 旁轴相机图片目录
    fotric_image_dir = os.path.join(image_dir, "Fotric_Camera")  # 红外相机图片目录
    fotric_data_dir = os.path.join(image_dir, "Fotric_Data")    # 红外相机数据目录
    
    #创建打印信息存储文件
    try:
        os.makedirs(ids_image_dir, exist_ok=True)
        os.makedirs(computer_image_dir, exist_ok=True)
        os.makedirs(fotric_image_dir, exist_ok=True)
        os.makedirs(fotric_data_dir, exist_ok=True)
        print(f"[目录信息] 基础目录: {base_dir}")
        print(f"[目录信息] 图像目录: {image_dir}")
        print(f"[目录信息] 红外目录: {fotric_image_dir}")
        print(f"[目录信息] 红外数据目录: {fotric_data_dir}")
        
        # 验证目录可写性
        test_file = os.path.join(fotric_image_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"[目录检查] 红外目录可写: True")
    except Exception as mkdir_err:
        print(f"[ERROR] 创建目录失败: {mkdir_err}")
    
    CSV_FILE = os.path.join(base_dir, f'task_{timestamp}/print_message.csv')
    # 添加时间戳同步相关字段
    HEADER = ['image_path','computer_image_path','timestamp','img_timestamp','coord_timestamp','time_diff_ms',
              'current_x','current_y','current_z',
              'flow_rate','feed_rate','z_offset','target_hotend','hot_end','bed','img_num',
              'flow_rate_class','feed_rate_class','z_offset_class','hotend_class',
              'fotric_temp_min','fotric_temp_max','fotric_temp_avg','fotric_image_path','fotric_data_path']
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(HEADER)
    
    # 重置IMAGE_COUNT为1
    IMAGE_COUNT = 1
    print(f"[线程] 开始图片捕获循环，保存路径: {save_directory.get()}")
    while is_recording:
        if not is_paused:
            try:
                # ========== 标准化采集参数控制 ==========
                if ACQUISITION_MODE == "标准化采集":
                    # 根据Z轴高度自动调整参数
                    current_z_pos = current_z  # 获取当前Z坐标
                    
                    # 查找当前Z高度对应的区间
                    height_updated = False
                    for idx, (z_start, z_end, speed, flow, desc) in enumerate(HEIGHT_RANGES):
                        if z_start <= current_z_pos < z_end:
                            if idx != current_height_range_idx:
                                current_height_range_idx = idx
                                FEED_RATE = speed
                                FLOW_RATE = flow
                                print(f"[标准化采集] Z高度 {current_z_pos:.2f}mm，进入区间 {desc}")
                                print(f"[标准化采集] 速度: {speed}%, 流量: {flow}%")
                                
                                # 发送参数修改（使用异步避免阻塞）
                                send_gcode_async(f"M220 S{speed}")
                                send_gcode_async(f"M221 S{flow}")
                                height_updated = True
                            break
                # =========================================
                
                # 普通采集的参数修改逻辑（保持原有逻辑）
                elif ACQUISITION_MODE == "普通采集" and PRINT_STATE:
                    time_to_change = 120  # 120秒(2分钟)自动改变一次参数
                    count = int(time_to_change/2)
                    i = 0
                    if(IMAGE_COUNT % count == 0):  # 120秒修改一次打印数据
                        
                        if(PARAM_LOOP):
                            # 循环参数
                            FLOW_RATE = PARAM_LOOP_LIST[i][2]
                            FEED_RATE = PARAM_LOOP_LIST[i][3]
                            Z_OFF = PARAM_LOOP_LIST[i][1]-PRIMARY_Z_OFF
                            TARGET_HOTEND = PARAM_LOOP_LIST[i][0]
                            i += 1
                            if(i > 9):
                                i = 0
                        else:
                            # 随机生成较大误差参数
                            pass
                        
                        # 随机获取打印数据
                        rate_options = [20, 30, 40, 50, 60, 70, 80, 90, 100, 
                        110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        
                        # z_off_options 包含了指定的浮点数
                        z_off_options = [-0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32]

                        # 使用 np.random.choice 从列表中随机抽取
                        FLOW_RATE = np.random.choice(rate_options)
                        FEED_RATE = np.random.choice(rate_options)
                        Z_OFF = np.random.choice(z_off_options)
                        
                        change_param_auto(FLOW_RATE, FEED_RATE, Z_OFF, TARGET_HOTEND)  # 自动修改参数
                #获取图像
                print(f"[第{IMAGE_COUNT}帧] 开始获取IDS相机图像...")
                # IDS相机图像（已自动旋转180度校正安装方向）
                ids_image_np = ids_process_img()
                print(f"[第{IMAGE_COUNT}帧] IDS图像获取成功，形状: {ids_image_np.shape}")
                
                print(f"[第{IMAGE_COUNT}帧] 获取旁轴相机图像...")
                # 旁轴相机图像
                computer_image_np = get_computer_camera_frame()
                
                # ========== 打印状态检查 ==========
                # 检查打印机是否真正在打印，避免在预热准备阶段记录数据
                if not is_printer_actually_printing():
                    if IMAGE_COUNT % 50 == 0:  # 每50帧输出一次提示
                        print(f"[第{IMAGE_COUNT}帧] 打印机未开始打印（预热/准备中），跳过记录")
                    IMAGE_COUNT += 1
                    time.sleep(capture_interval)
                    continue
                # ==================================
                
                # ========== 双时间戳记录 ==========
                # 获取图像采集时间戳（单位：秒）
                img_timestamp = time.time()
                current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S-%f")[:-2]
                
                # 获取最新的坐标时间戳（全局变量由M114线程更新）
                global last_coord_timestamp
                coord_timestamp = last_coord_timestamp
                
                # 计算时间差（毫秒）
                time_diff_ms = abs(img_timestamp - coord_timestamp) * 1000
                
                # 每10帧输出一次同步信息
                if IMAGE_COUNT % 10 == 0:
                    print(f"[时间同步] 图像t={img_timestamp:.3f}, 坐标t={coord_timestamp:.3f}, 差值={time_diff_ms:.1f}ms")
                # ==================================
                
                # 保存IDS相机图像
                ids_image_path = os.path.join(ids_image_dir, f"image-{IMAGE_COUNT}.jpg")
                # OpenCV BGR -> PIL RGB
                ids_img_rgb = cv2.cvtColor(ids_image_np, cv2.COLOR_BGR2RGB)
                ids_img = Image.fromarray(ids_img_rgb)
                if ids_img.mode == 'RGBA':
                    ids_img = ids_img.convert('RGB')
                ids_img.save(ids_image_path)
                
                # ========== 标准化采集记录区间控制 ==========
                should_record = True
                if ACQUISITION_MODE == "标准化采集":
                    current_z_pos = current_z
                    # 获取当前区间的记录范围（延迟0.5mm开始，提前0.5mm停止）
                    for idx, (z_start, z_end, speed, flow, desc) in enumerate(HEIGHT_RANGES):
                        if z_start <= current_z_pos < z_end:
                            # 0-5mm区间：完整记录0-5mm（不延迟开始，不提前结束）
                            if z_start == 0:
                                record_start = 0.0  # 从0mm开始
                                record_end = 5.0    # 到5mm结束
                            else:
                                record_start = z_start + 0.5  # 延迟0.5mm开始
                                record_end = z_end - 0.5      # 提前0.5mm停止
                            
                            if record_start <= current_z_pos < record_end:
                                should_record = True
                                if IMAGE_COUNT % 50 == 0:  # 每50帧输出一次记录状态
                                    print(f"[标准化采集] 记录中: Z={current_z_pos:.2f}mm, 区间{desc}")
                            else:
                                should_record = False
                                if IMAGE_COUNT % 50 == 0:
                                    print(f"[标准化采集] 跳过记录: Z={current_z_pos:.2f}mm (不在有效记录区间)")
                            break
                # =============================================
                
                if not should_record:
                    # 不记录此帧，但仍计数并等待
                    IMAGE_COUNT += 1
                    time.sleep(capture_interval)
                    continue
                
                # 保存旁轴相机图像
                computer_image_path = ""
                if computer_image_np is not None:
                    computer_image_path = os.path.join(computer_image_dir, f"image-{IMAGE_COUNT}.jpg")
                    computer_img = cv2.cvtColor(computer_image_np, cv2.COLOR_BGR2RGB)
                    computer_pil_img = Image.fromarray(computer_img)
                    computer_pil_img.save(computer_image_path)
                    print(f"[第{IMAGE_COUNT}帧] 旁轴相机图像已保存: {computer_image_path}")
                else:
                    computer_image_path = ""
                
                # 保存红外相机图像和数据
                fotric_image_path = ""
                fotric_data_path = ""
                fotric_temp_min_cached = 0.0
                fotric_temp_max_cached = 0.0
                fotric_temp_avg_cached = 0.0
                
                if fotric_enabled and fotric_device is not None and fotric_device.is_connected:
                    # 获取线程安全的温度数据
                    with fotric_lock:
                        fotric_temp_min_cached = fotric_temp_min
                        fotric_temp_max_cached = fotric_temp_max
                        fotric_temp_avg_cached = fotric_temp_avg
                    
                    try:
                        # 获取红外原始数据
                        thermal_data = fotric_device.get_thermal_data()
                        if thermal_data is not None:
                            print(f"[第{IMAGE_COUNT}帧] 开始保存红外相机数据...")
                            
                            # 保存原始热像数据为 NPZ 格式（高效压缩存储）
                            fotric_data_path = os.path.join(fotric_data_dir, f"thermal_data-{IMAGE_COUNT}.npz")
                            np.savez_compressed(
                                fotric_data_path,
                                thermal_data=thermal_data.astype(np.float32),
                                timestamp=current_time,
                                temp_min=float(np.min(thermal_data)),
                                temp_max=float(np.max(thermal_data)),
                                temp_avg=float(np.mean(thermal_data))
                            )
                            print(f"[第{IMAGE_COUNT}帧] 红外数据已保存: {fotric_data_path}")
                            
                            # 保存彩色热像图 (JPG 格式用于直观查看)
                            try:
                                if fotric_temp_max_cached > fotric_temp_min_cached:
                                    normalized = ((thermal_data - fotric_temp_min_cached) / (fotric_temp_max_cached - fotric_temp_min_cached) * 255).astype(np.uint8)
                                else:
                                    # 当没有温度范围时，用中灰色替代全黑
                                    normalized = np.full_like(thermal_data, 128, dtype=np.uint8)
                                
                                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                                
                                # 验证图像数据
                                if colored is None or colored.size == 0:
                                    print(f"[第{IMAGE_COUNT}帧] 错误: 彩色图像数据为空")
                                    fotric_image_path = ""
                                elif not os.path.exists(fotric_image_dir):
                                    print(f"[第{IMAGE_COUNT}帧] 错误: 红外图像目录不存在 {fotric_image_dir}")
                                    # 尝试创建目录
                                    try:
                                        os.makedirs(fotric_image_dir, exist_ok=True)
                                        print(f"[第{IMAGE_COUNT}帧] 已创建目录: {fotric_image_dir}")
                                    except Exception as mkdir_err:
                                        print(f"[第{IMAGE_COUNT}帧] 创建目录失败: {mkdir_err}")
                                    fotric_image_path = ""
                                else:
                                    fotric_image_path = os.path.join(fotric_image_dir, f"image-{IMAGE_COUNT}.jpg")
                                    fotric_image_path = os.path.normpath(fotric_image_path)
                                    
                                    try:
                                        parent_dir = os.path.dirname(fotric_image_path)
                                        if not os.path.exists(parent_dir):
                                            os.makedirs(parent_dir, exist_ok=True)
                                        
                                        # 【主方案】使用PIL库保存（兼容性最好）
                                        save_success = False
                                        try:
                                            pil_image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
                                            pil_image.save(fotric_image_path, quality=95)
                                            if os.path.exists(fotric_image_path):
                                                file_size = os.path.getsize(fotric_image_path)
                                                print(f"[第{IMAGE_COUNT}帧] 红外图像已保存: {fotric_image_path} ({file_size}字节)")
                                                save_success = True
                                        except Exception as pil_err:
                                            print(f"[第{IMAGE_COUNT}帧] PIL保存失败, 尝试cv2: {pil_err}")
                                        
                                        # 【备选1】cv2.imwrite
                                        if not save_success:
                                            result = cv2.imwrite(fotric_image_path, colored)
                                            if result and os.path.exists(fotric_image_path):
                                                file_size = os.path.getsize(fotric_image_path)
                                                print(f"[第{IMAGE_COUNT}帧] 红外图像已保存(cv2): {fotric_image_path} ({file_size}字节)")
                                                save_success = True
                                            else:
                                                print(f"[第{IMAGE_COUNT}帧] cv2.imwrite失败")
                                        
                                        # 【备选2】PNG格式
                                        if not save_success:
                                            png_path = fotric_image_path.replace('.jpg', '.png')
                                            result2 = cv2.imwrite(png_path, colored, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                                            if result2 and os.path.exists(png_path):
                                                file_size = os.path.getsize(png_path)
                                                print(f"[第{IMAGE_COUNT}帧] 红外图像已保存(PNG): {png_path} ({file_size}字节)")
                                                fotric_image_path = png_path
                                                save_success = True
                                        
                                        if not save_success:
                                            print(f"[第{IMAGE_COUNT}帧] 错误: 红外图像保存失败 - 路径:{fotric_image_path}")
                                            fotric_image_path = ""
                                            
                                    except Exception as write_err:
                                        print(f"[第{IMAGE_COUNT}帧] 保存异常: {write_err}")
                                        fotric_image_path = ""
                            except Exception as e2:
                                print(f"[第{IMAGE_COUNT}帧] 保存红外图像失败: {type(e2).__name__}: {e2}")
                                import traceback
                                traceback.print_exc()
                                fotric_image_path = ""
                            
                    except Exception as e:
                        print(f"[第{IMAGE_COUNT}帧] 获取/保存红外相机数据失败: {e}")
                        fotric_image_path = ""
                        fotric_data_path = ""
                
                if(IMAGE_COUNT %100 ==0):#40秒更新一次显示信息（从25改为100）
                    print(f"[✓] 已保存图片 #{IMAGE_COUNT}")  # 调试信息
                image_queue.put(f"image-{IMAGE_COUNT}.jpg")

                #获取打印信息
                print(f"[第{IMAGE_COUNT}帧] 获取打印参数分类...")
                print(f"[第{IMAGE_COUNT}帧] 调用get_print_param_class_origin()...")
                flow_rate_class,feed_rate_class,z_offset_class,hotend_class,bed,hot_end = get_print_param_class_origin()
                print(f"[第{IMAGE_COUNT}帧] 打印参数获取成功")
                #flow_rate_class,feed_rate_class,z_offset_class,hotend_class,bed,hot_end = get_print_param_class_by_model(ids_image_path,model)
                
                # 构建 CSV 数据行，包含旁轴相机、XYZ坐标和红外相机信息
                print(f"[第{IMAGE_COUNT}帧] 构建CSV行数据...")
                print(f"[第{IMAGE_COUNT}帧] 当前坐标: X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f}")
                row_data = [ids_image_path,computer_image_path,current_time,img_timestamp,coord_timestamp,time_diff_ms,
                            current_x,current_y,current_z,
                            FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,hot_end,bed,IMAGE_COUNT,
                            flow_rate_class,feed_rate_class,z_offset_class,hotend_class,
                            fotric_temp_min_cached,fotric_temp_max_cached,fotric_temp_avg_cached,
                            fotric_image_path,fotric_data_path]
                
                # 调试输出：显示坐标和相机路径
                if IMAGE_COUNT % 10 == 0:
                    print(f"[第{IMAGE_COUNT}帧] 坐标: X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f}")
                    print(f"[第{IMAGE_COUNT}帧] 旁轴相机: {computer_image_path}")
                    print(f"[第{IMAGE_COUNT}帧] 红外相机: {fotric_image_path}")
                if(IMAGE_COUNT %10 ==0):
                    FLOW_RATE_label.config(text=f"{flow_rate_class}")
                    FEED_RATE_label.config(text=f"{feed_rate_class}")
                    Z_OFF_label.config(text=f"{z_offset_class}")
                    HOTEND_label.config(text=f"{hotend_class}")
                #print(row_data)#调试用
                
                if(CLOSE_LOOP):
                    #自动闭环调控
                    flow_rate_offset = int(auto_close_loop(flow_rate_class,0.7,FLOW_RATE_LIST,0.2,40,-50))
                    feed_rate_offset = int(auto_close_loop(feed_rate_class,0.8,FEED_RATE_LIST,0.25,40,-50))
                    z_off_offset = round(auto_close_loop(z_offset_class,0.7,Z_OFF_LIST,0.2,0.16,-0.16),2)
                    hotend_offset = auto_close_loop(hotend_class,0.7,HOTEND_LIST,0.2,10,-10)
                    if(flow_rate_offset or feed_rate_offset or z_off_offset or hotend_offset):
                        if(IMAGE_COUNT % 5==0):#2秒更新一次
                            res_json = get_printer_status()
                            hot_end = res_json['temperature']['tool0']['actual']
                            if(abs(TARGET_HOTEND + hotend_offset - hot_end)>25):
                                hotend_offset = 0
                                TARGET_HOTEND = 200 + (int)(np.random.rand()*10-5)
                            print(f"闭环调控偏移量,flow_rate:{flow_rate_offset},feed_rate:{feed_rate_offset},z_off:{z_off_offset},hotend:{hotend_offset}")
                            close_loop_label.config(text=f"闭环调控状态偏移量 flow_rate:{flow_rate_offset},feed_rate:{feed_rate_offset},z_off:{z_off_offset},hotend:{hotend_offset}")
                            FLOW_RATE +=flow_rate_offset
                            FEED_RATE +=feed_rate_offset
                            Z_OFF +=z_off_offset
                            TARGET_HOTEND +=hotend_offset
                            #param_init()
                            change_param_auto(FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND)
                        
                #打开csv文件
                print(f"[第{IMAGE_COUNT}帧] 写入CSV...")
                with open(CSV_FILE, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_data)

                IMAGE_COUNT += 1
                print(f"[✓] 第{IMAGE_COUNT-1}帧保存成功，下一帧编号: {IMAGE_COUNT}")
                time.sleep(capture_interval)
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"[ERROR] 第{IMAGE_COUNT}帧捕获失败:\n{error_msg}")
                time.sleep(1)  # 出错后等待1秒再重试
        else:
            time.sleep(1)  # 暂停状态下，每秒检查一次是否恢复

#ids相机获取图像数据，处理过程
def ids_process_img():
    """
    获取IDS随轴相机图像（已旋转180度校正安装方向）
    如果IDS相机不可用，则使用替代方案：
    1. 尝试使用替代网络摄像头
    2. 如果都不可用，返回空白图像或旁轴相机图像
    
    注意：所有返回的图像都已经旋转180度，以校正摄像头倒装问题
    """
    global ids_camera_enabled, ids_alternative_camera
    
    result_image = None
    
    # ========== 方案1: IDS相机可用时，使用IDS相机 ==========
    if ids_camera_enabled and datastream is not None:
        try:
            buffer = datastream.WaitForFinishedBuffer(1000)
            if buffer is None:
                raise Exception("WaitForFinishedBuffer 返回 None - 可能是设备未就绪或连接失败")
            
            image_data = ids_peak_ipl_extension.BufferToImage(buffer)
            if image_data is None:
                raise Exception("BufferToImage 返回 None - 图像转换失败")
            
            # 从 IDS 图像获取数据
            width = image_data.Width()
            height = image_data.Height()
            pixel_format = image_data.PixelFormat()
            
            if width <= 0 or height <= 0:
                raise Exception(f"无效的图像尺寸: {width}x{height}")
            
            image_np = image_data.get_numpy()
            if image_np is None:
                raise Exception("get_numpy() 返回 None - 无法获取图像数据")
            
            # 根据像素格式处理
            if pixel_format == "RGB8":
                image_np = image_np.reshape((height, width, 3))
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            elif pixel_format == "Mono8":
                image_np = image_np.reshape((height, width))
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            
            datastream.QueueBuffer(buffer)
            result_image = image_np
            
        except Exception as e:
            print(f"[IDS相机] 获取图像出错: {e}，将使用替代方案")
            # 出错时降级到替代方案
    
    # ========== 方案2: IDS相机不可用时，使用替代网络摄像头 ==========
    if result_image is None:
        try:
            # 检查替代摄像头是否存在且已打开
            if 'ids_alternative_camera' in globals() and ids_alternative_camera is not None:
                ret, frame = ids_alternative_camera.read()
                if ret and frame is not None and frame.size > 0:
                    # 保持原始分辨率，不进行缩放
                    # 替代摄像头通常是1920x1080或1280x720
                    result_image = frame
        except Exception as e:
            print(f"[IDS替代] 获取替代摄像头图像失败: {e}")
    
    # ========== 方案3: 尝试使用旁轴相机图像作为后备 ==========
    if result_image is None:
        try:
            computer_frame = get_computer_camera_frame()
            if computer_frame is not None:
                # 保持原始分辨率
                result_image = computer_frame
        except:
            pass
    
    # ========== 方案4: 返回空白图像 ==========
    if result_image is None:
        print("[IDS替代] 所有摄像头都不可用，返回空白图像")
        blank_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 添加文字提示（适配1920x1080分辨率）
        cv2.putText(blank_img, "IDS Camera Offline", (700, 540), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.putText(blank_img, "Using Alternative Mode", (680, 620), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
        result_image = blank_img
    
    # ========== 旋转180度校正安装方向 ==========
    # 随轴相机是倒着安装的，需要旋转180度
    if result_image is not None:
        result_image = cv2.flip(result_image, -1)
    
    return result_image


# 文件保存位置
def select_save_directory():
    directory = filedialog.askdirectory()
    save_directory.set(directory)

def start_recording():
    global is_recording, record_thread, camera_thread,FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,PRIMARY_Z_OFF,computer_camera,INIT_MODEL,model
    interval_value = interval_entry.get()

    #参数初始化
    param_init()
    change_param_auto(FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,True)

    if not interval_value:
        messagebox.showwarning("警告", "请键入拍摄间隔时间")
        return
    
    # 检旁轴相机是否打开，如果失败则尝试重新初始化
    if computer_camera is None or not computer_camera.isOpened():
        print("[系统] 尝试重新初始化旁轴相机...")
        if not find_and_init_camera():
            messagebox.showwarning("警告", "旁轴相机未连接，请检查Nikon Webcam Utility驱动是否正常运行")
            return
    
    # 测试摄像头是否能读取有效的帧
    try:
        test_frame = get_computer_camera_frame()
        if test_frame is None or test_frame.size == 0:
            print("[警告] 摄像头无法读取有效帧")
            messagebox.showwarning("警告", "旁轴相机无法读取画面，请检查驱动状态")
            return
        print("[✓] 旁轴相机可正常读取")
    except Exception as e:
        print(f"[错误] 摄像头测试失败: {e}")
        messagebox.showerror("错误", f"摄像头测试失败: {e}")
        return
    
    if not is_recording:
        # ========== 录制前确认对话框（带模型选择）==========
        # 创建自定义对话框
        confirm_dialog = tk.Toplevel(root)
        confirm_dialog.title("录制确认")
        confirm_dialog.geometry("450x600")  # 增加高度避免按钮被遮挡
        confirm_dialog.transient(root)  # 设置为主窗口的模态对话框
        confirm_dialog.grab_set()
        confirm_dialog.resizable(False, False)
        
        # 获取采集模式
        acq_mode = acquisition_mode.get()
        global ACQUISITION_MODE, CURRENT_EXP_NUMBER
        ACQUISITION_MODE = acq_mode
        
        # 配置信息框架
        info_frame = tk.LabelFrame(confirm_dialog, text="录制配置", font=('Arial', 11, 'bold'), padx=10, pady=10)
        info_frame.pack(fill='x', padx=15, pady=10)
        
        # 显示配置信息
        tk.Label(info_frame, text=f"📷 拍摄间隔: {interval_value} 秒", font=('Arial', 10)).pack(anchor='w', pady=2)
        tk.Label(info_frame, text=f"📁 保存位置:", font=('Arial', 10)).pack(anchor='w', pady=2)
        save_path_label = tk.Label(info_frame, text=f"   {save_directory.get()}", font=('Arial', 9), fg='#666666', wraplength=380)
        save_path_label.pack(anchor='w', pady=2)
        
        # 采集模式显示
        mode_color = '#4CAF50' if acq_mode == "普通采集" else '#FF5722'
        tk.Label(info_frame, text=f"📝 采集模式: {acq_mode}", font=('Arial', 10, 'bold'), fg=mode_color).pack(anchor='w', pady=2)
        
        # 如果是标准化采集，显示实验编号和参数
        if acq_mode == "标准化采集":
            CURRENT_EXP_NUMBER = int(exp_number.get())
            exp_config = STANDARDIZED_CONFIG[CURRENT_EXP_NUMBER]
            z_final = PRIMARY_Z_OFF + exp_config["z_offset"]
            tk.Label(info_frame, text=f"🔬 实验编号: 第{CURRENT_EXP_NUMBER}组", font=('Arial', 10), fg='#2196F3').pack(anchor='w', pady=2)
            tk.Label(info_frame, text=f"   温度: {exp_config['temp']}°C, Z偏移: {exp_config['z_offset']}mm, Z最终: {z_final:.2f}mm", 
                    font=('Arial', 9), fg='#666666').pack(anchor='w', pady=1)
        
        # 状态控制
        status_frame = tk.LabelFrame(confirm_dialog, text="状态控制", font=('Arial', 11, 'bold'), padx=10, pady=10)
        status_frame.pack(fill='x', padx=15, pady=5)
        
        tk.Label(status_frame, text=f"• 自动修改参数: {PRINT_STATE}", font=('Arial', 10)).pack(anchor='w', pady=2)
        tk.Label(status_frame, text=f"• 闭环调控: {CLOSE_LOOP}", font=('Arial', 10), 
                fg='#FF5722' if CLOSE_LOOP else '#666666').pack(anchor='w', pady=2)
        tk.Label(status_frame, text=f"• 循环参数: {PARAM_LOOP}", font=('Arial', 10)).pack(anchor='w', pady=2)
        
        # 模型选择框架
        model_frame = tk.LabelFrame(confirm_dialog, text="模型选择", font=('Arial', 11, 'bold'), padx=10, pady=10)
        model_frame.pack(fill='x', padx=15, pady=10)
        
        selected_model = tk.StringVar(value="none")
        model_result = {"confirmed": False, "model_path": None}
        
        if CLOSE_LOOP:
            # 闭环调控开启，显示模型选择
            tk.Label(model_frame, text="请选择闭环调控使用的模型：", font=('Arial', 10)).pack(anchor='w', pady=5)
            
            # 可用模型列表
            models = [
                ("完整模型 (推荐)", "full/model_full.pt"),
                ("拼接融合模型", "concat_only/model_concat.pt"),
                ("单IDS模型", "ids_only/model_ids_only.pt"),
                ("无MMD模型", "no_mmd/model_no_mmd.pt"),
                ("RGB-only模型", "rgb_only/model_rgb_only.pt"),
            ]
            
            for i, (name, path) in enumerate(models):
                rb = tk.Radiobutton(model_frame, text=name, variable=selected_model, 
                                   value=path, font=('Arial', 10))
                rb.pack(anchor='w', pady=2)
                if i == 0:
                    selected_model.set(path)  # 默认选择第一个
        else:
            # 闭环调控关闭，不加载模型
            tk.Label(model_frame, text="⚠️ 闭环调控未启用，不使用模型预测", 
                    font=('Arial', 10), fg='#666666').pack(anchor='w', pady=10)
            tk.Label(model_frame, text="将使用阈值分类方法进行参数评估", 
                    font=('Arial', 9), fg='#999999').pack(anchor='w')
            selected_model.set("none")
        
        # 按钮框架
        btn_frame = tk.Frame(confirm_dialog)
        btn_frame.pack(fill='x', padx=15, pady=15)
        
        def on_confirm():
            model_result["confirmed"] = True
            if CLOSE_LOOP and selected_model.get() != "none":
                model_result["model_path"] = selected_model.get()
            confirm_dialog.destroy()
        
        def on_cancel():
            model_result["confirmed"] = False
            confirm_dialog.destroy()
        
        tk.Button(btn_frame, text="确定开始", command=on_confirm, 
                 font=('Arial', 10, 'bold'), bg='#4CAF50', fg='white', 
                 width=12, height=1).pack(side='left', padx=20)
        
        tk.Button(btn_frame, text="取消", command=on_cancel, 
                 font=('Arial', 10, 'bold'), bg='#f44336', fg='white', 
                 width=12, height=1).pack(side='right', padx=20)
        
        # 等待对话框关闭
        root.wait_window(confirm_dialog)
        
        # 检查用户是否确认
        if not model_result["confirmed"]:
            print("[系统] 用户取消录制")
            return
        
        # 根据选择加载模型
        if CLOSE_LOOP and model_result["model_path"]:
            model_full_path = Path(__file__).parent.parent / 'saved_models' / model_result["model_path"]
            print(f"[系统] 加载选定模型: {model_full_path}")
            try:
                from predict import load_model
                model = load_model(model_full_path)
                INIT_MODEL = True
                print(f"[✓] 模型加载完成: {model_result['model_path']}")
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {e}")
                return
        else:
            print("[系统] 闭环调控未启用，跳过模型加载")
        
        print("[系统] 用户确认录制，开始启动...")
        # =====================================
        
        is_recording = True
        capture_interval = float(interval_value)
        image_queue = queue.Queue()
        # 启动摄像头拍照线程
        camera_thread = threading.Thread(target=capture_images, args=(capture_interval, image_queue))
        camera_thread.daemon = True  # 设置为守护线程，应用退出时自动终止
        camera_thread.start()

def stop_recording():
    global is_paused
    if is_recording:
        is_paused = True

def continue_recording():
    global is_paused
    if is_recording:
        is_paused = False

def complete_experiment():
    global is_recording, computer_camera, camera_thread, ids_camera_enabled
    
    print("\n[系统] 正在结束实验并释放资源...")
    
    if is_recording:
        is_recording = False
        # 使用超时 join()，避免永久阻塞（最多等待5秒）
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=5)
            if camera_thread.is_alive():
                print("[警告] 图像捕获线程未能正常退出")
    
    # 关闭旁轴相机
    if computer_camera is not None and computer_camera.isOpened():
        computer_camera.release()
        print("✓ 旁轴相机已关闭")
    
    # 关闭IDS相机或替代摄像头
    if ids_camera_enabled and device is not None:
        # IDS相机模式
        try:
            nodemap_remote.FindNode("AcquisitionStop").Execute()
            datastream.StopAcquisition()
            ids_peak.Library.Close()
            print("✓ IDS相机已关闭")
        except Exception as e:
            print(f"[警告] 关闭IDS相机时出错: {e}")
    else:
        # 替代摄像头模式
        try:
            if 'ids_alternative_camera' in globals() and ids_alternative_camera is not None:
                ids_alternative_camera.release()
                print("✓ IDS替代摄像头已关闭")
        except Exception as e:
            print(f"[警告] 关闭IDS替代摄像头时出错: {e}")
    
    # 释放资源
    stop_websocket()
    
    root.destroy()
    print("[系统] 实验结束，资源已释放")
    # 关闭 Tkinter 主窗口

def test_save_single_frame():
    """
    测试保存单帧数据
    按照采集数据的逻辑保存一帧数据到测试目录
    """
    global IMAGE_COUNT, FLOW_RATE, FEED_RATE, Z_OFF, TARGET_HOTEND
    global current_x, current_y, current_z, ids_camera_enabled
    
    print("\n" + "="*70)
    print("[测试保存] 开始保存单帧数据...")
    print("="*70)
    
    try:
        # 创建测试保存目录
        base_dir = save_directory.get()
        test_dir = os.path.join(base_dir, "test_single_frame")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        test_task_dir = os.path.join(test_dir, f"test_{timestamp}")
        
        image_dir = os.path.join(test_task_dir, "images")
        ids_image_dir = os.path.join(image_dir, "IDS_Camera")
        computer_image_dir = os.path.join(image_dir, "Computer_Camera")
        fotric_image_dir = os.path.join(image_dir, "Fotric_Camera")
        fotric_data_dir = os.path.join(image_dir, "Fotric_Data")
        
        # 创建目录
        os.makedirs(ids_image_dir, exist_ok=True)
        os.makedirs(computer_image_dir, exist_ok=True)
        os.makedirs(fotric_image_dir, exist_ok=True)
        os.makedirs(fotric_data_dir, exist_ok=True)
        
        print(f"[测试保存] 保存目录: {test_task_dir}")
        
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S-%f")[:-2]
        
        # ========== 1. 获取并保存IDS相机图像 ==========
        print("[测试保存] 获取IDS相机图像...")
        # IDS相机图像（已自动旋转180度校正安装方向）
        ids_image_np = ids_process_img()
        
        ids_image_path = os.path.join(ids_image_dir, "test_image_IDS.jpg")
        # OpenCV BGR -> PIL RGB
        ids_img_rgb = cv2.cvtColor(ids_image_np, cv2.COLOR_BGR2RGB)
        ids_img = Image.fromarray(ids_img_rgb)
        if ids_img.mode == 'RGBA':
            ids_img = ids_img.convert('RGB')
        ids_img.save(ids_image_path, quality=95)
        print(f"[测试保存] ✓ IDS图像已保存: {ids_image_path}")
        
        # ========== 2. 获取并保存旁轴相机图像 ==========
        print("[测试保存] 获取旁轴相机图像...")
        computer_image_np = get_computer_camera_frame()
        
        computer_image_path = ""
        if computer_image_np is not None:
            computer_image_path = os.path.join(computer_image_dir, "test_image_Computer.jpg")
            computer_img = cv2.cvtColor(computer_image_np, cv2.COLOR_BGR2RGB)
            computer_pil_img = Image.fromarray(computer_img)
            computer_pil_img.save(computer_image_path, quality=95)
            print(f"[测试保存] ✓ 旁轴相机图像已保存: {computer_image_path}")
        else:
            print("[测试保存] ⚠ 旁轴相机图像获取失败")
        
        # ========== 3. 获取并保存红外相机图像和数据 ==========
        print("[测试保存] 获取红外相机图像...")
        fotric_image_path = ""
        fotric_data_path = ""
        fotric_temp_min_cached = 0.0
        fotric_temp_max_cached = 0.0
        fotric_temp_avg_cached = 0.0
        
        if fotric_enabled and fotric_device is not None and fotric_device.is_connected:
            with fotric_lock:
                fotric_temp_min_cached = fotric_temp_min
                fotric_temp_max_cached = fotric_temp_max
                fotric_temp_avg_cached = fotric_temp_avg
            
            try:
                thermal_data = fotric_device.get_thermal_data()
                if thermal_data is not None:
                    # 保存原始热像数据
                    fotric_data_path = os.path.join(fotric_data_dir, "test_thermal_data.npz")
                    np.savez_compressed(
                        fotric_data_path,
                        thermal_data=thermal_data.astype(np.float32),
                        timestamp=current_time,
                        temp_min=float(np.min(thermal_data)),
                        temp_max=float(np.max(thermal_data)),
                        temp_avg=float(np.mean(thermal_data))
                    )
                    print(f"[测试保存] ✓ 红外数据已保存: {fotric_data_path}")
                    
                    # 保存彩色热像图
                    if fotric_temp_max_cached > fotric_temp_min_cached:
                        normalized = ((thermal_data - fotric_temp_min_cached) / 
                                     (fotric_temp_max_cached - fotric_temp_min_cached) * 255).astype(np.uint8)
                    else:
                        normalized = np.full_like(thermal_data, 128, dtype=np.uint8)
                    
                    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                    fotric_image_path = os.path.join(fotric_image_dir, "test_image_Fotric.jpg")
                    cv2.imwrite(fotric_image_path, colored)
                    print(f"[测试保存] ✓ 红外图像已保存: {fotric_image_path}")
                else:
                    print("[测试保存] ⚠ 红外数据获取失败")
            except Exception as e:
                print(f"[测试保存] ⚠ 红外相机保存失败: {e}")
        else:
            print("[测试保存] ⚠ 红外相机未启用")
        
        # ========== 4. 获取打印状态 ==========
        print("[测试保存] 获取打印状态...")
        try:
            res_json = get_printer_status()
            bed = res_json['temperature']['bed']['actual']
            hot_end = res_json['temperature']['tool0']['actual']
            print(f"[测试保存] ✓ 打印状态获取成功: 热端={hot_end}°C, 热床={bed}°C")
        except Exception as e:
            print(f"[测试保存] ⚠ 获取打印状态失败: {e}")
            bed = 0.0
            hot_end = 0.0
        
        # ========== 5. 获取参数分类 ==========
        print("[测试保存] 获取参数分类...")
        try:
            flow_rate_class = get_param_class(FLOW_RATE, config.PARAM_THRESHOLDS["flow_rate"])
            feed_rate_class = get_param_class(FEED_RATE, config.PARAM_THRESHOLDS["feed_rate"])
            z_offset_class = get_param_class(Z_OFF, config.PARAM_THRESHOLDS["z_offset"])
            hotend_class = get_param_class(hot_end, config.PARAM_THRESHOLDS["hotend"])
            print(f"[测试保存] ✓ 参数分类: Flow={flow_rate_class}, Feed={feed_rate_class}, Z={z_offset_class}, Hotend={hotend_class}")
        except Exception as e:
            print(f"[测试保存] ⚠ 获取参数分类失败: {e}")
            flow_rate_class = feed_rate_class = z_offset_class = hotend_class = 1
        
        # ========== 6. 保存到CSV ==========
        # 获取时间戳
        img_timestamp = time.time()
        coord_timestamp = last_coord_timestamp
        time_diff_ms = abs(img_timestamp - coord_timestamp) * 1000
        
        csv_file = os.path.join(test_task_dir, "test_print_message.csv")
        header = ['image_path','computer_image_path','timestamp','img_timestamp','coord_timestamp','time_diff_ms',
                  'current_x','current_y','current_z',
                  'flow_rate','feed_rate','z_offset','target_hotend','hot_end','bed','img_num',
                  'flow_rate_class','feed_rate_class','z_offset_class','hotend_class',
                  'fotric_temp_min','fotric_temp_max','fotric_temp_avg','fotric_image_path','fotric_data_path']
        
        # 写入CSV
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            row_data = [ids_image_path, computer_image_path, current_time, img_timestamp, coord_timestamp, time_diff_ms,
                       current_x, current_y, current_z,
                       FLOW_RATE, FEED_RATE, Z_OFF, TARGET_HOTEND, hot_end, bed, "test",
                       flow_rate_class, feed_rate_class, z_offset_class, hotend_class,
                       fotric_temp_min_cached, fotric_temp_max_cached, fotric_temp_avg_cached,
                       fotric_image_path, fotric_data_path]
            writer.writerow(row_data)
        
        print(f"[测试保存] ✓ CSV已保存: {csv_file}")
        
        print("\n" + "="*70)
        print("[测试保存] 单帧数据保存完成！")
        print(f"保存位置: {test_task_dir}")
        print("="*70 + "\n")
        
        # 显示成功提示
        messagebox.showinfo("测试保存成功", f"单帧数据已保存到:\n{test_task_dir}")
        
    except Exception as e:
        print(f"[测试保存] ✗ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("测试保存失败", f"保存失败:\n{e}")

def get_random_huge_error_param():
    #随机生成较大误差参数
    rand = np.random.rand()#20~60,150~200
    flow_rate = int((rand * 80 + 20) if rand < 0.5 else ((rand - 0.5) * 50 + 150))
    rand = np.random.rand()#20~60,150~200
    feed_rate = int((rand * 80 + 20) if rand < 0.5 else ((rand - 0.5) * 50 + 150))
    rand = np.random.rand()#-0.08~-0.04,1.5~2.5
    z_off = rand * 0.04 - 0.08 if rand < 0.5 else (rand - 0.5) * 1.0 + 1.5
    rand = np.random.rand()#120~150,220~250
    target_hotend = int(rand * 60 + 120 if rand < 0.5 else (rand-0.5)* 60 + 220)
    return flow_rate,feed_rate,z_off,target_hotend

#普通版
def get_print_param_class_origin():
    res_json = get_printer_status()
    bed = res_json['temperature']['bed']['actual']
    hot_end = res_json['temperature']['tool0']['actual']
    flow_rate_class = get_param_class(FLOW_RATE, config.PARAM_THRESHOLDS["flow_rate"])
    feed_rate_class = get_param_class(FEED_RATE, config.PARAM_THRESHOLDS["feed_rate"])
    z_offset_class = get_param_class(Z_OFF, config.PARAM_THRESHOLDS["z_offset"])
    hotend_class = get_param_class(hot_end, config.PARAM_THRESHOLDS["hotend"])   
    return flow_rate_class, feed_rate_class, z_offset_class, hotend_class, bed, hot_end

#模型版 - 完整4模态预测
def get_print_param_class_by_model(ids_image_path, model):
    """
    使用完整4模态数据进行模型预测
    
    Args:
        ids_image_path: IDS相机图像路径
        model: 加载的预测模型
    
    Returns:
        tuple: (flow_rate_class, feed_rate_class, z_offset_class, hotend_class, bed, hot_end)
    """
    import torch
    from PIL import Image
    
    # 获取当前打印机状态和温度
    res_json = get_printer_status()
    bed = res_json['temperature']['bed']['actual']
    hot_end = res_json['temperature']['tool0']['actual']
    
    # ========== 1. 加载IDS随轴相机图像 (448x448) ==========
    ids_img = Image.open(ids_image_path).convert('RGB')
    ids_img = ids_img.resize((448, 448))
    ids_array = np.array(ids_img) / 255.0
    # ImageNet归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ids_array = (ids_array - mean) / std
    ids_tensor = torch.from_numpy(ids_array).permute(2, 0, 1).float()
    
    # ========== 2. 获取旁轴相机图像 (224x224) ==========
    computer_frame = get_computer_camera_frame()
    if computer_frame is not None:
        # OpenCV BGR -> RGB
        computer_frame_rgb = cv2.cvtColor(computer_frame, cv2.COLOR_BGR2RGB)
        computer_img = Image.fromarray(computer_frame_rgb).resize((224, 224))
        computer_array = np.array(computer_img) / 255.0
        computer_array = (computer_array - mean) / std
        computer_tensor = torch.from_numpy(computer_array).permute(2, 0, 1).float()
    else:
        # 如果旁轴相机不可用，使用IDS图像下采样
        computer_tensor = torch.from_numpy(
            cv2.resize(ids_array, (224, 224))
        ).permute(2, 0, 1).float()
    
    # ========== 3. 获取Fotric红外图像 (224x224) ==========
    global fotric_latest_frame, fotric_temp_min, fotric_temp_max, fotric_temp_avg, fotric_enabled
    
    if fotric_enabled and fotric_latest_frame is not None:
        with fotric_lock:
            thermal_data = fotric_latest_frame.copy()
            temp_min = fotric_temp_min
            temp_max = fotric_temp_max
            temp_avg = fotric_temp_avg
        
        # 创建伪彩色热像 (3通道)
        if temp_max > temp_min:
            normalized = ((thermal_data - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_data, dtype=np.uint8)
        
        # 应用热力图颜色映射 (JET)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        fotric_img = Image.fromarray(colored_rgb).resize((224, 224))
        fotric_array = np.array(fotric_img) / 255.0
        fotric_array = (fotric_array - mean) / std
        fotric_tensor = torch.from_numpy(fotric_array).permute(2, 0, 1).float()
        
        # 创建灰度热像 (1通道)
        thermal_resized = cv2.resize(thermal_data, (224, 224))
        # 温度归一化到 [0, 1] 范围
        thermal_norm = (thermal_resized - 20) / (250 - 20)  # 假设温度范围 20-250°C
        thermal_norm = np.clip(thermal_norm, 0, 1)
        thermal_tensor = torch.from_numpy(thermal_norm).unsqueeze(0).float()
    else:
        # 如果红外相机不可用，使用零填充
        fotric_tensor = torch.zeros(3, 224, 224)
        thermal_tensor = torch.zeros(1, 224, 224)
        temp_min = temp_max = temp_avg = 0.0
    
    # ========== 4. 构造工艺参数 (10维) ==========
    # [current_x, current_y, current_z, flow_rate, feed_rate, z_offset, hot_end, thermal_min, thermal_max, thermal_avg]
    global current_x, current_y, current_z, FLOW_RATE, FEED_RATE, Z_OFF, TARGET_HOTEND
    
    params = np.array([
        float(current_x),
        float(current_y), 
        float(current_z),
        float(FLOW_RATE),
        float(FEED_RATE),
        float(Z_OFF),
        float(hot_end),  # 使用实际温度而不是目标温度
        float(temp_min),
        float(temp_max),
        float(temp_avg)
    ], dtype=np.float32)
    
    # 参数标准化 (与训练时一致)
    # 前7个参数使用硬编码范围标准化到 [-1, 1]
    param_ranges = {
        'current_x': (0, 200),
        'current_y': (0, 200),
        'current_z': (0, 200),
        'flow_rate': (20, 200),
        'feed_rate': (20, 200),
        'z_offset': (-0.08, 0.32),
        'hot_end': (150, 250),
    }
    
    params_normalized = np.zeros_like(params)
    param_names = ['current_x', 'current_y', 'current_z', 'flow_rate', 'feed_rate', 'z_offset', 'hot_end']
    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[name]
        params_normalized[i] = 2 * (params[i] - min_val) / (max_val - min_val) - 1
        params_normalized[i] = np.clip(params_normalized[i], -1, 1)
    
    # 温度统计参数归一化 (范围 [20, 250])
    params_normalized[7] = (params[7] - 20) / (250 - 20)  # thermal_min
    params_normalized[8] = (params[8] - 20) / (250 - 20)  # thermal_max
    params_normalized[9] = (params[9] - 20) / (250 - 20)  # thermal_avg
    
    params_tensor = torch.from_numpy(params_normalized).float()
    
    # ========== 5. 构造batch_data字典 ==========
    batch_data = {
        'ids': ids_tensor,
        'computer': computer_tensor,
        'fotric': fotric_tensor,
        'thermal': thermal_tensor,
        'params': params_tensor,
    }
    
    # ========== 6. 调用模型预测 ==========
    try:
        from predict import predict_single
        pred_param = predict_single(batch_data, model)
        flow_rate_class = pred_param[0]
        feed_rate_class = pred_param[1]
        z_offset_class = pred_param[2]
        hotend_class = pred_param[3]
        print(f"[模型预测] Flow:{flow_rate_class} Feed:{feed_rate_class} Z:{z_offset_class} Hotend:{hotend_class}")
    except Exception as e:
        print(f"[模型预测错误] {e}, 使用阈值分类作为后备")
        # 后备方案：使用阈值分类
        flow_rate_class = get_param_class(FLOW_RATE, config.PARAM_THRESHOLDS["flow_rate"])
        feed_rate_class = get_param_class(FEED_RATE, config.PARAM_THRESHOLDS["feed_rate"])
        z_offset_class = get_param_class(Z_OFF, config.PARAM_THRESHOLDS["z_offset"])
        hotend_class = get_param_class(hot_end, config.PARAM_THRESHOLDS["hotend"])
    
    return flow_rate_class, feed_rate_class, z_offset_class, hotend_class, bed, hot_end

#改变是否修改参数
def update_print_state():
    global PRINT_STATE,CLOSE_LOOP,PARAM_LOOP
    if PRINT_STATE == False:
        PRINT_STATE = True
    else:
        PRINT_STATE = False
    state_label.config(text=f"自动修改状态: {PRINT_STATE},闭环调控状态：{CLOSE_LOOP},循环参数状态:{PARAM_LOOP}")

#改变是否闭环调控
def update_close_loop_state():
    global PRINT_STATE,CLOSE_LOOP,PARAM_LOOP
    if CLOSE_LOOP == False:
        CLOSE_LOOP = True
    else:
        CLOSE_LOOP = False
    state_label.config(text=f"自动修改状态: {PRINT_STATE},闭环调控状态：{CLOSE_LOOP},循环参数状态:{PARAM_LOOP}")

#改变是否循环参数
def update_param_loop_state():
    global PRINT_STATE,CLOSE_LOOP,PARAM_LOOP
    if PARAM_LOOP == False:
        PARAM_LOOP = True
    else:
        PARAM_LOOP = False
    state_label.config(text=f"自动修改状态: {PRINT_STATE},闭环调控状态：{CLOSE_LOOP},循环参数状态:{PARAM_LOOP}")

#菜单界面的修改参数按钮逻辑   
def change_param_by_button():
    global FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,PRIMARY_Z_OFF,CUR_Z_OFF
    FLOW_RATE = int(FLOW_RATE_entry.get())
    FEED_RATE = int(FEED_RATE_entry.get())
    Z_OFF = float(Z_OFF_entry.get())
    send_z_off = Z_OFF-CUR_Z_OFF
    CUR_Z_OFF = Z_OFF
    TARGET_HOTEND = int(TARGET_HOTEND_entry.get())
    print(f"[手动修改] 热端={TARGET_HOTEND}, 流量={FLOW_RATE}, 速率={FEED_RATE}, Z补偿={Z_OFF}")
    
    # 使用异步发送，不阻塞UI
    send_gcode_async(f"M104 S{TARGET_HOTEND}")
    send_gcode_async(f"M290 Z{send_z_off}")
    send_gcode_async(f"M221 S{FLOW_RATE}")
    send_gcode_async(f"M220 S{FEED_RATE}")

def change_param_auto(FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,init=None):
    TARGET_HOTEND = int(TARGET_HOTEND)
    global PRIMARY_Z_OFF,CUR_Z_OFF
    if init:
        send_gcode(f"M851 Z{PRIMARY_Z_OFF}")
        print(f"初始化Z轴补偿{PRIMARY_Z_OFF}")
    
    send_z_off = Z_OFF-CUR_Z_OFF
    CUR_Z_OFF = Z_OFF
    flow_rate.set(FLOW_RATE)
    feed_rate.set(FEED_RATE)
    z_off.set(Z_OFF)
    target_hotend.set(TARGET_HOTEND)
    
    # 使用异步方式发送所有G代码，避免阻塞主线程
    print(f"[参数修改] 发送命令: M104 S{TARGET_HOTEND}, M290 Z{send_z_off}, M221 S{FLOW_RATE}, M220 S{FEED_RATE}")
    send_gcode_async(f"M104 S{TARGET_HOTEND}")  # 设置热端温度
    send_gcode_async(f"M290 Z{send_z_off}")     # 调整Z轴
    send_gcode_async(f"M221 S{FLOW_RATE}")      # 设置流量
    send_gcode_async(f"M220 S{FEED_RATE}")      # 设置速度
    
    # 异步发送M114获取坐标，不阻塞
    if ws and is_websocket_connected:
        ws.send(json.dumps(["sendCommand", f"M114"]))

FLOW_RATE_LIST = deque(maxlen=10)
FEED_RATE_LIST = deque(maxlen=10)
Z_OFF_LIST = deque(maxlen=12)
HOTEND_LIST = deque(maxlen=15)
#根据类别,模式阈值，序列，最小量，最大增量和最大减量调控
def auto_close_loop(param_class,theta_mode,list_,lmin,max_increase,max_decrease):
    list_.append(param_class)
    counter = Counter(list_)
    most_common_value, most_common_count = counter.most_common(1)[0]
    if most_common_value == 1:
        return 0
    else:
        mode = most_common_count/len(list_)
        if mode > theta_mode:#接受这个模态
            #映射
            mapped = (mode-theta_mode)/(1-theta_mode)*(1-lmin)+lmin
            if most_common_value == 0:
                return max_increase*mapped
            elif most_common_value == 2:
                return max_decrease*mapped
        else:
            return 0


def param_init():
    global FEED_RATE,FLOW_RATE,Z_OFF,TARGET_HOTEND,HOTEND_TEMP,HOTEND_TEMP_DIRECTION
    FLOW_RATE = 100
    FEED_RATE = 100
    Z_OFF = 0
    TARGET_HOTEND = 200
    
    # 重置温度循环状态
    HOTEND_TEMP = 200
    HOTEND_TEMP_DIRECTION = 1
    
    flow_rate.set(FLOW_RATE)
    feed_rate.set(FEED_RATE)
    z_off.set(Z_OFF)
    target_hotend.set(TARGET_HOTEND)

'''
#旧随机生成参数
def param_ramdon():
    global FEED_RATE,FLOW_RATE,Z_OFF,TARGET_HOTEND
    #随机获取打印数据
    FLOW_RATE = np.random.randint(20,201)
    FEED_RATE = np.random.randint(20,201)
    Z_OFF = (np.random.randint(0,41)-8)*0.01
    TARGET_HOTEND = np.random.randint(150,251)
    flow_rate.set(FLOW_RATE)
    feed_rate.set(FEED_RATE)
    z_off.set(Z_OFF)
    target_hotend.set(TARGET_HOTEND)
''' 
# 新随机生成参数
def param_ramdon():
    global FEED_RATE, FLOW_RATE, Z_OFF, TARGET_HOTEND
    
    # 定义可选的列表范围
    # rate_options 包含了 20 到 200，步长为 10 的整数
    rate_options = [20, 30, 40, 50, 60, 70, 80, 90, 100, 
                    110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    
    # z_off_options 包含了指定的浮点数
    z_off_options = [-0.32,-0.28,-0.24,-0.20,-0.16,-0.08, -0.04, 0, 0.04, 0.08, 0.16, 0.20,0.24,0.28,0.32]
    
    

    # 随机获取打印数据
    # 使用 np.random.choice 从列表中随机抽取
    FLOW_RATE = np.random.choice(rate_options)
    FEED_RATE = np.random.choice(rate_options)
    Z_OFF = np.random.choice(z_off_options)
    
    # TARGET_HOTEND 保持不变

    # 将生成的值设置到对应的变量中
    flow_rate.set(FLOW_RATE)
    feed_rate.set(FEED_RATE)
    z_off.set(Z_OFF)
    target_hotend.set(TARGET_HOTEND)

# ========== 摄像头初始化 ==========
print("========== 初始化摄像头 ==========")

# 初始化红外相机 (Fotric 628ch)
fotric_config = CAMERA_CONFIG["fotric"]
print("正在初始化红外相机(Fotric 628ch)...")

try:
    if not fotric_config["enabled"]:
        print("[红外相机] 已在配置中禁用")
        fotric_enabled = False
        fotric_device = None
    else:
        # 使用真实的 FotricEnhancedDevice 进行连接
        fotric_device = FotricEnhancedDevice(
            ip=fotric_config["ip"],
            port=fotric_config["port"],
            username=fotric_config["username"],
            password=fotric_config["password"],
            simulation_mode=fotric_config["simulation_mode"],
            high_resolution=fotric_config["high_resolution"],
            update_rate=2.0,              # 2Hz 更新频率
            sample_density=40             # 采样密度
        )
        fotric_enabled = fotric_device.is_connected
        if fotric_enabled:
            print(f"✓ 红外相机初始化成功: {fotric_device.width}x{fotric_device.height} ({fotric_config['ip']}:{fotric_config['port']})")
        else:
            print("✗ 红外相机连接失败")
except Exception as e:
    print(f"✗ 红外相机初始化异常: {e}")
    fotric_enabled = False
    fotric_device = None

# 初始化旁轴相机
print("正在初始化旁轴相机...")

def find_and_init_camera():
    """枚举并找到虚拟摄像头（USB外接摄像头）"""
    global computer_camera, camera_opened
    
    config = CAMERA_CONFIG["computer"]
    if not config["enabled"]:
        print("[旁轴相机] 已在配置中禁用")
        camera_opened = False
        return False
    
    camera_id = config["device_id"]
    print(f"[摄像头检测] 正在初始化旁轴相机 (设备{camera_id})...")
    
    try:
        # ===== 使用DirectShow后端（Windows最快） =====
        backend = cv2.CAP_DSHOW  # DirectShow - 比AutoBackend快5-10倍
        computer_camera = cv2.VideoCapture(camera_id + backend)
        
        if not computer_camera.isOpened():
            print(f"[警告] DirectShow后端打开失败，尝试AUTO后端...")
            computer_camera = cv2.VideoCapture(camera_id)
            
            if not computer_camera.isOpened():
                print(f"✗ 无法打开摄像头设备 {camera_id}")
                camera_opened = False
                return False
        
        print(f"[初始化] 摄像头 {camera_id} 已打开，正在优化参数...")
        
        # ===== 立即设置缓冲区大小（加快初始化） =====
        computer_camera.set(cv2.CAP_PROP_BUFFERSIZE, config["buffer_size"])
        
        # ===== 禁用自动对焦（大幅加快初始化） =====
        if config["disable_autofocus"]:
            try:
                computer_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                print("[优化] 已禁用自动对焦")
            except:
                pass
        
        # ===== 禁用自动曝光 =====
        if config["disable_auto_exposure"]:
            try:
                computer_camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
                print("[优化] 已禁用自动曝光")
            except:
                pass
        
        # ===== 禁用自动白平衡 =====
        if config["disable_auto_whitebalance"]:
            try:
                computer_camera.set(cv2.CAP_PROP_AUTO_WB, 0)
                print("[优化] 已禁用自动白平衡")
            except:
                pass
        
        # ===== 设置分辨率 =====
        width, height = config["resolution"]
        computer_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        computer_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        computer_camera.set(cv2.CAP_PROP_FPS, config["fps"])
        
        # ===== 快速预热缓冲区 =====
        warm_up_frames = config["warm_up_frames"]
        print(f"[预热中] 清除缓冲区 ({warm_up_frames}帧)...")
        for j in range(warm_up_frames):
            ret, frame = computer_camera.read()
            if ret and frame is not None and j % 4 == 0:
                print(f"  [预热 {j+1}/{warm_up_frames}]")
            if j < warm_up_frames - 1:
                time.sleep(0.02)
        print("[完成] 缓冲区预热完成")
        
        # ===== 获取实际分辨率 =====
        actual_width = int(computer_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(computer_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = computer_camera.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ 旁轴摄像头初始化成功 (设备{camera_id}): {actual_width}x{actual_height} @ {actual_fps:.0f}FPS")
        camera_opened = True
        return True
        
    except Exception as e:
        print(f"[错误] 打开摄像头失败: {e}")
        import traceback
        traceback.print_exc()
        camera_opened = False
        return False

# 初始化摄像头
if not find_and_init_camera():
    print("[警告] 旁轴相机初始化失败，将在启动记录时重试")
    computer_camera = None

# 初始化 IDS peak（可选）
print("正在初始化IDS相机...")

# IDS相机相关全局变量
ids_camera_enabled = False
device = None
datastream = None
nodemap_remote = None
ids_alternative_camera_id = 2  # IDS替代摄像头设备ID（当IDS不可用时使用）
# 注意：设备2是Nikon Webcam Utility（虚拟驱动），不是真实摄像头
# 如果没有连接Nikon相机，设备2实际上不可用
# 真正的第三个摄像头应该是设备3

try:
    ids_peak.Library.Initialize()
    # 创建设备管理器
    device_manager = ids_peak.DeviceManager.Instance()
    # 更新设备列表
    device_manager.Update()
    
    # 检查是否有IDS相机
    if device_manager.Devices().empty():
        print("⚠️ 未找到 IDS 相机，将使用替代方案（网络摄像头）")
        ids_camera_enabled = False
    else:
        print("✓ 找到IDS相机")
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Exclusive)
        # 配置数据流
        nodemap_remote = device.RemoteDevice().NodeMaps()[0]
        auto_focus_node = nodemap_remote.FindNode("FocusAuto")
        if auto_focus_node.IsAvailable():
            # 正确设置枚举值的方法
            auto_focus_entries = auto_focus_node.Entries()
            for entry in auto_focus_entries:
                if entry.SymbolicValue() == "Off":
                    auto_focus_node.SetCurrentEntry(entry)
                    print("已关闭自动对焦")
                    break
        else:
            print("该相机不支持自动对焦功能")
        #设置固定焦距 
        nodemap_remote.FindNode("FocusStepper").SetValue(112)
        
        # 创建数据流
        datastream = device.DataStreams()[0].OpenDataStream()
        # 申请缓冲区
        payload_size = nodemap_remote.FindNode("PayloadSize").Value()
        for _ in range(10):
            buffer = datastream.AllocAndAnnounceBuffer(payload_size)
            datastream.QueueBuffer(buffer)
        # 开始采集
        datastream.StartAcquisition()
        nodemap_remote.FindNode("AcquisitionStart").Execute()
        
        ids_camera_enabled = True
        print("✓ IDS相机初始化成功")
        
except Exception as e:
    print(f"⚠️ IDS相机初始化失败: {e}")
    print("  将使用替代方案（网络摄像头）作为随轴相机")
    ids_camera_enabled = False
    device = None
    datastream = None
    nodemap_remote = None
    
# 如果IDS相机不可用，尝试初始化替代摄像头
if not ids_camera_enabled:
    ids_config = CAMERA_CONFIG["ids"]
    
    if not ids_config["enabled"]:
        print("\n[IDS替代] 随轴相机已在配置中禁用")
    else:
        print("\n[IDS替代] 尝试初始化网络摄像头作为随轴相机...")
        
        ids_alternative_camera = None
        
        # 从配置获取设备列表和要跳过的设备
        candidate_devices = ids_config["alternative_device_ids"]
        skip_devices = ids_config["skip_devices"]
        brightness_threshold = ids_config["brightness_threshold"]
        std_threshold = ids_config["std_threshold"]
        
        for dev_id in candidate_devices:
            # 跳过指定设备（如旁轴相机使用的设备）
            if dev_id in skip_devices:
                continue
                
            try:
                print(f"[IDS替代] 尝试打开设备 {dev_id}...")
                import cv2
                test_cam = cv2.VideoCapture(dev_id + cv2.CAP_DSHOW)
                
                if test_cam.isOpened():
                    # 测试读取一帧
                    ret, frame = test_cam.read()
                    if ret and frame is not None and frame.size > 0:
                        # 检查是否是Nikon虚拟摄像头（通常分辨率特殊或黑屏）
                        width = int(test_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(test_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # 简单的亮度检测
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        brightness = cv2.mean(gray)[0]
                        
                        print(f"  设备{dev_id}: {width}x{height}, 亮度={brightness:.1f}")
                        
                        # 如果亮度太低（接近黑屏），可能是Nikon虚拟摄像头或未连接
                        if brightness < brightness_threshold:
                            print(f"  ⚠️ 设备{dev_id} 画面太暗（亮度{brightness:.1f}），可能是虚拟摄像头或未连接")
                            test_cam.release()
                            continue
                        
                        # 额外检查：如果亮度适中但画面大部分是均匀的（可能是测试图案），也跳过
                        std_dev = np.std(gray)
                        if brightness < 30 and std_dev < std_threshold:
                            print(f"  ⚠️ 设备{dev_id} 画面过于均匀（std={std_dev:.1f}），可能是虚拟摄像头")
                            test_cam.release()
                            continue
                        
                        # 找到可用的摄像头
                        ids_alternative_camera = test_cam
                        ids_alternative_camera_id = dev_id
                        
                        # 从配置获取目标分辨率列表
                        target_resolutions = ids_config["fallback_resolutions"]
                        
                        actual_width = width
                        actual_height = height
                        
                        for target_w, target_h in target_resolutions:
                            ids_alternative_camera.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                            ids_alternative_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
                            ids_alternative_camera.set(cv2.CAP_PROP_FPS, ids_config["fps"])
                            ids_alternative_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            # 验证实际设置的分辨率
                            actual_width = int(ids_alternative_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(ids_alternative_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            if actual_width >= target_w * 0.9:  # 允许10%误差
                                print(f"  分辨率设置成功: {actual_width}x{actual_height}")
                                break
                            else:
                                print(f"  尝试分辨率 {target_w}x{target_h} 失败，实际为 {actual_width}x{actual_height}")
                        
                        print(f"✓ IDS替代摄像头初始化成功 (设备{dev_id}): {actual_width}x{actual_height}")
                        break
                    else:
                        print(f"  设备{dev_id} 无法读取有效帧")
                        test_cam.release()
                else:
                    print(f"  设备{dev_id} 无法打开")
                    test_cam.release()
                    
            except Exception as e:
                print(f"  设备{dev_id} 异常: {e}")
        
        if ids_alternative_camera is None:
            print("⚠️ 未找到可用的IDS替代摄像头")
            print("  提示：所有候选设备都不可用或被排除（Nikon虚拟驱动/画面太暗）")
            print("  将在界面上显示空白图像")

#---创建Tkinter窗口
root = tk.Tk()
root.title("3D打印监测与调控系统")
root.configure(bg='#f0f0f0')  # 设置背景色

# 设置窗口大小（适应1440x900及以上分辨率）
# 计算后的尺寸：左侧面板约1000x800，右侧面板约400x800
root.geometry("1400x850")
root.minsize(1280, 800)  # 最小窗口大小

#---设置字体样式
title_font = ('Arial', 12, 'bold')
label_font = ('Arial', 10)
param_label_font = ( 'Arial', 8)
button_font = ('Arial', 10, 'bold')
entry_font = ('Arial', 10)

# ========== 三相机并排布局 ==========
# 创建相机显示框架（左侧大区域）
camera_frame = tk.Frame(root, bg='#f0f0f0')
camera_frame.grid(row=0, column=0, rowspan=8, padx=10, pady=5, sticky='nsew')

# 旁轴相机标签 (左上)
computer_label_panel = tk.Frame(camera_frame, bg='#f0f0f0')
computer_label_panel.grid(row=0, column=0, padx=5, pady=5)
computer_camera_title = tk.Label(computer_label_panel, text="旁轴相机 (1920x1080)", bg='#f0f0f0', font=title_font)
computer_camera_title.pack()
# width/height是字符单位，对于640x480像素图像，大约需要80x25个字符
computer_video_label = tk.Label(computer_label_panel, bg='black', relief='ridge', bd=3)
computer_video_label.pack(padx=5, pady=5)

# IDS相机标签 (右上)
ids_label_panel = tk.Frame(camera_frame, bg='#f0f0f0')
ids_label_panel.grid(row=0, column=1, padx=5, pady=5)
ids_camera_title = tk.Label(ids_label_panel, text="随轴相机 (IDS/替代)", bg='#f0f0f0', font=title_font)
ids_camera_title.pack()
ids_video_label = tk.Label(ids_label_panel, bg='black', relief='ridge', bd=3)
ids_video_label.pack(padx=5, pady=5)

# 红外相机标签 (下方，跨两列)
fotric_label_panel = tk.Frame(camera_frame, bg='#f0f0f0')
fotric_label_panel.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
fotric_camera_title = tk.Label(fotric_label_panel, text="红外相机 (Fotric)", bg='#f0f0f0', font=title_font)
fotric_camera_title.pack()
fotric_video_label = tk.Label(fotric_label_panel, bg='black', relief='ridge', bd=3)
fotric_video_label.pack(padx=5, pady=5)

# 红外相机温度信息显示面板（放在红外图像下方）
fotric_temp_frame = tk.Frame(fotric_label_panel, bg='#e8e8e8')
fotric_temp_frame.pack(pady=2, fill='x')
tk.Label(fotric_temp_frame, text="温度统计", bg='#e8e8e8', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
fotric_temp_info = tk.Label(fotric_temp_frame, text="最小: -- | 平均: -- | 最大: --", bg='#e8e8e8', font=('Arial', 9), fg='#FF5722')
fotric_temp_info.pack(side='left', padx=10)

#---记录控制变量
is_recording = False
is_paused = False
record_thread = None
camera_thread = None
PRINT_STATE = False
CLOSE_LOOP = False
PARAM_LOOP = False

# ========== 右侧控制面板（统一在一个框架内）==========
right_panel = tk.Frame(root, bg='#f0f0f0', padx=10, pady=5, width=450)
right_panel.grid(row=0, column=1, rowspan=8, sticky='ns', padx=5, pady=5)
right_panel.grid_propagate(False)  # 防止子组件改变框架大小

#---拍摄间隔设置
interval_frame = tk.LabelFrame(right_panel, text="拍摄设置", bg='#f0f0f0', font=title_font)
interval_frame.pack(fill='x', pady=5)

tk.Label(interval_frame, text="拍摄间隔 (秒):", bg='#f0f0f0', font=label_font).grid(row=0, column=0, padx=5, pady=5)
interval_entry = tk.Entry(interval_frame, font=entry_font, width=8, relief='sunken', bd=2)
interval_entry.grid(row=0, column=1, padx=5, pady=5)
interval_entry.insert(0, "0.5")

#---保存位置设置
save_frame = tk.LabelFrame(right_panel, text="保存设置", bg='#f0f0f0', font=title_font)
save_frame.pack(fill='x', pady=5)

# 默认保存路径为G盘FDM最新数据文件夹
default_save_path = r"D:\College\Python_project\4Project\data\FDMdata"
save_directory = tk.StringVar(value=default_save_path)
tk.Label(save_frame, text="位置:", bg='#f0f0f0', font=label_font).grid(row=0, column=0, padx=5, pady=5)
save_entry = tk.Entry(save_frame, textvariable=save_directory, font=entry_font, width=20, relief='sunken', bd=2)
save_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(save_frame, text="选择", command=select_save_directory, font=button_font, 
          bg='#4CAF50', fg='white', relief='raised', width=6).grid(row=0, column=2, padx=5, pady=5)

#---状态控制
state_frame = tk.LabelFrame(right_panel, text="状态控制", bg='#f0f0f0', font=title_font)
state_frame.pack(fill='x', pady=5)

state_btn_frame = tk.Frame(state_frame, bg='#f0f0f0')
state_btn_frame.pack(fill='x', padx=3, pady=5)

tk.Button(state_btn_frame, text="修改参数(60s)", command=update_print_state, 
          font=button_font, bg='#9C27B0', fg='white', width=12).pack(side='left', padx=2)
tk.Button(state_btn_frame, text="闭环调控", command=update_close_loop_state, 
          font=button_font, bg='#3F51B5', fg='white', width=10).pack(side='left', padx=2)
tk.Button(state_btn_frame, text="循环参数", command=update_param_loop_state, 
          font=button_font, bg='#009688', fg='white', width=10).pack(side='left', padx=2)

state_label = tk.Label(state_frame, text=f"自动修改: {PRINT_STATE}, 闭环: {CLOSE_LOOP}, 循环: {PARAM_LOOP}", 
                       bg='#f0f0f0', font=label_font, fg='#333333')
state_label.pack(fill='x', padx=5, pady=2)

#---打印参数控制
param_frame = tk.LabelFrame(right_panel, text="打印参数控制", bg='#f0f0f0', font=title_font)
param_frame.pack(fill='x', pady=5)

# 参数标签和输入框
param_grid = tk.Frame(param_frame, bg='#f0f0f0')
param_grid.pack(fill='x', padx=5, pady=3)

tk.Label(param_grid, text="流量", bg='#f0f0f0', font=param_label_font).grid(row=0, column=0, padx=5)
tk.Label(param_grid, text="速度", bg='#f0f0f0', font=param_label_font).grid(row=0, column=1, padx=5)
tk.Label(param_grid, text="Z偏移", bg='#f0f0f0', font=param_label_font).grid(row=0, column=2, padx=5)
tk.Label(param_grid, text="温度", bg='#f0f0f0', font=param_label_font).grid(row=0, column=3, padx=5)

flow_rate = tk.StringVar(value=str(FLOW_RATE))
FLOW_RATE_entry = tk.Entry(param_grid, textvariable=flow_rate, font=entry_font, width=6, relief='sunken', bd=2)
FLOW_RATE_entry.grid(row=1, column=0, padx=5, pady=2)

feed_rate = tk.StringVar(value=str(FEED_RATE))
FEED_RATE_entry = tk.Entry(param_grid, textvariable=feed_rate, font=entry_font, width=6, relief='sunken', bd=2)
FEED_RATE_entry.grid(row=1, column=1, padx=5, pady=2)

z_off = tk.StringVar(value=str(Z_OFF))
Z_OFF_entry = tk.Entry(param_grid, textvariable=z_off, font=entry_font, width=6, relief='sunken', bd=2)
Z_OFF_entry.grid(row=1, column=2, padx=5, pady=2)

target_hotend = tk.StringVar(value=str(TARGET_HOTEND))
TARGET_HOTEND_entry = tk.Entry(param_grid, textvariable=target_hotend, font=entry_font, width=6, relief='sunken', bd=2)
TARGET_HOTEND_entry.grid(row=1, column=3, padx=5, pady=2)

# 预测分类结果
tk.Label(param_frame, text="预测分类结果(0,1,2)", bg='#f0f0f0', font=('Arial',10), fg='#333333').pack(pady=2)

# 参数状态标签
status_grid = tk.Frame(param_frame, bg='#f0f0f0')
status_grid.pack(fill='x', padx=5)

FLOW_RATE_label = tk.Label(status_grid, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
FLOW_RATE_label.grid(row=0, column=0, padx=15)
FEED_RATE_label = tk.Label(status_grid, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
FEED_RATE_label.grid(row=0, column=1, padx=15)
Z_OFF_label = tk.Label(status_grid, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
Z_OFF_label.grid(row=0, column=2, padx=15)
HOTEND_label = tk.Label(status_grid, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
HOTEND_label.grid(row=0, column=3, padx=15)

# 参数操作按钮
button_frame = tk.Frame(param_frame, bg='#f0f0f0')
button_frame.pack(fill='x', pady=5)

tk.Button(button_frame, text="修改参数", command=change_param_by_button, 
          font=button_font, bg='#2196F3', fg='white', width=10).pack(side='left', padx=3)
tk.Button(button_frame, text="参数回正", command=param_init, 
          font=button_font, bg='#4CAF50', fg='white', width=10).pack(side='left', padx=3)
tk.Button(button_frame, text="参数随机", command=param_ramdon, 
          font=button_font, bg='#FF9800', fg='white', width=10).pack(side='left', padx=3)

#---记录控制
record_frame = tk.LabelFrame(right_panel, text="记录控制", bg='#f0f0f0', font=title_font)
record_frame.pack(fill='x', pady=5)

# 采集模式和实验编号（同一行）
mode_frame = tk.Frame(record_frame, bg='#f0f0f0')
mode_frame.pack(fill='x', padx=5, pady=5)

tk.Label(mode_frame, text="采集模式:", bg='#f0f0f0', font=label_font).pack(side='left', padx=5)

# 采集模式下拉菜单
acquisition_mode = tk.StringVar(value="普通采集")
acquisition_mode_combo = ttk.Combobox(mode_frame, textvariable=acquisition_mode, 
                                       values=["普通采集", "标准化采集"], 
                                       width=12, state='readonly')
acquisition_mode_combo.pack(side='left', padx=5)

# 实验编号（始终显示，但普通采集时禁用）
tk.Label(mode_frame, text="编号:", bg='#f0f0f0', font=label_font).pack(side='left', padx=(15, 5))

exp_number = tk.StringVar(value="1")
exp_number_combo = ttk.Combobox(mode_frame, textvariable=exp_number,
                                 values=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                 width=5, state='disabled')  # 初始禁用
exp_number_combo.pack(side='left', padx=5)

# 显示当前实验参数
def update_exp_info(*args):
    if acquisition_mode.get() == "标准化采集":
        exp_number_combo.config(state='readonly')  # 启用实验编号选择
        # 标准化采集自动设置状态
        try:
            if 'state_label' in globals():
                state_label.config(text=f"自动修改: True, 闭环: False, 循环: False")
        except:
            pass
        global PRINT_STATE, CLOSE_LOOP, PARAM_LOOP
        PRINT_STATE = True
        CLOSE_LOOP = False
        PARAM_LOOP = False
    else:
        exp_number_combo.config(state='disabled')  # 禁用实验编号选择

acquisition_mode.trace('w', update_exp_info)

# 按钮区域
record_btn_frame = tk.Frame(record_frame, bg='#f0f0f0')
record_btn_frame.pack(fill='x', padx=3, pady=5)

tk.Button(record_btn_frame, text="开始", command=start_recording, 
          font=button_font, bg='#2196F3', fg='white', width=6).pack(side='left', padx=2)
tk.Button(record_btn_frame, text="停止", command=stop_recording,  
          font=button_font, bg='#f44336', fg='white', width=6).pack(side='left', padx=2)
tk.Button(record_btn_frame, text="继续", command=continue_recording, 
          font=button_font, bg='#FF9800', fg='white', width=6).pack(side='left', padx=2)
tk.Button(record_btn_frame, text="结束", command=complete_experiment, 
          font=button_font, bg='#607D8B', fg='white', width=6).pack(side='left', padx=2)

# 测试保存按钮（新的一行）
test_save_btn_frame = tk.Frame(record_frame, bg='#f0f0f0')
test_save_btn_frame.pack(fill='x', padx=3, pady=5)

tk.Button(test_save_btn_frame, text="测试保存", command=test_save_single_frame, 
          font=button_font, bg='#9C27B0', fg='white', width=10).pack(pady=3)

#---状态信息
status_frame = tk.LabelFrame(right_panel, text="状态信息", bg='#f0f0f0', font=title_font)
status_frame.pack(fill='x', pady=5)

close_loop_label = tk.Label(status_frame, text="闭环调控情况:", bg='#f0f0f0', font=label_font)
close_loop_label.pack(anchor='w', padx=5, pady=5)

#---OctoPrint 服务控制
octoprint_frame = tk.LabelFrame(right_panel, text="OctoPrint 服务", bg='#f0f0f0', font=title_font)
octoprint_frame.pack(fill='x', pady=5)

octoprint_status_label = tk.Label(octoprint_frame, text="状态: 检测中...", bg='#f0f0f0', font=label_font, fg='#666666')
octoprint_status_label.pack(anchor='w', padx=5, pady=3)

octoprint_btn_frame = tk.Frame(octoprint_frame, bg='#f0f0f0')
octoprint_btn_frame.pack(fill='x', padx=5, pady=3)

def toggle_octoprint_service():
    """启动/停止OctoPrint服务"""
    if is_octoprint_running():
        # 停止服务
        stop_octoprint_service()
        update_octoprint_ui()
    else:
        # 启动服务
        threading.Thread(target=lambda: [
            start_octoprint_service(),
            update_octoprint_ui()
        ], daemon=True).start()

def update_octoprint_ui():
    """更新OctoPrint状态UI"""
    if is_octoprint_running():
        octoprint_status_label.config(text="状态: 运行中", fg='#4CAF50')
        octoprint_toggle_btn.config(text="停止服务", bg='#f44336')
    else:
        octoprint_status_label.config(text="状态: 未运行", fg='#666666')
        octoprint_toggle_btn.config(text="启动服务", bg='#4CAF50')

octoprint_toggle_btn = tk.Button(octoprint_btn_frame, text="启动服务", command=toggle_octoprint_service,
                                 font=button_font, bg='#4CAF50', fg='white', width=12)
octoprint_toggle_btn.pack(side='left', padx=3)

#---打印机状态显示（替换原来的坐标显示）
printer_status_frame = tk.LabelFrame(right_panel, text="打印机状态", bg='#f0f0f0', font=title_font)
printer_status_frame.pack(fill='x', pady=5)

# 使用网格布局显示状态信息
status_grid = tk.Frame(printer_status_frame, bg='#f0f0f0')
status_grid.pack(fill='x', padx=5, pady=5)

# XYZ坐标（第一行）
tk.Label(status_grid, text="X:", bg='#f0f0f0', font=label_font).grid(row=0, column=0, sticky='w')
x_label = tk.Label(status_grid, text="0.00", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=8)
x_label.grid(row=0, column=1, sticky='w')

tk.Label(status_grid, text="Y:", bg='#f0f0f0', font=label_font).grid(row=0, column=2, sticky='w')
y_label = tk.Label(status_grid, text="0.00", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=8)
y_label.grid(row=0, column=3, sticky='w')

tk.Label(status_grid, text="Z:", bg='#f0f0f0', font=label_font).grid(row=0, column=4, sticky='w')
z_label = tk.Label(status_grid, text="0.00", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=8)
z_label.grid(row=0, column=5, sticky='w')

# 温度信息（第二行）
tk.Label(status_grid, text="尖端目标:", bg='#f0f0f0', font=label_font).grid(row=1, column=0, sticky='w', columnspan=2)
hotend_target_label = tk.Label(status_grid, text="200°C", bg='#f0f0f0', font=label_font, fg='#FF5722', width=8)
hotend_target_label.grid(row=1, column=2, sticky='w')

tk.Label(status_grid, text="实际:", bg='#f0f0f0', font=label_font).grid(row=1, column=3, sticky='w')
hotend_actual_label = tk.Label(status_grid, text="--°C", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=8)
hotend_actual_label.grid(row=1, column=4, sticky='w')

# 热床温度（第三行）
tk.Label(status_grid, text="热床目标:", bg='#f0f0f0', font=label_font).grid(row=2, column=0, sticky='w', columnspan=2)
bed_target_label = tk.Label(status_grid, text="--°C", bg='#f0f0f0', font=label_font, fg='#2196F3', width=8)
bed_target_label.grid(row=2, column=2, sticky='w')

tk.Label(status_grid, text="实际:", bg='#f0f0f0', font=label_font).grid(row=2, column=3, sticky='w')
bed_actual_label = tk.Label(status_grid, text="--°C", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=8)
bed_actual_label.grid(row=2, column=4, sticky='w')

# 打印参数（第四行）
tk.Label(status_grid, text="流量:", bg='#f0f0f0', font=label_font).grid(row=3, column=0, sticky='w')
flow_status_label = tk.Label(status_grid, text="100%", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=6)
flow_status_label.grid(row=3, column=1, sticky='w')

tk.Label(status_grid, text="速度:", bg='#f0f0f0', font=label_font).grid(row=3, column=2, sticky='w')
speed_status_label = tk.Label(status_grid, text="100%", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=6)
speed_status_label.grid(row=3, column=3, sticky='w')

# Z偏移和Z轴最终状态（第五行）
tk.Label(status_grid, text="Z偏移:", bg='#f0f0f0', font=label_font).grid(row=4, column=0, sticky='w')
z_offset_status_label = tk.Label(status_grid, text="0.00", bg='#f0f0f0', font=label_font, fg='#2c3e50', width=6)
z_offset_status_label.grid(row=4, column=1, sticky='w')

tk.Label(status_grid, text="Z最终:", bg='#f0f0f0', font=label_font).grid(row=4, column=2, sticky='w')
z_final_status_label = tk.Label(status_grid, text="-2.55", bg='#f0f0f0', font=label_font, fg='#9C27B0', width=8)
z_final_status_label.grid(row=4, column=3, sticky='w', columnspan=2)

# 打印机状态（第六行）
tk.Label(status_grid, text="状态:", bg='#f0f0f0', font=label_font).grid(row=5, column=0, sticky='w')
printer_state_label = tk.Label(status_grid, text="准备中", bg='#f0f0f0', font=label_font, fg='#666666', width=12)
printer_state_label.grid(row=5, column=1, sticky='w', columnspan=4)

# 保留原来的coordinates_label用于兼容（隐藏）
coordinates_label = tk.Label(
    printer_status_frame,
    text=f"X: {current_x:.2f}  Y: {current_y:.2f}  Z: {current_z:.2f}",
    font=label_font,
    bg='#f0f0f0',
    fg='#2c3e50'
)
# 配置行列权重
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=3)  # 相机区域占更多空间
root.grid_columnconfigure(1, weight=1)  # 控制面板占较少空间
camera_frame.grid_columnconfigure(0, weight=1)  # 旁轴相机列
camera_frame.grid_columnconfigure(1, weight=1)  # IDS相机列
camera_frame.grid_rowconfigure(0, weight=1)     # 第一行相机
camera_frame.grid_rowconfigure(1, weight=1)     # 第二行红外相机


if __name__ == "__main__":
    # 定义关闭窗口的回调函数
    def on_closing():
        """应用关闭时清理资源"""
        global fotric_device, computer_camera, is_recording, camera_thread, ids_camera_enabled
        
        print("\n[系统] 应用正在关闭...")
        is_recording = False  # 停止图像捕获
        time.sleep(0.2)
        
        # 等待图像捕获线程退出（最多5秒）
        if camera_thread and camera_thread.is_alive():
            try:
                camera_thread.join(timeout=5)
                if camera_thread.is_alive():
                    print("[警告] 图像捕获线程未能正常退出，强制关闭")
            except Exception as e:
                print(f"[警告] 线程退出异常: {e}")
        
        # 关闭红外相机
        if fotric_device is not None:
            try:
                fotric_device.disconnect()
                print("✓ 红外相机已关闭")
            except Exception as e:
                print(f"关闭红外相机失败: {e}")
        
        # 关闭旁轴相机
        if computer_camera is not None:
            try:
                computer_camera.release()
                print("✓ 旁轴相机已关闭")
            except Exception as e:
                print(f"关闭旁轴相机失败: {e}")
        
        # 关闭IDS相机或替代摄像头
        if ids_camera_enabled and device is not None:
            try:
                nodemap_remote.FindNode("AcquisitionStop").Execute()
                datastream.StopAcquisition()
                ids_peak.Library.Close()
                print("✓ IDS相机已关闭")
            except Exception as e:
                print(f"关闭IDS相机失败: {e}")
        else:
            # 关闭替代摄像头
            try:
                if 'ids_alternative_camera' in globals() and ids_alternative_camera is not None:
                    ids_alternative_camera.release()
                    print("✓ 随轴替代摄像头已关闭")
            except Exception as e:
                print(f"关闭替代摄像头失败: {e}")
        
        # 停止OctoPrint服务（如果是本程序启动的）
        stop_octoprint_service()
        
        print("[系统] 资源清理完毕，应用退出")
        
        stop_websocket()
        root.destroy()
    
    # 设置窗口关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)
    start_websocket()
    
    # 启动M114坐标获取定时器 (每500毫秒执行一次，与图像采集同频)
    def schedule_m114_update():
        """定时更新M114坐标"""
        update_m114_coordinates()
        root.after(500, schedule_m114_update)  # 0.5秒一次，与2Hz图像采集同步
    
    # 启动打印机状态显示更新定时器 (每2000毫秒执行一次)
    def schedule_printer_status_update():
        """定时更新打印机状态显示"""
        update_printer_status_display()
        root.after(2000, schedule_printer_status_update)
    
    # ========== OctoPrint 服务自动检测与启动 ==========
    def auto_start_octoprint():
        """自动检测并启动OctoPrint服务"""
        print("\n[OctoPrint] 检测服务状态...")
        
        if is_octoprint_running():
            print("[OctoPrint] 服务已在运行")
            update_octoprint_ui()
        else:
            print("[OctoPrint] 服务未运行，尝试自动启动...")
            print("[提示] 如需手动启动，请使用左侧'OctoPrint 服务'面板的按钮")
            
            # 在新线程中启动，避免阻塞UI
            def start_and_update():
                success = start_octoprint_service()
                if success:
                    print("[OctoPrint] 自动启动成功")
                else:
                    print("[OctoPrint] 自动启动失败，请检查:")
                    print("  1. 是否已安装OctoPrint: pip install octoprint")
                    print("  2. 是否在正确的conda/virtual环境中")
                    print("  3. 或手动启动: octoprint serve")
                # 在主线程中更新UI
                root.after(0, update_octoprint_ui)
            
            threading.Thread(target=start_and_update, daemon=True).start()
    
    # 程序启动后3秒自动检测/启动OctoPrint（给UI一点时间加载）
    root.after(3000, auto_start_octoprint)
    
    # 定时更新OctoPrint状态（每5秒）
    def schedule_octoprint_status_update():
        update_octoprint_ui()
        root.after(5000, schedule_octoprint_status_update)
    
    root.after(5000, schedule_octoprint_status_update)
    # =============================================
    
    # 启动定时器
    root.after(500, schedule_m114_update)
    root.after(1000, schedule_printer_status_update)
    
    update_frame()
    # 开始主循环
    root.mainloop()


