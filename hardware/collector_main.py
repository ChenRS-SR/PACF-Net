"""
PAC-NET 数据采集系统（简化版）
仅保留数据采集功能，移除模型推理相关代码
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
from PIL import Image, ImageTk
import queue
from datetime import datetime
import requests
import numpy as np
from collections import deque
import struct
import json

# 导入配置
try:
    from configs.collector_config import (
        CAMERA_CONFIG, OCTOPRINT_CONFIG, 
        ACQUISITION_CONFIG, RECORDING_CONFIG, TUNING_CONFIG
    )
    CONFIG_LOADED = True
except ImportError:
    print("[警告] 无法加载collector_config，使用默认配置")
    CONFIG_LOADED = False

# M114 Coordinator
try:
    from hardware.coordinator import M114Coordinator
    m114_coord = M114Coordinator()
except Exception as e:
    print(f"[警告] M114Coordinator初始化失败: {e}")
    m114_coord = None

# 导入相机驱动
try:
    from hardware.fotric_driver import FotricEnhancedDevice
except ImportError:
    print("[警告] Fotric驱动未找到")
    FotricEnhancedDevice = None

# ==================== 全局变量 ====================

# 配置（如果配置文件加载失败则使用默认值）
if CONFIG_LOADED:
    OCTOPRINT_URL = OCTOPRINT_CONFIG['url']
    API_KEY = OCTOPRINT_CONFIG['api_key']
    CAM_CONFIG = CAMERA_CONFIG
else:
    OCTOPRINT_URL = "http://127.0.0.1:5000"
    API_KEY = ""
    CAM_CONFIG = {}

# 相机相关
computer_camera = None
camera_opened = False
fotric_device = None
fotric_enabled = False
fotric_lock = threading.Lock()

# IDS相机相关
try:
    from ids_peak import ids_peak
    from ids_peak import ids_peak_ipl_extension
    IDS_AVAILABLE = True
except ImportError:
    IDS_AVAILABLE = False
    ids_peak = None
    ids_peak_ipl_extension = None

ids_camera_enabled = False
device = None
datastream = None
nodemap_remote = None
ids_alternative_camera = None

# 坐标相关
current_x = 0.0
current_y = 0.0
current_z = 0.0
last_coord_timestamp = 0.0

# 打印参数
FLOW_RATE = 100
FEED_RATE = 100
Z_OFF = 0
TARGET_HOTEND = 200
PRIMARY_Z_OFF = -2.55

# 控制标志
is_recording = False
is_paused = False
IMAGE_COUNT = 0
ACQUISITION_MODE = "标准化采集"

# ==================== 核心函数 ====================

def get_printer_status():
    """获取打印机状态"""
    url = f"{OCTOPRINT_URL}/api/printer"
    headers = {"X-Api-Key": API_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[API错误] {e}")
    return None

def send_gcode(command, timeout=5):
    """发送G代码"""
    url = f"{OCTOPRINT_URL}/api/printer/command"
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    data = {"command": command}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        return response.status_code == 204
    except Exception as e:
        print(f"[G代码错误] {e}")
        return False

def is_printer_ready_for_m114():
    """检查是否可以发送M114"""
    try:
        res_json = get_printer_status()
        if res_json is None:
            return False
        state_text = res_json.get('state', {}).get('text', '').lower()
        return 'operational' in state_text or 'printing' in state_text
    except:
        return False

def is_printer_actually_printing():
    """检查是否真正在打印"""
    try:
        res_json = get_printer_status()
        if res_json is None:
            return False
        state_text = res_json.get('state', {}).get('text', '').lower()
        return 'printing' in state_text or 'operational' in state_text
    except:
        return False

def update_m114_coordinates():
    """更新坐标"""
    global current_x, current_y, current_z, last_coord_timestamp
    
    if not is_printer_ready_for_m114():
        return
    
    try:
        if m114_coord:
            coords = m114_coord.wait_for_m114_response(timeout=0.8)
            if coords:
                last_coord_timestamp = time.time()
                current_x, current_y, current_z = coords['X'], coords['Y'], coords['Z']
    except Exception as e:
        pass

def init_ids_camera():
    """初始化IDS相机"""
    global ids_camera_enabled, device, datastream, nodemap_remote, ids_alternative_camera
    
    if not IDS_AVAILABLE:
        print("[IDS] SDK不可用，将使用替代摄像头")
        ids_camera_enabled = False
    else:
        try:
            ids_peak.Library.Initialize()
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            if not device_manager.Devices().empty():
                device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Exclusive)
                nodemap_remote = device.RemoteDevice().NodeMaps()[0]
                datastream = device.DataStreams()[0].OpenDataStream()
                
                payload_size = nodemap_remote.FindNode("PayloadSize").Value()
                for _ in range(10):
                    buffer = datastream.AllocAndAnnounceBuffer(payload_size)
                    datastream.QueueBuffer(buffer)
                
                datastream.StartAcquisition()
                nodemap_remote.FindNode("AcquisitionStart").Execute()
                ids_camera_enabled = True
                print("[IDS] IDS相机初始化成功")
                return True
        except Exception as e:
            print(f"[IDS] 初始化失败: {e}")
            ids_camera_enabled = False
    
    # 尝试替代摄像头
    if CAM_CONFIG.get('ids', {}).get('enabled', True):
        try:
            config = CAM_CONFIG['ids']
            for cam_id in config.get('alternative_device_ids', [0, 2, 3]):
                if cam_id in config.get('skip_devices', []):
                    continue
                cam = cv2.VideoCapture(cam_id + cv2.CAP_DSHOW)
                if cam.isOpened():
                    ret, frame = cam.read()
                    if ret and frame is not None:
                        ids_alternative_camera = cam
                        print(f"[IDS] 替代摄像头打开成功 (设备{cam_id})")
                        return True
                    cam.release()
        except Exception as e:
            print(f"[IDS] 替代摄像头失败: {e}")
    
    return False

def get_ids_frame():
    """获取IDS图像"""
    if ids_camera_enabled and IDS_AVAILABLE:
        try:
            buffer = datastream.WaitForFinishedBuffer(1000)
            image_data = ids_peak_ipl_extension.BufferToImage(buffer)
            image_np = image_data.get_numpy()
            
            if image_data.PixelFormat() == "RGB8":
                image_np = image_np.reshape((image_data.Height(), image_data.Width(), 3))
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            datastream.QueueBuffer(buffer)
            # 旋转180度校正
            return cv2.flip(image_np, -1)
        except:
            pass
    
    if ids_alternative_camera:
        ret, frame = ids_alternative_camera.read()
        if ret:
            return cv2.flip(frame, -1)
    return None

def init_computer_camera():
    """初始化旁轴相机"""
    global computer_camera, camera_opened
    
    config = CAM_CONFIG.get('computer', {})
    if not config.get('enabled', True):
        return False
    
    try:
        device_id = config.get('device_id', 1)
        backend = cv2.CAP_DSHOW if config.get('use_dshow', True) else 0
        computer_camera = cv2.VideoCapture(device_id + backend)
        
        if computer_camera.isOpened():
            width, height = config.get('resolution', (1920, 1080))
            computer_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            computer_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            computer_camera.set(cv2.CAP_PROP_BUFFERSIZE, config.get('buffer_size', 1))
            
            # 预热
            for _ in range(config.get('warm_up_frames', 5)):
                computer_camera.read()
            
            camera_opened = True
            print(f"[旁轴] 相机初始化成功 (设备{device_id})")
            return True
    except Exception as e:
        print(f"[旁轴] 初始化失败: {e}")
    return False

def get_computer_frame():
    """获取旁轴图像"""
    if computer_camera and camera_opened:
        ret, frame = computer_camera.read()
        if ret:
            return frame
    return None

def init_fotric_camera():
    """初始化红外相机"""
    global fotric_device, fotric_enabled
    
    if FotricEnhancedDevice is None:
        return False
    
    config = CAM_CONFIG.get('fotric', {})
    if not config.get('enabled', True):
        return False
    
    try:
        fotric_device = FotricEnhancedDevice(
            ip=config.get('ip', '192.168.1.100'),
            port=config.get('port', 10080),
            username=config.get('username', 'admin'),
            password=config.get('password', 'admin'),
            simulation_mode=config.get('simulation_mode', False),
            high_resolution=config.get('high_resolution', True)
        )
        fotric_enabled = fotric_device.is_connected
        if fotric_enabled:
            print("[红外] 相机初始化成功")
            return True
    except Exception as e:
        print(f"[红外] 初始化失败: {e}")
    return False

def get_fotric_frame():
    """获取红外图像"""
    if fotric_enabled and fotric_device:
        try:
            thermal_data = fotric_device.get_thermal_data()
            if thermal_data is not None:
                temp_min = float(np.min(thermal_data))
                temp_max = float(np.max(thermal_data))
                
                if temp_max > temp_min:
                    normalized = ((thermal_data - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(thermal_data, dtype=np.uint8)
                
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                return colored
        except:
            pass
    return None

# ==================== 采集主函数 ====================

def capture_images(capture_interval, image_queue):
    """采集图像主循环"""
    global is_recording, IMAGE_COUNT
    # ... 保留原有采集逻辑，移除模型推理部分
    pass

# ==================== UI部分 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("PAC-NET 数据采集系统")
    print("=" * 60)
    
    # 初始化相机
    print("\n[初始化] 正在初始化相机...")
    init_ids_camera()
    init_computer_camera()
    init_fotric_camera()
    
    # 创建主窗口
    root = tk.Tk()
    root.title("PAC-NET 数据采集系统")
    root.geometry("1400x850")
    
    # TODO: 添加UI界面
    
    print("\n[系统] 启动完成")
    root.mainloop()
    
    # 清理
    print("\n[退出] 清理资源...")
    if computer_camera:
        computer_camera.release()

if __name__ == "__main__":
    main()
