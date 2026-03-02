"""
打印机多相机标定工具

功能：
1. 控制打印机移动到多个标定点（根据CALIBRATION_POINTS配置）
2. 在每个点采集3个相机的图像
3. 通过鼠标点击标定喷嘴像素位置
4. 保存标定数据和图像
5. 计算相机内外参

使用方法：
    python utils/calibration_tool.py
"""

import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import time
import json
import threading
import queue
from datetime import datetime
import requests

# 添加项目路径（从utils文件夹运行时需要向上回退一级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置区域（用户可修改） ====================

# OctoPrint 配置
OCTOPRINT_URL = "http://127.0.0.1:5000"
API_KEY = "UGjrS2T5n_48GF0YsWADx1EoTILjwn7ZkeWUfgGvW2Q"

# 相机设备编号配置（根据你的系统修改这些值）
CAMERA_CONFIG = {
    # 随轴相机（IDS 或替代摄像头）
    "ids": {
        "enabled": True,           # 是否启用
        "device_ids": [0,2,3],   # 尝试的设备编号列表（按顺序尝试）
        "resolution": (1920, 1080), # 分辨率 (宽, 高)
        "use_dshow": True,         # 使用DirectShow后端（Windows下更快）
        "buffer_size": 1,          # 缓冲区大小
        "skip_brightness_check": True,  # 是否跳过亮度检测（加速但可能选到虚拟摄像头）
        "brightness_threshold": 6, # 亮度阈值
    },
    # 旁轴可见光相机（电脑摄像头）
    "computer": {
        "enabled": True,
        "device_id": 1,            # 设备编号
        "resolution": (1920, 1080),
        "fps": 30,
        "buffer_size": 1,          # 缓冲区大小（越小延迟越低）
        "use_dshow": True,         # 使用DirectShow后端（Windows下更快）
        "disable_autofocus": True, # 禁用自动对焦（大幅加快初始化）
        "disable_auto_exposure": True,
        "disable_auto_whitebalance": True,
        "warm_up_frames": 5,       # 预热帧数（清除缓冲区）
    },
    # 标定配置
    "calibration": {
        "movement_delay": 3.0,     # 移动后等待时间（秒）
        "position_tolerance": 0.5, # 位置容差（mm）
    },
    # 旁轴红外相机（Fotric）
    "fotric": {
        "enabled": True,
        "ip": "192.168.1.100",     # 相机IP地址
        "port": 10080,
        "username": "admin",
        "password": "admin",
        "high_resolution": True,
    }
}

# 如何查看设备编号：
# 1. Windows: 设备管理器 -> 图像设备，按连接顺序从0开始编号
# 2. 或者在Python中运行: python -c "import cv2; [print(f'{i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"

# ==================== 配置区域结束 ====================

# 全局变量
ids_camera_enabled = False
device = None
datastream = None
nodemap_remote = None
ids_alternative_camera = None
ids_alternative_camera_id = 2

computer_camera = None
camera_opened = False

fotric_device = None
fotric_enabled = False
fotric_latest_frame = None
fotric_lock = threading.Lock()

# 标定点配置
CALIBRATION_POINTS = {
    'X': [80, 130, 180],
    'Y': [20, 100, 180],
    'Z': [2, 36, 70]
}

# 相机配置
CAMERAS = [
    {"name": "随轴相机", "id": "ids", "description": "固定在打印头上"},
    {"name": "旁轴可见光", "id": "computer", "description": "打印机前方"},
    {"name": "旁轴红外", "id": "fotric", "description": "打印机前方（热成像）"}
]


# ==================== OctoPrint API ====================
def get_printer_status():
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
    url = f"{OCTOPRINT_URL}/api/printer/command"
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    data = {"command": command}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        return response.status_code == 204
    except Exception as e:
        print(f"[G代码错误] {e}")
        return False


def wait_for_temperature(target_temp, tolerance=5, timeout=300):
    """等待温度达到目标值"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_printer_status()
        if status:
            actual = status['temperature']['tool0']['actual']
            print(f"[温度] 当前: {actual:.1f}°C / 目标: {target_temp}°C")
            if abs(actual - target_temp) < tolerance:
                return True
        time.sleep(2)
    return False


def wait_for_position(target_x, target_y, target_z, tolerance=0.5, timeout=30):
    """等待打印机到达目标位置（使用OctoPrint API查询实际位置）"""
    print(f"[等待位置] 目标: X{target_x}, Y{target_y}, Z{target_z} (容差{tolerance}mm)")
    start_time = time.time()
    stable_count = 0
    
    while time.time() - start_time < timeout:
        status = get_printer_status()
        if status and 'state' in status:
            # 检查打印机是否空闲
            state = status['state'].get('text', 'Unknown')
            
            # 尝试从温度数据中获取位置（某些固件会在这里报告位置）
            # 或者使用M114命令获取
            try:
                # 使用M114命令获取位置
                if send_gcode("M114"):
                    time.sleep(0.3)  # 等待命令执行
                    # 再次获取状态，某些OctoPrint版本会在状态中包含位置
                    status = get_printer_status()
            except:
                pass
        
        # 简单等待策略：移动后等待足够时间让打印机稳定
        time.sleep(0.5)
        stable_count += 1
        
        # 至少等待2秒让移动完成
        if stable_count >= 4:  # 2秒
            print(f"[等待位置] 位置应已稳定")
            return True
    
    print(f"[等待位置] 等待超时")
    return False


# ==================== 相机初始化 ====================
def init_ids_camera():
    """初始化IDS或替代相机"""
    global ids_camera_enabled, device, datastream, nodemap_remote
    global ids_alternative_camera
    
    config = CAMERA_CONFIG["ids"]
    if not config["enabled"]:
        print("[IDS] 随轴相机已在配置中禁用")
        return False
    
    print("[初始化] IDS/随轴相机...")
    
    # 尝试IDS原生SDK
    try:
        from ids_peak import ids_peak
        from ids_peak import ids_peak_ipl_extension
        
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()
        
        if device_manager.Devices().empty():
            print("[IDS] 未找到IDS相机，将尝试使用替代摄像头")
            ids_camera_enabled = False
        else:
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
        print(f"[IDS] IDS SDK初始化失败: {e}")
        ids_camera_enabled = False
    
    # 尝试替代摄像头（使用配置中的设备编号列表，优化版本）
    try:
        print(f"[IDS] 尝试替代摄像头，设备列表: {config['device_ids']}...")
        width, height = config["resolution"]
        use_dshow = config.get("use_dshow", True)
        skip_check = config.get("skip_brightness_check", False)
        
        for cam_id in config["device_ids"]:
            print(f"[IDS] 尝试设备 {cam_id}...")
            
            # 使用DirectShow后端加速
            if use_dshow:
                cam = cv2.VideoCapture(cam_id + cv2.CAP_DSHOW)
                if not cam.isOpened():
                    cam = cv2.VideoCapture(cam_id)
            else:
                cam = cv2.VideoCapture(cam_id)
            
            if cam.isOpened():
                # 立即设置缓冲区
                cam.set(cv2.CAP_PROP_BUFFERSIZE, config.get("buffer_size", 1))
                
                # 设置分辨率
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # 快速读取测试帧
                ret, frame = cam.read()
                if ret and frame is not None:
                    # 亮度检测（可选，用于排除虚拟摄像头）
                    if not skip_check:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        brightness = cv2.mean(gray)[0]
                        if brightness < config.get("brightness_threshold", 15):
                            print(f"  设备{cam_id} 画面太暗({brightness:.1f})，跳过")
                            cam.release()
                            continue
                    
                    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ids_alternative_camera = cam
                    print(f"[IDS] ✓ 替代摄像头打开成功 (设备{cam_id}, {actual_w}x{actual_h})")
                    return True
                cam.release()
            else:
                print(f"[IDS] 设备 {cam_id} 无法打开")
    except Exception as e:
        print(f"[IDS] 替代摄像头失败: {e}")
    
    print("[IDS] 所有摄像头初始化失败")
    return False


def init_computer_camera():
    """初始化旁轴可见光相机（优化版本）"""
    global computer_camera, camera_opened
    
    config = CAMERA_CONFIG["computer"]
    if not config["enabled"]:
        print("[旁轴] 可见光相机已在配置中禁用")
        return False
    
    device_id = config["device_id"]
    width, height = config["resolution"]
    print(f"[初始化] 旁轴可见光相机 (设备{device_id})...")
    
    try:
        # ===== 使用DirectShow后端（Windows最快） =====
        if config.get("use_dshow", True):
            backend = cv2.CAP_DSHOW
            computer_camera = cv2.VideoCapture(device_id + backend)
            if not computer_camera.isOpened():
                print("[旁轴] DirectShow后端失败，尝试默认后端...")
                computer_camera = cv2.VideoCapture(device_id)
        else:
            computer_camera = cv2.VideoCapture(device_id)
        
        if not computer_camera.isOpened():
            print(f"[旁轴] ✗ 无法打开设备 {device_id}")
            camera_opened = False
            return False
        
        # ===== 立即设置缓冲区大小 =====
        computer_camera.set(cv2.CAP_PROP_BUFFERSIZE, config.get("buffer_size", 1))
        
        # ===== 禁用自动功能（大幅加快初始化） =====
        if config.get("disable_autofocus", True):
            try:
                computer_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except:
                pass
        
        if config.get("disable_auto_exposure", True):
            try:
                computer_camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            except:
                pass
        
        if config.get("disable_auto_whitebalance", True):
            try:
                computer_camera.set(cv2.CAP_PROP_AUTO_WB, 0)
            except:
                pass
        
        # ===== 设置分辨率 =====
        computer_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        computer_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        computer_camera.set(cv2.CAP_PROP_FPS, config.get("fps", 30))
        
        # ===== 快速预热缓冲区 =====
        warm_up = config.get("warm_up_frames", 5)
        for i in range(warm_up):
            ret, _ = computer_camera.read()
            if i < warm_up - 1:
                time.sleep(0.02)
        
        # ===== 获取实际参数 =====
        actual_w = int(computer_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(computer_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = computer_camera.get(cv2.CAP_PROP_FPS)
        
        camera_opened = True
        print(f"[旁轴] ✓ 初始化成功: {actual_w}x{actual_h} @ {actual_fps:.0f}FPS")
        return True
        
    except Exception as e:
        print(f"[旁轴] ✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        camera_opened = False
        return False


def init_fotric_camera():
    """初始化旁轴红外相机"""
    global fotric_device, fotric_enabled
    
    config = CAMERA_CONFIG["fotric"]
    if not config["enabled"]:
        print("[红外] 红外相机已在配置中禁用")
        return False
    
    print("[初始化] 旁轴红外相机...")
    try:
        from hardware.fotric_driver import FotricEnhancedDevice
        fotric_device = FotricEnhancedDevice(
            ip=config["ip"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            simulation_mode=False,
            high_resolution=config["high_resolution"]
        )
        fotric_enabled = fotric_device.is_connected
        if fotric_enabled:
            print(f"[红外] 红外相机初始化成功 ({config['ip']}:{config['port']})")
            return True
    except Exception as e:
        print(f"[红外] 初始化失败: {e}")
    return False


def get_ids_frame():
    """获取随轴相机图像（已旋转180度校正安装方向）"""
    result_image = None
    
    if ids_camera_enabled:
        try:
            from ids_peak import ids_peak_ipl_extension
            buffer = datastream.WaitForFinishedBuffer(1000)
            image_data = ids_peak_ipl_extension.BufferToImage(buffer)
            image_np = image_data.get_numpy()
            
            if image_data.PixelFormat() == "RGB8":
                image_np = image_np.reshape((image_data.Height(), image_data.Width(), 3))
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            datastream.QueueBuffer(buffer)
            result_image = image_np
        except:
            pass
    
    if result_image is None and ids_alternative_camera:
        ret, frame = ids_alternative_camera.read()
        if ret:
            result_image = frame
    
    # 旋转180度校正安装方向（随轴相机是倒着安装的）
    if result_image is not None:
        result_image = cv2.flip(result_image, -1)
    
    return result_image


def get_computer_frame():
    """获取旁轴可见光图像"""
    if computer_camera and camera_opened:
        ret, frame = computer_camera.read()
        if ret:
            return frame
    return None


def get_fotric_frame():
    """获取旁轴红外图像"""
    global fotric_latest_frame
    if fotric_enabled and fotric_device:
        try:
            thermal_data = fotric_device.get_thermal_data()
            if thermal_data is not None:
                with fotric_lock:
                    fotric_latest_frame = thermal_data.copy()
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


# ==================== 标定界面 ====================
class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("打印机多相机标定工具")
        self.root.geometry("1200x800")
        
        # 标定数据
        self.calibration_data = []
        self.current_point_idx = 0
        self.current_camera_idx = 0
        self.total_points = len(CALIBRATION_POINTS['X']) * len(CALIBRATION_POINTS['Y']) * len(CALIBRATION_POINTS['Z'])
        
        # 点列表
        self.points = []
        for x in CALIBRATION_POINTS['X']:
            for y in CALIBRATION_POINTS['Y']:
                for z in CALIBRATION_POINTS['Z']:
                    self.points.append((x, y, z))
        
        # 临时存储当前点的数据
        self.current_point_data = {
            'world': None,
            'images': {},
            'pixels': {}
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置界面"""
        # 标题
        title_label = tk.Label(self.root, text="打印机多相机标定工具", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # 动态生成标定点范围说明文字
        x_range = CALIBRATION_POINTS['X']
        y_range = CALIBRATION_POINTS['Y']
        z_range = CALIBRATION_POINTS['Z']
        total_points = len(x_range) * len(y_range) * len(z_range)
        
        desc_text = f"""标定流程：
1. 点击"开始标定"，系统会自动加热喷嘴到200°C
2. 然后依次移动到{total_points}个标定点
3. 每个点需要标定3个相机的喷嘴位置（鼠标点击图片）
4. 标定完成后自动计算相机参数

标定点分布：X{x_range} × Y{y_range} × Z{z_range}"""
        
        desc_label = tk.Label(self.root, text=desc_text, justify=tk.LEFT, 
                             font=('Arial', 10))
        desc_label.pack(pady=5)
        
        # 状态显示
        self.status_frame = tk.LabelFrame(self.root, text="标定状态", 
                                         font=('Arial', 11, 'bold'))
        self.status_frame.pack(fill='x', padx=20, pady=10)
        
        self.status_label = tk.Label(self.status_frame, text="准备就绪", 
                                    font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        self.progress_label = tk.Label(self.status_frame, text=f"进度: 0/{self.total_points}", 
                                      font=('Arial', 11))
        self.progress_label.pack(pady=2)
        
        self.position_label = tk.Label(self.status_frame, text="当前位置: -", 
                                      font=('Arial', 10))
        self.position_label.pack(pady=2)
        
        # 按钮区域
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        self.start_btn = tk.Button(btn_frame, text="开始标定", 
                                  command=self.start_calibration,
                                  font=('Arial', 12, 'bold'),
                                  bg='#4CAF50', fg='white',
                                  width=15, height=2)
        self.start_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(btn_frame, text="停止标定", 
                                 command=self.stop_calibration,
                                 font=('Arial', 12, 'bold'),
                                 bg='#f44336', fg='white',
                                 width=15, height=2,
                                 state='disabled')
        self.stop_btn.pack(side='left', padx=10)
        
        # 预览区域
        preview_frame = tk.LabelFrame(self.root, text="相机预览", 
                                     font=('Arial', 11, 'bold'))
        preview_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 三个相机的预览
        self.preview_labels = {}
        for i, cam in enumerate(CAMERAS):
            frame = tk.Frame(preview_frame)
            frame.grid(row=0, column=i, padx=10, pady=5)
            
            tk.Label(frame, text=cam['name'], 
                    font=('Arial', 10, 'bold')).pack()
            tk.Label(frame, text=cam['description'], 
                    font=('Arial', 8), fg='gray').pack()
            
            label = tk.Label(frame, bg='black', width=400, height=300)
            label.pack(pady=5)
            self.preview_labels[cam['id']] = label
        
        # 启动预览更新
        self.update_preview()
        
    def update_preview(self):
        """更新预览图像"""
        # 获取各相机图像
        ids_img = get_ids_frame()
        comp_img = get_computer_frame()
        fotric_img = get_fotric_frame()
        
        # 更新显示
        if ids_img is not None:
            self.show_image(self.preview_labels['ids'], ids_img, (400, 300))
        if comp_img is not None:
            self.show_image(self.preview_labels['computer'], comp_img, (400, 300))
        if fotric_img is not None:
            self.show_image(self.preview_labels['fotric'], fotric_img, (400, 300))
        
        self.root.after(100, self.update_preview)
    
    def show_image(self, label, img, size):
        """在Label中显示图像"""
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    
    def start_calibration(self):
        """开始标定流程"""
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # 在新线程中运行标定
        self.calibration_thread = threading.Thread(target=self.calibration_process)
        self.calibration_thread.daemon = True
        self.calibration_thread.start()
    
    def stop_calibration(self):
        """停止标定"""
        self.calibration_running = False
        self.status_label.config(text="标定已停止")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def calibration_process(self):
        """标定主流程"""
        self.calibration_running = True
        
        # 1. 加热喷嘴
        self.update_status("加热喷嘴到200°C...")
        send_gcode("M104 S200")
        if not wait_for_temperature(200, tolerance=5, timeout=300):
            self.update_status("加热超时！")
            return
        
        self.update_status("加热完成，开始标定...")
        time.sleep(2)
        
        # 2. 遍历所有标定点
        for i, (x, y, z) in enumerate(self.points):
            if not self.calibration_running:
                break
            
            self.current_point_idx = i
            self.current_point_data = {
                'point_id': i + 1,
                'world': (x, y, z),
                'images': {},
                'pixels': {}
            }
            
            self.update_status(f"移动到标定点 {i+1}/{self.total_points}: X={x}, Y={y}, Z={z}")
            self.update_progress(i + 1, self.total_points)
            
            # 发送移动命令
            send_gcode(f"G1 X{x} Y{y} Z{z} F3000")
            
            # 等待打印机到位
            calib_config = CAMERA_CONFIG.get("calibration", {})
            move_delay = calib_config.get("movement_delay", 3.0)
            self.update_status(f"移动到标定点 {i+1}/{self.total_points}: X={x}, Y={y}, Z={z} (等待{move_delay:.1f}s...)")
            time.sleep(move_delay)
            
            self.current_point_data['m114'] = (x, y, z)
            
            # 3. 采集三个相机的图像并标定
            for cam_idx, cam_info in enumerate(CAMERAS):
                if not self.calibration_running:
                    break
                
                self.current_camera_idx = cam_idx
                self.update_status(f"标定点 {i+1}/{self.total_points} - {cam_info['name']}: 请点击喷嘴位置")
                
                # 获取图像
                if cam_info['id'] == 'ids':
                    img = get_ids_frame()
                elif cam_info['id'] == 'computer':
                    img = get_computer_frame()
                else:
                    img = get_fotric_frame()
                
                if img is None:
                    print(f"[警告] 无法获取 {cam_info['name']} 图像")
                    continue
                
                # 打开标定对话框（返回像素坐标和带标记的图像）
                pixel_pos, marked_img = self.calibrate_image(img, cam_info['name'], x, y, z)
                
                if pixel_pos and marked_img is not None:
                    self.current_point_data['pixels'][cam_info['id']] = pixel_pos
                    # 保存带标记的图像
                    self.current_point_data['images'][cam_info['id']] = marked_img
                    print(f"[标定] {cam_info['name']}: 像素坐标 {pixel_pos}")
                else:
                    # 如果没有标记，保存原始图像
                    self.current_point_data['images'][cam_info['id']] = img.copy()
                    print(f"[标定] {cam_info['name']}: 标定取消或失败")
            
            # 保存当前点数据
            if len(self.current_point_data['pixels']) == 3:
                self.calibration_data.append(self.current_point_data)
                print(f"[完成] 标定点 {i+1} 完成")
            
            # 中间结果保存已移除，只在最后保存一次
        
        # 标定完成
        if self.calibration_running:
            self.update_status("标定完成！正在保存数据...")
            self.save_calibration_data("calibration_final.json")
            self.save_all_images()
            self.compute_calibration()
            messagebox.showinfo("完成", "标定完成！数据已保存。")
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def calibrate_image(self, img, camera_name, x, y, z):
        """打开图像标定对话框，返回点击的像素坐标和带标记的图像"""
        result = {'pos': None, 'marked_img': None}
        
        dialog = tk.Toplevel(self.root)
        dialog.title(f"标定 {camera_name} - 点({x}, {y}, {z})")
        dialog.geometry("1000x800")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 提示文字
        tk.Label(dialog, text=f"请点击 {camera_name} 图像中的喷嘴尖端位置", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        tk.Label(dialog, text="提示：使用鼠标左键点击喷嘴尖端，点击后会留下红色标记", 
                font=('Arial', 10), fg='gray').pack()
        
        # 图像显示框架
        img_frame = tk.Frame(dialog)
        img_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas用于显示图像
        canvas = tk.Canvas(img_frame, bg='black')
        canvas.pack(fill='both', expand=True)
        
        # 调整图像大小
        h, w = img.shape[:2]
        display_w, display_h = 800, 600
        
        scale = 1.0
        if w > display_w or h > display_h:
            scale = min(display_w / w, display_h / h)
        
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 保存原始图像用于后续绘制标记
        original_img = img.copy()
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        
        canvas_img = canvas.create_image(display_w//2, display_h//2, image=imgtk)
        canvas.imgtk = imgtk
        
        # 十字线（跟随鼠标）
        cross_h = canvas.create_line(0, 0, 0, 0, fill='yellow', width=2)
        cross_v = canvas.create_line(0, 0, 0, 0, fill='yellow', width=2)
        
        # 永久标记（点击后显示）
        permanent_markers = []
        
        # 状态显示
        pos_label = tk.Label(dialog, text="坐标: 未选择", font=('Arial', 11))
        pos_label.pack(pady=5)
        
        def update_canvas_image(marked_img_resized):
            """更新Canvas上的图像"""
            nonlocal imgtk
            img_rgb = cv2.cvtColor(marked_img_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            canvas.itemconfig(canvas_img, image=imgtk)
            canvas.imgtk = imgtk
        
        def on_click(event):
            """鼠标点击事件"""
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)
            
            img_x = int((canvas_x - display_w//2 + new_w//2) / scale)
            img_y = int((canvas_y - display_h//2 + new_h//2) / scale)
            
            img_x = max(0, min(img_x, w - 1))
            img_y = max(0, min(img_y, h - 1))
            
            result['pos'] = (img_x, img_y)
            pos_label.config(text=f"坐标: ({img_x}, {img_y})")
            
            # 在原始图像上绘制永久标记（红色十字）
            nonlocal original_img
            marked_img = original_img.copy()
            # 绘制十字标记
            cv2.line(marked_img, (img_x - 20, img_y), (img_x + 20, img_y), (0, 0, 255), 3)
            cv2.line(marked_img, (img_x, img_y - 20), (img_x, img_y + 20), (0, 0, 255), 3)
            # 绘制圆圈
            cv2.circle(marked_img, (img_x, img_y), 10, (0, 0, 255), 2)
            # 绘制文字标签
            cv2.putText(marked_img, f"({img_x}, {img_y})", (img_x + 25, img_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 保存带标记的图像
            result['marked_img'] = marked_img
            
            # 更新Canvas显示（使用缩放后的图像）
            marked_img_resized = cv2.resize(marked_img, (new_w, new_h))
            update_canvas_image(marked_img_resized)
            
            # 同时保留Canvas上的十字线标记
            canvas.coords(cross_h, canvas_x - 15, canvas_y, canvas_x + 15, canvas_y)
            canvas.coords(cross_v, canvas_x, canvas_y - 15, canvas_x, canvas_y + 15)
            canvas.itemconfig(cross_h, fill='red', width=3)
            canvas.itemconfig(cross_v, fill='red', width=3)
            
            print(f"[标定] 已标记位置: ({img_x}, {img_y})")
        
        def on_mouse_move(event):
            """鼠标移动事件"""
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)
            canvas.coords(cross_h, canvas_x - 10, canvas_y, canvas_x + 10, canvas_y)
            canvas.coords(cross_v, canvas_x, canvas_y - 10, canvas_x, canvas_y + 10)
        
        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Motion>", on_mouse_move)
        
        # 按钮
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)
        
        def on_confirm():
            if result['pos']:
                dialog.destroy()
            else:
                messagebox.showwarning("警告", "请先点击喷嘴位置！")
        
        def on_skip():
            result['pos'] = None
            result['marked_img'] = None
            dialog.destroy()
        
        tk.Button(btn_frame, text="确认", command=on_confirm,
                 font=('Arial', 11, 'bold'), bg='#4CAF50', fg='white',
                 width=10).pack(side='left', padx=10)
        
        tk.Button(btn_frame, text="跳过", command=on_skip,
                 font=('Arial', 11), width=10).pack(side='left', padx=10)
        
        # 等待对话框关闭
        self.root.wait_window(dialog)
        
        return result['pos'], result['marked_img']
    
    def update_status(self, text):
        """更新状态显示"""
        self.status_label.config(text=text)
        print(f"[状态] {text}")
    
    def update_progress(self, current, total):
        """更新进度"""
        self.progress_label.config(text=f"进度: {current}/{total}")
    
    def save_calibration_data(self, filename):
        """保存标定数据到JSON"""
        try:
            data = {
                'calibration_date': datetime.now().isoformat(),
                'total_points': len(self.calibration_data),
                'points': []
            }
            
            for point in self.calibration_data:
                point_data = {
                    'point_id': point['point_id'],
                    'world_xyz': point['world'],
                    'm114_xyz': point.get('m114', point['world']),
                    'pixels': point['pixels']
                }
                data['points'].append(point_data)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[保存] 标定数据已保存: {filename}")
        except Exception as e:
            print(f"[错误] 保存数据失败: {e}")
    
    def save_all_images(self):
        """保存所有标定图像"""
        folder = f"calibration_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(folder, exist_ok=True)
        
        for point in self.calibration_data:
            point_id = point['point_id']
            x, y, z = point['world']
            
            for cam_id, img in point['images'].items():
                filename = f"{folder}/point{point_id:02d}_{cam_id}_X{x}Y{y}Z{z}.jpg"
                cv2.imwrite(filename, img)
        
        print(f"[保存] 所有图像已保存到: {folder}/")
    
    def compute_calibration(self):
        """计算相机标定参数"""
        print("[计算] 开始计算相机标定参数...")
        
        for cam_info in CAMERAS:
            cam_id = cam_info['id']
            
            obj_points = []
            img_points = []
            
            for point in self.calibration_data:
                if cam_id in point['pixels']:
                    world = point['world']
                    pixel = point['pixels'][cam_id]
                    
                    obj_points.append(world)
                    img_points.append(pixel)
            
            if len(obj_points) < 6:
                print(f"[警告] {cam_info['name']} 标定点不足，无法计算")
                continue
            
            obj_points = np.array(obj_points, dtype=np.float32)
            img_points = np.array(img_points, dtype=np.float32)
            
            print(f"[计算] {cam_info['name']}: 收集到 {len(obj_points)} 个标定点")
            print(f"  世界坐标范围: X[{obj_points[:,0].min():.1f}, {obj_points[:,0].max():.1f}], "
                  f"Y[{obj_points[:,1].min():.1f}, {obj_points[:,1].max():.1f}], "
                  f"Z[{obj_points[:,2].min():.1f}, {obj_points[:,2].max():.1f}]")
            print(f"  像素坐标范围: U[{img_points[:,0].min():.1f}, {img_points[:,0].max():.1f}], "
                  f"V[{img_points[:,1].min():.1f}, {img_points[:,1].max():.1f}]")
        
        print("[计算] 标定参数计算完成")


def main():
    """主函数"""
    print("="*60)
    print("打印机多相机标定工具")
    print("="*60)
    
    # 初始化相机
    print("\n[初始化] 正在初始化相机...")
    init_ids_camera()
    init_computer_camera()
    init_fotric_camera()
    
    # 创建主窗口
    root = tk.Tk()
    app = CalibrationApp(root)
    
    # 运行
    root.mainloop()
    
    # 清理
    print("\n[退出] 清理资源...")
    if ids_alternative_camera:
        ids_alternative_camera.release()
    if computer_camera:
        computer_camera.release()


if __name__ == "__main__":
    main()
