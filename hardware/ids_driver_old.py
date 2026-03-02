import time
import os
import cv2
import threading
import tkinter as tk
from tkinter import filedialog
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
from predict import predict,load_model
#from utils.tools import get_param_class
from config import config
import struct
from Fotric_628ch_enhanced import FotricEnhancedDevice

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

#全局参数
FLOW_RATE = 100
FEED_RATE = 100
Z_OFF = 0
CUR_Z_OFF = 0
TARGET_HOTEND = 200
PRIMARY_Z_OFF = -2.55


#其他全局变量
IMAGE_COUNT = 0
INIT_MODEL = False
model = None

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

def update_frame():
    # 获取IDS相机图像
    ids_image_np = ids_process_img()
    ids_image_np = cv2.flip(ids_image_np, -1)  # 翻转180度
    
    # 获取旁轴相机图像
    computer_image_np = get_computer_camera_frame()
    
    # 获取Fotric红外相机图像
    fotric_image_np = get_fotric_camera_frame()
    
    # 转换IDS相机图像并显示
    ids_img = Image.fromarray(ids_image_np)
    ids_img = ids_img.resize((960, 360), Image.LANCZOS)
    ids_imgtk = ImageTk.PhotoImage(image=ids_img)
    ids_video_label.imgtk = ids_imgtk
    ids_video_label.configure(image=ids_imgtk)
    
    # 转换旁轴相机图像并显示
    if computer_image_np is not None:
        computer_img = Image.fromarray(cv2.cvtColor(computer_image_np, cv2.COLOR_BGR2RGB))
        computer_img = computer_img.resize((960, 360), Image.LANCZOS)
        computer_imgtk = ImageTk.PhotoImage(image=computer_img)
        computer_video_label.imgtk = computer_imgtk
        computer_video_label.configure(image=computer_imgtk)
    
    # 转换Fotric红外相机图像并显示
    if fotric_image_np is not None:
        fotric_img = Image.fromarray(cv2.cvtColor(fotric_image_np, cv2.COLOR_BGR2RGB))
        fotric_img = fotric_img.resize((480, 270), Image.LANCZOS)
        fotric_imgtk = ImageTk.PhotoImage(image=fotric_img)
        fotric_video_label.imgtk = fotric_imgtk
        fotric_video_label.configure(image=fotric_imgtk)
    
    # 更新温度显示面板
    update_temperature_display()

    # 每10毫秒更新一次画面
    ids_video_label.after(10, update_frame)



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
    
    print("[线程] 图片捕获线程已启动")
    
    # 创建图片存储子文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_dir = os.path.join(save_directory.get(), f"task_{timestamp}/images")
    ids_image_dir = os.path.join(image_dir, "IDS_Camera")  # IDS相机图片目录
    computer_image_dir = os.path.join(image_dir, "Computer_Camera")  # 旁轴相机图片目录
    fotric_image_dir = os.path.join(image_dir, "Fotric_Camera")  # 红外相机图片目录
    fotric_data_dir = os.path.join(image_dir, "Fotric_Data")    # 红外相机数据目录
    
    #创建打印信息存储文件
    os.makedirs(ids_image_dir, exist_ok=True)
    os.makedirs(computer_image_dir, exist_ok=True)
    os.makedirs(fotric_image_dir, exist_ok=True)
    os.makedirs(fotric_data_dir, exist_ok=True)
    
    CSV_FILE = os.path.join(save_directory.get(), f'task_{timestamp}/print_message.csv')
    HEADER = ['image_path','timestamp','flow_rate','feed_rate','z_offset',
              'target_hotend','hot_end','bed','img_num','flow_rate_class',
              'feed_rate_class','z_offset_class','hotend_class','fotric_temp_min',
              'fotric_temp_max','fotric_temp_avg','fotric_image_path','fotric_data_path']
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
                time_to_change = 180#180秒自动改变一次参数
                count = int(time_to_change/2)
                i = 0
                if(PRINT_STATE and IMAGE_COUNT %count ==0):#180秒修改一次打印数据time,前180秒不算
                    
                    
                    if(PARAM_LOOP):
                        #循环参数
                        FLOW_RATE = PARAM_LOOP_LIST[i][2]
                        FEED_RATE = PARAM_LOOP_LIST[i][3]
                        Z_OFF = PARAM_LOOP_LIST[i][1]-PRIMARY_Z_OFF
                        TARGET_HOTEND = PARAM_LOOP_LIST[i][0]
                        i +=1
                        if(i>9):
                            i=0
                    else:
                        #随机生成较大误差参数
                        #FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND = get_random_huge_error_param()
                        pass
                    

                    #随机获取打印数据
                    rate_options = [20, 30, 40, 50, 60, 70, 80, 90, 100, 
                    110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    
                    # z_off_options 包含了指定的浮点数
                    z_off_options = [-0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32]

                    # 使用 np.random.choice 从列表中随机抽取
                    FLOW_RATE = np.random.choice(rate_options)
                    FEED_RATE = np.random.choice(rate_options)
                    Z_OFF = np.random.choice(z_off_options)
                    '''
                    FLOW_RATE = np.random.randint(20,201)
                    FEED_RATE = np.random.randint(20,201)
                    Z_OFF = (np.random.randint(0,41)-8)*0.01
                    TARGET_HOTEND = np.random.randint(180,251)
                    '''
                    change_param_auto(FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND)#自动修改参数
                #获取图像
                print(f"[第{IMAGE_COUNT}帧] 开始获取IDS相机图像...")
                # IDS相机图像
                ids_image_np = ids_process_img()
                print(f"[第{IMAGE_COUNT}帧] IDS图像获取成功，形状: {ids_image_np.shape}")
                ids_image_np = cv2.flip(ids_image_np, -1)#翻转180度
                
                print(f"[第{IMAGE_COUNT}帧] 获取旁轴相机图像...")
                # 旁轴相机图像
                computer_image_np = get_computer_camera_frame()
                
                # 获取当前时间并格式化为 "年-月-日T小时:分钟:秒-毫秒"
                current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S-%f")[:-2]
                
                # 保存IDS相机图像
                ids_image_path = os.path.join(ids_image_dir, f"image-{IMAGE_COUNT}.jpg")
                ids_img = Image.fromarray(ids_image_np)
                if ids_img.mode == 'RGBA':
                    ids_img = ids_img.convert('RGB')
                ids_img.save(ids_image_path)
                
                # 保存旁轴相机图像
                computer_image_path = None
                if computer_image_np is not None:
                    computer_image_path = os.path.join(computer_image_dir, f"image-{IMAGE_COUNT}.jpg")
                    computer_img = cv2.cvtColor(computer_image_np, cv2.COLOR_BGR2RGB)
                    computer_pil_img = Image.fromarray(computer_img)
                    computer_pil_img.save(computer_image_path)
                
                # 保存红外相机图像和数据
                fotric_image_path = None
                fotric_data_path = None
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
                            
                            # 保存彩色热像图 (JPG 格式用于直观查看)
                            if fotric_temp_max_cached > fotric_temp_min_cached:
                                normalized = ((thermal_data - fotric_temp_min_cached) / (fotric_temp_max_cached - fotric_temp_min_cached) * 255).astype(np.uint8)
                            else:
                                normalized = np.zeros_like(thermal_data, dtype=np.uint8)
                            
                            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                            
                            fotric_image_path = os.path.join(fotric_image_dir, f"image-{IMAGE_COUNT}.jpg")
                            cv2.imwrite(fotric_image_path, colored)
                            
                    except Exception as e:
                        print(f"保存红外相机数据失败: {e}")
                
                if(IMAGE_COUNT %100 ==0):#40秒更新一次显示信息（从25改为100）
                    print(f"[✓] 已保存图片 #{IMAGE_COUNT}")  # 调试信息
                image_queue.put(f"image-{IMAGE_COUNT}.jpg")

                #获取打印信息
                print(f"[第{IMAGE_COUNT}帧] 获取打印参数分类...")
                print(f"[第{IMAGE_COUNT}帧] 调用get_print_param_class_origin()...")
                flow_rate_class,feed_rate_class,z_offset_class,hotend_class,bed,hot_end = get_print_param_class_origin()
                print(f"[第{IMAGE_COUNT}帧] 打印参数获取成功")
                #flow_rate_class,feed_rate_class,z_offset_class,hotend_class,bed,hot_end = get_print_param_class_by_model(ids_image_path,model)
                
                # 构建 CSV 数据行，包含红外相机信息
                print(f"[第{IMAGE_COUNT}帧] 构建CSV行数据...")
                row_data = [ids_image_path,current_time,FLOW_RATE,FEED_RATE,Z_OFF,
                            TARGET_HOTEND,hot_end,bed,IMAGE_COUNT,flow_rate_class,
                            feed_rate_class,z_offset_class,hotend_class,
                            fotric_temp_min_cached,fotric_temp_max_cached,fotric_temp_avg_cached,
                            fotric_image_path if fotric_image_path else "",
                            fotric_data_path if fotric_data_path else ""]
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
        
        datastream.QueueBuffer(buffer)
        return image_np
    except Exception as e:
        print(f"[IDS相机] ids_process_img 出错: {e}")
        raise


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
        tk.messagebox.showwarning("警告", "请键入拍摄间隔时间")
        return
    
    # 检旁轴相机是否打开，如果失败则尝试重新初始化
    if computer_camera is None or not computer_camera.isOpened():
        print("[系统] 尝试重新初始化旁轴相机...")
        if not find_and_init_camera():
            tk.messagebox.showwarning("警告", "旁轴相机未连接，请检查Nikon Webcam Utility驱动是否正常运行")
            return
    
    # 测试摄像头是否能读取有效的帧
    try:
        test_frame = get_computer_camera_frame()
        if test_frame is None or test_frame.size == 0:
            print("[警告] 摄像头无法读取有效帧")
            tk.messagebox.showwarning("警告", "旁轴相机无法读取画面，请检查驱动状态")
            return
        print("[✓] 旁轴相机可正常读取")
    except Exception as e:
        print(f"[错误] 摄像头测试失败: {e}")
        tk.messagebox.showerror("错误", f"摄像头测试失败: {e}")
        return
    
    if not is_recording:
        # 提前初始化模型（在启动录制前，而不是等到第一帧）
        if not INIT_MODEL:
            print("[系统] 正在初始化预测模型，请稍候...")
            try:
                model = load_model()
                INIT_MODEL = True
                print("[✓] 模型初始化完成")
            except Exception as e:
                print(f"[ERROR] 模型初始化失败: {e}")
                tk.messagebox.showerror("错误", f"模型初始化失败: {e}")
                return
        
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
    global is_recording, computer_camera, camera_thread
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
        print("旁轴相机已关闭")
    
    # 停止采集
    nodemap_remote.FindNode("AcquisitionStop").Execute()
    datastream.StopAcquisition()
    # 释放资源
    ids_peak.Library.Close()
    print("IDS相机已关闭")
    
    root.destroy()
    # 关闭 Tkinter 主窗口

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

#模型版
def get_print_param_class_by_model(image_path,model):
    pred_param = predict(image_path,model)
    flow_rate_class = pred_param[0]
    feed_rate_class = pred_param[1]
    z_offset_class = pred_param[2]
    hotend_class= pred_param[3]
    res_json = get_printer_status()
    bed = res_json['temperature']['bed']['actual']
    hot_end = res_json['temperature']['tool0']['actual']
    return flow_rate_class,feed_rate_class,z_offset_class,hotend_class,bed,hot_end

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
    global FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,PRIMARY_Z_OFF
    FLOW_RATE = int(FLOW_RATE_entry.get())
    FEED_RATE = int(FEED_RATE_entry.get())
    Z_OFF = float(Z_OFF_entry.get())
    send_z_off = Z_OFF-CUR_Z_OFF
    CUR_Z_OFF = Z_OFF
    TARGET_HOTEND = int(TARGET_HOTEND_entry.get())
    res_104 = send_gcode(f"M104 S{TARGET_HOTEND}")
    res_290 = send_gcode(f"M290 Z{send_z_off}")
    res_221 = send_gcode(f"M221 S{FLOW_RATE}")
    res_220 = send_gcode(f"M220 S{FEED_RATE}")
    print(f"修改打印参数,M104 S{TARGET_HOTEND},M290 Z{send_z_off},M221 S{FLOW_RATE},M220 S{FEED_RATE}")
    if(res_221 != True or res_104 != True or res_220 !=True or res_290 != True):
        print("G代码输入出错！")

def change_param_auto(FLOW_RATE,FEED_RATE,Z_OFF,TARGET_HOTEND,init=None):
    TARGET_HOTEND = int(TARGET_HOTEND)
    global PRIMARY_Z_OFF,CUR_Z_OFF
    if init:
        res_851 = send_gcode(f"M851 Z{PRIMARY_Z_OFF}")
        print(f"初始化Z轴补偿{PRIMARY_Z_OFF}")
        if(res_851 != True):
            print("G代码输入出错！")
    send_z_off = Z_OFF-CUR_Z_OFF
    CUR_Z_OFF = Z_OFF
    flow_rate.set(FLOW_RATE)
    feed_rate.set(FEED_RATE)
    z_off.set(Z_OFF)
    target_hotend.set(TARGET_HOTEND)
    #将打印数据发送给打印机
    res_104 = send_gcode(f"M104 S{TARGET_HOTEND}")
    res_290 = send_gcode(f"M290 Z{send_z_off}")
    res_221 = send_gcode(f"M221 S{FLOW_RATE}")
    res_220 = send_gcode(f"M220 S{FEED_RATE}")
    print(f"修改打印参数,M104 S{TARGET_HOTEND},M290 Z{send_z_off},M221 S{FLOW_RATE},M220 S{FEED_RATE}")
    if(res_221 != True or res_104 != True or res_220 !=True or res_290 != True):
        print("G代码输入出错！")

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
    global FEED_RATE,FLOW_RATE,Z_OFF,TARGET_HOTEND
    FLOW_RATE = 100
    FEED_RATE = 100
    Z_OFF = 0
    TARGET_HOTEND = 200
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
    z_off_options = [-0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32]

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
print("正在初始化红外相机(Fotric 628ch)...")
try:
    # 使用真实的 FotricEnhancedDevice 进行连接
    # 根据实际的摄像头 IP 地址和端口进行配置
    fotric_device = FotricEnhancedDevice(
        ip="192.168.1.100",           # 修改为实际的设备 IP
        port=10080,                   # Fotric 默认端口
        username="admin",
        password="admin",
        simulation_mode=False,        # 禁用模拟模式，使用真实连接
        high_resolution=True,         # 启用高分辨率 640x480
        update_rate=2.0,              # 2Hz 更新频率
        sample_density=40             # 采样密度
    )
    fotric_enabled = fotric_device.is_connected
    if fotric_enabled:
        print(f"✓ 红外相机初始化成功: {fotric_device.width}x{fotric_device.height}")
    else:
        print("✗ 红外相机连接失败")
except Exception as e:
    print(f"✗ 红外相机初始化异常: {e}")
    fotric_enabled = False
    fotric_device = None

# 初始化旁轴相机
print("正在初始化旁轴相机...")

def find_and_init_camera():
    """枚举并找到虚拟摄像头（Nikon Webcam Utility）"""
    global computer_camera, camera_opened
    
    print("[摄像头检测] 扫描所有可用摄像头（跳过设备0-电脑自带相机）...")
    for i in range(1, 10):  # 从设备1开始，跳过设备0
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 测试读取一帧
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frame_info = f"设备 {i}: 分辨率 {frame.shape[1]}x{frame.shape[0]}"
                    print(f"✓ {frame_info}")
                    cap.release()
                    time.sleep(0.5)  # 等待资源释放
                    
                    # 重新打开该摄像头进行初始化
                    computer_camera = cv2.VideoCapture(i)
                    time.sleep(1.5)  # 等待虚拟摄像头初始化完成
                    
                    # 尝试设置分辨率（虚拟摄像头可能不支持，会自动降级）
                    computer_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    computer_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    computer_camera.set(cv2.CAP_PROP_FPS, 30)
                    computer_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区，降低延迟
                    
                    # 预热摄像头：读取50帧来确保缓冲区清空（Nikon虚拟摄像头需要更多帧）
                    print("  [预热中] 清除缓冲区，请等待...")
                    for j in range(50):
                        ret, frame = computer_camera.read()
                        if ret and frame is not None and j % 10 == 0:
                            print(f"  [预热 {j}/50] 成功")
                        time.sleep(0.05)  # 减少延迟，快速清空缓冲区
                    print("  [预热完成] 缓冲区已清空")
                    
                    actual_width = int(computer_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(computer_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ 旁轴相机初始化成功 (设备{i}): {actual_width}x{actual_height}")
                    camera_opened = True
                    return True
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"  设备 {i} 检测异常: {e}")
            continue
    
    print("✗ 未找到可用的旁轴相机")
    camera_opened = False
    return False

# 初始化摄像头
if not find_and_init_camera():
    print("[警告] 旁轴相机初始化失败，将在启动记录时重试")
    computer_camera = None

# 初始化 IDS peak
print("正在初始化IDS相机...")
ids_peak.Library.Initialize()
# 创建设备管理器
device_manager = ids_peak.DeviceManager.Instance()
# 更新设备列表
device_manager.Update()
# 选择第一个相机（或遍历设备列表选择指定相机）
if device_manager.Devices().empty():
    print("未找到 IDS 相机")
    exit()
else:
    print("找到IDS相机")
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
#nodemap_remote.FindNode("Width").SetValue(960)
#nodemap_remote.FindNode("Height").SetValue(640)
#nodemap_remote.FindNode("PixelFormat").SetValue("RGB8")
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

#---创建Tkinter窗口
root = tk.Tk()
root.title("3D打印监测与调控系统")
root.configure(bg='#f0f0f0')  # 设置背景色

#---设置字体样式
title_font = ('Arial', 12, 'bold')
label_font = ('Arial', 10)
param_label_font = ( 'Arial', 8)
button_font = ('Arial', 10, 'bold')
entry_font = ('Arial', 10)

# 创建两个Label用于显示两个摄像头画面
# 旁轴相机标签
computer_label_panel = tk.Frame(root, bg='#f0f0f0')
computer_label_panel.grid(row=0, column=0, rowspan=5, padx=10, pady=5, sticky='nsew')
computer_camera_title = tk.Label(computer_label_panel, text="旁轴相机", bg='#f0f0f0', font=title_font)
computer_camera_title.pack()
computer_video_label = tk.Label(computer_label_panel, bg='black', relief='ridge', bd=3, width=480, height=270)
computer_video_label.pack(padx=5, pady=5)
# IDS相机标签
ids_label_panel = tk.Frame(root, bg='#f0f0f0')
ids_label_panel.grid(row=1, column=0, rowspan=5, padx=10, pady=5, sticky='nsew')
ids_camera_title = tk.Label(ids_label_panel, text="IDS 相机", bg='#f0f0f0', font=title_font)
ids_camera_title.pack()
ids_video_label = tk.Label(ids_label_panel, bg='black', relief='ridge', bd=3, width=480, height=270)
ids_video_label.pack(padx=5, pady=5)



# 红外相机标签
fotric_label_panel = tk.Frame(root, bg='#f0f0f0')
fotric_label_panel.grid(row=2, column=0, rowspan=5, padx=10, pady=5, sticky='nsew')
fotric_camera_title = tk.Label(fotric_label_panel, text="红外相机", bg='#f0f0f0', font=title_font)
fotric_camera_title.pack()
fotric_video_label = tk.Label(fotric_label_panel, bg='black', relief='ridge', bd=3, width=384, height=280)
fotric_video_label.pack(padx=5, pady=5)

# 红外相机温度信息显示面板
tk.Label(fotric_label_panel, text="温度统计信息", bg='#e8e8e8', font=('Arial', 10, 'bold')).pack(pady=2)
fotric_temp_info = tk.Label(fotric_label_panel, text="最小: -- | 平均: -- | 最大: --", bg='#e8e8e8', font=('Arial', 9), fg='#FF5722')
fotric_temp_info.pack(pady=3)

#---控制面板框架
control_panel = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
control_panel.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

#---拍摄间隔设置
interval_frame = tk.LabelFrame(control_panel, text="拍摄设置", bg='#f0f0f0', font=title_font)
interval_frame.grid(row=0, column=0, columnspan=4, sticky='ew', pady=10)

tk.Label(interval_frame, text="拍摄间隔时间 (秒):", bg='#f0f0f0', font=label_font).grid(row=0, column=0,pady=10)
interval_entry = tk.Entry(interval_frame, font=entry_font, width=8, relief='sunken', bd=2)
interval_entry.grid(row=0, column=1, padx=5)
interval_entry.insert(0, "2")  # 设置默认值为2秒

#---保存位置设置
save_frame = tk.LabelFrame(control_panel, text="保存设置", bg='#f0f0f0', font=title_font)
save_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=10)

save_directory = tk.StringVar()
tk.Label(save_frame, text="保存位置:", bg='#f0f0f0', font=label_font).grid(row=0, column=0)
save_entry = tk.Entry(save_frame, textvariable=save_directory, font=entry_font, width=25, relief='sunken', bd=2)
save_entry.grid(row=0, column=1, padx=5)
tk.Button(save_frame, text="选择文件夹", command=select_save_directory, font=button_font, bg='#4CAF50', fg='white', relief='raised').grid(row=0, column=2, padx=5,pady=10)

#---记录控制
is_recording = False
is_paused = False
record_thread = None
camera_thread = None
PRINT_STATE = False
CLOSE_LOOP = False
PARAM_LOOP = False

#---状态参数控制面板框架
data_control_panel = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
data_control_panel.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)

#---控制打印状态
state_frame = tk.LabelFrame(data_control_panel, text="状态控制", bg='#f0f0f0', font=title_font)
state_frame.grid(row=3, column=0, columnspan=4, sticky='ew', pady=10)

tk.Button(state_frame, text="修改参数(60s)", command=update_print_state, font=button_font, bg='#9C27B0', fg='white', width=13).grid(row=0, column=0, padx=3,pady=10)
tk.Button(state_frame, text="闭环调控", command=update_close_loop_state, font=button_font, bg='#3F51B5', fg='white', width=13).grid(row=0, column=1, padx=3,pady=10)
tk.Button(state_frame, text="循环参数", command=update_param_loop_state, font=button_font, bg='#009688', fg='white', width=13).grid(row=0, column=2, padx=3,pady=10)
state_label = tk.Label(state_frame, text=f"自动修改: {PRINT_STATE}, 闭环: {CLOSE_LOOP}, 循环: {PARAM_LOOP}", bg='#f0f0f0', font=label_font, fg='#333333')
state_label.grid(row=1, column=0, columnspan=4, pady=5)

#---控制打印参数，标签及输入框
param_frame = tk.LabelFrame(data_control_panel, text="打印参数控制", bg='#f0f0f0', font=title_font)
param_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=10)
# 参数标签
tk.Label(param_frame, text="FLOW_RATE", bg='#f0f0f0', font=param_label_font).grid(row=0, column=0)
tk.Label(param_frame, text="FEED_RATE", bg='#f0f0f0', font=param_label_font).grid(row=0, column=1)
tk.Label(param_frame, text="Z_OFF", bg='#f0f0f0', font=param_label_font).grid(row=0, column=2)
tk.Label(param_frame, text="TARGET_HOTEND", bg='#f0f0f0', font=param_label_font).grid(row=0, column=3)

# 参数输入框
flow_rate = tk.StringVar(value=str(FLOW_RATE))
FLOW_RATE_entry = tk.Entry(param_frame, textvariable=flow_rate, font=entry_font, width=8, relief='sunken', bd=2)
FLOW_RATE_entry.grid(row=1, column=0, padx=5, pady=2)

feed_rate = tk.StringVar(value=str(FEED_RATE))
FEED_RATE_entry = tk.Entry(param_frame, textvariable=feed_rate, font=entry_font, width=8, relief='sunken', bd=2)
FEED_RATE_entry.grid(row=1, column=1, padx=5, pady=2)

z_off = tk.StringVar(value=str(Z_OFF))
Z_OFF_entry = tk.Entry(param_frame, textvariable=z_off, font=entry_font, width=8, relief='sunken', bd=2)
Z_OFF_entry.grid(row=1, column=2, padx=5, pady=2)

target_hotend = tk.StringVar(value=str(TARGET_HOTEND))
TARGET_HOTEND_entry = tk.Entry(param_frame, textvariable=target_hotend, font=entry_font, width=8, relief='sunken', bd=2)
TARGET_HOTEND_entry.grid(row=1, column=3, padx=5, pady=2)

classify_label = tk.Label(param_frame,text="预测分类结果(0,1,2)",bg='#f0f0f0', font=("Arial",12), fg='#333333').grid(row=2, column=1,columnspan=2, pady=2)

# 参数状态标签
FLOW_RATE_label = tk.Label(param_frame, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
FLOW_RATE_label.grid(row=3, column=0, pady=2)
FEED_RATE_label = tk.Label(param_frame, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
FEED_RATE_label.grid(row=3, column=1, pady=2)
Z_OFF_label = tk.Label(param_frame, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
Z_OFF_label.grid(row=3, column=2, pady=2)
HOTEND_label = tk.Label(param_frame, text="暂无", bg='#f0f0f0', font=label_font, fg='#333333')
HOTEND_label.grid(row=3, column=3, pady=2)

# 参数操作按钮
button_frame = tk.Frame(param_frame, bg='#f0f0f0')
button_frame.grid(row=4, column=0, columnspan=4, pady=10)

tk.Button(button_frame, text="修改参数", command=change_param_by_button, 
          font=button_font, bg='#2196F3', fg='white',width=13).grid(row=0, column=0, padx=3)
tk.Button(button_frame, text="参数回正", command=param_init, 
          font=button_font, bg='#4CAF50', fg='white',width=13).grid(row=0, column=1, padx=3)
tk.Button(button_frame, text="参数随机", command=param_ramdon, 
          font=button_font, bg='#FF9800', fg='white',width=13).grid(row=0, column=2, padx=3)

#---状态显示框架
display_panel = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
display_panel.grid(row=3, column=1, sticky='nsew', padx=5, pady=5)

#---记录控制
record_frame = tk.LabelFrame(display_panel, text="记录控制", bg='#f0f0f0', font=title_font)
record_frame.grid(row=0, column=0, columnspan=4, sticky='ew', pady=10)

tk.Button(record_frame, text="开始实验", command=start_recording, font=button_font, bg='#2196F3', fg='white', width=9).grid(row=0, column=0, padx=5,pady=10)
tk.Button(record_frame, text="停止实验", command=stop_recording,  font=button_font, bg='#f44336', fg='white', width=9).grid(row=0, column=1, padx=5,pady=10)
tk.Button(record_frame, text="继续实验", command=continue_recording, font=button_font, bg='#FF9800', fg='white', width=9).grid(row=0, column=2, padx=5,pady=10)
tk.Button(record_frame, text="结束实验", command=complete_experiment, font=button_font, bg='#607D8B', fg='white', width=9).grid(row=0, column=3, padx=5,pady=10)

# 状态显示
status_frame = tk.LabelFrame(display_panel, text="状态信息", bg='#f0f0f0', font=title_font)
status_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=10)

close_loop_label = tk.Label(status_frame, text="闭环调控情况:", bg='#f0f0f0', font=label_font)
close_loop_label.grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=10)

# 配置行列权重
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
control_panel.grid_columnconfigure(0, weight=1)

if __name__ == "__main__":
    # 定义关闭窗口的回调函数
    def on_closing():
        """应用关闭时清理资源"""
        global fotric_device, computer_camera, is_recording, camera_thread
        
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
        
        print("[系统] 资源清理完毕，应用退出")
        root.destroy()
    
    # 设置窗口关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    update_frame()
    # 开始主循环
    root.mainloop()




# 数据展示

# v：振动速度 a：振动角度 t：温度 s：振动位移 f：振动频率

# 读取寄存器 从0x3a读取1个寄存器
# device.readReg(0x3a, 1)
# 获得读取结果
# device.get(str(0x3a))

# 写入寄存器 向0x65写入50 即修改检测周期为50hz
# device.writeReg(0x65, 50)
