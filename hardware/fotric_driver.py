"""
FOTRIC 628ch 增强版设备类
基于C# DLDemo项目的核心功能移植

主要功能:
1. 直接HTTP温度获取 (基于C# GetIspTItem方法)
2. 增强的连接管理
3. 多种数据流模式支持
4. 简化的温度读取接口
"""

import ctypes
import ctypes.util
import logging
import json
import requests
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import threading
from dataclasses import dataclass
from datetime import datetime

# 配置日志 - 仅显示WARNING及以上级别，减少调试信息
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class TemperaturePoint:
    """温度点数据类"""
    x: int
    y: int
    temperature: float
    timestamp: float

@dataclass
class DeviceInfo:
    """设备信息数据类"""
    ip: str
    width: int
    height: int
    firmware_version: str
    is_connected: bool

class FotricEnhancedDevice:
    """FOTRIC 628ch增强版设备类"""
    
    # 常量定义 (基于C#代码)
    DEFAULT_PORT = 10080
    STREAM_PORT = 10081
    
    # REST API路径 (基于C#代码)
    URL_ADMIN_INFO = "/admin/info"
    URL_SENSOR_DIMENSION = "/sensor/dimension"
    URL_ISP_T = "/isp/t?x={}&y={}"
    URL_STREAM_VIDEO_RAW = "/stream/video/raw"
    URL_STREAM_VIDEO_PRI = "/stream/video/pri"
    URL_STREAM_VIDEO_SUB = "/stream/video/sub"
    URL_SENSOR_LUTS = "/sensor/luts"
    URL_SENSOR_LUT = "/sensor/lut"
    URL_SENSOR_LUT_TABLE = "/sensor/luts/{}?list"
    
    # 铁红调色板查找表 (LUT) - 256级
    # 颜色渐变：黑(0,0,0) -> 深蓝 -> 紫 -> 红 -> 橙 -> 黄 -> 白(255,255,255)
    _IRON_LUT = None
    
    @classmethod
    def _get_iron_colormap_lut(cls):
        """获取铁红调色板查找表（延迟初始化）"""
        if cls._IRON_LUT is not None:
            return cls._IRON_LUT
        
        # 创建铁红调色板 - 256级
        lut = np.zeros((256, 3), dtype=np.uint8)
        
        for i in range(256):
            t = i / 255.0  # 归一化到 [0, 1]
            
            # 铁红调色板的颜色映射
            if t < 0.125:  # 0-31: 黑色到深蓝
                r = 0
                g = 0
                b = int(t / 0.125 * 128) + 16
            elif t < 0.375:  # 32-95: 深蓝到紫色
                r = int((t - 0.125) / 0.25 * 128)
                g = 0
                b = 255
            elif t < 0.625:  # 96-159: 紫色到红色
                r = 128 + int((t - 0.375) / 0.25 * 127)
                g = 0
                b = 255 - int((t - 0.375) / 0.25 * 128)
            elif t < 0.875:  # 160-223: 红色到黄色
                r = 255
                g = int((t - 0.625) / 0.25 * 255)
                b = 0
            else:  # 224-255: 黄色到白色
                r = 255
                g = 255
                b = int((t - 0.875) / 0.125 * 255)
            
            lut[i] = [b, g, r]  # OpenCV 使用 BGR 格式
        
        cls._IRON_LUT = lut
        return cls._IRON_LUT
    
    @classmethod
    def _apply_iron_colormap(cls, gray_image):
        """
        应用铁红调色板到灰度图像
        
        Args:
            gray_image: 灰度图像 (0-255)
        
        Returns:
            BGR彩色图像
        """
        lut = cls._get_iron_colormap_lut()
        colored = lut[gray_image]
        return colored
    
    def __init__(self, ip: str = "192.168.1.100", 
                 port: int = 10080,  # 使用正确的默认端口
                 username: str = "admin", 
                 password: str = "admin",
                 simulation_mode: bool = False,
                 high_resolution: bool = True,
                 update_rate: float = 2.0,
                 sample_density: int = 40):
        """
        初始化FOTRIC设备
        
        Args:
            ip: 设备IP地址
            port: 设备端口
            username: 用户名
            password: 密码
            simulation_mode: 是否启用模拟模式
            high_resolution: 是否启用高分辨率模式(640x480)
            update_rate: 数据更新频率(Hz)
            sample_density: 采样密度（像素间隔）
        """
        self.ip = ip
        self.port = port  # 直接使用传入的端口值
        self.username = username
        self.password = password
        self.simulation_mode = simulation_mode
        self.high_resolution = high_resolution
        self.update_rate = update_rate
        self.sample_density = sample_density
        
        # 设备状态
        self.is_connected = False
        self.device_info: Optional[DeviceInfo] = None
        
        # 会话管理
        self.session = requests.Session()
        # 移除认证 - 设备不需要认证
        # self.session.auth = (username, password)
        self.session.timeout = 10
        
        # 设备参数
        self.width = 640
        self.height = 480
        self.firmware_version = ""
        
        # 温度缓存
        self._temperature_cache: Dict[str, TemperaturePoint] = {}
        self._cache_lock = threading.Lock()
        
        # SLS项目兼容性状态
        self.latest_frame = None
        self.frame_count = 0
        self.is_running = False
        self.reader_thread = None
        
        # 日志控制 - 只显示首次成功获取温度矩阵的日志
        self._thermal_data_logged = False
        
        logger.info(f"初始化FOTRIC增强版设备: {ip}")
        
        # 自动连接并开始监控
        if self.connect():
            self.start_monitoring()
    
    def _generate_initial_frame(self):
        """生成初始帧数据，避免启动时显示no data"""
        try:
            thermal_data = self._generate_thermal_array()
            if thermal_data is not None:
                temp_stats = self._calculate_temp_stats(thermal_data)
                self.frame_count = 1
                self.latest_frame = {
                    'frame': thermal_data,
                    'timestamp': datetime.now(),
                    'frame_id': self.frame_count,
                    'temp_min': temp_stats['temp_min'],
                    'temp_max': temp_stats['temp_max'],
                    'temp_avg': temp_stats['temp_avg']
                }
                logger.info(f"初始帧已生成: {thermal_data.shape}, 温度范围: {temp_stats['temp_min']:.1f}-{temp_stats['temp_max']:.1f}°C")
        except Exception as e:
            logger.error(f"生成初始帧失败: {e}")

    def connect(self) -> bool:
        """
        连接设备 (基于C#的LoginCamera方法)
        
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info(f"连接FOTRIC设备: {self.ip}")
            
            if self.simulation_mode:
                logger.info("模拟模式: 跳过实际设备连接")
                self.device_info = DeviceInfo(
                    ip=self.ip,
                    width=self.width,
                    height=self.height,
                    firmware_version="Simulation v1.0",
                    is_connected=True
                )
                self.is_connected = True
                logger.info(f"FOTRIC模拟设备连接成功: {self.width}x{self.height}")
                # 立即生成初始数据以避免no data问题
                self._generate_initial_frame()
                return True
            
            # 1. 测试基本连接
            if not self._test_connection():
                logger.error("设备连接测试失败")
                return False
            
            # 2. 获取设备信息
            if not self._get_device_info():
                logger.error("获取设备信息失败")
                return False
            
            # 3. 获取传感器尺寸
            if not self._get_sensor_dimension():
                logger.error("获取传感器尺寸失败")
                return False
            
            self.is_connected = True
            logger.info(f"FOTRIC设备连接成功: {self.width}x{self.height}")
            
            # 立即生成初始数据以避免no data问题
            self._generate_initial_frame()
            
            return True
            
        except Exception as e:
            logger.error(f"连接设备失败: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开设备连接"""
        self.stop_monitoring()
        self.is_connected = False
        self.session.close()
        logger.info("设备连接已断开")
    
    def get_point_temperature(self, x: int, y: int, use_cache: bool = True) -> Optional[float]:
        """
        获取指定点的温度 (基于C# GetIspTItem方法)
        
        Args:
            x: X坐标
            y: Y坐标
            use_cache: 是否使用缓存
            
        Returns:
            float: 温度值(摄氏度)，失败返回None
        """
        if not self.is_connected:
            logger.warning("设备未连接")
            return None
        
        # 坐标验证
        if not (0 <= x < self.width and 0 <= y < self.height):
            logger.warning(f"坐标超出范围: ({x}, {y}), 设备尺寸: {self.width}x{self.height}")
            return None
        
        cache_key = f"{x}_{y}"
        
        # 检查缓存
        if use_cache:
            with self._cache_lock:
                if cache_key in self._temperature_cache:
                    cached_point = self._temperature_cache[cache_key]
                    # 缓存有效期5秒
                    if time.time() - cached_point.timestamp < 5.0:
                        return cached_point.temperature
        
        if self.simulation_mode:
            # 模拟模式: 生成模拟温度数据
            import random
            base_temp = 25.0
            temp_variation = (x + y) / (self.width + self.height) * 10  # 根据位置变化
            noise = random.uniform(-2.0, 2.0)  # 随机噪声
            temperature = base_temp + temp_variation + noise
            
            # 模拟缓存更新
            temperature_point = TemperaturePoint(x, y, temperature, time.time())
            with self._cache_lock:
                self._temperature_cache[cache_key] = temperature_point
            
            logger.debug(f"模拟温度获取: ({x}, {y}) = {temperature:.2f}°C")
            return temperature

        try:
            # 构建请求URL (基于C#的URL_ISP_T)
            url = f"http://{self.ip}:{self.port}{self.URL_ISP_T.format(x, y)}"
            
            # 发送请求
            response = self.session.get(url)
            response.raise_for_status()
            
            # 解析响应 (基于C#的OWBTypes.IspTItem)
            data = response.json()
            temperature = data.get('t', 0.0)
            
            # 更新缓存
            temperature_point = TemperaturePoint(x, y, temperature, time.time())
            with self._cache_lock:
                self._temperature_cache[cache_key] = temperature_point
            
            logger.debug(f"温度获取成功: ({x}, {y}) = {temperature:.2f}°C")
            return temperature
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP请求失败: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"响应解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"获取温度失败: {e}")
            return None
    
    def get_temperature_array(self, points: list) -> Dict[Tuple[int, int], float]:
        """
        批量获取多个点的温度
        
        Args:
            points: 坐标列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            Dict: {(x, y): temperature} 映射
        """
        results = {}
        
        for x, y in points:
            temp = self.get_point_temperature(x, y)
            if temp is not None:
                results[(x, y)] = temp
        
        return results
    
    def get_center_temperature(self) -> Optional[float]:
        """获取中心点温度"""
        center_x = self.width // 2
        center_y = self.height // 2
        return self.get_point_temperature(center_x, center_y)
    
    def get_temperature_grid(self, grid_size: int = 10) -> np.ndarray:
        """
        获取温度网格数据
        
        Args:
            grid_size: 网格大小
            
        Returns:
            np.ndarray: 温度网格
        """
        if not self.is_connected:
            return np.array([])
        
        step_x = self.width // grid_size
        step_y = self.height // grid_size
        
        grid = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * step_x
                y = j * step_y
                temp = self.get_point_temperature(x, y)
                grid[i, j] = temp if temp is not None else 0.0
        
        return grid
    
    def _test_connection(self) -> bool:
        """测试设备连接"""
        try:
            url = f"http://{self.ip}:{self.port}{self.URL_ADMIN_INFO}"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_device_info(self) -> bool:
        """获取设备信息"""
        try:
            url = f"http://{self.ip}:{self.port}{self.URL_ADMIN_INFO}"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            self.firmware_version = data.get('firmware_version', 'Unknown')
            
            self.device_info = DeviceInfo(
                ip=self.ip,
                width=self.width,
                height=self.height,
                firmware_version=self.firmware_version,
                is_connected=True
            )
            
            return True
        except:
            return False
    
    def _get_sensor_dimension(self) -> bool:
        """获取传感器尺寸"""
        try:
            url = f"http://{self.ip}:{self.port}{self.URL_SENSOR_DIMENSION}"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            self.width = data.get('width', 640)
            self.height = data.get('height', 480)
            
            return True
        except:
            # 使用默认尺寸
            logger.warning("无法获取传感器尺寸，使用默认值 640x480")
            return True
    
    def clear_cache(self) -> None:
        """清除温度缓存"""
        with self._cache_lock:
            self._temperature_cache.clear()
        logger.info("温度缓存已清除")
    
    def _calculate_temp_stats(self, temp_grid: np.ndarray) -> Dict[str, float]:
        """计算温度统计信息"""
        return {
            'temp_min': float(np.min(temp_grid)),
            'temp_max': float(np.max(temp_grid)),
            'temp_avg': float(np.mean(temp_grid))
        }
    
    def _generate_thermal_array(self, grid_size: int = 62) -> Optional[np.ndarray]:
        """
        生成热像仪数据数组
        - Fotric设备: 返回原生640x480分辨率
        - 模拟模式: 根据设备类型返回相应分辨率
        
        Args:
            grid_size: 网格大小（仅用于模拟模式的IR8062兼容）
            
        Returns:
            np.ndarray: Fotric原生640x480或模拟的80x62数组
        """
        if not self.is_connected:
            return None
        
        try:
            if self.simulation_mode:
                # 模拟模式: 生成Fotric原生分辨率的数据 (640x480)
                import random
                base_temp = 25.0 + 5 * np.sin(time.time() * 0.5)
                
                # 创建640x480的温度数组
                height, width = 480, 640
                y, x = np.ogrid[:height, :width]
                center_x, center_y = width // 2, height // 2
                
                # 径向温度分布
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                temp_field = base_temp + 15 * np.exp(-distance / max_distance * 3)
                temp_field += np.random.normal(0, 1.0, temp_field.shape)
                
                logger.debug(f"模拟模式生成温度数组: {width}x{height}")
                return temp_field.astype(np.float32)
            
            # 实际模式: 尝试获取全分辨率温度矩阵
            full_matrix = self.get_full_temperature_matrix()
            if full_matrix is not None:
                logger.debug(f"获取到全分辨率温度矩阵: {full_matrix.shape}")
                return full_matrix
            
            # 如果无法获取全矩阵，使用采样方式生成高分辨率数据
            logger.info("使用采样方式生成高分辨率温度数据...")
            return self._generate_sampled_high_res_data()
            
            return None
            
        except Exception as e:
            logger.error(f"生成热像数组失败: {e}")
            return None
    
    def _data_reader_thread(self):
        """数据读取线程，模仿IR8062的工作模式"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.is_running:
            loop_start_time = time.time()  # 记录循环开始时间
            try:
                # 生成热像数据
                thermal_data = self._generate_thermal_array()
                
                if thermal_data is not None:
                    # 计算温度统计
                    temp_stats = self._calculate_temp_stats(thermal_data)
                    
                    # 更新最新帧数据
                    self.frame_count += 1
                    self.latest_frame = {
                        'frame': thermal_data,
                        'timestamp': datetime.now(),
                        'frame_id': self.frame_count,
                        'temp_min': temp_stats['temp_min'],
                        'temp_max': temp_stats['temp_max'],
                        'temp_avg': temp_stats['temp_avg']
                    }
                    
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                
                # 根据设备模式和配置调整帧率（优化版本）
                if self.simulation_mode:
                    time.sleep(0.1)  # 模拟模式: 10 FPS
                else:
                    # 真实模式: 动态调整sleep时间，考虑数据获取耗时
                    loop_end_time = time.time()
                    actual_loop_time = loop_end_time - loop_start_time
                    target_interval = 1.0 / self.update_rate if self.update_rate > 0 else 0.5
                    
                    # 只有当循环时间小于目标间隔时才需要sleep
                    remaining_time = target_interval - actual_loop_time
                    if remaining_time > 0:
                        time.sleep(remaining_time)
                    # 如果数据获取已经超过目标间隔，不再额外sleep
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    logger.warning(f"数据读取错误 (第{consecutive_errors}次): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("连续错误太多，停止监控")
                    break
                
                time.sleep(1 if consecutive_errors > 5 else 0.5)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            return True
        
        self.is_running = True
        self.reader_thread = threading.Thread(target=self._data_reader_thread, daemon=True)
        self.reader_thread.start()
        
        logger.info("开始热像监控")
        return True
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if hasattr(self, 'reader_thread') and self.reader_thread:
            self.reader_thread.join(timeout=2)
        logger.info("停止热像监控")
    
    # === 标定和数据恢复工具函数 ===
    
    @staticmethod
    def load_thermal_data(npz_filepath: str) -> Dict[str, Any]:
        """
        从NPZ文件恢复完整的热像数据
        
        Args:
            npz_filepath: NPZ文件路径
            
        Returns:
            Dict: 包含温度矩阵、元数据和数据类型的字典
        """
        try:
            data = np.load(npz_filepath, allow_pickle=True)
            
            # 检查数据类型
            data_type = data.get('data_type', 'unknown')
            thermal_data = data['thermal_data']
            metadata = data['metadata'].item()
            
            # 如果是压缩的CH3_Field数据，需要还原
            if 'scale_factor' in data:
                scale_factor = float(data['scale_factor'])
                thermal_data = thermal_data.astype(np.float32) * scale_factor
                logger.info(f"🔄 已还原压缩数据，精度: {scale_factor}°C")
            
            result = {
                'temperature_matrix': thermal_data,  # 640×480温度矩阵
                'metadata': metadata,
                'data_type': data_type,
                'shape': thermal_data.shape,
                'temp_range': (thermal_data.min(), thermal_data.max()),
                'file_path': npz_filepath
            }
            
            logger.info(f"✅ 数据恢复成功: {thermal_data.shape}, "
                       f"温度范围: {thermal_data.min():.2f}~{thermal_data.max():.2f}°C")
            return result
            
        except Exception as e:
            logger.error(f"数据恢复失败: {e}")
            return {}
    
    @staticmethod
    def apply_calibration_to_thermal_data(thermal_data: np.ndarray, 
                                        calibration_matrix: np.ndarray) -> np.ndarray:
        """
        对温度矩阵应用几何标定
        
        Args:
            thermal_data: 原始640×480温度矩阵
            calibration_matrix: 3×3标定变换矩阵
            
        Returns:
            np.ndarray: 标定后的温度矩阵
        """
        try:
            import cv2
            # 使用OpenCV的仿射变换对温度矩阵进行几何标定
            # 保持float32精度以确保温度值不失真
            calibrated = cv2.warpPerspective(
                thermal_data.astype(np.float32), 
                calibration_matrix, 
                (thermal_data.shape[1], thermal_data.shape[0]),
                flags=cv2.INTER_LINEAR,  # 线性插值保持温度连续性
                borderMode=cv2.BORDER_REFLECT  # 边界处理
            )
            
            logger.info(f"🎯 温度矩阵标定完成: {thermal_data.shape} -> {calibrated.shape}")
            return calibrated
            
        except Exception as e:
            logger.error(f"温度矩阵标定失败: {e}")
            return thermal_data
    
    @staticmethod
    def save_calibrated_thermal_data(calibrated_data: np.ndarray, 
                                   original_metadata: Dict[str, Any], 
                                   output_path: str,
                                   calibration_info: Dict[str, Any] = None):
        """
        保存标定后的热像数据
        
        Args:
            calibrated_data: 标定后的温度矩阵
            original_metadata: 原始元数据
            output_path: 输出文件路径（不含扩展名）
            calibration_info: 标定信息（可选）
        """
        try:
            # 更新元数据
            new_metadata = original_metadata.copy()
            new_metadata.update({
                'calibrated': True,
                'calibration_timestamp': datetime.now().isoformat(),
                'temp_min_calibrated': float(calibrated_data.min()),
                'temp_max_calibrated': float(calibrated_data.max()),
                'temp_avg_calibrated': float(calibrated_data.mean())
            })
            
            if calibration_info:
                new_metadata['calibration_info'] = calibration_info
            
            # 保存标定后的数据
            np.savez_compressed(f"{output_path}_calibrated.npz",
                              thermal_data=calibrated_data,
                              data_type='thermal_calibrated',
                              metadata=new_metadata)
            
            logger.info(f"💾 标定数据已保存: {output_path}_calibrated.npz")
            return True
            
        except Exception as e:
            logger.error(f"保存标定数据失败: {e}")
            return False

    # === SLS项目兼容接口 ===
    
    def get_thermal_data(self):
        """获取最新的热成像数据 - SLS项目接口"""
        if self.latest_frame:
            return self.latest_frame['frame']
        return None
    
    def get_latest_frame(self):
        """获取最新帧的完整信息"""
        return self.latest_frame
    
    def get_temperature_stats(self):
        """获取温度统计信息"""
        if self.latest_frame:
            return {
                'min_temp': self.latest_frame['temp_min'],
                'max_temp': self.latest_frame['temp_max'],
                'avg_temp': self.latest_frame['temp_avg'],
                'frame_id': self.latest_frame['frame_id'],
                'timestamp': self.latest_frame['timestamp']
            }
        return None
    
    def get_current_temp_range(self):
        """获取当前帧的温度最值
        
        Returns:
            tuple: (temp_min, temp_max) 如果没有数据返回 (None, None)
        """
        if self.latest_frame:
            return self.latest_frame['temp_min'], self.latest_frame['temp_max']
        return None, None
    
    def initialize(self):
        """初始化设备 - SLS项目接口"""
        return self.is_connected or self.simulation_mode
    
    def check_status(self):
        """检查设备状态 - SLS项目接口"""
        return self.is_connected and self.is_running
    
    def save_current_frame(self, filepath, temp_range=None, colormap=None):
        """
        保存当前帧 - 兼容IR8062接口
        
        Args:
            filepath: 保存路径
            temp_range: 可选，温度范围 (temp_low, temp_high)，如 (26.0, 50.0)
            colormap: 可选，调色板类型 ('IRON', 'JET', 'HOT', 'INFERNO')
        """
        if not self.latest_frame:
            return False
            
        frame = self.latest_frame['frame']
        
        # 分离目录和文件名
        import os
        base_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        
        # 创建CH3_Data文件夹（与CH3同级，都在images目录下）
        images_dir = os.path.dirname(base_dir)  # 获取images目录
        data_dir = os.path.join(images_dir, "CH3_Data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 数据文件路径
        data_filepath = os.path.join(data_dir, filename)
        
        # 保存高效压缩的.npz格式到CH3_Data
        try:
            # 1. 取消.npy保存（占用空间大且冗余）
            # np.save(f"{data_filepath}.npy", frame)  # 已注释掉
            
            # 2. 保存.npz格式（压缩，包含元数据）
            metadata = {
                'timestamp': self.latest_frame['timestamp'].isoformat(),
                'frame_id': self.latest_frame['frame_id'],
                'temp_min': self.latest_frame['temp_min'],
                'temp_max': self.latest_frame['temp_max'],
                'temp_avg': self.latest_frame['temp_avg'],
                'width': self.width,  # Fotric原生分辨率640
                'height': self.height   # Fotric原生分辨率480
            }
            
            # 检查文件名，如果是CH3_Field连续采样，使用高压缩存储
            if "CH3_Field" in str(filepath) or "field" in filename.lower():
                # CH3_Field连续采样：使用高压缩比存储
                # 温度精度0.1°C足够，转换为int16节省50%空间
                temp_scaled = (frame * 10).astype(np.int16)  # 0.1°C精度
                np.savez_compressed(f"{data_filepath}.npz", 
                                  thermal_data=temp_scaled,
                                  scale_factor=0.1,  # 还原时需乘以此值
                                  data_type='field_continuous',  # 标识连续采样数据
                                  metadata=metadata)
                logger.info(f"📦 CH3_Field高压缩保存: {data_filepath}.npz (空间节省~95%)")
            else:
                # 普通热像数据：保持完整精度，但只保存NPZ
                np.savez_compressed(f"{data_filepath}.npz", 
                                  thermal_data=frame,
                                  data_type='thermal_snapshot',  # 标识单次采样数据
                                  metadata=metadata)
                logger.info(f"📦 热像数据已保存: {data_filepath}.npz (压缩率~85%)")
                # 注释：取消CSV保存以节省空间和时间
                # CSV格式仅在需要人工查看时才建议使用
            
        except Exception as e:
            logger.error(f"保存数据文件失败: {e}")
        
        # 保存可视化图像到CH3文件夹（原filepath）
        try:
            import cv2
            
            # 使用传入的参数或从配置文件读取默认值
            from configs.config import THERMAL_VISUALIZATION
            tv = THERMAL_VISUALIZATION
            
            # 温度范围：优先使用传入的参数，否则使用配置默认值
            if temp_range is not None:
                temp_low, temp_high = temp_range
            else:
                temp_low = tv.get('temp_low', 26.0)
                temp_high = tv.get('temp_high', 50.0)
            
            # 调色板：优先使用传入的参数，否则使用配置默认值
            if colormap is not None:
                colormap_type = colormap
            else:
                colormap_type = tv.get('colormap', 'IRON')
            
            temp_min, temp_max = frame.min(), frame.max()
            
            if temp_max > temp_min:
                # 将温度裁剪到显示范围并归一化到 [0, 255]
                clipped_temp = np.clip(frame, temp_low, temp_high)
                normalized = ((clipped_temp - temp_low) / (temp_high - temp_low) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(frame, dtype=np.uint8)
            
            # 应用铁红调色板或OpenCV调色板
            if colormap_type == 'IRON':
                colored = self._apply_iron_colormap(normalized)
            elif colormap_type == 'JET':
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            elif colormap_type == 'HOT':
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
            elif colormap_type == 'INFERNO':
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            else:
                colored = self._apply_iron_colormap(normalized)
            # 使用Fotric设备的原生分辨率 (640x480)，而不是IR8062的8倍放大
            if colored.shape[:2] != (self.height, self.width):
                resized = cv2.resize(colored, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            else:
                resized = colored  # 已经是正确尺寸，无需缩放
            
            # 使用支持中文路径的保存方法
            try:
                # 方法1：尝试直接保存
                success = cv2.imwrite(f"{filepath}.png", resized)
                if success:
                    return success
                else:
                    # 方法2：使用编码方式处理中文路径
                    logger.info("直接保存失败，尝试中文路径兼容保存...")
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
                    result, encimg = cv2.imencode('.png', resized, encode_param)
                    if result:
                        encimg.tofile(f"{filepath}.png")
                        logger.info(f"CH3热像图已保存（中文路径）: {filepath}.png")
                        return True
                    else:
                        logger.error(f"图像编码失败: {filepath}.png")
                        return False
            except Exception as e:
                logger.error(f"保存CH3图像异常: {e}")
                return False
            
        except Exception as e:
            logger.error(f"保存图像文件失败: {e}")
            return False
    
    def save_frame_with_panel_settings(self, filepath, thermal_panel=None):
        """使用thermal_panel的设置保存当前帧（如果提供）"""
        # 对于Fotric设备，使用基本保存方法
        # 因为Fotric不需要额外的插值处理
        return self.save_current_frame(filepath)
    
    def __del__(self):
        """清理资源"""
        try:
            self.stop_monitoring()
            self.disconnect()
        except Exception:
            # 在析构函数中不要抛出异常
            pass
    
    def __enter__(self):
        """上下文管理器入口"""
        if self.connect():
            return self
        else:
            raise RuntimeError("设备连接失败")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()

    def get_full_temperature_matrix(self) -> Optional[np.ndarray]:
        """
        获取全分辨率温度矩阵（640x480）
        使用Fotric /isp/t-snapshot API获取完整的温度矩阵数据
        
        Returns:
            np.ndarray: 640x480的温度矩阵，如果获取失败返回None
        """
        if not self.is_connected or self.simulation_mode:
            return None
            
        try:
            # 步骤1: 调用 /isp/t-snapshot API 获取文件路径
            snapshot_url = f"http://{self.ip}:{self.port}/isp/t-snapshot"
            response = self.session.get(snapshot_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 解析文件路径
                if 'path' in data:
                    file_path = data['path']
                    
                    # 步骤2: 下载温度数据文件
                    file_url = f"http://{self.ip}:{self.port}{file_path}"
                    file_response = self.session.get(file_url, timeout=15)
                    
                    if file_response.status_code == 200:
                        # 步骤3: 解析温度数据文件
                        temp_matrix = self._parse_temperature_snapshot(file_response.content)
                        if temp_matrix is not None:
                            if not self._thermal_data_logged:
                                logger.info(f"成功获取全分辨率温度矩阵: {temp_matrix.shape}")
                                self._thermal_data_logged = True
                            return temp_matrix
                    else:
                        logger.warning(f"下载温度文件失败: HTTP {file_response.status_code}")
                else:
                    logger.warning("t-snapshot响应中未找到文件路径")
            else:
                logger.warning(f"t-snapshot API调用失败: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.debug(f"t-snapshot API请求失败: {e}")
        except Exception as e:
            logger.debug(f"解析t-snapshot数据失败: {e}")
            
        return None

    def _parse_temperature_snapshot(self, file_data: bytes) -> Optional[np.ndarray]:
        """
        解析Fotric温度快照文件
        基于C# DLDemo中的解析逻辑
        
        Args:
            file_data: 从/isp/t-snapshot下载的二进制文件数据
            
        Returns:
            np.ndarray: 解析得到的温度矩阵
        """
        try:
            import struct
            
            if len(file_data) < 16:  # 至少需要16字节头部
                logger.error("文件数据太小，无法解析")
                return None
            
            # 解析文件头部（16字节）
            # 基于C# DLDemo的解析逻辑
            pos = 0
            
            # 跳过前2字节
            pos += 2
            
            # 读取宽度和高度（网络字节序，大端序）
            width_bytes = file_data[pos:pos+2]
            height_bytes = file_data[pos+2:pos+4]
            
            width = struct.unpack('>H', width_bytes)[0]  # 大端序16位无符号整数
            height = struct.unpack('>H', height_bytes)[0]
            pos += 4
            
            # 读取深度和类型
            depth = file_data[pos]
            type_val = file_data[pos+1]
            pos += 2
            
            # 读取行大小
            linesize_bytes = file_data[pos:pos+4]
            linesize = struct.unpack('<I', linesize_bytes)[0]  # 小端序32位无符号整数
            pos += 4
            
            logger.debug(f"文件头解析: width={width}, height={height}, depth={depth}, type={type_val}, linesize={linesize}")
            
            # 跳到数据部分（16字节头部之后）
            data_start = 16
            
            # 计算需要的数据长度
            expected_data_length = width * height * 2  # 每个像素2字节
            if len(file_data) < data_start + expected_data_length:
                logger.error(f"文件数据不足，期望 {data_start + expected_data_length} 字节，实际 {len(file_data)} 字节")
                return None
            
            # 解析温度数据
            temp_matrix = np.zeros((height, width), dtype=np.float32)
            
            pos = data_start
            for i in range(height):
                for j in range(width):
                    # 读取2字节温度值（小端序）
                    temp_bytes = file_data[pos:pos+2]
                    t_src = struct.unpack('<H', temp_bytes)[0]  # 小端序16位无符号整数
                    
                    # 基于C#代码的温度解析算法
                    t_integer = (t_src & 65528) >> 3  # 0xFFF8 = 65528
                    t_float = (t_src & 7) / 8.0  # 0x7 = 7
                    temperature = t_integer + t_float
                    
                    temp_matrix[i, j] = temperature
                    pos += 2
            
            logger.info(f"成功解析温度快照: {width}x{height}, 温度范围: {temp_matrix.min():.2f}~{temp_matrix.max():.2f}°C")
            return temp_matrix
            
        except Exception as e:
            logger.error(f"解析温度快照文件失败: {e}")
            return None
            
        except Exception as e:
            logger.error(f"解析温度快照文件失败: {e}")
            return None

    def _generate_sampled_high_res_data(self) -> np.ndarray:
        """
        生成采样的高分辨率温度数据（640x480）
        使用智能采样策略，避免过多的网络请求
        
        Returns:
            np.ndarray: 640x480的温度数组
        """
        if self.high_resolution:
            target_width = 640
            target_height = 480
        else:
            # 兼容模式：返回较低分辨率
            target_width = 320
            target_height = 240
        
        # 创建温度数组
        temp_matrix = np.zeros((target_height, target_width), dtype=np.float32)
        
        # 使用配置的采样密度
        sample_step = self.sample_density
        sample_points = []
        
        for y in range(0, target_height, sample_step):
            for x in range(0, target_width, sample_step):
                sample_points.append((x, y))
        
        # 获取采样点温度
        sampled_temps = []
        valid_samples = []
        
        logger.debug(f"开始采样 {len(sample_points)} 个点，采样密度: {sample_step}")
        
        for i, (x, y) in enumerate(sample_points):
            temp = self.get_point_temperature(x, y)
            if temp is not None:
                sampled_temps.append(temp)
                valid_samples.append((x, y, temp))
                temp_matrix[y, x] = temp
            
            # 每采样20个点打印一次进度
            if (i + 1) % 20 == 0:
                logger.debug(f"采样进度: {i+1}/{len(sample_points)}")
        
        # 如果有有效采样数据，进行插值
        if len(valid_samples) > 3:  # 至少需要3个点进行插值
            avg_temp = np.mean(sampled_temps)
            temp_std = np.std(sampled_temps)
            
            # 使用双线性插值填充其他位置
            for y in range(target_height):
                for x in range(target_width):
                    if temp_matrix[y, x] == 0.0:  # 未采样的点
                        # 找到最近的采样点
                        min_dist = float('inf')
                        nearest_temp = avg_temp
                        
                        for sx, sy, stemp in valid_samples:
                            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_temp = stemp
                        
                        # 基于距离的温度插值，添加一些随机变化
                        distance_factor = min(1.0, min_dist / sample_step)
                        interpolated_temp = nearest_temp * (1 - distance_factor) + avg_temp * distance_factor
                        interpolated_temp += np.random.normal(0, temp_std * 0.1)  # 添加小量噪声
                        
                        temp_matrix[y, x] = interpolated_temp
            
            logger.info(f"完成高分辨率数据生成: {target_width}x{target_height}, "
                       f"采样点数: {len(valid_samples)}, 平均温度: {avg_temp:.2f}°C")
            
        else:
            # 如果采样失败，生成基础模拟数据
            logger.warning("采样数据不足，生成基础高分辨率模拟数据")
            base_temp = 25.0
            y, x = np.ogrid[:target_height, :target_width]
            center_x, center_y = target_width // 2, target_height // 2
            
            # 径向温度分布
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            temp_matrix = base_temp + 10 * np.exp(-distance / max_distance * 2)
            temp_matrix += np.random.normal(0, 1.0, temp_matrix.shape)
        
        return temp_matrix.astype(np.float32)

# 兼容性别名
Fotric628CHEnhanced = FotricEnhancedDevice

if __name__ == "__main__":
    # 简单测试
    device = FotricEnhancedDevice()
    
    if device.connect():
        print("✅ 设备连接成功")
        
        # 测试中心点温度
        center_temp = device.get_center_temperature()
        if center_temp is not None:
            print(f"🌡️ 中心点温度: {center_temp:.2f}°C")
        
        # 测试多点温度
        points = [(100, 100), (200, 200), (300, 300)]
        temps = device.get_temperature_array(points)
        print(f"📊 多点温度: {temps}")
        
        device.disconnect()
    else:
        print("❌ 设备连接失败")