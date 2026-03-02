"""
自动化多相机图片裁剪脚本 (测试版)
功能：支持多个相机的自动化裁剪
- Computer_Camera: 基于 CSV 坐标和 H_RGB，裁剪 448*448
- IDS_Camera: 固定中心 (575, 289)，裁剪 448*448，进行对比度增强
- Fotric_Camera: 基于 CSV 坐标和 H_IR，从对应 NPZ 文件裁剪温度数据，裁剪 224*224，归一化后保存为 PNG
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# 批量处理的任务列表和对应的图片范围 (start, end)
TASK_LIST = [
    ("task_20260122_202652", 1, 1407),
    ("task_20260122_220644", 7, 1120),
    ("task_20260122_231127", 7, 742),
    ("task_20260122_234125", 9, 920),
    ("task_20260123_012915", 15, 1738),
    ("task_20260123_062549", 15, 710),
    ("task_20260123_073418", 20, 933),
    ("task_20260123_081221", 10, 812),
    ("task_20260123_155043", 7, 774),
    ("task_20260123_162714", 6, 840),
    ("task_20260124_072841", 1, 692),
    ("task_20260124_075811", 6, 922),
]



class ImageCropper:
    """多相机图片自动裁剪工具"""
    
    # 旁轴RGB相机的单应性矩阵 (Computer_Camera)
    H_RGB = np.array([
        [-1.57739488e+00, -3.01652536e+00 , 4.95179045e+02],
        [-1.89268649e+00, -2.49586238e+00 , 4.11121812e+02],
        [-4.60507661e-03, -4.59075942e-03 , 1.00000000e+00],
    ], dtype=np.float32)
    
    # 热像相机的单应性矩阵 (Fotric_Camera)
    H_IR = np.array([
        [-1.46859995e+01, -8.81963039e+00, 1.02122124e+03],
        [-8.58883405e+00, 8.90290676e-02, 3.32716381e+02],
        [-2.62457270e-02, -2.84697457e-02, 1.00000000e+00]
    ], dtype=np.float32)
    
    def __init__(self, source_dir, output_base_dir, use_csv_coords=True, generate_debug=False):
        """
        初始化裁剪工具
        
        Args:
            source_dir: 源数据目录 (包含 images/ 和 print_message.csv)
            output_base_dir: 输出基目录
            use_csv_coords: 是否使用 CSV 坐标（True）还是单应性矩阵（False）
            generate_debug: 是否生成 Debug 文件夹
        """
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir)
        self.use_csv_coords = use_csv_coords
        self.generate_debug = generate_debug
        
        # 源路径
        self.csv_path = self.source_dir / "print_message.csv"
        self.images_dir = self.source_dir / "images"
        self.computer_camera_dir = self.images_dir / "Computer_Camera"
        self.ids_camera_dir = self.images_dir / "IDS_Camera"
        self.fotric_camera_dir = self.images_dir / "Fotric_Camera"  # 伪彩色
        self.fotric_data_dir = self.images_dir / "Fotric_Data"
        
        # CSV 坐标文件
        self.computer_csv_coords = self.images_dir / "Computer_Camera_calibration.csv"
        self.fotric_csv_coords = self.images_dir / "Fotric_Camera_calibration.csv"
        
        # 输出路径
        task_name = self.source_dir.name
        self.output_task_dir = self.output_base_dir / task_name
        self.output_computer_camera_dir = self.output_task_dir / "Computer_Camera"
        self.output_ids_camera_dir = self.output_task_dir / "IDS_Camera"
        self.output_fotric_camera_dir = self.output_task_dir / "Fotric_Camera"  # 伪彩色裁剪
        self.output_fotric_data_images_dir = self.output_task_dir / "Fotric_data_images"  # 温度矩阵裁剪
        
        # Debug 文件夹
        self.output_computer_camera_debug_dir = self.output_task_dir / "Computer_Camera_Debug"
        self.output_fotric_camera_debug_dir = self.output_task_dir / "Fotric_Camera_Debug"
        self.output_fotric_data_images_debug_dir = self.output_task_dir / "Fotric_data_images_Debug"
        
        self.output_csv_path = self.output_task_dir / "print_message.csv"
        
        # 裁剪参数
        self.computer_crop_size = 224
        self.ids_crop_size = 448
        self.ids_center = (575, 289)
        self.fotric_crop_size = 224
        
        self.processed_data = []
        
        # CSV 坐标数据 (如果使用)
        self.computer_csv_data = {}  # {img_num: (u, v)}
        self.fotric_csv_data = {}    # {img_num: (u, v)}
        
    def validate_paths(self):
        """验证输入路径是否有效"""
        checks = [
            (self.csv_path.exists(), f"CSV 文件不存在: {self.csv_path}"),
            (self.computer_camera_dir.exists(), f"Computer_Camera 文件夹不存在"),
        ]
        
        for check, msg in checks:
            if not check:
                print(f"❌ {msg}")
                return False
        
        print(f"✓ 输入路径验证通过")
        return True
    
    def create_output_dirs(self):
        """创建输出目录"""
        self.output_computer_camera_dir.mkdir(parents=True, exist_ok=True)
        
        if self.ids_camera_dir.exists():
            self.output_ids_camera_dir.mkdir(parents=True, exist_ok=True)
        
        if self.fotric_camera_dir.exists():
            self.output_fotric_camera_dir.mkdir(parents=True, exist_ok=True)
        
        if self.fotric_data_dir.exists():
            self.output_fotric_data_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建Debug文件夹 (如果启用)
        if self.generate_debug:
            if self.computer_camera_dir.exists():
                self.output_computer_camera_debug_dir.mkdir(parents=True, exist_ok=True)
            if self.fotric_camera_dir.exists():
                self.output_fotric_camera_debug_dir.mkdir(parents=True, exist_ok=True)
            if self.fotric_data_dir.exists():
                self.output_fotric_data_images_debug_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 创建输出目录: {self.output_task_dir}")
    
    def load_csv_coordinates(self):
        """加载 CSV 坐标文件"""
        if not self.use_csv_coords:
            print("⊘ 使用单应性矩阵计算坐标")
            return True
        
        # 加载 Computer_Camera 坐标
        if self.computer_csv_coords.exists():
            try:
                df = pd.read_csv(self.computer_csv_coords)
                for _, row in df.iterrows():
                    img_name = row.get('image', '')
                    # 提取 image-xxx 中的 xxx
                    if img_name.startswith('image-'):
                        img_num = int(img_name.split('-')[1].split('.')[0])
                        u = int(row.get('u', 0))
                        v = int(row.get('v', 0))
                        self.computer_csv_data[img_num] = (u, v)
                print(f"✓ 加载 Computer_Camera CSV 坐标: {len(self.computer_csv_data)} 条")
            except Exception as e:
                print(f"⚠️  加载 Computer_Camera CSV 失败: {e}")
        else:
            print(f"⚠️  Computer_Camera CSV 不存在: {self.computer_csv_coords}")
        
        # 加载 Fotric_Camera 坐标
        if self.fotric_csv_coords.exists():
            try:
                df = pd.read_csv(self.fotric_csv_coords)
                for _, row in df.iterrows():
                    img_name = row.get('image', '')
                    if img_name.startswith('image-'):
                        img_num = int(img_name.split('-')[1].split('.')[0])
                        u = int(row.get('u', 0))
                        v = int(row.get('v', 0))
                        self.fotric_csv_data[img_num] = (u, v)
                print(f"✓ 加载 Fotric_Camera CSV 坐标: {len(self.fotric_csv_data)} 条")
            except Exception as e:
                print(f"⚠️  加载 Fotric_Camera CSV 失败: {e}")
        else:
            print(f"⚠️  Fotric_Camera CSV 不存在: {self.fotric_csv_coords}")
        
        return True
    
    def load_csv(self):
        """加载 CSV 文件"""
        # 尝试多种编码
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"✓ 加载 CSV 文件: {len(self.df)} 行数据 (编码: {encoding})")
                
                # 检查必需列
                required_cols = ['img_num', 'current_x', 'current_z']
                missing = [col for col in required_cols if col not in self.df.columns]
                if missing:
                    print(f"❌ CSV 缺少列: {missing}")
                    return False
                
                return True
            except Exception as e:
                continue
        
        # 所有编码都失败
        print(f"❌ 读取 CSV 失败: 尝试了所有编码方式 (utf-8-sig, utf-8, gbk, gb2312, latin1)")
        return False
    
    def transform_coords(self, current_x, current_z, homography_matrix):
        """
        将物理坐标转换为像素坐标
        
        Args:
            current_x, current_z: 物理坐标
            homography_matrix: 单应性矩阵 (H_RGB 或 H_IR)
            
        Returns:
            (center_u, center_v): 像素坐标，如果转换失败返回 None
        """
        try:
            physical_point = np.array([[[current_x, current_z]]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(physical_point, homography_matrix)
            
            center_u = int(pixel_point[0][0][0])
            center_v = int(pixel_point[0][0][1])
            
            return center_u, center_v
        except Exception as e:
            print(f"⚠️  坐标转换失败 (x={current_x}, z={current_z}): {e}")
            return None
    
    def enhance_contrast(self, img):
        """
        对比度增强 (CLAHE)
        
        Args:
            img: BGR 图片
            
        Returns:
            增强后的图片
        """
        try:
            # 转换为 LAB 色彩空间
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 应用 CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 合并并转换回 BGR
            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            print(f"⚠️  对比度增强失败: {e}")
            return img
    
    def crop_image(self, image_path, center_u, center_v, crop_size):
        """
        裁剪图片
        
        Args:
            image_path: 源图片路径
            center_u, center_v: 裁剪中心坐标
            crop_size: 裁剪大小
            
        Returns:
            裁剪后的图片，如果失败返回 None
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️  无法读取图片: {image_path}")
                return None
            
            h, w = img.shape[:2]
            crop_half = crop_size // 2
            
            # 计算裁剪区域
            left = max(0, center_u - crop_half)
            right = min(w, center_u + crop_half)
            top = max(0, center_v - crop_half)
            bottom = min(h, center_v + crop_half)
            
            # 检查是否能裁剪出完整的区域
            if (right - left) < crop_size or (bottom - top) < crop_size:
                print(f"⚠️  中心点过靠近边界 (u={center_u}, v={center_v})")
                return None
            
            cropped = img[top:bottom, left:right]
            return cropped
        except Exception as e:
            print(f"⚠️  裁剪失败: {e}")
            return None
    
    def process_computer_camera(self, row, img_num):
        """
        处理 Computer_Camera 图片
        
        Args:
            row: CSV 行数据
            img_num: 图片编号
            
        Returns:
            (成功状态, 中心坐标(u,v))
        """
        try:
            image_name = f"image-{img_num}.jpg"
            image_path = self.computer_camera_dir / image_name
            
            if not image_path.exists():
                return False, None
            
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                return False, None
            
            # 获取裁剪中心
            if self.use_csv_coords and img_num in self.computer_csv_data:
                center_u, center_v = self.computer_csv_data[img_num]
            else:
                # 使用单应性矩阵
                current_x = float(row['current_x'])
                current_z = float(row['current_z'])
                coords = self.transform_coords(current_x, current_z, self.H_RGB)
                if coords is None:
                    return False, None
                center_u, center_v = coords
            
            # 裁剪图片
            cropped_img = self.crop_image(image_path, center_u, center_v, self.computer_crop_size)
            if cropped_img is None:
                return False, None
            
            # 保存裁剪后的图片 (PNG 格式)
            output_image_name = f"image-{img_num}.png"
            output_image_path = self.output_computer_camera_dir / output_image_name
            cv2.imwrite(str(output_image_path), cropped_img)
            
            # 生成 Debug 可视化 (如果启用)
            if self.generate_debug:
                self.save_fotric_debug_visualization(img,self.computer_crop_size, center_u, center_v, img_num, 
                                                    self.output_computer_camera_debug_dir, is_thermal=False)
            
            return True, (center_u, center_v)
        except Exception as e:
            print(f"⚠️  Computer_Camera 处理失败: {e}")
            return False, None
    
    def process_ids_camera(self, img_num):
        """
        处理 IDS_Camera 图片 (固定中心，对比度增强)
        
        Args:
            img_num: 图片编号
            
        Returns:
            处理成功返回 True
        """
        try:
            image_name = f"image-{img_num}.jpg"
            image_path = self.ids_camera_dir / image_name
            
            if not image_path.exists():
                return False
            
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️  无法读取 IDS 图片: {image_path}")
                return False
            
            # 裁剪 (固定中心)
            center_u, center_v = self.ids_center
            cropped_img = self.crop_image(image_path, center_u, center_v, self.ids_crop_size)
            if cropped_img is None:
                return False
            
            # 对裁剪后的图片进行对比度增强
            cropped_img = self.enhance_contrast(cropped_img)
            
            # 保存 (PNG 格式)
            output_image_name = f"image-{img_num}.png"
            output_image_path = self.output_ids_camera_dir / output_image_name
            cv2.imwrite(str(output_image_path), cropped_img)
            
            return True
        except Exception as e:
            print(f"⚠️  IDS_Camera 处理失败: {e}")
            return False
    
    def save_fotric_debug_visualization(self, img,crop_size, center_u, center_v, img_num, debug_dir, is_thermal=False):
        """
        保存 Fotric 调试可视化图片 (标注中心点 + 裁剪区域)
        
        Args:
            img: 源图片（BGR 或灰度）
            center_u, center_v: 裁剪中心坐标
            img_num: 图片编号
            debug_dir: Debug 输出目录
            is_thermal: 是否是热像（灰度）图片
        """
        try:
            # 如果是灰度图，转换为BGR用于绘制
            if is_thermal:
                debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                debug_img = img.copy()
            
            crop_half = crop_size // 2
            left = max(0, center_u - crop_half)
            right = min(img.shape[1], center_u + crop_half)
            top = max(0, center_v - crop_half)
            bottom = min(img.shape[0], center_v + crop_half)
            
            # 绘制裁剪区域矩形 (红色线条)
            cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # 绘制中心点 (红色实心圆)
            cv2.circle(debug_img, (center_u, center_v), 6, (0, 0, 255), -1)
            # 绘制中心点的十字线
            cv2.line(debug_img, (center_u - 12, center_v), (center_u + 12, center_v), (0, 0, 255), 2)
            cv2.line(debug_img, (center_u, center_v - 12), (center_u, center_v + 12), (0, 0, 255), 2)
            
            # 添加文字标注
            cv2.putText(debug_img, f"Center: ({center_u}, {center_v})", 
                       (center_u + 15, center_v - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 255), 1)
            cv2.putText(debug_img, f"Size: {self.fotric_crop_size}x{self.fotric_crop_size}", 
                       (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 255), 1)
            
            # 保存调试图片
            debug_image_name = f"image-{img_num}.png"
            debug_image_path = debug_dir / debug_image_name
            cv2.imwrite(str(debug_image_path), debug_img)
            
        except Exception as e:
            print(f"⚠️  保存调试图片失败: {e}")
    
    def process_fotric_camera_pseudo_color(self, img_num):
        """
        处理 Fotric_Camera 伪彩色图片
        
        Args:
            img_num: 图片编号
            
        Returns:
            (成功状态, 中心坐标(u,v))
        """
        try:
            image_name = f"image-{img_num}.jpg"
            image_path = self.fotric_camera_dir / image_name
            
            if not image_path.exists():
                return False, None
            
            # 获取裁剪中心
            if img_num not in self.fotric_csv_data:
                return False, None
            
            center_u, center_v = self.fotric_csv_data[img_num]
            
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️  无法读取 Fotric_Camera 伪彩色图片: {image_path}")
                return False, None
            
            # 裁剪图片
            cropped_img = self.crop_image(image_path, center_u, center_v, self.fotric_crop_size)
            if cropped_img is None:
                return False, None
            
            # 保存裁剪后的图片 (PNG 格式)
            output_image_name = f"image-{img_num}.png"
            output_image_path = self.output_fotric_camera_dir / output_image_name
            cv2.imwrite(str(output_image_path), cropped_img)
            
            # 生成 Debug 可视化 (如果启用)
            if self.generate_debug:
                self.save_fotric_debug_visualization(img,self.fotric_crop_size, center_u, center_v, img_num, 
                                                    self.output_fotric_camera_debug_dir, is_thermal=False)
            
            return True, (center_u, center_v)
        except Exception as e:
            print(f"⚠️  Fotric_Camera 伪彩色处理失败: {e}")
            return False, None
    
    def process_fotric_data_thermal(self, img_num):
        """
        处理 Fotric_data 温度矩阵图片
        
        Args:
            img_num: 图片编号
            
        Returns:
            (成功状态, 中心坐标(u,v))
        """
        try:
            # 获取裁剪中心
            if img_num not in self.fotric_csv_data:
                return False, None
            
            center_u, center_v = self.fotric_csv_data[img_num]
            
            # 对应的 NPZ 文件路径 (thermal_data-{img_num}.npz)
            npz_name = f"thermal_data-{img_num}.npz"
            npz_path = self.fotric_data_dir / npz_name
            
            if not npz_path.exists():
                return False, None
            
            # 加载 NPZ 文件获取温度矩阵
            data = np.load(npz_path)
            temps = data['thermal_data']  # 温度数据存储在 'thermal_data' 字段
            
            crop_half = self.fotric_crop_size // 2
            
            # 计算裁剪区域
            left = max(0, center_u - crop_half)
            right = min(temps.shape[1], center_u + crop_half)
            top = max(0, center_v - crop_half)
            bottom = min(temps.shape[0], center_v + crop_half)
            
            # 检查是否能裁剪出完整的区域
            if (right - left) < self.fotric_crop_size or (bottom - top) < self.fotric_crop_size:
                print(f"⚠️  热像中心点过靠近边界 (u={center_u}, v={center_v})")
                return False, None
            
            # 裁剪温度矩阵
            cropped_temps = temps[top:bottom, left:right]
            
            # 绝对温度归一化
            # 温度范围 [20, 250]，归一化到 [0, 1]
            normalized = (cropped_temps - 20) / (250 - 20)
            normalized = np.clip(normalized, 0.0, 1.0)  # 限制在 [0, 1]
            
            # 转换为 uint8
            img_uint8 = (normalized * 255).astype(np.uint8)
            
            # 保存为单通道 PNG
            image_name = f"image-{img_num}.png"
            output_image_path = self.output_fotric_data_images_dir / image_name
            cv2.imwrite(str(output_image_path), img_uint8)
            
            # 生成 Debug 可视化 (如果启用) - 使用原始温度矩阵的可视化
            if self.generate_debug:
                # 将整个原始温度矩阵转换为可视化图像（用于 debug）
                temps_normalized = (temps - 20) / (250 - 20)
                temps_normalized = np.clip(temps_normalized, 0.0, 1.0)
                temps_vis = (temps_normalized * 255).astype(np.uint8)
                self.save_fotric_debug_visualization(temps_vis,self.fotric_crop_size, center_u, center_v, img_num, 
                                                    self.output_fotric_data_images_debug_dir, is_thermal=True)
            
            return True, (center_u, center_v)
        except Exception as e:
            print(f"⚠️  Fotric_data 温度矩阵处理失败: {e}")
            return False, None
    
    def process_images(self, start_num=1, end_num=9999):
        """
        处理指定范围的图片
        
        Args:
            start_num: 起始图片编号
            end_num: 结束图片编号
        """
        print(f"\n{'='*70}")
        print(f"开始处理图片 (范围: {start_num} - {end_num})")
        print(f"{'='*70}")
        
        processed_count = 0
        skipped_count = 0
        
        # 检查各相机目录是否存在
        has_computer = self.computer_camera_dir.exists()
        has_ids = self.ids_camera_dir.exists()
        has_fotric_camera = self.fotric_camera_dir.exists()
        has_fotric_data = self.fotric_data_dir.exists()
        
        print(f"相机检测: Computer={has_computer}, IDS={has_ids}, Fotric_Camera={has_fotric_camera}, Fotric_data={has_fotric_data}\n")
        
        # 遍历 CSV 每一行
        for idx, row in self.df.iterrows():
            img_num = int(row['img_num'])
            
            # 检查范围
            if img_num < start_num or img_num > end_num:
                continue
            
            success = False
            computer_center = None
            fotric_center = None
            
            # 处理 Computer_Camera
            if has_computer:
                success_comp, center_comp = self.process_computer_camera(row, img_num)
                if success_comp:
                    success = True
                    computer_center = center_comp
            
            # 处理 IDS_Camera
            if has_ids:
                success_ids = self.process_ids_camera(img_num)
                if success_ids and not success:
                    success = True
            
            # 处理 Fotric_Camera 伪彩色
            if has_fotric_camera:
                success_fc, center_fc = self.process_fotric_camera_pseudo_color(img_num)
                if success_fc:
                    success = True
                    fotric_center = center_fc
            
            # 处理 Fotric_data 温度矩阵
            if has_fotric_data:
                success_fd, center_fd = self.process_fotric_data_thermal(img_num)
                if success_fd:
                    success = True
                    if fotric_center is None:
                        fotric_center = center_fd
            
            if success:
                # 添加裁剪坐标到行数据
                if computer_center:
                    row['computer_image_crop_u'] = computer_center[0]
                    row['computer_image_crop_v'] = computer_center[1]
                else:
                    row['computer_image_crop_u'] = None
                    row['computer_image_crop_v'] = None
                
                if fotric_center:
                    row['fotric_camera_crop_u'] = fotric_center[0]
                    row['fotric_camera_crop_v'] = fotric_center[1]
                else:
                    row['fotric_camera_crop_u'] = None
                    row['fotric_camera_crop_v'] = None
                
                # 保存数据行信息
                row_copy = row.copy()
                self.processed_data.append(row_copy)
                processed_count += 1
                
                # 每 50 张图片打印一次进度
                if processed_count % 50 == 0:
                    print(f"  ✓ 已处理 {processed_count} 张图片")
            else:
                skipped_count += 1
        
        print(f"\n{'='*70}")
        print(f"处理完成!")
        print(f"  ✓ 成功处理: {processed_count} 张")
        print(f"  ⊘ 跳过: {skipped_count} 张")
        print(f"{'='*70}\n")
        
        return processed_count > 0
    def save_csv(self):
        """保存处理结果到 CSV"""
        try:
            output_df = pd.DataFrame(self.processed_data)
            
            # 更新路径列
            for idx, row in output_df.iterrows():
                img_num = int(row['img_num'])
                image_name = f"image-{img_num}.png"
                
                # IDS_Camera (image_path)
                output_df.loc[idx, 'image_path'] = str(self.output_ids_camera_dir / image_name)
                
                # Computer_Camera (computer_image_path)
                output_df.loc[idx, 'computer_image_path'] = str(self.output_computer_camera_dir / image_name)
                
                # Fotric_Camera 伪彩色 (fotric_image_path)
                output_df.loc[idx, 'fotric_image_path'] = str(self.output_fotric_camera_dir / image_name)
                
                # Fotric_data 温度矩阵 (fotric_data_image_path)
                output_df.loc[idx, 'fotric_data_image_path'] = str(self.output_fotric_data_images_dir / image_name)
            
            # 删除原来的 fotric_data_path 列（如果存在）
            if 'fotric_data_path' in output_df.columns:
                output_df.drop('fotric_data_path', axis=1, inplace=True)
            
            # 尝试使用 UTF-8-sig 编码保存，如果失败则使用 GBK
            try:
                output_df.to_csv(self.output_csv_path, index=False, encoding='utf-8-sig')
                print(f"✓ 保存 CSV 文件: {self.output_csv_path} (编码: utf-8-sig)")
            except:
                output_df.to_csv(self.output_csv_path, index=False, encoding='gbk')
                print(f"✓ 保存 CSV 文件: {self.output_csv_path} (编码: gbk)")
            
            print(f"  包含 {len(output_df)} 条记录")
            return True
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")
            return False
    
    def run(self, start_num=1, end_num=9999):
        """
        运行完整的裁剪流程
        
        Args:
            start_num: 起始图片编号
            end_num: 结束图片编号
        """
        print(f"\n{'#'*70}")
        print(f"# 自动化多相机图片裁剪工具")
        print(f"{'#'*70}\n")
        
        # 验证路径
        if not self.validate_paths():
            return False
        
        # 创建输出目录
        self.create_output_dirs()
        
        # 加载 CSV
        if not self.load_csv():
            return False
        
        # 加载 CSV 坐标
        if self.use_csv_coords:
            self.load_csv_coordinates()
        
        # 处理图片
        if not self.process_images(start_num=start_num, end_num=end_num):
            print("❌ 未成功处理任何图片")
            return False
        
        # 保存结果 CSV
        if not self.save_csv():
            return False
        
        print(f"✓ 全部完成！")
        print(f"  输出目录: {self.output_task_dir}")
        print(f"  ✓ Computer_Camera: {self.output_computer_camera_dir}")
        print(f"  ✓ IDS_Camera: {self.output_ids_camera_dir}")
        print(f"  ✓ Fotric_Camera (伪彩色): {self.output_fotric_camera_dir}")
        print(f"  ✓ Fotric_data_images (温度矩阵): {self.output_fotric_data_images_dir}")
        
        if self.generate_debug:
            print(f"  🔍 Computer_Camera_Debug: {self.output_computer_camera_debug_dir}")
            print(f"  🔍 Fotric_Camera_Debug: {self.output_fotric_camera_debug_dir}")
            print(f"  🔍 Fotric_data_images_Debug: {self.output_fotric_data_images_debug_dir}")
        
        print(f"  📝 CSV 结果: {self.output_csv_path}")
        
        return True


def main():
    """主函数"""
    import sys
    
    print("\n" + "="*70)
    print("自动化多相机图片裁剪工具")
    print("="*70 + "\n")
    
    # 1. 选择任务
    print("可用的任务列表:")
    for i, (task_name, start, end) in enumerate(TASK_LIST, 1):
        print(f"  {i}. {task_name} (image-{start} 到 image-{end})")
    print(f"  13. 处理全部任务 (1-12)")
    
    task_choice = input("\n请选择任务编号 (默认 1): ").strip()
    
    # 处理全部任务
    if task_choice == '13':
        print("\n您选择了处理全部任务。")
        
        # 2. 选择裁剪方式
        print("\n裁剪中心获取方式:")
        print("  1. 使用 CSV 文件中的坐标 (推荐)")
        print("  2. 使用单应性矩阵计算")
        
        coord_choice = input("\n请选择方式 (默认 1): ").strip()
        use_csv_coords = coord_choice != '2' if coord_choice else True
        
        print(f"  ✓ 选择: {'CSV 坐标' if use_csv_coords else '单应性矩阵'}")
        
        # 3. 选择是否生成 Debug 文件夹
        print("\nDebug 文件夹:")
        print("  1. 生成 Debug 文件夹 (标注中心点和裁剪区域)")
        print("  2. 不生成")
        
        debug_choice = input("\n请选择 (默认 2): ").strip()
        generate_debug = debug_choice == '1'
        
        print(f"  ✓ 选择: {'生成 Debug' if generate_debug else '不生成 Debug'}")
        
        # 确认开始
        print("\n" + "="*70)
        print("警告：将处理全部 12 个任务，这可能需要很长时间！")
        print("="*70)
        confirm = input("确认开始处理？(y/n，默认 n): ").strip().lower()
        
        if confirm != 'y':
            print("已取消处理")
            return
        
        # 批量处理所有任务
        base_dir = r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01"
        output_base_dir = Path(base_dir + "_Crop")
        
        successful_tasks = 0
        failed_tasks = 0
        
        for task_idx, (task_name, start_num, end_num) in enumerate(TASK_LIST, 1):
            print(f"\n{'#'*70}")
            print(f"# 处理任务 {task_idx}/12: {task_name}")
            print(f"{'#'*70}")
            
            source_dir = Path(base_dir) / task_name
            
            # 创建裁剪工具
            cropper = ImageCropper(str(source_dir), str(output_base_dir), 
                                  use_csv_coords=use_csv_coords, 
                                  generate_debug=generate_debug)
            
            # 运行处理
            success = cropper.run(start_num=start_num, end_num=end_num)
            
            if success:
                successful_tasks += 1
                print(f"\n✓ 任务 {task_idx}/12 处理成功！")
            else:
                failed_tasks += 1
                print(f"\n❌ 任务 {task_idx}/12 处理失败")
        
        # 最终统计
        print(f"\n{'='*70}")
        print(f"全部任务处理完成！")
        print(f"{'='*70}")
        print(f"✓ 成功: {successful_tasks}/12 个任务")
        print(f"❌ 失败: {failed_tasks}/12 个任务")
        print(f"输出目录: {output_base_dir}")
        
        return
    
    # 单个任务处理
    try:
        task_idx = int(task_choice) - 1 if task_choice else 0
        if task_idx < 0 or task_idx >= len(TASK_LIST):
            print("❌ 无效的任务编号")
            return
        task_name, start_num, end_num = TASK_LIST[task_idx]
    except ValueError:
        print("❌ 请输入有效的数字")
        return
    
    # 2. 选择裁剪方式
    print("\n裁剪中心获取方式:")
    print("  1. 使用 CSV 文件中的坐标 (推荐)")
    print("  2. 使用单应性矩阵计算")
    
    coord_choice = input("\n请选择方式 (默认 1): ").strip()
    use_csv_coords = coord_choice != '2' if coord_choice else True
    
    print(f"  ✓ 选择: {'CSV 坐标' if use_csv_coords else '单应性矩阵'}")
    
    # 3. 选择是否生成 Debug 文件夹
    print("\nDebug 文件夹:")
    print("  1. 生成 Debug 文件夹 (标注中心点和裁剪区域)")
    print("  2. 不生成")
    
    debug_choice = input("\n请选择 (默认 2): ").strip()
    generate_debug = debug_choice == '1'
    
    print(f"  ✓ 选择: {'生成 Debug' if generate_debug else '不生成 Debug'}")
    
    # 配置路径
    base_dir = r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01"
    source_dir = Path(base_dir) / task_name
    output_base_dir = Path(base_dir + "_Crop")
    
    print(f"\n配置信息:")
    print(f"  源目录: {source_dir}")
    print(f"  输出目录: {output_base_dir}")
    print(f"  处理范围: image-{start_num} 到 image-{end_num}")
    print(f"  裁剪方式: {'CSV 坐标' if use_csv_coords else '单应性矩阵'}")
    print(f"  Debug: {'启用' if generate_debug else '禁用'}")
    
    # 创建裁剪工具
    cropper = ImageCropper(str(source_dir), str(output_base_dir), 
                          use_csv_coords=use_csv_coords, 
                          generate_debug=generate_debug)
    
    # 运行处理
    success = cropper.run(start_num=start_num, end_num=end_num)
    
    if success:
        print("\n✓ 处理成功！")
    else:
        print("\n❌ 处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()