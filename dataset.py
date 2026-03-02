import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


class ParameterScaler:
    """
    参数标准化器 (Parameter Scaler)
    
    用于标准化工艺参数，确保数据一致性。
    
    关键点 (Data Leakage 防护)：
    - 必须在训练集上 fit，计算 mean 和 std
    - 验证集和测试集必须使用训练集的 mean 和 std
    - 不能在 val/test 数据上重新 fit 或重新计算统计量
    """
    
    def __init__(self):
        self.scalers = {}  # 为每个参数维度存储独立的 scaler
        self.param_names = [
            'current_x', 'current_y', 'current_z',
            'flow_rate', 'feed_rate', 'z_offset', 'hot_end',
            'thermal_min', 'thermal_max', 'thermal_avg'
        ]
    
    def fit(self, data_list):
        """
        从参数列表中拟合 scaler
        
        Args:
            data_list: 参数数组列表，每个是 numpy array (10,)
        """
        data_array = np.array(data_list)  # (N, 10)
        
        for i, param_name in enumerate(self.param_names):
            scaler = StandardScaler()
            scaler.fit(data_array[:, i:i+1])  # reshape to (N, 1)
            self.scalers[param_name] = scaler
        
        print(f"✓ ParameterScaler 已在训练集上 fit (样本数: {len(data_list)})")
    
    def transform(self, params):
        """
        标准化参数
        
        Args:
            params: numpy array (10,)
        
        Returns:
            标准化后的参数 (10,)
        """
        if not self.scalers:
            raise ValueError("❌ ParameterScaler 未被 fit，请先调用 fit() 方法")
        
        normalized = np.zeros_like(params)
        for i, param_name in enumerate(self.param_names):
            normalized[i] = self.scalers[param_name].transform([[params[i]]])[0, 0]
        
        return normalized
    
    def save(self, save_path):
        """保存 scaler 到文件"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"✓ ParameterScaler 已保存至: {save_path}")
    
    def load(self, load_path):
        """从文件加载 scaler"""
        with open(load_path, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"✓ ParameterScaler 已加载自: {load_path}")



class FDMDefectDataset(Dataset):
    """
    FDM 打印缺陷数据集
    
    数据结构：
    - Computer_Camera: 旁轴相机图片 (224x224) - RGB 通道 (PIL 读取)
    - Fotric_Camera: 旁轴红外伪彩色图片 (224x224) - RGB 通道 (PIL 读取)
    - Fotric_data_images: 温度矩阵灰度图 (224x224) - 单通道灰度
    - IDS_Camera: 随轴相机图片 (448x448) - RGB 通道 (PIL 读取)
    
    通道顺序说明：
    - 所有 RGB 图像通过 PIL 读取，自动为 RGB 顺序（与 ImageNet 预训练权重一致）
    - 避免使用 cv2.imread（默认 BGR），会导致颜色通道反序
    
    CSV包含列：
    - task_id: 任务编号
    - computer_image_path, fotric_image_path, fotric_data_image_path, image_path
    - current_x, current_y, current_z: 打印头坐标 (mm)
    - flow_rate, flow_rate_class: 流量及分类 [20-200]
    - feed_rate, feed_rate_class: 进给速率及分类 [20-200]
    - z_offset, z_offset_class: Z轴偏移及分类 [-0.08-0.32]
    - hot_end, hotend_class: 打印头温度及分类 [150-250]°C
    - thermal_min, thermal_max, thermal_average: 温度统计 (°C)
    """
    
    def __init__(self, csv_path, data_root=None, is_train=True, parameter_scaler=None):
        """
        Args:
            csv_path: CSV 文件路径 (train.csv, val.csv, test.csv 或 all_data.csv)
            data_root: 数据根目录，如果为None则从CSV路径推断
            is_train: 是否为训练模式 (用于数据增强)
            parameter_scaler: ParameterScaler 对象，用于标准化工艺参数
                             - 训练集：None，内部计算；或者传入已拟合的 scaler
                             - 验证/测试集：必须传入训练集的 scaler（防止信息泄露）
        """
        self.data_frame = pd.read_csv(csv_path)
        self.is_train = is_train
        self.parameter_scaler = parameter_scaler
        
        # 推断数据根目录
        if data_root is None:
            # 从CSV路径推断根目录 (202601FDM_Series01_Crop)
            self.data_root = Path(csv_path).parent.parent
        else:
            self.data_root = Path(data_root)
        
        print(f"✓ 加载数据集: {len(self)} 样本 (is_train={is_train})")
        print(f"  数据根目录: {self.data_root}")
        if parameter_scaler is not None:
            print(f"  使用外部 ParameterScaler (防止 Data Leakage)")
        else:
            print(f"  未传入 ParameterScaler，参数将使用原始范围标准化")
        
        # --------------------------
        # 图像预处理 (每种相机独立处理)
        # 注：所有图像通过 PIL 读取，通道顺序为 RGB（与 ImageNet 预训练权重一致）
        # --------------------------
        
        # 1. Computer_Camera (旁轴RGB相机, 224x224, RGB 通道)
        self.computer_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 数据增强版本 (训练时) - 增强强度以防止过拟合
        if is_train:
            self.computer_transform_aug = transforms.Compose([
                transforms.RandomRotation(degrees=20),  # 增加旋转角度（在 PIL Image 上）
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # 更多仿射变换
                transforms.ToTensor(),  # 转换为张量
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # 颜色抖动（需要张量）
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # 模糊（需要张量）
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # 随机遮挡（需要张量）
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.computer_transform_aug = self.computer_transform
        
        # 2. IDS_Camera (随轴相机, 448x448, RGB 通道)
        self.ids_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if is_train:
            self.ids_transform_aug = transforms.Compose([
                transforms.RandomRotation(degrees=20),  # 增加旋转角度（在 PIL Image 上）
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # 更多仿射变换
                transforms.ToTensor(),  # 转换为张量
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # 颜色抖动（需要张量）
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # 随机遮挡（需要张量）
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.ids_transform_aug = self.ids_transform
        
        # 3. Fotric_Camera (伪彩色热像, 224x224, RGB 通道)
        # 警告：色彩顺序很重要，错误的顺序会导致特征完全不同
        self.fotric_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if is_train:
            self.fotric_transform_aug = transforms.Compose([
                transforms.RandomRotation(degrees=15),  # 增加旋转角度（在 PIL Image 上）
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 加入仿射变换
                transforms.ToTensor(),  # 转换为张量
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动（需要张量）
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # 随机遮挡（需要张量）
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.fotric_transform_aug = self.fotric_transform
        
        # 4. Fotric_data_images (温度矩阵灰度图, 224x224, 单通道)
        self.thermal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图单通道归一化
        ])
        
        if is_train:
            self.thermal_transform_aug = transforms.Compose([
                transforms.RandomRotation(degrees=15),  # 增加旋转角度（在 PIL Image 上）
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 仿射变换
                transforms.ToTensor(),  # 转换为张量
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # 随机遮挡（需要张量）
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.thermal_transform_aug = self.thermal_transform
        
        # 初始化硬编码参数范围（总是初始化，作为后备）
        # _extract_params() 中如果有外部 scaler 会覆盖这些值
        self._init_param_stats_simple()
    
    def _init_param_stats_simple(self):
        """初始化硬编码的参数范围（用于后向兼容）"""
        # 仅在没有 ParameterScaler 时使用
        self.param_ranges = {
            'current_x': (0, 200),      # X坐标 (mm)
            'current_y': (0, 200),      # Y坐标 (mm)
            'current_z': (0, 200),      # Z坐标 (mm)
            'flow_rate': (20, 200),     # 流量 [20-200]
            'feed_rate': (20, 200),     # 进给速度 [20-200]
            'z_offset': (-0.08, 0.32),  # Z轴偏移 [-0.08-0.32]
            'hot_end': (150, 250),       # 打印头温度 [150-250]°C
        }
    
    def _normalize_param_with_range(self, value, param_name):
        """使用硬编码范围标准化参数到 [-1, 1]"""
        if self.param_ranges is None:
            raise ValueError("❌ 参数范围未初始化，请传入 ParameterScaler")
        
        min_val, max_val = self.param_ranges.get(param_name, (0, 1))
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        return np.clip(normalized, -1, 1)
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        try:
            # 1. 读取四种图像
            computer_img = self._load_image(row, 'computer_image_path', 'RGB')
            ids_img = self._load_image(row, 'image_path', 'RGB')  # image_path 是 IDS_Camera
            fotric_img = self._load_image(row, 'fotric_image_path', 'RGB')
            thermal_img = self._load_image(row, 'fotric_data_image_path', 'L')  # 灰度图
            
            # 2. 应用预处理和增强
            if self.is_train:
                computer_tensor = self.computer_transform_aug(computer_img)
                ids_tensor = self.ids_transform_aug(ids_img)
                fotric_tensor = self.fotric_transform_aug(fotric_img)
                thermal_tensor = self.thermal_transform_aug(thermal_img)
            else:
                computer_tensor = self.computer_transform(computer_img)
                ids_tensor = self.ids_transform(ids_img)
                fotric_tensor = self.fotric_transform(fotric_img)
                thermal_tensor = self.thermal_transform(thermal_img)
            
            # 3. 提取工艺参数（包含温度统计）
            params = self._extract_params(row)  # (10,) - [x, y, z, flow_rate, feed_rate, z_offset, hot_end, thermal_min, thermal_max, thermal_avg]
            param_tensor = torch.tensor(params, dtype=torch.float32)
            
            # 4. 提取分类标签
            labels = torch.tensor([
                int(row['flow_rate_class']),
                int(row['feed_rate_class']),
                int(row['z_offset_class']),
                int(row['hotend_class'])
            ], dtype=torch.long)
            
            return {
                'computer': computer_tensor,      # (3, 224, 224)
                'ids': ids_tensor,                # (3, 448, 448)
                'fotric': fotric_tensor,          # (3, 224, 224)
                'thermal': thermal_tensor,        # (1, 224, 224)
                'params': param_tensor,           # (10,) - [x, y, z, flow_rate, feed_rate, z_offset, hot_end, thermal_min, thermal_max, thermal_avg]
                'labels': labels,                 # (4,) - [flow_rate_class, feed_rate_class, z_offset_class, hotend_class]
                'task_id': int(row.get('task_id', 0))  # 任务ID
            }
        
        except Exception as e:
            print(f"❌ 加载样本 {idx} 失败: {e}")
            raise
    
    def _load_image(self, row, column_name, mode='RGB'):
        """
        加载图像
        
        Args:
            row: DataFrame 行数据
            column_name: 列名
            mode: 图像模式 ('RGB' 或 'L')
        
        Returns:
            PIL Image 对象
        """
        img_path = row[column_name]
        
        # 处理相对路径
        if isinstance(img_path, str):
            path_obj = Path(img_path)
            # 如果是相对路径，相对于 data_root
            if not path_obj.is_absolute():
                img_path = self.data_root / path_obj
        
        img = Image.open(img_path).convert(mode)
        return img
    
    def _extract_params(self, row):
        """
        提取并标准化工艺参数及温度统计
        
        优先使用 ParameterScaler，如果未传入则使用硬编码范围标准化
        
        Returns:
            标准化后的参数数组 [x, y, z, flow_rate, feed_rate, z_offset, hot_end, thermal_min, thermal_max, thermal_avg]
            共10维
        """
        # 提取原始工艺参数值
        raw_params = np.array([
            float(row['current_x']),
            float(row['current_y']),
            float(row['current_z']),
            float(row['flow_rate']),
            float(row['feed_rate']),
            float(row['z_offset']),
            float(row['hot_end']),
        ], dtype=np.float32)
        
        # 提取温度统计
        thermal_stats = self._extract_thermal_stats(row)
        
        # 合并为完整参数向量 (10,)
        combined_params = np.concatenate([raw_params, thermal_stats], dtype=np.float32)
        
        # 选择标准化方法：优先外部 scaler，否则使用硬编码范围
        if self.parameter_scaler is not None:
            # 使用外部 ParameterScaler（验证/测试集）
            combined_params = self.parameter_scaler.transform(combined_params)
        else:
            # 使用硬编码范围（训练集）
            standardized = np.zeros_like(combined_params)
            # 前 7 个参数用硬编码范围
            for i in range(7):
                param_name = ['current_x', 'current_y', 'current_z', 'flow_rate', 'feed_rate', 'z_offset', 'hot_end'][i]
                standardized[i] = self._normalize_param_with_range(combined_params[i], param_name)
            # 温度统计参数不标准化（保持原始值）
            standardized[7:] = combined_params[7:]
            combined_params = standardized
        
        return combined_params
    
    def _extract_thermal_stats(self, row):
        """
        提取并归一化温度统计信息
        
        Returns:
            归一化后的温度统计 [min_temp, max_temp, avg_temp]
        """
        # 尝试多种可能的列名
        thermal_min_cols = ['thermal_min', 'thermal_minimum', 'temp_min', 'min_temp']
        thermal_max_cols = ['thermal_max', 'thermal_maximum', 'temp_max', 'max_temp']
        thermal_avg_cols = ['thermal_average', 'thermal_avg', 'temp_avg', 'avg_temp', 'thermal_mean']
        
        # 查找存在的列
        thermal_min = None
        thermal_max = None
        thermal_avg = None
        
        for col in thermal_min_cols:
            if col in row.index and pd.notna(row[col]):
                thermal_min = float(row[col])
                break
        
        for col in thermal_max_cols:
            if col in row.index and pd.notna(row[col]):
                thermal_max = float(row[col])
                break
        
        for col in thermal_avg_cols:
            if col in row.index and pd.notna(row[col]):
                thermal_avg = float(row[col])
                break
        
        # 如果找不到温度统计，使用默认值
        if thermal_min is None:
            thermal_min = 0
        if thermal_max is None:
            thermal_max = 1
        if thermal_avg is None:
            thermal_avg = 0.5
        
        # 温度归一化 (范围 [20, 250])
        thermal_min_norm = (thermal_min - 20) / (250 - 20)
        thermal_max_norm = (thermal_max - 20) / (250 - 20)
        thermal_avg_norm = (thermal_avg - 20) / (250 - 20)
        
        return np.array([thermal_min_norm, thermal_max_norm, thermal_avg_norm], dtype=np.float32)


def create_data_loaders(data_root, batch_size=32, num_workers=4, scaler_save_path='./scaler.pkl'):
    """
    创建数据加载器，并确保参数标准化的一致性 (防止 Data Leakage)
    
    流程：
    1. 加载训练集 (不使用 scaler)
    2. 从训练集提取参数，拟合 ParameterScaler
    3. 保存 scaler 到文件
    4. 使用训练集的 scaler 加载验证集和测试集
    
    Args:
        data_root: 数据根目录
        batch_size: 批大小
        num_workers: 工作线程数
        scaler_save_path: scaler 保存路径
    
    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """
    from torch.utils.data import DataLoader
    
    data_root = Path(data_root)
    
    print(f"\n{'='*70}")
    print("数据加载器初始化 (防止 Data Leakage)")
    print(f"{'='*70}")
    
    # 第一步：加载训练集 (不使用 scaler)
    print("\n[1/4] 加载训练集...")
    train_dataset = FDMDefectDataset(
        data_root / 'train.csv',
        data_root=data_root.parent,
        is_train=True,
        parameter_scaler=None
    )
    
    # 第二步：从训练集提取参数并拟合 scaler
    print("\n[2/4] 从训练集提取参数...")
    param_list = []
    for i in range(len(train_dataset)):
        row = train_dataset.data_frame.iloc[i]
        params = train_dataset._extract_params(row)
        param_list.append(params)
    
    print(f"✓ 已提取 {len(param_list)} 个参数向量")
    
    # 拟合 scaler
    print("\n[3/4] 拟合 ParameterScaler...")
    scaler = ParameterScaler()
    scaler.fit(param_list)
    
    # 保存 scaler
    scaler_path = Path(scaler_save_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(str(scaler_path))
    
    # 第三步：使用训练集的 scaler 加载验证集和测试集
    print("\n[4/4] 加载验证集和测试集 (使用训练集的 scaler)...")
    
    val_dataset = FDMDefectDataset(
        data_root / 'val.csv',
        data_root=data_root.parent,
        is_train=False,
        parameter_scaler=scaler
    )
    
    test_dataset = FDMDefectDataset(
        data_root / 'test.csv',
        data_root=data_root.parent,
        is_train=False,
        parameter_scaler=scaler
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n{'='*70}")
    print(f"✓ 数据加载器初始化完成！")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    print(f"  Scaler: {scaler_path}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    # 测试数据集和 scaler 加载
    data_root = Path(r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop")
    
    print("="*70)
    print("FDM 数据集 + ParameterScaler 加载测试")
    print("="*70)
    
    # 使用 create_data_loaders 正确初始化
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        data_root,
        batch_size=4,
        num_workers=0,
        scaler_save_path='./test_scaler.pkl'
    )
    
    # 测试一个 batch
    print("\n测试加载一个 batch...")
    for batch in train_loader:
        print(f"\nBatch 内容:")
        print(f"  Computer camera: {batch['computer'].shape}")
        print(f"  IDS camera: {batch['ids'].shape}")
        print(f"  Fotric camera: {batch['fotric'].shape}")
        print(f"  Thermal image: {batch['thermal'].shape}")
        print(f"  Parameters: {batch['params'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        
        print(f"\n参数标准化验证 (使用 ParameterScaler):")
        print(f"  参数范围: [{batch['params'].min():.4f}, {batch['params'].max():.4f}]")
        print(f"  参数均值: {batch['params'].mean():.4f}")
        print(f"  参数std: {batch['params'].std():.4f}")
        break
    
    print("\n✓ 数据集加载成功！")
    print("✓ Scaler 已保存至 ./test_scaler.pkl")