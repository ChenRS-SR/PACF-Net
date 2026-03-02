"""
预测脚本：用于实时推理和闭环调控验证

Usage:
    # 单张图片预测
    python predict.py --model_path saved_models/.../model_full.pt --image_path test.jpg
    
    # 批量预测文件夹
    python predict.py --model_path saved_models/.../model_full.pt --image_folder ./test_images
    
    # 实时预测（用于闭环）
    python predict.py --model_path saved_models/.../model_full.pt --mode realtime --interval 1.0
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import pickle
from PIL import Image
import time

from model import PACFNet
from dataset import FDMDefectDataset


class Predictor:
    """
    PACF-NET 预测器
    
    支持单张图片、批量图片和实时预测模式
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径 (.pt)
            device: 计算设备，None则自动选择
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # 加载 scaler
        self.scaler_path = self.model_path.parent / 'scaler.pkl'
        self.scaler = None
        if self.scaler_path.exists():
            try:
                with open(self.scaler_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                # 兼容两种格式：dict 或 ParameterScaler 对象
                if isinstance(loaded_data, dict):
                    from dataset import ParameterScaler
                    self.scaler = ParameterScaler()
                    self.scaler.scalers = loaded_data
                    print(f"✓ Scaler 加载成功 (dict格式)")
                else:
                    self.scaler = loaded_data
                    print(f"✓ Scaler 加载成功")
            except Exception as e:
                print(f"⚠️ Scaler 加载失败: {e}")
                print("  将使用默认参数标准化")
                self.scaler = None
        else:
            print(f"⚠️ 未找到 scaler: {self.scaler_path}")
        
        # 加载模型
        self.model = self._load_model()
        
        # 类别映射
        self.task_names = ['Flow Rate', 'Feed Rate', 'Z Offset', 'Hotend Temp']
        self.class_names = ['Low', 'Normal', 'High']
    
    def _detect_variant(self):
        """从模型路径检测变体类型"""
        path_str = str(self.model_path).lower()
        if 'concat' in path_str:
            return 'concat_only'
        elif 'no-mmd' in path_str or 'no_mmd' in path_str:
            return 'no_mmd'
        elif 'rgb-only' in path_str or 'rgb_only' in path_str:
            return 'rgb_only'
        elif 'ids-only' in path_str or 'ids_only' in path_str:
            return 'ids_only'
        return 'full'
    
    def _load_model(self):
        """加载模型"""
        variant = self._detect_variant()
        print(f"✓ 检测到变体: {variant}")
        model = PACFNet(variant=variant).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ 模型加载成功: {self.model_path}")
        return model
    
    def _load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """
        加载并预处理单张图片
        
        注意：这是简化版本，实际使用时需要根据具体模态调整
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        
        # 归一化 (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        return img_array
    
    def predict_single(self, batch_data):
        """
        单批次预测
        
        Args:
            batch_data: 符合模型输入格式的字典
        
        Returns:
            predictions: 预测结果字典
        """
        with torch.no_grad():
            # 准备输入
            for key in ['ids', 'computer', 'fotric', 'thermal', 'params']:
                if key in batch_data:
                    batch_data[key] = batch_data[key].to(self.device)
            
            # 推理
            outputs = self.model(batch_data, labels=None)
            
            # 解析预测结果
            predictions = {}
            for i, task in enumerate(['flow_rate', 'feed_rate', 'z_offset', 'hot_end']):
                pred_class = torch.argmax(outputs[task], dim=1).item()
                probs = torch.softmax(outputs[task], dim=1).cpu().numpy()[0]
                predictions[self.task_names[i]] = {
                    'class': self.class_names[pred_class],
                    'class_id': pred_class,
                    'confidence': float(probs[pred_class]),
                    'probabilities': {
                        self.class_names[j]: float(probs[j]) 
                        for j in range(3)
                    }
                }
        
        return predictions
    
    def predict_batch(self, dataloader):
        """
        批量预测
        
        Args:
            dataloader: PyTorch DataLoader
        
        Returns:
            all_predictions: 所有预测结果列表
        """
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch_data = {
                    'ids': batch['ids'].to(self.device),
                    'computer': batch['computer'].to(self.device),
                    'fotric': batch['fotric'].to(self.device),
                    'thermal': batch['thermal'].to(self.device),
                    'params': batch['params'].to(self.device),
                }
                
                outputs = self.model(batch_data, labels=None)
                
                batch_preds = []
                for i in range(len(batch['ids'])):
                    preds = {}
                    for j, task in enumerate(['flow_rate', 'feed_rate', 'z_offset', 'hot_end']):
                        pred_class = torch.argmax(outputs[task][i]).item()
                        preds[self.task_names[j]] = self.class_names[pred_class]
                    batch_preds.append(preds)
                
                all_predictions.extend(batch_preds)
        
        return all_predictions
    
    def format_output(self, predictions):
        """格式化预测输出"""
        lines = []
        lines.append("=" * 50)
        for task_name, result in predictions.items():
            lines.append(f"{task_name:12s}: {result['class']:6s} (置信度: {result['confidence']:.2%})")
        lines.append("=" * 50)
        return "\n".join(lines)


def main(args):
    # 初始化预测器
    predictor = Predictor(args.model_path, device=args.device)
    
    if args.mode == 'single':
        # 单张图片预测
        print(f"\n预测图片: {args.image_path}")
        
        # 这里需要构建完整的 batch_data
        # 简化示例：实际使用时需要根据采集系统提供的数据构造
        print("⚠️ 单张图片预测需要构造完整的4模态输入")
        print("请使用 eval_matrix.py 或 visualize.py 进行完整评估")
    
    elif args.mode == 'realtime':
        # 实时预测模式（用于闭环）
        print(f"\n启动实时预测模式 (间隔: {args.interval}s)")
        print("按 Ctrl+C 停止")
        
        try:
            while True:
                # TODO: 从采集系统获取数据
                # batch_data = get_data_from_acquisition_system()
                # predictions = predictor.predict_single(batch_data)
                # print(predictor.format_output(predictions))
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n实时预测已停止")
    
    else:
        print(f"未知模式: {args.mode}")


# ==================== 模块级别兼容函数 ====================
# 用于兼容 ids_websocket.py 等旧代码调用

# 全局模型缓存
_model_cache = {}

def load_model(model_path=None):
    """
    加载模型（兼容旧接口）
    
    Args:
        model_path: 模型路径，None则使用默认路径
    
    Returns:
        Predictor: 预测器实例
    """
    global _model_cache
    
    # 默认模型路径
    if model_path is None:
        # 尝试从默认位置加载
        default_paths = [
            Path(__file__).parent / 'saved_models' / 'full' / 'model_full.pt',
            Path(__file__).parent / 'saved_models' / 'pacs-net_full' / 'model_full.pt',
            Path(__file__).parent / 'saved_models' / 'model_full.pt',
            Path(__file__).parent / 'model_full.pt',
        ]
        for p in default_paths:
            if p.exists():
                model_path = p
                print(f"[load_model] 找到默认模型: {model_path}")
                break
        
        if model_path is None:
            print("[load_model] 错误: 未找到默认模型")
            print("[load_model] 查找路径:")
            for p in default_paths:
                print(f"  - {p} (存在: {p.exists()})")
            raise FileNotFoundError("未找到默认模型，请指定 model_path")
    
    model_path = Path(model_path)
    
    # 缓存机制
    cache_key = str(model_path)
    if cache_key not in _model_cache:
        print(f"[load_model] 加载模型: {model_path}")
        _model_cache[cache_key] = Predictor(model_path)
    
    return _model_cache[cache_key]


def predict_single(batch_data, model=None):
    """
    单批次预测（完整4模态版本）
    
    Args:
        batch_data: dict with keys:
            - 'ids': IDS随轴相机图像 (B, 3, 448, 448) 或 (3, 448, 448)
            - 'computer': 旁轴RGB相机图像 (B, 3, 224, 224) 或 (3, 224, 224)
            - 'fotric': 伪彩色热像 (B, 3, 224, 224) 或 (3, 224, 224)
            - 'thermal': 灰度热像 (B, 1, 224, 224) 或 (1, 224, 224)
            - 'params': 工艺参数 (B, 10) 或 (10,)
        model: Predictor 实例，None则自动加载
    
    Returns:
        tuple: (flow_rate_class, feed_rate_class, z_offset_class, hotend_class)
               其中每个值是 0, 1, 2 分别代表 Low, Normal, High
    """
    import torch
    
    if model is None:
        model = load_model()
    
    # 确保输入是4D/2D张量，如果不是则添加batch维度
    for key in ['ids', 'computer', 'fotric', 'thermal', 'params']:
        if key in batch_data:
            tensor = batch_data[key]
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            if tensor.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 1:  # (10,) -> (1, 10)
                tensor = tensor.unsqueeze(0)
            batch_data[key] = tensor.to(model.device)
    
    # 预测
    predictions = model.predict_single(batch_data)
    
    # 转换为类别编号 (0, 1, 2)
    class_map = {'Low': 0, 'Normal': 1, 'High': 2}
    
    flow_rate_class = class_map[predictions['Flow Rate']['class']]
    feed_rate_class = class_map[predictions['Feed Rate']['class']]
    z_offset_class = class_map[predictions['Z Offset']['class']]
    hotend_class = class_map[predictions['Hotend Temp']['class']]
    
    return flow_rate_class, feed_rate_class, z_offset_class, hotend_class


# 保留旧接口用于单图像路径预测（简化版本）
def predict_single_from_path(image_path, model=None):
    """
    从单张图片路径预测（简化版本，用于快速测试）
    
    注意：此函数使用零值填充其他模态，仅用于测试，不代表真实性能。
    实际使用时请使用 predict_single() 传入完整的4模态数据。
    
    Args:
        image_path: 图片路径
        model: Predictor 实例，None则自动加载
    
    Returns:
        tuple: (flow_rate_class, feed_rate_class, z_offset_class, hotend_class)
    """
    import torch
    from PIL import Image
    
    if model is None:
        model = load_model()
    
    device = model.device
    
    # 加载图片
    img = Image.open(image_path).convert('RGB')
    
    # IDS图像: 448x448
    ids_img = img.resize((448, 448))
    ids_array = np.array(ids_img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ids_array = (ids_array - mean) / std
    ids_tensor = torch.from_numpy(ids_array).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # 其他图像: 224x224
    small_img = img.resize((224, 224))
    small_array = np.array(small_img) / 255.0
    small_array = (small_array - mean) / std
    small_tensor = torch.from_numpy(small_array).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # 构造简化输入（注意：这是测试用的，其他模态用零填充）
    batch_data = {
        'ids': ids_tensor,
        'computer': small_tensor,
        'fotric': small_tensor,
        'thermal': torch.zeros(1, 1, 224, 224).to(device),
        'params': torch.zeros(1, 10).to(device),
    }
    
    return predict_single(batch_data, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PACF-NET 预测脚本')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    
    # 预测模式
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'realtime'],
                       help='预测模式')
    parser.add_argument('--image_path', type=str, default=None,
                       help='单张图片路径')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='实时预测间隔 (秒)')
    
    args = parser.parse_args()
    main(args)
