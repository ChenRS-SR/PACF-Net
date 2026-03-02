"""
评估脚本：加载训练好的模型，在测试集上生成混淆矩阵

Usage:
    python eval_matrix.py --model_path saved_models/20260208_042805/best_model.pt --output_dir results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

from dataset import FDMDefectDataset
from model import PACFNet


def detect_variant_from_path(model_path):
    """从模型路径自动检测变体类型"""
    path_str = str(model_path).lower()
    
    if 'concat' in path_str:
        return 'concat_only'
    elif 'no-mmd' in path_str or 'no_mmd' in path_str:
        return 'no_mmd'
    elif 'rgb-only' in path_str or 'rgb_only' in path_str:
        return 'rgb_only'
    elif 'ids-only' in path_str or 'ids_only' in path_str:
        return 'ids_only'
    else:
        return 'full'


def load_model_and_scaler(model_path, device):
    """
    加载模型和对应的 scaler
    
    Args:
        model_path: 模型文件路径 (.pt)
        device: 计算设备
    
    Returns:
        model: 加载好的模型
        scaler: 参数标准化器
    """
    model_path = Path(model_path)
    
    # 自动检测变体类型
    variant = detect_variant_from_path(model_path)
    print(f"✓ 检测到变体: {variant}")
    
    # 加载模型
    model = PACFNet(variant=variant).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 处理可能的 DataParallel 包装
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ 模型加载成功: {model_path}")
    
    # 尝试加载同目录下的 scaler
    scaler_path = model_path.parent / 'scaler.pkl'
    scaler = None
    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # 兼容两种格式：dict 或 ParameterScaler 对象
            if isinstance(loaded_data, dict):
                from dataset import ParameterScaler
                scaler = ParameterScaler()
                scaler.scalers = loaded_data
                print(f"✓ Scaler 加载成功 (dict格式): {scaler_path}")
            else:
                scaler = loaded_data
                print(f"✓ Scaler 加载成功: {scaler_path}")
        except Exception as e:
            print(f"⚠️ Scaler 加载失败: {e}")
            print("  将使用默认参数标准化")
            scaler = None
    else:
        print(f"⚠️ 未找到 scaler 文件: {scaler_path}")
    
    return model, scaler


def plot_confusion_matrix(y_true, y_pred, task_name, class_names, save_path, figsize=(8, 6)):
    """
    绘制高清混淆矩阵图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        task_name: 任务名称
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像尺寸
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比（按行归一化）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(cm_norm, annot=False, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Proportion'})
    
    # 在每个单元格中添加数量和百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            
            # 根据数值选择文字颜色
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            
            ax.text(j + 0.5, i + 0.5, f'{count}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=12, color=text_color,
                   fontweight='bold')
    
    # 设置标题和标签
    ax.set_title(f'{task_name}\nAcc={acc:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # 调整标签
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    # 保存高清图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 已保存: {save_path}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def plot_combined_confusion_matrices(results, class_names, save_path, figsize=(16, 12)):
    """
    将四个任务的混淆矩阵绘制在一张图上
    
    Args:
        results: 各任务的结果字典
        class_names: 类别名称
        save_path: 保存路径
        figsize: 图像尺寸
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    task_names = ['Flow Rate', 'Feed Rate', 'Z Offset', 'Hotend Temp']
    
    for idx, (task_key, task_name, ax) in enumerate(zip(
        ['flow_rate', 'feed_rate', 'z_offset', 'hot_end'], 
        task_names, 
        axes
    )):
        y_true = results[task_key]['y_true']
        y_pred = results[task_key]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        acc = results[task_key]['accuracy']
        
        # 绘制热力图
        sns.heatmap(cm_norm, annot=False, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   vmin=0, vmax=1, ax=ax, cbar=False)
        
        # 添加数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{count}\n({pct:.1f}%)',
                       ha='center', va='center', fontsize=10, color=text_color)
        
        ax.set_title(f'{task_name}\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    plt.suptitle('Confusion Matrices for All Tasks', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 已保存综合图: {save_path}")


def evaluate_model(model, test_loader, device):
    """
    在测试集上评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        results: 包含预测结果和标签的字典
    """
    model.eval()
    
    # 收集所有预测和标签
    all_preds = {
        'flow_rate': [],
        'feed_rate': [],
        'z_offset': [],
        'hot_end': []
    }
    all_labels = {
        'flow_rate': [],
        'feed_rate': [],
        'z_offset': [],
        'hot_end': []
    }
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[推理]"):
            # 准备输入数据
            batch_data = {
                'ids': batch['ids'].to(device),
                'computer': batch['computer'].to(device),
                'fotric': batch['fotric'].to(device),
                'thermal': batch['thermal'].to(device),
                'params': batch['params'].to(device),
            }
            
            # 模型推理
            outputs = model(batch_data, labels=None)
            
            # 提取标签
            labels_tensor = batch['labels']  # (B, 4)
            
            # 收集每个任务的结果
            all_preds['flow_rate'].extend(torch.argmax(outputs['flow_rate'], dim=1).cpu().numpy())
            all_preds['feed_rate'].extend(torch.argmax(outputs['feed_rate'], dim=1).cpu().numpy())
            all_preds['z_offset'].extend(torch.argmax(outputs['z_offset'], dim=1).cpu().numpy())
            all_preds['hot_end'].extend(torch.argmax(outputs['hot_end'], dim=1).cpu().numpy())
            
            all_labels['flow_rate'].extend(labels_tensor[:, 0].numpy())
            all_labels['feed_rate'].extend(labels_tensor[:, 1].numpy())
            all_labels['z_offset'].extend(labels_tensor[:, 2].numpy())
            all_labels['hot_end'].extend(labels_tensor[:, 3].numpy())
    
    # 整理结果
    results = {}
    task_names = ['flow_rate', 'feed_rate', 'z_offset', 'hot_end']
    
    for task in task_names:
        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        results[task] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results


def print_summary_table(results):
    """
    打印评估结果汇总表
    
    Args:
        results: 各任务的结果字典
    """
    print("\n" + "="*70)
    print("评估结果汇总")
    print("="*70)
    print(f"{'Task':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-"*70)
    
    task_display_names = {
        'flow_rate': 'Flow Rate',
        'feed_rate': 'Feed Rate',
        'z_offset': 'Z Offset',
        'hot_end': 'Hotend Temp'
    }
    
    for task in ['flow_rate', 'feed_rate', 'z_offset', 'hot_end']:
        r = results[task]
        print(f"{task_display_names[task]:<15} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f}")
    
    # 计算平均指标
    avg_acc = np.mean([results[t]['accuracy'] for t in results])
    avg_f1 = np.mean([results[t]['f1'] for t in results])
    
    print("-"*70)
    print(f"{'Average':<15} {avg_acc:>10.4f} {'-'*10} {'-'*10} {avg_f1:>10.4f}")
    print("="*70)


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 加载模型和 scaler
    model, scaler = load_model_and_scaler(args.model_path, device)
    
    # 加载测试数据集
    print(f"\n加载测试数据集: {args.test_csv}")
    test_dataset = FDMDefectDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        is_train=False,
        parameter_scaler=scaler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✓ 测试集大小: {len(test_dataset)} 样本")
    
    # 运行评估
    results = evaluate_model(model, test_loader, device)
    
    # 打印汇总表
    print_summary_table(results)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制混淆矩阵
    print(f"\n生成混淆矩阵图...")
    class_names = ['Low', 'Normal', 'High']
    task_names = {
        'flow_rate': 'Flow Rate',
        'feed_rate': 'Feed Rate', 
        'z_offset': 'Z Offset',
        'hot_end': 'Hotend Temperature'
    }
    
    # 单独保存每个任务的混淆矩阵
    for task in ['flow_rate', 'feed_rate', 'z_offset', 'hot_end']:
        save_path = output_dir / f'cm_{task}.png'
        plot_confusion_matrix(
            results[task]['y_true'],
            results[task]['y_pred'],
            task_names[task],
            class_names,
            save_path
        )
    
    # 保存综合图
    combined_path = output_dir / 'cm_all_tasks.png'
    plot_combined_confusion_matrices(results, class_names, combined_path)
    
    print(f"\n✓ 所有结果已保存到: {output_dir.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PACF-NET 评估脚本 - 生成混淆矩阵')
    
    # 模型和数据路径
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径 (e.g., saved_models/20260208_042805/best_model.pt)')
    parser.add_argument('--test_csv', type=str,
                       default=r'D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop\test.csv',
                       help='测试集 CSV 文件路径')
    parser.add_argument('--data_root', type=str,
                       default=r'D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop',
                       help='数据根目录')
    
    # 输出设置
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='输出目录路径')
    
    # 推理设置
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作线程数')
    
    args = parser.parse_args()
    main(args)
