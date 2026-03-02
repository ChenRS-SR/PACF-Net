"""
可视化脚本：t-SNE 特征分布图 和 Attention Map 热力图

Usage:
    # t-SNE 可视化 (证明 MMD 有效性)
    python visualize.py --model_path saved_models/full/model_full.pt --mode tsne
    
    # Attention Map 可视化
    python visualize.py --model_path saved_models/full/model_full.pt --mode attention --sample_idx 10
    
    # 同时生成两种图
    python visualize.py --model_path saved_models/full/model_Full.pt --mode all
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle
import cv2
from PIL import Image

from dataset import FDMDefectDataset
from model import PACFNet


def detect_variant_from_path(model_path):
    """
    从模型路径自动检测变体类型
    
    例如：
    - saved_models/concat-only_xxx/model_concat.pt -> 'concat_only'
    - saved_models/no-mmd_xxx/model_no_mmd.pt -> 'no_mmd'
    - saved_models/full_xxx/model_full.pt -> 'full'
    """
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
    """加载模型和 scaler"""
    model_path = Path(model_path)
    
    # 自动检测变体类型
    variant = detect_variant_from_path(model_path)
    print(f"✓ 检测到变体: {variant}")
    
    model = PACFNet(variant=variant).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ 模型加载成功: {model_path}")
    
    # 加载 scaler
    scaler_path = model_path.parent / 'scaler.pkl'
    scaler = None
    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # 兼容两种格式：dict 或 ParameterScaler 对象
            if isinstance(loaded_data, dict):
                # 如果是 dict，包装成 ParameterScaler
                from dataset import ParameterScaler
                scaler = ParameterScaler()
                scaler.scalers = loaded_data
                print(f"✓ Scaler 加载成功 (dict格式): {scaler_path}")
            else:
                # 已经是 ParameterScaler 对象
                scaler = loaded_data
                print(f"✓ Scaler 加载成功: {scaler_path}")
        except Exception as e:
            print(f"⚠️ Scaler 加载失败: {e}")
            print("  可能原因: NumPy版本不兼容 (保存时用的NumPy版本与当前不同)")
            print("  解决方案: 重新训练模型，或在相同NumPy版本环境下加载")
            print("  当前将不使用scaler (可能导致推理结果不准确)")
            scaler = None
    else:
        print(f"⚠️ 未找到 scaler 文件: {scaler_path}")
    
    return model, scaler


def extract_features_for_tsne(model, dataloader, device, max_samples=1000):
    """
    提取用于 t-SNE 可视化的特征
    
    Returns:
        features_dict: 包含各种特征的字典
        labels_dict: 包含标签的字典
    """
    model.eval()
    
    all_features = {
        'f_local': [],  # IDS 特征
        'f_rgb': [],    # RGB 特征
        'f_thermal': [], # 热成像特征
        'fused': [],    # 融合后特征
    }
    
    all_labels = {
        'flow_rate': [],
        'feed_rate': [],
        'z_offset': [],
        'hot_end': [],
    }
    
    print(f"\n提取特征用于 t-SNE (最多 {max_samples} 个样本)...")
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[特征提取]"):
            if sample_count >= max_samples:
                break
            
            batch_data = {
                'ids': batch['ids'].to(device),
                'computer': batch['computer'].to(device),
                'fotric': batch['fotric'].to(device),
                'thermal': batch['thermal'].to(device),
                'params': batch['params'].to(device),
            }
            
            # 使用 extract_features 方法
            features = model.extract_features(batch_data, return_attention=False)
            
            # 收集特征
            all_features['f_local'].append(features['f_local'].cpu().numpy())
            all_features['f_rgb'].append(features['f_rgb'].cpu().numpy())
            all_features['f_thermal'].append(features['f_thermal'].cpu().numpy())
            all_features['fused'].append(features['fused_feat'].cpu().numpy())
            
            # 收集标签
            labels_tensor = batch['labels']
            all_labels['flow_rate'].extend(labels_tensor[:, 0].numpy())
            all_labels['feed_rate'].extend(labels_tensor[:, 1].numpy())
            all_labels['z_offset'].extend(labels_tensor[:, 2].numpy())
            all_labels['hot_end'].extend(labels_tensor[:, 3].numpy())
            
            sample_count += batch['ids'].size(0)
    
    # 合并所有批次
    for key in all_features:
        all_features[key] = np.concatenate(all_features[key], axis=0)[:max_samples]
    for key in all_labels:
        all_labels[key] = np.array(all_labels[key])[:max_samples]
    
    print(f"✓ 提取了 {len(all_features['f_local'])} 个样本的特征")
    
    return all_features, all_labels


def plot_tsne_dual_features(features_dict, labels, label_name, save_path, figsize=(14, 6)):
    """
    绘制 t-SNE 对比图：展示 MMD 对齐前后的特征分布
    
    左图：RGB vs Local (对齐前，MMD之前)
    右图：融合后的特征按类别着色
    """
    # 准备数据：对比 RGB 和 Local 特征 (MMD 对齐前)
    f_rgb = features_dict['f_rgb']
    f_local = features_dict['f_local']
    
    # 合并 RGB 和 Local 特征用于 t-SNE
    n_samples = len(f_rgb)
    combined_pre = np.vstack([f_rgb, f_local])  # (2N, 512)
    
    # 标签：0=RGB, 1=Local
    domain_labels = np.array([0] * n_samples + [1] * n_samples)
    
    # 运行 t-SNE
    print(f"  运行 t-SNE (pre-alignment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embedded_pre = tsne.fit_transform(combined_pre)
    
    # 运行 t-SNE (融合后的特征，按类别着色)
    print(f"  运行 t-SNE (fused features)...")
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embedded_fused = tsne2.fit_transform(features_dict['fused'])
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：MMD 对齐前 (RGB vs Local)
    ax1 = axes[0]
    rgb_mask = domain_labels == 0
    local_mask = domain_labels == 1
    
    ax1.scatter(embedded_pre[rgb_mask, 0], embedded_pre[rgb_mask, 1],
               c='royalblue', label='RGB Features', alpha=0.5, s=20)
    ax1.scatter(embedded_pre[local_mask, 0], embedded_pre[local_mask, 1],
               c='coral', label='Local (IDS) Features', alpha=0.5, s=20)
    ax1.set_title('Before MMD Alignment\n(RGB vs Local Features)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：融合后的特征按类别着色
    ax2 = axes[1]
    class_names = ['Low', 'Normal', 'High']
    colors = ['#e74c3c', '#2ecc71', '#3498db']  # 红、绿、蓝
    
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        ax2.scatter(embedded_fused[mask, 0], embedded_fused[mask, 1],
                   c=colors[class_idx], label=f'{class_name}', alpha=0.6, s=25)
    
    ax2.set_title(f'After Fusion\n({label_name} Classes)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(title='Class')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature Distribution Visualization - {label_name}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 已保存: {save_path}")


def plot_tsne_all_tasks(features_dict, labels_dict, save_path, figsize=(16, 12)):
    """
    绘制所有四个任务的 t-SNE 对比图 (2x2 布局)
    """
    task_names = ['Flow Rate', 'Feed Rate', 'Z Offset', 'Hotend Temp']
    task_keys = ['flow_rate', 'feed_rate', 'z_offset', 'hot_end']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (task_name, task_key, ax) in enumerate(zip(task_names, task_keys, axes)):
        print(f"  生成 {task_name} 的 t-SNE 图...")
        
        # 运行 t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embedded = tsne.fit_transform(features_dict['fused'])
        
        labels = labels_dict[task_key]
        class_names = ['Low', 'Normal', 'High']
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        
        for class_idx, class_name in enumerate(class_names):
            mask = labels == class_idx
            if mask.sum() > 0:
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=colors[class_idx], label=f'{class_name}', alpha=0.6, s=25)
        
        ax.set_title(task_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.legend(title='Class', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('t-SNE Feature Distribution by Task', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 已保存综合图: {save_path}")


def visualize_attention_maps(model, dataset, sample_indices, device, output_dir):
    """
    可视化指定样本的 Attention Map
    
    Args:
        model: 模型
        dataset: 数据集
        sample_indices: 样本索引列表
        device: 设备
        output_dir: 输出目录
    """
    model.eval()
    output_dir = Path(output_dir)
    
    # 定义特征源名称和对应的颜色
    source_names = ['Local (IDS)'] * 49 + ['RGB'] * 49 + ['Thermal'] * 49
    source_colors = ['coral'] * 49 + ['royalblue'] * 49 + ['gold'] * 49
    
    print(f"\n生成 Attention Map 可视化...")
    
    for idx in tqdm(sample_indices, desc="[Attention Map]"):
        # 获取样本
        sample = dataset[idx]
        
        # 准备输入数据
        batch_data = {
            'ids': sample['ids'].unsqueeze(0).to(device),
            'computer': sample['computer'].unsqueeze(0).to(device),
            'fotric': sample['fotric'].unsqueeze(0).to(device),
            'thermal': sample['thermal'].unsqueeze(0).to(device),
            'params': sample['params'].unsqueeze(0).to(device),
        }
        
        # 获取标签
        labels = sample['labels'].numpy()
        task_names = ['Flow', 'Feed', 'Z', 'Hotend']
        label_strs = [f"{task_names[i]}:{['L','N','H'][labels[i]]}" for i in range(4)]
        
        with torch.no_grad():
            features = model.extract_features(batch_data, return_attention=True)
            attn_weights = features['attn_weights']  # (1, 1, 147) 或 None (ConcatFusion)
            outputs = features['outputs']
        
        # 预测结果
        preds = {k: torch.argmax(v, dim=1).item() for k, v in outputs.items()}
        pred_strs = [f"{task_names[i]}:{['L','N','H'][preds[['flow_rate','feed_rate','z_offset','hot_end'][i]]]}" 
                     for i in range(4)]
        
        # 检查是否有 Attention Weights (ConcatFusion 返回 None)
        if attn_weights is None:
            # 对于 ConcatFusion 变体，跳过 Attention Map 可视化
            # 只保存输入图像和预测信息
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # IDS 图像
            ids_img = sample['ids'].permute(1, 2, 0).numpy()
            ids_img = ids_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            ids_img = np.clip(ids_img, 0, 1)
            axes[0].imshow(ids_img)
            axes[0].set_title('IDS Camera', fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            # RGB 图像
            rgb_img = sample['computer'].permute(1, 2, 0).numpy()
            rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_img = np.clip(rgb_img, 0, 1)
            axes[1].imshow(rgb_img)
            axes[1].set_title('RGB Camera', fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            # Thermal 图像
            thermal_img = sample['thermal'][0].numpy()
            thermal_img = thermal_img * 0.5 + 0.5
            axes[2].imshow(thermal_img, cmap='hot')
            axes[2].set_title('Thermal', fontsize=10, fontweight='bold')
            axes[2].axis('off')
            
            # 文本信息
            axes[3].text(0.1, 0.7, f'Sample {idx}\n\nTrue:\n' + '\n'.join(label_strs) + 
                        '\n\nPred:\n' + '\n'.join(pred_strs) + 
                        '\n\n(ConcatFusion:\nNo Attention)', 
                        fontsize=12, verticalalignment='top', family='monospace')
            axes[3].set_xlim(0, 1)
            axes[3].set_ylim(0, 1)
            axes[3].axis('off')
            
            plt.suptitle(f'ConcatFusion - Sample {idx} (No Attention Mechanism)', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            save_path = output_dir / f'sample_{idx}_no_attention.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            continue
        
        # 提取注意力权重 (147,)
        attn = attn_weights[0, 0].cpu().numpy()
        
        # 创建可视化图
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 显示原始图像
        # IDS 图像 (448x448)
        ax_ids = fig.add_subplot(gs[0, 0])
        ids_img = sample['ids'].permute(1, 2, 0).numpy()
        ids_img = ids_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ids_img = np.clip(ids_img, 0, 1)
        ax_ids.imshow(ids_img)
        ax_ids.set_title('IDS Camera (448×448)', fontsize=10, fontweight='bold')
        ax_ids.axis('off')
        
        # RGB 图像 (224x224)
        ax_rgb = fig.add_subplot(gs[0, 1])
        rgb_img = sample['computer'].permute(1, 2, 0).numpy()
        rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_img = np.clip(rgb_img, 0, 1)
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title('RGB Camera (224×224)', fontsize=10, fontweight='bold')
        ax_rgb.axis('off')
        
        # 热成像 (224x224)
        ax_thermal = fig.add_subplot(gs[0, 2])
        thermal_img = sample['thermal'][0].numpy()
        thermal_img = thermal_img * 0.5 + 0.5  # 反标准化
        ax_thermal.imshow(thermal_img, cmap='hot')
        ax_thermal.set_title('Thermal (224×224)', fontsize=10, fontweight='bold')
        ax_thermal.axis('off')
        
        # 注意力权重条形图
        ax_bar = fig.add_subplot(gs[1, :])
        x_pos = np.arange(147)
        bars = ax_bar.bar(x_pos, attn, color=source_colors, alpha=0.7, edgecolor='none')
        ax_bar.axvline(x=49, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax_bar.axvline(x=98, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax_bar.set_xlabel('Spatial Position Index', fontsize=11)
        ax_bar.set_ylabel('Attention Weight', fontsize=11)
        ax_bar.set_title('Cross-Attention Weights across Visual Sources', fontsize=12, fontweight='bold')
        ax_bar.set_xlim(0, 147)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='coral', alpha=0.7, label='Local (IDS)'),
            Patch(facecolor='royalblue', alpha=0.7, label='RGB'),
            Patch(facecolor='gold', alpha=0.7, label='Thermal')
        ]
        ax_bar.legend(handles=legend_elements, loc='upper right')
        
        # 空间注意力热力图 (7x7 布局)
        # Local
        ax_local_heatmap = fig.add_subplot(gs[2, 0])
        local_attn = attn[:49].reshape(7, 7)
        sns.heatmap(local_attn, ax=ax_local_heatmap, cmap='Reds', 
                   cbar=True, square=True, xticklabels=False, yticklabels=False)
        ax_local_heatmap.set_title('Local (IDS) Attention\n(7×7 spatial)', fontsize=10, fontweight='bold')
        
        # RGB
        ax_rgb_heatmap = fig.add_subplot(gs[2, 1])
        rgb_attn = attn[49:98].reshape(7, 7)
        sns.heatmap(rgb_attn, ax=ax_rgb_heatmap, cmap='Blues',
                   cbar=True, square=True, xticklabels=False, yticklabels=False)
        ax_rgb_heatmap.set_title('RGB Attention\n(7×7 spatial)', fontsize=10, fontweight='bold')
        
        # Thermal
        ax_thermal_heatmap = fig.add_subplot(gs[2, 2])
        thermal_attn = attn[98:].reshape(7, 7)
        sns.heatmap(thermal_attn, ax=ax_thermal_heatmap, cmap='YlOrBr',
                   cbar=True, square=True, xticklabels=False, yticklabels=False)
        ax_thermal_heatmap.set_title('Thermal Attention\n(7×7 spatial)', fontsize=10, fontweight='bold')
        
        # 主标题
        plt.suptitle(f'Sample {idx}\nTrue: {" | ".join(label_strs)}\nPred: {" | ".join(pred_strs)}',
                    fontsize=12, fontweight='bold', y=0.98)
        
        # 保存
        save_path = output_dir / f'attention_sample_{idx}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
    print(f"✓ 已保存 {len(sample_indices)} 个 Attention Map 到 {output_dir}")


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 加载模型
    model, scaler = load_model_and_scaler(args.model_path, device)
    
    # 创建输出目录（根据模型所在子目录命名）
    # 例如：saved_models/full/model_Full.pt -> visualization/full/
    model_subdir = Path(args.model_path).parent.name
    output_dir = Path(args.output_dir) / model_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")
    
    # 加载数据集
    print(f"\n加载数据集...")
    
    if args.mode in ['tsne', 'all']:
        # t-SNE 使用测试集
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
            num_workers=args.num_workers
        )
        
        # 提取特征
        features_dict, labels_dict = extract_features_for_tsne(
            model, test_loader, device, max_samples=args.max_samples
        )
        
        # 生成各任务的 t-SNE 图
        task_names = ['Flow_Rate', 'Feed_Rate', 'Z_Offset', 'Hotend_Temp']
        task_keys = ['flow_rate', 'feed_rate', 'z_offset', 'hot_end']
        
        print(f"\n生成 t-SNE 可视化...")
        for task_name, task_key in zip(task_names, task_keys):
            save_path = output_dir / f'tsne_{task_name}.png'
            plot_tsne_dual_features(
                features_dict, 
                labels_dict[task_key], 
                task_name.replace('_', ' '),
                save_path
            )
        
        # 生成综合图
        combined_path = output_dir / 'tsne_all_tasks.png'
        plot_tsne_all_tasks(features_dict, labels_dict, combined_path)
        
        print(f"\n✓ t-SNE 可视化完成，保存到: {output_dir}")
    
    if args.mode in ['attention', 'all']:
        # Attention Map 可视化
        # 使用完整数据集以便选择特定样本
        full_dataset = FDMDefectDataset(
            csv_path=args.test_csv,
            data_root=args.data_root,
            is_train=False,
            parameter_scaler=scaler
        )
        
        # 选择样本索引
        if args.sample_indices:
            indices = [int(x) for x in args.sample_indices.split(',')]
        else:
            # 随机选择 N 个样本
            np.random.seed(42)
            indices = np.random.choice(len(full_dataset), args.num_attention_samples, replace=False).tolist()
        
        # 创建 attention_maps 子目录
        attention_dir = output_dir / 'attention_maps'
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        visualize_attention_maps(
            model, full_dataset, indices, device, 
            attention_dir
        )
        
        print(f"\n✓ Attention Map 可视化完成")
    
    print(f"\n{'='*60}")
    print(f"所有可视化结果已保存到: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PACF-NET 可视化脚本')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--test_csv', type=str,
                       default=r'D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop\test.csv',
                       help='测试集 CSV 路径')
    parser.add_argument('--data_root', type=str,
                       default=r'D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop',
                       help='数据根目录')
    
    # 可视化模式
    parser.add_argument('--mode', type=str, default='all', choices=['tsne', 'attention', 'all'],
                       help='可视化模式: tsne=特征分布, attention=注意力图, all=全部')
    
    # t-SNE 设置
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='t-SNE 最大样本数')
    
    # Attention Map 设置
    parser.add_argument('--sample_indices', type=str, default=None,
                       help='指定样本索引，如 "10,25,100" (默认随机选择)')
    parser.add_argument('--num_attention_samples', type=int, default=5,
                       help='随机选择 Attention Map 样本数')
    
    # 通用设置
    parser.add_argument('--output_dir', type=str, default='visualization',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    args = parser.parse_args()
    main(args)
