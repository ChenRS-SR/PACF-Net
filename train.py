import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

from model import PACFNet
from dataset import create_data_loaders


class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理类别不均衡问题
    在验证集标签分布偏离训练集时特别有效
    
    公式: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    gamma: focusing parameter，越大越关注困难样本
    alpha: 平衡参数
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) 模型输出的 logits
            targets: (N,) 真实标签
        """
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # 正确类别的概率
        focal_loss = (1 - pt) ** self.gamma * ce
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    防止模型对训练集过拟合，提高泛化能力
    
    Args:
        smoothing: 平滑系数 (0.0 ~ 1.0)，通常设为 0.1
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) 模型输出的 logits
            targets: (N,) 真实标签
        """
        num_classes = logits.size(1)
        # 将 hard label 转换为 soft label
        # 正确类别: 1 - smoothing + smoothing/num_classes
        # 错误类别: smoothing/num_classes
        confidence = 1.0 - self.smoothing
        smooth_label = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), confidence)
        smooth_label += self.smoothing / num_classes
        
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_label * log_probs).sum(dim=1).mean()
        return loss


def prepare_batch_data(batch, device):
    """
    将 DataLoader 返回的字典转换为模型需要的格式
    
    Args:
        batch: DataLoader 返回的字典，包含 {computer, ids, fotric, thermal, params, labels, task_id}
        device: 计算设备
    
    Returns:
        batch_data: 模型输入字典 {ids, computer, fotric, thermal, params}
        labels_dict: 标签字典 {flow_rate, feed_rate, z_offset, hot_end}
    """
    # 准备 batch_data (模型的第一个输入)
    batch_data = {
        'ids': batch['ids'].to(device),
        'computer': batch['computer'].to(device),
        'fotric': batch['fotric'].to(device),
        'thermal': batch['thermal'].to(device),
        'params': batch['params'].to(device),
    }
    
    # 准备 labels_dict (模型的第二个输入)
    # dataset 返回的 labels 是 (B, 4) 张量，顺序为 [flow_rate, feed_rate, z_offset, hotend]
    labels_tensor = batch['labels'].to(device)  # (B, 4)
    labels_dict = {
        'flow_rate': labels_tensor[:, 0],
        'feed_rate': labels_tensor[:, 1],
        'z_offset': labels_tensor[:, 2],
        'hot_end': labels_tensor[:, 3],
    }
    
    return batch_data, labels_dict


def calculate_accuracy(outputs, labels_dict):
    """
    计算四个任务的准确率
    
    Args:
        outputs: 模型输出 {flow_rate, feed_rate, z_offset, hot_end}
        labels_dict: 标签字典
    
    Returns:
        dict: 包含四个任务的准确率和平均值
              {'flow_rate': ..., 'feed_rate': ..., 'z_offset': ..., 'hot_end': ..., 'mean': ...}
    """
    flow_acc = (outputs['flow_rate'].argmax(1) == labels_dict['flow_rate']).float().mean()
    feed_acc = (outputs['feed_rate'].argmax(1) == labels_dict['feed_rate']).float().mean()
    z_acc = (outputs['z_offset'].argmax(1) == labels_dict['z_offset']).float().mean()
    hotend_acc = (outputs['hot_end'].argmax(1) == labels_dict['hot_end']).float().mean()
    
    return {
        'flow_rate': flow_acc.item(),
        'feed_rate': feed_acc.item(),
        'z_offset': z_acc.item(),
        'hot_end': hotend_acc.item(),
        'mean': (flow_acc + feed_acc + z_acc + hotend_acc).item() / 4
    }


def inference(model, data_loader, device):
    """
    推理模式 - 仅输出预测结果，不计算损失
    用于测试集或部署场景
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        all_outputs: 所有预测结果列表
    """
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[推理]", ncols=80):
            batch_data, _ = prepare_batch_data(batch, device)  # 不需要 labels
            
            # 推理模式：仅返回 outputs
            outputs = model(batch_data, labels=None)
            all_outputs.append(outputs)
    
    return all_outputs


def train_epoch(model, train_loader, optimizer, device, epoch=0, total_epochs=100, 
                use_focal=False, use_label_smoothing=True, alpha_mmd=1.0):
    """
    训练一个 epoch，支持 MMD Warmup 策略
    
    MMD Warmup: 前 5 个 epoch 只训练 MMD（冻结分类头），强制特征先对齐
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_mmd_loss = 0.0
    num_batches = 0
    
    # MMD Warmup 策略：前 5 个 epoch 只训练 MMD Loss，冻结分类头
    # 注意：变体A (no_mmd) alpha_mmd=0，跳过 warmup
    is_warmup = (epoch < 5) and (alpha_mmd > 0)
    if is_warmup:
        # 冻结分类头
        for param in model.task_head.parameters():
            param.requires_grad = False
    else:
        # 解冻分类头
        for param in model.task_head.parameters():
            param.requires_grad = True
    
    # 针对不同任务的类别权重（基于训练集分布计算，并针对验证集分布偏移调整）
    # 权重 = 中位数频率 / 类别频率，使少数类获得更高权重
    # Flow: [21%, 61%, 18%] -> 权重 [2.9, 1.0, 3.4]
    # Feed: [24%, 54%, 23%] -> 权重 [2.3, 1.0, 2.4]  
    #       注意：验证集"偏高"仅7.1%，需特别加强 -> 提高权重到5.0
    # Z: [7%, 72%, 22%] -> 权重 [10.3, 1.0, 3.3]
    # Hotend: [7%, 74%, 19%] -> 权重 [10.6, 1.0, 3.9]
    flow_weights = torch.tensor([2.9, 1.0, 3.4], device=device)
    feed_weights = torch.tensor([2.3, 1.0, 5.0], device=device)  # 加强"偏高"类，应对验证集偏移
    z_weights = torch.tensor([10.3, 1.0, 3.3], device=device)
    hotend_weights = torch.tensor([10.6, 1.0, 3.9], device=device)
    
    flow_loss_fn = nn.CrossEntropyLoss(weight=flow_weights, label_smoothing=0.1 if use_label_smoothing else 0.0)
    feed_loss_fn = nn.CrossEntropyLoss(weight=feed_weights, label_smoothing=0.1 if use_label_smoothing else 0.0)
    z_loss_fn = nn.CrossEntropyLoss(weight=z_weights, label_smoothing=0.1 if use_label_smoothing else 0.0)
    hotend_loss_fn = nn.CrossEntropyLoss(weight=hotend_weights, label_smoothing=0.1 if use_label_smoothing else 0.0)
    
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25) if use_focal else None
    
    pbar = tqdm(train_loader, desc="[训练]", ncols=100)
    for batch in pbar:
        batch_data, labels_dict = prepare_batch_data(batch, device)
        
        optimizer.zero_grad()
        outputs, loss_dict, _ = model(batch_data, labels_dict)
        
        if is_warmup:
            # MMD Warmup: 只训练 MMD Loss
            mmd_loss = loss_dict['mmd_loss']
            # 检查 MMD 是否为 NaN
            if torch.isnan(mmd_loss):
                print(f"[Epoch {epoch+1}] Warning: MMD is NaN, skipping batch")
                continue
            loss = mmd_loss * alpha_mmd
            ce_loss = 0.0
        else:
            # 正常训练：CE + MMD
            ce_loss = 0.0
            ce_loss += flow_loss_fn(outputs['flow_rate'], labels_dict['flow_rate'])
            ce_loss += feed_loss_fn(outputs['feed_rate'], labels_dict['feed_rate'])
            ce_loss += z_loss_fn(outputs['z_offset'], labels_dict['z_offset'])
            ce_loss += hotend_loss_fn(outputs['hot_end'], labels_dict['hot_end'])
            
            mmd_loss = loss_dict['mmd_loss']
            # 检查损失是否为 NaN
            if torch.isnan(ce_loss) or torch.isnan(mmd_loss):
                print(f"[Epoch {epoch+1}] Warning: CE={ce_loss.item()}, MMD={mmd_loss.item()}, skipping batch")
                continue
            loss = ce_loss + alpha_mmd * mmd_loss
        
        # 检查总损失是否为 NaN
        if torch.isnan(loss):
            print(f"[Epoch {epoch+1}] Warning: Total loss is NaN, skipping batch")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += ce_loss if isinstance(ce_loss, (int, float)) else ce_loss.item()
        total_mmd_loss += loss_dict['mmd_loss'].item()
        num_batches += 1
        
        # 实时显示
        if is_warmup:
            pbar.set_postfix({
                'mode': 'WARMUP',
                'mmd': f'{loss_dict["mmd_loss"].item():.4f}'
            })
        else:
            pbar.set_postfix({
                'mode': 'NORMAL',
                'total': f'{loss.item():.4f}',
                'ce': f'{(ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss):.4f}',
                'mmd': f'{loss_dict["mmd_loss"].item():.4f}'
            })
    
    return {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'mmd_loss': total_mmd_loss / num_batches,
        'is_warmup': is_warmup,
    }


def validate(model, val_loader, device, use_focal=False, epoch=0, alpha_mmd=1.0):
    """
    验证一个 epoch
    
    在 MMD Warmup 阶段（epoch < 5），不计算分类准确率
    """
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_mmd_loss = 0.0
    total_accs = {'flow_rate': 0.0, 'feed_rate': 0.0, 'z_offset': 0.0, 'hot_end': 0.0, 'mean': 0.0}
    num_batches = 0
    
    is_warmup = (epoch < 5)
    
    # 验证时使用标准 CrossEntropy（不加权重，公平评估）
    ce_loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[验证]", ncols=100)
        for batch in pbar:
            batch_data, labels_dict = prepare_batch_data(batch, device)
            
            # 模型返回 (outputs, loss_dict, intermediate_features)
            outputs, loss_dict, _ = model(batch_data, labels_dict)
            
            if is_warmup:
                # Warmup 阶段：只看 MMD Loss
                ce_loss = 0.0
                mmd_loss = loss_dict['mmd_loss']
                # 如果 MMD 是 NaN，用 0 代替
                if torch.isnan(mmd_loss):
                    mmd_loss = torch.tensor(0.0, device=device)
                loss = mmd_loss * alpha_mmd
            else:
                # 正常阶段：计算标准 CE（不加权重，公平评估）+ MMD
                ce_loss = 0.0
                ce_loss += ce_loss_fn(outputs['flow_rate'], labels_dict['flow_rate'])
                ce_loss += ce_loss_fn(outputs['feed_rate'], labels_dict['feed_rate'])
                ce_loss += ce_loss_fn(outputs['z_offset'], labels_dict['z_offset'])
                ce_loss += ce_loss_fn(outputs['hot_end'], labels_dict['hot_end'])
                
                mmd_loss = loss_dict['mmd_loss']
                # 如果 MMD 是 NaN，用 0 代替
                if torch.isnan(mmd_loss):
                    mmd_loss = torch.tensor(0.0, device=device)
                loss = ce_loss + alpha_mmd * mmd_loss
            
            # 计算四个任务的准确率
            acc_dict = calculate_accuracy(outputs, labels_dict)
            
            total_loss += loss.item()
            total_ce_loss += (ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss)
            total_mmd_loss += loss_dict['mmd_loss'].item()
            for key in total_accs:
                total_accs[key] += acc_dict[key]
            num_batches += 1
            
            # 实时显示当前 batch 的指标
            if is_warmup:
                pbar.set_postfix({
                    'mode': 'WARMUP',
                    'mmd': f'{loss_dict["mmd_loss"].item():.4f}',
                })
            else:
                pbar.set_postfix({
                    'mode': 'NORMAL',
                    'total': f'{loss.item():.4f}',
                    'ce': f'{(ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss):.4f}',
                    'mmd': f'{loss_dict["mmd_loss"].item():.4f}',
                    'acc': f'{acc_dict["mean"]:.4f}'
                })
    
    avg_accs = {k: v / num_batches for k, v in total_accs.items()}
    
    return {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'mmd_loss': total_mmd_loss / num_batches,
        'accs': avg_accs,
        'is_warmup': is_warmup
    }


def select_variant():
    """
    交互式选择消融实验变体
    
    Returns:
        variant: 变体名称
        model_name: 保存时的模型文件名
        alpha_mmd: MMD损失权重
    """
    print("\n" + "="*70)
    print("请选择要训练的模型变体 (Ablation Study)")
    print("="*70)
    print("1. Full Model (完整模型)                -> model_full.pt")
    print("2. Variant A - No MMD (无MMD)           -> model_no_mmd.pt")
    print("3. Variant B - Concat Only (拼接)       -> model_concat.pt")
    print("4. Variant C-1 - RGB+IDS (无热成像)     -> model_rgb_only.pt")
    print("5. Variant C-2 - IDS Only (单模态)      -> model_ids_only.pt")
    print("="*70)
    
    while True:
        try:
            choice = input("请输入选项 (1-5): ").strip()
            if choice == '1':
                return 'full', 'model_full.pt', None  # None表示使用args.mmd_weight
            elif choice == '2':
                return 'no_mmd', 'model_no_mmd.pt', 0.0
            elif choice == '3':
                return 'concat_only', 'model_concat.pt', None
            elif choice == '4':
                return 'rgb_only', 'model_rgb_only.pt', None
            elif choice == '5':
                return 'ids_only', 'model_ids_only.pt', None
            else:
                print("❌ 无效选项，请重新输入 (1-5)")
        except KeyboardInterrupt:
            print("\n用户取消")
            exit(0)


def main(args):
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 选择消融实验变体
    variant, model_filename, variant_mmd_weight = select_variant()
    
    # 确定MMD权重 (变体A强制为0，其他使用命令行参数或默认值)
    if variant_mmd_weight is not None:
        effective_mmd_weight = variant_mmd_weight
    else:
        effective_mmd_weight = args.mmd_weight
    
    # 初始化模型 (传入变体参数)
    model = PACFNet(variant=variant).to(device)
    print(f"✓ 模型已加载到 {device}")
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"✓ 优化器: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  注：已针对分布偏移和过拟合调整超参数")
    print(f"      - 学习率提高到 {args.lr} (加速收敛)")
    print(f"      - 权重衰减增加到 {args.weight_decay} (更强正则化)")
    print(f"      - MMD 权重: {effective_mmd_weight} {'(变体A: 禁用MMD)' if variant == 'no_mmd' else '(特征对齐)'}")
    
    # 创建以时间戳命名的模型保存文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在变体名称前加上标识
    variant_prefix = variant.replace('_', '-')
    save_dir = Path('saved_models') / f"{variant_prefix}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = save_dir / model_filename
    scaler_path = save_dir / 'scaler.pkl'
    log_path = save_dir / 'train.log'  # 训练日志文件
    
    print(f"✓ 模型保存目录: {save_dir}")
    print(f"✓ 模型文件名: {model_filename}")
    print(f"✓ 日志文件: {log_path}")
    
    # 初始化日志文件，记录训练配置
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write(f"训练配置\n")
        f.write(f"{'='*100}\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"变体: {variant}\n")
        f.write(f"模型文件: {model_filename}\n")
        f.write(f"Data Root: {args.data_root}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"MMD Weight: {effective_mmd_weight}\n")
        f.write(f"Num Workers: {args.num_workers}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\n{'='*100}\n")
        f.write(f"训练日志\n")
        f.write(f"{'='*100}\n\n")
    
    # 创建数据加载器
    print(f"\n加载数据集...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler_save_path=str(scaler_path)
    )
    
    # 诊断：检查标签分布
    print(f"\n{'='*100}")
    print(f"📊 标签分布诊断与数据漂移检测 [变体: {variant}]")
    print(f"{'='*100}")
    print("\n⚠️  验证集存在分布偏移 (尤其速度 Feed)：[偏低 17.5%, 正常 75.4%, 偏高 7.1%]")
    print("   训练集分布: [偏低 23.7%, 正常 53.8%, 偏高 22.5%]")
    print("   '偏高'类在验证集严重欠采样，模型难以学习")
    print("\n改进措施：")
    print(f"  ✓ 学习率: {args.lr} (适当提高加速收敛)")
    print(f"  ✓ 权重衰减: {args.weight_decay} (强正则化抑制过拟合)")
    if variant == 'no_mmd':
        print(f"  ✓ MMD 权重: {effective_mmd_weight} (变体A: 禁用MMD)")
    elif variant == 'concat_only':
        print(f"  ✓ MMD 权重: {effective_mmd_weight} (变体B: 拼接融合)")
    elif variant in ['rgb_only', 'ids_only']:
        print(f"  ✓ MMD 权重: {effective_mmd_weight} (变体C: 单模态)")
    else:
        print(f"  ✓ MMD 权重: {effective_mmd_weight} (加强跨域特征对齐)")
    print(f"  ✓ Dropout: 0.5 (增强正则化)")
    print(f"  ✓ 早停耐心: {args.patience} (给更多收敛时间)")
    print(f"{'='*100}\n")
    
    # 训练循环
    print(f"\n{'='*100}")
    print(f"开始训练: {args.epochs} 个 epoch [变体: {variant}]")
    if variant == 'no_mmd':
        print(f"  - Warmup 阶段: 禁用 (变体A无MMD)")
        print(f"  - 正常阶段: 只训练 CE (无MMD)")
    else:
        print(f"  - Warmup 阶段 (epoch 1-5): 只训练 MMD，冻结分类头")
        print(f"  - 正常阶段 (epoch 6+): 训练 CE + MMD，保存最佳模型")
    print(f"  - MMD 权重: {effective_mmd_weight}")
    print(f"  - 早停耐心值: {args.patience} 个 epoch")
    print(f"  - 改进: 空间注意力机制 + 强正则化 (dropout=0.5)")
    print(f"{'='*100}\n")
    
    # 早停相关变量
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in tqdm(range(args.epochs), desc="[Epoch]", ncols=100):
        # 训练（支持 MMD Warmup，变体A禁用MMD）
        use_focal = (epoch >= 5)  # 从第6个epoch开始使用focal loss
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, 
            epoch=epoch, total_epochs=args.epochs, 
            use_focal=use_focal, use_label_smoothing=True,
            alpha_mmd=effective_mmd_weight
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, device, 
            use_focal=use_focal, epoch=epoch,
            alpha_mmd=effective_mmd_weight
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型（只在非 warmup 阶段，基于平均准确率）
        flag = ""
        is_best = False
        if not val_metrics['is_warmup']:
            current_acc = val_metrics['accs']['mean']
            
            if epoch == 5:  # 第一个正常 epoch
                best_val_acc = current_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                flag = "✓ [首个]"
                is_best = True
            elif current_acc > best_val_acc:  # 准确率越高越好
                best_val_acc = current_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                flag = "✓ [最佳]"
                patience_counter = 0  # 重置早停计数器
                is_best = True
            else:
                patience_counter += 1
                flag = f"[{patience_counter}/{args.patience}]"
        
        # 详细打印四个参数的准确率和损失分量
        warmup_tag = "[WARMUP]" if train_metrics['is_warmup'] else "[NORMAL]"
        acc_str = (f"Flow:{val_metrics['accs']['flow_rate']:.4f} "
                  f"Feed:{val_metrics['accs']['feed_rate']:.4f} "
                  f"Z:{val_metrics['accs']['z_offset']:.4f} "
                  f"Hotend:{val_metrics['accs']['hot_end']:.4f}")
        
        # Warmup 阶段不显示准确率，因为分类头未训练
        if val_metrics['is_warmup']:
            log_line = (f"Epoch {epoch+1:3d}/{args.epochs} {warmup_tag} | "
                       f"TrMMD:{train_metrics['mmd_loss']:.4f} | "
                       f"VaMMD:{val_metrics['mmd_loss']:.4f} | "
                       f"(分类头冻结中...)")
        else:
            log_line = (f"Epoch {epoch+1:3d}/{args.epochs} {warmup_tag} | "
                       f"TrCE:{train_metrics['ce_loss']:.4f} TrMMD:{train_metrics['mmd_loss']:.4f} | "
                       f"VaCE:{val_metrics['ce_loss']:.4f} VaMMD:{val_metrics['mmd_loss']:.4f} | "
                       f"Acc[{acc_str}] {flag} | LR:{current_lr:.2e}")
        
        # 同时打印到控制台和日志文件
        print(log_line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
        
        # 早停检查
        if not val_metrics['is_warmup'] and patience_counter >= args.patience:
            print(f"\n⚠️  早停触发！连续 {args.patience} 个 epoch 验证准确率未提升")
            print(f"   最佳准确率: {best_val_acc:.4f} @ Epoch {best_epoch+1}")
            break
    
    print(f"\n{'='*100}")
    print(f"训练完成！最佳模型已保存至: {best_model_path}")
    print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch+1})")
    print(f"{'='*100}")
    
    # 在日志中记录训练完成
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"训练完成！最佳模型已保存至: {best_model_path}\n")
        f.write(f"最佳验证准确率: {best_val_acc:.6f} (Epoch {best_epoch+1})\n")
        if patience_counter >= args.patience:
            f.write(f"早停触发: 连续 {args.patience} 个 epoch 未提升\n")
        f.write(f"{'='*100}\n")
    
    # 在测试集上进行推理
    print(f"\n{'='*100}")
    print(f"在测试集上进行推理...")
    print(f"{'='*100}\n")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 推理模式 (仅输出预测结果，不计算损失)
    test_outputs = inference(model, test_loader, device)
    print(f"✓ 推理完成！共处理 {len(test_loader)} 个 batch")
    print(f"  每个 batch 返回 4 个任务的预测: {{flow_rate, feed_rate, z_offset, hot_end}}")
    print(f"\n第一个 batch 的输出形状:")
    for key, val in test_outputs[0].items():
        print(f"  {key}: {val.shape}")
    
    # 在日志中记录推理完成
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"推理完成！共处理 {len(test_loader)} 个 batch\n")
        f.write(f"每个 batch 返回 4 个任务的预测: {{flow_rate, feed_rate, z_offset, hot_end}}\n")
        f.write(f"\n第一个 batch 的输出形状:\n")
        for key, val in test_outputs[0].items():
            f.write(f"  {key}: {val.shape}\n")
        f.write(f"\n保存位置: {log_path.parent}\n")
        f.write(f"{'='*100}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PACF-NET 训练脚本')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, 
                       default=r'D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop',
                       help='数据集根目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练 epoch 数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率 (提高到1e-4加速收敛)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减 (增加到5e-4抑制过拟合)')
    parser.add_argument('--mmd_weight', type=float, default=2.0, help='MMD 损失权重 (提高到2.0加强特征对齐)')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值 (增加到20，给模型更多收敛时间)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作线程数')
    
    args = parser.parse_args()
    
    main(args)