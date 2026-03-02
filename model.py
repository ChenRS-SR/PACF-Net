import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F


class VisualBackbone(nn.Module):
    """
    多源异构视觉骨干网络 (Multi-Source Visual Backbone)
    
    包含三个并行的ResNet-18塔结构，分别处理不同的视觉输入：
    
    1. 局部纹理塔 (Local Texture Stream)
       - 输入：IDS 随轴工业相机，448*448*3 (RGB)
       - 特征图尺寸：7×7 (通过 AdaptiveAvgPool2d 统一)
       - 作用：捕捉喷嘴尖端的微米级缺陷，高分辨率特性提供清晰的本征特征
       - 输出：F_local (B, 512, 7, 7) 空间特征 / (B, 512) 全局特征
    
    2. 环境感知塔 (Environmental Context Stream)
       - RGB通道：旁轴相机，224*224*3 (RGB)
       - 特征图尺寸：7×7 (原生 ResNet18 输出)
       - 作用：提供宏观几何形貌和宽的背景视野
       - 输出：F_rgb (B, 512, 7, 7) 空间特征 / (B, 512) 全局特征
    
    3. 热成像塔 (Thermal Stream)
       - 输入：伪彩色热像 (3通道) + 灰度温度矩阵 (1通道) = 224*224*4
       - 特征图尺寸：7×7 (原生 ResNet18 输出)
       - 作用：提供熔池温度场与冷却趋势信息
       - 输出：F_thermal (B, 512, 7, 7) 空间特征 / (B, 512) 全局特征
    
    重要改进：
    - 所有塔在最后都添加 AdaptiveAvgPool2d((7, 7))，统一空间尺寸
    - IDS 原本输出 14×14，现缩放到 7×7（与 RGB/IR 保持一致）
    - 这样可以直接做点对点的 Attention 或 MMD 对齐
    """
    
    def __init__(self):
        super().__init__()
        
        # 1. 局部纹理塔 - 处理IDS随轴相机 (448×448×3)
        self.local_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.local_backbone = nn.Sequential(*list(self.local_backbone.children())[:-1])
        # 统一特征图尺寸到 7×7 (IDS 原本是 14×14，特殊处理)
        self.local_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 2. RGB环境感知塔 - 处理旁轴RGB (224×224×3)
        self.rgb_context_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.rgb_context_backbone = nn.Sequential(*list(self.rgb_context_backbone.children())[:-1])
        # RGB 原本输出 7×7，这里保持一致
        self.rgb_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 3. 热成像塔 - 处理伪彩色+灰度热像 (224×224×4)
        self.thermal_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # 修改第一层卷积以适配4通道输入
        self.thermal_backbone.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.thermal_backbone = nn.Sequential(*list(self.thermal_backbone.children())[:-1])
        # IR 原本输出 7×7，这里保持一致
        self.thermal_pool = nn.AdaptiveAvgPool2d((7, 7))
    
    def forward(self, ids, computer, fotric, thermal):
        """
        Args:
            ids: IDS随轴相机图像 (B, 3, 448, 448)
            computer: 旁轴RGB相机图像 (B, 3, 224, 224)
            fotric: 伪彩色热像 (B, 3, 224, 224)
            thermal: 灰度热像 (B, 1, 224, 224)
        
        Returns:
            dict 包含：
            - 'f_local_spatial': 局部纹理空间特征 (B, 512, 7, 7)
            - 'f_rgb_spatial': RGB环境空间特征 (B, 512, 7, 7)
            - 'f_thermal_spatial': 热成像空间特征 (B, 512, 7, 7)
            - 'f_local': 局部纹理全局特征 (B, 512)
            - 'f_rgb': RGB环境全局特征 (B, 512)
            - 'f_thermal': 热成像全局特征 (B, 512)
        """
        # 1. 局部纹理特征提取
        ids_feat = self.local_backbone(ids)  # (B, 512, 14, 14)
        ids_feat = self.local_pool(ids_feat)  # (B, 512, 7, 7) - 统一尺寸
        f_local_spatial = ids_feat  # 保存空间特征用于 MMD/Attention
        f_local = ids_feat.mean(dim=(2, 3))  # (B, 512) - 全局平均池化
        
        # 2. RGB环境特征提取
        rgb_feat = self.rgb_context_backbone(computer)  # (B, 512, 7, 7)
        rgb_feat = self.rgb_pool(rgb_feat)  # (B, 512, 7, 7)
        f_rgb_spatial = rgb_feat
        f_rgb = rgb_feat.mean(dim=(2, 3))  # (B, 512) - 全局平均池化
        
        # 3. 热成像特征提取 - 堆叠伪彩色和灰度
        thermal_combined = torch.cat([fotric, thermal], dim=1)  # (B, 4, 224, 224)
        thermal_feat = self.thermal_backbone(thermal_combined)  # (B, 512, 7, 7)
        thermal_feat = self.thermal_pool(thermal_feat)  # (B, 512, 7, 7)
        f_thermal_spatial = thermal_feat
        f_thermal = thermal_feat.mean(dim=(2, 3))  # (B, 512) - 全局平均池化
        
        return {
            'f_local_spatial': f_local_spatial,
            'f_rgb_spatial': f_rgb_spatial,
            'f_thermal_spatial': f_thermal_spatial,
            'f_local': f_local,
            'f_rgb': f_rgb,
            'f_thermal': f_thermal,
        }


class ProcessIntentEmbedding(nn.Module):
    """
    工艺意图嵌入模块 (Process Intent Embedding)
    
    将10维的工艺状态向量转换为高维的语义嵌入向量（Semantic Embedding Vector）
    
    输入向量维度说明：
    - [0-2]: 空间状态 (Spatial State) - 喷头坐标 (X, Y, Z)
    - [3-6]: 控制意图 (Control Intent) - 流量、速度、Z偏移、温度
    - [7-9]: 热统计特征 (Thermal Statistics) - 最高温、最低温、平均温
    
    这个向量不仅编码了当前的打印指令，还隐含了当前的热-力学状态预期。
    """
    
    def __init__(self, input_dim=10, embed_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, params):
        """
        Args:
            params: 10维工艺状态向量 (B, 10)
        
        Returns:
            语义嵌入向量 (B, 512)
        """
        return self.mlp(params)


class CrossAttentionFusion(nn.Module):
    """
    基于交叉注意力的因果融合机制 (Cross-Attention Fusion) - 改进版
    
    改进点：
    - 使用空间特征序列 (7x7=49 tokens) 替代全局池化后的单token
    - Q: 工艺意图向量 (B, 1, 512)
    - K/V: 多源视觉特征拼接 (B, 147, 512) - 3个源 x 49个空间位置
    - 实现真正的空间注意力机制，模型可以关注特定空间位置
    
    工作原理：
    - Query (Q)：由工艺意图嵌入向量生成
        代表"当前的工艺条件下，我们预期会发生什么？"
    - Key (K) & Value (V)：由空间视觉特征生成
        代表"相机实际看到了什么？在哪个位置？"
    - 融合逻辑：网络根据工艺参数动态调整对视觉特征的空间关注权重
    """
    
    def __init__(self, intent_dim=512, visual_dim=512, num_heads=8, spatial_size=7):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_patches = spatial_size * spatial_size  # 49
        
        # 投影层：将不同源的特征投影到统一维度
        self.local_proj = nn.Conv2d(visual_dim, intent_dim, kernel_size=1)
        self.rgb_proj = nn.Conv2d(visual_dim, intent_dim, kernel_size=1)
        self.thermal_proj = nn.Conv2d(visual_dim, intent_dim, kernel_size=1)
        
        # 多头交叉注意力：意图向量作为Query，空间视觉特征作为Key/Value
        self.cross_attn = nn.MultiheadAttention(
            intent_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # 融合层
        self.fusion_proj = nn.Sequential(
            nn.Linear(intent_dim * 2, intent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.LayerNorm(intent_dim)
        )
        
        # 可学习的空间位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches * 3, intent_dim) * 0.02)
    
    def forward(self, f_local_spatial, f_rgb_spatial, f_thermal_spatial, intent_feat, return_attn=False):
        """
        Args:
            f_local_spatial: 局部纹理空间特征 (B, 512, 7, 7)
            f_rgb_spatial: RGB环境空间特征 (B, 512, 7, 7)
            f_thermal_spatial: 热成像空间特征 (B, 512, 7, 7)
            intent_feat: 工艺意图嵌入 (B, 512)
            return_attn: 是否返回注意力权重 (用于可视化)
        
        Returns:
            如果 return_attn=False:
                融合后的特征 (B, 512)
            如果 return_attn=True:
                (融合后的特征 (B, 512), 注意力权重 (B, 1, 147))
        """
        B = intent_feat.size(0)
        
        # 1. 投影并展平空间特征
        # (B, 512, 7, 7) -> (B, 512, 49) -> (B, 49, 512)
        local_tokens = self.local_proj(f_local_spatial).flatten(2).permute(0, 2, 1)
        rgb_tokens = self.rgb_proj(f_rgb_spatial).flatten(2).permute(0, 2, 1)
        thermal_tokens = self.thermal_proj(f_thermal_spatial).flatten(2).permute(0, 2, 1)
        
        # 2. 拼接三个源的空间tokens: (B, 49*3, 512) = (B, 147, 512)
        visual_tokens = torch.cat([local_tokens, rgb_tokens, thermal_tokens], dim=1)
        
        # 3. 添加位置编码
        visual_tokens = visual_tokens + self.pos_embed[:, :visual_tokens.size(1), :]
        
        # 4. 交叉注意力: Q=意图向量, K/V=视觉tokens
        q = intent_feat.unsqueeze(1)  # (B, 1, 512)
        k = v = visual_tokens  # (B, 147, 512)
        
        attn_output, attn_weights = self.cross_attn(q, k, v)  # (B, 1, 512), (B, 1, 147)
        attn_output = attn_output.squeeze(1)  # (B, 512)
        
        # 5. 融合：结合原始意图向量和注意力输出
        combined = torch.cat([intent_feat, attn_output], dim=1)  # (B, 1024)
        fused = self.fusion_proj(combined)  # (B, 512)
        
        if return_attn:
            return fused, attn_weights
        return fused


class ConcatFusion(nn.Module):
    """
    消融实验变体B：简单拼接融合 (无注意力机制)
    
    直接将三个视觉特征全局池化后拼接，再与工艺意图拼接
    用于验证注意力机制的有效性
    """
    
    def __init__(self, visual_dim=512, intent_dim=512, num_sources=3):
        super().__init__()
        self.num_sources = num_sources
        
        # 输入维度: num_sources * visual_dim (视觉) + intent_dim (工艺)
        input_dim = num_sources * visual_dim + intent_dim
        
        # 降维投影层
        self.fusion_proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.LayerNorm(512)
        )
    
    def forward(self, f_local_spatial, f_rgb_spatial, f_thermal_spatial, intent_feat, return_attn=False):
        """
        Args:
            f_local_spatial: 局部纹理空间特征 (B, 512, 7, 7)
            f_rgb_spatial: RGB环境空间特征 (B, 512, 7, 7)
            f_thermal_spatial: 热成像空间特征 (B, 512, 7, 7)
            intent_feat: 工艺意图嵌入 (B, 512)
            return_attn: 兼容性参数，始终返回None
        
        Returns:
            融合后的特征 (B, 512)
        """
        # 全局平均池化
        f_local = f_local_spatial.mean(dim=(2, 3))   # (B, 512)
        f_rgb = f_rgb_spatial.mean(dim=(2, 3))       # (B, 512)
        f_thermal = f_thermal_spatial.mean(dim=(2, 3))  # (B, 512)
        
        # 拼接所有特征
        combined = torch.cat([f_local, f_rgb, f_thermal, intent_feat], dim=1)  # (B, 512*3 + 512)
        
        # 投影降维
        fused = self.fusion_proj(combined)  # (B, 512)
        
        if return_attn:
            return fused, None
        return fused


class MultiTaskHead(nn.Module):
    """
    多任务解耦诊断头 (Multi-Task Decoupled Head)
    
    网络的输出端不采用单一的分类器，而是设计了四个并行的全连接层，
    分别对应四个核心工艺参数的分类诊断：
    
    - flow_rate_head: 流量状态 → [偏低, 正常, 偏高] (3分类)
    - feed_rate_head: 速度状态 → [偏低, 正常, 偏高] (3分类)
    - z_offset_head: Z轴状态 → [偏低, 正常, 偏高] (3分类)
    - hot_end_head: 温度状态 → [偏低, 正常, 偏高] (3分类)
    
    这种设计允许网络学习每个参数特有的特征表示，
    同时共享来自融合模块的基础特征。
    
    改进：
    - 添加 BatchNorm 稳定训练
    - 提高 Dropout 率 (0.2 -> 0.5) 增强正则化，抑制过拟合
    - 添加中间层增强表达能力
    """
    
    def __init__(self, in_dim=512, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # 流量状态分支
        self.flow_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 速度状态分支 (验证集分布偏移严重，增加容量)
        self.feed_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Z轴状态分支
        self.z_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 温度状态分支
        self.hotend_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用 Xavier 初始化，帮助分类头更快收敛"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fused_feat):
        """
        Args:
            fused_feat: 融合后的特征 (B, 512)
        
        Returns:
            dict: 四个任务的输出
                - 'flow_rate': (B, 3) - 流量分类logits
                - 'feed_rate': (B, 3) - 速度分类logits
                - 'z_offset': (B, 3) - Z轴分类logits
                - 'hot_end': (B, 3) - 温度分类logits
        """
        return {
            'flow_rate': self.flow_head(fused_feat),
            'feed_rate': self.feed_head(fused_feat),
            'z_offset': self.z_head(fused_feat),
            'hot_end': self.hotend_head(fused_feat)
        }


class PACFNet(nn.Module):
    """
    完整的 PACF-NET 网络架构 (支持消融实验)
    
    支持的变体:
    - 'full': 完整模型 (默认)
    - 'no_mmd': 变体A - 无MMD损失 (通过设置mmd_weight=0实现)
    - 'concat_only': 变体B - 使用简单拼接代替注意力
    - 'rgb_only': 变体C-1 - 只使用RGB模态 (IDS + Computer RGB)
    - 'ids_only': 变体C-2 - 只使用IDS模态
    
    网络流程：
    1. 多源输入处理：IDS (448*448*3) + Computer RGB (224*224*3) + Fotric热像 (224*224*4)
    2. 视觉特征提取：三个并行ResNet-18塔提取不同尺度和光谱特征
    3. 工艺意图嵌入：10维参数向量 → 512维语义嵌入
    4. 因果融合：基于交叉注意力的动态特征融合 (或拼接融合)
    5. 多任务诊断：四个并行的分类头输出打印状态诊断
    """
    
    def __init__(self, variant='full'):
        """
        Args:
            variant: 模型变体
                'full': 完整模型
                'no_mmd': 变体A (无MMD，通过训练时设置weight=0实现)
                'concat_only': 变体B (拼接融合，无注意力)
                'rgb_only': 变体C-1 (仅RGB模态: IDS + Computer)
                'ids_only': 变体C-2 (仅IDS模态)
        """
        super().__init__()
        
        self.variant = variant
        
        # 模块堆栈
        self.visual_backbone = VisualBackbone()
        self.intent_embed = ProcessIntentEmbedding(input_dim=10, embed_dim=512)
        
        # 根据变体选择融合模块
        if variant == 'concat_only':
            # 变体B: 简单拼接融合
            self.cross_attn = ConcatFusion(visual_dim=512, intent_dim=512, num_sources=3)
        else:
            # 完整模型或其他变体: 交叉注意力融合
            self.cross_attn = CrossAttentionFusion(intent_dim=512, visual_dim=512, num_heads=8)
        
        self.task_head = MultiTaskHead(in_dim=512, num_classes=3)
        
        # MMD损失计算用的高斯核函数
        self.mmd_kernel_bandwidth = 1.0
        
        print(f"✓ PACFNet 初始化完成: variant='{variant}'")
    
    def compute_mmd_loss(self, f_local, f_rgb, hotend_mask):
        """
        计算最大均值差异 (Maximum Mean Discrepancy, MMD) 损失
        用于局部纹理特征和RGB特征之间的因果对齐
        
        保证同一热应力条件下的不同视角特征分布一致
        
        正确的 MMD 计算：
        MMD²(X, Y) = E[k(x, x')] - 2E[k(x, y)] + E[k(y, y')]
        其中 k 是 RBF 核函数
        
        Args:
            f_local: 局部纹理特征 (B, 512)
            f_rgb: RGB环境特征 (B, 512)
            hotend_mask: 热端温度范围标签 (B,) - 用于分组，同组内计算MMD
        
        Returns:
            mmd_loss: 标量损失值（越大表示分布差异越大，需要最小化）
        """
        device = f_local.device
        
        # 检查输入是否包含 NaN 或 Inf
        if torch.isnan(f_local).any() or torch.isnan(f_rgb).any():
            print(f"[MMD Warning] 输入特征包含 NaN!")
            return torch.tensor(0.0, device=device)
        if torch.isinf(f_local).any() or torch.isinf(f_rgb).any():
            print(f"[MMD Warning] 输入特征包含 Inf!")
            return torch.tensor(0.0, device=device)
        
        mmd_loss = torch.tensor(0.0, device=device)
        
        # 根据热端温度分组，计算同组内不同视角（IDS vs RGB）的特征分布差异
        unique_hotends = torch.unique(hotend_mask)
        n_groups = 0
        
        for hotend_class in unique_hotends:
            mask = (hotend_mask == hotend_class)
            n_samples = mask.sum().item()
            
            # 每组至少要有2个样本才能计算MMD
            if n_samples < 2:
                continue
            
            # 获取该组内的局部特征和RGB特征
            local_feats = f_local[mask]  # (N, 512)
            rgb_feats = f_rgb[mask]      # (N, 512)
            
            # 归一化特征，防止数值过大
            local_feats = F.normalize(local_feats, p=2, dim=1)
            rgb_feats = F.normalize(rgb_feats, p=2, dim=1)
            
            # 计算 MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
            n = local_feats.shape[0]
            
            # 1. E[k(x, x')] - 局部特征内部的相似度
            # 使用更稳定的计算方式，避免显式构造大的距离矩阵
            dist_local = torch.cdist(local_feats, local_feats, p=2) ** 2
            k_local = torch.exp(-dist_local / (2 * self.mmd_kernel_bandwidth ** 2))
            # 去掉对角线（自身与自身的距离为0，k=1）
            mask_diag = ~torch.eye(n, dtype=torch.bool, device=device)
            e_local = k_local[mask_diag].mean()
            
            # 2. E[k(y, y')] - RGB特征内部的相似度
            dist_rgb = torch.cdist(rgb_feats, rgb_feats, p=2) ** 2
            k_rgb = torch.exp(-dist_rgb / (2 * self.mmd_kernel_bandwidth ** 2))
            e_rgb = k_rgb[mask_diag].mean()
            
            # 3. E[k(x, y)] - 局部特征和RGB特征之间的相似度
            dist_cross = torch.cdist(local_feats, rgb_feats, p=2) ** 2
            k_cross = torch.exp(-dist_cross / (2 * self.mmd_kernel_bandwidth ** 2))
            e_cross = k_cross.mean()
            
            # MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
            # 我们希望最小化这个值，使得两个分布更接近
            mmd_sq = e_local - 2 * e_cross + e_rgb
            
            # 确保 MMD² 非负且有限（数值误差处理）
            if torch.isnan(mmd_sq) or torch.isinf(mmd_sq):
                continue
            mmd_sq = torch.clamp(mmd_sq, min=0.0, max=10.0)
            
            mmd_loss = mmd_loss + torch.sqrt(mmd_sq)
            n_groups += 1
        
        if n_groups > 0:
            mmd_loss = mmd_loss / n_groups
        else:
            mmd_loss = torch.tensor(0.0, device=device)
        
        return mmd_loss
    
    def _mask_modalities(self, batch_data):
        """
        根据变体类型屏蔽不需要的模态 (消融实验C变体)
        
        Args:
            batch_data: 输入数据字典
        
        Returns:
            处理后的 batch_data
        """
        if self.variant == 'rgb_only':
            # 变体C-1: 保留 IDS + RGB (可见光)，只屏蔽 Thermal (热成像)
            # 用于验证热成像模态的贡献
            # 使用随机噪声代替全零，避免NaN
            batch_data['fotric'] = torch.randn_like(batch_data['fotric']) * 0.01
            batch_data['thermal'] = torch.randn_like(batch_data['thermal']) * 0.01
        
        elif self.variant == 'ids_only':
            # 变体C-2: 只保留 IDS 模态，屏蔽 RGB 和 Thermal
            # 单模态基线，使用随机噪声避免NaN
            batch_data['computer'] = torch.randn_like(batch_data['computer']) * 0.01
            batch_data['fotric'] = torch.randn_like(batch_data['fotric']) * 0.01
            batch_data['thermal'] = torch.randn_like(batch_data['thermal']) * 0.01
        
        return batch_data
    
    def forward(self, batch_data, labels=None):
        """
        前向传播
        
        Args:
            batch_data: dict，包含
                - 'ids': IDS随轴相机 (B, 3, 448, 448)
                - 'computer': 旁轴RGB相机 (B, 3, 224, 224)
                - 'fotric': 伪彩色热像 (B, 3, 224, 224)
                - 'thermal': 灰度热像 (B, 1, 224, 224)
                - 'params': 工艺参数 (B, 10)
            
            labels: dict，包含
                - 'flow_rate': (B,) - 流量分类标签
                - 'feed_rate': (B,) - 速度分类标签
                - 'z_offset': (B,) - Z轴分类标签
                - 'hot_end': (B,) - 温度分类标签
        
        Returns:
            训练时：(outputs, total_loss, intermediate_features)
                - outputs: dict，包含四个任务的logits
                - total_loss: 总损失（分类+MMD）
                - intermediate_features: 中间特征字典，用于可视化和分析
            推理时：
                - outputs: dict，包含四个任务的logits
        """
        # 0. 根据变体屏蔽不需要的模态 (消融实验)
        batch_data = self._mask_modalities(batch_data)
        
        # 1. 提取视觉特征 (同时获取空间和全局特征)
        backbone_outputs = self.visual_backbone(
            batch_data['ids'],
            batch_data['computer'],
            batch_data['fotric'],
            batch_data['thermal']
        )
        
        # 解包视觉特征
        f_local_spatial = backbone_outputs['f_local_spatial']  # (B, 512, 7, 7)
        f_rgb_spatial = backbone_outputs['f_rgb_spatial']      # (B, 512, 7, 7)
        f_thermal_spatial = backbone_outputs['f_thermal_spatial']  # (B, 512, 7, 7)
        f_local = backbone_outputs['f_local']  # (B, 512)
        f_rgb = backbone_outputs['f_rgb']      # (B, 512)
        f_thermal = backbone_outputs['f_thermal']  # (B, 512)
        
        # 2. 计算因果对齐损失 (MMD) - 使用全局特征
        mmd_loss = torch.tensor(0.0, device=f_local.device)
        if labels is not None and self.variant != 'no_mmd':
            # 注意：no_mmd变体在训练时通过alpha_mmd=0来实现
            # 这里仍然计算mmd_loss用于日志记录
            mmd_loss = self.compute_mmd_loss(
                f_local,
                f_rgb=f_rgb,
                hotend_mask=labels['hot_end']
            )
        
        # 3. 工艺意图嵌入
        intent_feat = self.intent_embed(batch_data['params'])  # (B, 512)
        
        # 4. 因果融合 (使用空间特征)
        fused_feat = self.cross_attn(
            f_local_spatial, f_rgb_spatial, f_thermal_spatial, intent_feat
        )  # (B, 512)
        
        # 5. 多任务诊断输出
        outputs = self.task_head(fused_feat)
        
        # 保存中间特征用于分析和可视化
        intermediate_features = {
            'f_local_spatial': f_local_spatial,
            'f_rgb_spatial': f_rgb_spatial,
            'f_thermal_spatial': f_thermal_spatial,
            'f_local': f_local,
            'f_rgb': f_rgb,
            'f_thermal': f_thermal,
            'intent_feat': intent_feat,
            'fused_feat': fused_feat,
        }
        
        # 训练/验证时计算总损失（只要有 labels）
        if labels is not None:
            # 各任务的交叉熵损失（在 train.py 中会重新计算，这里用于调试）
            ce_loss = 0.0
            ce_loss += F.cross_entropy(outputs['flow_rate'], labels['flow_rate'])
            ce_loss += F.cross_entropy(outputs['feed_rate'], labels['feed_rate'])
            ce_loss += F.cross_entropy(outputs['z_offset'], labels['z_offset'])
            ce_loss += F.cross_entropy(outputs['hot_end'], labels['hot_end'])
            
            # 注意：总损失的权重由调用者（train.py）控制
            # 这里返回的 loss_dict 中的值用于监控和调试
            loss_dict = {
                'ce_loss': ce_loss,
                'mmd_loss': mmd_loss,
            }
            
            return outputs, loss_dict, intermediate_features
        else:
            # 推理模式，不计算损失
            return outputs
    
    def extract_features(self, batch_data, return_attention=False):
        """
        提取中间特征用于可视化分析
        
        Args:
            batch_data: 输入数据字典
            return_attention: 是否返回注意力权重
        
        Returns:
            dict: 包含各种中间特征
        """
        features = {}
        
        # 1. 提取视觉特征
        backbone_outputs = self.visual_backbone(
            batch_data['ids'],
            batch_data['computer'],
            batch_data['fotric'],
            batch_data['thermal']
        )
        
        features['f_local'] = backbone_outputs['f_local']  # (B, 512) IDS特征
        features['f_rgb'] = backbone_outputs['f_rgb']      # (B, 512) RGB特征
        features['f_thermal'] = backbone_outputs['f_thermal']  # (B, 512) 热成像特征
        features['f_local_spatial'] = backbone_outputs['f_local_spatial']
        features['f_rgb_spatial'] = backbone_outputs['f_rgb_spatial']
        features['f_thermal_spatial'] = backbone_outputs['f_thermal_spatial']
        
        # 2. 工艺意图嵌入
        intent_feat = self.intent_embed(batch_data['params'])
        features['intent_feat'] = intent_feat
        
        # 3. 因果融合 (可选择返回注意力)
        if return_attention:
            fused_feat, attn_weights = self.cross_attn(
                features['f_local_spatial'],
                features['f_rgb_spatial'],
                features['f_thermal_spatial'],
                intent_feat,
                return_attn=True
            )
            features['attn_weights'] = attn_weights  # (B, 1, 147)
        else:
            fused_feat = self.cross_attn(
                features['f_local_spatial'],
                features['f_rgb_spatial'],
                features['f_thermal_spatial'],
                intent_feat,
                return_attn=False
            )
        
        features['fused_feat'] = fused_feat
        
        # 4. 分类输出
        outputs = self.task_head(fused_feat)
        features['outputs'] = outputs
        
        return features