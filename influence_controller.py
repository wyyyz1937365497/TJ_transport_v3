"""
决策层：影响力驱动控制器 (Influence-Driven Controller)
动态计算每辆车的影响力得分 Score_i = α·Importance_GNN + β·Impact_Predicted
仅对 Top-K（如 K=5）关键车辆执行强化学习控制，其余车辆使用 IDM 模型跟驰
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class InfluenceDrivenController(nn.Module):
    """
    影响力驱动控制器
    1. 计算每辆车的影响力得分
    2. 选择Top-K最具影响力的ICV车辆
    3. 为选中的车辆生成控制动作
    """
    
    def __init__(self, gnn_dim: int = 256, world_dim: int = 256, global_dim: int = 16,
                 hidden_dim: int = 128, action_dim: int = 2, top_k: int = 5):
        super().__init__()
        
        self.top_k = top_k
        self.action_dim = action_dim
        
        # 影响力权重参数
        self.alpha = nn.Parameter(torch.tensor(1.0))  # GNN重要性权重
        self.beta = nn.Parameter(torch.tensor(5.0))    # 预测影响力权重
        
        # 1. 全局上下文编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )
        
        # 2. 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_dim + 64 + world_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. GNN重要性评分网络
        self.gnn_importance_scorer = nn.Sequential(
            nn.Linear(gnn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 4. 预测影响力评分网络
        self.predicted_impact_scorer = nn.Sequential(
            nn.Linear(world_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 5. 动作生成网络
        self.action_generator = nn.ModuleDict({
            'acceleration': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()  # 输出范围[-1, 1]
            ),
            'lane_change': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # 输出概率[0, 1]
            )
        })
        
        # 6. 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, gnn_embedding: torch.Tensor, world_predictions: torch.Tensor,
                global_metrics: torch.Tensor, vehicle_ids: List[str], 
                is_icv: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播
        Args:
            gnn_embedding: [N, 256] GNN嵌入
            world_predictions: [N, 5, 257] 世界模型预测
            global_metrics: [B, 16] 全局交通指标
            vehicle_ids: [N] 车辆ID列表
            is_icv: [N] 是否为智能网联车
        Returns:
            包含选中车辆ID、控制动作等的字典
        """
        batch_size = gnn_embedding.size(0)
        
        # 1. 处理全局特征
        global_features = self.global_encoder(global_metrics)  # [B, 64]
        
        # 2. 融合特征
        # 取世界模型的平均预测
        avg_world_pred = world_predictions.mean(dim=1) if world_predictions.dim() == 3 else world_predictions
        
        # 重复全局特征以匹配批次大小
        global_features_expanded = global_features.repeat(batch_size, 1)
        
        # 融合
        fused_input = torch.cat([
            gnn_embedding,
            global_features_expanded,
            avg_world_pred
        ], dim=1)  # [N, 256+64+256] = [N, 576]
        
        fused_features = self.fusion_layer(fused_input)  # [N, 128]
        
        # 3. 计算ICV车辆的影响力得分
        icv_mask = is_icv.bool()
        icv_indices = torch.where(icv_mask)[0]
        
        if len(icv_indices) == 0:
            return {
                'selected_vehicle_ids': [],
                'selected_indices': [],
                'raw_actions': torch.zeros(0, self.action_dim),
                'influence_scores': torch.zeros(0),
                'value_estimates': torch.zeros(0)
            }
        
        icv_features = fused_features[icv_mask]  # [N_icv, 128]
        icv_gnn_emb = gnn_embedding[icv_mask]  # [N_icv, 256]
        icv_world_pred = avg_world_pred[icv_mask]  # [N_icv, 256]
        
        # 计算GNN重要性得分
        gnn_importance = self.gnn_importance_scorer(icv_gnn_emb).squeeze(-1)  # [N_icv]
        
        # 计算预测影响力得分
        predicted_impact = self.predicted_impact_scorer(icv_world_pred).squeeze(-1)  # [N_icv]
        
        # 综合影响力得分：Score_i = α·Importance_GNN + β·Impact_Predicted
        influence_scores = (self.alpha * gnn_importance + 
                          self.beta * predicted_impact) / (self.alpha + self.beta)  # [N_icv]
        
        # 4. 选择Top-K车辆
        k = min(self.top_k, len(icv_indices))
        top_k_scores, top_k_indices = torch.topk(influence_scores, k, largest=True, sorted=True)
        
        selected_indices = icv_indices[top_k_indices]  # [K]
        selected_vehicle_ids = [vehicle_ids[i] for i in selected_indices.cpu().numpy()]
        
        # 5. 为选中车辆生成动作
        selected_features = fused_features[selected_indices]  # [K, 128]
        
        # 生成加速度动作
        accel_actions = self.action_generator['acceleration'](selected_features)  # [K, 1]
        
        # 生成换道概率
        lane_actions = self.action_generator['lane_change'](selected_features)  # [K, 1]
        
        # 组合动作
        raw_actions = torch.cat([accel_actions, lane_actions], dim=1)  # [K, 2]
        
        # 6. 价值估计
        value_estimates = self.value_network(fused_features).squeeze(-1)  # [N]
        
        return {
            'selected_vehicle_ids': selected_vehicle_ids,
            'selected_indices': selected_indices.cpu().numpy().tolist(),
            'raw_actions': raw_actions,
            'influence_scores': influence_scores,
            'value_estimates': value_estimates,
            'top_k_scores': top_k_scores,
            'gnn_importance': gnn_importance,
            'predicted_impact': predicted_impact
        }


class IDMModel(nn.Module):
    """
    智能驾驶员模型 (Intelligent Driver Model)
    用于非受控车辆的跟驰行为
    """
    
    def __init__(self, 
                 desired_speed: float = 30.0,
                 safe_time_headway: float = 1.5,
                 max_accel: float = 2.0,
                 comfortable_decel: float = 1.5,
                 min_gap: float = 2.0,
                 exponent: float = 4.0):
        super().__init__()
        
        self.register_buffer('desired_speed', torch.tensor(desired_speed))
        self.register_buffer('safe_time_headway', torch.tensor(safe_time_headway))
        self.register_buffer('max_accel', torch.tensor(max_accel))
        self.register_buffer('comfortable_decel', torch.tensor(comfortable_decel))
        self.register_buffer('min_gap', torch.tensor(min_gap))
        self.register_buffer('exponent', torch.tensor(exponent))
    
    def forward(self, speed: torch.Tensor, leader_speed: torch.Tensor, 
                gap: torch.Tensor) -> torch.Tensor:
        """
        计算IDM加速度
        Args:
            speed: [N] 当前车辆速度
            leader_speed: [N] 前车速度
            gap: [N] 车间距
        Returns:
            acceleration: [N] 计算的加速度
        """
        # 防止除零
        gap = torch.clamp(gap, min=self.min_gap)
        
        # 自由流项
        free_flow = 1.0 - (speed / self.desired_speed) ** self.exponent
        
        # 交互项
        delta_v = speed - leader_speed
        desired_gap = (self.min_gap + 
                       speed * self.safe_time_headway +
                       (speed * delta_v) / (2 * torch.sqrt(self.max_accel * self.comfortable_decel)))
        
        interaction = (desired_gap / gap) ** 2
        
        # IDM加速度
        acceleration = self.max_accel * (free_flow - interaction)
        
        return acceleration


class TopKSelector(nn.Module):
    """
    Top-K选择器
    基于影响力得分选择最具影响力的K个智能体
    """
    
    def __init__(self, top_k: int = 5, temperature: float = 1.0):
        super().__init__()
        
        self.top_k = top_k
        self.temperature = temperature
    
    def forward(self, scores: torch.Tensor, is_icv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择Top-K智能体
        Args:
            scores: [N] 影响力得分
            is_icv: [N] 是否为智能网联车
        Returns:
            selected_indices: [K] 选中的索引
            selected_scores: [K] 选中的得分
        """
        # 只考虑ICV车辆
        icv_mask = is_icv.bool()
        icv_indices = torch.where(icv_mask)[0]
        
        if len(icv_indices) == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)
        
        # 获取ICV车辆的得分
        icv_scores = scores[icv_indices]
        
        # 应用温度缩放
        scaled_scores = icv_scores / self.temperature
        
        # 选择Top-K
        k = min(self.top_k, len(icv_indices))
        top_k_scores, top_k_local_indices = torch.topk(scaled_scores, k, largest=True)
        
        # 转换为全局索引
        selected_indices = icv_indices[top_k_local_indices]
        
        return selected_indices, top_k_scores
