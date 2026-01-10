"""
决策层：影响力驱动控制器 (Influence-Driven Controller)
动态计算每辆车的影响力得分 Score_i = α·Importance_GNN + β·Impact_Predicted
仅对 Top-K（如 K=5）关键车辆执行强化学习控制，其余车辆使用 IDM 模型跟驰
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        
        # 1. 全局上下文编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )
        
        # 2. 特征融合层 - 修复维度问题
        # 输入维度: gnn_dim (256) + 64 + world_dim (256) = 576
        fusion_input_dim = gnn_dim + 64 + world_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. 影响力评分网络
        self.influence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 4. 动作生成网络
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
        
        # 5. 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 可学习的权重参数
        self.register_parameter('alpha', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('beta', nn.Parameter(torch.tensor(5.0)))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None and not isinstance(m.bias, bool):
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
        if world_predictions.dim() == 3:
            avg_world_pred = world_predictions.mean(dim=1)  # [N, 257]
        else:
            avg_world_pred = world_predictions  # [N, 256]
        
        # 重复全局特征以匹配批次大小
        global_features_expanded = global_features.repeat(batch_size, 1)
        
        # 融合: gnn_embedding (256) + global_features (64) + avg_world_pred (256/257) = 576/577
        # 我们需要确保维度匹配
        if avg_world_pred.size(-1) == 257:
            avg_world_pred = avg_world_pred[:, :256]  # 截断到256维
        
        fused_input = torch.cat([
            gnn_embedding,
            global_features_expanded,
            avg_world_pred
        ], dim=1)  # [N, 576]
        
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
        influence_scores = self.influence_scorer(icv_features).squeeze(-1)  # [N_icv]
        
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
            'top_k_scores': top_k_scores
        }
    
    def compute_influence(self, gnn_importance: torch.Tensor, 
                         predicted_impact: torch.Tensor) -> torch.Tensor:
        """
        计算综合影响力得分
        Args:
            gnn_importance: [N] GNN重要性得分
            predicted_impact: [N] 预测影响力得分
        Returns:
            influence_scores: [N] 综合影响力得分
        """
        # 使用可学习的权重参数
        alpha = torch.clamp(self.alpha, min=0.1, max=10.0)
        beta = torch.clamp(self.beta, min=0.1, max=10.0)
        
        influence_scores = alpha * gnn_importance + beta * predicted_impact
        return influence_scores
    
    def compute_action_loss(self, actions: torch.Tensor, target_actions: torch.Tensor,
                           advantages: torch.Tensor) -> torch.Tensor:
        """
        计算动作损失
        Args:
            actions: [B, 2] 采取的动作
            target_actions: [B, 2] 目标动作
            advantages: [B] 优势函数
        Returns:
            loss: 动作损失
        """
        # 加速动作损失（MSE）
        accel_loss = F.mse_loss(actions[:, 0], target_actions[:, 0])
        
        # 换道动作损失（BCE）
        lane_loss = F.binary_cross_entropy(actions[:, 1], target_actions[:, 1])
        
        # 优势加权
        weighted_accel_loss = (accel_loss * advantages.abs()).mean()
        weighted_lane_loss = (lane_loss * advantages.abs()).mean()
        
        # 总损失
        total_loss = weighted_accel_loss + 0.5 * weighted_lane_loss
        
        return total_loss
    
    def compute_value_loss(self, value_estimates: torch.Tensor, 
                          returns: torch.Tensor) -> torch.Tensor:
        """
        计算价值损失
        Args:
            value_estimates: [N] 价值估计
            returns: [N] 实际回报
        Returns:
            loss: 价值损失
        """
        return F.mse_loss(value_estimates, returns)


class TopKSelector(nn.Module):
    """
    Top-K选择器
    根据影响力得分选择Top-K车辆
    """
    
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
    
    def forward(self, influence_scores: torch.Tensor, 
                icv_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择Top-K车辆
        Args:
            influence_scores: [N] 影响力得分
            icv_mask: [N] ICV掩码
        Returns:
            selected_indices: [K] 选中车辆索引
            selected_scores: [K] 选中车辆得分
        """
        # 仅考虑ICV车辆
        masked_scores = influence_scores * icv_mask.float()
        
        # 选择Top-K
        k = min(self.top_k, icv_mask.sum().item())
        if k == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([])
        
        top_k_scores, top_k_indices = torch.topk(masked_scores, k, largest=True)
        
        return top_k_indices, top_k_scores


class IDMController:
    """
    IDM (Intelligent Driver Model) 控制器
    用于非受控车辆的跟驰行为
    """
    
    def __init__(self, desired_speed: float = 30.0, 
                 safe_time_headway: float = 1.5,
                 min_gap: float = 2.0,
                 max_accel: float = 2.0,
                 comfortable_decel: float = 3.0):
        self.desired_speed = desired_speed
        self.safe_time_headway = safe_time_headway
        self.min_gap = min_gap
        self.max_accel = max_accel
        self.comfortable_decel = comfortable_decel
    
    def compute_acceleration(self, ego_speed: float, leader_speed: float,
                            gap: float) -> float:
        """
        计算IDM加速度
        Args:
            ego_speed: 自车速度
            leader_speed: 前车速度
            gap: 车距
        Returns:
            acceleration: 计算的加速度
        """
        delta_v = ego_speed - leader_speed
        
        # 期望车距
        desired_gap = self.min_gap + ego_speed * self.safe_time_headway + \
                     (ego_speed * delta_v) / (2 * np.sqrt(
                         self.max_accel * self.comfortable_decel))
        
        # IDM加速度公式
        accel = self.max_accel * (1 - (ego_speed / self.desired_speed)**4 - 
                                  (desired_gap / max(gap, 0.1))**2)
        
        return float(accel)
