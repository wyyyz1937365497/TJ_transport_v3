"""
智能交通协同控制系统 - 完整神经网络架构
包含：感知层、预测层、决策层、安全层
严格遵守竞赛规则，仅控制25%的智能车辆
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import time

# 导入各个组件
from risk_sensitive_gnn import RiskSensitiveGNN
from progressive_world_model import ProgressiveWorldModel
from influence_controller import InfluenceDrivenController
from safety_shield import DualModeSafetyShield


class TrafficController(nn.Module):
    """
    智能交通协同控制神经网络
    架构：Risk-Sensitive GNN + Progressive World Model + Influence-Driven Controller + Dual-mode Safety Shield
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 1. 感知层：风险敏感GNN
        self.risk_gnn = RiskSensitiveGNN(
            node_dim=config.get('node_dim', 9),
            edge_dim=config.get('edge_dim', 4),
            hidden_dim=config.get('gnn_hidden_dim', 64),
            output_dim=config.get('gnn_output_dim', 256),
            num_layers=config.get('gnn_layers', 3),
            heads=config.get('gnn_heads', 4)
        )
        
        # 2. 预测层：渐进式世界模型
        self.world_model = ProgressiveWorldModel(
            input_dim=config.get('gnn_output_dim', 256),
            hidden_dim=config.get('world_hidden_dim', 128),
            future_steps=config.get('future_steps', 5),
            num_phases=2
        )
        
        # 3. 决策层：影响力驱动控制器
        self.controller = InfluenceDrivenController(
            gnn_dim=config.get('gnn_output_dim', 256),
            world_dim=config.get('gnn_output_dim', 256),
            global_dim=config.get('global_dim', 16),
            hidden_dim=config.get('controller_hidden_dim', 128),
            action_dim=config.get('action_dim', 2),
            top_k=config.get('top_k', 5)
        )
        
        # 4. 安全层：双模态安全屏障
        self.safety_shield = DualModeSafetyShield(
            ttc_threshold=config.get('ttc_threshold', 2.0),
            thw_threshold=config.get('thw_threshold', 1.5),
            max_accel=config.get('max_accel', 2.0),
            max_decel=config.get('max_decel', -3.0),
            emergency_decel=config.get('emergency_decel', -5.0),
            max_lane_change_speed=config.get('max_lane_change_speed', 5.0)
        )
        
        # 5. 约束优化参数
        self.register_buffer('lagrange_multiplier', torch.tensor(1.0))
        self.cost_limit = config.get('cost_limit', 0.1)
        self.lambda_lr = config.get('lambda_lr', 0.01)
        
        # 6. 缓存机制
        self.gnn_cache = {}
        self.cache_timeout = config.get('cache_timeout', 10)  # 缓存10步
        
        print("✅ 交通控制神经网络初始化完成!")
        print(f"   - GNN维度: {config.get('gnn_output_dim', 256)}")
        print(f"   - 预测步长: {config.get('future_steps', 5)}")
        print(f"   - 控制车辆数: {config.get('top_k', 5)}")
        
    def forward(self, batch: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        前向传播，生成控制指令
        """
        # 1. 感知层：GNN特征提取
        gnn_embedding = self._get_gnn_embedding(batch, step)
        
        # 2. 预测层：未来状态预测
        world_predictions = self.world_model(gnn_embedding)
        
        # 3. 决策层：影响力计算与动作生成
        controller_output = self.controller(
            gnn_embedding=gnn_embedding,
            world_predictions=world_predictions,
            global_metrics=batch['global_metrics'],
            vehicle_ids=batch['vehicle_ids'],
            is_icv=batch['is_icv']
        )
        
        # 4. 安全层：动作安全化
        safe_actions = self.safety_shield(
            raw_actions=controller_output['raw_actions'],
            vehicle_states=batch['vehicle_states'],
            selected_vehicle_indices=controller_output['selected_indices']
        )
        
        # 5. 组合输出
        output = {
            'selected_vehicle_ids': controller_output['selected_vehicle_ids'],
            'safe_actions': safe_actions,
            'influence_scores': controller_output['influence_scores'],
            'level1_interventions': safe_actions['level1_interventions'],
            'level2_interventions': safe_actions['level2_interventions'],
            'gnn_embedding': gnn_embedding,
            'world_predictions': world_predictions
        }
        
        return output
    
    def _get_gnn_embedding(self, batch: Dict[str, Any], step: int) -> torch.Tensor:
        """带缓存的GNN推理"""
        # 生成缓存键
        cache_key = str(hash(str(batch['vehicle_ids']) + str(batch['edge_indices'].shape)))
        
        # 检查缓存
        if cache_key in self.gnn_cache and step - self.gnn_cache[cache_key]['step'] < self.cache_timeout:
            return self.gnn_cache[cache_key]['embedding']
        
        # 构建图数据
        graph_data = self._build_graph(batch)
        
        # GNN推理
        with torch.no_grad():
            gnn_embedding = self.risk_gnn(graph_data)
        
        # 更新缓存
        self.gnn_cache[cache_key] = {
            'embedding': gnn_embedding,
            'step': step
        }
        
        return gnn_embedding
    
    def _build_graph(self, batch: Dict[str, Any]):
        """构建图神经网络输入"""
        # 节点特征
        node_features = batch['node_features']  # [N, 9]
        
        # 边索引
        edge_index = batch['edge_indices']  # [2, E]
        
        # 边特征
        edge_features = batch['edge_features']  # [E, 4]
        
        # 创建简单的图数据对象
        graph = {
            'x': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_features
        }
        
        return graph
    
    def update_lagrange_multiplier(self, mean_cost: float):
        """更新拉格朗日乘子"""
        if mean_cost > self.cost_limit:
            self.lagrange_multiplier *= (1 + self.lambda_lr)
        else:
            self.lagrange_multiplier *= (1 - self.lambda_lr)
        
        # 限制范围
        self.lagrange_multiplier = torch.clamp(self.lagrange_multiplier, 0.1, 10.0)
        
        return self.lagrange_multiplier.item()
