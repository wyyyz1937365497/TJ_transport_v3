import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traci
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import time
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class RiskSensitiveGNN(nn.Module):
    """
    风险敏感图神经网络
    输入：车辆节点特征(9维) + 交互边特征(4维)
    输出：256维全局嵌入
    """

    def __init__(self, node_dim: int = 9, edge_dim: int = 4, hidden_dim: int = 64,
                 output_dim: int = 256, num_layers: int = 3, heads: int = 4):
        super().__init__()

        # 1. 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2. 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )

        # 3. 风险注意力机制
        self.risk_attention = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 4. GNN层 - 改进版（带残差连接和层归一化）
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=hidden_dim // 2,
                    dropout=0.1
                )
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # 5. 输出投影层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, graph: Data) -> torch.Tensor:
        """
        前向传播
        Args:
            graph: 包含x, edge_index, edge_attr的PyG数据对象
        Returns:
            global_embedding: [N, 256] 全局嵌入
        """
        # 1. 编码节点和边特征
        node_features = self.node_encoder(graph.x)  # [N, 64]
        edge_features = self.edge_encoder(graph.edge_attr)  # [E, 32]

        # 2. 计算风险注意力权重
        if edge_features.size(0) > 0:
            src_nodes = graph.edge_index[0]
            risk_input = torch.cat([
                node_features[src_nodes],
                edge_features
            ], dim=1)  # [E, 96]
            risk_weights = self.risk_attention(risk_input)  # [E, 1]
        else:
            risk_weights = None

        # 3. GNN传播 - 改进版（残差连接）
        x = node_features
        for i, (layer, norm) in enumerate(zip(self.gnn_layers, self.norm_layers)):
            residual = x
            x = layer(x, graph.edge_index, edge_attr=edge_features)
            x = F.relu(x)
            x = norm(x + residual)  # 残差连接 + 层归一化

        # 4. 输出投影
        global_embedding = self.output_layer(x)  # [N, 256]

        return global_embedding


class ProgressiveWorldModel(nn.Module):
    """
    渐进式世界模型
    阶段1：预测下一时刻状态
    阶段2：预测未来5步状态 + 冲突概率
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128,
                 future_steps: int = 5, num_phases: int = 2):
        super().__init__()

        self.future_steps = future_steps
        self.num_phases = num_phases
        self.current_phase = 1

        # 1. 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2. 基础动力学分支 (Phase 1)
        self.dynamics_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 3. 风险演化分支 (Phase 2)
        self.risk_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 192),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(192, input_dim + 1)  # 状态 + 冲突概率
            ) for _ in range(future_steps)
        ])

        # 4. 辅助分类器
        self.conflict_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LSTM)):
                if hasattr(m, 'weight') and m.weight is not None and isinstance(m.weight, torch.Tensor):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
                    nn.init.constant_(m.bias, 0)

    def set_phase(self, phase: int):
        """设置训练阶段"""
        self.current_phase = phase
        print(f"🔄 世界模型切换到阶段 {phase}")

    def forward(self, gnn_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            gnn_embedding: [N, 256] GNN输出嵌入
        Returns:
            predictions: 
                Phase 1: [N, 256] 下一时刻状态
                Phase 2: [N, 5, 257] 未来5步状态 + 冲突概率
        """
        batch_size = gnn_embedding.size(0)

        # 1. 共享编码
        encoded = self.shared_encoder(gnn_embedding)  # [N, 128]

        if self.current_phase == 1:
            # Phase 1: 基础动力学预测
            # 重塑为LSTM输入格式 [N, 1, 128]
            lstm_input = encoded.unsqueeze(1)

            # LSTM预测
            lstm_output, _ = self.dynamics_lstm(lstm_input)  # [N, 1, 128]

            # 预测下一时刻状态
            next_state = self.risk_decoders[0](lstm_output.squeeze(1))[:, :-1]  # [N, 256]

            return next_state

        else:
            # Phase 2: 风险演化预测
            predictions = []

            # 为每个未来步生成预测
            for t in range(self.future_steps):
                # 使用相同的编码但添加时间步信息
                time_input = encoded + 0.1 * t * torch.ones_like(encoded)
                pred = self.risk_decoders[t](time_input)  # [N, 257]
                predictions.append(pred.unsqueeze(1))

            # 合并预测 [N, 5, 257]
            predictions = torch.cat(predictions, dim=1)

            return predictions


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

        # 2. 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_dim + 64 + 257, 384),
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
        # 处理world_predictions的不同维度
        if world_predictions.dim() == 3:
            # Phase 2: [N, 5, 257] -> 取平均得到 [N, 257]
            avg_world_pred = world_predictions.mean(dim=1)
        elif world_predictions.dim() == 2:
            # Phase 1: [N, 256] -> 需要padding到257维
            avg_world_pred = torch.cat([
                world_predictions,
                torch.zeros(batch_size, 1, device=world_predictions.device)
            ], dim=1)
        else:
            avg_world_pred = world_predictions

        # 重复全局特征以匹配批次大小
        global_features_expanded = global_features.repeat(batch_size, 1)

        # 融合
        fused_input = torch.cat([
            gnn_embedding,
            global_features_expanded,
            avg_world_pred
        ], dim=1)  # [N, 256+64+257] = [N, 577]

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

        selected_indices = icv_indices[top_k_indices.cpu()]  # [K]
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


class DualModeSafetyShield(nn.Module):
    """
    双模态安全屏障
    Level 1: 动作裁剪（软约束）
    Level 2: 紧急制动（硬约束）
    """

    def __init__(self, ttc_threshold: float = 2.0, thw_threshold: float = 1.5,
                 max_accel: float = 2.0, max_decel: float = -3.0,
                 emergency_decel: float = -5.0, max_lane_change_speed: float = 5.0):
        super().__init__()

        self.ttc_threshold = ttc_threshold
        self.thw_threshold = thw_threshold
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.emergency_decel = emergency_decel
        self.max_lane_change_speed = max_lane_change_speed

        # 安全参数（可学习）
        self.register_parameter('learnable_max_accel', nn.Parameter(torch.tensor(max_accel)))
        self.register_parameter('learnable_max_decel', nn.Parameter(torch.tensor(max_decel)))
        self.register_parameter('learnable_emergency_decel', nn.Parameter(torch.tensor(emergency_decel)))

    def forward(self, raw_actions: torch.Tensor, vehicle_states: Dict[str, Any],
                selected_vehicle_indices: List[int]) -> Dict[str, Any]:
        """
        安全屏障前向传播
        Args:
            raw_actions: [K, 2] 原始控制动作（加速度，换道概率）
            vehicle_states: 车辆状态字典
            selected_vehicle_indices: 选中车辆索引列表
        Returns:
            安全化后的动作和干预统计
        """
        if len(selected_vehicle_indices) == 0:
            return {
                'safe_actions': torch.zeros(0, 2),
                'level1_interventions': 0,
                'level2_interventions': 0
            }

        # Level 1: 动作裁剪
        level1_actions, level1_interventions = self._level1_clipping(
            raw_actions, vehicle_states, selected_vehicle_indices
        )

        # Level 2: 紧急安全检查
        level2_actions, level2_interventions = self._level2_emergency_check(
            level1_actions, vehicle_states, selected_vehicle_indices
        )

        total_level1 = torch.sum(level1_interventions).item()
        total_level2 = torch.sum(level2_interventions).item()

        return {
            'safe_actions': level2_actions,
            'level1_interventions': total_level1,
            'level2_interventions': total_level2
        }

    def _level1_clipping(self, raw_actions: torch.Tensor, vehicle_states: Dict[str, Any],
                         selected_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Level 1: 基础动作裁剪"""
        k = len(selected_indices)
        safe_actions = raw_actions.clone()
        intervention_mask = torch.zeros(k, dtype=torch.bool)

        for i, idx in enumerate(selected_indices):
            veh_id = vehicle_states['ids'][idx]

            if veh_id not in vehicle_states['data']:
                continue

            vehicle = vehicle_states['data'][veh_id]
            current_speed = vehicle['speed']

            # 1. 加速度裁剪
            raw_accel = raw_actions[i, 0].item()

            # 动态调整加速度限制（基于速度）
            dynamic_max_accel = self.max_accel * (1 - current_speed / 30.0)  # 高速时减小加速度
            dynamic_max_decel = self.max_decel * (1 + current_speed / 30.0)  # 高速时增大减速度

            safe_accel = max(min(raw_accel, dynamic_max_accel), dynamic_max_decel)

            if abs(safe_accel - raw_accel) > 0.1:  # 干预阈值
                intervention_mask[i] = True

            # 2. 换道限制
            raw_lane_change = raw_actions[i, 1].item()
            safe_lane_change = raw_lane_change

            # 仅在低速时允许换道
            if current_speed > self.max_lane_change_speed:
                safe_lane_change = 0.0
                if raw_lane_change > 0.5:
                    intervention_mask[i] = True

            # 更新安全动作
            safe_actions[i, 0] = safe_accel
            safe_actions[i, 1] = safe_lane_change

        return safe_actions, intervention_mask

    def _level2_emergency_check(self, actions: torch.Tensor, vehicle_states: Dict[str, Any],
                                selected_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Level 2: 紧急安全检查"""
        k = len(selected_indices)
        final_actions = actions.clone()
        emergency_mask = torch.zeros(k, dtype=torch.bool)

        for i, idx in enumerate(selected_indices):
            veh_id = vehicle_states['ids'][idx]

            if veh_id not in vehicle_states['data']:
                continue

            ego_vehicle = vehicle_states['data'][veh_id]
            leader_vehicle = self._find_leader(veh_id, ego_vehicle, vehicle_states['data'])

            if leader_vehicle:
                # 计算TTC和THW
                ttc = self._calculate_ttc(ego_vehicle, leader_vehicle)
                thw = self._calculate_thw(ego_vehicle, leader_vehicle)

                # 检查紧急条件
                if ttc < self.ttc_threshold or thw < self.thw_threshold:
                    # 紧急制动
                    final_actions[i, 0] = self.emergency_decel
                    final_actions[i, 1] = 0.0  # 取消换道
                    emergency_mask[i] = True

        return final_actions, emergency_mask

    def _find_leader(self, ego_id: str, ego: Dict[str, Any], all_vehicles: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """找到前车"""
        min_distance = float('inf')
        leader = None

        for veh_id, vehicle in all_vehicles.items():
            if veh_id == ego_id:
                continue

            # 检查是否在同一车道
            if vehicle['lane_id'] != ego['lane_id']:
                continue

            # 检查是否在前方
            if vehicle['position'] <= ego['position']:
                continue

            distance = vehicle['position'] - ego['position']
            if distance < min_distance:
                min_distance = distance
                leader = vehicle

        return leader if min_distance < 100 else None  # 100米内

    def _calculate_ttc(self, ego: Dict[str, Any], leader: Dict[str, Any]) -> float:
        """计算碰撞时间TTC"""
        relative_speed = ego['speed'] - leader['speed']
        distance = leader['position'] - ego['position']

        if relative_speed <= 0:
            return float('inf')  # 不会碰撞

        ttc = distance / relative_speed
        return max(0.1, ttc)  # 防止除零

    def _calculate_thw(self, ego: Dict[str, Any], leader: Dict[str, Any]) -> float:
        """计算车头时距THW"""
        distance = leader['position'] - ego['position']
        if ego['speed'] <= 0:
            return float('inf')

        thw = distance / ego['speed']
        return max(0.1, thw)  # 防止除零


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

    def _build_graph(self, batch: Dict[str, Any]) -> Data:
        """构建图神经网络输入"""
        # 节点特征
        node_features = batch['node_features']  # [N, 9]

        # 边索引
        edge_index = batch['edge_indices']  # [2, E]

        # 边特征
        edge_features = batch['edge_features']  # [E, 4]

        # 创建PyG数据对象
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )

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


class NeuralTrafficController:
    """
    神经交通控制器，集成到SUMO竞赛框架
    """

    def __init__(self, config_path: str = None):
        # 默认配置
        self.config = {
            'node_dim': 9,
            'edge_dim': 4,
            'gnn_hidden_dim': 64,
            'gnn_output_dim': 256,
            'gnn_layers': 3,
            'gnn_heads': 4,
            'world_hidden_dim': 128,
            'future_steps': 5,
            'controller_hidden_dim': 128,
            'global_dim': 16,
            'top_k': 5,
            'ttc_threshold': 2.0,
            'thw_threshold': 1.5,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'emergency_decel': -5.0,
            'max_lane_change_speed': 5.0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'cache_timeout': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_path': None
        }

        # 加载配置文件
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.config.update(config_data)

        # 初始化神经网络
        self.device = torch.device(self.config['device'])
        self.model = TrafficController(self.config).to(self.device)

        # 加载预训练模型
        if self.config.get('model_path') and os.path.exists(self.config['model_path']):
            try:
                checkpoint = torch.load(self.config['model_path'], map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 加载预训练模型: {self.config['model_path']}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")

        # 统计信息
        self.total_interventions = 0
        self.total_emergency_interventions = 0
        self.total_controlled_vehicles = 0

        print(f"🚀 神经交通控制器初始化完成! 设备: {self.device}")

    def build_model_input(self, vehicle_data: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        构建模型输入
        """
        # 1. 收集车辆特征
        vehicle_ids = list(vehicle_data.keys())
        node_features = []
        is_icv_list = []

        for i, veh_id in enumerate(vehicle_ids):
            vehicle = vehicle_data[veh_id]

            # 节点特征: [位置, 速度, 加速度, 车道, 剩余距离, 完成率, 类型, 时间, 步长]
            position = vehicle.get('position', 0.0)
            speed = vehicle.get('speed', 0.0)
            acceleration = vehicle.get('acceleration', 0.0)
            lane_index = vehicle.get('lane_index', 0)
            remaining_distance = vehicle.get('remaining_distance', 1000.0)
            completion_rate = vehicle.get('completion_rate', 0.0)
            is_icv = 1.0 if vehicle.get('is_icv', False) else 0.0  # ICV标志
            current_time = step * 0.1  # 时间(秒)
            time_step = 0.1  # 步长

            features = [
                position,
                speed,
                acceleration,
                lane_index,
                remaining_distance,
                completion_rate,
                is_icv,
                current_time,
                time_step
            ]

            node_features.append(features)
            is_icv_list.append(vehicle.get('is_icv', False))

        # 2. 构建交互图
        edge_indices = []
        edge_features = []

        # 连接相近车辆，考虑实际的车辆位置和车道
        for i, veh_id_i in enumerate(vehicle_ids):
            for j, veh_id_j in enumerate(vehicle_ids):
                if i == j:
                    continue

                # 获取车辆信息
                vehicle_i = vehicle_data[veh_id_i]
                vehicle_j = vehicle_data[veh_id_j]
                
                # 获取车辆位置和速度
                pos_i = vehicle_i.get('position', 0.0)
                pos_j = vehicle_j.get('position', 0.0)
                speed_i = vehicle_i.get('speed', 0.0)
                speed_j = vehicle_j.get('speed', 0.0)
                
                # 获取车道信息
                lane_i = vehicle_i.get('lane_id', '')
                lane_j = vehicle_j.get('lane_id', '')

                # 计算距离
                distance = abs(pos_i - pos_j)
                
                # 只有在同一条车道上或距离很近的情况下才建立连接
                if lane_i == lane_j or distance < 50:  # 50米内或同车道
                    edge_indices.append([i, j])

                    # 边特征: [相对距离, 相对速度, TTC, THW]
                    rel_distance = distance
                    rel_speed = abs(speed_i - speed_j)

                    # 计算TTC (Time To Collision) 和 THW (Time Headway)
                    # TTC = distance / closing_speed (如果接近的话)
                    closing_speed = abs(speed_i - speed_j)
                    if speed_i > speed_j and closing_speed > 0.1:
                        # 车辆i在追车辆j的情况
                        ttc = rel_distance / closing_speed if closing_speed > 0 else float('inf')
                    else:
                        # 不会追尾
                        ttc = float('inf')

                    # THW = distance / speed_of_rear_vehicle (对于后车而言)
                    rear_speed = min(speed_i, speed_j)
                    thw = rel_distance / rear_speed if rear_speed > 0 else float('inf')

                    edge_features.append([
                        rel_distance,
                        rel_speed,
                        min(ttc, 100.0),  # 限制TTC最大值
                        min(thw, 100.0)   # 限制THW最大值
                    ])

        # 3. 全局交通指标
        global_metrics = self._calculate_global_metrics(vehicle_data, step)

        # 4. 转换为张量
        batch = {
            'node_features': torch.tensor(node_features, dtype=torch.float32).to(self.device),
            'edge_indices': torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device) if edge_indices else torch.zeros((2, 0), dtype=torch.long).to(self.device),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32).to(self.device) if edge_features else torch.zeros((0, 4), dtype=torch.float32).to(self.device),
            'global_metrics': torch.tensor(global_metrics, dtype=torch.float32).unsqueeze(0).to(self.device),
            'vehicle_ids': vehicle_ids,
            'is_icv': torch.tensor(is_icv_list, dtype=torch.bool).to(self.device),
            'vehicle_states': {
                'ids': vehicle_ids,
                'data': vehicle_data
            }
        }

        return batch

    def _calculate_global_metrics(self, vehicle_data: Dict[str, Any], step: int) -> List[float]:
        """
        计算全局交通指标
        """
        speeds = [v['speed'] for v in vehicle_data.values()]
        positions = [v['position'] for v in vehicle_data.values()]
        accelerations = [v['acceleration'] for v in vehicle_data.values()]

        avg_speed = np.mean(speeds) if speeds else 0.0
        speed_std = np.std(speeds) if len(speeds) > 1 else 0.0
        avg_accel = np.mean(np.abs(accelerations)) if accelerations else 0.0
        vehicle_count = len(vehicle_data)

        # 16维全局指标
        metrics = [
            avg_speed, speed_std, avg_accel, vehicle_count,
            step * 0.1,  # 当前时间
            min(positions) if positions else 0.0,  # 最小位置
            max(positions) if positions else 0.0,  # 最大位置
            np.mean(positions) if positions else 0.0,  # 平均位置
            len([v for v in vehicle_data.values() if v.get('is_icv', False)]),  # ICV数量
            vehicle_count - len([v for v in vehicle_data.values() if v.get('is_icv', False)]),  # 非ICV数量
            np.sum([v['speed'] for v in vehicle_data.values() if v.get('is_icv', False)]) if vehicle_data else 0.0,  # ICV总速度
            np.sum([v['speed'] for v in vehicle_data.values() if not v.get('is_icv', False)]) if vehicle_data else 0.0,  # 非ICV总速度
            avg_speed * vehicle_count,  # 总流量
            speed_std * vehicle_count,  # 总波动
            avg_accel * vehicle_count,  # 总加速度
            step % 100  # 周期性特征
        ]

        return metrics

    def apply_control(self, vehicle_data: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        应用控制算法
        """
        # 1. 构建模型输入
        batch = self.build_model_input(vehicle_data, step)

        # 2. 模型推理
        with torch.no_grad():
            output = self.model(batch, step)

        # 3. 应用安全动作
        control_results = self._apply_safe_actions(output, vehicle_data)

        # 4. 更新统计
        self.total_interventions += output['level1_interventions'] + output['level2_interventions']
        self.total_emergency_interventions += output['level2_interventions']
        self.total_controlled_vehicles += len(output['selected_vehicle_ids'])

        # 5. 调试输出
        if step % 100 == 0:
            print(f"[Step {step}] 控制: {len(output['selected_vehicle_ids'])}辆, "
                  f"干预: {output['level1_interventions'] + output['level2_interventions']}, "
                  f"紧急: {output['level2_interventions']}")

        return control_results

    def _apply_safe_actions(self, output: Dict[str, Any], vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用安全动作到SUMO
        """
        results = {
            'controlled_vehicles': [],
            'actions_applied': [],
            'safety_interventions': output['level1_interventions'] + output['level2_interventions'],
            'emergency_interventions': output['level2_interventions']
        }

        for i, veh_id in enumerate(output['selected_vehicle_ids']):
            if veh_id not in vehicle_data:
                continue

            try:
                action = output['safe_actions']['safe_actions'][i]
                accel_action = action[0].item() * 5.0  # [-1,1] -> [-5,5]
                lane_action = action[1].item() > 0.5  # 概率转布尔

                # 应用加速度控制
                current_speed = traci.vehicle.getSpeed(veh_id)
                new_speed = max(0.0, current_speed + accel_action * 0.1)  # 0.1秒步长

                traci.vehicle.setSpeedMode(veh_id, 0)  # 关闭SUMO自动控制
                traci.vehicle.setSpeed(veh_id, new_speed)

                # 记录控制结果
                results['controlled_vehicles'].append(veh_id)
                results['actions_applied'].append({
                    'acceleration': accel_action,
                    'lane_change': lane_action,
                    'new_speed': new_speed
                })

            except traci.TraCIException as e:
                continue
            except Exception as e:
                continue

        return results