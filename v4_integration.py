"""
v4.0 完整集成主类
协调感知层、预测层、决策层、安全约束模块和事件触发机制之间的数据交互

数据流：
Observation → RiskSensitiveGNN → ProgressiveWorldModel → InfluenceDrivenController 
           → DualModeSafetyShield → EventTriggeredController → Safe Actions
"""

import numpy as np
import torch
import torch.nn as nn
import traci
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time


# 导入所有核心组件
from risk_sensitive_gnn import RiskSensitiveGNN, GraphAttentionLayer
from progressive_world_model import ProgressiveWorldModel
from influence_controller import InfluenceDrivenController, IDMController
from safety_shield import DualModeSafetyShield, SafetyReward, ActionClipper
from event_triggered_controller import EventTriggeredController, EventType
from sumo_rl_env_optimized import SUMORLEnvironmentOptimized, TraCISubscriptionManager


class IntegrationPhase(Enum):
    """集成阶段枚举"""
    PHASE_1_INITIALIZATION = "phase_1_initialization"
    PHASE_2_PERCEPTION = "phase_2_perception"
    PHASE_3_PREDICTION = "phase_3_prediction"
    PHASE_4_DECISION = "phase_4_decision"
    PHASE_5_SAFETY = "phase_5_safety"
    PHASE_6_EVENT_TRIGGER = "phase_6_event_trigger"
    PHASE_7_ACTION_APPLICATION = "phase_7_action_application"


@dataclass
class IntegrationState:
    """集成状态数据类"""
    phase: IntegrationPhase
    timestamp: float
    vehicle_count: int
    icv_count: int
    hv_count: int
    avg_speed: float
    total_reward: float
    safety_interventions: int
    control_updates: int
    processing_time: float


class V4CompleteIntegration:
    """
    v4.0 完整集成主类
    
    协调所有组件之间的数据交互，实现完整的交通控制流程
    """
    
    def __init__(self,
                 sumo_cfg_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 gnn_hidden_dim: int = 256,
                 gnn_num_heads: int = 4,
                 world_model_hidden_dim: int = 256,
                 top_k: int = 5,
                 icv_penetration: float = 0.25,
                 control_interval: float = 10.0,
                 use_gui: bool = False,
                 max_steps: int = 3600):
        """
        初始化 v4.0 完整集成系统
        
        Args:
            sumo_cfg_path: SUMO 配置文件路径
            device: 计算设备
            gnn_hidden_dim: GNN 隐藏层维度
            gnn_num_heads: GNN 注意力头数
            world_model_hidden_dim: 世界模型隐藏层维度
            top_k: Top-K 选择的车辆数
            icv_penetration: ICV 渗透率
            control_interval: 控制间隔（秒）
            use_gui: 是否使用 GUI
            max_steps: 最大仿真步数
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.device = device
        self.top_k = top_k
        self.icv_penetration = icv_penetration
        self.control_interval = control_interval
        self.use_gui = use_gui
        self.max_steps = max_steps
        
        print("=" * 80)
        print("🚀 v4.0 完整集成系统初始化")
        print("=" * 80)
        
        # 初始化 SUMO 环境
        print("\n[1/6] 初始化 SUMO 环境...")
        self.sumo_env = SUMORLEnvironmentOptimized(
            sumo_cfg_path=sumo_cfg_path,
            use_gui=use_gui,
            max_steps=max_steps,
            use_subscription=True
        )
        
        # 初始化感知层 - Risk-Sensitive GNN
        print("\n[2/6] 初始化感知层 - Risk-Sensitive GNN...")
        self.risk_sensitive_gnn = RiskSensitiveGNN(
            node_feature_dim=9,
            edge_feature_dim=4,
            hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        # 初始化预测层 - Progressive World Model
        print("\n[3/6] 初始化预测层 - Progressive World Model...")
        self.progressive_world_model = ProgressiveWorldModel(
            state_dim=9,
            hidden_dim=world_model_hidden_dim,
            latent_dim=64,
            num_layers=2,
            device=device
        ).to(device)
        
        # 初始化决策层 - Influence-Driven Controller
        print("\n[4/6] 初始化决策层 - Influence-Driven Controller...")
        self.influence_controller = InfluenceDrivenController(
            state_dim=9,
            hidden_dim=128,
            top_k=top_k,
            device=device
        ).to(device)
        
        # 初始化 IDM 控制器（用于非 ICV 车辆）
        self.idm_controller = IDMController()
        
        # 初始化安全约束模块 - Dual-Mode Safety Shield
        print("\n[5/6] 初始化安全约束模块 - Dual-Mode Safety Shield...")
        self.safety_shield = DualModeSafetyShield(
            ttc_threshold=2.0,
            thw_threshold=1.5,
            emergency_deceleration=-4.5,
            max_acceleration=3.0,
            min_acceleration=-3.0
        )
        
        # 初始化安全奖励计算器
        self.safety_reward = SafetyReward(
            emergency_penalty=-100.0,
            warning_penalty=-10.0,
            safe_reward=1.0
        )
        
        # 初始化动作裁剪器
        self.action_clipper = ActionClipper(
            max_acceleration=3.0,
            min_acceleration=-3.0,
            max_speed=30.0,
            min_speed=0.0
        )
        
        # 初始化事件触发控制器
        print("\n[6/6] 初始化事件触发控制器...")
        self.event_controller = EventTriggeredController(
            control_interval=control_interval,
            emergency_ttc_threshold=1.5,
            high_risk_ttc_threshold=2.0,
            congestion_speed_threshold=5.0,
            congestion_ratio_threshold=0.6
        )
        
        # 系统状态
        self.current_step = 0
        self.last_control_time = 0.0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.safety_interventions = 0
        self.control_updates = 0
        
        # GNN 缓存
        self.gnn_cache = {}
        self.gnn_cache_timeout = 10  # 步数
        
        # 统计信息
        self.integration_stats = {
            'phase_times': {},
            'total_phases': 0,
            'successful_phases': 0,
            'failed_phases': 0
        }
        
        print("\n" + "=" * 80)
        print("✅ v4.0 完整集成系统初始化完成")
        print("=" * 80)
        print(f"设备: {device}")
        print(f"Top-K: {top_k}")
        print(f"ICV 渗透率: {icv_penetration}")
        print(f"控制间隔: {control_interval}s")
        print(f"最大步数: {max_steps}")
        print("=" * 80)
    
    def reset(self) -> Dict[str, Any]:
        """
        重置集成系统
        
        Returns:
            initial_observation: 初始观测
        """
        print("\n🔄 重置 v4.0 集成系统...")
        
        # 重置 SUMO 环境
        initial_observation = self.sumo_env.reset()
        
        # 重置系统状态
        self.current_step = 0
        self.last_control_time = 0.0
        self.total_reward = 0.0
        self.safety_interventions = 0
        self.control_updates = 0
        
        # 清空 GNN 缓存
        self.gnn_cache.clear()
        
        # 重置事件触发控制器
        self.event_controller.reset()
        
        print("✅ 系统重置完成")
        
        return initial_observation
    
    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步完整的集成流程
        
        Returns:
            observation: 观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        start_time = time.time()
        
        try:
            # Phase 1: 获取观测
            observation = self._phase_get_observation()
            
            # Phase 2: 感知层 - Risk-Sensitive GNN
            gnn_embeddings = self._phase_perception(observation)
            
            # Phase 3: 预测层 - Progressive World Model
            predictions = self._phase_prediction(observation, gnn_embeddings)
            
            # Phase 4: 决策层 - Influence-Driven Controller
            control_actions = self._phase_decision(observation, gnn_embeddings, predictions)
            
            # Phase 5: 安全约束 - Dual-Mode Safety Shield
            safe_actions, safety_info = self._phase_safety(observation, control_actions)
            
            # Phase 6: 事件触发检查
            should_control, event_type = self._phase_event_trigger(observation, safety_info)
            
            # Phase 7: 应用动作
            if should_control:
                self._phase_action_application(observation, safe_actions)
                self.control_updates += 1
            
            # 执行 SUMO 仿真步
            step_observation, step_reward, done, step_info = self.sumo_env.step({})
            
            # 计算总奖励
            total_reward = step_reward + self._compute_integration_reward(safety_info)
            self.total_reward += total_reward
            
            # 更新统计
            self.current_step += 1
            self.safety_interventions += safety_info.get('intervention_count', 0)
            
            # 构建返回信息
            info = {
                'step': self.current_step,
                'total_reward': self.total_reward,
                'step_reward': step_reward,
                'safety_interventions': self.safety_interventions,
                'control_updates': self.control_updates,
                'event_type': event_type.name if event_type else None,
                'vehicle_count': len(observation.get('vehicle_ids', [])),
                'processing_time': time.time() - start_time,
                'safety_info': safety_info,
                'predictions': predictions
            }
            
            return observation, total_reward, done, info
        
        except Exception as e:
            print(f"❌ 集成流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回错误状态
            info = {
                'step': self.current_step,
                'total_reward': self.total_reward,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            
            return observation, 0.0, True, info
    
    def _phase_get_observation(self) -> Dict[str, Any]:
        """
        Phase 1: 获取观测
        
        Returns:
            observation: 观测数据
        """
        phase_start = time.time()
        
        # 从 SUMO 环境获取观测
        observation = self.sumo_env._get_observation()
        
        # 更新统计
        self.integration_stats['phase_times']['get_observation'] = \
            self.integration_stats['phase_times'].get('get_observation', 0) + (time.time() - phase_start)
        
        return observation
    
    def _phase_perception(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Phase 2: 感知层 - Risk-Sensitive GNN
        
        Args:
            observation: 观测数据
            
        Returns:
            gnn_embeddings: GNN 嵌入
        """
        phase_start = time.time()
        
        vehicle_data = observation.get('vehicle_data', {})
        vehicle_ids = observation.get('vehicle_ids', [])
        
        if not vehicle_ids:
            return {'embeddings': None, 'importance': None}
        
        # 检查缓存
        cache_key = frozenset(vehicle_ids)
        if cache_key in self.gnn_cache:
            cached_data = self.gnn_cache[cache_key]
            if self.current_step - cached_data['step'] < self.gnn_cache_timeout:
                return cached_data['data']
        
        # 构建图数据
        node_features, edge_indices, edge_features = self._build_graph_data(vehicle_data, vehicle_ids)
        
        if node_features is None:
            return {'embeddings': None, 'importance': None}
        
        # 执行 GNN 前向传播
        with torch.no_grad():
            embeddings = self.risk_sensitive_gnn(node_features, edge_indices, edge_features)
        
        # 计算重要性分数（基于嵌入的范数）
        importance = torch.norm(embeddings, dim=1).cpu().numpy()
        
        # 缓存结果
        gnn_output = {
            'embeddings': embeddings,
            'importance': importance,
            'vehicle_ids': vehicle_ids
        }
        
        self.gnn_cache[cache_key] = {
            'data': gnn_output,
            'step': self.current_step
        }
        
        # 更新统计
        self.integration_stats['phase_times']['perception'] = \
            self.integration_stats['phase_times'].get('perception', 0) + (time.time() - phase_start)
        
        return gnn_output
    
    def _phase_prediction(self, 
                         observation: Dict[str, Any], 
                         gnn_embeddings: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Phase 3: 预测层 - Progressive World Model
        
        Args:
            observation: 观测数据
            gnn_embeddings: GNN 嵌入
            
        Returns:
            predictions: 预测结果
        """
        phase_start = time.time()
        
        if gnn_embeddings['embeddings'] is None:
            return {
                'next_states': None,
                'flow_evolution': None,
                'risk_evolution': None,
                'conflict_probability': None
            }
        
        vehicle_data = observation.get('vehicle_data', {})
        vehicle_ids = observation.get('vehicle_ids', [])
        
        # 构建状态张量
        state_tensor = self._build_state_tensor(vehicle_data, vehicle_ids)
        
        if state_tensor is None:
            return {
                'next_states': None,
                'flow_evolution': None,
                'risk_evolution': None,
                'conflict_probability': None
            }
        
        # 执行世界模型前向传播
        with torch.no_grad():
            predictions = self.progressive_world_model(state_tensor)
        
        # 更新统计
        self.integration_stats['phase_times']['prediction'] = \
            self.integration_stats['phase_times'].get('prediction', 0) + (time.time() - phase_start)
        
        return predictions
    
    def _phase_decision(self,
                       observation: Dict[str, Any],
                       gnn_embeddings: Dict[str, torch.Tensor],
                       predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 4: 决策层 - Influence-Driven Controller
        
        Args:
            observation: 观测数据
            gnn_embeddings: GNN 嵌入
            predictions: 预测结果
            
        Returns:
            control_actions: 控制动作
        """
        phase_start = time.time()
        
        vehicle_data = observation.get('vehicle_data', {})
        vehicle_ids = observation.get('vehicle_ids', [])
        
        if not vehicle_ids:
            return {
                'selected_vehicle_ids': [],
                'actions': None,
                'influence_scores': None
            }
        
        # 筛选 ICV 车辆
        icv_vehicles = [v for v in vehicle_ids 
                       if vehicle_data.get(v, {}).get('is_icv', False)]
        
        if not icv_vehicles:
            return {
                'selected_vehicle_ids': [],
                'actions': None,
                'influence_scores': None
            }
        
        # 构建状态张量
        state_tensor = self._build_state_tensor(vehicle_data, icv_vehicles)
        
        if state_tensor is None:
            return {
                'selected_vehicle_ids': [],
                'actions': None,
                'influence_scores': None
            }
        
        # 获取 GNN 重要性
        gnn_importance = gnn_embeddings.get('importance')
        
        # 执行影响力控制器前向传播
        with torch.no_grad():
            control_output = self.influence_controller(
                state_tensor, 
                gnn_importance
            )
        
        # 更新统计
        self.integration_stats['phase_times']['decision'] = \
            self.integration_stats['phase_times'].get('decision', 0) + (time.time() - phase_start)
        
        return control_output
    
    def _phase_safety(self,
                      observation: Dict[str, Any],
                      control_actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Phase 5: 安全约束 - Dual-Mode Safety Shield
        
        Args:
            observation: 观测数据
            control_actions: 控制动作
            
        Returns:
            safe_actions: 安全动作
            safety_info: 安全信息
        """
        phase_start = time.time()
        
        selected_ids = control_actions.get('selected_vehicle_ids', [])
        actions = control_actions.get('actions')
        
        if not selected_ids or actions is None:
            return {}, {'intervention_count': 0, 'level': 0}
        
        vehicle_data = observation.get('vehicle_data', {})
        
        # 初始化安全信息
        safety_info = {
            'intervention_count': 0,
            'level': 0,
            'emergency_count': 0,
            'warning_count': 0,
            'clipped_count': 0
        }
        
        # 应用安全屏障
        safe_actions = {}
        for i, veh_id in enumerate(selected_ids):
            if i >= actions.shape[0]:
                continue
            
            try:
                action_vec = actions[i]
                accel_action = action_vec[0].item() * 5.0  # [-1,1] -> [-5,5]
                
                # 获取车辆状态
                veh_data = vehicle_data.get(veh_id, {})
                speed = veh_data.get('speed', 0.0)
                road_id = veh_data.get('road_id', '')
                lane_id = veh_data.get('lane_id', '')
                
                # Level 1: 动作裁剪
                clipped_accel, is_clipped = self.action_clipper.clip_acceleration(
                    accel_action, speed
                )
                
                if is_clipped:
                    safety_info['clipped_count'] += 1
                    safety_info['level'] = max(safety_info['level'], 1)
                
                # Level 2: 紧急制动检查
                is_emergency, is_warning = self.safety_shield.check_emergency_conditions(
                    veh_id, speed, road_id, lane_id
                )
                
                if is_emergency:
                    clipped_accel = self.safety_shield.emergency_deceleration
                    safety_info['emergency_count'] += 1
                    safety_info['intervention_count'] += 1
                    safety_info['level'] = 2
                elif is_warning:
                    safety_info['warning_count'] += 1
                    safety_info['intervention_count'] += 1
                
                safe_actions[veh_id] = {
                    'acceleration': clipped_accel,
                    'is_emergency': is_emergency,
                    'is_warning': is_warning
                }
            
            except Exception as e:
                continue
        
        # 更新统计
        self.integration_stats['phase_times']['safety'] = \
            self.integration_stats['phase_times'].get('safety', 0) + (time.time() - phase_start)
        
        return safe_actions, safety_info
    
    def _phase_event_trigger(self,
                            observation: Dict[str, Any],
                            safety_info: Dict[str, Any]) -> Tuple[bool, Optional[EventType]]:
        """
        Phase 6: 事件触发检查
        
        Args:
            observation: 观测数据
            safety_info: 安全信息
            
        Returns:
            should_control: 是否应该执行控制
            event_type: 事件类型
        """
        phase_start = time.time()
        
        # 检查是否应该触发控制
        should_control, event_type = self.event_controller.should_trigger_control(
            observation, safety_info, self.current_step * self.sumo_env.step_length
        )
        
        # 更新统计
        self.integration_stats['phase_times']['event_trigger'] = \
            self.integration_stats['phase_times'].get('event_trigger', 0) + (time.time() - phase_start)
        
        return should_control, event_type
    
    def _phase_action_application(self,
                                 observation: Dict[str, Any],
                                 safe_actions: Dict[str, Any]):
        """
        Phase 7: 应用动作到 SUMO
        
        Args:
            observation: 观测数据
            safe_actions: 安全动作
        """
        phase_start = time.time()
        
        vehicle_data = observation.get('vehicle_data', {})
        
        # 应用控制动作到 ICV 车辆
        for veh_id, action in safe_actions.items():
            try:
                if veh_id not in vehicle_data:
                    continue
                
                accel = action['acceleration']
                current_speed = vehicle_data[veh_id].get('speed', 0.0)
                
                # 计算新速度
                new_speed = max(0.0, current_speed + accel * 0.1)
                
                # 应用速度控制
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, new_speed)
            
            except Exception as e:
                continue
        
        # 更新统计
        self.integration_stats['phase_times']['action_application'] = \
            self.integration_stats['phase_times'].get('action_application', 0) + (time.time() - phase_start)
    
    def _build_graph_data(self, 
                          vehicle_data: Dict[str, Any], 
                          vehicle_ids: List[str]) -> Tuple[Optional[torch.Tensor], 
                                                          Optional[torch.Tensor], 
                                                          Optional[torch.Tensor]]:
        """
        构建图数据
        
        Args:
            vehicle_data: 车辆数据
            vehicle_ids: 车辆ID列表
            
        Returns:
            node_features: 节点特征
            edge_indices: 边索引
            edge_features: 边特征
        """
        if not vehicle_ids:
            return None, None, None
        
        # 构建节点特征 [N, 9]
        node_features_list = []
        for veh_id in vehicle_ids:
            data = vehicle_data.get(veh_id, {})
            features = [
                data.get('speed', 0.0) / 30.0,
                data.get('acceleration', 0.0) / 5.0,
                data.get('lane_index', 0) / 3.0,
                data.get('position', 0.0) / 1000.0,
                data.get('is_icv', 0.0),
                1.0 if data.get('vehicle_class', '') == 'passenger' else 0.0,
                0.0, 0.0, 0.0  # 预留特征
            ]
            node_features_list.append(features)
        
        node_features = torch.tensor(node_features_list, dtype=torch.float32).to(self.device)
        
        # 构建边（基于空间邻近性）
        edge_indices = []
        edge_features = []
        
        for i, veh_id_i in enumerate(vehicle_ids):
            for j, veh_id_j in enumerate(vehicle_ids):
                if i >= j:
                    continue
                
                data_i = vehicle_data.get(veh_id_i, {})
                data_j = vehicle_data.get(veh_id_j, {})
                
                # 计算距离
                pos_i = data_i.get('position', 0.0)
                pos_j = data_j.get('position', 0.0)
                distance = abs(pos_i - pos_j)
                
                # 只连接距离小于 100 米的车辆
                if distance < 100.0:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    
                    # 计算边特征（TTC 和 THW）
                    speed_i = data_i.get('speed', 0.0)
                    speed_j = data_j.get('speed', 0.0)
                    
                    if speed_i > 0 and speed_j > 0:
                        ttc = distance / abs(speed_i - speed_j + 1e-6)
                        ttc = min(max(ttc, 0.1), 10.0)
                        thw = distance / max(speed_i, 1e-6)
                        thw = min(max(thw, 0.1), 5.0)
                    else:
                        ttc = 10.0
                        thw = 5.0
                    
                    edge_feature = [
                        1.0 / ttc,
                        1.0 / thw,
                        1.0 / (distance + 1e-6),
                        1.0
                    ]
                    
                    edge_features.append(edge_feature)
                    edge_features.append(edge_feature)
        
        if not edge_indices:
            return node_features, None, None
        
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        edge_features = torch.tensor(edge_features, dtype=torch.float32).to(self.device)
        
        return node_features, edge_indices, edge_features
    
    def _build_state_tensor(self, 
                           vehicle_data: Dict[str, Any], 
                           vehicle_ids: List[str]) -> Optional[torch.Tensor]:
        """
        构建状态张量
        
        Args:
            vehicle_data: 车辆数据
            vehicle_ids: 车辆ID列表
            
        Returns:
            state_tensor: 状态张量
        """
        if not vehicle_ids:
            return None
        
        state_list = []
        for veh_id in vehicle_ids:
            data = vehicle_data.get(veh_id, {})
            state = [
                data.get('speed', 0.0) / 30.0,
                data.get('acceleration', 0.0) / 5.0,
                data.get('lane_index', 0) / 3.0,
                data.get('position', 0.0) / 1000.0,
                data.get('is_icv', 0.0),
                1.0 if data.get('vehicle_class', '') == 'passenger' else 0.0,
                0.0, 0.0, 0.0  # 预留特征
            ]
            state_list.append(state)
        
        state_tensor = torch.tensor(state_list, dtype=torch.float32).to(self.device)
        
        return state_tensor
    
    def _compute_integration_reward(self, safety_info: Dict[str, Any]) -> float:
        """
        计算集成奖励
        
        Args:
            safety_info: 安全信息
            
        Returns:
            reward: 奖励值
        """
        # 基础奖励
        reward = 0.0
        
        # 安全干预惩罚
        reward += safety_info.get('emergency_count', 0) * self.safety_reward.emergency_penalty
        reward += safety_info.get('warning_count', 0) * self.safety_reward.warning_penalty
        
        # 控制更新奖励（鼓励有效控制）
        if safety_info.get('level', 0) == 0:
            reward += self.safety_reward.safe_reward
        
        return reward
    
    def get_integration_state(self) -> IntegrationState:
        """
        获取当前集成状态
        
        Returns:
            state: 集成状态
        """
        return IntegrationState(
            phase=IntegrationPhase.PHASE_7_ACTION_APPLICATION,
            timestamp=time.time(),
            vehicle_count=len(self.sumo_env.vehicle_ids),
            icv_count=sum([1 for v in self.sumo_env.vehicle_ids 
                          if self.sumo_env._get_observation().get('vehicle_data', {}).get(v, {}).get('is_icv', False)]),
            hv_count=0,
            avg_speed=0.0,
            total_reward=self.total_reward,
            safety_interventions=self.safety_interventions,
            control_updates=self.control_updates,
            processing_time=0.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'safety_interventions': self.safety_interventions,
            'control_updates': self.control_updates,
            'avg_reward': self.total_reward / max(self.current_step, 1),
            'integration_stats': self.integration_stats.copy(),
            'gnn_cache_size': len(self.gnn_cache)
        }
    
    def close(self):
        """关闭集成系统"""
        print("\n🔄 关闭 v4.0 集成系统...")
        
        # 关闭 SUMO 环境
        self.sumo_env.close()
        
        # 清空缓存
        self.gnn_cache.clear()
        
        print("✅ 系统已关闭")
    
    def save_checkpoint(self, path: str):
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            'risk_sensitive_gnn': self.risk_sensitive_gnn.state_dict(),
            'progressive_world_model': self.progressive_world_model.state_dict(),
            'influence_controller': self.influence_controller.state_dict(),
            'statistics': self.get_statistics()
        }
        
        torch.save(checkpoint, path)
        print(f"✅ 检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.risk_sensitive_gnn.load_state_dict(checkpoint['risk_sensitive_gnn'])
        self.progressive_world_model.load_state_dict(checkpoint['progressive_world_model'])
        self.influence_controller.load_state_dict(checkpoint['influence_controller'])
        
        print(f"✅ 检查点已加载: {path}")


def main():
    """主函数 - 演示 v4.0 完整集成系统"""
    print("🚀 v4.0 完整集成系统演示")
    
    # 创建集成系统
    integration = V4CompleteIntegration(
        sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
        device='cpu',  # 使用 CPU 以避免 CUDA 问题
        gnn_hidden_dim=256,
        gnn_num_heads=4,
        world_model_hidden_dim=256,
        top_k=5,
        icv_penetration=0.25,
        control_interval=10.0,
        use_gui=False,
        max_steps=100
    )
    
    try:
        # 重置系统
        observation = integration.reset()
        
        print(f"\n初始观测:")
        print(f"  车辆数: {len(observation['vehicle_ids'])}")
        print(f"  全局指标: {observation['global_metrics'][:4]}")
        
        # 运行仿真
        for step in range(50):
            observation, reward, done, info = integration.step()
            
            if step % 10 == 0:
                print(f"\n[Step {step+1}]")
                print(f"  奖励: {reward:.4f}")
                print(f"  总奖励: {info['total_reward']:.2f}")
                print(f"  安全干预: {info['safety_interventions']}")
                print(f"  控制更新: {info['control_updates']}")
                print(f"  事件类型: {info['event_type']}")
                print(f"  处理时间: {info['processing_time']:.4f}s")
            
            if done:
                break
        
        # 打印统计信息
        stats = integration.get_statistics()
        print(f"\n{'='*80}")
        print("📊 集成系统统计")
        print(f"{'='*80}")
        print(f"总步数: {stats['current_step']}")
        print(f"总奖励: {stats['total_reward']:.2f}")
        print(f"平均奖励: {stats['avg_reward']:.4f}")
        print(f"安全干预: {stats['safety_interventions']}")
        print(f"控制更新: {stats['control_updates']}")
        print(f"GNN 缓存大小: {stats['gnn_cache_size']}")
        print(f"\n各阶段耗时:")
        for phase, time_cost in stats['integration_stats']['phase_times'].items():
            print(f"  {phase}: {time_cost:.4f}s")
        print(f"{'='*80}")
        
    finally:
        integration.close()


if __name__ == "__main__":
    main()
