"""
安全层：双模态安全屏障 (Dual-mode Safety Shield)
Level 1：动作裁剪（加速度限幅、速度非负）
Level 2：TTC < 2.0s 时强制紧急制动，并在训练中施加巨大负奖励以避免进入高危状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


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
        
        # 统计信息
        self.register_buffer('total_level1_interventions', torch.tensor(0))
        self.register_buffer('total_level2_interventions', torch.tensor(0))
    
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
        
        # 更新统计
        self.total_level1_interventions += total_level1
        self.total_level2_interventions += total_level2
        
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
            current_speed = vehicle.get('speed', 0.0)
            
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
            leader_vehicle = self._find_leader(ego_vehicle, vehicle_states['data'])
            
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
    
    def _find_leader(self, ego: Dict[str, Any], all_vehicles: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """找到前车"""
        min_distance = float('inf')
        leader = None
        
        for veh_id, vehicle in all_vehicles.items():
            if veh_id == ego.get('id'):
                continue
            
            # 检查是否在同一车道
            if vehicle.get('lane_id') != ego.get('lane_id'):
                continue
            
            # 检查是否在前方
            ego_pos = ego.get('position', 0.0)
            veh_pos = vehicle.get('position', 0.0)
            if veh_pos <= ego_pos:
                continue
            
            distance = veh_pos - ego_pos
            if distance < min_distance:
                min_distance = distance
                leader = vehicle
        
        return leader if min_distance < 100 else None  # 100米内
    
    def _calculate_ttc(self, ego: Dict[str, Any], leader: Dict[str, Any]) -> float:
        """计算碰撞时间TTC"""
        ego_speed = ego.get('speed', 0.0)
        leader_speed = leader.get('speed', 0.0)
        ego_pos = ego.get('position', 0.0)
        leader_pos = leader.get('position', 0.0)
        
        relative_speed = ego_speed - leader_speed
        distance = leader_pos - ego_pos
        
        if relative_speed <= 0:
            return float('inf')  # 不会碰撞
        
        ttc = distance / relative_speed
        return max(0.1, ttc)  # 防止除零
    
    def _calculate_thw(self, ego: Dict[str, Any], leader: Dict[str, Any]) -> float:
        """计算车头时距THW"""
        ego_speed = ego.get('speed', 0.0)
        ego_pos = ego.get('position', 0.0)
        leader_pos = leader.get('position', 0.0)
        
        distance = leader_pos - ego_pos
        if ego_speed <= 0:
            return float('inf')
        
        thw = distance / ego_speed
        return max(0.1, thw)  # 防止除零
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_level1_interventions.zero_()
        self.total_level2_interventions.zero_()
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            'total_level1_interventions': self.total_level1_interventions.item(),
            'total_level2_interventions': self.total_level2_interventions.item()
        }


class SafetyReward(nn.Module):
    """
    安全奖励函数
    在训练中为高危状态施加巨大负奖励
    """
    
    def __init__(self, 
                 ttc_threshold: float = 2.0,
                 thw_threshold: float = 1.5,
                 emergency_penalty: float = -100.0,
                 warning_penalty: float = -10.0):
        super().__init__()
        
        self.ttc_threshold = ttc_threshold
        self.thw_threshold = thw_threshold
        self.emergency_penalty = emergency_penalty
        self.warning_penalty = warning_penalty
    
    def forward(self, ttc: torch.Tensor, thw: torch.Tensor) -> torch.Tensor:
        """
        计算安全奖励
        Args:
            ttc: [N] 碰撞时间
            thw: [N] 车头时距
        Returns:
            reward: [N] 安全奖励
        """
        # 紧急情况：TTC < 2.0s 或 THW < 1.5s
        emergency_mask = (ttc < self.ttc_threshold) | (thw < self.thw_threshold)
        
        # 警告情况：TTC < 3.0s 或 THW < 2.0s
        warning_mask = (ttc < self.ttc_threshold * 1.5) | (thw < self.thw_threshold * 1.5)
        warning_mask = warning_mask & (~emergency_mask)
        
        # 计算奖励
        reward = torch.zeros_like(ttc)
        reward[emergency_mask] = self.emergency_penalty
        reward[warning_mask] = self.warning_penalty
        
        return reward


class ActionClipper(nn.Module):
    """
    动作裁剪器
    确保动作在安全范围内
    """
    
    def __init__(self, 
                 max_accel: float = 2.0,
                 max_decel: float = -3.0,
                 min_speed: float = 0.0,
                 max_speed: float = 30.0):
        super().__init__()
        
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.min_speed = min_speed
        self.max_speed = max_speed
    
    def forward(self, actions: torch.Tensor, current_speeds: torch.Tensor) -> torch.Tensor:
        """
        裁剪动作
        Args:
            actions: [N, 2] 原始动作（加速度，换道概率）
            current_speeds: [N] 当前速度
        Returns:
            clipped_actions: [N, 2] 裁剪后的动作
        """
        clipped_actions = actions.clone()
        
        # 裁剪加速度
        clipped_actions[:, 0] = torch.clamp(
            clipped_actions[:, 0], 
            min=self.max_decel, 
            max=self.max_accel
        )
        
        # 确保速度非负
        predicted_speeds = current_speeds + clipped_actions[:, 0] * 0.1  # 0.1秒步长
        speed_mask = predicted_speeds < self.min_speed
        clipped_actions[speed_mask, 0] = (self.min_speed - current_speeds[speed_mask]) / 0.1
        
        # 确保速度不超过最大值
        speed_mask = predicted_speeds > self.max_speed
        clipped_actions[speed_mask, 0] = (self.max_speed - current_speeds[speed_mask]) / 0.1
        
        # 裁剪换道概率
        clipped_actions[:, 1] = torch.clamp(clipped_actions[:, 1], min=0.0, max=1.0)
        
        return clipped_actions
