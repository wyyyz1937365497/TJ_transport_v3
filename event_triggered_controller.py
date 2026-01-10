"""
事件触发控制器
实现事件触发 + 定时兜底的控制周期
默认10秒，高危事件可中断
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import time


class EventType(Enum):
    """事件类型枚举"""
    NORMAL = "normal"           # 正常情况
    HIGH_RISK = "high_risk"     # 高风险事件
    CONGESTION = "congestion"   # 拥堵事件
    EMERGENCY = "emergency"     # 紧急事件
    CONTROL_UPDATE = "control_update"  # 控制更新事件


class EventTriggeredController:
    """
    事件触发控制器
    
    功能：
    1. 事件触发：检测高风险、拥堵等事件，触发控制更新
    2. 定时兜底：默认10秒周期性更新控制
    3. 高危中断：TTC < 2.0s 时立即中断并执行紧急控制
    """
    
    def __init__(self, 
                 control_interval: float = 10.0,  # 默认10秒
                 ttc_threshold: float = 2.0,
                 thw_threshold: float = 1.5,
                 congestion_threshold: float = 5.0,
                 speed_variance_threshold: float = 10.0):
        """
        初始化事件触发控制器
        
        Args:
            control_interval: 控制更新间隔（秒）
            ttc_threshold: TTC 阈值（秒）
            thw_threshold: THW 阈值（秒）
            congestion_threshold: 拥堵速度阈值（m/s）
            speed_variance_threshold: 速度方差阈值
        """
        self.control_interval = control_interval
        self.ttc_threshold = ttc_threshold
        self.thw_threshold = thw_threshold
        self.congestion_threshold = congestion_threshold
        self.speed_variance_threshold = speed_variance_threshold
        
        # 控制状态
        self.last_control_time = 0.0
        self.current_step = 0
        self.last_control_step = 0
        self.control_history = []
        
        # 事件统计
        self.event_counts = {
            EventType.NORMAL: 0,
            EventType.HIGH_RISK: 0,
            EventType.CONGESTION: 0,
            EventType.EMERGENCY: 0,
            EventType.CONTROL_UPDATE: 0
        }
        
        # 缓存上次控制动作
        self.last_control_action = None
        self.last_selected_vehicles = None
        
        print(f"✅ 事件触发控制器初始化完成")
        print(f"   控制间隔: {control_interval}s")
        print(f"   TTC 阈值: {ttc_threshold}s")
        print(f"   THW 阈值: {thw_threshold}s")
    
    def should_trigger_control(self, 
                            observation: Dict[str, Any],
                            current_time: float,
                            step: int) -> Tuple[bool, EventType, Dict[str, Any]]:
        """
        判断是否应该触发控制更新
        
        Args:
            observation: 当前观测
            current_time: 当前时间（秒）
            step: 当前步数
            
        Returns:
            should_trigger: 是否触发
            event_type: 事件类型
            event_info: 事件详细信息
        """
        self.current_step = step
        
        # 1. 检查紧急事件（最高优先级）
        emergency_result = self._check_emergency_events(observation)
        if emergency_result['is_emergency']:
            self.event_counts[EventType.EMERGENCY] += 1
            return True, EventType.EMERGENCY, emergency_result
        
        # 2. 检查高风险事件
        high_risk_result = self._check_high_risk_events(observation)
        if high_risk_result['is_high_risk']:
            self.event_counts[EventType.HIGH_RISK] += 1
            return True, EventType.HIGH_RISK, high_risk_result
        
        # 3. 检查拥堵事件
        congestion_result = self._check_congestion_events(observation)
        if congestion_result['is_congestion']:
            self.event_counts[EventType.CONGESTION] += 1
            return True, EventType.CONGESTION, congestion_result
        
        # 4. 定时兜底：检查是否达到控制间隔
        time_since_last_control = current_time - self.last_control_time
        if time_since_last_control >= self.control_interval:
            self.event_counts[EventType.CONTROL_UPDATE] += 1
            event_info = {
                'time_since_last_control': time_since_last_control,
                'scheduled_update': True
            }
            return True, EventType.CONTROL_UPDATE, event_info
        
        # 5. 正常情况：不需要触发
        self.event_counts[EventType.NORMAL] += 1
        return False, EventType.NORMAL, {'reason': 'normal_operation'}
    
    def _check_emergency_events(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查紧急事件
        
        Args:
            observation: 当前观测
            
        Returns:
            result: 紧急事件检测结果
        """
        vehicle_data = observation.get('vehicle_data', {})
        
        if not vehicle_data:
            return {'is_emergency': False}
        
        # 检查是否有车辆 TTC 或 THW 低于阈值
        emergency_vehicles = []
        
        for veh_id, vehicle in vehicle_data.items():
            # 获取前车
            leader = self._find_leader(vehicle, vehicle_data)
            
            if leader:
                # 计算 TTC
                ttc = self._calculate_ttc(vehicle, leader)
                
                # 计算 THW
                thw = self._calculate_thw(vehicle, leader)
                
                # 检查是否达到紧急阈值
                if ttc < self.ttc_threshold or thw < self.thw_threshold:
                    emergency_vehicles.append({
                        'vehicle_id': veh_id,
                        'ttc': ttc,
                        'thw': thw,
                        'speed': vehicle.get('speed', 0.0)
                    })
        
        if emergency_vehicles:
            return {
                'is_emergency': True,
                'emergency_vehicles': emergency_vehicles,
                'min_ttc': min(v['ttc'] for v in emergency_vehicles),
                'min_thw': min(v['thw'] for v in emergency_vehicles),
                'count': len(emergency_vehicles)
            }
        
        return {'is_emergency': False}
    
    def _check_high_risk_events(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查高风险事件
        
        Args:
            observation: 当前观测
            
        Returns:
            result: 高风险事件检测结果
        """
        vehicle_data = observation.get('vehicle_data', {})
        
        if not vehicle_data:
            return {'is_high_risk': False}
        
        # 检查 TTC < 3.0s 或 THW < 2.0s 的车辆
        high_risk_vehicles = []
        
        for veh_id, vehicle in vehicle_data.items():
            leader = self._find_leader(vehicle, vehicle_data)
            
            if leader:
                ttc = self._calculate_ttc(vehicle, leader)
                thw = self._calculate_thw(vehicle, leader)
                
                # 放宽的阈值（警告级别）
                if ttc < self.ttc_threshold * 1.5 or thw < self.thw_threshold * 1.5:
                    high_risk_vehicles.append({
                        'vehicle_id': veh_id,
                        'ttc': ttc,
                        'thw': thw,
                        'speed': vehicle.get('speed', 0.0)
                    })
        
        if high_risk_vehicles:
            return {
                'is_high_risk': True,
                'high_risk_vehicles': high_risk_vehicles,
                'min_ttc': min(v['ttc'] for v in high_risk_vehicles),
                'min_thw': min(v['thw'] for v in high_risk_vehicles),
                'count': len(high_risk_vehicles)
            }
        
        return {'is_high_risk': False}
    
    def _check_congestion_events(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查拥堵事件
        
        Args:
            observation: 当前观测
            
        Returns:
            result: 拥堵事件检测结果
        """
        vehicle_data = observation.get('vehicle_data', {})
        
        if not vehicle_data:
            return {'is_congestion': False}
        
        # 计算速度统计
        speeds = [v.get('speed', 0.0) for v in vehicle_data.values()]
        
        if not speeds:
            return {'is_congestion': False}
        
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        min_speed = np.min(speeds)
        
        # 检查拥堵条件
        is_congested = (
            avg_speed < self.congestion_threshold or  # 平均速度过低
            speed_std > self.speed_variance_threshold or  # 速度波动过大
            min_speed < 1.0  # 有车辆几乎停止
        )
        
        if is_congested:
            return {
                'is_congestion': True,
                'avg_speed': avg_speed,
                'speed_std': speed_std,
                'min_speed': min_speed,
                'vehicle_count': len(speeds),
                'congestion_reason': self._get_congestion_reason(avg_speed, speed_std, min_speed)
            }
        
        return {'is_congestion': False}
    
    def _get_congestion_reason(self, avg_speed: float, speed_std: float, min_speed: float) -> str:
        """获取拥堵原因"""
        reasons = []
        
        if avg_speed < self.congestion_threshold:
            reasons.append(f"低平均速度 ({avg_speed:.2f} m/s)")
        
        if speed_std > self.speed_variance_threshold:
            reasons.append(f"高速度波动 ({speed_std:.2f} m/s)")
        
        if min_speed < 1.0:
            reasons.append(f"车辆停止 ({min_speed:.2f} m/s)")
        
        return ", ".join(reasons)
    
    def _find_leader(self, ego: Dict[str, Any], all_vehicles: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """找到前车"""
        min_distance = float('inf')
        leader = None
        
        ego_pos = ego.get('position', 0.0)
        ego_lane_id = ego.get('lane_id', '')
        
        for veh_id, vehicle in all_vehicles.items():
            if veh_id == ego.get('id'):
                continue
            
            # 检查是否在同一车道
            if vehicle.get('lane_id') != ego_lane_id:
                continue
            
            # 检查是否在前方
            veh_pos = vehicle.get('position', 0.0)
            if veh_pos <= ego_pos:
                continue
            
            distance = veh_pos - ego_pos
            if distance < min_distance:
                min_distance = distance
                leader = vehicle
        
        return leader if min_distance < 100 else None  # 100米内
    
    def _calculate_ttc(self, ego: Dict[str, Any], leader: Dict[str, Any]) -> float:
        """计算碰撞时间 TTC"""
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
        """计算车头时距 THW"""
        ego_speed = ego.get('speed', 0.0)
        ego_pos = ego.get('position', 0.0)
        leader_pos = leader.get('position', 0.0)
        
        distance = leader_pos - ego_pos
        if ego_speed <= 0:
            return float('inf')
        
        thw = distance / ego_speed
        return max(0.1, thw)  # 防止除零
    
    def record_control(self, 
                     control_action: Dict[str, Any],
                     selected_vehicles: List[str],
                     event_type: EventType,
                     current_time: float,
                     step: int):
        """
        记录控制动作
        
        Args:
            control_action: 控制动作
            selected_vehicles: 选中的车辆
            event_type: 触发的事件类型
            current_time: 当前时间
            step: 当前步数
        """
        # 更新时间
        self.last_control_time = current_time
        self.last_control_step = step
        
        # 缓存动作
        self.last_control_action = control_action
        self.last_selected_vehicles = selected_vehicles
        
        # 记录历史
        self.control_history.append({
            'step': step,
            'time': current_time,
            'event_type': event_type.value,
            'selected_vehicles': selected_vehicles,
            'action': control_action
        })
        
        # 限制历史长度
        if len(self.control_history) > 1000:
            self.control_history = self.control_history[-1000:]
    
    def get_last_control(self) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
        """
        获取上次控制动作
        
        Returns:
            last_action: 上次控制动作
            last_vehicles: 上次选中的车辆
        """
        return self.last_control_action, self.last_selected_vehicles
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            stats: 统计信息字典
        """
        total_events = sum(self.event_counts.values())
        
        stats = {
            'total_events': total_events,
            'event_counts': {event.value: count for event, count in self.event_counts.items()},
            'event_percentages': {
                event.value: (count / total_events * 100) if total_events > 0 else 0.0
                for event, count in self.event_counts.items()
            },
            'control_history_length': len(self.control_history),
            'current_step': self.current_step,
            'last_control_step': self.last_control_step
        }
        
        return stats
    
    def reset(self):
        """重置控制器状态"""
        self.last_control_time = 0.0
        self.current_step = 0
        self.last_control_step = 0
        self.control_history = []
        self.last_control_action = None
        self.last_selected_vehicles = None
        
        # 重置事件统计
        for event_type in self.event_counts:
            self.event_counts[event_type] = 0


class EventTriggeredTrainer:
    """
    事件触发训练器
    结合事件触发机制和神经网络控制
    """
    
    def __init__(self, 
                 neural_controller,
                 event_controller: EventTriggeredController,
                 config: Dict[str, Any]):
        """
        初始化事件触发训练器
        
        Args:
            neural_controller: 神经网络控制器
            event_controller: 事件触发控制器
            config: 训练配置
        """
        self.neural_controller = neural_controller
        self.event_controller = event_controller
        self.config = config
        
        # 训练统计
        self.training_stats = {
            'total_steps': 0,
            'total_controls': 0,
            'event_triggered_controls': 0,
            'time_triggered_controls': 0,
            'emergency_interventions': 0
        }
        
        print("✅ 事件触发训练器初始化完成")
    
    def run_episode(self, env, max_steps: int = 3600) -> Dict[str, Any]:
        """
        运行一个 episode
        
        Args:
            env: SUMO 环境
            max_steps: 最大步数
            
        Returns:
            episode_stats: episode 统计信息
        """
        # 重置环境
        observation = env.reset()
        
        episode_reward = 0.0
        step = 0
        current_time = 0.0
        
        while step < max_steps:
            # 判断是否需要触发控制
            should_trigger, event_type, event_info = self.event_controller.should_trigger_control(
                observation, current_time, step
            )
            
            if should_trigger:
                # 构建批次
                batch = self._build_batch(observation, step)
                
                # 执行控制
                with torch.no_grad():
                    output = self.neural_controller(batch, step)
                
                # 记录控制
                self.event_controller.record_control(
                    control_action=output,
                    selected_vehicles=output['selected_vehicle_ids'],
                    event_type=event_type,
                    current_time=current_time,
                    step=step
                )
                
                # 更新统计
                self.training_stats['total_controls'] += 1
                
                if event_type == EventType.CONTROL_UPDATE:
                    self.training_stats['time_triggered_controls'] += 1
                else:
                    self.training_stats['event_triggered_controls'] += 1
                
                if event_type == EventType.EMERGENCY:
                    self.training_stats['emergency_interventions'] += 1
                
                # 应用控制动作
                action = {
                    'selected_vehicle_ids': output['selected_vehicle_ids'],
                    'safe_actions': output['safe_actions']
                }
                
                observation, reward, done, info = env.step(action)
            else:
                # 使用上次控制动作
                last_action, last_vehicles = self.event_controller.get_last_control()
                
                if last_action is not None and last_vehicles is not None:
                    action = {
                        'selected_vehicle_ids': last_vehicles,
                        'safe_actions': last_action
                    }
                    observation, reward, done, info = env.step(action)
                else:
                    # 没有上次动作，直接执行一步
                    observation, reward, done, info = env.step({})
            
            episode_reward += reward
            step += 1
            current_time += 0.1  # 假设步长为0.1秒
            
            # 进度报告
            if step % 100 == 0:
                print(f"[Step {step}] 奖励: {episode_reward:.2f}, "
                      f"事件: {event_type.value}")
            
            if done:
                break
        
        # 获取统计
        event_stats = self.event_controller.get_statistics()
        
        episode_stats = {
            'total_reward': episode_reward,
            'total_steps': step,
            'avg_reward': episode_reward / step if step > 0 else 0.0,
            'event_stats': event_stats,
            'training_stats': self.training_stats.copy()
        }
        
        print(f"📊 Episode完成! 总奖励: {episode_reward:.2f}")
        print(f"   总控制次数: {self.training_stats['total_controls']}")
        print(f"   事件触发: {self.training_stats['event_triggered_controls']}")
        print(f"   定时触发: {self.training_stats['time_triggered_controls']}")
        print(f"   紧急干预: {self.training_stats['emergency_interventions']}")
        
        return episode_stats
    
    def _build_batch(self, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
        """构建训练批次"""
        vehicle_data = observation['vehicle_data']
        vehicle_ids = observation['vehicle_ids']
        
        if not vehicle_data:
            return None
        
        # 收集车辆特征
        node_features = []
        is_icv_list = []
        
        for veh_id in vehicle_ids:
            vehicle = vehicle_data[veh_id]
            features = [
                vehicle.get('position', 0.0),
                vehicle.get('speed', 0.0),
                vehicle.get('acceleration', 0.0),
                vehicle.get('lane_index', 0),
                1000.0,
                0.5,
                1.0 if vehicle.get('is_icv', False) else 0.0,
                step * 0.1,
                0.1
            ]
            node_features.append(features)
            is_icv_list.append(vehicle.get('is_icv', False))
        
        # 构建边特征
        edge_indices = []
        edge_features = []
        
        for i, veh_id_i in enumerate(vehicle_ids):
            for j, veh_id_j in enumerate(vehicle_ids):
                if i == j:
                    continue
                
                pos_i = vehicle_data[veh_id_i].get('position', 0.0)
                pos_j = vehicle_data[veh_id_j].get('position', 0.0)
                speed_i = vehicle_data[veh_id_i].get('speed', 0.0)
                speed_j = vehicle_data[veh_id_j].get('speed', 0.0)
                
                distance = abs(pos_i - pos_j)
                if distance < 50:
                    edge_indices.append([i, j])
                    
                    rel_distance = distance
                    rel_speed = abs(speed_i - speed_j)
                    
                    ttc = rel_distance / max(rel_speed, 0.1) if rel_speed > 0 else 100
                    thw = rel_distance / max(speed_i, 0.1) if speed_i > 0 else 100
                    
                    edge_features.append([rel_distance, rel_speed, min(ttc, 10), min(thw, 10)])
        
        # 转换为张量
        device = next(self.neural_controller.parameters()).device
        
        batch = {
            'node_features': torch.tensor(node_features, dtype=torch.float32).to(device),
            'edge_indices': torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device) if edge_indices else torch.zeros((2, 0), dtype=torch.long).to(device),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32).to(device) if edge_features else torch.zeros((0, 4), dtype=torch.float32).to(device),
            'global_metrics': torch.tensor(observation['global_metrics'], dtype=torch.float32).unsqueeze(0).to(device),
            'vehicle_ids': vehicle_ids,
            'is_icv': torch.tensor(is_icv_list, dtype=torch.bool).to(device),
            'vehicle_states': {
                'ids': vehicle_ids,
                'data': vehicle_data
            }
        }
        
        return batch


def main():
    """主函数 - 演示事件触发控制器"""
    print("🚀 事件触发控制器演示")
    
    # 创建事件触发控制器
    event_controller = EventTriggeredController(
        control_interval=10.0,
        ttc_threshold=2.0,
        thw_threshold=1.5,
        congestion_threshold=5.0,
        speed_variance_threshold=10.0
    )
    
    # 模拟观测
    mock_observation = {
        'vehicle_data': {
            'veh_0': {
                'id': 'veh_0',
                'position': 100.0,
                'speed': 15.0,
                'acceleration': 0.0,
                'lane_index': 0,
                'lane_id': 'E1_0',
                'is_icv': True
            },
            'veh_1': {
                'id': 'veh_1',
                'position': 120.0,
                'speed': 14.0,
                'acceleration': -0.5,
                'lane_index': 0,
                'lane_id': 'E1_0',
                'is_icv': False
            }
        },
        'vehicle_ids': ['veh_0', 'veh_1'],
        'global_metrics': [15.0, 0.5, 0.2, 2, 10.0, 100.0, 200.0, 150.0, 1, 1, 15.0, 14.0, 30.0, 1.0, 0.4, 10]
    }
    
    # 测试事件触发
    for step in range(20):
        current_time = step * 0.1
        
        should_trigger, event_type, event_info = event_controller.should_trigger_control(
            mock_observation, current_time, step
        )
        
        if should_trigger:
            print(f"\n[Step {step}] 触发控制: {event_type.value}")
            print(f"   事件信息: {event_info}")
            
            # 记录控制
            event_controller.record_control(
                control_action={'test_action': True},
                selected_vehicles=['veh_0'],
                event_type=event_type,
                current_time=current_time,
                step=step
            )
    
    # 打印统计
    stats = event_controller.get_statistics()
    print(f"\n{'='*60}")
    print("📊 事件触发统计")
    print(f"{'='*60}")
    print(f"总事件数: {stats['total_events']}")
    print(f"事件分布: {stats['event_counts']}")
    print(f"事件百分比: {stats['event_percentages']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
