"""
SUMO集成模块
将神经网络控制器集成到SUMO仿真环境中
"""

import torch
import numpy as np
import traci
from typing import Dict, List, Tuple, Any, Optional
import os
import json


class SUMOEnvironment:
    """
    SUMO环境封装
    提供标准化的环境接口
    """
    
    def __init__(self, sumo_cfg: str, gui: bool = False):
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.connected = False
        
        # 统计信息
        self.step_count = 0
        self.total_reward = 0.0
        
    def start(self):
        """启动SUMO仿真"""
        if self.connected:
            return
        
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--no-warnings", "true"]
        
        traci.start(sumo_cmd)
        self.connected = True
        print(f"✅ SUMO仿真已启动: {self.sumo_cfg}")
    
    def step(self) -> Dict[str, Any]:
        """
        执行一步仿真
        Returns:
            observation: 观测数据
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        traci.simulationStep()
        self.step_count += 1
        
        # 收集观测
        observation = self._collect_observation()
        
        # 计算奖励
        reward = self._compute_reward(observation)
        self.total_reward += reward
        
        # 检查是否结束
        done = traci.simulation.getMinExpectedNumber() == 0
        
        info = {
            'step': self.step_count,
            'total_reward': self.total_reward
        }
        
        return observation, reward, done, info
    
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        if self.connected:
            traci.close()
        
        self.step_count = 0
        self.total_reward = 0.0
        self.start()
        
        return self._collect_observation()
    
    def close(self):
        """关闭环境"""
        if self.connected:
            traci.close()
            self.connected = False
    
    def _collect_observation(self) -> Dict[str, Any]:
        """
        收集观测数据
        使用配置的ICV车辆列表而非随机哈希
        """
        vehicle_ids = traci.vehicle.getIDList()
        
        vehicle_data = {}
        for veh_id in vehicle_ids:
            try:
                vehicle_data[veh_id] = {
                    'position': traci.vehicle.getLanePosition(veh_id),
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'acceleration': traci.vehicle.getAcceleration(veh_id),
                    'lane_index': traci.vehicle.getLaneIndex(veh_id),
                    'lane_id': traci.vehicle.getLaneID(veh_id),
                    'road_id': traci.vehicle.getRoadID(veh_id),
                    'is_icv': self._is_icv_vehicle(veh_id)
                }
            except Exception as e:
                import logging
                logging.warning(f"获取车辆 {veh_id} 数据失败: {e}")
                continue
        
        # 全局指标
        global_metrics = self._compute_global_metrics(vehicle_data)
        
        observation = {
            'vehicle_data': vehicle_data,
            'global_metrics': global_metrics,
            'vehicle_ids': list(vehicle_data.keys())
        }
        
        return observation
    
    def _is_icv_vehicle(self, veh_id: str) -> bool:
        """
        判断车辆是否为ICV（智能网联车）
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            is_icv: 是否为ICV
        """
        # 方法1: 从车辆类型判断（推荐）
        try:
            vehicle_class = traci.vehicle.getVehicleClass(veh_id)
            if vehicle_class == "custom1" or vehicle_class == "emergency":
                return True
        except:
            pass
        
        # 方法2: 从车辆类型ID判断
        try:
            vtype = traci.vehicle.getTypeID(veh_id)
            if "icv" in vtype.lower() or "autonomous" in vtype.lower():
                return True
        except:
            pass
        
        # 方法3: 使用确定性哈希（用于演示，生产环境应使用配置）
        import hashlib
        hash_value = int(hashlib.md5(veh_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < 25  # 25% ICV渗透率
    
    def _compute_global_metrics(self, vehicle_data: Dict[str, Any]) -> List[float]:
        """
        计算全局交通指标
        基于真实车辆状态计算16维指标
        """
        if not vehicle_data:
            return [0.0] * 16
        
        speeds = [v['speed'] for v in vehicle_data.values()]
        positions = [v['position'] for v in vehicle_data.values()]
        accelerations = [v.get('acceleration', 0.0) for v in vehicle_data.values()]
        
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        avg_accel = np.mean(accelerations)
        vehicle_count = len(vehicle_data)
        
        # ICV统计
        icv_vehicles = [v for v in vehicle_data.values() if v.get('is_icv', False)]
        hv_vehicles = [v for v in vehicle_data.values() if not v.get('is_icv', False)]
        
        icv_count = len(icv_vehicles)
        hv_count = len(hv_vehicles)
        
        icv_total_speed = sum([v['speed'] for v in icv_vehicles])
        hv_total_speed = sum([v['speed'] for v in hv_vehicles])
        
        metrics = [
            avg_speed,
            speed_std,
            avg_accel,
            float(vehicle_count),
            self.step_count * 0.1,  # 时间
            min(positions) if positions else 0.0,
            max(positions) if positions else 0.0,
            np.mean(positions) if positions else 0.0,
            float(icv_count),
            float(hv_count),
            icv_total_speed,
            hv_total_speed,
            avg_speed * vehicle_count,  # 总流量
            speed_std * vehicle_count,  # 总波动
            avg_accel * vehicle_count,  # 总加速度
            self.step_count % 100  # 周期性特征
        ]
        
        return metrics
    
    def _compute_reward(self, observation: Dict[str, Any]) -> float:
        """
        计算奖励 - 基于真实交通指标
        考虑：流量效率、安全、稳定性
        """
        vehicle_data = observation['vehicle_data']
        
        if not vehicle_data:
            return 0.0
        
        speeds = [v['speed'] for v in vehicle_data.values()]
        accelerations = [v.get('acceleration', 0.0) for v in vehicle_data.values()]
        
        # 1. 流量效率奖励
        avg_speed = np.mean(speeds) if speeds else 0.0
        flow_efficiency = avg_speed / 30.0  # 归一化到[0,1]
        
        # 2. 稳定性惩罚
        speed_std = np.std(speeds) if len(speeds) > 1 else 0.0
        accel_std = np.std(accelerations) if len(accelerations) > 1 else 0.0
        stability_penalty = (speed_std / 10.0 + accel_std / 5.0) * 0.5
        
        # 3. 安全评估
        safety_penalty = 0.0
        for vehicle in vehicle_data.values():
            speed = vehicle.get('speed', 0.0)
            accel = vehicle.get('acceleration', 0.0)
            
            # 检查危险驾驶行为
            if speed > 35.0:  # 超速
                safety_penalty += (speed - 35.0) * 0.1
            if accel < -4.0:  # 急刹车
                safety_penalty += (-accel - 4.0) * 0.2
            if accel > 3.0:  # 急加速
                safety_penalty += (accel - 3.0) * 0.1
        
        # 4. 综合奖励
        reward = (
            flow_efficiency * 10.0           # 流量效率权重
            - stability_penalty * 2.0         # 稳定性惩罚权重
            - safety_penalty * 5.0            # 安全惩罚权重
        )
        
        return reward


class SUMOIntegration:
    """
    SUMO集成控制器
    连接神经网络控制器和SUMO环境
    """
    
    def __init__(self, neural_controller, sumo_cfg: str, gui: bool = False):
        self.neural_controller = neural_controller
        self.sumo_env = SUMOEnvironment(sumo_cfg, gui)
        
        # 统计信息
        self.control_stats = {
            'total_interventions': 0,
            'total_emergency_interventions': 0,
            'total_controlled_vehicles': 0,
            'step_records': []
        }
        
        print("✅ SUMO集成控制器初始化完成")
    
    def run_episode(self, max_steps: int = 3600) -> Dict[str, Any]:
        """
        运行一个episode
        Args:
            max_steps: 最大步数
        Returns:
            episode_stats: episode统计信息
        """
        # 重置环境
        observation = self.sumo_env.reset()
        
        episode_reward = 0.0
        step = 0
        
        while step < max_steps:
            # 构建模型输入
            batch = self._build_model_input(observation, step)
            
            # 应用控制
            control_results = self._apply_control(batch, observation, step)
            
            # 执行仿真步
            observation, reward, done, info = self.sumo_env.step()
            
            episode_reward += reward
            step += 1
            
            # 记录统计
            if step % 100 == 0:
                print(f"[Step {step}] 奖励: {episode_reward:.2f}, "
                      f"控制车辆: {len(control_results['controlled_vehicles'])}")
            
            if done:
                break
        
        # 关闭环境
        self.sumo_env.close()
        
        episode_stats = {
            'total_reward': episode_reward,
            'total_steps': step,
            'avg_reward': episode_reward / step if step > 0 else 0.0,
            'control_stats': self.control_stats
        }
        
        print(f"📊 Episode完成! 总奖励: {episode_reward:.2f}, 平均奖励: {episode_stats['avg_reward']:.2f}")
        
        return episode_stats
    
    def _build_model_input(self, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
        """构建模型输入"""
        vehicle_data = observation['vehicle_data']
        vehicle_ids = observation['vehicle_ids']
        
        # 1. 收集车辆特征
        node_features = []
        is_icv_list = []
        
        for veh_id in vehicle_ids:
            vehicle = vehicle_data[veh_id]
            
            # 节点特征: [位置, 速度, 加速度, 车道, 剩余距离, 完成率, 类型, 时间, 步长]
            features = [
                vehicle.get('position', 0.0),
                vehicle.get('speed', 0.0),
                vehicle.get('acceleration', 0.0),
                vehicle.get('lane_index', 0),
                1000.0,  # 剩余距离（简化）
                0.5,  # 完成率（简化）
                1.0 if vehicle.get('is_icv', False) else 0.0,
                step * 0.1,
                0.1
            ]
            
            node_features.append(features)
            is_icv_list.append(vehicle.get('is_icv', False))
        
        # 2. 构建交互图
        edge_indices = []
        edge_features = []
        
        # 简化版：连接相近车辆
        for i, veh_id_i in enumerate(vehicle_ids):
            for j, veh_id_j in enumerate(vehicle_ids):
                if i == j:
                    continue
                
                pos_i = vehicle_data[veh_id_i].get('position', 0.0)
                pos_j = vehicle_data[veh_id_j].get('position', 0.0)
                speed_i = vehicle_data[veh_id_i].get('speed', 0.0)
                speed_j = vehicle_data[veh_id_j].get('speed', 0.0)
                
                distance = abs(pos_i - pos_j)
                if distance < 50:  # 50米内
                    edge_indices.append([i, j])
                    
                    rel_distance = distance
                    rel_speed = abs(speed_i - speed_j)
                    
                    ttc = rel_distance / max(rel_speed, 0.1) if rel_speed > 0 else 100
                    thw = rel_distance / max(speed_i, 0.1) if speed_i > 0 else 100
                    
                    edge_features.append([rel_distance, rel_speed, min(ttc, 10), min(thw, 10)])
        
        # 3. 转换为张量
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
    
    def _apply_control(self, batch: Dict[str, Any], observation: Dict[str, Any], step: int) -> Dict[str, Any]:
        """应用控制"""
        results = {
            'controlled_vehicles': [],
            'actions_applied': [],
            'safety_interventions': 0,
            'emergency_interventions': 0
        }
        
        # 模型推理
        with torch.no_grad():
            output = self.neural_controller(batch, step)
        
        # 应用安全动作
        for i, veh_id in enumerate(output['selected_vehicle_ids']):
            if veh_id not in observation['vehicle_data']:
                continue
            
            try:
                action = output['safe_actions'][i]
                accel_action = action[0].item() * 5.0  # [-1,1] -> [-5,5]
                
                current_speed = traci.vehicle.getSpeed(veh_id)
                new_speed = max(0.0, current_speed + accel_action * 0.1)
                
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, new_speed)
                
                results['controlled_vehicles'].append(veh_id)
                results['actions_applied'].append({
                    'acceleration': accel_action,
                    'new_speed': new_speed
                })
                
            except Exception as e:
                continue
        
        # 更新统计
        results['safety_interventions'] = output['level1_interventions'] + output['level2_interventions']
        results['emergency_interventions'] = output['level2_interventions']
        
        self.control_stats['total_interventions'] += results['safety_interventions']
        self.control_stats['total_emergency_interventions'] += results['emergency_interventions']
        self.control_stats['total_controlled_vehicles'] += len(results['controlled_vehicles'])
        
        if step % 100 == 0:
            self.control_stats['step_records'].append({
                'step': step,
                'controlled_vehicles': len(results['controlled_vehicles']),
                'interventions': results['safety_interventions'],
                'emergency_interventions': results['emergency_interventions']
            })
        
        return results
