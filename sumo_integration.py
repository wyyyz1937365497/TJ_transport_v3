"""
集成到SUMO竞赛框架
将神经网络控制器集成到SUMO仿真环境中
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
import json

from neural_traffic_controller import TrafficController


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
            'device': 'cpu',  # 使用CPU以确保兼容性
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
            features = [
                vehicle.get('position', 0.0),
                vehicle.get('speed', 0.0),
                vehicle.get('acceleration', 0.0),
                vehicle.get('lane_index', 0),
                vehicle.get('remaining_distance', 1000.0),
                vehicle.get('completion_rate', 0.0),
                1.0 if vehicle.get('is_icv', False) else 0.0,  # ICV标志
                step * 0.1,  # 时间(秒)
                0.1  # 步长
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
                
                # 计算距离
                pos_i = vehicle_data[veh_id_i].get('position', 0.0)
                pos_j = vehicle_data[veh_id_j].get('position', 0.0)
                speed_i = vehicle_data[veh_id_i].get('speed', 0.0)
                speed_j = vehicle_data[veh_id_j].get('speed', 0.0)
                
                distance = abs(pos_i - pos_j)
                if distance < 50:  # 50米内
                    edge_indices.append([i, j])
                    
                    # 边特征: [相对距离, 相对速度, TTC, THW]
                    rel_distance = distance
                    rel_speed = abs(speed_i - speed_j)
                    
                    # 估算TTC和THW
                    ttc = rel_distance / max(rel_speed, 0.1) if rel_speed > 0 else 100
                    thw = rel_distance / max(speed_i, 0.1) if speed_i > 0 else 100
                    
                    edge_features.append([rel_distance, rel_speed, min(ttc, 10), min(thw, 10)])
        
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
                action = output['safe_actions'][i]
                accel_action = action[0].item() * 5.0  # [-1,1] -> [-5,5]
                lane_action = action[1].item() > 0.5  # 概率转布尔
                
                # 记录控制结果
                results['controlled_vehicles'].append(veh_id)
                results['actions_applied'].append({
                    'acceleration': accel_action,
                    'lane_change': lane_action,
                    'speed': vehicle_data[veh_id].get('speed', 0.0)
                })
                
            except Exception as e:
                continue
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        """
        return {
            'total_interventions': self.total_interventions,
            'total_emergency_interventions': self.total_emergency_interventions,
            'total_controlled_vehicles': self.total_controlled_vehicles
        }
    
    def reset_statistics(self):
        """
        重置统计信息
        """
        self.total_interventions = 0
        self.total_emergency_interventions = 0
        self.total_controlled_vehicles = 0


def create_sumo_controller(config_path: str = None) -> NeuralTrafficController:
    """
    创建SUMO控制器
    """
    return NeuralTrafficController(config_path)
