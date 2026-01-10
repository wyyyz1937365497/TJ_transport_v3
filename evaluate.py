"""
评估脚本
评估控制器性能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm

from sumo_integration import create_sumo_controller


class Evaluator:
    """
    评估器
    """
    
    def __init__(self, model_path: str, config_path: str = None):
        # 创建控制器
        self.controller = create_sumo_controller(config_path)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.controller.device)
            self.controller.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 加载模型: {model_path}")
        
        # 评估指标
        self.metrics = {
            'total_steps': 0,
            'total_reward': 0.0,
            'avg_speed': 0.0,
            'speed_std': 0.0,
            'throughput': 0.0,
            'intervention_count': 0,
            'emergency_count': 0,
            'controlled_vehicles': 0
        }
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 3600) -> Dict[str, Any]:
        """
        评估控制器
        """
        print(f"🔍 开始评估 ({num_episodes} episodes, {max_steps} steps each)...")
        
        for episode in range(num_episodes):
            episode_reward = self._run_episode(max_steps)
            
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        # 计算平均指标
        avg_metrics = {
            'avg_reward': self.metrics['total_reward'] / num_episodes,
            'avg_speed': self.metrics['avg_speed'] / num_episodes,
            'speed_std': self.metrics['speed_std'] / num_episodes,
            'throughput': self.metrics['throughput'] / num_episodes,
            'avg_interventions': self.metrics['intervention_count'] / num_episodes,
            'avg_emergency': self.metrics['emergency_count'] / num_episodes,
            'avg_controlled': self.metrics['controlled_vehicles'] / num_episodes
        }
        
        print("\n📊 评估结果:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return avg_metrics
    
    def _run_episode(self, max_steps: int) -> float:
        """
        运行一个episode
        """
        # 重置控制器统计
        self.controller.reset_statistics()
        
        # 模拟SUMO环境
        total_reward = 0.0
        speeds = []
        
        for step in range(max_steps):
            # 生成模拟车辆数据
            vehicle_data = self._generate_vehicle_data(step)
            
            # 应用控制
            control_results = self.controller.apply_control(vehicle_data, step)
            
            # 计算奖励
            reward = self._calculate_reward(vehicle_data, control_results)
            total_reward += reward
            
            # 记录速度
            speeds.extend([v['speed'] for v in vehicle_data.values()])
            
            # 每100步输出
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}")
        
        # 更新指标
        self.metrics['total_steps'] += max_steps
        self.metrics['total_reward'] += total_reward
        self.metrics['avg_speed'] += np.mean(speeds)
        self.metrics['speed_std'] += np.std(speeds)
        self.metrics['throughput'] += len(speeds) / max_steps
        
        stats = self.controller.get_statistics()
        self.metrics['intervention_count'] += stats['total_interventions']
        self.metrics['emergency_count'] += stats['total_emergency_interventions']
        self.metrics['controlled_vehicles'] += stats['total_controlled_vehicles']
        
        return total_reward
    
    def _generate_vehicle_data(self, step: int) -> Dict[str, Any]:
        """
        生成模拟车辆数据
        注意：在实际评估中，应该使用真实的SUMO环境数据
        此方法仅用于演示，生产环境应从SUMO获取真实数据
        """
        import warnings
        warnings.warn(
            "使用模拟数据进行评估。在实际生产环境中，"
            "应该使用真实的SUMO仿真数据。",
            RuntimeWarning
        )
        
        vehicle_data = {}
        
        # 基于物理规律生成更真实的车辆数据
        num_vehicles = int(10 + 10 * np.sin(step * 0.01))
        
        for i in range(num_vehicles):
            veh_id = f"veh_{step}_{i}"
            
            # 基于车道和位置生成更合理的数据
            lane_index = np.random.randint(0, 3)
            position = np.random.uniform(0, 1000) + lane_index * 50  # 不同车道偏移
            
            # 速度基于位置（接近终点可能减速）
            base_speed = 15.0
            speed = base_speed + np.random.normal(0, 3.0)
            speed = max(5.0, min(30.0, speed))  # 限制在合理范围
            
            # 加速度基于速度变化
            acceleration = np.random.normal(0, 0.5)
            acceleration = max(-3.0, min(2.0, acceleration))
            
            # 剩余距离和完成率
            remaining_distance = max(0.0, 1000.0 - position)
            completion_rate = position / 1000.0
            
            vehicle_data[veh_id] = {
                'position': position,
                'speed': speed,
                'acceleration': acceleration,
                'lane_index': lane_index,
                'remaining_distance': remaining_distance,
                'completion_rate': completion_rate,
                'is_icv': np.random.random() < 0.25,  # 25% ICV
                'id': veh_id,
                'lane_id': f"lane_{lane_index}"
            }
        
        return vehicle_data
    
    def _calculate_reward(self, vehicle_data: Dict[str, Any],
                        control_results: Dict[str, Any]) -> float:
        """
        计算奖励 - 基于真实交通指标
        考虑：流量效率、安全、稳定性、控制成本
        """
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
        
        # 4. 控制成本
        intervention_cost = control_results.get('safety_interventions', 0) * 0.05
        emergency_cost = control_results.get('emergency_interventions', 0) * 0.5
        
        # 5. 综合奖励
        reward = (
            flow_efficiency * 10.0           # 流量效率权重
            - stability_penalty * 2.0         # 稳定性惩罚权重
            - safety_penalty * 5.0            # 安全惩罚权重
            - intervention_cost                # 控制成本
            - emergency_cost                   # 紧急干预成本
        )
        
        return reward
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """保存评估结果"""
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ 评估结果已保存到: {save_path}")


def main():
    """主函数"""
    # 配置
    model_path = 'models/traffic_controller_v1.pth'
    config_path = None
    num_episodes = 10
    max_steps = 3600
    results_path = 'results/evaluation_results.json'
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 创建评估器
    evaluator = Evaluator(model_path, config_path)
    
    # 评估
    results = evaluator.evaluate(num_episodes=num_episodes, max_steps=max_steps)
    
    # 保存结果
    evaluator.save_results(results, results_path)
    
    print("\n🎉 评估完成!")


if __name__ == "__main__":
    main()
