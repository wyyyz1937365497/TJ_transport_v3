"""
基于SUMO-RL框架的训练脚本
使用SUMO仿真环境训练神经网络控制器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional
import os
import json
from tqdm import tqdm

from neural_traffic_controller import TrafficController
from sumo_rl_env import SUMORLEnvironment, create_sumo_env


class SUMORLTrainer:
    """
    基于SUMO-RL的训练器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        # 创建SUMO环境
        self.env = create_sumo_env(
            sumo_cfg_path=config['sumo_cfg_path'],
            use_gui=config.get('use_gui', False),
            max_steps=config.get('max_steps', 3600),
            seed=config.get('seed', None)
        )
        
        # 创建神经网络控制器
        self.controller = TrafficController(config['model']).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.controller.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 训练统计
        self.episode_rewards = []
        self.episode_stats = []
        self.best_reward = float('-inf')
        
        # 模型保存路径
        self.save_dir = config.get('save_dir', 'models')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("✅ 训练器初始化完成!")
    
    def build_batch(self, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        构建训练批次
        
        Args:
            observation: 环境观测
            step: 当前步数
        
        Returns:
            batch: 训练批次
        """
        vehicle_data = observation['vehicle_data']
        vehicle_ids = observation['vehicle_ids']
        
        if not vehicle_data:
            return None
        
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
        batch = {
            'node_features': torch.tensor(node_features, dtype=torch.float32).to(self.device),
            'edge_indices': torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device) if edge_indices else torch.zeros((2, 0), dtype=torch.long).to(self.device),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32).to(self.device) if edge_features else torch.zeros((0, 4), dtype=torch.float32).to(self.device),
            'global_metrics': torch.tensor(observation['global_metrics'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'vehicle_ids': vehicle_ids,
            'is_icv': torch.tensor(is_icv_list, dtype=torch.bool).to(self.device),
            'vehicle_states': {
                'ids': vehicle_ids,
                'data': vehicle_data
            }
        }
        
        return batch
    
    def run_episode(self, episode_num: int) -> Dict[str, float]:
        """
        运行一个episode
        
        Args:
            episode_num: episode编号
        
        Returns:
            stats: episode统计信息
        """
        # 重置环境
        observation = self.env.reset()
        
        episode_reward = 0.0
        step = 0
        
        # 设置世界模型阶段
        phase = self.config.get('training_phase', 1)
        self.controller.world_model.set_phase(phase)
        
        # Episode循环
        while step < self.env.max_steps:
            # 构建批次
            batch = self.build_batch(observation, step)
            
            if batch is None:
                # 没有车辆，直接执行一步
                observation, reward, done, info = self.env.step({})
                episode_reward += reward
                step += 1
                continue
            
            # 前向传播
            with torch.no_grad():
                output = self.controller(batch, step)
            
            # 构建动作
            action = {
                'selected_vehicle_ids': output['selected_vehicle_ids'],
                'safe_actions': output['safe_actions']
            }
            
            # 执行一步
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            step += 1
            
            # 进度报告
            if step % 100 == 0:
                print(f"[Episode {episode_num}] Step {step}/{self.env.max_steps}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Vehicles: {info['vehicle_count']}")
            
            if done:
                break
        
        # 获取统计信息
        env_stats = self.env.get_episode_statistics()
        
        stats = {
            'episode': episode_num,
            'total_reward': episode_reward,
            'total_steps': step,
            'avg_reward': episode_reward / max(step, 1),
            'vehicle_count': env_stats['vehicle_count']
        }
        
        return stats
    
    def train(self, num_episodes: int = 100):
        """
        训练主循环
        
        Args:
            num_episodes: 训练episode数
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始训练")
        print(f"{'='*60}")
        print(f"总episodes: {num_episodes}")
        print(f"最大步数: {self.env.max_steps}")
        print(f"学习率: {self.config.get('learning_rate', 1e-4)}")
        print(f"训练阶段: {self.config.get('training_phase', 1)}")
        print(f"{'='*60}\n")
        
        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            # 运行episode
            stats = self.run_episode(episode)
            
            # 记录统计
            self.episode_rewards.append(stats['total_reward'])
            self.episode_stats.append(stats)
            
            # 更新学习率
            self.scheduler.step(stats['avg_reward'])
            
            # 保存最佳模型
            if stats['avg_reward'] > self.best_reward:
                self.best_reward = stats['avg_reward']
                self.save_model('best_model.pth')
                print(f"🎉 新的最佳模型! 平均奖励: {self.best_reward:.2f}")
            
            # 定期保存
            if episode % 10 == 0:
                self.save_model(f'checkpoint_episode_{episode}.pth')
                self.save_training_log()
            
            # 打印统计
            print(f"\nEpisode {episode} 完成:")
            print(f"  总奖励: {stats['total_reward']:.2f}")
            print(f"  平均奖励: {stats['avg_reward']:.4f}")
            print(f"  总步数: {stats['total_steps']}")
            print(f"  车辆数: {stats['vehicle_count']}")
            print(f"  最佳平均奖励: {self.best_reward:.4f}\n")
        
        print(f"\n{'='*60}")
        print("✅ 训练完成!")
        print(f"{'='*60}")
        print(f"最终最佳平均奖励: {self.best_reward:.4f}")
        
        # 保存最终模型
        self.save_model('final_model.pth')
        self.save_training_log()
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.save_dir, filename)
        torch.save({
            'episode': len(self.episode_rewards),
            'model_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'config': self.config
        }, model_path)
        print(f"💾 模型已保存: {model_path}")
    
    def save_training_log(self):
        """保存训练日志"""
        log_path = os.path.join(self.save_dir, 'training_log.json')
        
        log_data = {
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_stats': self.episode_stats,
            'best_reward': float(self.best_reward)
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📊 训练日志已保存: {log_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """主函数"""
    # 默认配置
    default_config = {
        'sumo_cfg_path': '仿真环境-初赛/sumo.sumocfg',
        'use_gui': False,
        'max_steps': 3600,
        'seed': 42,
        
        'model': {
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
            'device': 'cpu'
        },
        
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_episodes': 100,
        'training_phase': 1,
        'save_dir': 'models'
    }
    
    # 加载配置文件（如果存在）
    config_path = 'train_sumo_rl_config.json'
    if os.path.exists(config_path):
        config = load_config(config_path)
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    else:
        config = default_config
        # 保存默认配置
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"📝 默认配置已保存到: {config_path}")
    
    # 创建训练器
    trainer = SUMORLTrainer(config)
    
    # 开始训练
    trainer.train(num_episodes=config['num_episodes'])


if __name__ == "__main__":
    main()
