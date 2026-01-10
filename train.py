"""
训练脚本
包含三阶段训练流程：
Phase 1：世界模型预训练
Phase 2：安全RL训练
Phase 3：约束优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from neural_traffic_controller import TrafficController
from sumo_integration import create_sumo_controller


class TrafficDataset(Dataset):
    """
    交通数据集
    用于训练世界模型
    """
    
    def __init__(self, data_path: str = None, num_samples: int = 1000):
        self.num_samples = num_samples
        
        # 如果没有提供数据路径，生成模拟数据
        if data_path is None or not os.path.exists(data_path):
            self.data = self._generate_mock_data(num_samples)
        else:
            self.data = self._load_data(data_path)
    
    def _generate_mock_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """生成模拟数据"""
        data = []
        for _ in range(num_samples):
            # 模拟车辆数据
            num_vehicles = np.random.randint(5, 20)
            vehicle_data = {}
            
            for i in range(num_vehicles):
                veh_id = f"veh_{i}"
                vehicle_data[veh_id] = {
                    'position': np.random.uniform(0, 1000),
                    'speed': np.random.uniform(5, 25),
                    'acceleration': np.random.uniform(-2, 2),
                    'lane_index': np.random.randint(0, 3),
                    'remaining_distance': np.random.uniform(100, 1000),
                    'completion_rate': np.random.uniform(0, 1),
                    'is_icv': np.random.random() < 0.25,
                    'id': veh_id,
                    'lane_id': f"lane_{np.random.randint(0, 3)}"
                }
            
            data.append({
                'vehicle_data': vehicle_data,
                'step': np.random.randint(0, 3600)
            })
        
        return data
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class Trainer:
    """
    训练器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化模型
        self.model = TrafficController(config['model']).to(config['device'])
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # 训练统计
        self.training_stats = {
            'phase1_rewards': [],
            'phase2_rewards': [],
            'phase3_rewards': []
        }
    
    def train_phase1(self, num_epochs: int, batch_size: int = 64):
        """
        Phase 1: 世界模型预训练
        """
        print("🔄 Phase 1: 世界模型预训练...")
        
        # 设置世界模型为Phase 1
        self.model.world_model.set_phase(1)
        
        # 创建数据集
        dataset = TrafficDataset(num_samples=1000)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # 模拟训练过程
                loss = self._train_phase1_step(batch_data)
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Phase 1 - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # 更新学习率
            self.scheduler.step(-avg_loss)
        
        print("✅ Phase 1 完成!")
    
    def _train_phase1_step(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Phase 1 单步训练
        """
        self.optimizer.zero_grad()
        
        # 模拟前向传播
        vehicle_data = batch_data['vehicle_data']
        step = batch_data['step']
        
        # 构建输入
        batch = self._build_training_batch(vehicle_data, step)
        
        # 前向传播
        gnn_embedding = self.model.risk_gnn(self.model._build_graph(batch))
        predictions = self.model.world_model(gnn_embedding)
        
        # 计算损失
        targets = self._generate_targets(gnn_embedding)
        loss = self.mse_loss(predictions, targets)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def train_phase2(self, num_epochs: int, batch_size: int = 64):
        """
        Phase 2: 安全RL训练
        """
        print("🔄 Phase 2: 安全RL训练...")
        
        # 设置世界模型为Phase 2
        self.model.world_model.set_phase(2)
        
        # 创建数据集
        dataset = TrafficDataset(num_samples=1000)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        for epoch in range(num_epochs):
            total_reward = 0.0
            num_batches = 0
            
            for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # 模拟训练过程
                reward = self._train_phase2_step(batch_data)
                
                total_reward += reward
                num_batches += 1
            
            avg_reward = total_reward / num_batches if num_batches > 0 else 0.0
            self.training_stats['phase2_rewards'].append(avg_reward)
            
            print(f"Phase 2 - Epoch {epoch+1}/{num_epochs}, Reward: {avg_reward:.4f}")
            
            # 更新学习率
            self.scheduler.step(avg_reward)
            
            # 更新拉格朗日乘子
            if epoch % 5 == 0:
                self.model.update_lagrange_multiplier(avg_reward)
        
        print("✅ Phase 2 完成!")
    
    def _train_phase2_step(self, batch_data: Dict[str, Any]) -> float:
        """
        Phase 2 单步训练
        """
        self.optimizer.zero_grad()
        
        # 模拟前向传播
        vehicle_data = batch_data['vehicle_data']
        step = batch_data['step']
        
        # 构建输入
        batch = self._build_training_batch(vehicle_data, step)
        
        # 前向传播
        output = self.model(batch, step)
        
        # 计算奖励
        reward = self._calculate_reward(output, vehicle_data)
        
        # 反向传播（简化版）
        loss = -reward
        loss.backward()
        self.optimizer.step()
        
        return reward.item()
    
    def train_phase3(self, num_epochs: int, batch_size: int = 64):
        """
        Phase 3: 约束优化
        """
        print("🔄 Phase 3: 约束优化...")
        
        # 创建数据集
        dataset = TrafficDataset(num_samples=1000)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        for epoch in range(num_epochs):
            total_reward = 0.0
            num_batches = 0
            
            for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # 模拟训练过程
                reward = self._train_phase3_step(batch_data)
                
                total_reward += reward
                num_batches += 1
            
            avg_reward = total_reward / num_batches if num_batches > 0 else 0.0
            self.training_stats['phase3_rewards'].append(avg_reward)
            
            print(f"Phase 3 - Epoch {epoch+1}/{num_epochs}, Reward: {avg_reward:.4f}")
            
            # 更新学习率
            self.scheduler.step(avg_reward)
        
        print("✅ Phase 3 完成!")
    
    def _train_phase3_step(self, batch_data: Dict[str, Any]) -> float:
        """
        Phase 3 单步训练
        """
        self.optimizer.zero_grad()
        
        # 模拟前向传播
        vehicle_data = batch_data['vehicle_data']
        step = batch_data['step']
        
        # 构建输入
        batch = self._build_training_batch(vehicle_data, step)
        
        # 前向传播
        output = self.model(batch, step)
        
        # 计算约束奖励
        reward = self._calculate_constrained_reward(output, vehicle_data)
        
        # 反向传播
        loss = -reward
        loss.backward()
        self.optimizer.step()
        
        return reward.item()
    
    def _build_training_batch(self, vehicle_data: Dict[str, Any], step: int) -> Dict[str, Any]:
        """构建训练批次"""
        # 简化版：使用sumo_integration中的方法
        from sumo_integration import NeuralTrafficController
        
        controller = NeuralTrafficController()
        return controller.build_model_input(vehicle_data, step)
    
    def _generate_targets(self, gnn_embedding: torch.Tensor) -> torch.Tensor:
        """生成训练目标"""
        # 简化版：使用噪声版本的嵌入作为目标
        noise = torch.randn_like(gnn_embedding) * 0.1
        return gnn_embedding + noise
    
    def _calculate_reward(self, output: Dict[str, Any], vehicle_data: Dict[str, Any]) -> torch.Tensor:
        """计算奖励"""
        # 简化版奖励函数
        avg_speed = np.mean([v['speed'] for v in vehicle_data.values()])
        speed_std = np.std([v['speed'] for v in vehicle_data.values()])
        intervention_cost = (output['level1_interventions'] + output['level2_interventions']) * 0.1
        
        # 奖励 = 速度奖励 - 不稳定惩罚 - 干预成本
        reward = avg_speed * 0.1 - speed_std * 0.5 - intervention_cost
        
        return torch.tensor(reward, dtype=torch.float32)
    
    def _calculate_constrained_reward(self, output: Dict[str, Any], vehicle_data: Dict[str, Any]) -> torch.Tensor:
        """计算约束奖励"""
        # 基础奖励
        base_reward = self._calculate_reward(output, vehicle_data)
        
        # 约束惩罚
        constraint_penalty = self.model.lagrange_multiplier * (
            (output['level1_interventions'] + output['level2_interventions']) / 100.0
        )
        
        return base_reward - constraint_penalty
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        print(f"✅ 模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', {})
        print(f"✅ 模型已从 {path} 加载")


def main():
    """主函数"""
    # 加载配置
    config = {
        'training': {
            'phase1_epochs': 50,
            'phase2_epochs': 200,
            'phase3_epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.0003,
            'weight_decay': 0.0001
        },
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
            'top_k': 5
        },
        'safety': {
            'ttc_threshold': 2.0,
            'thw_threshold': 1.5,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'emergency_decel': -5.0,
            'max_lane_change_speed': 5.0
        },
        'constraint': {
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'alpha': 1.0,
            'beta': 5.0
        },
        'device': 'cpu',
        'save_path': 'models/traffic_controller_v1.pth'
    }
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # Phase 1: 世界模型预训练
    trainer.train_phase1(
        num_epochs=config['training']['phase1_epochs'],
        batch_size=config['training']['batch_size']
    )
    
    # Phase 2: 安全RL训练
    trainer.train_phase2(
        num_epochs=config['training']['phase2_epochs'],
        batch_size=config['training']['batch_size']
    )
    
    # Phase 3: 约束优化
    trainer.train_phase3(
        num_epochs=config['training']['phase3_epochs'],
        batch_size=config['training']['batch_size']
    )
    
    # 保存模型
    trainer.save_model(config['save_path'])
    
    print("🎉 训练完成!")


if __name__ == "__main__":
    main()
