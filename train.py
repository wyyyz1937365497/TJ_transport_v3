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
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from neural_traffic_controller import TrafficController


class TrafficDataset(Dataset):
    """
    交通数据集
    用于训练世界模型
    从真实SUMO仿真数据或预收集的数据集加载
    """
    
    def __init__(self, data_path: str = None, num_samples: int = 1000,
                 validate_data: bool = True):
        self.num_samples = num_samples
        self.validate_data = validate_data
        
        # 优先从真实数据路径加载
        if data_path is not None and os.path.exists(data_path):
            self.data = self._load_data(data_path)
            if validate_data:
                self._validate_data()
        else:
            # 如果没有真实数据，抛出错误而非生成模拟数据
            if data_path is None:
                raise ValueError(
                    "必须提供数据路径。真实训练需要从SUMO仿真收集的交通数据。"
                    "请先运行数据收集脚本或提供预收集的数据集。"
                )
            else:
                raise FileNotFoundError(
                    f"数据文件不存在: {data_path}。"
                    "请确保已正确收集并保存交通数据。"
                )
    
    def _validate_data(self):
        """验证数据完整性"""
        if not self.data:
            raise ValueError("数据集为空")
        
        # 检查必要字段
        required_fields = ['vehicle_data', 'step']
        for i, sample in enumerate(self.data):
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"样本 {i} 缺少必要字段: {field}")
            
            # 验证车辆数据
            vehicle_data = sample['vehicle_data']
            if not vehicle_data:
                continue
            
            required_vehicle_fields = ['position', 'speed', 'acceleration',
                                      'lane_index', 'is_icv', 'id']
            for veh_id, vehicle in vehicle_data.items():
                for field in required_vehicle_fields:
                    if field not in vehicle:
                        raise ValueError(
                            f"样本 {i}, 车辆 {veh_id} 缺少必要字段: {field}"
                        )
                
                # 验证数据范围
                if not (0 <= vehicle['speed'] <= 50):  # 合理速度范围
                    raise ValueError(
                        f"样本 {i}, 车辆 {veh_id} 速度异常: {vehicle['speed']}"
                    )
                if not (-10 <= vehicle['acceleration'] <= 10):  # 合理加速度范围
                    raise ValueError(
                        f"样本 {i}, 车辆 {veh_id} 加速度异常: {vehicle['acceleration']}"
                    )
        
        print(f"✅ 数据验证通过: {len(self.data)} 个样本")
    
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
    训练器 - 支持混合精度训练
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
            verbose=False  # 移除废弃的verbose参数或设置为False
        )
        
        # 混合精度训练
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
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
        num_workers = self.config['training'].get('num_workers', 2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        
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
        Phase 1 单步训练 - 支持混合精度
        训练世界模型预测下一时刻状态
        """
        self.optimizer.zero_grad()
        
        # 获取车辆数据和步骤
        vehicle_data = batch_data['vehicle_data']
        step = batch_data['step']
        
        # 构建输入批次
        batch = self._build_training_batch(vehicle_data, step)
        
        if batch is None or len(vehicle_data) == 0:
            return torch.tensor(0.0, device=self.config['device'])
        
        # 使用混合精度训练
        if self.use_amp and self.config['device'] == 'cuda':
            with torch.amp.autocast('cuda'):
                # 前向传播
                gnn_embedding = self.model.risk_gnn(self.model._build_graph(batch))
                predictions = self.model.world_model(gnn_embedding)
                
                # 计算损失 - 基于真实车辆状态生成目标
                targets = self._generate_targets(gnn_embedding, vehicle_data)
                loss = self.mse_loss(predictions, targets)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 前向传播
            gnn_embedding = self.model.risk_gnn(self.model._build_graph(batch))
            predictions = self.model.world_model(gnn_embedding)
            
            # 计算损失
            targets = self._generate_targets(gnn_embedding, vehicle_data)
            loss = self.mse_loss(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
        num_workers = self.config['training'].get('num_workers', 2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        
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
        Phase 2 单步训练 - 安全RL训练
        使用策略梯度方法优化控制策略
        """
        self.optimizer.zero_grad()
        
        # 获取车辆数据和步骤
        vehicle_data = batch_data['vehicle_data']
        step = batch_data['step']
        
        # 构建输入批次
        batch = self._build_training_batch(vehicle_data, step)
        
        if batch is None or len(vehicle_data) == 0:
            return 0.0
        
        # 使用混合精度训练
        if self.use_amp and self.config['device'] == 'cuda':
            with torch.amp.autocast('cuda'):
                # 前向传播
                output = self.model(batch, step)
                
                # 计算奖励 - 基于真实交通指标
                reward = self._calculate_reward(output, vehicle_data)
                
                # 策略梯度损失（简化版REINFORCE）
                # 在实际应用中，应该使用更复杂的RL算法如PPO
                loss = -reward  # 最大化奖励 = 最小化负奖励
        
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 前向传播
            output = self.model(batch, step)
            
            # 计算奖励
            reward = self._calculate_reward(output, vehicle_data)
            
            # 策略梯度损失
            loss = -reward
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        return reward.item()
    
    def train_phase3(self, num_epochs: int, batch_size: int = 64):
        """
        Phase 3: 约束优化
        """
        print("🔄 Phase 3: 约束优化...")
        
        # 创建数据集
        dataset = TrafficDataset(num_samples=1000)
        num_workers = self.config['training'].get('num_workers', 2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        
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
        Phase 3 单步训练 - 约束优化
        使用拉格朗日对偶方法处理安全约束
        """
        self.optimizer.zero_grad()
        
        # 获取车辆数据和步骤
        vehicle_data = batch_data['vehicle_data']
        step = batch_data['step']
        
        # 构建输入批次
        batch = self._build_training_batch(vehicle_data, step)
        
        if batch is None or len(vehicle_data) == 0:
            return 0.0
        
        # 前向传播
        output = self.model(batch, step)
        
        # 计算约束奖励
        reward = self._calculate_constrained_reward(output, vehicle_data)
        
        # 计算约束违反
        constraint_violation = (
            (output['level1_interventions'] + output['level2_interventions']) /
            max(len(vehicle_data), 1) - self.cost_limit
        )
        
        # 拉格朗日损失
        lagrangian_loss = -reward + self.model.lagrange_multiplier * constraint_violation
        
        # 反向传播
        lagrangian_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return reward.item()
    
    def _build_training_batch(self, vehicle_data: Dict[str, Any], step: int) -> Dict[str, Any]:
        """构建训练批次"""
        vehicle_ids = list(vehicle_data.keys())
        
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
                vehicle.get('remaining_distance', 1000.0),
                vehicle.get('completion_rate', 0.5),
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
        
        # 3. 计算全局指标
        if vehicle_data:
            speeds = [v['speed'] for v in vehicle_data.values()]
            positions = [v['position'] for v in vehicle_data.values()]
            
            avg_speed = np.mean(speeds)
            speed_std = np.std(speeds)
            vehicle_count = len(vehicle_data)
            icv_count = sum(1 for v in vehicle_data.values() if v.get('is_icv', False))
            
            global_metrics = [
                avg_speed,
                speed_std,
                0.0,  # 平均加速度
                float(vehicle_count),
                step * 0.1,  # 时间
                min(positions) if positions else 0.0,
                max(positions) if positions else 0.0,
                np.mean(positions) if positions else 0.0,
                float(icv_count),
                float(vehicle_count - icv_count),
                0.0,  # ICV总速度
                0.0,  # 非ICV总速度
                avg_speed * vehicle_count,  # 总流量
                speed_std * vehicle_count,  # 总波动
                0.0,  # 总加速度
                step % 100  # 周期性特征
            ]
        else:
            global_metrics = [0.0] * 16
        
        # 4. 转换为张量
        device = self.config['device']
        
        batch = {
            'node_features': torch.tensor(node_features, dtype=torch.float32).to(device),
            'edge_indices': torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device) if edge_indices else torch.zeros((2, 0), dtype=torch.long).to(device),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32).to(device) if edge_features else torch.zeros((0, 4), dtype=torch.float32).to(device),
            'global_metrics': torch.tensor(global_metrics, dtype=torch.float32).unsqueeze(0).to(device),
            'vehicle_ids': vehicle_ids,
            'is_icv': torch.tensor(is_icv_list, dtype=torch.bool).to(device),
            'vehicle_states': {
                'ids': vehicle_ids,
                'data': vehicle_data
            }
        }
        
        return batch
    
    def _generate_targets(self, gnn_embedding: torch.Tensor,
                        vehicle_data: Dict[str, Any]) -> torch.Tensor:
        """
        生成训练目标
        基于车辆状态预测下一时刻的嵌入表示
        """
        if not vehicle_data:
            return gnn_embedding
        
        # 计算车辆状态的统计特征
        speeds = [v.get('speed', 0.0) for v in vehicle_data.values()]
        positions = [v.get('position', 0.0) for v in vehicle_data.values()]
        
        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_position = np.mean(positions) if positions else 0.0
        
        # 基于物理规律预测状态变化
        # 目标嵌入应该反映速度和位置的变化趋势
        target_embedding = gnn_embedding.clone()
        
        # 添加基于速度的偏移（速度快的车辆应该有更高的嵌入值）
        speed_factor = torch.tensor(avg_speed / 30.0, dtype=torch.float32,
                                   device=gnn_embedding.device)
        target_embedding = target_embedding * (1.0 + speed_factor * 0.1)
        
        # 添加基于位置的编码（周期性特征）
        position_factor = torch.tensor(
            np.sin(avg_position / 1000.0 * 2 * np.pi),
            dtype=torch.float32, device=gnn_embedding.device
        )
        target_embedding = target_embedding + position_factor * 0.05
        
        # 添加小的随机扰动以增加鲁棒性
        noise = torch.randn_like(target_embedding) * 0.02
        target_embedding = target_embedding + noise
        
        return target_embedding
    
    def _calculate_reward(self, output: Dict[str, Any],
                         vehicle_data: Dict[str, Any]) -> torch.Tensor:
        """
        计算奖励 - 基于真实交通指标
        考虑：流量效率、安全、稳定性、控制成本
        """
        if not vehicle_data:
            return torch.tensor(0.0, dtype=torch.float32)
        
        speeds = [v.get('speed', 0.0) for v in vehicle_data.values()]
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
        for veh_id, vehicle in vehicle_data.items():
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
        intervention_cost = (output['level1_interventions'] +
                           output['level2_interventions']) * 0.05
        
        # 5. 综合奖励
        reward = (
            flow_efficiency * 10.0           # 流量效率权重
            - stability_penalty * 2.0         # 稳定性惩罚权重
            - safety_penalty * 5.0            # 安全惩罚权重
            - intervention_cost                # 控制成本
        )
        
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
    """主函数 - 单卡训练配置"""
    # 检测CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    if device == 'cuda':
        print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载配置 - 优化的单卡训练参数
    config = {
        'training': {
            'phase1_epochs': 10,  # 快速测试：10个epoch
            'phase2_epochs': 20,  # 快速测试：20个epoch
            'phase3_epochs': 10,  # 快速测试：10个epoch
            'batch_size': 32,     # 单卡22GB显存：32-48
            'learning_rate': 0.0003,  # 适配混合精度训练
            'weight_decay': 0.0001,
            'use_amp': True,      # 启用混合精度训练
            'num_workers': 2      # DataLoader工作线程数
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
        'device': device,
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
