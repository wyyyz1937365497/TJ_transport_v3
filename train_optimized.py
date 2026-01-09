import torch
import json
import os
import numpy as np
from neural_traffic_controller import TrafficController, ProgressiveWorldModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch_geometric.data import Data


class TrainingConfig:
    """训练配置管理"""
    def __init__(self, config_path='train_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_phase_config(self, phase):
        """获取特定阶段的配置"""
        return self.config['training'][f'phase{phase}']


class MultiTaskLoss(nn.Module):
    """多任务加权损失函数"""
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'state': 1.0,
            'conflict': 1.5,
            'safety': 2.0
        }
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        """
        计算多任务损失
        predictions: {'states': ..., 'conflicts': ..., 'safety': ...}
        targets: {'states': ..., 'conflicts': ..., 'safety': ...}
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 状态预测损失
        if 'states' in predictions and 'states' in targets:
            state_loss = self.mse_loss(predictions['states'], targets['states'])
            loss_dict['state'] = state_loss.item()
            total_loss += self.weights['state'] * state_loss
        
        # 冲突预测损失
        if 'conflicts' in predictions and 'conflicts' in targets:
            conflict_loss = self.bce_loss(
                predictions['conflicts'].sigmoid(),
                targets['conflicts']
            )
            loss_dict['conflict'] = conflict_loss.item()
            total_loss += self.weights['conflict'] * conflict_loss
        
        # 安全指标损失
        if 'safety' in predictions and 'safety' in targets:
            safety_loss = self.mse_loss(predictions['safety'], targets['safety'])
            loss_dict['safety'] = safety_loss.item()
            total_loss += self.weights['safety'] * safety_loss
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


class CurriculumScheduler:
    """课程学习调度器"""
    def __init__(self, total_epochs, initial_difficulty=0.3):
        self.total_epochs = total_epochs
        self.initial_difficulty = initial_difficulty
    
    def get_difficulty(self, epoch):
        """获取当前训练难度"""
        progress = epoch / self.total_epochs
        
        if progress < 0.3:
            # 初期：简单场景
            return 0.3 + progress * 0.7  # 0.3 -> 0.5
        elif progress < 0.7:
            # 中期：中等场景
            return 0.5 + (progress - 0.3) * 0.5  # 0.5 -> 0.7
        else:
            # 后期：复杂场景
            return 0.7 + (progress - 0.7) * 0.3  # 0.7 -> 1.0
    
    def get_batch_importance_weights(self, batch_size, difficulty):
        """根据难度获取批次重要性权重"""
        # 模拟：困难样本权重更高
        weights = np.random.exponential(difficulty, size=batch_size)
        return torch.tensor(weights / weights.sum() * batch_size).float()


class DataAugmentation:
    """交通数据增强"""
    @staticmethod
    def augment_vehicle_state(state, augment_prob=0.5):
        """增强车辆状态数据"""
        if np.random.random() > augment_prob:
            return state
        
        state = state.clone()
        
        # 速度扰动 ±10%
        if 'speed' in state:
            state['speed'] = state['speed'] * (1 + np.random.uniform(-0.1, 0.1))
        
        # 位置偏移 ±5米
        if 'position' in state:
            state['position'] = state['position'] + np.random.uniform(-5, 5)
        
        # 加速度噪声
        if 'acceleration' in state:
            state['acceleration'] = state['acceleration'] + np.random.normal(0, 0.5)
        
        return state
    
    @staticmethod
    def augment_edge_features(edge_features, dropout_rate=0.1):
        """随机移除交互边"""
        if np.random.random() < dropout_rate:
            mask = torch.rand(edge_features.size(0)) > 0.1
            return edge_features[mask]
        return edge_features


class TrainingMonitor:
    """训练监控和早停机制"""
    def __init__(self, patience=15, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait_count = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'phase': []
        }
    
    def update(self, val_loss, learning_rate, phase):
        """更新监控状态"""
        self.history['train_loss'].append(val_loss)
        self.history['learning_rate'].append(learning_rate)
        self.history['phase'].append(phase)
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait_count = 0
            if self.verbose:
                print(f"✅ 新的最优损失: {self.best_loss:.4f}")
            return True
        else:
            self.wait_count += 1
            if self.verbose and self.wait_count % 5 == 0:
                print(f"⚠️ 验证集无改进 {self.wait_count}/{self.patience}")
            return False
    
    def should_stop(self):
        """判断是否应该早停"""
        return self.wait_count >= self.patience


class OptimizedTrainer:
    """优化的训练器"""
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )
        
        # 学习率调度器：余弦退火 + 预热
        total_epochs = (
            config['training']['phase1_epochs'] +
            config['training']['phase2_epochs'] +
            config['training']['phase3_epochs']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 多任务损失
        self.loss_fn = MultiTaskLoss(weights={
            'state': 1.0,
            'conflict': 1.5,
            'safety': 2.0
        })
        
        # 课程学习
        self.curriculum = CurriculumScheduler(total_epochs)
        
        # 监控
        self.monitor = TrainingMonitor(patience=15)
        
        # 数据增强
        self.augmentation = DataAugmentation()
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    def generate_batch_data(self, batch_size=64, difficulty=1.0):
        """生成训练批次"""
        dummy_node_features = torch.randn(batch_size, self.config['model']['node_dim']).to(self.device)
        dummy_edge_index = torch.randint(0, batch_size, (2, batch_size * 2)).to(self.device)
        dummy_edge_attr = torch.randn(batch_size * 2, self.config['model']['edge_dim']).to(self.device)
        
        # 应用数据增强
        if np.random.random() < 0.3:
            dummy_edge_attr = self.augmentation.augment_edge_features(dummy_edge_attr)
        
        batch_data = {
            'node_features': dummy_node_features,
            'edge_indices': dummy_edge_index,
            'edge_features': dummy_edge_attr,
            'global_metrics': torch.randn(1, self.config['model']['global_dim']).to(self.device),
            'vehicle_ids': [f'veh_{i}' for i in range(batch_size)],
            'is_icv': torch.rand(batch_size) > (0.75 - 0.25 * difficulty),  # 难度越高，ICV越多
            'vehicle_states': {
                'ids': [f'veh_{i}' for i in range(batch_size)],
                'data': {f'veh_{i}': {
                    'position': torch.randn(1).item(),
                    'speed': torch.randn(1).item(),
                    'acceleration': torch.randn(1).item(),
                    'lane_id': f'edge_{i % 10}'
                } for i in range(batch_size)}
            }
        }
        
        return batch_data
    
    def train_phase(self, phase, num_epochs):
        """训练单个阶段"""
        print(f"\n{'='*60}")
        print(f"🔄 阶段{phase}训练开始...")
        print(f"{'='*60}")
        
        self.model.world_model.set_phase(phase)
        
        for epoch in range(num_epochs):
            # 获取课程学习难度
            difficulty = self.curriculum.get_difficulty(epoch)
            
            # 生成批次
            batch_data = self.generate_batch_data(
                batch_size=self.config['training']['batch_size'],
                difficulty=difficulty
            )
            
            # 前向传播
            self.model.train()
            self.optimizer.zero_grad()
            
            # 构建图数据
            from torch_geometric.data import Data
            graph_data = Data(
                x=batch_data['node_features'],
                edge_index=batch_data['edge_indices'],
                edge_attr=batch_data['edge_features']
            ).to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                # GNN推理
                gnn_out = self.model.risk_gnn(graph_data)
                
                # 阶段特定的训练
                if phase == 1:
                    # Phase 1: 预测下一时刻状态
                    pred_next = self.model.world_model(gnn_out)
                    target_next = torch.randn_like(gnn_out).to(self.device)
                    
                    loss = F.mse_loss(pred_next, target_next)
                    loss_dict = {'total': loss.item()}
                
                elif phase == 2:
                    # Phase 2: 预测未来5步状态 + 冲突概率
                    pred_future = self.model.world_model(gnn_out)
                    target_future = torch.randn(batch_data['node_features'].size(0), 5, 257).to(self.device)
                    
                    # 分离预测
                    pred_states = pred_future[..., :-1]  # 状态部分
                    pred_conflicts = pred_future[..., -1]  # 冲突部分
                    
                    target_states = target_future[..., :-1]
                    target_conflicts = target_future[..., -1]
                    
                    predictions = {
                        'states': pred_states,
                        'conflicts': pred_conflicts.unsqueeze(-1)
                    }
                    targets = {
                        'states': target_states,
                        'conflicts': target_conflicts.unsqueeze(-1)
                    }
                    
                    loss, loss_dict = self.loss_fn(predictions, targets)
                
                else:  # phase == 3
                    # Phase 3: 端到端优化
                    output = self.model(batch_data, epoch)
                    
                    # 计算组合损失
                    target_safety = torch.randn(len(output.get('selected_indices', [])), 2).to(self.device)
                    
                    if 'world_predictions' in output:
                        predictions = {
                            'states': output['world_predictions'][..., :-1],
                            'conflicts': output['world_predictions'][..., -1].unsqueeze(-1),
                            'safety': torch.randn_like(target_safety).to(self.device)
                        }
                        targets = {
                            'states': torch.randn_like(predictions['states']).to(self.device),
                            'conflicts': torch.rand_like(predictions['conflicts']).to(self.device),
                            'safety': target_safety
                        }
                        loss, loss_dict = self.loss_fn(predictions, targets)
                    else:
                        loss = torch.tensor(0.0).to(self.device)
                        loss_dict = {'total': 0.0}
            
            # 反向传播（使用梯度缩放器）
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # 记录日志
            if epoch % max(1, num_epochs // 10) == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Phase {phase} - Epoch {epoch:3d}/{num_epochs} | "
                      f"Loss: {loss_dict['total']:.4f} | "
                      f"Difficulty: {difficulty:.2f} | "
                      f"LR: {lr:.2e}")
        
        print(f"✅ 阶段{phase}训练完成!")
    
    def train(self):
        """执行完整训练流程"""
        # Phase 1: 世界模型预训练
        self.train_phase(1, self.config['training']['phase1_epochs'])
        
        # Phase 2: 世界模型风险预测
        self.train_phase(2, self.config['training']['phase2_epochs'])
        
        # Phase 3: 端到端微调
        self.train_phase(3, self.config['training']['phase3_epochs'])
        
        print("\n✅ 全部训练完成!")


def train_traffic_controller(config: dict):
    """
    优化的训练流程
    """
    print("🔧 开始优化训练交通控制器...")
    
    # 初始化模型
    model = TrafficController(config['model'])
    
    # 设置设备
    device = torch.device(config['model']['device'])
    model = model.to(device)
    
    # 创建优化的训练器
    trainer = OptimizedTrainer(model, config, device)
    
    # 执行训练
    trainer.train()
    
    return model


def save_model(model, config, save_path):
    """保存训练好的模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, save_path)
    
    print(f"✅ 模型已保存到: {save_path}")


def main():
    # 加载配置
    with open('train_config.json', 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("📊 优化版交通控制器训练")
    print("=" * 60)
    print(f"模型配置: {config['model']}")
    print(f"训练配置: {config['training']}")
    
    # 训练模型
    trained_model = train_traffic_controller(config)
    
    # 保存模型
    save_path = config['training']['save_path']
    save_model(trained_model, config, save_path)
    
    print("\n" + "=" * 60)
    print("训练流程完成!")
    print("=" * 60)


if __name__ == "__main__":
    import torch.nn as nn
    main()
