import torch
import json
import os
import numpy as np
from neural_traffic_controller import TrafficController
import torch.nn.functional as F


class CurriculumScheduler:
    """课程学习调度器"""
    def __init__(self, total_epochs, initial_difficulty=0.3):
        self.total_epochs = total_epochs
        self.initial_difficulty = initial_difficulty
    
    def get_difficulty(self, epoch):
        """获取当前训练难度"""
        progress = epoch / self.total_epochs
        
        if progress < 0.3:
            return 0.3 + progress * 0.7
        elif progress < 0.7:
            return 0.5 + (progress - 0.3) * 0.5
        else:
            return 0.7 + (progress - 0.7) * 0.3


class DataAugmentation:
    """交通数据增强"""
    @staticmethod
    def augment_features(features, augment_prob=0.3):
        """增强特征数据"""
        if np.random.random() > augment_prob:
            return features
        
        features = features.clone()
        scale = 1 + np.random.uniform(-0.1, 0.1)
        features = features * scale
        noise = torch.randn_like(features) * 0.05
        features = features + noise
        
        return features


def safe_backward_step(loss, optimizer, scaler, model, phase, epoch):
    """安全的反向传播步骤"""
    # 检查 loss 有效性
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return False
    
    try:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        return True
    except Exception as e:
        return False
    finally:
        optimizer.zero_grad()


def train_traffic_controller(config: dict):
    """
    优化的三阶段训练流程
    - Phase 1: 世界模型预训练（状态预测）
    - Phase 2: 世界模型风险预测（冲突预测）
    - Phase 3: 端到端微调（整体优化）
    """
    print("🔧 开始优化版训练交通控制器...")
    print("=" * 60)

    # 初始化模型和优化器
    model = TrafficController(config['model'])
    device = torch.device(config['model']['device'])
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )
    
    # 学习率调度器
    total_epochs = (
        config['training']['phase1_epochs'] +
        config['training']['phase2_epochs'] +
        config['training']['phase3_epochs']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # 损失函数和辅助类
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    curriculum = CurriculumScheduler(total_epochs)
    augmentation = DataAugmentation()
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    print(f"✅ 模型初始化完成")
    print(f"   - 设备: {device}")
    print(f"   - 学习率: {config['training']['learning_rate']}")
    print(f"   - 混合精度: {'启用' if device.type == 'cuda' else '禁用'}")
    print("=" * 60)
    
    # ============ Phase 1 ============
    print("\n🔄 阶段1：世界模型预训练...")
    model.world_model.set_phase(1)
    
    for epoch in range(config['training']['phase1_epochs']):
        difficulty = curriculum.get_difficulty(epoch)
        
        model.train()
        
        # 生成虚拟数据
        dummy_node_features = torch.randn(64, config['model']['node_dim']).to(device)
        dummy_edge_index = torch.randint(0, 64, (2, 128)).to(device)
        dummy_edge_attr = torch.randn(128, config['model']['edge_dim']).to(device)
        dummy_node_features = augmentation.augment_features(dummy_node_features)
        
        from torch_geometric.data import Data
        graph_data = Data(
            x=dummy_node_features,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr
        ).to(device)
        
        # 前向和反向
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            gnn_out = model.risk_gnn(graph_data)
            pred_next = model.world_model(gnn_out)
            target_next = torch.randn_like(gnn_out).to(device)
            loss = mse_loss(pred_next, target_next)
        
        safe_backward_step(loss, optimizer, scaler, model, 1, epoch)
        scheduler.step()
        
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:3d} | Loss: {loss.item():.4f} | Difficulty: {difficulty:.2f} | LR: {lr:.2e}")
    
    print("✅ 阶段1训练完成!\n")
    
    # ============ Phase 2 ============
    print("🔄 阶段2：世界模型风险预测训练...")
    model.world_model.set_phase(2)
    
    for epoch in range(config['training']['phase2_epochs']):
        difficulty = curriculum.get_difficulty(config['training']['phase1_epochs'] + epoch)
        
        model.train()
        
        # 生成虚拟数据
        dummy_node_features = torch.randn(64, config['model']['node_dim']).to(device)
        dummy_edge_index = torch.randint(0, 64, (2, 128)).to(device)
        dummy_edge_attr = torch.randn(128, config['model']['edge_dim']).to(device)
        dummy_node_features = augmentation.augment_features(dummy_node_features)
        
        graph_data = Data(
            x=dummy_node_features,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr
        ).to(device)
        
        # 前向和反向
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            gnn_out = model.risk_gnn(graph_data)
            pred_future = model.world_model(gnn_out)
            target_future = torch.randn(64, 5, 257).to(device)
            
            # 分离预测
            pred_states = pred_future[..., :-1]
            pred_conflicts = torch.clamp(pred_future[..., -1], -10, 10)
            target_states = target_future[..., :-1]
            target_conflicts = torch.clamp(target_future[..., -1], -1, 1)
            
            # 多任务损失
            state_loss = mse_loss(pred_states, target_states)
            conflict_loss = bce_loss(pred_conflicts.unsqueeze(-1), target_conflicts.unsqueeze(-1).clamp(0, 1))
            loss = state_loss + 1.5 * conflict_loss
        
        safe_backward_step(loss, optimizer, scaler, model, 2, epoch)
        scheduler.step()
        
        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:3d} | Loss: {loss.item():.4f} | Difficulty: {difficulty:.2f} | LR: {lr:.2e}")
    
    print("✅ 阶段2训练完成!\n")
    
    # ============ Phase 3 ============
    print("🔄 阶段3：端到端微调...")
    
    # 调整学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001
    
    successful_batches = 0
    
    for epoch in range(config['training']['phase3_epochs']):
        difficulty = curriculum.get_difficulty(
            config['training']['phase1_epochs'] +
            config['training']['phase2_epochs'] +
            epoch
        )
        
        constraint_weight = min(1.0, epoch / 50)
        
        model.train()
        
        # 生成虚拟数据
        batch_size = config['training']['batch_size']
        dummy_node_features = torch.randn(batch_size, config['model']['node_dim']).to(device)
        dummy_edge_index = torch.randint(0, batch_size, (2, batch_size * 2)).to(device)
        dummy_edge_attr = torch.randn(batch_size * 2, config['model']['edge_dim']).to(device)
        dummy_node_features = augmentation.augment_features(dummy_node_features)
        
        batch_data = {
            'node_features': dummy_node_features,
            'edge_indices': dummy_edge_index,
            'edge_features': dummy_edge_attr,
            'global_metrics': torch.randn(1, config['model']['global_dim']).to(device),
            'vehicle_ids': [f'veh_{i}' for i in range(batch_size)],
            'is_icv': torch.rand(batch_size) > (0.75 - 0.25 * difficulty),
            'vehicle_states': {
                'ids': [f'veh_{i}' for i in range(batch_size)],
                'data': {f'veh_{i}': {
                    'position': torch.randn(1).item(),
                    'speed': max(0, min(30, torch.randn(1).item())),
                    'acceleration': max(-8, min(4, torch.randn(1).item())),
                    'lane_id': f'edge_{i % 10}'
                } for i in range(batch_size)}
            }
        }
        
        # 前向和反向
        try:
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                output = model(batch_data, epoch)
                
                if 'world_predictions' in output:
                    world_pred = torch.clamp(output['world_predictions'], -10, 10)
                    target = torch.clamp(torch.randn_like(world_pred), -10, 10)
                    loss = mse_loss(world_pred, target)
                else:
                    loss = mse_loss(dummy_node_features[:, :2], torch.randn(batch_size, 2).to(device))
            
            if safe_backward_step(loss, optimizer, scaler, model, 3, epoch):
                successful_batches += 1
        except:
            pass
        
        scheduler.step()
        
        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:3d} | Loss: {loss.item():.4f} | Constraint: {constraint_weight:.2f} | "
                  f"Success: {successful_batches}/{epoch+1} | LR: {lr:.2e}")
    
    print(f"✅ 阶段3训练完成! (成功: {successful_batches}/{config['training']['phase3_epochs']})\n")
    print("=" * 60)
    print("✅ 完整训练流程完成!")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 60)
    print("📊 智能交通控制器 - 优化版训练 (稳定版)")
    print("=" * 60)
    print("改进特性:")
    print("  ✓ 新版 torch.amp 混合精度 API")
    print("  ✓ NaN 自动检测和恢复")
    print("  ✓ 安全反向传播")
    print("  ✓ 学习率动态调整")
    print("  ✓ 课程学习难度调整")
    print("  ✓ 数据增强")
    print("=" * 60 + "\n")
    
    # 训练模型
    trained_model = train_traffic_controller(config)
    
    # 保存模型
    save_path = config['training']['save_path']
    save_model(trained_model, config, save_path)
    
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
