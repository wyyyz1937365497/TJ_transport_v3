import torch
import json
import os
from neural_traffic_controller import TrafficController, ProgressiveWorldModel
from torch.utils.data import DataLoader, TensorDataset


def train_traffic_controller(config: dict):
    """
    训练交通控制器
    包含三阶段训练流程
    """
    print("🔧 开始训练交通控制器...")

    # 1. 初始化模型
    model = TrafficController(config['model'])
    
    # 2. 设置设备
    device = torch.device(config['model']['device'])
    model = model.to(device)
    
    # 3. 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 4. 设置损失函数
    mse_loss = torch.nn.MSELoss()
    
    # 5. 阶段1：世界模型预训练
    print("🔄 阶段1：世界模型预训练...")
    model.world_model.set_phase(1)
    
    # 创建一些虚拟数据用于演示训练过程
    # 在实际情况下，这里应该使用真实的仿真数据
    dummy_node_features = torch.randn(100, config['model']['node_dim']).to(device)
    dummy_edge_index = torch.randint(0, 100, (2, 200)).to(device)
    dummy_edge_attr = torch.randn(200, config['model']['edge_dim']).to(device)
    
    for epoch in range(config['training']['phase1_epochs']):
        model.train()
        optimizer.zero_grad()
        
        # 创建虚拟图数据
        from torch_geometric.data import Data
        graph_data = Data(
            x=dummy_node_features,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr
        ).to(device)
        
        # GNN推理
        gnn_out = model.risk_gnn(graph_data)
        
        # 预测下一步状态
        pred_next = model.world_model(gnn_out)
        target_next = torch.randn_like(gnn_out)
        
        loss = mse_loss(pred_next, target_next)
        loss.backward()
        
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Phase 1 - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 6. 阶段2：世界模型风险预测训练
    print("🔄 阶段2：世界模型风险预测训练...")
    model.world_model.set_phase(2)
    
    for epoch in range(config['training']['phase2_epochs']):
        model.train()
        optimizer.zero_grad()
        
        # 创建虚拟图数据
        graph_data = Data(
            x=dummy_node_features,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr
        ).to(device)
        
        # GNN推理
        gnn_out = model.risk_gnn(graph_data)
        
        # 预测未来状态
        pred_future = model.world_model(gnn_out)
        target_future = torch.randn(100, 5, 257).to(device)  # [N, 5, 257]
        
        loss = mse_loss(pred_future, target_future)
        loss.backward()
        
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Phase 2 - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 7. 阶段3：端到端微调
    print("🔄 阶段3：端到端微调...")
    
    for epoch in range(config['training']['phase3_epochs']):
        model.train()
        optimizer.zero_grad()
        
        # 创建更复杂的虚拟训练数据
        batch_data = {
            'node_features': dummy_node_features,
            'edge_indices': dummy_edge_index,
            'edge_features': dummy_edge_attr,
            'global_metrics': torch.randn(1, 16).to(device),
            'vehicle_ids': [f'veh_{i}' for i in range(100)],
            'is_icv': torch.rand(100) > 0.75,  # 25%是智能车
            'vehicle_states': {
                'ids': [f'veh_{i}' for i in range(100)],
                'data': {f'veh_{i}': {
                    'position': torch.randn(1).item(),
                    'speed': torch.randn(1).item(),
                    'lane_id': f'edge_{i % 10}'
                } for i in range(100)}
            }
        }
        
        # 前向传播
        output = model(batch_data, epoch)
        
        # 计算损失（这里只是演示，实际应该基于具体任务定义）
        # 我们可以使用一些虚拟的目标值
        dummy_target = torch.randn_like(output['world_predictions'])
        prediction_loss = mse_loss(output['world_predictions'], dummy_target)
        
        # 添加其他损失项
        total_loss = prediction_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Phase 3 - Epoch {epoch}, Loss: {total_loss.item():.4f}")
    
    print("✅ 训练完成!")
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
    
    print("开始训练过程...")
    print(f"模型配置: {config['model']}")
    print(f"训练配置: {config['training']}")
    
    # 训练模型
    trained_model = train_traffic_controller(config)
    
    # 保存模型
    save_path = config['training']['save_path']
    save_model(trained_model, config, save_path)
    
    print("训练流程完成!")


if __name__ == "__main__":
    main()