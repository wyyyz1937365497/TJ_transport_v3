"""
系统测试脚本
测试所有组件是否正常工作
"""

import torch
import numpy as np
from typing import Dict, List, Any
import os

# 导入所有组件
from neural_traffic_controller import TrafficController
from risk_sensitive_gnn import RiskSensitiveGNN, GraphAttentionLayer
from progressive_world_model import ProgressiveWorldModel
from influence_controller import InfluenceDrivenController, IDMController
from safety_shield import DualModeSafetyShield, SafetyReward, ActionClipper


def test_risk_sensitive_gnn():
    """测试风险敏感GNN"""
    print("🧪 测试 Risk-Sensitive GNN...")
    
    # 创建模型
    model = RiskSensitiveGNN(
        node_dim=9,
        edge_dim=4,
        hidden_dim=64,
        output_dim=256,
        num_layers=3,
        heads=4
    )
    
    # 创建测试数据
    num_nodes = 10
    num_edges = 20
    graph = {
        'x': torch.randn(num_nodes, 9),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'edge_attr': torch.randn(num_edges, 4)
    }
    
    # 前向传播
    output = model(graph)
    
    assert output.shape == (num_nodes, 256), f"输出形状错误: {output.shape}"
    print(f"   ✅ Risk-Sensitive GNN 测试通过! 输出形状: {output.shape}")
    
    return True


def test_progressive_world_model():
    """测试渐进式世界模型"""
    print("🧪 测试 Progressive World Model...")
    
    # 创建模型
    model = ProgressiveWorldModel(
        input_dim=256,
        hidden_dim=128,
        future_steps=5,
        num_phases=2
    )
    
    # 创建测试数据
    gnn_embedding = torch.randn(10, 256)
    
    # 测试 Phase 1
    model.set_phase(1)
    output_phase1 = model(gnn_embedding)
    assert output_phase1.shape == (10, 256), f"Phase 1 输出形状错误: {output_phase1.shape}"
    print(f"   ✅ Phase 1 测试通过! 输出形状: {output_phase1.shape}")
    
    # 测试 Phase 2
    model.set_phase(2)
    output_phase2 = model(gnn_embedding)
    assert output_phase2.shape == (10, 5, 257), f"Phase 2 输出形状错误: {output_phase2.shape}"
    print(f"   ✅ Phase 2 测试通过! 输出形状: {output_phase2.shape}")
    
    # 测试损失计算
    targets = torch.randn(10, 5, 257)
    loss_dict = model.compute_loss(output_phase2, targets, phase=2)
    assert 'total_loss' in loss_dict, "损失字典缺少 total_loss"
    print(f"   ✅ 损失计算测试通过! 总损失: {loss_dict['total_loss'].item():.4f}")
    
    return True


def test_influence_controller():
    """测试影响力驱动控制器"""
    print("🧪 测试 Influence-Driven Controller...")
    
    # 创建模型
    model = InfluenceDrivenController(
        gnn_dim=256,
        world_dim=256,
        global_dim=16,
        hidden_dim=128,
        action_dim=2,
        top_k=5
    )
    
    # 创建测试数据
    gnn_embedding = torch.randn(10, 256)
    world_predictions = torch.randn(10, 5, 257)
    global_metrics = torch.randn(1, 16)
    vehicle_ids = [f"veh_{i}" for i in range(10)]
    is_icv = torch.tensor([True, False, True, False, True, False, True, False, True, False])
    
    # 前向传播
    output = model(gnn_embedding, world_predictions, global_metrics, vehicle_ids, is_icv)
    
    assert 'selected_vehicle_ids' in output, "输出缺少 selected_vehicle_ids"
    assert 'raw_actions' in output, "输出缺少 raw_actions"
    assert len(output['selected_vehicle_ids']) <= 5, "选中车辆数超过 top_k"
    print(f"   ✅ Influence-Driven Controller 测试通过!")
    print(f"      选中车辆: {output['selected_vehicle_ids']}")
    print(f"      动作形状: {output['raw_actions'].shape}")
    
    return True


def test_safety_shield():
    """测试双模态安全屏障"""
    print("🧪 测试 Dual-Mode Safety Shield...")
    
    # 创建模型
    model = DualModeSafetyShield(
        ttc_threshold=2.0,
        thw_threshold=1.5,
        max_accel=2.0,
        max_decel=-3.0,
        emergency_decel=-5.0,
        max_lane_change_speed=5.0
    )
    
    # 创建测试数据
    raw_actions = torch.randn(5, 2)
    vehicle_states = {
        'ids': ['veh_0', 'veh_1', 'veh_2', 'veh_3', 'veh_4'],
        'data': {
            'veh_0': {'position': 100.0, 'speed': 10.0, 'lane_id': 'lane_0', 'id': 'veh_0'},
            'veh_1': {'position': 150.0, 'speed': 15.0, 'lane_id': 'lane_0', 'id': 'veh_1'},
            'veh_2': {'position': 200.0, 'speed': 20.0, 'lane_id': 'lane_0', 'id': 'veh_2'},
            'veh_3': {'position': 250.0, 'speed': 12.0, 'lane_id': 'lane_0', 'id': 'veh_3'},
            'veh_4': {'position': 300.0, 'speed': 18.0, 'lane_id': 'lane_0', 'id': 'veh_4'}
        }
    }
    selected_indices = [0, 1, 2, 3, 4]
    
    # 前向传播
    output = model(raw_actions, vehicle_states, selected_indices)
    
    assert 'safe_actions' in output, "输出缺少 safe_actions"
    assert 'level1_interventions' in output, "输出缺少 level1_interventions"
    assert 'level2_interventions' in output, "输出缺少 level2_interventions"
    print(f"   ✅ Dual-Mode Safety Shield 测试通过!")
    print(f"      Level 1 干预: {output['level1_interventions']}")
    print(f"      Level 2 干预: {output['level2_interventions']}")
    
    return True


def test_traffic_controller():
    """测试完整交通控制器"""
    print("🧪 测试 Traffic Controller...")
    
    # 创建配置
    config = {
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
    }
    
    # 创建模型
    model = TrafficController(config)
    
    # 创建测试批次
    batch = {
        'node_features': torch.randn(10, 9),
        'edge_indices': torch.randint(0, 10, (2, 20)),
        'edge_features': torch.randn(20, 4),
        'global_metrics': torch.randn(1, 16),
        'vehicle_ids': [f"veh_{i}" for i in range(10)],
        'is_icv': torch.tensor([True, False, True, False, True, False, True, False, True, False]),
        'vehicle_states': {
            'ids': [f"veh_{i}" for i in range(10)],
            'data': {
                f"veh_{i}": {
                    'position': float(i * 50),
                    'speed': float(10 + i),
                    'lane_id': 'lane_0',
                    'id': f"veh_{i}"
                } for i in range(10)
            }
        }
    }
    
    # 前向传播
    output = model(batch, step=0)
    
    assert 'selected_vehicle_ids' in output, "输出缺少 selected_vehicle_ids"
    assert 'safe_actions' in output, "输出缺少 safe_actions"
    assert 'gnn_embedding' in output, "输出缺少 gnn_embedding"
    assert 'world_predictions' in output, "输出缺少 world_predictions"
    print(f"   ✅ Traffic Controller 测试通过!")
    print(f"      选中车辆: {output['selected_vehicle_ids']}")
    print(f"      GNN 嵌入形状: {output['gnn_embedding'].shape}")
    print(f"      世界预测形状: {output['world_predictions'].shape}")
    
    return True


def test_idm_controller():
    """测试IDM控制器"""
    print("🧪 测试 IDM Controller...")
    
    # 创建模型
    model = IDMController()
    
    # 创建测试数据
    ego_speed = 10.0
    leader_speed = 12.0
    gap = 30.0
    
    # 计算加速度
    acceleration = model.compute_acceleration(ego_speed, leader_speed, gap)
    
    # acceleration可能是float或tensor，转换为float
    if isinstance(acceleration, torch.Tensor):
        acceleration = acceleration.item()
    
    assert isinstance(acceleration, float), f"输出类型错误: {type(acceleration)}"
    print(f"   ✅ IDM Controller 测试通过!")
    print(f"      加速度: {acceleration:.4f}")
    
    return True


def test_safety_reward():
    """测试安全奖励函数"""
    print("🧪 测试 Safety Reward...")
    
    # 创建模型
    model = SafetyReward()
    
    # 创建测试数据
    ttc = torch.tensor([1.5, 2.5, 3.5])
    thw = torch.tensor([1.0, 2.0, 3.0])
    
    # 计算奖励
    reward = model(ttc, thw)
    
    assert reward.shape == (3,), f"输出形状错误: {reward.shape}"
    print(f"   ✅ Safety Reward 测试通过!")
    print(f"      奖励: {reward}")
    
    return True


def test_action_clipper():
    """测试动作裁剪器"""
    print("🧪 测试 Action Clipper...")
    
    # 创建模型
    model = ActionClipper()
    
    # 创建测试数据
    actions = torch.randn(5, 2)
    current_speeds = torch.tensor([10.0, 15.0, 20.0, 25.0, 30.0])
    
    # 裁剪动作
    clipped_actions = model(actions, current_speeds)
    
    assert clipped_actions.shape == (5, 2), f"输出形状错误: {clipped_actions.shape}"
    print(f"   ✅ Action Clipper 测试通过!")
    print(f"      原始动作: {actions}")
    print(f"      裁剪后: {clipped_actions}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🚀 开始系统测试")
    print("=" * 60)
    print()
    
    tests = [
        ("Risk-Sensitive GNN", test_risk_sensitive_gnn),
        ("Progressive World Model", test_progressive_world_model),
        ("Influence-Driven Controller", test_influence_controller),
        ("Dual-Mode Safety Shield", test_safety_shield),
        ("Traffic Controller", test_traffic_controller),
        ("IDM Controller", test_idm_controller),
        ("Safety Reward", test_safety_reward),
        ("Action Clipper", test_action_clipper)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 所有测试通过!")
    else:
        print(f"⚠️  {failed} 个测试失败，请检查代码")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
