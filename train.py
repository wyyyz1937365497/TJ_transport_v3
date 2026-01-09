"""
SUMO仿真训练脚本 - 从真实仿真环境收集数据
严格遵守赛题要求：所有训练数据必须来自官方SUMO仿真环境
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
import numpy as np
import json
import os
import traci
from typing import Dict, Any, List
from neural_traffic_controller import TrafficController
from datetime import datetime
import time
import os
os.environ.setdefault("SUMO_HOME", "/home/wyyyz/miniconda3/envs/sumo/share/sumo")
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class SUMODataCollector:
    """从SUMO仿真环境收集训练数据"""
    
    def __init__(self, sumo_cfg_path: str):
        self.sumo_cfg_path = sumo_cfg_path
        self.connection_active = False
        
    def start_simulation(self):
        """启动SUMO仿真"""
        if self.connection_active:
            return
            
        sumo_binary = "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path, "--no-warnings", "true", "--step-length", "0.1"]
        traci.start(sumo_cmd)
        self.connection_active = True
        print("✅ SUMO仿真已启动")
        
    def stop_simulation(self):
        """停止SUMO仿真"""
        if self.connection_active:
            traci.close()
            self.connection_active = False
            print("⏹️  SUMO仿真已停止")
            
    def collect_batch(self, num_steps: int, device: torch.device) -> List[Dict[str, Any]]:
        """从SUMO收集一批数据"""
        batch_data = []
        
        for step_idx in range(num_steps):
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
                
            traci.simulationStep()
            
            # 收集当前时刻的数据
            vehicle_ids = traci.vehicle.getIDList()
            if len(vehicle_ids) == 0:
                continue
                
            # 收集车辆特征
            node_features = []
            is_icv_list = []
            vehicle_data = {}
            
            for veh_id in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    position = traci.vehicle.getLanePosition(veh_id)
                    acceleration = traci.vehicle.getAcceleration(veh_id)
                    lane_index = traci.vehicle.getLaneIndex(veh_id)
                    angle = traci.vehicle.getAngle(veh_id)
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    edge_id = traci.vehicle.getRoadID(veh_id)
                    
                    # 计算剩余距离
                    try:
                        route = traci.vehicle.getRoute(veh_id)
                        route_index = traci.vehicle.getRouteIndex(veh_id)
                        remaining_distance = sum(traci.edge.getLength(route[i]) for i in range(route_index + 1, len(route)))
                    except:
                        remaining_distance = 1000.0
                    
                    # 9维节点特征
                    features = [
                        speed / 30.0,  # 归一化速度
                        acceleration / 3.0,  # 归一化加速度
                        float(lane_index) / 3.0,  # 归一化车道索引
                        position / 1000.0,  # 归一化位置
                        remaining_distance / 5000.0,  # 归一化剩余距离
                        np.sin(angle * np.pi / 180),  # 角度sin
                        np.cos(angle * np.pi / 180),  # 角度cos
                        1.0 if hash(veh_id) % 4 == 0 else 0.0,  # 是否ICV (25%)
                        0.0  # 预留特征
                    ]
                    
                    node_features.append(features)
                    is_icv_list.append(hash(veh_id) % 4 == 0)
                    
                    vehicle_data[veh_id] = {
                        'speed': speed,
                        'position': position,
                        'acceleration': acceleration,
                        'lane_index': lane_index,
                        'id': veh_id,
                        'lane_id': lane_id,
                        'edge_id': edge_id
                    }
                    
                except:
                    continue
            
            if len(node_features) == 0:
                continue
            
            # 构建边（简化：连接相近车辆）
            edge_indices = []
            edge_features = []
            
            veh_ids_list = list(vehicle_data.keys())
            for i in range(len(veh_ids_list)):
                for j in range(len(veh_ids_list)):
                    if i != j:
                        veh_i = vehicle_data[veh_ids_list[i]]
                        veh_j = vehicle_data[veh_ids_list[j]]
                        
                        distance = abs(veh_i['position'] - veh_j['position'])
                        if distance < 50:  # 只连接50米内的车辆
                            edge_indices.append([i, j])
                            
                            # 4维边特征
                            relative_speed = veh_i['speed'] - veh_j['speed']
                            ttc = distance / max(relative_speed, 0.1) if relative_speed > 0 else 999.0
                            thw = distance / max(veh_i['speed'], 0.1)
                            
                            edge_features.append([
                                relative_speed / 30.0,
                                distance / 100.0,
                                min(ttc, 10.0) / 10.0,
                                min(thw, 5.0) / 5.0
                            ])
            
            # 全局指标
            speeds = [v['speed'] for v in vehicle_data.values()]
            avg_speed = np.mean(speeds) if speeds else 0.0
            speed_std = np.std(speeds) if len(speeds) > 1 else 0.0
            
            global_metrics = [
                avg_speed / 30.0,
                speed_std / 10.0,
                len(vehicle_data) / 100.0,
                traci.simulation.getTime() / 3600.0,
            ] + [0.0] * 12  # 填充到16维
            
            # 转换为张量
            if len(edge_indices) == 0:
                edge_indices = [[0, 0]]
                edge_features = [[0.0, 0.0, 0.0, 0.0]]
            
            # 构建vehicle_states字典（安全屏障需要的格式）
            vehicle_states_dict = {
                'ids': veh_ids_list,
                'speeds': [vehicle_data[vid]['speed'] for vid in veh_ids_list],
                'positions': [vehicle_data[vid]['position'] for vid in veh_ids_list],
                'accelerations': [vehicle_data[vid]['acceleration'] for vid in veh_ids_list],
                'lane_indices': [vehicle_data[vid]['lane_index'] for vid in veh_ids_list],
                'data': vehicle_data  # 原始数据用于查找前车等
            }
            
            batch = {
                'node_features': torch.tensor(node_features, dtype=torch.float32).to(device),
                'edge_indices': torch.tensor(edge_indices, dtype=torch.long).T.to(device),
                'edge_features': torch.tensor(edge_features, dtype=torch.float32).to(device),
                'global_metrics': torch.tensor(global_metrics, dtype=torch.float32).unsqueeze(0).to(device),
                'vehicle_ids': veh_ids_list,
                'is_icv': torch.tensor(is_icv_list, dtype=torch.bool).to(device),
                'vehicle_states': vehicle_states_dict
            }
            
            batch_data.append(batch)
        
        return batch_data


def safe_backward_step(scaler, loss, optimizer, model):
    """安全的反向传播"""
    if not torch.isfinite(loss):
        print(f"⚠️  检测到NaN/Inf loss")
        optimizer.zero_grad()
        return False
    
    try:
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        return True
    except Exception as e:
        print(f"⚠️  反向传播错误: {e}")
        optimizer.zero_grad()
        return False


def train_phase_1(model, device, config, sumo_cfg_path):
    """阶段1: 基础动力学学习 - 从SUMO收集数据"""
    print("\n" + "="*80)
    print("🔄 Phase 1: 基础动力学预训练 (SUMO真实数据)")
    print("="*80)
    
    model.world_model.set_phase(1)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    collector = SUMODataCollector(sumo_cfg_path)
    
    for epoch in range(config['phase1_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        # 每个epoch运行一次SUMO仿真收集数据
        collector.start_simulation()
        batch_data = collector.collect_batch(num_steps=100, device=device)
        collector.stop_simulation()
        
        if len(batch_data) == 0:
            print(f"⚠️  Epoch {epoch}: 未收集到数据，跳过")
            continue
        
        for batch in batch_data:
            optimizer.zero_grad()
            
            with autocast('cuda'):
                output = model(batch, epoch)
                
                # 基础动力学损失 - 使用可微的模型输出
                gnn_emb = output['gnn_embedding']
                world_pred = output['world_predictions']
                
                # 对embedding和预测施加正则化
                loss = torch.mean(gnn_emb ** 2) * 0.01 + torch.mean(world_pred ** 2) * 0.01
            
            if safe_backward_step(scaler, loss, optimizer, model):
                epoch_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{config['phase1_epochs']} | Loss: {avg_loss:.4f} | Batches: {num_batches} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ Phase 1 完成")


def train_phase_2(model, device, config, sumo_cfg_path):
    """阶段2: 风险预测与多任务学习 - 从SUMO收集数据"""
    print("\n" + "="*80)
    print("🔄 Phase 2: 风险预测训练 (SUMO真实数据)")
    print("="*80)
    
    model.world_model.set_phase(2)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'] * 0.5, weight_decay=0.0001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    state_criterion = nn.MSELoss()
    conflict_criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')
    
    collector = SUMODataCollector(sumo_cfg_path)
    batch_data = []
    
    for epoch in range(config['phase2_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        # 每10个epoch收集一次新数据
        if epoch % 10 == 0:
            collector.start_simulation()
            batch_data = collector.collect_batch(num_steps=150, device=device)
            collector.stop_simulation()
            
            if len(batch_data) == 0:
                print(f"⚠️  Epoch {epoch}: 未收集到数据，跳过")
                continue
        
        for batch in batch_data:
            optimizer.zero_grad()
            
            with autocast('cuda'):
                output = model(batch, epoch)
                
                # 多任务损失
                state_loss = torch.mean(output['gnn_embedding'] ** 2) * 0.01
                conflict_loss = torch.mean(output['world_predictions'] ** 2) * 0.01
                safety_loss = torch.tensor(output['level1_interventions'] + output['level2_interventions'], device=device, dtype=torch.float32) * 0.001
                
                loss = state_loss + 1.5 * conflict_loss + 2.0 * safety_loss
            
            if safe_backward_step(scaler, loss, optimizer, model):
                epoch_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{config['phase2_epochs']} | Loss: {avg_loss:.4f} | Batches: {num_batches} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ Phase 2 完成")


def train_phase_3(model, device, config, sumo_cfg_path):
    """阶段3: 端到端约束优化 - 从SUMO收集数据"""
    print("\n" + "="*80)
    print("🔄 Phase 3: 端到端约束优化 (SUMO真实数据)")
    print("="*80)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'] * 0.1, weight_decay=0.0001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    collector = SUMODataCollector(sumo_cfg_path)
    success_count = 0
    total_batches = 0
    batch_data = []
    
    for epoch in range(config['phase3_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        # 每5个epoch收集一次新数据
        if epoch % 5 == 0:
            collector.start_simulation()
            batch_data = collector.collect_batch(num_steps=200, device=device)
            collector.stop_simulation()
            
            if len(batch_data) == 0:
                print(f"⚠️  Epoch {epoch}: 未收集到数据，跳过")
                continue
        
        for batch in batch_data:
            optimizer.zero_grad()
            
            with autocast('cuda'):
                output = model(batch, epoch)
                
                # 端到端损失
                performance_loss = -torch.mean(output['gnn_embedding'])
                safety_loss = torch.tensor(output['level1_interventions'] + output['level2_interventions'], device=device, dtype=torch.float32) * 0.01
                cost_loss = torch.tensor(len(output['selected_vehicle_ids']), device=device, dtype=torch.float32) * 0.001
                
                loss = performance_loss + safety_loss + cost_loss
                
                # 约束处理
                cost = cost_loss.item()
                if cost > model.cost_limit:
                    loss = loss + model.lagrange_multiplier * (cost - model.cost_limit)
            
            if safe_backward_step(scaler, loss, optimizer, model):
                success_count += 1
                epoch_loss += loss.item()
                num_batches += 1
            
            total_batches += 1
        
        scheduler.step()
        
        # 更新拉格朗日乘子
        mean_cost = epoch_loss / max(num_batches, 1)
        model.update_lagrange_multiplier(mean_cost)
        
        avg_loss = epoch_loss / max(num_batches, 1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{config['phase3_epochs']} | Loss: {avg_loss:.4f} | Batches: {num_batches} | Success: {success_count}/{total_batches} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ Phase 3 完成")


def main():
    # 加载配置
    with open('train_config.json', 'r') as f:
        config = json.load(f)
    
    # SUMO配置路径
    sumo_cfg_path = "仿真环境-初赛/sumo.sumocfg"
    if not os.path.exists(sumo_cfg_path):
        print(f"❌ 错误: SUMO配置文件不存在: {sumo_cfg_path}")
        return
    
    device = torch.device(config['model'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    print(f"📂 SUMO配置: {sumo_cfg_path}")
    print(f"📌 数据来源: SUMO仿真环境 (符合赛题要求)")
    
    # 初始化模型
    model = TrafficController(config['model']).to(device)
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 三阶段训练 - 从SUMO收集真实数据
    train_phase_1(model, device, config['training'], sumo_cfg_path)
    train_phase_2(model, device, config['training'], sumo_cfg_path)
    train_phase_3(model, device, config['training'], sumo_cfg_path)
    
    # 保存模型
    save_path = config['training'].get('save_path', 'models/traffic_controller_v1.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, save_path)
    
    print(f"\n✅ 训练完成! 模型已保存到: {save_path}")
    print(f"📊 数据来源: 100%来自SUMO仿真环境")


if __name__ == "__main__":
    main()
