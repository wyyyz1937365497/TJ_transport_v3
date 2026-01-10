"""
预测层：渐进式世界模型 (Progressive World Model)
分两阶段训练：
Phase 1：仅预测下一时刻车辆状态（位置、速度），学习基础动力学
Phase 2：冻结特征提取器，解耦输出为 z_flow（流演化）与 z_risk（风险演化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class ProgressiveWorldModel(nn.Module):
    """
    渐进式世界模型
    阶段1：预测下一时刻状态
    阶段2：预测未来5步状态 + 冲突概率
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, 
                 future_steps: int = 5, num_phases: int = 2):
        super().__init__()
        
        self.future_steps = future_steps
        self.num_phases = num_phases
        self.current_phase = 1
        
        # 1. 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. 基础动力学分支 (Phase 1)
        self.dynamics_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 3. 风险演化分支 (Phase 2)
        self.risk_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 192),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(192, input_dim + 1)  # 状态 + 冲突概率
            ) for _ in range(future_steps)
        ])
        
        # 4. 辅助分类器
        self.conflict_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LSTM)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_phase(self, phase: int):
        """设置训练阶段"""
        self.current_phase = phase
        print(f"🔄 世界模型切换到阶段 {phase}")
    
    def forward(self, gnn_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            gnn_embedding: [N, 256] GNN输出嵌入
        Returns:
            predictions: 
                Phase 1: [N, 256] 下一时刻状态
                Phase 2: [N, 5, 257] 未来5步状态 + 冲突概率
        """
        batch_size = gnn_embedding.size(0)
        
        # 1. 共享编码
        encoded = self.shared_encoder(gnn_embedding)  # [N, 128]
        
        if self.current_phase == 1:
            # Phase 1: 基础动力学预测
            # 重塑为LSTM输入格式 [N, 1, 128]
            lstm_input = encoded.unsqueeze(1)
            
            # LSTM预测
            lstm_output, _ = self.dynamics_lstm(lstm_input)  # [N, 1, 128]
            
            # 预测下一时刻状态
            next_state = self.risk_decoders[0](lstm_output.squeeze(1))[:, :-1]  # [N, 256]
            
            return next_state
        
        else:
            # Phase 2: 风险演化预测
            predictions = []
            
            # 为每个未来步生成预测
            for t in range(self.future_steps):
                # 使用相同的编码但添加时间步信息
                time_input = encoded + 0.1 * t * torch.ones_like(encoded)
                pred = self.risk_decoders[t](time_input)  # [N, 257]
                predictions.append(pred.unsqueeze(1))
            
            # 合并预测 [N, 5, 257]
            predictions = torch.cat(predictions, dim=1)
            
            return predictions
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                     phase: int = None) -> Dict[str, torch.Tensor]:
        """
        计算损失
        Args:
            predictions: 模型预测
            targets: 真实目标
            phase: 当前训练阶段
        Returns:
            loss_dict: 包含各项损失的字典
        """
        if phase is None:
            phase = self.current_phase
        
        loss_dict = {}
        
        if phase == 1:
            # Phase 1: 仅计算状态预测的MSE损失
            mse_loss = F.mse_loss(predictions, targets)
            loss_dict['mse_loss'] = mse_loss
            loss_dict['total_loss'] = mse_loss
        
        else:
            # Phase 2: 联合优化轨迹MSE与冲突分类损失
            # 状态预测损失
            state_pred = predictions[:, :, :-1]  # [N, 5, 256]
            state_target = targets[:, :, :-1]  # [N, 5, 256]
            mse_loss = F.mse_loss(state_pred, state_target)
            
            # 冲突分类损失
            conflict_pred = predictions[:, :, -1]  # [N, 5]
            conflict_target = targets[:, :, -1]  # [N, 5]
            bce_loss = F.binary_cross_entropy(conflict_pred, conflict_target)
            
            # 总损失
            total_loss = mse_loss + 0.5 * bce_loss
            
            loss_dict['mse_loss'] = mse_loss
            loss_dict['bce_loss'] = bce_loss
            loss_dict['total_loss'] = total_loss
        
        return loss_dict


class FlowEvolutionDecoder(nn.Module):
    """
    流演化解码器
    预测交通流的演化趋势
    """
    
    def __init__(self, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            hidden_state: [N, hidden_dim] 隐藏状态
        Returns:
            flow_state: [N, output_dim] 流状态
        """
        return self.decoder(hidden_state)


class RiskEvolutionDecoder(nn.Module):
    """
    风险演化解码器
    预测风险状态的演化
    """
    
    def __init__(self, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # 风险分类头
        self.risk_classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            hidden_state: [N, hidden_dim] 隐藏状态
        Returns:
            risk_state: [N, output_dim] 风险状态
            risk_prob: [N, 1] 风险概率
        """
        risk_state = self.decoder(hidden_state)
        risk_prob = self.risk_classifier(risk_state)
        return risk_state, risk_prob
