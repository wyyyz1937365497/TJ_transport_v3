import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveWorldModel(nn.Module):
    """渐进式世界模型"""
    
    def __init__(self, input_dim=256, hidden_dim=128, num_future_steps=5, num_phases=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.num_phases = num_phases
        
        # Phase 1: 基础动力学预测
        self.phase1_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.phase1_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Phase 2: 风险演化预测
        self.phase2_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 为每个未来步骤创建预测器
        self.phase2_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_future_steps)
        ])
        
        # 当前阶段
        self.current_phase = 1
    
    def forward(self, state, action=None):
        """
        前向传播
        如果action不为None，则将其与state连接作为输入
        """
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        
        if self.current_phase == 1:
            # Phase 1: 预测下一时刻状态
            h = self.phase1_encoder(x)
            next_state = self.phase1_decoder(h)
            return next_state.unsqueeze(1)  # [batch, 1, state_dim]
        else:
            # Phase 2: 预测未来多个时刻状态
            h = self.phase2_encoder(x)
            future_states = []
            
            for decoder in self.phase2_decoders:
                future_state = decoder(h)
                future_states.append(future_state)
            
            return torch.stack(future_states, dim=1)  # [batch, future_steps, state_dim]
    
    def set_phase(self, phase):
        """设置当前训练阶段"""
        if phase in [1, 2]:
            self.current_phase = phase
        else:
            raise ValueError("Phase must be 1 or 2")
    
    def get_phase(self):
        """获取当前阶段"""
        return self.current_phase