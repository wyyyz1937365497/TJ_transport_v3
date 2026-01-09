import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class TrafficTorchModel(TorchModelV2, nn.Module):
    """
    v4.0架构的完整PyTorch实现
    集成：感知层 + 预测层 + 决策层 + 约束优化
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.config = model_config["custom_model_config"]
        
        # --- 1. 感知层：Risk-Sensitive GNN ---
        self.gnn = self._build_gnn()
        
        # --- 2. 预测层：Progressive World Model ---
        self.world_model = self._build_world_model()
        
        # --- 3. 决策层：影响力驱动选择器 ---
        self.influence_selector = self._build_influence_selector()
        
        # --- 4. Agent网络 ---
        self.actor = self._build_actor_network()
        self.critic = self._build_critic_network()
        self.cost_critic = self._build_cost_critic_network()
        
        # --- 5. 约束优化参数 ---
        self.register_buffer("lagrange_multiplier", torch.tensor(1.0))
        self.cost_limit = self.config.get("cost_limit", 10.0)
        self.lambda_lr = self.config.get("lambda_lr", 0.01)
        
        # 缓存
        self._last_value = None
        self._last_cost_value = None
        
    def _build_gnn(self):
        """构建风险敏感GNN"""
        # 简化的GNN实现，实际应用中需要更复杂的图网络
        class SimpleGNN(nn.Module):
            def __init__(self, input_dim=256, hidden_dim=128, output_dim=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.ReLU()
                )
            
            def forward(self, x):
                return self.net(x)
        
        return SimpleGNN(
            input_dim=256,
            hidden_dim=self.config.get("gnn_hidden_dim", 128),
            output_dim=self.config.get("gnn_output_dim", 256)
        )
    
    def _build_world_model(self):
        """构建渐进式世界模型"""
        class ProgressiveWorldModel(nn.Module):
            def __init__(self, input_dim=256, hidden_dim=128, num_future_steps=5):
                super().__init__()
                self.num_future_steps = num_future_steps
                
                # 编码器
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                # 解码器：为每个未来步骤预测状态
                self.decoders = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim)
                    ) for _ in range(num_future_steps)
                ])
            
            def forward(self, x):
                encoded = self.encoder(x)
                predictions = []
                
                for decoder in self.decoders:
                    pred = decoder(encoded)
                    predictions.append(pred)
                
                return torch.stack(predictions, dim=1)  # [B, num_future_steps, input_dim]
        
        return ProgressiveWorldModel(
            input_dim=self.config.get("gnn_output_dim", 256),
            hidden_dim=self.config.get("world_hidden_dim", 128),
            num_future_steps=self.config.get("future_steps", 5)
        )
    
    def _build_influence_selector(self):
        """构建影响力驱动选择器"""
        class InfluenceSelector(nn.Module):
            def __init__(self, input_dim=256, hidden_dim=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            
            def forward(self, x):
                return torch.sigmoid(self.net(x))
        
        return InfluenceSelector(
            input_dim=self.config.get("gnn_output_dim", 256),
            hidden_dim=self.config.get("selector_hidden_dim", 64)
        )
    
    def _build_actor_network(self):
        """构建Actor网络"""
        input_dim = self.config.get("gnn_output_dim", 256) * 2  # GNN + 预测
        
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.shape[0] * self.action_space.shape[1])
        )
    
    def _build_critic_network(self):
        """构建Critic网络"""
        input_dim = self.config.get("gnn_output_dim", 256) * 2
        
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _build_cost_critic_network(self):
        """构建Cost Critic网络"""
        input_dim = self.config.get("gnn_output_dim", 256) * 2
        
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        前向传播，整合v4.0所有组件
        """
        # 1. 提取观测
        obs = input_dict["obs"]
        gnn_embedding = obs["gnn_embedding"]  # [B, 256]
        world_prediction = obs["world_prediction"]  # [B, 5, 256]
        
        # 2. 融合GNN和世界模型特征
        world_features = torch.mean(world_prediction, dim=1)  # [B, 256]
        combined_features = torch.cat([gnn_embedding, world_features], dim=1)  # [B, 512]
        
        # 3. Actor: 生成动作
        action_logits = self.actor(combined_features)
        action_logits = action_logits.view(-1, self.action_space.shape[0], self.action_space.shape[1])
        
        # 4. Critic: 评估状态价值
        self._last_value = self.critic(combined_features)
        self._last_cost_value = self.cost_critic(combined_features)
        
        return action_logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        """标准价值函数"""
        assert self._last_value is not None
        return self._last_value.squeeze(1)
    
    def cost_value_function(self):
        """成本价值函数"""
        assert self._last_cost_value is not None
        return self._last_cost_value.squeeze(1)
    
    def update_lagrange_multiplier(self, mean_ep_cost):
        """更新拉格朗日乘子"""
        if mean_ep_cost > self.cost_limit:
            self.lagrange_multiplier.data *= (1 + self.lambda_lr)
        else:
            self.lagrange_multiplier.data *= (1 - self.lambda_lr)
        self.lagrange_multiplier.data = torch.clamp(self.lagrange_multiplier.data, 0.0, 100.0)