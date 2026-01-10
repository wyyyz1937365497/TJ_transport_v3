"""
Ray RLlib 自定义模型包装器
将TrafficController模型无缝集成到Ray RLlib的训练流程中

功能说明：
1. 创建TrafficControllerModel类，继承自ray.rllib.models.ModelV2
2. 在forward方法中返回TrafficController实例
3. Ray RLlib会自动处理设备放置（CPU/GPU）、梯度计算、参数更新等
4. 确保TrafficController的forward方法与Ray RLlib兼容
5. 保留所有现有的GNN前向传播逻辑和世界模型预测逻辑

模型配置参数：
- node_dim: 9
- edge_dim: 4
- gnn_hidden_dim: 64
- gnn_output_dim: 256
- gnn_layers: 3
- gnn_heads: 4
- world_hidden_dim: 128
- future_steps: 5
- controller_hidden_dim: 128
- global_dim: 16
- top_k: 5
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import gymnasium as gym
from ray.rllib.models import ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

# 导入TrafficController及其组件
from neural_traffic_controller import TrafficController
from risk_sensitive_gnn import RiskSensitiveGNN
from progressive_world_model import ProgressiveWorldModel
from influence_controller import InfluenceDrivenController
from safety_shield import DualModeSafetyShield


class TrafficControllerModel(TorchModelV2, nn.Module):
    """
    TrafficController的Ray RLlib包装器
    
    该类将TrafficController模型包装为Ray RLlib兼容的模型，
    支持分布式训练、GPU加速、自动梯度计算等功能。
    
    架构：
    - 输入：交通状态观测（包含车辆节点特征、边特征、全局指标等）
    - 内部：RiskSensitiveGNN + ProgressiveWorldModel + InfluenceDrivenController + DualModeSafetyShield
    - 输出：动作分布（连续动作空间）和价值估计
    
    Ray RLlib集成要点：
    1. 继承TorchModelV2以支持PyTorch后端
    2. 实现__init__方法初始化模型
    3. 实现forward方法进行前向传播
    4. 实现value_function方法返回价值估计
    5. 支持设备自动放置（CPU/GPU）
    """
    
    def __init__(self, obs_space: gym.spaces.Space, 
                 action_space: gym.spaces.Space,
                 num_outputs: int, 
                 model_config: ModelConfigDict,
                 name: str):
        """
        初始化TrafficControllerModel
        
        参数说明：
            obs_space: 观测空间，应为包含以下键的Dict空间：
                - node_features: 车辆节点特征 [N, 9]
                - edge_indices: 边索引 [2, E]
                - edge_features: 边特征 [E, 4]
                - global_metrics: 全局交通指标 [16]
                - vehicle_ids: 车辆ID列表
                - is_icv: 是否为智能网联车 [N]
                - vehicle_states: 车辆状态字典
            action_space: 动作空间，应为Box空间 [2]（加速度，换道概率）
            num_outputs: 输出维度，与action_space维度相同
            model_config: 模型配置字典，包含以下键：
                - node_dim: 节点特征维度（默认9）
                - edge_dim: 边特征维度（默认4）
                - gnn_hidden_dim: GNN隐藏层维度（默认64）
                - gnn_output_dim: GNN输出维度（默认256）
                - gnn_layers: GNN层数（默认3）
                - gnn_heads: GNN注意力头数（默认4）
                - world_hidden_dim: 世界模型隐藏层维度（默认128）
                - future_steps: 未来预测步数（默认5）
                - controller_hidden_dim: 控制器隐藏层维度（默认128）
                - global_dim: 全局特征维度（默认16）
                - top_k: 选择的最具影响力车辆数（默认5）
                - action_dim: 动作维度（默认2）
            name: 模型名称
        """
        # 初始化父类
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # 从model_config中提取配置参数
        self.config = {
            'node_dim': model_config.get('node_dim', 9),
            'edge_dim': model_config.get('edge_dim', 4),
            'gnn_hidden_dim': model_config.get('gnn_hidden_dim', 64),
            'gnn_output_dim': model_config.get('gnn_output_dim', 256),
            'gnn_layers': model_config.get('gnn_layers', 3),
            'gnn_heads': model_config.get('gnn_heads', 4),
            'world_hidden_dim': model_config.get('world_hidden_dim', 128),
            'future_steps': model_config.get('future_steps', 5),
            'controller_hidden_dim': model_config.get('controller_hidden_dim', 128),
            'global_dim': model_config.get('global_dim', 16),
            'top_k': model_config.get('top_k', 5),
            'action_dim': model_config.get('action_dim', 2),
            # 安全参数
            'ttc_threshold': model_config.get('ttc_threshold', 2.0),
            'thw_threshold': model_config.get('thw_threshold', 1.5),
            'max_accel': model_config.get('max_accel', 2.0),
            'max_decel': model_config.get('max_decel', -3.0),
            'emergency_decel': model_config.get('emergency_decel', -5.0),
            'max_lane_change_speed': model_config.get('max_lane_change_speed', 5.0),
            # 约束优化参数
            'cost_limit': model_config.get('cost_limit', 0.1),
            'lambda_lr': model_config.get('lambda_lr', 0.01),
            # 缓存参数
            'cache_timeout': model_config.get('cache_timeout', 10)
        }
        
        # 创建TrafficController实例
        self.traffic_controller = TrafficController(self.config)
        
        # 动作输出层（将TrafficController的输出映射到动作空间）
        # 注意：TrafficController已经包含了动作生成网络，这里主要用于适配RLlib的输出格式
        self.action_output = nn.Sequential(
            nn.Linear(self.config['action_dim'], self.config['action_dim']),
            nn.Tanh()  # 输出范围[-1, 1]，后续会映射到实际动作范围
        )
        
        # 价值函数头（用于Actor-Critic算法）
        self.value_head = nn.Sequential(
            nn.Linear(self.config['gnn_output_dim'], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 保存价值函数输出（在forward中计算）
        self._value_out = None
        
        # 打印初始化信息
        print("=" * 60)
        print("🚀 TrafficControllerModel (Ray RLlib) 初始化完成!")
        print("=" * 60)
        print(f"📊 模型配置:")
        print(f"   - 节点维度: {self.config['node_dim']}")
        print(f"   - 边维度: {self.config['edge_dim']}")
        print(f"   - GNN隐藏维度: {self.config['gnn_hidden_dim']}")
        print(f"   - GNN输出维度: {self.config['gnn_output_dim']}")
        print(f"   - GNN层数: {self.config['gnn_layers']}")
        print(f"   - GNN注意力头数: {self.config['gnn_heads']}")
        print(f"   - 世界模型隐藏维度: {self.config['world_hidden_dim']}")
        print(f"   - 未来预测步数: {self.config['future_steps']}")
        print(f"   - 控制器隐藏维度: {self.config['controller_hidden_dim']}")
        print(f"   - 全局维度: {self.config['global_dim']}")
        print(f"   - Top-K车辆数: {self.config['top_k']}")
        print(f"   - 动作维度: {self.config['action_dim']}")
        print(f"🛡️  安全参数:")
        print(f"   - TTC阈值: {self.config['ttc_threshold']}s")
        print(f"   - THW阈值: {self.config['thw_threshold']}s")
        print(f"   - 最大加速度: {self.config['max_accel']} m/s²")
        print(f"   - 最大减速度: {self.config['max_decel']} m/s²")
        print(f"   - 紧急减速度: {self.config['emergency_decel']} m/s²")
        print("=" * 60)
    
    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], 
                state: List[TensorType], 
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        """
        前向传播方法（Ray RLlib接口）
        
        该方法是Ray RLlib训练循环的核心，负责：
        1. 从input_dict中提取观测数据
        2. 构建TrafficController所需的输入格式
        3. 调用TrafficController进行推理
        4. 返回动作分布和价值估计
        
        参数说明：
            input_dict: 输入字典，包含：
                - obs: 观测数据（Dict空间）
                - obs_flat: 展平的观测数据（如果使用）
            state: RNN状态（本模型不使用RNN，保持为空列表）
            seq_lens: 序列长度（本模型不使用序列，保持为None）
        
        返回：
            Tuple[TensorType, List[TensorType]]: 
                - 动作logits或分布参数
                - RNN状态（本模型为空列表）
        
        注意：
            - Ray RLlib会自动处理batch维度
            - 设备放置由Ray RLlib自动管理
            - 梯度计算由Ray RLlib自动处理
        """
        # 1. 从input_dict中提取观测数据
        obs = input_dict["obs"]
        
        # 2. 构建TrafficController所需的batch格式
        # 注意：obs应该是一个Dict，包含以下键
        batch = self._prepare_batch(obs)
        
        # 3. 调用TrafficController进行前向传播
        # 使用虚拟step参数（实际训练时由RLlib管理）
        controller_output = self.traffic_controller(batch, step=0)
        
        # 4. 提取动作和价值估计
        # 获取安全动作（已经过安全屏障处理）
        safe_actions = controller_output['safe_actions']  # [K, 2]
        
        # 获取GNN嵌入用于价值估计
        gnn_embedding = controller_output['gnn_embedding']  # [N, 256]
        
        # 5. 计算价值函数（对所有车辆的平均）
        if len(gnn_embedding) > 0:
            value_out = self.value_head(gnn_embedding).mean(dim=0)  # [1]
        else:
            value_out = torch.zeros(1, device=gnn_embedding.device)
        
        # 保存价值输出供value_function方法使用
        self._value_out = value_out
        
        # 6. 处理动作输出
        # 如果有选中的车辆，返回其动作；否则返回零动作
        if len(safe_actions) > 0:
            # 对选中的车辆动作取平均（简化处理）
            # 实际应用中可能需要更复杂的处理逻辑
            action_output = safe_actions.mean(dim=0, keepdim=True)  # [1, 2]
        else:
            # 如果没有选中车辆，返回零动作
            action_output = torch.zeros(1, self.config['action_dim'], 
                                       device=gnn_embedding.device)
        
        # 7. 返回动作输出和空状态列表
        # Ray RLlib期望输出形状为[batch_size, num_outputs]
        return action_output, state
    
    def _prepare_batch(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备TrafficController所需的batch格式
        
        该方法将Ray RLlib的观测格式转换为TrafficController期望的格式。
        
        参数说明：
            obs: Ray RLlib的观测数据（Dict空间）
        
        返回：
            batch: TrafficController所需的batch字典
        """
        # 创建batch字典
        batch = {}
        
        # 1. 节点特征 [N, 9]
        batch['node_features'] = self._ensure_tensor(obs['node_features'])
        
        # 2. 边索引 [2, E]
        batch['edge_indices'] = self._ensure_tensor(obs['edge_indices'], dtype=torch.long)
        
        # 3. 边特征 [E, 4]
        batch['edge_features'] = self._ensure_tensor(obs['edge_features'])
        
        # 4. 全局指标 [16]
        batch['global_metrics'] = self._ensure_tensor(obs['global_metrics'])
        
        # 5. 车辆ID列表
        batch['vehicle_ids'] = obs['vehicle_ids']
        
        # 6. ICV掩码 [N]
        batch['is_icv'] = self._ensure_tensor(obs['is_icv'], dtype=torch.float32)
        
        # 7. 车辆状态字典
        batch['vehicle_states'] = obs['vehicle_states']
        
        return batch
    
    def _ensure_tensor(self, data: Union[np.ndarray, torch.Tensor], 
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        确保数据为torch.Tensor类型，并移动到正确的设备
        
        参数说明：
            data: 输入数据（numpy数组或torch张量）
            dtype: 目标数据类型
        
        返回：
            tensor: 转换后的torch张量
        """
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype)
        else:
            tensor = torch.tensor(data, dtype=dtype)
        
        # 确保张量在正确的设备上（与模型参数相同）
        device = next(self.parameters()).device
        return tensor.to(device)
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        返回价值函数估计（Ray RLlib接口）
        
        该方法用于Actor-Critic算法，返回当前状态的价值估计。
        
        返回：
            value_out: 价值估计 [batch_size]
        
        注意：
            - 该方法必须在forward之后调用
            - 价值估计在forward中计算并存储在self._value_out中
        """
        assert self._value_out is not None, "value_function() called before forward()"
        return self._value_out.squeeze(-1)  # [batch_size]


class TrafficControllerModelV2(TrafficControllerModel):
    """
    TrafficControllerModel的V2版本
    
    该版本提供了更灵活的接口，支持：
    1. 直接返回所有选中车辆的动作（而非平均）
    2. 支持多智能体场景
    3. 提供更详细的信息输出
    """
    
    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], 
                state: List[TensorType], 
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        """
        前向传播方法（V2版本）
        
        该版本返回所有选中车辆的动作，而非简单平均。
        """
        # 1. 从input_dict中提取观测数据
        obs = input_dict["obs"]
        
        # 2. 构建TrafficController所需的batch格式
        batch = self._prepare_batch(obs)
        
        # 3. 调用TrafficController进行前向传播
        controller_output = self.traffic_controller(batch, step=0)
        
        # 4. 提取安全动作
        safe_actions = controller_output['safe_actions']  # [K, 2]
        
        # 5. 获取GNN嵌入用于价值估计
        gnn_embedding = controller_output['gnn_embedding']  # [N, 256]
        
        # 6. 计算价值函数
        if len(gnn_embedding) > 0:
            value_out = self.value_head(gnn_embedding).mean(dim=0)  # [1]
        else:
            value_out = torch.zeros(1, device=gnn_embedding.device)
        
        # 保存价值输出
        self._value_out = value_out
        
        # 7. 处理动作输出
        # 如果有选中的车辆，返回第一个车辆的动作（作为代表）
        # 或者可以返回所有动作的拼接
        if len(safe_actions) > 0:
            # 返回第一个选中车辆的动作
            action_output = safe_actions[0:1]  # [1, 2]
        else:
            action_output = torch.zeros(1, self.config['action_dim'], 
                                       device=gnn_embedding.device)
        
        # 8. 保存额外的信息供后续使用
        self._controller_output = controller_output
        
        # 9. 返回动作输出和空状态列表
        return action_output, state
    
    def get_controller_output(self) -> Dict[str, Any]:
        """
        获取TrafficController的完整输出
        
        该方法可以用于获取详细的控制信息，包括：
        - 选中的车辆ID
        - 影响力得分
        - 安全干预统计
        - GNN嵌入
        - 世界模型预测
        
        返回：
            controller_output: TrafficController的完整输出字典
        """
        return getattr(self, '_controller_output', {})


def register_traffic_controller_model():
    """
    注册TrafficControllerModel到Ray RLlib的ModelCatalog
    
    该函数应在训练脚本开始时调用，以便Ray RLlib能够识别和使用自定义模型。
    
    使用示例：
        from ray_model import register_traffic_controller_model
        
        # 注册模型
        register_traffic_controller_model()
        
        # 在配置中使用
        config = {
            "model": {
                "custom_model": "traffic_controller_model",
                "custom_model_config": {
                    "node_dim": 9,
                    "edge_dim": 4,
                    "gnn_hidden_dim": 64,
                    "gnn_output_dim": 256,
                    "gnn_layers": 3,
                    "gnn_heads": 4,
                    "world_hidden_dim": 128,
                    "future_steps": 5,
                    "controller_hidden_dim": 128,
                    "global_dim": 16,
                    "top_k": 5,
                    "action_dim": 2
                }
            }
        }
    """
    from ray.rllib.models import ModelCatalog
    
    # 注册模型
    ModelCatalog.register_custom_model("traffic_controller_model", TrafficControllerModel)
    ModelCatalog.register_custom_model("traffic_controller_model_v2", TrafficControllerModelV2)
    
    print("✅ TrafficControllerModel已注册到Ray RLlib ModelCatalog")
    print("   - traffic_controller_model: 基础版本")
    print("   - traffic_controller_model_v2: 增强版本")


# 如果直接运行此文件，执行注册
if __name__ == "__main__":
    register_traffic_controller_model()
    print("\n📝 模型注册完成！现在可以在Ray RLlib配置中使用:")
    print("   config['model']['custom_model'] = 'traffic_controller_model'")
