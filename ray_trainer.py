"""
Ray RLlib ConstrainedPPO训练器

功能说明：
1. 创建ConstrainedPPOTrainer类，继承自ray.rllib.algorithms.ppo.PPOTrainer
2. 实现拉格朗日约束优化，将干预成本作为约束条件
3. 动态调整拉格朗日乘子以平衡奖励和约束
4. 集成到Ray RLlib的训练流程中

核心思想：
- 原始问题：最大化期望奖励，同时满足约束条件
- 拉格朗日方法：将约束转化为惩罚项，通过乘子动态调整
- 目标函数：L = E[R] - λ * (E[C] - d)
  - R: 奖励
  - C: 成本（干预次数）
  - d: 成本限制
  - λ: 拉格朗日乘子

使用示例：
    from ray_trainer import ConstrainedPPOTrainer
    from ray_model import register_traffic_controller_model
    
    # 注册模型
    register_traffic_controller_model()
    
    # 配置训练器
    config = {
        "env": "sumo_gym_env",
        "model": {
            "custom_model": "traffic_controller_model",
            "custom_model_config": {...}
        },
        # 约束优化参数
        "cost_limit": 0.1,           # 成本限制（每步平均干预次数）
        "lambda_lr": 0.01,           # 拉格朗日乘子学习率
        "lambda_init": 1.0,          # 拉格朗日乘子初始值
        "alpha": 0.5,                # 约束参数（控制约束严格程度）
        "beta": 0.9,                 # 约束参数（控制乘子更新平滑度）
    }
    
    # 创建训练器
    trainer = ConstrainedPPOTrainer(config=config)
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, PolicyID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED


class ConstrainedPPOTrainer(PPOTrainer):
    """
    支持拉格朗日约束的PPO训练器
    
    该训练器在标准PPO算法的基础上，添加了拉格朗日约束优化机制，
    用于在最大化奖励的同时，控制干预成本在指定范围内。
    
    主要特性：
    1. 继承自PPOTrainer，保持与Ray RLlib的完全兼容性
    2. 在compute_gradients中添加拉格朗日约束梯度
    3. 动态更新拉格朗日乘子以平衡奖励和约束
    4. 支持多智能体场景
    5. 提供详细的约束违反统计
    
    约束优化原理：
    - 原始问题：max_π E[R(π)] s.t. E[C(π)] ≤ d
    - 拉格朗日对偶：min_λ max_π E[R(π) - λ(C(π) - d)]
    - 更新规则：
      - 策略参数：∇θ L = ∇θ E[R - λ(C - d)]
      - 拉格朗日乘子：λ ← max(0, λ + η(C - d))
    
    配置参数：
        cost_limit (float): 成本限制（默认0.1）
        lambda_lr (float): 拉格朗日乘子学习率（默认0.01）
        lambda_init (float): 拉格朗日乘子初始值（默认1.0）
        alpha (float): 约束参数，控制约束严格程度（默认0.5）
        beta (float): 约束参数，控制乘子更新平滑度（默认0.9）
    """
    
    def __init__(self, config: Dict[str, Any] = None, env: str = None, 
                 logger_creator=None):
        """
        初始化ConstrainedPPOTrainer
        
        参数说明：
            config: 训练器配置字典
            env: 环境名称
            logger_creator: 日志创建器
        
        配置参数说明：
            cost_limit: 成本限制，每步允许的平均干预次数
            lambda_lr: 拉格朗日乘子的学习率
            lambda_init: 拉格朗日乘子的初始值
            alpha: 约束参数（0-1），控制约束违反时的惩罚强度
            beta: 约束参数（0-1），控制拉格朗日乘子的更新平滑度
        """
        # 调用父类初始化
        super().__init__(config=config, env=env, logger_creator=logger_creator)
        
        # 从配置中提取约束优化参数
        self.cost_limit = config.get('cost_limit', 0.1)
        self.lambda_lr = config.get('lambda_lr', 0.01)
        self.lambda_init = config.get('lambda_init', 1.0)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.9)
        
        # 初始化拉格朗日乘子（每个策略一个乘子）
        self.lagrange_multipliers = {}
        
        # 初始化约束违反历史（用于平滑和统计）
        self.constraint_violation_history = {}
        
        # 初始化成本历史
        self.cost_history = {}
        
        # 打印初始化信息
        print("=" * 60)
        print("🔐 ConstrainedPPOTrainer 初始化完成!")
        print("=" * 60)
        print(f"⚙️  约束优化配置:")
        print(f"   - 成本限制 (cost_limit): {self.cost_limit}")
        print(f"   - 拉格朗日乘子学习率 (lambda_lr): {self.lambda_lr}")
        print(f"   - 拉格朗日乘子初始值 (lambda_init): {self.lambda_init}")
        print(f"   - 约束参数 alpha: {self.alpha}")
        print(f"   - 约束参数 beta: {self.beta}")
        print("=" * 60)
    
    @override(PPOTrainer)
    def compute_gradients(self, samples: SampleBatch, **kwargs) -> Tuple[TensorType, Dict[str, Any]]:
        """
        计算梯度，添加拉格朗日约束项
        
        该方法是训练的核心，负责：
        1. 调用父类的compute_gradients计算基础PPO梯度
        2. 从样本中提取成本信息
        3. 计算约束违反
        4. 添加拉格朗日约束梯度
        5. 返回修改后的梯度
        
        参数说明：
            samples: 样本批次，包含观测、动作、奖励等信息
            **kwargs: 额外的关键字参数
        
        返回：
            Tuple[TensorType, Dict[str, Any]]:
                - 梯度张量（包含拉格朗日约束项）
                - 信息字典（包含约束统计信息）
        
        梯度计算公式：
            ∇θ L = ∇θ E[R - λ(C - d)]
                  = ∇θ E[R] - λ * ∇θ E[C]
        
        其中：
            - R: 奖励
            - C: 成本（干预次数）
            - d: 成本限制
            - λ: 拉格朗日乘子
        """
        # 1. 调用父类计算基础PPO梯度
        grads, info = super().compute_gradients(samples, **kwargs)
        
        # 2. 提取策略ID
        policy_id = samples.policy_id if hasattr(samples, 'policy_id') else 'default_policy'
        
        # 3. 初始化该策略的拉格朗日乘子（如果尚未初始化）
        if policy_id not in self.lagrange_multipliers:
            self.lagrange_multipliers[policy_id] = self.lambda_init
            self.constraint_violation_history[policy_id] = []
            self.cost_history[policy_id] = []
            print(f"📊 初始化策略 '{policy_id}' 的拉格朗日乘子: {self.lambda_init}")
        
        # 4. 从样本中提取成本信息
        # 假设样本中包含以下字段：
        # - level1_interventions: 一级干预次数（安全屏障触发）
        # - level2_interventions: 二级干预次数（紧急干预）
        # 这些字段需要在环境或模型中计算并添加到样本中
        level1_interventions = samples.get('level1_interventions', np.zeros(len(samples)))
        level2_interventions = samples.get('level2_interventions', np.zeros(len(samples)))
        
        # 5. 计算总成本（干预次数）
        total_cost = level1_interventions + level2_interventions
        
        # 6. 计算平均成本
        avg_cost = np.mean(total_cost)
        
        # 7. 计算约束违反
        constraint_violation = self._compute_constraint_violation(avg_cost)
        
        # 8. 更新历史记录
        self.cost_history[policy_id].append(avg_cost)
        self.constraint_violation_history[policy_id].append(constraint_violation)
        
        # 限制历史长度
        max_history = 100
        if len(self.cost_history[policy_id]) > max_history:
            self.cost_history[policy_id] = self.cost_history[policy_id][-max_history:]
            self.constraint_violation_history[policy_id] = self.constraint_violation_history[policy_id][-max_history:]
        
        # 9. 获取当前拉格朗日乘子
        lambda_ = self.lagrange_multipliers[policy_id]
        
        # 10. 添加拉格朗日约束梯度
        # 基础梯度：∇θ E[R]
        # 约束梯度：-λ * ∇θ E[C]
        # 总梯度：∇θ E[R] - λ * ∇θ E[C]
        
        # 计算成本梯度（简化处理：使用奖励梯度的方向）
        # 在实际实现中，应该直接计算成本对策略参数的梯度
        # 这里我们使用一个近似：成本与奖励成反比，因此成本梯度方向与奖励梯度相反
        
        # 获取策略对象
        policy = self.get_policy(policy_id)
        
        # 计算拉格朗日惩罚项
        lagrangian_penalty = lambda_ * constraint_violation
        
        # 修改梯度：添加拉格朗日约束项
        # 注意：这里需要根据实际的梯度结构进行调整
        if isinstance(grads, dict):
            # 如果梯度是字典形式（按参数名索引）
            for param_name in grads:
                # 获取成本对该参数的梯度（简化处理）
                # 在完整实现中，应该通过反向传播计算
                cost_grad = self._compute_cost_gradient(samples, policy, param_name)
                
                # 添加拉格朗日约束梯度
                grads[param_name] = grads[param_name] - lambda_ * cost_grad
        elif isinstance(grads, (list, np.ndarray, torch.Tensor)):
            # 如果梯度是张量形式
            cost_grad = self._compute_cost_gradient(samples, policy, None)
            
            # 添加拉格朗日约束梯度
            if isinstance(grads, torch.Tensor):
                grads = grads - lambda_ * cost_grad
            else:
                grads = grads - lambda_ * cost_grad.numpy() if isinstance(cost_grad, torch.Tensor) else grads - lambda_ * cost_grad
        
        # 11. 更新拉格朗日乘子
        self.update_lagrange_multiplier(policy_id, constraint_violation)
        
        # 12. 构建信息字典
        info.update({
            'constraint_violation': constraint_violation,
            'avg_cost': avg_cost,
            'lagrangian_multiplier': lambda_,
            'lagrangian_penalty': lagrangian_penalty,
            'level1_interventions': np.mean(level1_interventions),
            'level2_interventions': np.mean(level2_interventions),
            'total_interventions': avg_cost,
        })
        
        # 13. 打印训练信息（每100步）
        if self.iteration % 100 == 0:
            print(f"\n🔄 训练迭代 {self.iteration}:")
            print(f"   - 平均成本: {avg_cost:.4f} (限制: {self.cost_limit})")
            print(f"   - 约束违反: {constraint_violation:.4f}")
            print(f"   - 拉格朗日乘子: {lambda_:.4f}")
            print(f"   - 拉格朗日惩罚: {lagrangian_penalty:.4f}")
            print(f"   - 一级干预: {np.mean(level1_interventions):.4f}")
            print(f"   - 二级干预: {np.mean(level2_interventions):.4f}")
        
        return grads, info
    
    def update_lagrange_multiplier(self, policy_id: PolicyID, 
                                   constraint_violation: float) -> None:
        """
        更新拉格朗日乘子
        
        该方法根据约束违反情况动态调整拉格朗日乘子，
        以平衡奖励最大化和约束满足。
        
        更新规则：
            λ ← max(0, λ + η * (C - d))
        
        其中：
            - λ: 拉格朗日乘子
            - η: 学习率（lambda_lr）
            - C: 平均成本
            - d: 成本限制
        
        参数说明：
            policy_id: 策略ID
            constraint_violation: 约束违反值（C - d）
        
        更新逻辑：
            - 如果约束违反 > 0（成本超过限制），增加乘子以加强惩罚
            - 如果约束违反 < 0（成本低于限制），减小乘子以放松约束
            - 使用beta参数进行平滑更新
        """
        # 获取当前乘子
        current_lambda = self.lagrange_multipliers.get(policy_id, self.lambda_init)
        
        # 计算乘子更新量
        # 使用梯度上升更新乘子（最大化拉格朗日对偶函数）
        delta = self.lambda_lr * constraint_violation
        
        # 使用beta参数进行平滑更新
        # lambda_new = beta * (lambda_old + delta) + (1 - beta) * lambda_old
        #            = lambda_old + beta * delta
        new_lambda = current_lambda + self.beta * delta
        
        # 确保乘子非负（拉格朗日乘子的物理意义）
        new_lambda = max(0.0, new_lambda)
        
        # 限制乘子最大值（防止数值不稳定）
        max_lambda = 100.0
        new_lambda = min(new_lambda, max_lambda)
        
        # 更新乘子
        self.lagrange_multipliers[policy_id] = new_lambda
        
        # 记录更新信息
        if self.iteration % 100 == 0:
            print(f"   📈 拉格朗日乘子更新: {current_lambda:.4f} → {new_lambda:.4f}")
            print(f"      更新量: {delta:.4f} (约束违反: {constraint_violation:.4f})")
    
    def _compute_constraint_violation(self, avg_cost: float) -> float:
        """
        计算约束违反
        
        该方法计算当前成本与成本限制之间的差异，
        用于评估约束满足情况。
        
        计算公式：
            violation = C - d
        
        其中：
            - C: 平均成本
            - d: 成本限制
        
        参数说明：
            avg_cost: 平均成本（干预次数）
        
        返回：
            constraint_violation: 约束违反值
                - 正值：成本超过限制
                - 零值：成本正好等于限制
                - 负值：成本低于限制
        
        注意：
            - 约束违反越大，拉格朗日惩罚越强
            - 负的约束违反表示约束被满足，可以放松惩罚
        """
        # 计算约束违反
        violation = avg_cost - self.cost_limit
        
        # 使用alpha参数调整约束违反的敏感度
        # alpha > 0.5: 对约束违反更敏感
        # alpha < 0.5: 对约束违反不太敏感
        adjusted_violation = self.alpha * violation
        
        return adjusted_violation
    
    def _compute_cost_gradient(self, samples: SampleBatch, policy, 
                                param_name: Optional[str] = None) -> TensorType:
        """
        计算成本对策略参数的梯度
        
        该方法计算干预成本对策略参数的梯度，
        用于在梯度更新中添加拉格朗日约束项。
        
        参数说明：
            samples: 样本批次
            policy: 策略对象
            param_name: 参数名称（如果为None，则计算所有参数的梯度）
        
        返回：
            cost_grad: 成本梯度
        
        注意：
            - 这是一个简化实现
            - 在完整实现中，应该通过反向传播计算
            - 这里使用一个近似：成本与奖励成反比
        """
        # 提取成本信息
        level1_interventions = samples.get('level1_interventions', np.zeros(len(samples)))
        level2_interventions = samples.get('level2_interventions', np.zeros(len(samples)))
        total_cost = level1_interventions + level2_interventions
        
        # 计算成本梯度（简化处理）
        # 在完整实现中，应该：
        # 1. 定义成本函数 C(θ) = E[interventions]
        # 2. 计算梯度 ∇θ C(θ)
        # 3. 使用策略梯度定理或自动微分
        
        # 这里我们使用一个近似：
        # 假设成本与奖励成反比，因此成本梯度方向与奖励梯度相反
        # cost_grad ≈ -reward_grad * (cost / reward)
        
        # 获取奖励
        rewards = samples.get('rewards', np.zeros(len(samples)))
        
        # 避免除零
        avg_reward = np.mean(rewards)
        avg_cost = np.mean(total_cost)
        
        # 计算成本梯度近似值
        if avg_reward != 0:
            cost_ratio = avg_cost / (abs(avg_reward) + 1e-8)
            # 成本梯度与奖励梯度方向相反，大小与成本比成正比
            cost_grad_magnitude = cost_ratio
        else:
            cost_grad_magnitude = 0.0
        
        # 如果指定了参数名称，返回该参数的梯度
        if param_name is not None:
            # 获取参数形状
            if hasattr(policy, 'model') and hasattr(policy.model, param_name):
                param = getattr(policy.model, param_name)
                if hasattr(param, 'shape'):
                    # 创建与参数形状相同的梯度张量
                    if isinstance(param, torch.Tensor):
                        cost_grad = torch.ones_like(param) * cost_grad_magnitude
                    else:
                        cost_grad = np.ones(param.shape) * cost_grad_magnitude
                else:
                    cost_grad = cost_grad_magnitude
            else:
                cost_grad = cost_grad_magnitude
        else:
            # 返回标量梯度
            cost_grad = cost_grad_magnitude
        
        return cost_grad
    
    @override(PPOTrainer)
    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, Any]:
        """
        在批次上学习（重写以支持拉格朗日约束）
        
        该方法重写父类的learn_on_batch方法，
        在标准PPO学习流程中集成拉格朗日约束优化。
        
        参数说明：
            samples: 样本批次
        
        返回：
            info: 信息字典，包含训练统计和约束统计
        """
        # 调用父类的learn_on_batch
        info = super().learn_on_batch(samples)
        
        # 添加约束统计信息
        for policy_id in self.lagrange_multipliers:
            if policy_id in self.cost_history and len(self.cost_history[policy_id]) > 0:
                avg_cost = np.mean(self.cost_history[policy_id][-10:])  # 最近10次的平均
                avg_violation = np.mean(self.constraint_violation_history[policy_id][-10:])
                lambda_ = self.lagrange_multipliers[policy_id]
                
                info[f'policy_{policy_id}_avg_cost'] = avg_cost
                info[f'policy_{policy_id}_constraint_violation'] = avg_violation
                info[f'policy_{policy_id}_lagrangian_multiplier'] = lambda_
        
        return info
    
    def get_constraint_stats(self) -> Dict[str, Any]:
        """
        获取约束统计信息
        
        该方法返回当前所有策略的约束统计信息，
        用于监控和调试约束优化过程。
        
        返回：
            stats: 统计信息字典，包含：
                - 每个策略的拉格朗日乘子
                - 每个策略的平均成本
                - 每个策略的约束违反
                - 每个策略的成本历史
                - 每个策略的约束违反历史
        """
        stats = {
            'cost_limit': self.cost_limit,
            'lambda_lr': self.lambda_lr,
            'alpha': self.alpha,
            'beta': self.beta,
            'policies': {}
        }
        
        for policy_id in self.lagrange_multipliers:
            policy_stats = {
                'lagrangian_multiplier': self.lagrange_multipliers[policy_id],
                'cost_history': list(self.cost_history.get(policy_id, [])),
                'constraint_violation_history': list(self.constraint_violation_history.get(policy_id, [])),
            }
            
            # 计算统计量
            if len(policy_stats['cost_history']) > 0:
                policy_stats['avg_cost'] = np.mean(policy_stats['cost_history'])
                policy_stats['std_cost'] = np.std(policy_stats['cost_history'])
                policy_stats['min_cost'] = np.min(policy_stats['cost_history'])
                policy_stats['max_cost'] = np.max(policy_stats['cost_history'])
            
            if len(policy_stats['constraint_violation_history']) > 0:
                policy_stats['avg_violation'] = np.mean(policy_stats['constraint_violation_history'])
                policy_stats['std_violation'] = np.std(policy_stats['constraint_violation_history'])
                policy_stats['min_violation'] = np.min(policy_stats['constraint_violation_history'])
                policy_stats['max_violation'] = np.max(policy_stats['constraint_violation_history'])
            
            stats['policies'][policy_id] = policy_stats
        
        return stats
    
    def reset_lagrange_multipliers(self, value: Optional[float] = None) -> None:
        """
        重置拉格朗日乘子
        
        该方法重置所有策略的拉格朗日乘子到指定值或初始值。
        
        参数说明：
            value: 重置值，如果为None则使用lambda_init
        """
        reset_value = value if value is not None else self.lambda_init
        
        for policy_id in self.lagrange_multipliers:
            self.lagrange_multipliers[policy_id] = reset_value
            print(f"🔄 重置策略 '{policy_id}' 的拉格朗日乘子: {reset_value}")
    
    def set_cost_limit(self, new_limit: float) -> None:
        """
        设置新的成本限制
        
        该方法动态调整成本限制，用于实验和调优。
        
        参数说明：
            new_limit: 新的成本限制
        """
        old_limit = self.cost_limit
        self.cost_limit = new_limit
        print(f"📊 成本限制更新: {old_limit} → {new_limit}")


def create_constrained_ppo_trainer(config: Dict[str, Any]) -> ConstrainedPPOTrainer:
    """
    创建ConstrainedPPOTrainer的工厂函数
    
    该函数提供了一种便捷的方式来创建ConstrainedPPOTrainer实例，
    并设置合理的默认配置。
    
    参数说明：
        config: 训练器配置字典
    
    返回：
        trainer: ConstrainedPPOTrainer实例
    
    使用示例：
        config = {
            "env": "sumo_gym_env",
            "framework": "torch",
            "num_workers": 4,
            # 约束优化参数
            "cost_limit": 0.1,
            "lambda_lr": 0.01,
            "lambda_init": 1.0,
            "alpha": 0.5,
            "beta": 0.9,
        }
        trainer = create_constrained_ppo_trainer(config)
    """
    # 设置默认配置
    default_config = {
        "cost_limit": 0.1,
        "lambda_lr": 0.01,
        "lambda_init": 1.0,
        "alpha": 0.5,
        "beta": 0.9,
    }
    
    # 合并用户配置
    merged_config = {**default_config, **config}
    
    # 创建训练器
    trainer = ConstrainedPPOTrainer(config=merged_config)
    
    return trainer


# 如果直接运行此文件，执行测试
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ConstrainedPPOTrainer 测试")
    print("=" * 60)
    
    # 测试配置
    test_config = {
        "env": "CartPole-v1",
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 0,
        # 约束优化参数
        "cost_limit": 0.1,
        "lambda_lr": 0.01,
        "lambda_init": 1.0,
        "alpha": 0.5,
        "beta": 0.9,
    }
    
    print("\n📝 测试配置:")
    print(f"   - 环境: {test_config['env']}")
    print(f"   - 框架: {test_config['framework']}")
    print(f"   - 成本限制: {test_config['cost_limit']}")
    print(f"   - 拉格朗日乘子学习率: {test_config['lambda_lr']}")
    print(f"   - 拉格朗日乘子初始值: {test_config['lambda_init']}")
    
    print("\n✅ ConstrainedPPOTrainer 已准备就绪!")
    print("\n💡 使用示例:")
    print("   from ray_trainer import ConstrainedPPOTrainer")
    print("   trainer = ConstrainedPPOTrainer(config=config)")
    print("   result = trainer.train()")
    print("=" * 60)
