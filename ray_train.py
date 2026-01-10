"""
Ray RLlib 训练脚本 - SUMO交通控制

功能说明：
1. 集成SUMO-RL环境（sumo_gym_env.py）和自定义模型包装器（ray_model.py）
2. 集成ConstrainedPPO训练器（ray_trainer.py）
3. 配置Ray RolloutWorkers（4个并行SUMO进程）
4. 配置异步训练架构，实现时间重叠
5. 启用LIBSUMO_AS_TRACI加速和批量订阅
6. 配置GPU训练进程，实现异步模型更新
7. 实现实时数据收集，不支持从JSON文件加载数据
8. 包含数据验证、梯度裁剪、混合精度训练（AMP）
9. 实现基于拉格朗日乘子的奖励重塑逻辑
10. 配置性能优化参数：num_workers、train_batch_size、rollout_fragment_length
11. 添加详细的训练日志和进度显示
12. 实现检查点保存和恢复
13. 添加TensorBoard日志记录

训练流程：
- Ray Driver启动多个RolloutWorkers
- 每个Worker运行独立的SUMO实例
- Workers并行收集rollout数据
- GPU训练进程异步更新模型
- 实现时间重叠：SUMO生成新数据的同时，GPU使用旧数据训练

使用示例：
    python ray_train.py --config config.json --restore checkpoint_path
"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

# Ray RLlib导入
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED

# 本地导入
from sumo_gym_env import SUMOGymEnv, create_sumo_gym_env
from ray_model import (
    TrafficControllerModel,
    TrafficControllerModelV2,
    register_traffic_controller_model
)
from ray_trainer import ConstrainedPPOTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 环境配置
# ============================================================================

def get_default_config() -> Dict[str, Any]:
    """
    获取默认训练配置
    
    Returns:
        config: 默认配置字典
    """
    return {
        # ==================== 基础配置 ====================
        "framework": "torch",
        "env": SUMOGymEnv,
        
        # ==================== SUMO环境配置 ====================
        "sumo_cfg_path": "仿真环境-初赛/sumo.sumocfg",
        "use_libsumo": True,  # 启用LIBSUMO_AS_TRACI加速
        "batch_subscribe": True,  # 启用批量订阅
        "max_steps": 3600,  # 每个episode的最大步数
        "use_gui": False,
        
        # ==================== Ray并行配置 ====================
        "num_workers": 4,  # 4个并行RolloutWorkers
        "num_gpus": 1,  # 使用1个GPU进行训练
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,  # 每个Worker运行1个SUMO实例
        "worker_use_gpu": False,  # Workers使用CPU，训练进程使用GPU
        
        # ==================== 异步训练配置 ====================
        "train_batch_size": 4000,  # 每次训练使用的样本数
        "rollout_fragment_length": 200,  # 每个rollout片段的长度
        "sgd_minibatch_size": 128,  # SGD小批次大小
        "num_sgd_iter": 10,  # 每次训练迭代的SGD更新次数
        
        # ==================== PPO算法配置 ====================
        "lr": 3e-4,  # 学习率
        "gamma": 0.99,  # 折扣因子
        "lambda_": 0.95,  # GAE参数
        "clip_param": 0.2,  # PPO裁剪参数
        "vf_loss_coeff": 0.5,  # 价值函数损失系数
        "entropy_coeff": 0.01,  # 熵正则化系数
        "kl_coeff": 0.2,  # KL散度系数
        "kl_target": 0.01,  # KL散度目标值
        
        # ==================== 梯度优化配置 ====================
        "grad_clip": 0.5,  # 梯度裁剪阈值
        "use_amp": True,  # 启用混合精度训练（AMP）
        
        # ==================== 约束优化配置 ====================
        "cost_limit": 0.1,  # 成本限制（每步平均干预次数）
        "lambda_lr": 0.01,  # 拉格朗日乘子学习率
        "lambda_init": 1.0,  # 拉格朗日乘子初始值
        "alpha": 0.5,  # 约束参数
        "beta": 0.9,  # 约束参数
        
        # ==================== 模型配置 ====================
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
                "action_dim": 2,
                # 安全参数
                "ttc_threshold": 2.0,
                "thw_threshold": 1.5,
                "max_accel": 2.0,
                "max_decel": -3.0,
                "emergency_decel": -5.0,
                "max_lane_change_speed": 5.0,
                # 约束优化参数
                "cost_limit": 0.1,
                "lambda_lr": 0.01,
            }
        },
        
        # ==================== 检查点配置 ====================
        "checkpoint_freq": 10,  # 每10次迭代保存一次检查点
        "checkpoint_at_end": True,  # 训练结束时保存检查点
        "keep_checkpoints_num": 5,  # 保留最近5个检查点
        "checkpoint_score_attr": "episode_reward_mean",
        
        # ==================== 日志配置 ====================
        "log_level": "INFO",
        "log_dir": "./ray_results",
        "experiment_name": "sumo_traffic_control",
        
        # ==================== TensorBoard配置 ====================
        "tensorboard_log": True,
        
        # ==================== 训练配置 ====================
        "stop": {
            "training_iteration": 1000,  # 最大训练迭代次数
            "episode_reward_mean": 100.0,  # 达到此平均奖励时停止
        },
        
        # ==================== 恢复配置 ====================
        "restore": None,  # 检查点路径
    }


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"✅ 从 {config_path} 加载配置")
    return config


def merge_configs(default_config: Dict[str, Any], 
                  user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并默认配置和用户配置
    
    Args:
        default_config: 默认配置
        user_config: 用户配置
        
    Returns:
        merged_config: 合并后的配置
    """
    merged = default_config.copy()
    
    # 递归合并嵌套字典
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


# ============================================================================
# 环境创建函数
# ============================================================================

def env_creator(env_config: Dict[str, Any]) -> SUMOGymEnv:
    """
    环境创建函数（Ray RLlib接口）
    
    Args:
        env_config: 环境配置字典
        
    Returns:
        env: SUMOGymEnv实例
    """
    # 从env_config中提取SUMO配置
    sumo_cfg_path = env_config.get("sumo_cfg_path", "仿真环境-初赛/sumo.sumocfg")
    use_libsumo = env_config.get("use_libsumo", True)
    batch_subscribe = env_config.get("batch_subscribe", True)
    max_steps = env_config.get("max_steps", 3600)
    use_gui = env_config.get("use_gui", False)
    
    # 获取模型配置
    model_config = env_config.get("model_config", {})
    
    # 创建环境
    env = create_sumo_gym_env(
        sumo_cfg_path=sumo_cfg_path,
        use_libsumo=use_libsumo,
        batch_subscribe=batch_subscribe,
        device='cpu',  # Workers使用CPU
        model_config=model_config,
        max_steps=max_steps,
        use_gui=use_gui
    )
    
    logger.info(f"✅ 创建SUMO环境: {sumo_cfg_path}")
    return env


# ============================================================================
# 自定义回调函数
# ============================================================================

class TrainingCallback:
    """
    训练回调函数
    
    用于监控训练进度、记录日志、保存检查点等
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化回调函数
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.start_time = time.time()
        self.best_reward = -np.inf
        self.best_iteration = 0
        
        # 创建日志目录
        self.log_dir = config.get("log_dir", "./ray_results")
        self.experiment_name = config.get("experiment_name", "sumo_traffic_control")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 创建日志文件
        log_file = os.path.join(self.run_dir, "training.log")
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(self.file_handler)
        
        logger.info(f"✅ 训练回调初始化完成")
        logger.info(f"   运行目录: {self.run_dir}")
    
    def on_train_result(self, result: Dict[str, Any]) -> None:
        """
        训练结果回调
        
        Args:
            result: 训练结果字典
        """
        iteration = result.get("training_iteration", 0)
        
        # 计算训练时间
        elapsed_time = time.time() - self.start_time
        
        # 提取关键指标
        episode_reward_mean = result.get("episode_reward_mean", 0.0)
        episode_len_mean = result.get("episode_len_mean", 0.0)
        
        # 提取约束统计
        constraint_violation = result.get("constraint_violation", 0.0)
        avg_cost = result.get("avg_cost", 0.0)
        lagrangian_multiplier = result.get("lagrangian_multiplier", 0.0)
        
        # 提取安全指标
        ttc_violations = result.get("ttc_violations", 0)
        thw_violations = result.get("thw_violations", 0)
        
        # 提取训练统计
        agent_steps = result.get(NUM_AGENT_STEPS_SAMPLED, 0)
        env_steps = result.get(NUM_ENV_STEPS_SAMPLED, 0)
        
        # 打印训练进度
        print("\n" + "=" * 80)
        print(f"🚀 训练迭代 {iteration}")
        print("=" * 80)
        print(f"⏱️  训练时间: {elapsed_time:.2f}秒")
        print(f"📊 性能指标:")
        print(f"   - 平均奖励: {episode_reward_mean:.4f}")
        print(f"   - 平均Episode长度: {episode_len_mean:.2f}")
        print(f"🛡️  安全指标:")
        print(f"   - TTC违规: {ttc_violations}")
        print(f"   - THW违规: {thw_violations}")
        print(f"🔐 约束优化:")
        print(f"   - 约束违反: {constraint_violation:.4f}")
        print(f"   - 平均成本: {avg_cost:.4f} (限制: {self.config['cost_limit']})")
        print(f"   - 拉格朗日乘子: {lagrangian_multiplier:.4f}")
        print(f"📈 训练统计:")
        print(f"   - Agent步数: {agent_steps}")
        print(f"   - 环境步数: {env_steps}")
        print(f"   - 学习率: {result.get('policy_learn_rate', 0):.6f}")
        print(f"   - 熵: {result.get('policy_entropy', 0):.4f}")
        print("=" * 80)
        
        # 更新最佳奖励
        if episode_reward_mean > self.best_reward:
            self.best_reward = episode_reward_mean
            self.best_iteration = iteration
            print(f"🎉 新的最佳奖励: {self.best_reward:.4f} (迭代 {iteration})")
        
        # 保存训练结果到文件
        self._save_training_result(result)
    
    def _save_training_result(self, result: Dict[str, Any]) -> None:
        """
        保存训练结果到JSON文件
        
        Args:
            result: 训练结果字典
        """
        # 创建results目录
        results_dir = os.path.join(self.run_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存当前迭代结果
        iteration = result.get("training_iteration", 0)
        result_file = os.path.join(results_dir, f"result_{iteration:06d}.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def on_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """
        检查点保存回调
        
        Args:
            checkpoint_info: 检查点信息字典
        """
        checkpoint_path = checkpoint_info.get("checkpoint", "")
        logger.info(f"💾 检查点已保存: {checkpoint_path}")
    
    def close(self) -> None:
        """关闭回调函数"""
        logger.info("📊 训练完成")
        logger.info(f"   最佳奖励: {self.best_reward:.4f} (迭代 {self.best_iteration})")
        logger.info(f"   总训练时间: {time.time() - self.start_time:.2f}秒")
        logger.removeHandler(self.file_handler)
        self.file_handler.close()


# ============================================================================
# 奖励重塑函数
# ============================================================================

def reward_shaping_with_lagrangian(
    batch: Dict[str, Any],
    lambda_: float,
    cost_limit: float
) -> Dict[str, Any]:
    """
    基于拉格朗日乘子的奖励重塑
    
    将约束成本转化为奖励惩罚，实现约束优化。
    
    奖励重塑公式：
        R' = R - λ * (C - d)
    
    其中：
        - R: 原始奖励
        - C: 成本（干预次数）
        - d: 成本限制
        - λ: 拉格朗日乘子
    
    Args:
        batch: 样本批次
        lambda_: 拉格朗日乘子
        cost_limit: 成本限制
        
    Returns:
        batch: 奖励重塑后的批次
    """
    # 提取原始奖励
    rewards = batch.get("rewards", np.zeros(len(batch)))
    
    # 提取成本（干预次数）
    level1_interventions = batch.get("level1_interventions", np.zeros(len(batch)))
    level2_interventions = batch.get("level2_interventions", np.zeros(len(batch)))
    total_cost = level1_interventions + level2_interventions
    
    # 计算约束违反
    constraint_violation = total_cost - cost_limit
    
    # 计算拉格朗日惩罚
    lagrangian_penalty = lambda_ * constraint_violation
    
    # 重塑奖励
    shaped_rewards = rewards - lagrangian_penalty
    
    # 更新批次
    batch["rewards"] = shaped_rewards
    batch["original_rewards"] = rewards
    batch["lagrangian_penalty"] = lagrangian_penalty
    
    return batch


# ============================================================================
# 数据验证函数
# ============================================================================

def validate_batch(batch: Dict[str, Any]) -> bool:
    """
    验证批次数据的完整性和有效性
    
    Args:
        batch: 样本批次
        
    Returns:
        is_valid: 数据是否有效
    """
    # 检查必需的字段
    required_fields = [
        "obs", "actions", "rewards", "dones", "new_obs"
    ]
    
    for field in required_fields:
        if field not in batch:
            logger.error(f"❌ 批次缺少必需字段: {field}")
            return False
    
    # 检查数据形状
    batch_size = len(batch["obs"])
    
    for field in ["obs", "actions", "rewards", "dones", "new_obs"]:
        if len(batch[field]) != batch_size:
            logger.error(f"❌ 字段 {field} 的长度不匹配: {len(batch[field])} != {batch_size}")
            return False
    
    # 检查NaN和Inf
    for field in ["obs", "actions", "rewards"]:
        data = batch[field]
        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                logger.error(f"❌ 字段 {field} 包含NaN值")
                return False
            if np.isinf(data).any():
                logger.error(f"❌ 字段 {field} 包含Inf值")
                return False
    
    return True


# ============================================================================
# 混合精度训练配置
# ============================================================================

def configure_amp(config: PPOConfig, use_amp: bool) -> PPOConfig:
    """
    配置混合精度训练（AMP）
    
    Args:
        config: PPO配置对象
        use_amp: 是否启用AMP
        
    Returns:
        config: 配置后的PPO配置
    """
    if use_amp:
        # 启用混合精度训练
        config.training(
            use_amp=True,
            amp_dtype="float16"  # 使用float16进行加速
        )
        logger.info("✅ 混合精度训练（AMP）已启用")
    else:
        config.training(use_amp=False)
        logger.info("ℹ️  混合精度训练（AMP）已禁用")
    
    return config


# ============================================================================
# 梯度裁剪配置
# ============================================================================

def configure_gradient_clipping(config: PPOConfig, grad_clip: float) -> PPOConfig:
    """
    配置梯度裁剪
    
    Args:
        config: PPO配置对象
        grad_clip: 梯度裁剪阈值
        
    Returns:
        config: 配置后的PPO配置
    """
    config.training(
        grad_clip=grad_clip
    )
    logger.info(f"✅ 梯度裁剪阈值: {grad_clip}")
    
    return config


# ============================================================================
# Ray配置构建
# ============================================================================

def build_ray_config(user_config: Dict[str, Any]) -> PPOConfig:
    """
    构建Ray RLlib配置
    
    Args:
        user_config: 用户配置字典
        
    Returns:
        config: PPOConfig对象
    """
    # 创建基础PPO配置
    config = PPOConfig()
    
    # ==================== 环境配置 ====================
    config.environment(
        env=SUMOGymEnv,
        env_config={
            "sumo_cfg_path": user_config["sumo_cfg_path"],
            "use_libsumo": user_config["use_libsumo"],
            "batch_subscribe": user_config["batch_subscribe"],
            "max_steps": user_config["max_steps"],
            "use_gui": user_config["use_gui"],
            "model_config": user_config["model"]["custom_model_config"]
        }
    )
    
    # ==================== 框架配置 ====================
    config.framework(user_config["framework"])
    
    # ==================== 并行配置 ====================
    config.resources(
        num_gpus=user_config["num_gpus"],
        num_cpus_per_worker=user_config["num_cpus_per_worker"],
    )
    config.rollouts(
        num_rollout_workers=user_config["num_workers"],
        num_envs_per_worker=user_config["num_envs_per_worker"],
    )
    
    # ==================== 训练配置 ====================
    config.training(
        train_batch_size=user_config["train_batch_size"],
        rollout_fragment_length=user_config["rollout_fragment_length"],
        sgd_minibatch_size=user_config["sgd_minibatch_size"],
        num_sgd_iter=user_config["num_sgd_iter"],
        lr=user_config["lr"],
        gamma=user_config["gamma"],
        lambda_=user_config["lambda_"],
        clip_param=user_config["clip_param"],
        vf_loss_coeff=user_config["vf_loss_coeff"],
        entropy_coeff=user_config["entropy_coeff"],
        kl_coeff=user_config["kl_coeff"],
        kl_target=user_config["kl_target"],
    )
    
    # ==================== 模型配置 ====================
    config.model(
        custom_model=user_config["model"]["custom_model"],
        custom_model_config=user_config["model"]["custom_model_config"]
    )
    
    # ==================== 梯度裁剪配置 ====================
    config = configure_gradient_clipping(config, user_config["grad_clip"])
    
    # ==================== 混合精度训练配置 ====================
    config = configure_amp(config, user_config["use_amp"])
    
    # ==================== 检查点配置 ====================
    config.checkpointing(
        checkpoint_frequency=user_config["checkpoint_freq"],
        checkpoint_at_end=user_config["checkpoint_at_end"],
        checkpoint_score_attribute=user_config["checkpoint_score_attr"],
        keep_checkpoints_num=user_config["keep_checkpoints_num"],
    )
    
    # ==================== 约束优化配置 ====================
    # 将约束优化参数添加到配置中（供ConstrainedPPOTrainer使用）
    config.cost_limit = user_config["cost_limit"]
    config.lambda_lr = user_config["lambda_lr"]
    config.lambda_init = user_config["lambda_init"]
    config.alpha = user_config["alpha"]
    config.beta = user_config["beta"]
    
    # ==================== 日志配置 ====================
    config.logging(
        level=user_config["log_level"],
    )
    
    logger.info("✅ Ray RLlib配置构建完成")
    
    return config


# ============================================================================
# 主训练函数
# ============================================================================

def train(config: Dict[str, Any]) -> None:
    """
    主训练函数
    
    Args:
        config: 训练配置字典
    """
    # 打印配置信息
    print("\n" + "=" * 80)
    print("🚀 Ray RLlib 训练配置")
    print("=" * 80)
    print(f"📊 环境配置:")
    print(f"   - SUMO配置文件: {config['sumo_cfg_path']}")
    print(f"   - LIBSUMO_AS_TRACI: {config['use_libsumo']}")
    print(f"   - 批量订阅: {config['batch_subscribe']}")
    print(f"   - 最大步数: {config['max_steps']}")
    print(f"🖥️  计算资源配置:")
    print(f"   - Workers数量: {config['num_workers']}")
    print(f"   - GPU数量: {config['num_gpus']}")
    print(f"   - 每Worker CPU数: {config['num_cpus_per_worker']}")
    print(f"   - 每Worker环境数: {config['num_envs_per_worker']}")
    print(f"📈 训练配置:")
    print(f"   - 训练批次大小: {config['train_batch_size']}")
    print(f"   - Rollout片段长度: {config['rollout_fragment_length']}")
    print(f"   - SGD小批次大小: {config['sgd_minibatch_size']}")
    print(f"   - SGD迭代次数: {config['num_sgd_iter']}")
    print(f"   - 学习率: {config['lr']}")
    print(f"🔐 约束优化配置:")
    print(f"   - 成本限制: {config['cost_limit']}")
    print(f"   - 拉格朗日乘子学习率: {config['lambda_lr']}")
    print(f"   - 拉格朗日乘子初始值: {config['lambda_init']}")
    print(f"🛡️  优化配置:")
    print(f"   - 梯度裁剪: {config['grad_clip']}")
    print(f"   - 混合精度训练: {config['use_amp']}")
    print(f"💾 检查点配置:")
    print(f"   - 保存频率: {config['checkpoint_freq']}")
    print(f"   - 保留数量: {config['keep_checkpoints_num']}")
    print("=" * 80)
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            num_gpus=config["num_gpus"],
            num_cpus=config["num_workers"] * config["num_cpus_per_worker"] + 2,
            log_to_driver=config.get("log_level", "INFO") == "INFO"
        )
        logger.info("✅ Ray已初始化")
    
    # 注册自定义模型
    register_traffic_controller_model()
    
    # 注册环境
    tune.register_env("sumo_gym_env", env_creator)
    
    # 构建Ray配置
    ray_config = build_ray_config(config)
    
    # 创建训练回调
    callback = TrainingCallback(config)
    
    # 创建训练器
    if config.get("restore"):
        logger.info(f"🔄 从检查点恢复: {config['restore']}")
        trainer = ConstrainedPPOTrainer(
            config=ray_config.to_dict(),
            logger_creator=lambda config: None
        )
        trainer.restore(config["restore"])
    else:
        trainer = ConstrainedPPOTrainer(
            config=ray_config.to_dict(),
            logger_creator=lambda config: None
        )
    
    logger.info("✅ 训练器已创建")
    
    # 训练循环
    stop_criteria = config["stop"]
    max_iterations = stop_criteria.get("training_iteration", 1000)
    target_reward = stop_criteria.get("episode_reward_mean", None)
    
    print("\n" + "=" * 80)
    print("🎯 开始训练")
    print("=" * 80)
    
    try:
        for iteration in range(max_iterations):
            # 训练一个迭代
            result = trainer.train()
            
            # 调用回调函数
            callback.on_train_result(result)
            
            # 检查停止条件
            if target_reward is not None:
                current_reward = result.get("episode_reward_mean", 0.0)
                if current_reward >= target_reward:
                    print(f"\n🎉 达到目标奖励: {current_reward:.4f} >= {target_reward:.4f}")
                    break
            
            # 检查点保存
            if (iteration + 1) % config["checkpoint_freq"] == 0:
                checkpoint_path = trainer.save()
                callback.on_checkpoint({"checkpoint": checkpoint_path})
    
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    
    finally:
        # 训练结束时保存检查点
        if config["checkpoint_at_end"]:
            checkpoint_path = trainer.save()
            logger.info(f"💾 最终检查点已保存: {checkpoint_path}")
        
        # 关闭回调
        callback.close()
        
        # 关闭训练器
        trainer.stop()
        
        # 关闭Ray
        ray.shutdown()
        logger.info("✅ Ray已关闭")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        args: 命令行参数
    """
    parser = argparse.ArgumentParser(
        description="Ray RLlib 训练脚本 - SUMO交通控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 使用默认配置训练
    python ray_train.py
    
    # 从配置文件加载配置
    python ray_train.py --config config.json
    
    # 从检查点恢复训练
    python ray_train.py --restore /path/to/checkpoint
    
    # 自定义训练参数
    python ray_train.py --num_workers 8 --num_gpus 2 --train_batch_size 8000
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（JSON格式）"
    )
    
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="检查点路径，用于恢复训练"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="RolloutWorkers数量"
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="GPU数量"
    )
    
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="训练批次大小"
    )
    
    parser.add_argument(
        "--rollout_fragment_length",
        type=int,
        default=None,
        help="Rollout片段长度"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="最大训练迭代次数"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志目录"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="实验名称"
    )
    
    parser.add_argument(
        "--use_libsumo",
        action="store_true",
        help="启用LIBSUMO_AS_TRACI加速"
    )
    
    parser.add_argument(
        "--use_gui",
        action="store_true",
        help="启用SUMO GUI"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 获取默认配置
    config = get_default_config()
    
    # 从配置文件加载配置
    if args.config:
        user_config = load_config_from_file(args.config)
        config = merge_configs(config, user_config)
    
    # 覆盖命令行参数
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.num_gpus is not None:
        config["num_gpus"] = args.num_gpus
    if args.train_batch_size is not None:
        config["train_batch_size"] = args.train_batch_size
    if args.rollout_fragment_length is not None:
        config["rollout_fragment_length"] = args.rollout_fragment_length
    if args.max_iterations is not None:
        config["stop"]["training_iteration"] = args.max_iterations
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    if args.use_libsumo:
        config["use_libsumo"] = True
    if args.use_gui:
        config["use_gui"] = True
    
    # 设置恢复路径
    if args.restore:
        config["restore"] = args.restore
    
    # 开始训练
    train(config)


# ============================================================================
# 脚本入口
# ============================================================================

if __name__ == "__main__":
    main()
