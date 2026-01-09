import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks
import os
import json
import time
import argparse

def train_traffic_agent():
    """
    三阶段训练主函数
    """
    
    # 1. 初始化Ray
    ray.init(
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        object_store_memory=2e9,  # 2GB
        num_cpus=16,  # 根据机器配置调整
        num_gpus=1
    )
    
    # 2. 注册自定义模型和环境
    from ray.rllib.env import register_env
    from src.models.traffic_torch_model import TrafficTorchModel
    from src.env.custom_sumo_env import CustomSumoEnv
    
    register_env("CustomSumoEnv", lambda cfg: CustomSumoEnv(cfg))
    ModelCatalog.register_custom_model("traffic_torch_model", TrafficTorchModel)
    
    # 3. 三阶段训练
    training_phases = [
        ("phase1_world_model", train_world_model_phase1),
        ("phase2_safe_rl", configure_phase2_training),
        ("phase3_constrained", configure_phase3_training)
    ]
    
    for phase_name, phase_config_fn in training_phases:
        print(f"Starting training phase: {phase_name}")
        
        if "phase1" in phase_name:
            # 阶段1：世界模型预训练（非RL）
            world_model = phase_config_fn({
                "buffer_size": 100000,
                "batch_size": 256,
                "epochs": 50,
                "lr": 1e-4,
                "pretrain_episodes": 100,
                "state_dim": 8,
                "action_dim": 1
            })
            
            # 保存预训练模型
            import torch
            torch.save(world_model.state_dict(), f"models/world_model_phase1.pt")
            print(f"Phase 1 completed. Model saved to models/world_model_phase1.pt")
            
        else:
            # 阶段2和3：RL训练
            config = phase_config_fn()
            
            # 设置训练名称
            config["name"] = f"TrafficAgent_{phase_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            config["local_dir"] = "./ray_results"
            
            # 启动训练
            analysis = tune.run(
                "PPO",
                config=config,
                stop={
                    "training_iteration": 500 if "phase2" in phase_name else 1000,
                    "episode_reward_mean": 1000.0  # 目标奖励
                },
                checkpoint_at_end=True,
                verbose=2,
                progress_reporter=tune.CLIReporter(
                    metric_columns=["episode_reward_mean", "episode_cost_mean", "episodes_total"]
                )
            )
            
            # 保存最佳检查点
            best_checkpoint = analysis.get_best_checkpoint(
                metric="episode_reward_mean",
                mode="max"
            )
            print(f"Phase {phase_name} completed. Best checkpoint: {best_checkpoint}")
    
    # 4. 关闭Ray
    ray.shutdown()
    print("All training phases completed successfully!")


def train_world_model_phase1(config):
    """
    阶段1：基础动力学学习
    目标：准确预测下一时刻车辆状态
    """
    from src.dataset.trajectory_buffer import TrajectoryBuffer
    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as F
    
    # 1. 创建数据集
    buffer = TrajectoryBuffer(
        max_size=config.get("buffer_size", 100000),
        state_dim=config.get("state_dim", 8),
        action_dim=config.get("action_dim", 1)
    )
    
    # 2. 数据收集（使用随机策略或IDM）
    from src.env.custom_sumo_env import CustomSumoEnv
    env = CustomSumoEnv(config)
    for episode in range(config.get("pretrain_episodes", 100)):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 随机动作或IDM动作
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 存储(s, a, s')三元组
            buffer.add(
                state=obs["gnn_embedding"],
                action=action,
                next_state=next_obs["gnn_embedding"]
            )
            
            obs = next_obs
    
    # 3. 创建数据加载器
    dataloader = DataLoader(
        buffer,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=4
    )
    
    # 4. 训练世界模型
    from src.models.progressive_world_model import ProgressiveWorldModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = ProgressiveWorldModel(config)
    world_model.to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=config.get("lr", 1e-4))
    
    for epoch in range(config.get("epochs", 50)):
        for batch in dataloader:
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            next_states = batch["next_state"].to(device)
            
            # 预测下一状态
            pred_next_states = world_model(states, actions)
            
            # 计算损失
            loss = F.mse_loss(pred_next_states, next_states)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return world_model


def configure_phase2_training():
    """配置阶段2：带安全屏障的RL训练"""
    
    config = {
        # 环境配置
        "env": "CustomSumoEnv",
        "env_config": {
            "sumo_cfg_path": "official.sumocfg",
            "net_file_path": "official.net.xml",
            "route_file_path": "official.rou.xml",
            "max_steps": 3600,
            "control_ratio": 0.25,
            "num_controlled": 5,
            "safety_shield_enabled": True,  # 启用安全屏障
            "ttc_threshold": 2.0,
            "thw_threshold": 1.5
        },
        
        # 模型配置
        "model": {
            "custom_model": "traffic_torch_model",
            "custom_model_config": {
                "gnn_output_dim": 256,
                "world_hidden_dim": 128,
                "future_steps": 5,
                "actor_hidden_dim": 128,
                "cost_limit": 100.0,  # 宽松的成本限制
                "lambda_lr": 0.001
            },
            "vf_share_layers": False
        },
        
        # 算法配置
        "framework": "torch",
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 1.0,
        
        # 训练配置
        "lr": 1e-4,
        "train_batch_size": 4000,
        "rollout_fragment_length": 500,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 10,
        
        # 并行配置
        "num_workers": 8,
        "num_gpus": 1,
        "num_envs_per_worker": 1,
        "remote_worker_envs": True,
        
        # 回放配置
        "batch_mode": "truncate_episodes",
        
        # 评估配置
        "evaluation_interval": 10,
        "evaluation_num_episodes": 5,
        
        # 检查点配置
        "checkpoint_freq": 10,
        "keep_checkpoints_num": 5,
        
        # 自定义策略
        # "policy_class": CustomPPOTorchPolicy  # 在实际实现中添加
    }
    
    return config


def configure_phase3_training():
    """配置阶段3：约束优化训练"""
    
    config = configure_phase2_training()  # 继承阶段2配置
    
    # 增强约束优化
    config["model"]["custom_model_config"].update({
        "cost_limit": 10.0,  # 严格的成本限制
        "lambda_lr": 0.01,   # 更快的乘子更新
        "cost_sensitive_training": True
    })
    
    # 调整训练参数
    config.update({
        "lr": 5e-5,  # 降低学习率
        "entropy_coeff": 0.005,  # 降低熵系数
        "kl_coeff": 0.2,  # 增加KL约束
        "kl_target": 0.01,
        
        # 增加训练强度
        "train_batch_size": 8000,
        "num_sgd_iter": 15,
        
        # 启用课程学习
        "curriculum_enabled": True,
        "curriculum_config": {
            "cost_limit_schedule": [
                {"episode": 0, "limit": 100.0},
                {"episode": 100, "limit": 50.0},
                {"episode": 200, "limit": 25.0},
                {"episode": 300, "limit": 10.0}
            ]
        }
    })
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traffic Control Agent")
    parser.add_argument('--phase', type=str, default='all', choices=['phase1', 'phase2', 'phase3', 'all'],
                        help='Which training phase to run')
    args = parser.parse_args()
    
    train_traffic_agent()