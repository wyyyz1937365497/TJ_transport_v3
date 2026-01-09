import os
import json
import ray
from ray import tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from sumo_env_adapter import SumoEnvironmentAdapter
from neural_traffic_controller import TrafficController


def env_creator(config):
    """环境创建器，用于注册到Ray中"""
    return SumoEnvironmentAdapter(
        sumo_cfg_path=config["sumo_cfg_path"],
        max_steps=config.get("max_steps", 3600),
        step_interval=config.get("step_interval", 5)
    )


def train_with_ray():
    """使用Ray RLlib进行训练"""
    # 加载配置
    with open('train_config.json', 'r') as f:
        config = json.load(f)
    
    # 初始化Ray
    ray.init(num_cpus=config["parallel"]["num_workers"] + 2)  # 为Ray额外留出一些CPU
    
    # 注册环境
    register_env("sumo_env", env_creator)
    
    # 创建模型配置
    model_config = config["model"].copy()
    model_config["device"] = "cpu"  # 在并行环境中使用CPU
    
    # 配置算法
    algo_config = (
        PPOConfig()
        .environment(
            env="sumo_env",
            env_config={
                "sumo_cfg_path": "仿真环境-初赛/sumo.sumocfg",  # SUMO配置文件路径
                "max_steps": 3600,
                "step_interval": 5
            }
        )
        .rollouts(
            num_rollout_workers=config["parallel"]["num_workers"],
            num_envs_per_worker=config["parallel"]["env_per_worker"],
            # batch_mode="complete_episodes"
        )
        .training(
            gamma=0.99,
            lr=0.0003,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            vf_loss_coeff=0.1,
            entropy_coeff=0.05
        )
        .framework(framework="torch")
        .resources(num_gpus=0)  # 使用CPU进行训练
    )
    
    # 创建算法实例
    algo = algo_config.build()
    
    # 开始训练
    print("开始使用Ray RLlib进行分布式训练...")
    print(f"使用 {config['parallel']['num_workers']} 个工作进程，每个工作进程运行 {config['parallel']['env_per_worker']} 个环境")
    
    # 训练循环
    for i in range(100):  # 训练100次迭代
        result = algo.train()
        
        print(f"Iteration {i}: reward={result['episode_reward_mean']}, "
              f"episode_len={result['episode_len_mean']}")
        
        # 每10次迭代保存一次模型
        if i % 10 == 0:
            checkpoint_dir = algo.save("checkpoints/ray_checkpoint_{:06d}".format(i))
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    # 保存最终模型
    final_checkpoint_dir = algo.save("checkpoints/final_ray_model")
    print(f"Final model saved to {final_checkpoint_dir}")
    
    # 清理资源
    ray.shutdown()
    
    print("Ray训练完成！")


if __name__ == "__main__":
    train_with_ray()