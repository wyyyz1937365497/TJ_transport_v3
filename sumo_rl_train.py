import os
import json
import ray
from ray import tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from sumo_env_adapter import SumoEnvironmentAdapter


def env_creator(env_config: EnvContext):
    """环境创建器，用于注册到Ray中"""
    # 从配置中获取参数
    sumo_cfg_path = env_config.get("sumo_cfg_path", "仿真环境-初赛/sumo.sumocfg")
    max_steps = env_config.get("max_steps", 3600)
    step_interval = env_config.get("step_interval", 5)
    worker_index = env_config.worker_index if hasattr(env_config, 'worker_index') else 0
    vector_index = env_config.vector_index if hasattr(env_config, 'vector_index') else 0
    
    # 计算唯一的端口号以避免冲突
    base_port = env_config.get("base_port", 8813)
    port_offset = worker_index * env_config.get("envs_per_worker", 2) + vector_index
    port = base_port + port_offset
    
    # 创建环境实例
    return SumoEnvironmentAdapter(
        sumo_cfg_path=sumo_cfg_path,
        max_steps=max_steps,
        step_interval=step_interval,
        port=port
    )


def train_with_sumo_rl():
    """使用Ray RLlib和SUMO-RL进行多实例训练"""
    # 加载配置
    with open('train_config.json', 'r') as f:
        config = json.load(f)
    
    # 初始化Ray
    num_workers = config["parallel"]["num_workers"]
    ray.init(num_cpus=num_workers + 2)  # 为Ray额外留出一些CPU
    
    # 注册环境
    register_env("sumo_env", env_creator)
    
    # 配置算法
    algo_config = (
        PPOConfig()
        .environment(
            env="sumo_env",
            env_config={
                "sumo_cfg_path": "仿真环境-初赛/sumo.sumocfg",  # SUMO配置文件路径
                "max_steps": 1000,  # 减少单次episode长度以加快训练
                "step_interval": 5,
                "base_port": config["parallel"]["base_port"],
                "envs_per_worker": config["parallel"]["env_per_worker"]
            }
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=config["parallel"]["env_per_worker"],
            rollout_fragment_length=200
        )
        .training(
            gamma=0.99,
            lr=tune.grid_search([0.0003, 0.001]),
            train_batch_size=4000,
            sgd_minibatch_size=128,
            vf_loss_coeff=0.1,
            entropy_coeff=0.05
        )
        .framework(framework="torch")
        .resources(num_gpus=2)  # 使用CPU进行训练
    )
    
    # 创建算法实例
    algo = algo_config.build()
    
    # 开始训练
    print("开始使用Ray RLlib和SUMO-RL进行多实例分布式训练...")
    print(f"使用 {num_workers} 个工作进程，每个工作进程运行 {config['parallel']['env_per_worker']} 个环境")
    print(f"基础端口: {config['parallel']['base_port']}")
    
    # 训练循环
    for i in range(50):  # 训练50次迭代
        result = algo.train()
        
        print(f"Iteration {i}: reward={result['episode_reward_mean']:.2f}, "
              f"episode_len={result['episode_len_mean']:.2f}, "
              f"policy_loss={result.get('learner', {}).get('default_policy', {}).get('learner_loss', 'N/A')}")
        
        # 每10次迭代保存一次模型
        if i % 10 == 0:
            checkpoint_dir = algo.save(f"checkpoints/ray_sumo_rl_checkpoint_{i:03d}")
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    # 保存最终模型
    final_checkpoint_dir = algo.save("checkpoints/final_sumo_rl_model")
    print(f"Final model saved to {final_checkpoint_dir}")
    
    # 清理资源
    ray.shutdown()
    
    print("SUMO-RL多实例训练完成！")


if __name__ == "__main__":
    train_with_sumo_rl()