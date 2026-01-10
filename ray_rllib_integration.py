"""
Ray RLlib 集成模块
利用 Ray RLlib 的多进程并行能力，同时跑多个 SUMO 实例
实现分布式强化学习训练
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional
import os
import time

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.tf.tf_modelv2 import TFModelV2
    from ray.rllib.utils import try_import_tf
    from ray.tune.registry import register_env
    tf = try_import_tf()
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  Ray RLlib 未安装，分布式训练功能不可用")
    print("   安装命令: pip install ray[rllib]")


from sumo_rl_env import SUMORLEnvironment
from neural_traffic_controller import TrafficController


class SUMORayEnvironment(gym.Env):
    """
    Ray RLlib 兼容的 SUMO 环境
    实现 Gymnasium 标准接口
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 Ray SUMO 环境
        
        Args:
            config: 环境配置字典
        """
        if config is None:
            config = {}
        
        self.config = config
        
        # SUMO 配置
        self.sumo_cfg_path = config.get('sumo_cfg_path', '仿真环境-初赛/sumo.sumocfg')
        self.use_gui = config.get('use_gui', False)
        self.max_steps = config.get('max_steps', 3600)
        self.seed_val = config.get('seed', None)
        
        # 初始化 SUMO 环境
        self.sumo_env = SUMORLEnvironment(
            sumo_cfg_path=self.sumo_cfg_path,
            use_gui=self.use_gui,
            max_steps=self.max_steps,
            seed=self.seed_val
        )
        
        # 动作空间
        # 动作是字典形式，但在 RLlib 中需要展平
        # 这里我们使用 MultiDiscrete 或者 Box 空间
        # 简化版：使用连续动作空间 [加速度, 换道概率] * top_k
        self.top_k = config.get('top_k', 5)
        self.action_dim = 2 * self.top_k  # [accel1, lane1, accel2, lane2, ...]
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # 观测空间 - 动态计算
        # 包含：节点特征、边特征、全局指标
        self.observation_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(None, 9),  # 动态车辆数
                dtype=np.float32
            ),
            'edge_indices': gym.spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(2, None),  # 动态边数
                dtype=np.int32
            ),
            'edge_features': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(None, 4),  # 动态边数
                dtype=np.float32
            ),
            'global_metrics': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(16,),
                dtype=np.float32
            ),
            'is_icv': gym.spaces.Box(
                low=0,
                high=1,
                shape=(None,),  # 动态车辆数
                dtype=np.int8
            )
        })
        
        # 环境状态
        self.current_step = 0
        self.episode_reward = 0.0
        self.vehicle_ids = []
        
        print(f"✅ Ray SUMO 环境初始化完成")
        print(f"   动作空间维度: {self.action_dim}")
        print(f"   Top-K: {self.top_k}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        if seed is not None:
            self.seed_val = seed
        
        # 重置 SUMO 环境
        observation = self.sumo_env.reset()
        
        # 转换为 Gym 格式
        gym_observation = self._convert_to_gym_observation(observation)
        
        # 重置状态
        self.current_step = 0
        self.episode_reward = 0.0
        self.vehicle_ids = observation['vehicle_ids']
        
        # 额外信息
        info = {
            'vehicle_count': len(self.vehicle_ids),
            'step': self.current_step
        }
        
        return gym_observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 动作数组 [accel1, lane1, accel2, lane2, ...]
            
        Returns:
            observation: 观测
            reward: 奖励
            terminated: 是否终止（正常结束）
            truncated: 是否截断（超时等）
            info: 额外信息
        """
        # 重塑动作
        action = action.reshape(-1, 2)  # [K, 2]
        
        # 构建 SUMO 环境期望的动作格式
        # 选择前 top_k 辆车（简化版）
        selected_vehicle_ids = self.vehicle_ids[:self.top_k] if len(self.vehicle_ids) >= self.top_k else self.vehicle_ids
        
        safe_actions = torch.tensor(action[:len(selected_vehicle_ids)], dtype=torch.float32)
        
        sumo_action = {
            'selected_vehicle_ids': selected_vehicle_ids,
            'safe_actions': safe_actions
        }
        
        # 执行一步
        observation, reward, done, info = self.sumo_env.step(sumo_action)
        
        # 转换观测
        gym_observation = self._convert_to_gym_observation(observation)
        
        # 更新状态
        self.current_step += 1
        self.episode_reward += reward
        self.vehicle_ids = observation['vehicle_ids']
        
        # 判断终止和截断
        terminated = done
        truncated = self.current_step >= self.max_steps
        
        # 更新额外信息
        info.update({
            'episode_reward': self.episode_reward,
            'step': self.current_step,
            'vehicle_count': len(self.vehicle_ids)
        })
        
        return gym_observation, reward, terminated, truncated, info
    
    def _convert_to_gym_observation(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        将 SUMO 观测转换为 Gym 格式
        
        Args:
            observation: SUMO 观测
            
        Returns:
            gym_observation: Gym 格式观测
        """
        vehicle_data = observation['vehicle_data']
        vehicle_ids = observation['vehicle_ids']
        
        if not vehicle_ids:
            return {
                'node_features': np.zeros((0, 9), dtype=np.float32),
                'edge_indices': np.zeros((2, 0), dtype=np.int32),
                'edge_features': np.zeros((0, 4), dtype=np.float32),
                'global_metrics': np.array(observation['global_metrics'], dtype=np.float32),
                'is_icv': np.zeros((0,), dtype=np.int8)
            }
        
        # 构建节点特征
        node_features = []
        is_icv_list = []
        
        for veh_id in vehicle_ids:
            vehicle = vehicle_data[veh_id]
            features = [
                vehicle.get('position', 0.0),
                vehicle.get('speed', 0.0),
                vehicle.get('acceleration', 0.0),
                vehicle.get('lane_index', 0),
                1000.0,  # 剩余距离（简化）
                0.5,  # 完成率（简化）
                1.0 if vehicle.get('is_icv', False) else 0.0,
                self.current_step * 0.1,
                0.1
            ]
            node_features.append(features)
            is_icv_list.append(1 if vehicle.get('is_icv', False) else 0)
        
        # 构建边特征（简化版）
        edge_indices = []
        edge_features = []
        
        for i, veh_id_i in enumerate(vehicle_ids):
            for j, veh_id_j in enumerate(vehicle_ids):
                if i == j:
                    continue
                
                pos_i = vehicle_data[veh_id_i].get('position', 0.0)
                pos_j = vehicle_data[veh_id_j].get('position', 0.0)
                speed_i = vehicle_data[veh_id_i].get('speed', 0.0)
                speed_j = vehicle_data[veh_id_j].get('speed', 0.0)
                
                distance = abs(pos_i - pos_j)
                if distance < 50:
                    edge_indices.append([i, j])
                    
                    rel_distance = distance
                    rel_speed = abs(speed_i - speed_j)
                    
                    ttc = rel_distance / max(rel_speed, 0.1) if rel_speed > 0 else 100
                    thw = rel_distance / max(speed_i, 0.1) if speed_i > 0 else 100
                    
                    edge_features.append([rel_distance, rel_speed, min(ttc, 10), min(thw, 10)])
        
        # 转换为 numpy 数组
        gym_observation = {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_indices': np.array(edge_indices, dtype=np.int32).T if edge_indices else np.zeros((2, 0), dtype=np.int32),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 4), dtype=np.float32),
            'global_metrics': np.array(observation['global_metrics'], dtype=np.float32),
            'is_icv': np.array(is_icv_list, dtype=np.int8)
        }
        
        return gym_observation
    
    def close(self):
        """关闭环境"""
        self.sumo_env.close()
    
    def render(self, mode: str = 'human'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
        """
        if mode == 'human' and self.use_gui:
            # SUMO GUI 已经在运行
            pass
        elif mode == 'rgb_array':
            # 返回截图（需要额外实现）
            pass
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """获取 episode 统计信息"""
        return {
            'total_steps': self.current_step,
            'total_reward': self.episode_reward,
            'avg_reward': self.episode_reward / max(self.current_step, 1),
            'vehicle_count': len(self.vehicle_ids)
        }


def create_ray_sumo_env(config: Dict[str, Any]) -> SUMORayEnvironment:
    """
    创建 Ray SUMO 环境的工厂函数
    
    Args:
        config: 环境配置
        
    Returns:
        Ray SUMO 环境实例
    """
    return SUMORayEnvironment(config)


class RayRLlibTrainer:
    """
    Ray RLlib 训练器
    实现分布式强化学习训练
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Ray RLlib 训练器
        
        Args:
            config: 训练配置
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray RLlib 未安装，无法使用分布式训练")
        
        self.config = config
        
        # 初始化 Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=config.get('num_cpus', 8),
                num_gpus=config.get('num_gpus', 1),
                ignore_reinit_error=True
            )
            print(f"✅ Ray 初始化完成")
            print(f"   CPUs: {config.get('num_cpus', 8)}")
            print(f"   GPUs: {config.get('num_gpus', 1)}")
        
        # 注册环境
        register_env("sumo_ray", create_ray_sumo_env)
        
        # 配置算法
        self.algorithm = PPO(config=self._get_rllib_config())
        
        print("✅ Ray RLlib 训练器初始化完成")
    
    def _get_rllib_config(self) -> Dict[str, Any]:
        """
        获取 RLlib 配置
        
        Returns:
            RLlib 配置字典
        """
        config = {
            # 环境配置
            "env": "sumo_ray",
            "env_config": {
                "sumo_cfg_path": self.config.get('sumo_cfg_path', '仿真环境-初赛/sumo.sumocfg'),
                "use_gui": self.config.get('use_gui', False),
                "max_steps": self.config.get('max_steps', 3600),
                "seed": self.config.get('seed', None),
                "top_k": self.config.get('top_k', 5)
            },
            
            # 并行配置
            "num_workers": self.config.get('num_workers', 4),  # 并行环境数
            "num_envs_per_worker": self.config.get('num_envs_per_worker', 2),  # 每个worker的环境数
            "train_batch_size": self.config.get('train_batch_size', 4000),
            "sgd_minibatch_size": self.config.get('sgd_minibatch_size', 128),
            
            # PPO 配置
            "lr": self.config.get('learning_rate', 3e-4),
            "gamma": self.config.get('gamma', 0.99),
            "lambda": self.config.get('lambda', 0.95),
            "clip_param": self.config.get('clip_param', 0.2),
            "vf_clip_param": self.config.get('vf_clip_param', 10.0),
            "entropy_coeff": self.config.get('entropy_coeff', 0.01),
            "vf_loss_coeff": self.config.get('vf_loss_coeff', 0.5),
            
            # 网络配置
            "model": {
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            
            # 训练配置
            "num_sgd_iter": self.config.get('num_sgd_iter', 10),
            "framework": "torch",
            
            # 资源配置
            "num_gpus": self.config.get('num_gpus', 1),
            "num_cpus_per_worker": self.config.get('num_cpus_per_worker', 1),
        }
        
        return config
    
    def train(self, num_iterations: int = 100):
        """
        训练模型
        
        Args:
            num_iterations: 训练迭代次数
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始 Ray RLlib 分布式训练")
        print(f"{'='*60}")
        print(f"迭代次数: {num_iterations}")
        print(f"并行环境数: {self.config.get('num_workers', 4) * self.config.get('num_envs_per_worker', 2)}")
        print(f"{'='*60}\n")
        
        for i in range(num_iterations):
            # 执行一次训练迭代
            result = self.algorithm.train()
            
            # 打印统计信息
            print(f"\n迭代 {i+1}/{num_iterations}:")
            print(f"  平均奖励: {result['episode_reward_mean']:.4f}")
            print(f"  最小奖励: {result['episode_reward_min']:.4f}")
            print(f"  最大奖励: {result['episode_reward_max']:.4f}")
            print(f"  Episode 长度: {result['episode_len_mean']:.2f}")
            print(f"  学习率: {result['info']['learner']['cur_lr']:.6f}")
            print(f"  熵: {result['info']['learner']['entropy']:.4f}")
            
            # 定期保存检查点
            if (i + 1) % 10 == 0:
                checkpoint_path = self.algorithm.save()
                print(f"💾 检查点已保存: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print("✅ 训练完成!")
        print(f"{'='*60}")
        
        # 保存最终模型
        final_checkpoint = self.algorithm.save()
        print(f"💾 最终模型已保存: {final_checkpoint}")
        
        return final_checkpoint
    
    def evaluate(self, num_episodes: int = 10):
        """
        评估模型
        
        Args:
            num_episodes: 评估 episode 数
        """
        print(f"\n{'='*60}")
        print(f"📊 开始评估")
        print(f"{'='*60}")
        print(f"Episode 数: {num_episodes}")
        print(f"{'='*60}\n")
        
        total_rewards = []
        total_steps = []
        
        for episode in range(num_episodes):
            # 创建评估环境
            env = SUMORayEnvironment(self.config)
            obs, info = env.reset()
            
            episode_reward = 0.0
            done = False
            truncated = False
            steps = 0
            
            while not (done or truncated):
                # 获取动作
                action = self.algorithm.compute_single_action(obs)
                
                # 执行一步
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"奖励={episode_reward:.2f}, 步数={steps}")
            
            env.close()
        
        # 统计结果
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_steps = np.mean(total_steps)
        
        print(f"\n{'='*60}")
        print("📊 评估结果")
        print(f"{'='*60}")
        print(f"平均奖励: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"{'='*60}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_steps': avg_steps,
            'all_rewards': total_rewards
        }
    
    def close(self):
        """关闭训练器"""
        self.algorithm.stop()
        if ray.is_initialized():
            ray.shutdown()
        print("✅ Ray RLlib 训练器已关闭")


def main():
    """主函数 - 演示 Ray RLlib 集成"""
    if not RAY_AVAILABLE:
        print("❌ Ray RLlib 未安装，无法运行分布式训练")
        print("   安装命令: pip install ray[rllib]")
        return
    
    # 训练配置
    config = {
        'sumo_cfg_path': '仿真环境-初赛/sumo.sumocfg',
        'use_gui': False,
        'max_steps': 3600,
        'seed': 42,
        'top_k': 5,
        
        # Ray 配置
        'num_cpus': 8,
        'num_gpus': 1,
        'num_workers': 4,
        'num_envs_per_worker': 2,
        
        # 训练配置
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_param': 0.2,
        'train_batch_size': 4000,
        'sgd_minibatch_size': 128,
        'num_sgd_iter': 10,
        
        # 评估配置
        'num_evaluation_episodes': 10
    }
    
    # 创建训练器
    trainer = RayRLlibTrainer(config)
    
    # 训练
    try:
        checkpoint_path = trainer.train(num_iterations=100)
        
        # 评估
        eval_results = trainer.evaluate(num_episodes=10)
        
        print(f"\n✅ 训练和评估完成!")
        print(f"   最终模型: {checkpoint_path}")
        print(f"   评估奖励: {eval_results['avg_reward']:.4f}")
        
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
