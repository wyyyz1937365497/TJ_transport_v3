"""
Ray推理脚本 - SUMO交通控制分布式推理

功能说明：
1. 加载训练好的TrafficController模型Checkpoint
2. 初始化Ray RemoteActor，将模型部署到远程节点
3. 连接到SUMO环境（使用SUMO-RL）
4. 在SUMO环境中实时收集车辆状态数据
5. 使用模型进行推理，生成控制动作
6. 应用控制动作到SUMO环境
7. 保留主动车辆调度ICV和安全屏障功能
8. 支持分布式推理，多个SUMO实例并行运行
9. 添加详细的推理日志和统计信息
10. 支持从配置文件加载模型路径和SUMO配置

推理流程：
- Ray Driver启动多个Remote Actors
- 每个Actor连接到独立的SUMO实例
- Actors并行进行推理，提升吞吐量
- 实时收集数据并进行决策
- 输出推理结果和性能指标

使用示例：
    # 基础推理
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --config config.json
    
    # 多实例并行推理
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --num_instances 4
    
    # 使用GUI可视化
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --use_gui
"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import defaultdict, deque

# Ray导入
import ray
from ray import serve
from ray.serve import Deployment

# 本地导入
from neural_traffic_controller import TrafficController
from safety_shield import DualModeSafetyShield
from influence_controller import InfluenceDrivenController, IDMController
from sumo_gym_env import SUMOGymEnv, create_sumo_gym_env

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 默认配置
# ============================================================================

def get_default_config() -> Dict[str, Any]:
    """
    获取默认推理配置
    
    Returns:
        config: 默认配置字典
    """
    return {
        # ==================== 模型配置 ====================
        "checkpoint_path": "models/traffic_controller_v1.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # ==================== SUMO环境配置 ====================
        "sumo_cfg_path": "仿真环境-初赛/sumo.sumocfg",
        "use_libsumo": True,
        "batch_subscribe": True,
        "max_steps": 3600,
        "use_gui": False,
        
        # ==================== Ray分布式配置 ====================
        "num_instances": 1,  # SUMO实例数量
        "num_cpus_per_instance": 1,
        "num_gpus": 0,  # 推理通常使用CPU，如需GPU可设置
        "ray_address": None,  # Ray集群地址，None表示本地
        
        # ==================== 模型超参数 ====================
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
        
        # ==================== 安全参数 ====================
        "ttc_threshold": 2.0,
        "thw_threshold": 1.5,
        "max_accel": 2.0,
        "max_decel": -3.0,
        "emergency_decel": -5.0,
        "max_lane_change_speed": 5.0,
        
        # ==================== 推理参数 ====================
        "warmup_steps": 10,  # 预热步数
        "log_interval": 10,  # 日志输出间隔
        "save_results": True,
        "results_dir": "./inference_results",
        
        # ==================== 日志配置 ====================
        "log_level": "INFO",
        "verbose": False,
    }


# ============================================================================
# Ray Remote Actor - 推理引擎
# ============================================================================

@ray.remote(num_cpus=1)
class InferenceActor:
    """
    Ray Remote Actor - 推理引擎
    
    每个Actor负责：
    1. 加载TrafficController模型
    2. 连接到独立的SUMO实例
    3. 实时收集车辆状态数据
    4. 执行模型推理
    5. 应用控制动作
    6. 记录统计信息
    """
    
    def __init__(
        self,
        actor_id: int,
        config: Dict[str, Any],
        checkpoint_path: str
    ):
        """
        初始化InferenceActor
        
        Args:
            actor_id: Actor ID
            config: 推理配置字典
            checkpoint_path: 模型检查点路径
        """
        self.actor_id = actor_id
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # 设备配置
        self.device = torch.device(config["device"])
        
        # 统计信息
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.inference_times = deque(maxlen=100)
        self.safety_interventions = {
            'level1': 0,
            'level2': 0
        }
        self.vehicle_control_stats = defaultdict(int)
        
        # 初始化模型和环境
        self.model = None
        self.env = None
        self.safety_shield = None
        self.idm_controller = None
        
        logger.info(f"✅ Actor {actor_id} 初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化模型和环境
        
        Returns:
            success: 是否初始化成功
        """
        try:
            # 1. 加载模型
            self._load_model()
            
            # 2. 初始化安全屏障
            self._init_safety_shield()
            
            # 3. 初始化IDM控制器
            self._init_idm_controller()
            
            # 4. 创建SUMO环境
            self._create_environment()
            
            logger.info(f"✅ Actor {self.actor_id} 模型和环境初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ Actor {self.actor_id} 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_model(self):
        """加载TrafficController模型"""
        # 构建模型配置
        model_config = {
            'node_dim': self.config['node_dim'],
            'edge_dim': self.config['edge_dim'],
            'gnn_hidden_dim': self.config['gnn_hidden_dim'],
            'gnn_output_dim': self.config['gnn_output_dim'],
            'gnn_layers': self.config['gnn_layers'],
            'gnn_heads': self.config['gnn_heads'],
            'world_hidden_dim': self.config['world_hidden_dim'],
            'future_steps': self.config['future_steps'],
            'controller_hidden_dim': self.config['controller_hidden_dim'],
            'global_dim': self.config['global_dim'],
            'top_k': self.config['top_k'],
            'action_dim': self.config['action_dim'],
            # 安全参数
            'ttc_threshold': self.config['ttc_threshold'],
            'thw_threshold': self.config['thw_threshold'],
            'max_accel': self.config['max_accel'],
            'max_decel': self.config['max_decel'],
            'emergency_decel': self.config['emergency_decel'],
            'max_lane_change_speed': self.config['max_lane_change_speed'],
        }
        
        # 创建模型
        self.model = TrafficController(model_config).to(self.device)
        self.model.eval()  # 推理模式
        
        # 加载检查点
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 处理不同的检查点格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"✅ Actor {self.actor_id} 模型加载成功: {self.checkpoint_path}")
        else:
            logger.warning(f"⚠️  检查点文件不存在: {self.checkpoint_path}，使用随机初始化模型")
    
    def _init_safety_shield(self):
        """初始化安全屏障"""
        self.safety_shield = DualModeSafetyShield(
            ttc_threshold=self.config['ttc_threshold'],
            thw_threshold=self.config['thw_threshold'],
            max_accel=self.config['max_accel'],
            max_decel=self.config['max_decel'],
            emergency_decel=self.config['emergency_decel'],
            max_lane_change_speed=self.config['max_lane_change_speed']
        )
    
    def _init_idm_controller(self):
        """初始化IDM控制器"""
        self.idm_controller = IDMController(
            desired_speed=30.0,
            safe_time_headway=self.config['thw_threshold'],
            min_gap=2.0,
            max_accel=self.config['max_accel'],
            comfortable_decel=abs(self.config['max_decel'])
        )
    
    def _create_environment(self):
        """创建SUMO环境"""
        self.env = create_sumo_gym_env(
            sumo_cfg_path=self.config['sumo_cfg_path'],
            use_libsumo=self.config['use_libsumo'],
            batch_subscribe=self.config['batch_subscribe'],
            device='cpu',  # 环境使用CPU
            model_config={
                'node_dim': self.config['node_dim'],
                'edge_dim': self.config['edge_dim'],
                'gnn_hidden_dim': self.config['gnn_hidden_dim'],
                'gnn_output_dim': self.config['gnn_output_dim'],
                'gnn_layers': self.config['gnn_layers'],
                'gnn_heads': self.config['gnn_heads'],
                'world_hidden_dim': self.config['world_hidden_dim'],
                'future_steps': self.config['future_steps'],
                'controller_hidden_dim': self.config['controller_hidden_dim'],
                'global_dim': self.config['global_dim'],
                'top_k': self.config['top_k'],
                'ttc_threshold': self.config['ttc_threshold'],
                'thw_threshold': self.config['thw_threshold'],
                'max_accel': self.config['max_accel'],
                'max_decel': self.config['max_decel'],
                'emergency_decel': self.config['emergency_decel'],
                'max_lane_change_speed': self.config['max_lane_change_speed'],
            },
            max_steps=self.config['max_steps'],
            use_gui=self.config['use_gui']
        )
    
    def run_episode(self) -> Dict[str, Any]:
        """
        运行一个完整的推理episode
        
        Returns:
            episode_stats: Episode统计信息
        """
        # 重置环境
        observation, info = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        
        episode_data = {
            'steps': [],
            'rewards': [],
            'vehicle_counts': [],
            'safety_metrics': [],
            'inference_times': [],
            'controlled_vehicles': []
        }
        
        logger.info(f"🚀 Actor {self.actor_id} 开始Episode {self.episode_count}")
        
        try:
            while True:
                # 执行推理步骤
                step_result = self._run_inference_step(observation)
                
                # 记录数据
                episode_data['steps'].append(self.step_count)
                episode_data['rewards'].append(step_result['reward'])
                episode_data['vehicle_counts'].append(step_result['vehicle_count'])
                episode_data['safety_metrics'].append(step_result['safety_metrics'])
                episode_data['inference_times'].append(step_result['inference_time'])
                episode_data['controlled_vehicles'].append(step_result['controlled_vehicles'])
                
                # 更新统计
                self.total_reward += step_result['reward']
                self.safety_interventions['level1'] += step_result.get('level1_interventions', 0)
                self.safety_interventions['level2'] += step_result.get('level2_interventions', 0)
                
                # 检查终止条件
                if step_result['done'] or step_result['truncated']:
                    break
                
                # 更新观测
                observation = step_result['observation']
                self.step_count += 1
                
                # 日志输出
                if self.step_count % self.config['log_interval'] == 0:
                    self._log_step_info(step_result)
        
        except Exception as e:
            logger.error(f"❌ Actor {self.actor_id} Episode {self.episode_count} 发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.episode_count += 1
            self.env.close()
        
        # 计算episode统计
        episode_stats = {
            'actor_id': self.actor_id,
            'episode_id': self.episode_count - 1,
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.step_count, 1),
            'avg_vehicle_count': np.mean(episode_data['vehicle_counts']),
            'avg_inference_time': np.mean(episode_data['inference_times']),
            'total_level1_interventions': self.safety_interventions['level1'],
            'total_level2_interventions': self.safety_interventions['level2'],
            'episode_data': episode_data
        }
        
        logger.info(f"✅ Actor {self.actor_id} Episode {episode_stats['episode_id']} 完成")
        logger.info(f"   总步数: {episode_stats['total_steps']}")
        logger.info(f"   总奖励: {episode_stats['total_reward']:.2f}")
        logger.info(f"   平均推理时间: {episode_stats['avg_inference_time']*1000:.2f}ms")
        logger.info(f"   Level1干预: {episode_stats['total_level1_interventions']}")
        logger.info(f"   Level2干预: {episode_stats['total_level2_interventions']}")
        
        return episode_stats
    
    def _run_inference_step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单步推理
        
        Args:
            observation: 当前观测
            
        Returns:
            step_result: 步骤结果
        """
        start_time = time.time()
        
        # 1. 准备批次数据
        batch = self._prepare_batch(observation)
        
        # 2. 运行模型推理
        with torch.no_grad():
            controller_output = self.model(batch, self.step_count)
        
        # 3. 提取控制动作
        raw_actions = controller_output.get('safe_actions', torch.zeros(0, 2))
        selected_vehicle_ids = controller_output.get('selected_vehicle_ids', [])
        selected_indices = controller_output.get('selected_indices', [])
        
        # 4. 应用安全屏障
        if len(selected_indices) > 0 and raw_actions.size(0) > 0:
            safety_output = self.safety_shield(
                raw_actions,
                batch['vehicle_states'],
                selected_indices
            )
            safe_actions = safety_output['safe_actions']
            level1_interventions = safety_output['level1_interventions']
            level2_interventions = safety_output['level2_interventions']
        else:
            safe_actions = raw_actions
            level1_interventions = 0
            level2_interventions = 0
        
        # 5. 应用动作到环境
        self.env._apply_actions(selected_vehicle_ids, {'actions': safe_actions})
        
        # 6. 推进仿真
        import traci
        traci.simulationStep()
        
        # 7. 获取新观测
        new_observation = self.env._get_observation()
        
        # 8. 计算奖励
        reward = self.env._calculate_reward(new_observation)
        
        # 9. 计算安全指标
        safety_metrics = self.env._calculate_safety_metrics(new_observation)
        
        # 10. 检查终止条件
        done, truncated = self.env._check_termination()
        
        # 计算推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return {
            'observation': new_observation,
            'reward': reward,
            'done': done,
            'truncated': truncated,
            'vehicle_count': len(new_observation['vehicle_ids']),
            'safety_metrics': safety_metrics,
            'inference_time': inference_time,
            'controlled_vehicles': len(selected_vehicle_ids),
            'level1_interventions': level1_interventions,
            'level2_interventions': level2_interventions
        }
    
    def _prepare_batch(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备模型输入批次
        
        Args:
            observation: 观测数据
            
        Returns:
            batch: 批次数据
        """
        batch = {
            'node_features': torch.tensor(
                observation['node_features'], dtype=torch.float32
            ).to(self.device),
            'edge_indices': torch.tensor(
                observation['edge_indices'], dtype=torch.long
            ).to(self.device),
            'edge_features': torch.tensor(
                observation['edge_features'], dtype=torch.float32
            ).to(self.device),
            'global_metrics': torch.tensor(
                observation['global_metrics'], dtype=torch.float32
            ).unsqueeze(0).to(self.device),
            'vehicle_ids': observation['vehicle_ids'].tolist(),
            'is_icv': torch.tensor(
                observation['is_icv'], dtype=torch.bool
            ).to(self.device),
            'vehicle_states': {
                'ids': observation['vehicle_ids'].tolist(),
                'data': observation.get('vehicle_data', {})
            }
        }
        return batch
    
    def _log_step_info(self, step_result: Dict[str, Any]):
        """输出步骤信息"""
        logger.info(
            f"Actor {self.actor_id} Step {self.step_count}: "
            f"Reward={step_result['reward']:.4f}, "
            f"Vehicles={step_result['vehicle_count']}, "
            f"Controlled={step_result['controlled_vehicles']}, "
            f"Time={step_result['inference_time']*1000:.2f}ms"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取Actor统计信息
        
        Returns:
            stats: 统计信息字典
        """
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0.0
        
        return {
            'actor_id': self.actor_id,
            'episode_count': self.episode_count,
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'avg_inference_time': avg_inference_time,
            'level1_interventions': self.safety_interventions['level1'],
            'level2_interventions': self.safety_interventions['level2']
        }


# ============================================================================
# 分布式推理协调器
# ============================================================================

class DistributedInferenceCoordinator:
    """
    分布式推理协调器
    
    负责：
    1. 初始化Ray集群
    2. 创建和管理多个InferenceActor
    3. 协调并行推理任务
    4. 收集和聚合统计信息
    5. 保存推理结果
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        """
        初始化协调器
        
        Args:
            config: 推理配置字典
            checkpoint_path: 模型检查点路径
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Ray配置
        self.num_instances = config['num_instances']
        self.num_cpus_per_instance = config['num_cpus_per_instance']
        self.num_gpus = config['num_gpus']
        self.ray_address = config.get('ray_address', None)
        
        # Actors
        self.actors = []
        
        # 统计信息
        self.start_time = None
        self.episode_stats = []
        
        logger.info("✅ 分布式推理协调器初始化完成")
    
    def initialize_ray(self):
        """初始化Ray集群"""
        if not ray.is_initialized():
            # 计算资源需求
            num_cpus = self.num_instances * self.num_cpus_per_instance + 2  # +2 for driver
            
            ray.init(
                address=self.ray_address,
                num_gpus=self.num_gpus,
                num_cpus=num_cpus,
                ignore_reinit_error=True,
                log_to_driver=self.config['log_level'] == 'INFO'
            )
            logger.info(f"✅ Ray集群初始化成功")
            logger.info(f"   地址: {self.ray_address or 'local'}")
            logger.info(f"   CPUs: {num_cpus}")
            logger.info(f"   GPUs: {self.num_gpus}")
        else:
            logger.info("ℹ️  Ray集群已初始化")
    
    def create_actors(self) -> bool:
        """
        创建InferenceActor实例
        
        Returns:
            success: 是否创建成功
        """
        try:
            self.actors = []
            
            for i in range(self.num_instances):
                # 创建Actor
                actor = InferenceActor.remote(
                    actor_id=i,
                    config=self.config,
                    checkpoint_path=self.checkpoint_path
                )
                
                # 初始化Actor
                success = ray.get(actor.initialize.remote())
                
                if success:
                    self.actors.append(actor)
                    logger.info(f"✅ Actor {i} 创建并初始化成功")
                else:
                    logger.error(f"❌ Actor {i} 初始化失败")
                    return False
            
            logger.info(f"✅ 所有 {len(self.actors)} 个Actor创建成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 创建Actor失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_inference(self, num_episodes: int = 1) -> List[Dict[str, Any]]:
        """
        运行分布式推理
        
        Args:
            num_episodes: 每个Actor运行的episode数量
            
        Returns:
            all_episode_stats: 所有episode的统计信息
        """
        self.start_time = time.time()
        self.episode_stats = []
        
        logger.info("=" * 80)
        logger.info("🚀 开始分布式推理")
        logger.info("=" * 80)
        logger.info(f"   Actor数量: {len(self.actors)}")
        logger.info(f"   每Actor Episode数: {num_episodes}")
        logger.info(f"   总Episode数: {len(self.actors) * num_episodes}")
        logger.info("=" * 80)
        
        try:
            # 并行运行推理
            futures = []
            for actor in self.actors:
                for _ in range(num_episodes):
                    futures.append(actor.run_episode.remote())
            
            # 收集结果
            episode_stats = ray.get(futures)
            self.episode_stats = episode_stats
            
            # 打印汇总统计
            self._print_summary_stats()
            
            return episode_stats
            
        except Exception as e:
            logger.error(f"❌ 推理过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _print_summary_stats(self):
        """打印汇总统计信息"""
        if not self.episode_stats:
            return
        
        # 计算汇总统计
        total_episodes = len(self.episode_stats)
        total_steps = sum(s['total_steps'] for s in self.episode_stats)
        total_reward = sum(s['total_reward'] for s in self.episode_stats)
        avg_reward = total_reward / total_episodes
        avg_inference_time = np.mean([s['avg_inference_time'] for s in self.episode_stats])
        total_level1 = sum(s['total_level1_interventions'] for s in self.episode_stats)
        total_level2 = sum(s['total_level2_interventions'] for s in self.episode_stats)
        
        elapsed_time = time.time() - self.start_time
        throughput = total_steps / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "=" * 80)
        print("📊 分布式推理汇总统计")
        print("=" * 80)
        print(f"⏱️  总运行时间: {elapsed_time:.2f}秒")
        print(f"📈 性能指标:")
        print(f"   - 总Episode数: {total_episodes}")
        print(f"   - 总步数: {total_steps}")
        print(f"   - 总奖励: {total_reward:.2f}")
        print(f"   - 平均奖励: {avg_reward:.4f}")
        print(f"   - 平均推理时间: {avg_inference_time*1000:.2f}ms")
        print(f"   - 吞吐量: {throughput:.2f} 步/秒")
        print(f"🛡️  安全指标:")
        print(f"   - Level1干预总数: {total_level1}")
        print(f"   - Level2干预总数: {total_level2}")
        print(f"   - 平均干预率: {(total_level1 + total_level2) / total_steps * 100:.2f}%")
        print("=" * 80)
    
    def save_results(self, results_dir: Optional[str] = None):
        """
        保存推理结果
        
        Args:
            results_dir: 结果保存目录
        """
        if not self.config.get('save_results', True):
            return
        
        results_dir = results_dir or self.config.get('results_dir', './inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存episode统计
        stats_file = os.path.join(results_dir, f"inference_stats_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.episode_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 推理结果已保存: {stats_file}")
        
        # 保存汇总统计
        summary_file = os.path.join(results_dir, f"inference_summary_{timestamp}.json")
        summary = {
            'config': self.config,
            'checkpoint_path': self.checkpoint_path,
            'total_episodes': len(self.episode_stats),
            'total_steps': sum(s['total_steps'] for s in self.episode_stats),
            'total_reward': sum(s['total_reward'] for s in self.episode_stats),
            'avg_reward': sum(s['total_reward'] for s in self.episode_stats) / max(len(self.episode_stats), 1),
            'avg_inference_time': np.mean([s['avg_inference_time'] for s in self.episode_stats]),
            'total_level1_interventions': sum(s['total_level1_interventions'] for s in self.episode_stats),
            'total_level2_interventions': sum(s['total_level2_interventions'] for s in self.episode_stats),
            'elapsed_time': time.time() - self.start_time,
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 汇总统计已保存: {summary_file}")
    
    def shutdown(self):
        """关闭协调器"""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("✅ Ray集群已关闭")


# ============================================================================
# 配置文件加载
# ============================================================================

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
# 命令行接口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        args: 命令行参数
    """
    parser = argparse.ArgumentParser(
        description="Ray推理脚本 - SUMO交通控制分布式推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基础推理
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth
    
    # 从配置文件加载
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --config config.json
    
    # 多实例并行推理
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --num_instances 4
    
    # 使用GUI可视化
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --use_gui
    
    # 连接到Ray集群
    python ray_inference.py --checkpoint models/traffic_controller_v1.pth --ray-address ray://localhost:10001
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（JSON格式）"
    )
    
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="SUMO实例数量（并行Actor数）"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="每个Actor运行的episode数量"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="每个episode的最大步数"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help="计算设备"
    )
    
    parser.add_argument(
        "--sumo-cfg",
        type=str,
        default=None,
        help="SUMO配置文件路径"
    )
    
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="启用SUMO GUI"
    )
    
    parser.add_argument(
        "--use-libsumo",
        action="store_true",
        help="启用LIBSUMO_AS_TRACI加速"
    )
    
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray集群地址（如：ray://localhost:10001）"
    )
    
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="日志输出间隔"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="结果保存目录"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出"
    )
    
    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

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
    config['checkpoint_path'] = args.checkpoint
    
    if args.num_instances is not None:
        config['num_instances'] = args.num_instances
    if args.device is not None:
        config['device'] = args.device
    if args.max_steps is not None:
        config['max_steps'] = args.max_steps
    if args.sumo_cfg is not None:
        config['sumo_cfg_path'] = args.sumo_cfg
    if args.use_gui:
        config['use_gui'] = True
    if args.use_libsumo:
        config['use_libsumo'] = True
    if args.ray_address is not None:
        config['ray_address'] = args.ray_address
    if args.log_interval is not None:
        config['log_interval'] = args.log_interval
    if args.results_dir is not None:
        config['results_dir'] = args.results_dir
    if args.verbose:
        config['verbose'] = True
        config['log_level'] = 'DEBUG'
    
    # 打印配置信息
    print("\n" + "=" * 80)
    print("🚀 Ray推理配置")
    print("=" * 80)
    print(f"📁 检查点路径: {config['checkpoint_path']}")
    print(f"🖥️  计算设备: {config['device']}")
    print(f"🌐 SUMO配置: {config['sumo_cfg_path']}")
    print(f"🔧 LIBSUMO: {config['use_libsumo']}")
    print(f"📺 GUI: {config['use_gui']}")
    print(f"⚙️  实例数量: {config['num_instances']}")
    print(f"📊 最大步数: {config['max_steps']}")
    print(f"🔗 Ray地址: {config['ray_address'] or 'local'}")
    print("=" * 80)
    
    # 创建协调器
    coordinator = DistributedInferenceCoordinator(
        config=config,
        checkpoint_path=config['checkpoint_path']
    )
    
    try:
        # 初始化Ray
        coordinator.initialize_ray()
        
        # 创建Actors
        if not coordinator.create_actors():
            logger.error("❌ Actor创建失败，退出")
            return
        
        # 运行推理
        episode_stats = coordinator.run_inference(num_episodes=args.num_episodes)
        
        # 保存结果
        coordinator.save_results()
        
    except KeyboardInterrupt:
        print("\n⚠️  推理被用户中断")
    
    except Exception as e:
        logger.error(f"❌ 推理过程发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭协调器
        coordinator.shutdown()


# ============================================================================
# 脚本入口
# ============================================================================

if __name__ == "__main__":
    main()
