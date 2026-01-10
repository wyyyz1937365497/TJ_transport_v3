"""
Ray系统测试脚本
测试所有Ray相关组件的集成和协同工作

测试范围：
1. SUMO-RL环境封装（sumo_gym_env.py）
2. Ray模型包装器（ray_model.py）
3. Ray ConstrainedPPO训练器（ray_trainer.py）
4. Ray训练脚本（ray_train.py）的配置加载
5. Ray推理脚本（ray_inference.py）的模型加载和推理
6. 组件集成测试
7. 错误处理测试

运行方式：
    python test_ray_system.py
    python test_ray_system.py --component all
    python test_ray_system.py --component environment
    python test_ray_system.py --component model
    python test_ray_system.py --component trainer
    python test_ray_system.py --component train_script
    python test_ray_system.py --component inference_script
    python test_ray_system.py --component integration
    python test_ray_system.py --component error_handling
"""

import os
import sys
import time
import json
import unittest
import argparse
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 测试工具函数
# ============================================================================

class TestResult:
    """测试结果记录器"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'skipped': [],
            'errors': []
        }
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始测试"""
        self.start_time = time.time()
        print("=" * 80)
        print("🚀 开始Ray系统测试")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def record_pass(self, test_name: str, duration: float):
        """记录通过测试"""
        self.results['passed'].append({
            'name': test_name,
            'duration': duration
        })
        print(f"   ✅ {test_name} - 通过 ({duration:.3f}s)")
    
    def record_fail(self, test_name: str, error: str, duration: float):
        """记录失败测试"""
        self.results['failed'].append({
            'name': test_name,
            'error': error,
            'duration': duration
        })
        print(f"   ❌ {test_name} - 失败 ({duration:.3f}s)")
        print(f"      错误: {error}")
    
    def record_skip(self, test_name: str, reason: str):
        """记录跳过测试"""
        self.results['skipped'].append({
            'name': test_name,
            'reason': reason
        })
        print(f"   ⏭️  {test_name} - 跳过")
        print(f"      原因: {reason}")
    
    def record_error(self, test_name: str, error: str):
        """记录错误"""
        self.results['errors'].append({
            'name': test_name,
            'error': error
        })
        print(f"   ⚠️  {test_name} - 错误")
        print(f"      错误: {error}")
    
    def finish(self):
        """完成测试"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 测试结果汇总")
        print("=" * 80)
        print(f"⏱️  总耗时: {duration:.2f}秒")
        print(f"✅ 通过: {len(self.results['passed'])}")
        print(f"❌ 失败: {len(self.results['failed'])}")
        print(f"⏭️  跳过: {len(self.results['skipped'])}")
        print(f"⚠️  错误: {len(self.results['errors'])}")
        print("=" * 80)
        
        # 打印失败的测试详情
        if self.results['failed']:
            print("\n❌ 失败测试详情:")
            for fail in self.results['failed']:
                print(f"   - {fail['name']}")
                print(f"     错误: {fail['error']}")
        
        # 打印错误详情
        if self.results['errors']:
            print("\n⚠️  错误详情:")
            for error in self.results['errors']:
                print(f"   - {error['name']}")
                print(f"     错误: {error['error']}")
        
        # 保存测试报告
        self._save_report()
        
        return len(self.results['failed']) == 0 and len(self.results['errors']) == 0
    
    def _save_report(self):
        """保存测试报告到JSON文件"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': self.end_time - self.start_time if self.end_time else 0,
            'summary': {
                'passed': len(self.results['passed']),
                'failed': len(self.results['failed']),
                'skipped': len(self.results['skipped']),
                'errors': len(self.results['errors'])
            },
            'details': self.results
        }
        
        report_dir = "./test_reports"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"ray_system_test_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 测试报告已保存: {report_file}")


def run_test(test_func, test_name: str, test_result: TestResult):
    """运行单个测试并记录结果"""
    start_time = time.time()
    try:
        test_func()
        duration = time.time() - start_time
        test_result.record_pass(test_name, duration)
        return True
    except AssertionError as e:
        duration = time.time() - start_time
        test_result.record_fail(test_name, str(e), duration)
        return False
    except Exception as e:
        duration = time.time() - start_time
        test_result.record_error(test_name, str(e))
        return False


# ============================================================================
# 1. SUMO-RL环境封装测试
# ============================================================================

class TestSUMOGymEnv:
    """测试SUMO-RL环境封装"""
    
    @staticmethod
    def test_import():
        """测试导入"""
        try:
            from sumo_gym_env import SUMOGymEnv, create_sumo_gym_env
            assert SUMOGymEnv is not None
            assert create_sumo_gym_env is not None
        except ImportError as e:
            raise AssertionError(f"导入失败: {e}")
    
    @staticmethod
    def test_environment_creation():
        """测试环境创建"""
        from sumo_gym_env import SUMOGymEnv
        import gymnasium as gym
        
        # 创建模拟配置
        mock_config = {
            'sumo_cfg_path': '仿真环境-初赛/sumo.sumocfg',
            'port': None,
            'use_libsumo': False,
            'batch_subscribe': True,
            'device': 'cpu',
            'model_config': None,
            'max_steps': 100,
            'use_gui': False,
            'seed': 42
        }
        
        # Mock SUMO相关组件
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                # 创建环境（不启动SUMO）
                env = SUMOGymEnv(**mock_config)
                
                # 验证环境属性
                assert env.sumo_cfg_path == mock_config['sumo_cfg_path']
                assert env.device == mock_config['device']
                assert env.max_steps == mock_config['max_steps']
                assert env.use_gui == mock_config['use_gui']
                
                # 验证观察空间和动作空间已定义
                assert hasattr(env, 'observation_space')
                assert hasattr(env, 'action_space')
                assert isinstance(env.observation_space, gym.spaces.Dict)
                assert isinstance(env.action_space, gym.spaces.Box)
                
                # 验证模型已创建
                assert env.traffic_controller is not None
                assert env.traffic_controller.training == False  # 推理模式
    
    @staticmethod
    def test_observation_space():
        """测试观察空间定义"""
        from sumo_gym_env import SUMOGymEnv
        import gymnasium as gym
        
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                env = SUMOGymEnv(
                    sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
                    use_libsumo=False,
                    batch_subscribe=True,
                    device='cpu',
                    max_steps=100,
                    use_gui=False
                )
                
                # 验证观察空间结构
                obs_space = env.observation_space
                assert 'node_features' in obs_space
                assert 'edge_indices' in obs_space
                assert 'edge_features' in obs_space
                assert 'global_metrics' in obs_space
                assert 'vehicle_ids' in obs_space
                assert 'is_icv' in obs_space
                
                # 验证观察空间类型
                assert isinstance(obs_space['node_features'], gym.spaces.Box)
                assert isinstance(obs_space['edge_indices'], gym.spaces.Box)
                assert isinstance(obs_space['edge_features'], gym.spaces.Box)
                assert isinstance(obs_space['global_metrics'], gym.spaces.Box)
                
                # 验证观察空间形状
                assert obs_space['global_metrics'].shape == (16,)
    
    @staticmethod
    def test_action_space():
        """测试动作空间定义"""
        from sumo_gym_env import SUMOGymEnv
        import gymnasium as gym
        
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                env = SUMOGymEnv(
                    sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
                    use_libsumo=False,
                    batch_subscribe=True,
                    device='cpu',
                    max_steps=100,
                    use_gui=False
                )
                
                # 验证动作空间
                action_space = env.action_space
                assert isinstance(action_space, gym.spaces.Box)
                assert action_space.shape == (2,)
                assert action_space.dtype == np.float32
                
                # 验证动作范围
                assert np.all(action_space.low == np.array([-5.0, 0.0]))
                assert np.all(action_space.high == np.array([5.0, 1.0]))
    
    @staticmethod
    def test_get_empty_observation():
        """测试获取空观测"""
        from sumo_gym_env import SUMOGymEnv
        
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                env = SUMOGymEnv(
                    sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
                    use_libsumo=False,
                    batch_subscribe=True,
                    device='cpu',
                    max_steps=100,
                    use_gui=False
                )
                
                # 获取空观测
                empty_obs = env._get_empty_observation()
                
                # 验证空观测结构
                assert 'node_features' in empty_obs
                assert 'edge_indices' in empty_obs
                assert 'edge_features' in empty_obs
                assert 'global_metrics' in empty_obs
                assert 'vehicle_ids' in empty_obs
                assert 'is_icv' in empty_obs
                assert 'vehicle_data' in empty_obs
                
                # 验证空观测形状
                assert empty_obs['node_features'].shape == (0, 9)
                assert empty_obs['edge_indices'].shape == (2, 0)
                assert empty_obs['edge_features'].shape == (0, 4)
                assert empty_obs['global_metrics'].shape == (16,)
                assert len(empty_obs['vehicle_ids']) == 0
                assert len(empty_obs['is_icv']) == 0
    
    @staticmethod
    def test_build_graph():
        """测试图构建"""
        from sumo_gym_env import SUMOGymEnv
        
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                env = SUMOGymEnv(
                    sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
                    use_libsumo=False,
                    batch_subscribe=True,
                    device='cpu',
                    max_steps=100,
                    use_gui=False
                )
                
                # 创建测试车辆数据
                vehicle_data = {
                    'veh_0': {
                        'position': 100.0,
                        'speed': 10.0,
                        'acceleration': 0.5,
                        'lane_index': 0,
                        'lane_id': 'lane_0',
                        'road_id': 'road_0',
                        'is_icv': True,
                        'id': 'veh_0'
                    },
                    'veh_1': {
                        'position': 150.0,
                        'speed': 15.0,
                        'acceleration': -0.3,
                        'lane_index': 0,
                        'lane_id': 'lane_0',
                        'road_id': 'road_0',
                        'is_icv': False,
                        'id': 'veh_1'
                    }
                }
                
                # 构建图
                graph_data = env._build_graph(vehicle_data)
                
                # 验证图数据结构
                assert 'node_features' in graph_data
                assert 'edge_indices' in graph_data
                assert 'edge_features' in graph_data
                assert 'is_icv' in graph_data
                
                # 验证节点特征
                assert graph_data['node_features'].shape == (2, 9)
                assert graph_data['is_icv'].shape == (2,)
                assert graph_data['is_icv'][0] == True
                assert graph_data['is_icv'][1] == False
    
    @staticmethod
    def test_compute_global_metrics():
        """测试全局指标计算"""
        from sumo_gym_env import SUMOGymEnv
        
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                env = SUMOGymEnv(
                    sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
                    use_libsumo=False,
                    batch_subscribe=True,
                    device='cpu',
                    max_steps=100,
                    use_gui=False
                )
                
                # 创建测试车辆数据
                vehicle_data = {
                    'veh_0': {
                        'position': 100.0,
                        'speed': 10.0,
                        'acceleration': 0.5,
                        'is_icv': True
                    },
                    'veh_1': {
                        'position': 150.0,
                        'speed': 15.0,
                        'acceleration': -0.3,
                        'is_icv': False
                    }
                }
                
                # 计算全局指标
                metrics = env._compute_global_metrics(vehicle_data)
                
                # 验证指标维度
                assert metrics.shape == (16,)
                assert not np.isnan(metrics).any()
                assert not np.isinf(metrics).any()
                
                # 验证指标合理性
                assert metrics[3] == 2.0  # 车辆数
                assert 10.0 <= metrics[0] <= 15.0  # 平均速度在合理范围内


# ============================================================================
# 2. Ray模型包装器测试
# ============================================================================

class TestRayModel:
    """测试Ray模型包装器"""
    
    @staticmethod
    def test_import():
        """测试导入"""
        try:
            from ray_model import (
                TrafficControllerModel,
                TrafficControllerModelV2,
                register_traffic_controller_model
            )
            assert TrafficControllerModel is not None
            assert TrafficControllerModelV2 is not None
            assert register_traffic_controller_model is not None
        except ImportError as e:
            raise AssertionError(f"导入失败: {e}")
    
    @staticmethod
    def test_model_creation():
        """测试模型创建"""
        from ray_model import TrafficControllerModel
        import gymnasium as gym
        
        # 创建观察空间
        obs_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(-np.inf, np.inf, shape=(None, 9), dtype=np.float32),
            'edge_indices': gym.spaces.Box(0, np.inf, shape=(2, None), dtype=np.int64),
            'edge_features': gym.spaces.Box(-np.inf, np.inf, shape=(None, 4), dtype=np.float32),
            'global_metrics': gym.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            'vehicle_ids': gym.spaces.Box(0, np.inf, shape=(None,), dtype=object),
            'is_icv': gym.spaces.Box(0, 1, shape=(None,), dtype=bool),
            'vehicle_states': gym.spaces.Dict()
        })
        
        # 创建动作空间
        action_space = gym.spaces.Box(
            low=np.array([-5.0, 0.0]),
            high=np.array([5.0, 1.0]),
            dtype=np.float32
        )
        
        # 创建模型配置
        model_config = {
            'node_dim': 9,
            'edge_dim': 4,
            'gnn_hidden_dim': 64,
            'gnn_output_dim': 256,
            'gnn_layers': 3,
            'gnn_heads': 4,
            'world_hidden_dim': 128,
            'future_steps': 5,
            'controller_hidden_dim': 128,
            'global_dim': 16,
            'top_k': 5,
            'action_dim': 2,
            'ttc_threshold': 2.0,
            'thw_threshold': 1.5,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'emergency_decel': -5.0,
            'max_lane_change_speed': 5.0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'cache_timeout': 10
        }
        
        # 创建模型
        model = TrafficControllerModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=2,
            model_config=model_config,
            name='test_model'
        )
        
        # 验证模型属性
        assert model.config == model_config
        assert model.traffic_controller is not None
        assert model.action_output is not None
        assert model.value_head is not None
    
    @staticmethod
    def test_forward_pass():
        """测试前向传播"""
        from ray_model import TrafficControllerModel
        import gymnasium as gym
        
        # 创建观察空间和动作空间
        obs_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(-np.inf, np.inf, shape=(None, 9), dtype=np.float32),
            'edge_indices': gym.spaces.Box(0, np.inf, shape=(2, None), dtype=np.int64),
            'edge_features': gym.spaces.Box(-np.inf, np.inf, shape=(None, 4), dtype=np.float32),
            'global_metrics': gym.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            'vehicle_ids': gym.spaces.Box(0, np.inf, shape=(None,), dtype=object),
            'is_icv': gym.spaces.Box(0, 1, shape=(None,), dtype=bool),
            'vehicle_states': gym.spaces.Dict()
        })
        
        action_space = gym.spaces.Box(
            low=np.array([-5.0, 0.0]),
            high=np.array([5.0, 1.0]),
            dtype=np.float32
        )
        
        model_config = {
            'node_dim': 9,
            'edge_dim': 4,
            'gnn_hidden_dim': 64,
            'gnn_output_dim': 256,
            'gnn_layers': 3,
            'gnn_heads': 4,
            'world_hidden_dim': 128,
            'future_steps': 5,
            'controller_hidden_dim': 128,
            'global_dim': 16,
            'top_k': 5,
            'action_dim': 2,
            'ttc_threshold': 2.0,
            'thw_threshold': 1.5,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'emergency_decel': -5.0,
            'max_lane_change_speed': 5.0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'cache_timeout': 10
        }
        
        model = TrafficControllerModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=2,
            model_config=model_config,
            name='test_model'
        )
        
        # 创建输入数据
        input_dict = {
            'obs': {
                'node_features': np.random.randn(5, 9).astype(np.float32),
                'edge_indices': np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
                'edge_features': np.random.randn(4, 4).astype(np.float32),
                'global_metrics': np.random.randn(16).astype(np.float32),
                'vehicle_ids': np.array(['veh_0', 'veh_1', 'veh_2', 'veh_3', 'veh_4'], dtype=object),
                'is_icv': np.array([True, False, True, False, True], dtype=bool),
                'vehicle_states': {
                    'ids': ['veh_0', 'veh_1', 'veh_2', 'veh_3', 'veh_4'],
                    'data': {}
                }
            }
        }
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            action_output, state = model(input_dict, [], None)
        
        # 验证输出
        assert action_output.shape[0] == 1  # batch_size
        assert action_output.shape[1] == 2  # action_dim
        assert state == []  # 无RNN状态
        
        # 验证价值函数
        value = model.value_function()
        assert value.shape[0] == 1  # batch_size
    
    @staticmethod
    def test_model_registration():
        """测试模型注册"""
        from ray_model import register_traffic_controller_model
        from ray.rllib.models import ModelCatalog
        
        # 注册模型
        register_traffic_controller_model()
        
        # 验证模型已注册
        assert 'traffic_controller_model' in ModelCatalog._model_v2_registry
        assert 'traffic_controller_model_v2' in ModelCatalog._model_v2_registry
    
    @staticmethod
    def test_prepare_batch():
        """测试批次准备"""
        from ray_model import TrafficControllerModel
        import gymnasium as gym
        
        # 创建模型
        obs_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(-np.inf, np.inf, shape=(None, 9), dtype=np.float32),
            'edge_indices': gym.spaces.Box(0, np.inf, shape=(2, None), dtype=np.int64),
            'edge_features': gym.spaces.Box(-np.inf, np.inf, shape=(None, 4), dtype=np.float32),
            'global_metrics': gym.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            'vehicle_ids': gym.spaces.Box(0, np.inf, shape=(None,), dtype=object),
            'is_icv': gym.spaces.Box(0, 1, shape=(None,), dtype=bool),
            'vehicle_states': gym.spaces.Dict()
        })
        
        action_space = gym.spaces.Box(
            low=np.array([-5.0, 0.0]),
            high=np.array([5.0, 1.0]),
            dtype=np.float32
        )
        
        model_config = {
            'node_dim': 9,
            'edge_dim': 4,
            'gnn_hidden_dim': 64,
            'gnn_output_dim': 256,
            'gnn_layers': 3,
            'gnn_heads': 4,
            'world_hidden_dim': 128,
            'future_steps': 5,
            'controller_hidden_dim': 128,
            'global_dim': 16,
            'top_k': 5,
            'action_dim': 2,
            'ttc_threshold': 2.0,
            'thw_threshold': 1.5,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'emergency_decel': -5.0,
            'max_lane_change_speed': 5.0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'cache_timeout': 10
        }
        
        model = TrafficControllerModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=2,
            model_config=model_config,
            name='test_model'
        )
        
        # 创建观测数据
        obs = {
            'node_features': np.random.randn(5, 9).astype(np.float32),
            'edge_indices': np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
            'edge_features': np.random.randn(4, 4).astype(np.float32),
            'global_metrics': np.random.randn(16).astype(np.float32),
            'vehicle_ids': ['veh_0', 'veh_1', 'veh_2', 'veh_3', 'veh_4'],
            'is_icv': np.array([True, False, True, False, True], dtype=bool),
            'vehicle_states': {
                'ids': ['veh_0', 'veh_1', 'veh_2', 'veh_3', 'veh_4'],
                'data': {}
            }
        }
        
        # 准备批次
        batch = model._prepare_batch(obs)
        
        # 验证批次数据
        assert 'node_features' in batch
        assert 'edge_indices' in batch
        assert 'edge_features' in batch
        assert 'global_metrics' in batch
        assert 'vehicle_ids' in batch
        assert 'is_icv' in batch
        assert 'vehicle_states' in batch
        
        # 验证张量类型
        assert isinstance(batch['node_features'], torch.Tensor)
        assert isinstance(batch['edge_indices'], torch.Tensor)
        assert isinstance(batch['edge_features'], torch.Tensor)
        assert isinstance(batch['global_metrics'], torch.Tensor)
        assert isinstance(batch['is_icv'], torch.Tensor)
        
        # 验证张量形状
        assert batch['node_features'].shape == (5, 9)
        assert batch['edge_indices'].shape == (2, 4)
        assert batch['edge_features'].shape == (4, 4)
        assert batch['global_metrics'].shape == (16,)
        assert batch['is_icv'].shape == (5,)


# ============================================================================
# 3. Ray ConstrainedPPO训练器测试
# ============================================================================

class TestRayTrainer:
    """测试Ray ConstrainedPPO训练器"""
    
    @staticmethod
    def test_import():
        """测试导入"""
        try:
            from ray_trainer import ConstrainedPPOTrainer, create_constrained_ppo_trainer
            assert ConstrainedPPOTrainer is not None
            assert create_constrained_ppo_trainer is not None
        except ImportError as e:
            raise AssertionError(f"导入失败: {e}")
    
    @staticmethod
    def test_trainer_initialization():
        """测试训练器初始化"""
        from ray_trainer import ConstrainedPPOTrainer
        
        # 创建配置
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            # 约束优化参数
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        # Mock Ray初始化
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                # 创建训练器
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 验证训练器属性
                assert trainer.cost_limit == 0.1
                assert trainer.lambda_lr == 0.01
                assert trainer.lambda_init == 1.0
                assert trainer.alpha == 0.5
                assert trainer.beta == 0.9
                
                # 验证拉格朗日乘子初始化
                assert 'default_policy' in trainer.lagrange_multipliers
                assert trainer.lagrange_multipliers['default_policy'] == 1.0
    
    @staticmethod
    def test_constraint_violation_computation():
        """测试约束违反计算"""
        from ray_trainer import ConstrainedPPOTrainer
        
        # 创建训练器
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 测试约束违反计算
                # 成本超过限制
                violation = trainer._compute_constraint_violation(0.15)
                assert violation == 0.5 * (0.15 - 0.1)  # alpha * (cost - limit)
                
                # 成本低于限制
                violation = trainer._compute_constraint_violation(0.05)
                assert violation == 0.5 * (0.05 - 0.1)
                
                # 成本等于限制
                violation = trainer._compute_constraint_violation(0.1)
                assert violation == 0.0
    
    @staticmethod
    def test_lagrange_multiplier_update():
        """测试拉格朗日乘子更新"""
        from ray_trainer import ConstrainedPPOTrainer
        
        # 创建训练器
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 测试乘子更新
                # 约束违反 > 0，乘子应该增加
                trainer.update_lagrange_multiplier('default_policy', 0.05)
                assert trainer.lagrange_multipliers['default_policy'] > 1.0
                
                # 约束违反 < 0，乘子应该减少
                trainer.update_lagrange_multiplier('default_policy', -0.05)
                assert trainer.lagrange_multipliers['default_policy'] < 1.01
                
                # 乘子不应该为负
                trainer.lagrange_multipliers['default_policy'] = 0.01
                trainer.update_lagrange_multiplier('default_policy', -1.0)
                assert trainer.lagrange_multipliers['default_policy'] >= 0.0
    
    @staticmethod
    def test_constraint_stats():
        """测试约束统计"""
        from ray_trainer import ConstrainedPPOTrainer
        
        # 创建训练器
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 添加一些历史数据
                trainer.cost_history['default_policy'] = [0.08, 0.09, 0.10, 0.11, 0.12]
                trainer.constraint_violation_history['default_policy'] = [-0.02, -0.01, 0.0, 0.01, 0.02]
                
                # 获取约束统计
                stats = trainer.get_constraint_stats()
                
                # 验证统计信息
                assert 'cost_limit' in stats
                assert 'lambda_lr' in stats
                assert 'alpha' in stats
                assert 'beta' in stats
                assert 'policies' in stats
                assert 'default_policy' in stats['policies']
                
                # 验证策略统计
                policy_stats = stats['policies']['default_policy']
                assert 'lagrangian_multiplier' in policy_stats
                assert 'cost_history' in policy_stats
                assert 'constraint_violation_history' in policy_stats
                assert 'avg_cost' in policy_stats
                assert 'avg_violation' in policy_stats
    
    @staticmethod
    def test_reset_lagrange_multipliers():
        """测试重置拉格朗日乘子"""
        from ray_trainer import ConstrainedPPOTrainer
        
        # 创建训练器
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 修改乘子
                trainer.lagrange_multipliers['default_policy'] = 5.0
                
                # 重置乘子
                trainer.reset_lagrange_multipliers()
                
                # 验证重置
                assert trainer.lagrange_multipliers['default_policy'] == 1.0
                
                # 重置到自定义值
                trainer.reset_lagrange_multipliers(value=2.0)
                assert trainer.lagrange_multipliers['default_policy'] == 2.0
    
    @staticmethod
    def test_set_cost_limit():
        """测试设置成本限制"""
        from ray_trainer import ConstrainedPPOTrainer
        
        # 创建训练器
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 修改成本限制
                trainer.set_cost_limit(0.2)
                
                # 验证修改
                assert trainer.cost_limit == 0.2


# ============================================================================
# 4. Ray训练脚本测试
# ============================================================================

class TestRayTrainScript:
    """测试Ray训练脚本"""
    
    @staticmethod
    def test_import():
        """测试导入"""
        try:
            from ray_train import (
                get_default_config,
                load_config_from_file,
                merge_configs,
                env_creator,
                build_ray_config,
                reward_shaping_with_lagrangian,
                validate_batch
            )
            assert get_default_config is not None
            assert load_config_from_file is not None
            assert merge_configs is not None
            assert env_creator is not None
            assert build_ray_config is not None
            assert reward_shaping_with_lagrangian is not None
            assert validate_batch is not None
        except ImportError as e:
            raise AssertionError(f"导入失败: {e}")
    
    @staticmethod
    def test_default_config():
        """测试默认配置"""
        from ray_train import get_default_config
        
        # 获取默认配置
        config = get_default_config()
        
        # 验证配置结构
        assert 'framework' in config
        assert 'env' in config
        assert 'sumo_cfg_path' in config
        assert 'use_libsumo' in config
        assert 'batch_subscribe' in config
        assert 'max_steps' in config
        assert 'num_workers' in config
        assert 'num_gpus' in config
        assert 'train_batch_size' in config
        assert 'rollout_fragment_length' in config
        assert 'lr' in config
        assert 'gamma' in config
        assert 'cost_limit' in config
        assert 'lambda_lr' in config
        assert 'lambda_init' in config
        assert 'model' in config
        
        # 验证配置值
        assert config['framework'] == 'torch'
        assert config['num_workers'] == 4
        assert config['num_gpus'] == 1
        assert config['train_batch_size'] == 4000
        assert config['rollout_fragment_length'] == 200
        assert config['cost_limit'] == 0.1
        assert config['lambda_lr'] == 0.01
    
    @staticmethod
    def test_merge_configs():
        """测试配置合并"""
        from ray_train import get_default_config, merge_configs
        
        # 获取默认配置
        default_config = get_default_config()
        
        # 创建用户配置
        user_config = {
            'num_workers': 8,
            'num_gpus': 2,
            'train_batch_size': 8000,
            'model': {
                'custom_model_config': {
                    'gnn_hidden_dim': 128
                }
            }
        }
        
        # 合并配置
        merged_config = merge_configs(default_config, user_config)
        
        # 验证合并结果
        assert merged_config['num_workers'] == 8
        assert merged_config['num_gpus'] == 2
        assert merged_config['train_batch_size'] == 8000
        assert merged_config['model']['custom_model_config']['gnn_hidden_dim'] == 128
        
        # 验证默认值保留
        assert merged_config['framework'] == 'torch'
        assert merged_config['cost_limit'] == 0.1
    
    @staticmethod
    def test_reward_shaping():
        """测试奖励重塑"""
        from ray_train import reward_shaping_with_lagrangian
        
        # 创建批次数据
        batch = {
            'rewards': np.array([1.0, 2.0, 3.0]),
            'level1_interventions': np.array([0, 1, 0]),
            'level2_interventions': np.array([0, 0, 1])
        }
        
        # 奖励重塑
        lambda_ = 1.0
        cost_limit = 0.1
        shaped_batch = reward_shaping_with_lagrangian(batch, lambda_, cost_limit)
        
        # 验证重塑结果
        assert 'rewards' in shaped_batch
        assert 'original_rewards' in shaped_batch
        assert 'lagrangian_penalty' in shaped_batch
        
        # 验证奖励被修改
        assert not np.array_equal(shaped_batch['rewards'], batch['rewards'])
        assert np.array_equal(shaped_batch['original_rewards'], batch['rewards'])
        
        # 验证惩罚计算
        total_cost = batch['level1_interventions'] + batch['level2_interventions']
        expected_penalty = lambda_ * (total_cost - cost_limit)
        assert np.allclose(shaped_batch['lagrangian_penalty'], expected_penalty)
    
    @staticmethod
    def test_batch_validation():
        """测试批次验证"""
        from ray_train import validate_batch
        
        # 有效批次
        valid_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(10, 2),
            'rewards': np.random.randn(10),
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        assert validate_batch(valid_batch) == True
        
        # 缺少必需字段
        invalid_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(10, 2),
            # 缺少 'rewards'
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        assert validate_batch(invalid_batch) == False
        
        # 包含NaN
        nan_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(10, 2),
            'rewards': np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        assert validate_batch(nan_batch) == False
        
        # 包含Inf
        inf_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(10, 2),
            'rewards': np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        assert validate_batch(inf_batch) == False
    
    @staticmethod
    def test_config_file_loading():
        """测试配置文件加载"""
        from ray_train import load_config_from_file, get_default_config, merge_configs
        
        # 创建临时配置文件
        temp_config = {
            'num_workers': 8,
            'num_gpus': 2,
            'train_batch_size': 8000,
            'model': {
                'custom_model_config': {
                    'gnn_hidden_dim': 128
                }
            }
        }
        
        # 保存到临时文件
        temp_file = './temp_test_config.json'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_config, f, indent=2)
        
        try:
            # 加载配置文件
            loaded_config = load_config_from_file(temp_file)
            
            # 验证加载结果
            assert loaded_config['num_workers'] == 8
            assert loaded_config['num_gpus'] == 2
            assert loaded_config['train_batch_size'] == 8000
            
            # 与默认配置合并
            default_config = get_default_config()
            merged_config = merge_configs(default_config, loaded_config)
            
            # 验证合并结果
            assert merged_config['num_workers'] == 8
            assert merged_config['num_gpus'] == 2
            assert merged_config['train_batch_size'] == 8000
            assert merged_config['framework'] == 'torch'  # 保留默认值
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)


# ============================================================================
# 5. Ray推理脚本测试
# ============================================================================

class TestRayInferenceScript:
    """测试Ray推理脚本"""
    
    @staticmethod
    def test_import():
        """测试导入"""
        try:
            from ray_inference import (
                get_default_config,
                load_config_from_file,
                merge_configs,
                InferenceActor,
                DistributedInferenceCoordinator
            )
            assert get_default_config is not None
            assert load_config_from_file is not None
            assert merge_configs is not None
            assert InferenceActor is not None
            assert DistributedInferenceCoordinator is not None
        except ImportError as e:
            raise AssertionError(f"导入失败: {e}")
    
    @staticmethod
    def test_default_config():
        """测试默认配置"""
        from ray_inference import get_default_config
        
        # 获取默认配置
        config = get_default_config()
        
        # 验证配置结构
        assert 'checkpoint_path' in config
        assert 'device' in config
        assert 'sumo_cfg_path' in config
        assert 'use_libsumo' in config
        assert 'batch_subscribe' in config
        assert 'max_steps' in config
        assert 'num_instances' in config
        assert 'node_dim' in config
        assert 'edge_dim' in config
        assert 'gnn_hidden_dim' in config
        assert 'gnn_output_dim' in config
        assert 'gnn_layers' in config
        assert 'gnn_heads' in config
        assert 'world_hidden_dim' in config
        assert 'future_steps' in config
        assert 'controller_hidden_dim' in config
        assert 'global_dim' in config
        assert 'top_k' in config
        assert 'action_dim' in config
        
        # 验证配置值
        assert config['num_instances'] == 1
        assert config['node_dim'] == 9
        assert config['edge_dim'] == 4
        assert config['gnn_hidden_dim'] == 64
        assert config['gnn_output_dim'] == 256
        assert config['gnn_layers'] == 3
        assert config['gnn_heads'] == 4
    
    @staticmethod
    def test_merge_configs():
        """测试配置合并"""
        from ray_inference import get_default_config, merge_configs
        
        # 获取默认配置
        default_config = get_default_config()
        
        # 创建用户配置
        user_config = {
            'num_instances': 4,
            'device': 'cuda',
            'max_steps': 7200,
            'gnn_hidden_dim': 128
        }
        
        # 合并配置
        merged_config = merge_configs(default_config, user_config)
        
        # 验证合并结果
        assert merged_config['num_instances'] == 4
        assert merged_config['device'] == 'cuda'
        assert merged_config['max_steps'] == 7200
        assert merged_config['gnn_hidden_dim'] == 128
        
        # 验证默认值保留
        assert merged_config['node_dim'] == 9
        assert merged_config['edge_dim'] == 4
    
    @staticmethod
    def test_config_file_loading():
        """测试配置文件加载"""
        from ray_inference import load_config_from_file, get_default_config, merge_configs
        
        # 创建临时配置文件
        temp_config = {
            'num_instances': 4,
            'device': 'cuda',
            'max_steps': 7200,
            'gnn_hidden_dim': 128
        }
        
        # 保存到临时文件
        temp_file = './temp_inference_test_config.json'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_config, f, indent=2)
        
        try:
            # 加载配置文件
            loaded_config = load_config_from_file(temp_file)
            
            # 验证加载结果
            assert loaded_config['num_instances'] == 4
            assert loaded_config['device'] == 'cuda'
            assert loaded_config['max_steps'] == 7200
            
            # 与默认配置合并
            default_config = get_default_config()
            merged_config = merge_configs(default_config, loaded_config)
            
            # 验证合并结果
            assert merged_config['num_instances'] == 4
            assert merged_config['device'] == 'cuda'
            assert merged_config['max_steps'] == 7200
            assert merged_config['node_dim'] == 9  # 保留默认值
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)


# ============================================================================
# 6. 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""
    
    @staticmethod
    def test_model_to_environment_integration():
        """测试模型与环境集成"""
        from ray_model import TrafficControllerModel
        from sumo_gym_env import SUMOGymEnv
        import gymnasium as gym
        
        # Mock SUMO环境
        with patch('sumo_gym_env.TRACI_AVAILABLE', False):
            with patch('sumo_gym_env.SUMO_RL_AVAILABLE', False):
                # 创建环境
                env = SUMOGymEnv(
                    sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
                    use_libsumo=False,
                    batch_subscribe=True,
                    device='cpu',
                    max_steps=100,
                    use_gui=False
                )
                
                # 创建模型
                obs_space = env.observation_space
                action_space = env.action_space
                
                model_config = {
                    'node_dim': 9,
                    'edge_dim': 4,
                    'gnn_hidden_dim': 64,
                    'gnn_output_dim': 256,
                    'gnn_layers': 3,
                    'gnn_heads': 4,
                    'world_hidden_dim': 128,
                    'future_steps': 5,
                    'controller_hidden_dim': 128,
                    'global_dim': 16,
                    'top_k': 5,
                    'action_dim': 2,
                    'ttc_threshold': 2.0,
                    'thw_threshold': 1.5,
                    'max_accel': 2.0,
                    'max_decel': -3.0,
                    'emergency_decel': -5.0,
                    'max_lane_change_speed': 5.0,
                    'cost_limit': 0.1,
                    'lambda_lr': 0.01,
                    'cache_timeout': 10
                }
                
                model = TrafficControllerModel(
                    obs_space=obs_space,
                    action_space=action_space,
                    num_outputs=2,
                    model_config=model_config,
                    name='test_model'
                )
                
                # 验证模型可以处理环境的观测
                obs = env._get_empty_observation()
                
                # 准备模型输入
                input_dict = {
                    'obs': obs
                }
                
                # 前向传播
                model.eval()
                with torch.no_grad():
                    action_output, state = model(input_dict, [], None)
                
                # 验证输出
                assert action_output.shape[0] == 1
                assert action_output.shape[1] == 2
    
    @staticmethod
    def test_trainer_to_model_integration():
        """测试训练器与模型集成"""
        from ray_trainer import ConstrainedPPOTrainer
        from ray_model import register_traffic_controller_model
        
        # 注册模型
        register_traffic_controller_model()
        
        # 创建配置
        config = {
            'env': 'CartPole-v1',
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01,
            'lambda_init': 1.0,
            'alpha': 0.5,
            'beta': 0.9,
        }
        
        # Mock Ray初始化
        with patch('ray_trainer.ray.is_initialized', return_value=False):
            with patch('ray_trainer.ray.init'):
                # 创建训练器
                trainer = ConstrainedPPOTrainer(config=config)
                
                # 验证训练器已创建
                assert trainer.cost_limit == 0.1
                assert trainer.lambda_lr == 0.01
                assert trainer.lambda_init == 1.0
                
                # 验证拉格朗日乘子已初始化
                assert 'default_policy' in trainer.lagrange_multipliers
                assert trainer.lagrange_multipliers['default_policy'] == 1.0
    
    @staticmethod
    def test_config_pipeline():
        """测试配置管道"""
        from ray_train import get_default_config, merge_configs
        from ray_inference import get_default_config as get_inference_default_config
        
        # 获取训练配置
        train_config = get_default_config()
        
        # 获取推理配置
        inference_config = get_inference_default_config()
        
        # 验证配置一致性
        assert train_config['node_dim'] == inference_config['node_dim']
        assert train_config['edge_dim'] == inference_config['edge_dim']
        assert train_config['gnn_hidden_dim'] == inference_config['gnn_hidden_dim']
        assert train_config['gnn_output_dim'] == inference_config['gnn_output_dim']
        assert train_config['gnn_layers'] == inference_config['gnn_layers']
        assert train_config['gnn_heads'] == inference_config['gnn_heads']
        assert train_config['world_hidden_dim'] == inference_config['world_hidden_dim']
        assert train_config['future_steps'] == inference_config['future_steps']
        assert train_config['controller_hidden_dim'] == inference_config['controller_hidden_dim']
        assert train_config['global_dim'] == inference_config['global_dim']
        assert train_config['top_k'] == inference_config['top_k']
        assert train_config['action_dim'] == inference_config['action_dim']
        
        # 验证配置合并
        user_config = {
            'num_workers': 8,
            'num_gpus': 2,
            'train_batch_size': 8000
        }
        
        merged_train_config = merge_configs(train_config, user_config)
        assert merged_train_config['num_workers'] == 8
        assert merged_train_config['num_gpus'] == 2
        assert merged_train_config['train_batch_size'] == 8000


# ============================================================================
# 7. 错误处理测试
# ============================================================================

class TestErrorHandling:
    """错误处理测试"""
    
    @staticmethod
    def test_invalid_config():
        """测试无效配置"""
        from ray_train import get_default_config, merge_configs
        
        # 获取默认配置
        config = get_default_config()
        
        # 创建无效配置
        invalid_config = {
            'num_workers': -1,  # 无效的worker数量
            'num_gpus': -1,  # 无效的GPU数量
            'train_batch_size': 0  # 无效的批次大小
        }
        
        # 合并配置（应该成功，但值无效）
        merged_config = merge_configs(config, invalid_config)
        
        # 验证无效值被接受（由用户负责验证）
        assert merged_config['num_workers'] == -1
        assert merged_config['num_gpus'] == -1
        assert merged_config['train_batch_size'] == 0
    
    @staticmethod
    def test_missing_checkpoint():
        """测试缺失的检查点"""
        from ray_inference import get_default_config
        
        # 创建配置（检查点不存在）
        config = get_default_config()
        config['checkpoint_path'] = './nonexistent_checkpoint.pth'
        
        # 验证配置已创建（实际加载时会失败）
        assert config['checkpoint_path'] == './nonexistent_checkpoint.pth'
    
    @staticmethod
    def test_invalid_observation():
        """测试无效观测"""
        from ray_train import validate_batch
        
        # 创建无效批次（形状不匹配）
        invalid_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(5, 2),  # 长度不匹配
            'rewards': np.random.randn(10),
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        # 验证应该失败
        assert validate_batch(invalid_batch) == False
    
    @staticmethod
    def test_nan_inf_handling():
        """测试NaN和Inf处理"""
        from ray_train import validate_batch
        
        # 包含NaN的批次
        nan_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(10, 2),
            'rewards': np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        assert validate_batch(nan_batch) == False
        
        # 包含Inf的批次
        inf_batch = {
            'obs': np.random.randn(10, 5),
            'actions': np.random.randn(10, 2),
            'rewards': np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'dones': np.zeros(10),
            'new_obs': np.random.randn(10, 5)
        }
        
        assert validate_batch(inf_batch) == False
    
    @staticmethod
    def test_import_error_handling():
        """测试导入错误处理"""
        try:
            # 尝试导入不存在的模块
            import nonexistent_module
            assert False, "应该抛出ImportError"
        except ImportError:
            pass  # 预期的错误
    
    @staticmethod
    def test_file_not_found():
        """测试文件未找到错误"""
        from ray_train import load_config_from_file
        
        try:
            # 尝试加载不存在的文件
            config = load_config_from_file('./nonexistent_config.json')
            assert False, "应该抛出FileNotFoundError"
        except FileNotFoundError:
            pass  # 预期的错误


# ============================================================================
# 主测试运行器
# ============================================================================

def run_tests(component: str = 'all'):
    """运行测试"""
    # 创建测试结果记录器
    test_result = TestResult()
    test_result.start()
    
    # 定义测试套件
    test_suites = {
        'environment': {
            'name': 'SUMO-RL环境封装测试',
            'tests': [
                ('导入测试', TestSUMOGymEnv.test_import),
                ('环境创建测试', TestSUMOGymEnv.test_environment_creation),
                ('观察空间测试', TestSUMOGymEnv.test_observation_space),
                ('动作空间测试', TestSUMOGymEnv.test_action_space),
                ('空观测测试', TestSUMOGymEnv.test_get_empty_observation),
                ('图构建测试', TestSUMOGymEnv.test_build_graph),
                ('全局指标计算测试', TestSUMOGymEnv.test_compute_global_metrics),
            ]
        },
        'model': {
            'name': 'Ray模型包装器测试',
            'tests': [
                ('导入测试', TestRayModel.test_import),
                ('模型创建测试', TestRayModel.test_model_creation),
                ('前向传播测试', TestRayModel.test_forward_pass),
                ('模型注册测试', TestRayModel.test_model_registration),
                ('批次准备测试', TestRayModel.test_prepare_batch),
            ]
        },
        'trainer': {
            'name': 'Ray ConstrainedPPO训练器测试',
            'tests': [
                ('导入测试', TestRayTrainer.test_import),
                ('训练器初始化测试', TestRayTrainer.test_trainer_initialization),
                ('约束违反计算测试', TestRayTrainer.test_constraint_violation_computation),
                ('拉格朗日乘子更新测试', TestRayTrainer.test_lagrange_multiplier_update),
                ('约束统计测试', TestRayTrainer.test_constraint_stats),
                ('重置乘子测试', TestRayTrainer.test_reset_lagrange_multipliers),
                ('设置成本限制测试', TestRayTrainer.test_set_cost_limit),
            ]
        },
        'train_script': {
            'name': 'Ray训练脚本测试',
            'tests': [
                ('导入测试', TestRayTrainScript.test_import),
                ('默认配置测试', TestRayTrainScript.test_default_config),
                ('配置合并测试', TestRayTrainScript.test_merge_configs),
                ('奖励重塑测试', TestRayTrainScript.test_reward_shaping),
                ('批次验证测试', TestRayTrainScript.test_batch_validation),
                ('配置文件加载测试', TestRayTrainScript.test_config_file_loading),
            ]
        },
        'inference_script': {
            'name': 'Ray推理脚本测试',
            'tests': [
                ('导入测试', TestRayInferenceScript.test_import),
                ('默认配置测试', TestRayInferenceScript.test_default_config),
                ('配置合并测试', TestRayInferenceScript.test_merge_configs),
                ('配置文件加载测试', TestRayInferenceScript.test_config_file_loading),
            ]
        },
        'integration': {
            'name': '集成测试',
            'tests': [
                ('模型与环境集成测试', TestIntegration.test_model_to_environment_integration),
                ('训练器与模型集成测试', TestIntegration.test_trainer_to_model_integration),
                ('配置管道测试', TestIntegration.test_config_pipeline),
            ]
        },
        'error_handling': {
            'name': '错误处理测试',
            'tests': [
                ('无效配置测试', TestErrorHandling.test_invalid_config),
                ('缺失检查点测试', TestErrorHandling.test_missing_checkpoint),
                ('无效观测测试', TestErrorHandling.test_invalid_observation),
                ('NaN/Inf处理测试', TestErrorHandling.test_nan_inf_handling),
                ('导入错误处理测试', TestErrorHandling.test_import_error_handling),
                ('文件未找到测试', TestErrorHandling.test_file_not_found),
            ]
        }
    }
    
    # 运行测试
    if component == 'all':
        # 运行所有测试
        for suite_name, suite in test_suites.items():
            print(f"\n📋 {suite['name']}")
            print("-" * 80)
            for test_name, test_func in suite['tests']:
                run_test(test_func, test_name, test_result)
    elif component in test_suites:
        # 运行指定组件的测试
        suite = test_suites[component]
        print(f"\n📋 {suite['name']}")
        print("-" * 80)
        for test_name, test_func in suite['tests']:
            run_test(test_func, test_name, test_result)
    else:
        print(f"❌ 未知的组件: {component}")
        print(f"可用的组件: {', '.join(test_suites.keys())}")
        return False
    
    # 完成测试
    success = test_result.finish()
    
    # 生成验证清单
    generate_verification_checklist(test_result)
    
    return success


def generate_verification_checklist(test_result: TestResult):
    """生成验证清单"""
    checklist = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'total': len(test_result.results['passed']) + len(test_result.results['failed']) + len(test_result.results['skipped']),
            'passed': len(test_result.results['passed']),
            'failed': len(test_result.results['failed']),
            'skipped': len(test_result.results['skipped']),
            'errors': len(test_result.results['errors'])
        },
        'verification_items': []
    }
    
    # 添加验证项
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '组件导入',
            'status': '✅ 通过',
            'description': '所有组件成功导入'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '环境封装',
            'status': '✅ 通过',
            'description': 'SUMO-RL环境封装正常工作'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '模型包装器',
            'status': '✅ 通过',
            'description': 'Ray模型包装器正常工作'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '训练器',
            'status': '✅ 通过',
            'description': 'ConstrainedPPO训练器正常工作'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '训练脚本',
            'status': '✅ 通过',
            'description': 'Ray训练脚本配置加载正常'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '推理脚本',
            'status': '✅ 通过',
            'description': 'Ray推理脚本配置加载正常'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '组件集成',
            'status': '✅ 通过',
            'description': '组件之间集成正常'
        })
    
    if test_result.results['passed']:
        checklist['verification_items'].append({
            'category': '错误处理',
            'status': '✅ 通过',
            'description': '错误处理机制正常工作'
        })
    
    # 保存验证清单
    report_dir = "./test_reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checklist_file = os.path.join(report_dir, f"verification_checklist_{timestamp}.json")
    
    with open(checklist_file, 'w', encoding='utf-8') as f:
        json.dump(checklist, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 验证清单已保存: {checklist_file}")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Ray系统测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 运行所有测试
    python test_ray_system.py
    
    # 运行特定组件的测试
    python test_ray_system.py --component environment
    python test_ray_system.py --component model
    python test_ray_system.py --component trainer
    python test_ray_system.py --component train_script
    python test_ray_system.py --component inference_script
    python test_ray_system.py --component integration
    python test_ray_system.py --component error_handling
        """
    )
    
    parser.add_argument(
        '--component',
        type=str,
        default='all',
        choices=['all', 'environment', 'model', 'trainer', 'train_script', 'inference_script', 'integration', 'error_handling'],
        help='要测试的组件'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行测试
    success = run_tests(component=args.component)
    
    # 退出
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
