import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import traci
import time
import os

class WorldModelDataCollector:
    """
    世界模型预训练数据收集器
    支持随机策略和IDM策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_buffer_size = config.get("max_buffer_size", 100000)
        self.state_dim = config.get("state_dim", 8)
        self.action_dim = config.get("action_dim", 1)
        self.future_steps = config.get("future_steps", 5)
        
        # 初始化缓冲区
        self.buffer = {
            'states': np.zeros((self.max_buffer_size, self.state_dim), dtype=np.float32),
            'actions': np.zeros((self.max_buffer_size, self.action_dim), dtype=np.float32),
            'next_states': np.zeros((self.max_buffer_size, self.state_dim), dtype=np.float32),
            'rewards': np.zeros((self.max_buffer_size, 1), dtype=np.float32),
            'dones': np.zeros((self.max_buffer_size, 1), dtype=np.float32),
            'vehicle_ids': [''] * self.max_buffer_size,
            'positions': np.zeros((self.max_buffer_size, 2), dtype=np.float32),
            'size': 0,
            'ptr': 0
        }
        
        # IDM参数
        self.idm_config = {
            'desired_speed': config.get("idm_desired_speed", 30.0),
            'safe_time_headway': config.get("idm_safe_time_headway", 1.5),
            'max_acceleration': config.get("idm_max_acceleration", 2.0),
            'comfortable_braking': config.get("idm_comfortable_braking", 3.0),
            'exponent': config.get("idm_exponent", 4.0)
        }
    
    def collect_data(self, num_episodes: int = 100, strategy: str = "mixed"):
        """
        收集训练数据
        strategy: "random", "idm", "mixed"
        """
        from src.env.custom_sumo_env import CustomSumoEnv
        
        env = CustomSumoEnv(self.config)
        
        for episode in range(num_episodes):
            print(f"Collecting data - Episode {episode + 1}/{num_episodes}")
            obs, info = env.reset()
            
            episode_data = []
            done = False
            
            while not done:
                # 选择收集策略
                if strategy == "random":
                    action = self._random_action(env)
                elif strategy == "idm":
                    action = self._idm_action(env, obs)
                else:  # mixed
                    if np.random.random() < 0.5:
                        action = self._random_action(env)
                    else:
                        action = self._idm_action(env, obs)
                
                next_obs, reward, done, truncated, info = env.step(action)
                
                # 提取状态特征
                state = self._extract_state_features(obs, env)
                next_state = self._extract_state_features(next_obs, env)
                
                # 存储数据
                self._add_to_buffer(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    done=done,
                    vehicle_ids=info.get("controlled_vehicles", []),
                    positions=info.get("vehicle_positions", [])
                )
                
                obs = next_obs
            
            # 保存检查点
            if episode % 10 == 0:
                self.save_buffer(f"buffers/data_collector_checkpoint_{episode}.npz")
        
        env.close()
        print(f"Data collection completed. Total samples: {self.buffer['size']}")
    
    def _random_action(self, env):
        """生成随机动作"""
        num_controlled = env.num_controlled
        return np.random.uniform(
            env.action_space.low, 
            env.action_space.high, 
            size=(num_controlled, 1)
        )
    
    def _idm_action(self, env, obs):
        """IDM跟驰模型生成动作"""
        vehicle_states = obs["vehicle_states"]
        num_controlled = env.num_controlled
        
        actions = np.zeros((num_controlled, 1))
        
        for i in range(min(num_controlled, len(vehicle_states["ids"]))):
            vid = vehicle_states["ids"][i]
            if vid in env._vehicle_data:
                vehicle = env._vehicle_data[vid]
                leader = self._find_leader(vehicle, env._vehicle_data)
                
                if leader:
                    # IDM公式
                    speed = vehicle["speed"]
                    leader_speed = leader["speed"]
                    gap = self._calculate_gap(vehicle, leader)
                    
                    desired_speed = self.idm_config['desired_speed']
                    safe_headway = self.idm_config['safe_time_headway']
                    max_accel = self.idm_config['max_acceleration']
                    comfortable_braking = self.idm_config['comfortable_braking']
                    exponent = self.idm_config['exponent']
                    
                    # 计算自由流加速度
                    free_flow_accel = max_accel * (1 - (speed / desired_speed) ** exponent)
                    
                    # 计算跟驰加速度
                    if gap > 0:
                        speed_diff = speed - leader_speed
                        braking_term = (speed * speed_diff) / (2 * np.sqrt(max_accel * comfortable_braking))
                        follow_accel = -max_accel * (braking_term / gap) ** 2
                    else:
                        follow_accel = -comfortable_braking
                    
                    # 综合加速度
                    accel = free_flow_accel + follow_accel
                    actions[i, 0] = np.clip(accel, env.action_space.low[0], env.action_space.high[0])
        
        return actions
    
    def _find_leader(self, vehicle, vehicle_data):
        """查找前车"""
        # 简化的前车查找逻辑
        # 在实际实现中，需要根据路网拓扑和车辆位置确定前车
        return None
    
    def _calculate_gap(self, vehicle, leader):
        """计算与前车的距离"""
        # 简化的距离计算
        # 在实际实现中，需要根据路网拓扑精确计算距离
        return 50.0
    
    def _extract_state_features(self, obs, env):
        """从观测中提取状态特征"""
        features = np.zeros(self.state_dim)
        
        if "gnn_embedding" in obs:
            # 使用GNN嵌入作为状态特征
            gnn_emb = obs["gnn_embedding"]
            features[:min(len(gnn_emb), self.state_dim)] = gnn_emb[:self.state_dim]
        
        return features
    
    def _add_to_buffer(self, **kwargs):
        """添加数据到缓冲区"""
        idx = self.buffer['ptr']
        
        self.buffer['states'][idx] = kwargs['state']
        self.buffer['actions'][idx] = kwargs['action'].flatten()[:self.action_dim]
        self.buffer['next_states'][idx] = kwargs['next_state']
        self.buffer['rewards'][idx] = kwargs['reward']
        self.buffer['dones'][idx] = float(kwargs['done'])
        self.buffer['vehicle_ids'][idx] = kwargs.get('vehicle_ids', [''])[0] if kwargs.get('vehicle_ids') else ''
        self.buffer['positions'][idx] = kwargs.get('positions', [0, 0])[0] if kwargs.get('positions') else [0, 0]
        
        self.buffer['ptr'] = (self.buffer['ptr'] + 1) % self.max_buffer_size
        self.buffer['size'] = min(self.buffer['size'] + 1, self.max_buffer_size)
    
    def save_buffer(self, filename: str):
        """保存缓冲区到文件"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        np.savez(
            filename,
            states=self.buffer['states'][:self.buffer['size']],
            actions=self.buffer['actions'][:self.buffer['size']],
            next_states=self.buffer['next_states'][:self.buffer['size']],
            rewards=self.buffer['rewards'][:self.buffer['size']],
            dones=self.buffer['dones'][:self.buffer['size']],
            vehicle_ids=self.buffer['vehicle_ids'][:self.buffer['size']],
            positions=self.buffer['positions'][:self.buffer['size']],
            size=self.buffer['size']
        )
        print(f"Buffer saved to {filename}")
    
    def load_buffer(self, filename: str):
        """从文件加载缓冲区"""
        if not os.path.exists(filename):
            print(f"Buffer file {filename} not found. Starting with empty buffer.")
            return
        
        data = np.load(filename, allow_pickle=True)
        
        size = min(data['size'], self.max_buffer_size)
        self.buffer['states'][:size] = data['states'][:size]
        self.buffer['actions'][:size] = data['actions'][:size]
        self.buffer['next_states'][:size] = data['next_states'][:size]
        self.buffer['rewards'][:size] = data['rewards'][:size]
        self.buffer['dones'][:size] = data['dones'][:size]
        self.buffer['vehicle_ids'][:size] = data['vehicle_ids'][:size].tolist()
        self.buffer['positions'][:size] = data['positions'][:size]
        self.buffer['size'] = size
        self.buffer['ptr'] = size % self.max_buffer_size
        
        print(f"Buffer loaded from {filename}. Total samples: {size}")