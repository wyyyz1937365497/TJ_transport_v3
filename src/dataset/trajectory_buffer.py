import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryBuffer:
    """
    交通轨迹数据缓冲区
    用于存储和管理交通仿真中的轨迹数据
    """
    
    def __init__(self, max_size=100000, state_dim=8, action_dim=1):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化缓冲区
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, next_state, reward=0, done=False):
        """
        添加经验样本
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        随机采样一批数据
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
    
    def get_all_data(self):
        """
        获取所有数据
        """
        return {
            'states': torch.FloatTensor(self.states[:self.size]),
            'actions': torch.FloatTensor(self.actions[:self.size]),
            'next_states': torch.FloatTensor(self.next_states[:self.size]),
            'rewards': torch.FloatTensor(self.rewards[:self.size]),
            'dones': torch.FloatTensor(self.dones[:self.size])
        }
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'next_state': self.next_states[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx]
        }


class WorldModelDataset(Dataset):
    """世界模型训练数据集"""
    
    def __init__(self, buffer_file: str, future_steps: int = 5):
        self.future_steps = future_steps
        # 加载缓冲区数据
        self.buffer = TrajectoryBuffer()
        # 这里应该加载保存的缓冲区数据
        # 实际实现中，需要从文件加载数据
        pass
    
    def __len__(self):
        return len(self.buffer) - self.future_steps
    
    def __getitem__(self, idx):
        # 获取当前状态
        item = self.buffer[idx]
        current_state = item['state']
        current_action = item['action']
        
        # 获取未来状态序列
        future_states = []
        for i in range(self.future_steps):
            future_idx = idx + i + 1
            if future_idx < len(self.buffer):
                future_item = self.buffer[future_idx]
                future_states.append(future_item['next_state'])
            else:
                # 如果未来状态不够，用最后一个状态填充
                future_states.append(current_state)
        
        future_states = np.stack(future_states)
        
        return {
            'state': torch.FloatTensor(current_state),
            'action': torch.FloatTensor(current_action),
            'future_states': torch.FloatTensor(future_states)
        }