"""
基于sumo-rl框架的SUMO环境封装
提供标准化的Gymnasium接口用于强化学习训练
"""

import numpy as np
import traci
from typing import Dict, List, Tuple, Any, Optional
import os
import xml.etree.ElementTree as ET


class SUMORLEnvironment:
    """
    SUMO强化学习环境
    实现Gymnasium风格的接口
    """
    
    def __init__(self, 
                 sumo_cfg_path: str,
                 use_gui: bool = False,
                 max_steps: int = 3600,
                 seed: Optional[int] = None):
        """
        初始化SUMO环境
        
        Args:
            sumo_cfg_path: SUMO配置文件路径
            use_gui: 是否使用GUI
            max_steps: 最大仿真步数
            seed: 随机种子
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.seed_val = seed
        
        # 环境状态
        self.current_step = 0
        self.connected = False
        self.vehicle_ids = []
        
        # 解析配置
        self.net_file = None
        self.routes_file = None
        self.step_length = 1.0
        self._parse_config()
        
        # 动作和观察空间
        self.action_space_dim = 2  # [加速度, 换道概率]
        self.observation_space_dim = None  # 动态计算
        
        # 统计信息
        self.total_reward = 0.0
        self.episode_rewards = []
        
        print(f"✅ SUMO RL环境初始化完成")
        print(f"   配置文件: {sumo_cfg_path}")
        print(f"   最大步数: {max_steps}")
        print(f"   GUI: {use_gui}")
    
    def _parse_config(self):
        """解析SUMO配置文件"""
        try:
            tree = ET.parse(self.sumo_cfg_path)
            root = tree.getroot()
            config_dir = os.path.dirname(self.sumo_cfg_path)
            
            # 获取路网和路径文件
            for input_elem in root.findall('.//input'):
                net_file = input_elem.find('net-file')
                if net_file is not None:
                    net_file_path = net_file.get('value')
                    if not os.path.isabs(net_file_path):
                        net_file_path = os.path.join(config_dir, net_file_path)
                    self.net_file = net_file_path
                
                route_files = input_elem.find('route-files')
                if route_files is not None:
                    route_file_path = route_files.get('value')
                    if not os.path.isabs(route_file_path):
                        route_file_path = os.path.join(config_dir, route_file_path)
                    self.routes_file = route_file_path
            
            # 获取时间步长
            time_step = root.find('.//step-length')
            if time_step is not None:
                self.step_length = float(time_step.get('value', 1.0))
            
            print(f"   网络文件: {self.net_file}")
            print(f"   路径文件: {self.routes_file}")
            print(f"   时间步长: {self.step_length}s")
            
        except Exception as e:
            print(f"⚠️  配置解析失败: {e}")
    
    def reset(self) -> Dict[str, Any]:
        """
        重置环境
        
        Returns:
            observation: 初始观测
        """
        # 关闭现有连接
        if self.connected:
            try:
                traci.close()
            except:
                pass
        
        # 重置状态
        self.current_step = 0
        self.total_reward = 0.0
        self.vehicle_ids = []
        
        # 启动SUMO
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]
        
        if self.seed_val is not None:
            sumo_cmd.extend(["--seed", str(self.seed_val)])
        
        try:
            traci.start(sumo_cmd)
            self.connected = True
            print(f"🚀 SUMO环境已重置")
        except Exception as e:
            print(f"❌ SUMO启动失败: {e}")
            raise
        
        # 执行第一步以初始化车辆
        traci.simulationStep()
        self.current_step += 1
        
        # 获取初始观测
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步仿真
        
        Args:
            action: 动作字典
                - selected_vehicle_ids: 选中的车辆ID列表
                - safe_actions: 安全动作 [K, 2]
        
        Returns:
            observation: 观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 应用控制动作
        self._apply_action(action)
        
        # 执行仿真步
        traci.simulationStep()
        self.current_step += 1
        
        # 获取观测
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._compute_reward(observation)
        self.total_reward += reward
        
        # 检查是否结束
        done = (self.current_step >= self.max_steps) or \
                (traci.simulation.getMinExpectedNumber() <= 0 and self.current_step > 100)
        
        # 额外信息
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'vehicle_count': len(self.vehicle_ids)
        }
        
        return observation, reward, done, info
    
    def close(self):
        """关闭环境"""
        if self.connected:
            try:
                traci.close()
                self.connected = False
                print("✅ SUMO环境已关闭")
            except Exception as e:
                print(f"⚠️  关闭SUMO时出错: {e}")
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        获取当前观测
        使用配置的ICV车辆列表而非随机哈希
        """
        # 获取所有车辆ID
        self.vehicle_ids = traci.vehicle.getIDList()
        
        if not self.vehicle_ids:
            return {
                'vehicle_data': {},
                'global_metrics': self._compute_global_metrics({}),
                'vehicle_ids': []
            }
        
        # 收集车辆数据
        vehicle_data = {}
        for veh_id in self.vehicle_ids:
            try:
                # 使用配置的ICV列表或基于车辆类型的判断
                # 在实际应用中，应该从配置文件或车辆类型中读取
                is_icv = self._is_icv_vehicle(veh_id)
                
                vehicle_data[veh_id] = {
                    'position': traci.vehicle.getLanePosition(veh_id),
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'acceleration': traci.vehicle.getAcceleration(veh_id),
                    'lane_index': traci.vehicle.getLaneIndex(veh_id),
                    'lane_id': traci.vehicle.getLaneID(veh_id),
                    'road_id': traci.vehicle.getRoadID(veh_id),
                    'is_icv': is_icv,
                    'id': veh_id
                }
            except Exception as e:
                # 记录错误但继续处理其他车辆
                import logging
                logging.warning(f"获取车辆 {veh_id} 数据失败: {e}")
                continue
        
        # 计算全局指标
        global_metrics = self._compute_global_metrics(vehicle_data)
        
        observation = {
            'vehicle_data': vehicle_data,
            'global_metrics': global_metrics,
            'vehicle_ids': list(vehicle_data.keys())
        }
        
        return observation
    
    def _is_icv_vehicle(self, veh_id: str) -> bool:
        """
        判断车辆是否为ICV（智能网联车）
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            is_icv: 是否为ICV
        """
        # 方法1: 从车辆类型判断（推荐）
        try:
            vehicle_class = traci.vehicle.getVehicleClass(veh_id)
            if vehicle_class == "custom1" or vehicle_class == "emergency":
                return True
        except:
            pass
        
        # 方法2: 从车辆类型ID判断
        try:
            vtype = traci.vehicle.getTypeID(veh_id)
            if "icv" in vtype.lower() or "autonomous" in vtype.lower():
                return True
        except:
            pass
        
        # 方法3: 使用确定性哈希（用于演示，生产环境应使用配置）
        # 注意：这种方法在真实应用中不推荐，应该使用明确的配置
        import hashlib
        hash_value = int(hashlib.md5(veh_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < 25  # 25% ICV渗透率
        
        # 计算全局指标
        global_metrics = self._compute_global_metrics(vehicle_data)
        
        observation = {
            'vehicle_data': vehicle_data,
            'global_metrics': global_metrics,
            'vehicle_ids': list(vehicle_data.keys())
        }
        
        return observation
    
    def _compute_global_metrics(self, vehicle_data: Dict[str, Any]) -> List[float]:
        """计算全局交通指标"""
        if not vehicle_data:
            return [0.0] * 16
        
        speeds = [v['speed'] for v in vehicle_data.values()]
        positions = [v['position'] for v in vehicle_data.values()]
        accelerations = [v['acceleration'] for v in vehicle_data.values()]
        
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        avg_accel = np.mean(np.abs(accelerations))
        vehicle_count = len(vehicle_data)
        
        # 16维全局指标
        metrics = [
            avg_speed,
            speed_std,
            avg_accel,
            float(vehicle_count),
            self.current_step * self.step_length,  # 当前时间
            min(positions) if positions else 0.0,
            max(positions) if positions else 0.0,
            np.mean(positions) if positions else 0.0,
            len([v for v in vehicle_data.values() if v.get('is_icv', False)]),
            vehicle_count - len([v for v in vehicle_data.values() if v.get('is_icv', False)]),
            np.sum([v['speed'] for v in vehicle_data.values() if v.get('is_icv', False)]),
            np.sum([v['speed'] for v in vehicle_data.values() if not v.get('is_icv', False)]),
            avg_speed * vehicle_count,
            speed_std * vehicle_count,
            avg_accel * vehicle_count,
            self.current_step % 100
        ]
        
        return metrics
    
    def _apply_action(self, action: Dict[str, Any]):
        """应用控制动作"""
        if 'selected_vehicle_ids' not in action or 'safe_actions' not in action:
            return
        
        selected_ids = action['selected_vehicle_ids']
        safe_actions = action['safe_actions']
        
        for i, veh_id in enumerate(selected_ids):
            if i >= len(safe_actions):
                continue
            
            try:
                action_vec = safe_actions[i]
                accel_action = action_vec[0].item() * 5.0  # [-1,1] -> [-5,5]
                
                current_speed = traci.vehicle.getSpeed(veh_id)
                new_speed = max(0.0, current_speed + accel_action * 0.1)
                
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, new_speed)
                
            except Exception as e:
                continue
    
    def _compute_reward(self, observation: Dict[str, Any]) -> float:
        """计算奖励"""
        vehicle_data = observation['vehicle_data']
        
        if not vehicle_data:
            return 0.0
        
        speeds = [v['speed'] for v in vehicle_data.values()]
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        
        # 奖励函数：速度奖励 - 不稳定惩罚
        reward = avg_speed * 0.1 - speed_std * 0.5
        
        return reward
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """获取episode统计信息"""
        return {
            'total_steps': self.current_step,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.current_step, 1),
            'vehicle_count': len(self.vehicle_ids)
        }


def create_sumo_env(sumo_cfg_path: str, **kwargs) -> SUMORLEnvironment:
    """
    创建SUMO环境的工厂函数
    
    Args:
        sumo_cfg_path: SUMO配置文件路径
        **kwargs: 其他参数
    
    Returns:
        SUMO RL环境实例
    """
    return SUMORLEnvironment(sumo_cfg_path, **kwargs)
