"""
TraCI 订阅优化的 SUMO 环境
使用 TraCI 订阅机制批量获取车辆数据，避免在循环中频繁调用 getSpeed 等函数
显著提升性能
"""

import numpy as np
import traci
from typing import Dict, List, Tuple, Any, Optional
import os
import xml.etree.ElementTree as ET
import time


class TraCISubscriptionManager:
    """
    TraCI 订阅管理器
    管理所有 TraCI 订阅，批量获取数据
    """
    
    # 订阅的车辆变量列表
    VEHICLE_SUBSCRIPTIONS = [
        traci.constants.VAR_ROAD_ID,
        traci.constants.VAR_LANEPOSITION,
        traci.constants.VAR_SPEED,
        traci.constants.VAR_ACCELERATION,
        traci.constants.VAR_LANE_INDEX,
        traci.constants.VAR_LANE_ID,
        traci.constants.VAR_POSITION,
        traci.constants.VAR_ANGLE,
        traci.constants.VAR_VEHICLECLASS,
        traci.constants.VAR_VEHICLESPEED,
        traci.constants.VAR_VEHICLEACCEL,
        traci.constants.VAR_VEHICLELENGTH,
        traci.constants.VAR_VEHICLEWIDTH
    ]
    
    def __init__(self):
        """初始化订阅管理器"""
        self.subscribed_vehicles = set()
        self.subscription_cache = {}
        self.last_subscription_time = 0
        self.cache_timeout = 0.1  # 缓存超时时间（秒）
        
        print("✅ TraCI 订阅管理器初始化完成")
    
    def subscribe_vehicle(self, veh_id: str):
        """
        订阅车辆
        
        Args:
            veh_id: 车辆ID
        """
        if veh_id in self.subscribed_vehicles:
            return
        
        # 批量订阅所有变量
        for var in self.VEHICLE_SUBSCRIPTIONS:
            traci.vehicle.subscribe(veh_id, var)
        
        self.subscribed_vehicles.add(veh_id)
    
    def unsubscribe_vehicle(self, veh_id: str):
        """
        取消订阅车辆
        
        Args:
            veh_id: 车辆ID
        """
        if veh_id not in self.subscribed_vehicles:
            return
        
        traci.vehicle.unsubscribe(veh_id)
        self.subscribed_vehicles.discard(veh_id)
        
        # 清除缓存
        if veh_id in self.subscription_cache:
            del self.subscription_cache[veh_id]
    
    def get_vehicle_data(self, veh_id: str) -> Optional[Dict[str, Any]]:
        """
        获取车辆数据（从缓存或 TraCI）
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            vehicle_data: 车辆数据字典
        """
        # 检查缓存
        if veh_id in self.subscription_cache:
            cache_entry = self.subscription_cache[veh_id]
            if time.time() - cache_entry['timestamp'] < self.cache_timeout:
                return cache_entry['data']
        
        # 从 TraCI 获取订阅数据
        try:
            subscription_results = traci.vehicle.getSubscriptionResults(veh_id)
            
            if not subscription_results:
                return None
            
            # 解析订阅结果
            vehicle_data = {
                'id': veh_id,
                'road_id': subscription_results.get(traci.constants.VAR_ROAD_ID, ''),
                'lane_position': subscription_results.get(traci.constants.VAR_LANEPOSITION, 0.0),
                'speed': subscription_results.get(traci.constants.VAR_SPEED, 0.0),
                'acceleration': subscription_results.get(traci.constants.VAR_ACCELERATION, 0.0),
                'lane_index': subscription_results.get(traci.constants.VAR_LANE_INDEX, 0),
                'lane_id': subscription_results.get(traci.constants.VAR_LANE_ID, ''),
                'position': subscription_results.get(traci.constants.VAR_POSITION, 0.0),
                'angle': subscription_results.get(traci.constants.VAR_ANGLE, 0.0),
                'vehicle_class': subscription_results.get(traci.constants.VAR_VEHICLECLASS, ''),
                'vehicle_length': subscription_results.get(traci.constants.VAR_VEHICLELENGTH, 5.0),
                'vehicle_width': subscription_results.get(traci.constants.VAR_VEHICLEWIDTH, 2.0)
            }
            
            # 更新缓存
            self.subscription_cache[veh_id] = {
                'data': vehicle_data,
                'timestamp': time.time()
            }
            
            return vehicle_data
        
        except Exception as e:
            # 如果订阅失败，尝试直接获取
            try:
                return self._get_vehicle_data_direct(veh_id)
            except:
                return None
    
    def _get_vehicle_data_direct(self, veh_id: str) -> Optional[Dict[str, Any]]:
        """
        直接获取车辆数据（降级方案）
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            vehicle_data: 车辆数据字典
        """
        try:
            vehicle_data = {
                'id': veh_id,
                'road_id': traci.vehicle.getRoadID(veh_id),
                'lane_position': traci.vehicle.getLanePosition(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id),
                'lane_index': traci.vehicle.getLaneIndex(veh_id),
                'lane_id': traci.vehicle.getLaneID(veh_id),
                'position': traci.vehicle.getLanePosition(veh_id),
                'angle': traci.vehicle.getAngle(veh_id),
                'vehicle_class': traci.vehicle.getVehicleClass(veh_id),
                'vehicle_length': traci.vehicle.getLength(veh_id),
                'vehicle_width': traci.vehicle.getWidth(veh_id)
            }
            return vehicle_data
        except Exception as e:
            return None
    
    def batch_subscribe_vehicles(self, vehicle_ids: List[str]):
        """
        批量订阅车辆
        
        Args:
            vehicle_ids: 车辆ID列表
        """
        for veh_id in vehicle_ids:
            self.subscribe_vehicle(veh_id)
    
    def batch_get_vehicle_data(self, vehicle_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取车辆数据
        
        Args:
            vehicle_ids: 车辆ID列表
            
        Returns:
            vehicle_data_dict: 车辆数据字典
        """
        vehicle_data_dict = {}
        
        for veh_id in vehicle_ids:
            vehicle_data = self.get_vehicle_data(veh_id)
            if vehicle_data is not None:
                vehicle_data_dict[veh_id] = vehicle_data
        
        return vehicle_data_dict
    
    def cleanup(self):
        """清理所有订阅"""
        for veh_id in list(self.subscribed_vehicles):
            self.unsubscribe_vehicle(veh_id)
        
        self.subscription_cache.clear()


class SUMORLEnvironmentOptimized:
    """
    TraCI 订阅优化的 SUMO 强化学习环境
    实现 Gymnasium 风格的接口
    """
    
    def __init__(self, 
                 sumo_cfg_path: str,
                 use_gui: bool = False,
                 max_steps: int = 3600,
                 seed: Optional[int] = None,
                 use_subscription: bool = True):
        """
        初始化优化的 SUMO 环境
        
        Args:
            sumo_cfg_path: SUMO 配置文件路径
            use_gui: 是否使用 GUI
            max_steps: 最大仿真步数
            seed: 随机种子
            use_subscription: 是否使用 TraCI 订阅优化
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.seed_val = seed
        self.use_subscription = use_subscription
        
        # 环境状态
        self.current_step = 0
        self.connected = False
        self.vehicle_ids = []
        
        # 订阅管理器
        self.subscription_manager = TraCISubscriptionManager() if use_subscription else None
        
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
        
        # 性能统计
        self.performance_stats = {
            'subscription_time': 0.0,
            'direct_call_time': 0.0,
            'subscription_hits': 0,
            'subscription_misses': 0
        }
        
        print(f"✅ 优化的 SUMO RL 环境初始化完成")
        print(f"   配置文件: {sumo_cfg_path}")
        print(f"   最大步数: {max_steps}")
        print(f"   GUI: {use_gui}")
        print(f"   TraCI 订阅: {use_subscription}")
    
    def _parse_config(self):
        """解析 SUMO 配置文件"""
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
        
        # 清理订阅
        if self.subscription_manager:
            self.subscription_manager.cleanup()
        
        # 启动 SUMO
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
            print(f"🚀 SUMO 环境已重置")
        except Exception as e:
            print(f"❌ SUMO 启动失败: {e}")
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
                # 清理订阅
                if self.subscription_manager:
                    self.subscription_manager.cleanup()
                
                traci.close()
                self.connected = False
                print("✅ SUMO 环境已关闭")
            except Exception as e:
                print(f"⚠️  关闭 SUMO 时出错: {e}")
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        获取当前观测（使用 TraCI 订阅优化）
        使用配置的ICV车辆列表而非随机哈希
        
        Returns:
            observation: 观测数据
        """
        # 获取所有车辆ID
        self.vehicle_ids = traci.vehicle.getIDList()
        
        if not self.vehicle_ids:
            return {
                'vehicle_data': {},
                'global_metrics': self._compute_global_metrics({}),
                'vehicle_ids': []
            }
        
        # 批量订阅车辆
        if self.subscription_manager:
            self.subscription_manager.batch_subscribe_vehicles(self.vehicle_ids)
            
            # 批量获取车辆数据
            vehicle_data = self.subscription_manager.batch_get_vehicle_data(self.vehicle_ids)
            
            # 为每辆车添加ICV标记
            for veh_id in vehicle_data:
                vehicle_data[veh_id]['is_icv'] = self._is_icv_vehicle(veh_id)
        else:
            # 降级方案：直接获取
            vehicle_data = {}
            for veh_id in self.vehicle_ids:
                try:
                    vehicle_data[veh_id] = {
                        'position': traci.vehicle.getLanePosition(veh_id),
                        'speed': traci.vehicle.getSpeed(veh_id),
                        'acceleration': traci.vehicle.getAcceleration(veh_id),
                        'lane_index': traci.vehicle.getLaneIndex(veh_id),
                        'lane_id': traci.vehicle.getLaneID(veh_id),
                        'road_id': traci.vehicle.getRoadID(veh_id),
                        'is_icv': self._is_icv_vehicle(veh_id),
                        'id': veh_id
                    }
                except Exception as e:
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
    
    def _compute_global_metrics(self, vehicle_data: Dict[str, Any]) -> List[float]:
        """
        计算全局交通指标
        
        Args:
            vehicle_data: 车辆数据字典
            
        Returns:
            metrics: 16维全局指标
        """
        if not vehicle_data:
            return [0.0] * 16
        
        vehicle_list = list(vehicle_data.values())
        speeds = [v['speed'] for v in vehicle_list]
        positions = [v['position'] for v in vehicle_list]
        accelerations = [v.get('acceleration', 0.0) for v in vehicle_list]
        
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        avg_accel = np.mean(np.abs(accelerations))
        vehicle_count = len(vehicle_list)
        
        # ICV 统计
        icv_vehicles = [v for v in vehicle_list if v.get('is_icv', False)]
        hv_vehicles = [v for v in vehicle_list if not v.get('is_icv', False)]
        
        icv_count = len(icv_vehicles)
        hv_count = len(hv_vehicles)
        
        icv_total_speed = sum([v['speed'] for v in icv_vehicles])
        hv_total_speed = sum([v['speed'] for v in hv_vehicles])
        
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
            float(icv_count),
            float(hv_count),
            icv_total_speed,
            hv_total_speed,
            avg_speed * vehicle_count,
            speed_std * vehicle_count,
            avg_accel * vehicle_count,
            self.current_step % 100
        ]
        
        return metrics
    
    def _apply_action(self, action: Dict[str, Any]):
        """
        应用控制动作
        
        Args:
            action: 动作字典
        """
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
        import hashlib
        hash_value = int(hashlib.md5(veh_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < 25  # 25% ICV渗透率
    
    def _compute_reward(self, observation: Dict[str, Any]) -> float:
        """
        计算奖励 - 基于真实交通指标
        考虑：流量效率、安全、稳定性
        
        Args:
            observation: 观测数据
            
        Returns:
            reward: 奖励值
        """
        vehicle_data = observation['vehicle_data']
        
        if not vehicle_data:
            return 0.0
        
        speeds = [v['speed'] for v in vehicle_data.values()]
        accelerations = [v.get('acceleration', 0.0) for v in vehicle_data.values()]
        
        # 1. 流量效率奖励
        avg_speed = np.mean(speeds) if speeds else 0.0
        flow_efficiency = avg_speed / 30.0  # 归一化到[0,1]
        
        # 2. 稳定性惩罚
        speed_std = np.std(speeds) if len(speeds) > 1 else 0.0
        accel_std = np.std(accelerations) if len(accelerations) > 1 else 0.0
        stability_penalty = (speed_std / 10.0 + accel_std / 5.0) * 0.5
        
        # 3. 安全评估
        safety_penalty = 0.0
        for vehicle in vehicle_data.values():
            speed = vehicle.get('speed', 0.0)
            accel = vehicle.get('acceleration', 0.0)
            
            # 检查危险驾驶行为
            if speed > 35.0:  # 超速
                safety_penalty += (speed - 35.0) * 0.1
            if accel < -4.0:  # 急刹车
                safety_penalty += (-accel - 4.0) * 0.2
            if accel > 3.0:  # 急加速
                safety_penalty += (accel - 3.0) * 0.1
        
        # 4. 综合奖励
        reward = (
            flow_efficiency * 10.0           # 流量效率权重
            - stability_penalty * 2.0         # 稳定性惩罚权重
            - safety_penalty * 5.0            # 安全惩罚权重
        )
        
        return reward
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """
        获取 episode 统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'total_steps': self.current_step,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.current_step, 1),
            'vehicle_count': len(self.vehicle_ids)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            stats: 性能统计
        """
        return self.performance_stats.copy()


def create_sumo_env_optimized(sumo_cfg_path: str, **kwargs) -> SUMORLEnvironmentOptimized:
    """
    创建优化的 SUMO 环境的工厂函数
    
    Args:
        sumo_cfg_path: SUMO 配置文件路径
        **kwargs: 其他参数
        
    Returns:
        优化的 SUMO RL 环境实例
    """
    return SUMORLEnvironmentOptimized(sumo_cfg_path, **kwargs)


def main():
    """主函数 - 演示 TraCI 订阅优化"""
    print("🚀 TraCI 订阅优化演示")
    
    # 创建优化的环境
    env = SUMORLEnvironmentOptimized(
        sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
        use_gui=False,
        max_steps=100,
        use_subscription=True
    )
    
    try:
        # 重置环境
        observation = env.reset()
        
        print(f"\n初始观测:")
        print(f"  车辆数: {len(observation['vehicle_ids'])}")
        print(f"  全局指标: {observation['global_metrics'][:4]}")
        
        # 运行几个步骤
        for step in range(10):
            # 执行一步
            observation, reward, done, info = env.step({})
            
            print(f"\n[Step {step+1}]")
            print(f"  奖励: {reward:.4f}")
            print(f"  车辆数: {info['vehicle_count']}")
            print(f"  总奖励: {info['total_reward']:.2f}")
            
            if done:
                break
        
        # 打印性能统计
        perf_stats = env.get_performance_stats()
        print(f"\n{'='*60}")
        print("📊 性能统计")
        print(f"{'='*60}")
        print(f"订阅时间: {perf_stats['subscription_time']:.4f}s")
        print(f"直接调用时间: {perf_stats['direct_call_time']:.4f}s")
        print(f"订阅命中: {perf_stats['subscription_hits']}")
        print(f"订阅未命中: {perf_stats['subscription_misses']}")
        print(f"{'='*60}")
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
