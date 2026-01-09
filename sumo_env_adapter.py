import gymnasium as gym
import numpy as np
import torch
import traci
import sumolib
from gymnasium import spaces
from typing import Dict, Any, List, Optional
from neural_traffic_controller import NeuralTrafficController


class SumoEnvironmentAdapter(gym.Env):
    """
    SUMO环境适配器，用于与神经网络控制器配合
    """
    
    def __init__(self, sumo_cfg_path: str, max_steps: int = 3600, step_interval: int = 5, port=None):
        super(SumoEnvironmentAdapter, self).__init__()
        
        self.sumo_cfg_path = sumo_cfg_path
        self.max_steps = max_steps
        self.step_interval = step_interval  # 每隔几个仿真步执行一次控制
        self.current_step = 0
        self.port = port
        
        # 初始化SUMO
        self._initialize_sumo()
        
        # 初始化神经控制器
        self.controller = NeuralTrafficController()
        
        # 定义动作空间和观测空间
        # 动作空间：对选定车辆的加速和换道操作
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10, 2), dtype=np.float32  # 最多10辆车，每辆车2个动作
        )
        
        # 观测空间：车辆状态信息
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(100, 9),  # 最多100辆车，每辆车9个特征
                dtype=np.float32
            ),
            'edge_indices': spaces.Box(
                low=0, high=100, 
                shape=(2, 200),  # 最多200条边
                dtype=np.int64
            ),
            'global_metrics': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(16,),  # 16个全局指标
                dtype=np.float32
            )
        })
        
        print("✅ SUMO环境适配器初始化完成!")

    def _initialize_sumo(self):
        """初始化SUMO仿真环境"""
        sumo_binary = "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path, "--no-warnings", "true"]
        
        # 如果指定了端口，则添加远程端口参数
        if self.port is not None:
            sumo_cmd.extend(["--remote-port", str(self.port)])
        
        traci.start(sumo_cmd)
        print(f"✅ SUMO仿真环境启动成功! 端口: {self.port if self.port else 'default'}")

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            super().reset(seed=seed)
        
        # 关闭当前仿真
        traci.close()
        
        # 重新初始化
        self._initialize_sumo()
        self.current_step = 0
        
        # 获取初始观测
        obs = self._get_observation()
        info = {"episode_step": self.current_step}
        
        return obs, info

    def step(self, action=None):
        """执行一步仿真"""
        # 执行SUMO仿真一步
        traci.simulationStep()
        self.current_step += 1
        
        # 根据步间隔决定是否应用神经网络控制
        if self.current_step % self.step_interval == 0:
            # 收集当前车辆数据
            vehicle_data = self._collect_current_vehicle_data()
            
            # 应用神经网络控制
            if vehicle_data:
                try:
                    control_results = self.controller.apply_control(vehicle_data, self.current_step)
                    
                    # 更新统计信息
                    if hasattr(self.controller, 'total_interventions'):
                        self.controller.total_interventions += control_results.get('safety_interventions', 0)
                    if hasattr(self.controller, 'total_emergency_interventions'):
                        self.controller.total_emergency_interventions += control_results.get('emergency_interventions', 0)
                    if hasattr(self.controller, 'total_controlled_vehicles'):
                        self.controller.total_controlled_vehicles += len(control_results.get('controlled_vehicles', []))
                        
                except Exception as e:
                    print(f"⚠️ 神经控制执行错误: {e}")
        
        # 获取新的观测
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 判断是否结束
        terminated = self.current_step >= self.max_steps
        truncated = traci.simulation.getMinExpectedNumber() <= 0
        
        # 获取额外信息
        info = {
            "episode_step": self.current_step,
            "active_vehicles": len(traci.vehicle.getIDList()),
            "cumulative_arrived": traci.simulation.getArrivedNumber(),
            "cumulative_departed": traci.simulation.getDepartedNumber()
        }
        
        return observation, reward, terminated, truncated, info

    def _collect_current_vehicle_data(self) -> Dict[str, Any]:
        """收集当前车辆数据"""
        vehicle_data = {}
        vehicle_ids = traci.vehicle.getIDList()

        for veh_id in vehicle_ids:
            try:
                # 确定是否为ICV (25%概率)
                is_icv = hash(veh_id) % 100 < 25

                # 获取车辆位置
                try:
                    position = traci.vehicle.getLanePosition(veh_id)
                except:
                    position = 0.0

                # 获取速度
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                except:
                    speed = 0.0

                # 获取加速度
                try:
                    acceleration = traci.vehicle.getAcceleration(veh_id)
                except:
                    acceleration = 0.0

                # 获取车道索引
                try:
                    lane_index = traci.vehicle.getLaneIndex(veh_id)
                except:
                    lane_index = 0

                # 获取车道ID
                try:
                    lane_id = traci.vehicle.getLaneID(veh_id)
                except:
                    lane_id = ""

                # 获取路线信息
                try:
                    route = traci.vehicle.getRoute(veh_id)
                    route_index = traci.vehicle.getRouteIndex(veh_id)
                except:
                    route = []
                    route_index = 0

                # 计算剩余距离（基于路线）
                remaining_distance = 0.0
                if route and route_index < len(route):
                    try:
                        # 获取当前路段长度
                        current_edge_length = traci.lane.getLength(f"{lane_id}_0") if lane_id else 0
                        remaining_distance += current_edge_length - position
                        
                        # 计算剩余路段的长度
                        for i in range(route_index + 1, len(route)):
                            edge_id = route[i]
                            try:
                                edge_length = traci.edge.getLength(edge_id)
                                remaining_distance += edge_length
                            except:
                                remaining_distance += 100  # 默认长度
                    except:
                        remaining_distance = 1000.0  # 默认值

                # 计算完成率
                try:
                    total_route_length = sum(
                        traci.edge.getLength(edge_id) for edge_id in route
                        if traci.edge.getLength(edge_id) > 0
                    ) if route else 1.0
                    completed_distance = max(0.0, total_route_length - remaining_distance)
                    completion_rate = min(1.0, completed_distance / max(total_route_length, 1.0))
                except:
                    completion_rate = 0.0

                # 获取当前边缘ID
                try:
                    edge_id = traci.vehicle.getRoadID(veh_id)
                except:
                    edge_id = ""

                vehicle_data[veh_id] = {
                    'position': position,
                    'speed': speed,
                    'acceleration': acceleration,
                    'lane_index': lane_index,
                    'remaining_distance': remaining_distance,
                    'completion_rate': completion_rate,
                    'is_icv': is_icv,
                    'id': veh_id,
                    'lane_id': lane_id,
                    'edge_id': edge_id,
                    'route_index': route_index
                }
            except Exception as e:
                print(f"Error collecting data for vehicle {veh_id}: {e}")
                continue

        return vehicle_data

    def _get_observation(self) -> Dict[str, Any]:
        """获取当前环境观测"""
        # 收集车辆数据
        vehicle_ids = traci.vehicle.getIDList()
        
        # 初始化观测数据结构
        node_features = np.zeros((100, 9), dtype=np.float32)
        edge_indices = np.zeros((2, 200), dtype=np.int64)
        global_metrics = np.zeros(16, dtype=np.float32)
        
        # 收集完整的车辆数据
        vehicle_data_list = []
        for veh_id in vehicle_ids[:100]:  # 最多取100辆车
            try:
                # 获取车辆信息
                position = traci.vehicle.getLanePosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                acceleration = traci.vehicle.getAcceleration(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                
                # 获取路线和距离信息
                route = traci.vehicle.getRoute(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)
                
                # 计算剩余距离
                remaining_distance = 0.0
                if route and route_index < len(route):
                    try:
                        # 获取当前路段长度
                        lane_id = traci.vehicle.getLaneID(veh_id)
                        current_edge_length = traci.lane.getLength(f"{lane_id}_0") if lane_id else 0
                        remaining_distance += current_edge_length - position
                        
                        # 计算剩余路段的长度
                        for i in range(route_index + 1, len(route)):
                            edge_id = route[i]
                            try:
                                edge_length = traci.edge.getLength(edge_id)
                                remaining_distance += edge_length
                            except:
                                remaining_distance += 100  # 默认长度
                    except:
                        remaining_distance = 1000.0  # 默认值

                # 计算完成率
                try:
                    total_route_length = sum(
                        traci.edge.getLength(edge_id) for edge_id in route
                        if traci.edge.getLength(edge_id) > 0
                    ) if route else 1.0
                    completed_distance = max(0.0, total_route_length - remaining_distance)
                    completion_rate = min(1.0, completed_distance / max(total_route_length, 1.0))
                except:
                    completion_rate = 0.0
                
                # 节点特征: [位置, 速度, 加速度, 车道, 剩余距离, 完成率, 类型, 时间, 步长]
                is_icv = 1.0 if (hash(veh_id) % 100 < 25) else 0.0  # 25%是智能车
                
                node_features[len(vehicle_data_list)] = [
                    position,
                    speed,
                    acceleration,
                    lane_index,
                    remaining_distance,
                    completion_rate,
                    is_icv,
                    self.current_step * 0.1,  # 时间
                    0.1       # 步长
                ]
                
                vehicle_data_list.append({
                    'id': veh_id,
                    'position': position,
                    'speed': speed,
                    'lane_id': traci.vehicle.getLaneID(veh_id)
                })
            except Exception as e:
                print(f"Error getting data for vehicle {veh_id}: {e}")
                continue
        
        # 构建边索引（基于车辆间的距离）
        edge_idx = 0
        for i in range(len(vehicle_data_list)):
            for j in range(i+1, len(vehicle_data_list)):
                # 计算车辆之间的距离（基于车道和位置）
                veh_i = vehicle_data_list[i]
                veh_j = vehicle_data_list[j]
                
                # 如果在同一条车道上，计算相对距离
                if veh_i['lane_id'] == veh_j['lane_id']:
                    distance = abs(veh_i['position'] - veh_j['position'])
                    if distance < 50:  # 距离小于50米则建立连接
                        if edge_idx < 200:
                            edge_indices[0][edge_idx] = i
                            edge_indices[1][edge_idx] = j
                            edge_idx += 1
                # 如果不在同一条车道但距离可能较近，可以进一步判断
                # （这里为了简化只考虑同一车道的情况，可以根据需要扩展）
        
        # 计算全局指标
        speeds = [data['speed'] for data in vehicle_data_list]
        positions = [data['position'] for data in vehicle_data_list]
        
        if speeds:
            avg_speed = np.mean(speeds)
            speed_std = np.std(speeds)
        else:
            avg_speed = 0.0
            speed_std = 0.0
            
        if positions:
            avg_position = np.mean(positions)
            min_position = min(positions) if positions else 0.0
            max_position = max(positions) if positions else 0.0
        else:
            avg_position = 0.0
            min_position = 0.0
            max_position = 0.0
            
        # 计算加速度相关指标
        accelerations = []
        for veh_id in vehicle_ids:
            try:
                acc = traci.vehicle.getAcceleration(veh_id)
                accelerations.append(acc)
            except:
                continue
        
        avg_accel = np.mean(accelerations) if accelerations else 0.0
        abs_avg_accel = np.mean(np.abs(accelerations)) if accelerations else 0.0
        
        # 统计ICV和普通车辆
        icv_count = sum(1 for vid in vehicle_ids if hash(vid) % 100 < 25)
        regular_count = len(vehicle_ids) - icv_count
        
        # ICV和普通车辆的速度总和
        icv_speed_sum = sum(traci.vehicle.getSpeed(vid) for vid in vehicle_ids if hash(vid) % 100 < 25)
        regular_speed_sum = sum(traci.vehicle.getSpeed(vid) for vid in vehicle_ids if hash(vid) % 100 >= 25)
        
        global_metrics = np.array([
            avg_speed,           # 平均速度
            speed_std,           # 速度标准差
            abs_avg_accel,       # 平均绝对加速度
            len(vehicle_ids),    # 车辆总数
            self.current_step * 0.1,  # 当前时间
            min_position,        # 最小位置
            max_position,        # 最大位置
            avg_position,        # 平均位置
            icv_count,           # ICV数量
            regular_count,       # 非ICV数量
            icv_speed_sum,       # ICV总速度
            regular_speed_sum,   # 非ICV总速度
            avg_speed * len(vehicle_ids),  # 总流量
            speed_std * len(vehicle_ids),  # 总波动
            abs_avg_accel * len(vehicle_ids),  # 总加速度变化
            self.current_step % 100  # 周期性特征
        ], dtype=np.float32)
        
        return {
            'node_features': node_features,
            'edge_indices': edge_indices,
            'global_metrics': global_metrics
        }

    def _compute_reward(self) -> float:
        """计算奖励函数"""
        # 获取当前所有车辆的速度
        vehicle_ids = traci.vehicle.getIDList()
        if not vehicle_ids:
            return 0.0
        
        speeds = [traci.vehicle.getSpeed(vid) for vid in vehicle_ids]
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        # 计算速度的标准差（越小越好，表示交通流畅）
        speed_std = np.std(speeds) if len(speeds) > 1 else 0.0
        
        # 奖励 = 平均速度 * 权重 - 速度方差 * 权重
        reward = avg_speed * 0.1 - speed_std * 0.05
        
        # 如果有车辆停止（速度为0），给予小惩罚
        stopped_cars = sum(1 for s in speeds if s < 0.1)
        reward -= stopped_cars * 0.01
        
        return reward

    def render(self, mode='human'):
        """渲染环境（暂时不需要实现）"""
        pass

    def close(self):
        """关闭环境"""
        if traci.isLoaded():
            traci.close()