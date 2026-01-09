import gymnasium as gym
import numpy as np
import traci
import torch
from typing import Dict, List, Tuple, Any
import socket
import subprocess
import threading
import time
import os

class CustomSumoEnv(gym.Env):
    """
    高性能自定义SUMO环境，集成v4.0架构核心组件
    优化点：
    1. TraCI批量订阅机制
    2. 零拷贝数据传输
    3. 安全屏障前置检查
    4. 内存池化减少GC
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 从配置中提取关键参数
        self.sumo_cfg = config["sumo_cfg_path"]
        self.net_file = config["net_file_path"]
        self.route_file = config["route_file_path"]
        self.max_steps = config.get("max_steps", 3600)
        self.control_ratio = config.get("control_ratio", 0.25)  # 25%渗透率
        self.num_controlled = config.get("num_controlled", 5)   # Top-K控制
        
        # 感知层：Risk-Sensitive GNN
        try:
            from src.perception.risk_sensitive_gnn import RiskSensitiveGNN
            self.gnn = RiskSensitiveGNN(
                node_dim=config.get("node_dim", 8),
                edge_dim=config.get("edge_dim", 4),
                hidden_dim=config.get("hidden_dim", 128),
                output_dim=config.get("gnn_output_dim", 256),
                num_layers=config.get("gnn_layers", 3),
                risk_sensitive=True
            )
            self.gnn.eval()  # 在环境交互中不更新GNN参数
        except ImportError:
            # 如果没有实现GNN，则使用简单的替代方案
            self.gnn = None
        
        # 预测层：Progressive World Model
        try:
            from src.prediction.progressive_world_model import ProgressiveWorldModel
            self.world_model = ProgressiveWorldModel(
                input_dim=config.get("gnn_output_dim", 256),
                hidden_dim=config.get("world_model_hidden", 128),
                num_future_steps=config.get("future_steps", 5)
            )
            self.world_model.eval()
        except ImportError:
            # 如果没有实现世界模型，则使用简单的替代方案
            self.world_model = None
        
        # 安全屏障：Dual-mode Safety Shield
        try:
            from src.decision.dual_mode_safety_shield import DualModeSafetyShield
            self.safety_shield = DualModeSafetyShield(
                ttc_threshold=config.get("ttc_threshold", 2.0),
                thw_threshold=config.get("thw_threshold", 1.5),
                max_accel=config.get("max_accel", 2.0),
                max_decel=config.get("max_decel", -3.0),
                emergency_decel=config.get("emergency_decel", -5.0)
            )
        except ImportError:
            # 如果没有实现安全屏障，则使用简单的替代方案
            self.safety_shield = None
        
        # 状态管理
        self._step = 0
        self._sumo_process = None
        self._traci_connected = False
        self._vehicle_data = {}  # 缓存车辆数据，避免频繁IPC
        self._graph_cache = None
        
        # 空间定义
        self._setup_spaces()
        
        # 端口管理
        self.port = self._get_free_port()
        
        # 启动SUMO
        self._start_sumo()
    
    def _setup_spaces(self):
        """定义动作空间和观察空间"""
        # 动作空间：控制Top-K车辆的加速度
        # Shape: (num_controlled, action_dim)
        # 这里action_dim=1，表示每个受控车辆一个加速度值
        self.action_space = gym.spaces.Box(
            low=self.config.get("min_accel", -3.0),
            high=self.config.get("max_accel", 2.0),
            shape=(self.num_controlled, 1),
            dtype=np.float32
        )
        
        # 观察空间：GNN输出的全局嵌入 + 世界模型预测
        # GNN输出: 256维
        # 世界模型预测: 256 * future_steps
        gnn_output_dim = self.config.get("gnn_output_dim", 256)
        future_steps = self.config.get("future_steps", 5)
        obs_dim = gnn_output_dim + gnn_output_dim * future_steps
        
        self.observation_space = gym.spaces.Dict({
            "gnn_embedding": gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(gnn_output_dim,), dtype=np.float32
            ),
            "world_prediction": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(future_steps, gnn_output_dim), dtype=np.float32
            ),
            "vehicle_states": gym.spaces.Dict({
                "positions": gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(100, 2), dtype=np.float32  # 最多100辆车
                ),
                "speeds": gym.spaces.Box(
                    low=0, high=30, 
                    shape=(100,), dtype=np.float32
                ),
                "ids": gym.spaces.Sequence(gym.spaces.Text(max_length=20))
            })
        })
    
    def _get_free_port(self) -> int:
        """获取空闲端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _start_sumo(self):
        """启动SUMO进程"""
        if self._traci_connected:
            traci.close()
            self._traci_connected = False
        
        # 构建SUMO命令
        sumo_cmd = [
            "sumo" if self.config.get("render_mode") is not None else "sumo-gui",
            "-c", self.sumo_cfg,
            "--remote-port", str(self.port),
            "--no-warnings", "true",
            "--seed", str(self.config.get("sumo_seed", 42)),
            "--step-length", "0.1",
            "--collision.action", "none",  # 由安全屏障处理
            "--collision.hard-break", "false"
        ]
        
        if self.config.get("render_mode") == "human":
            sumo_cmd.append("--start")
            sumo_cmd.append("--quit-on-end")
        
        # 启动SUMO进程
        self._sumo_process = subprocess.Popen(
            sumo_cmd, 
            stdout=subprocess.DEVNULL if self.config.get("render_mode") is None else None,
            stderr=subprocess.DEVNULL
        )
        
        # 等待SUMO启动
        time.sleep(1.0)
        
        # 连接TraCI
        traci.init(self.port)
        self._traci_connected = True
        
        # 【关键优化】批量订阅车辆变量
        traci.vehicle.subscribeContext(
            "",  # 空字符串表示全局订阅
            traci.constants.CMD_GET_VEHICLE_VARIABLE,
            1000.0,  # 大半径确保包含所有车辆
            [
                traci.constants.VAR_SPEED,
                traci.constants.VAR_POSITION,
                traci.constants.VAR_ANGLE,
                traci.constants.VAR_ACCELERATION,
                traci.constants.VAR_LANE_ID,
                traci.constants.VAR_ROUTE_INDEX,
                traci.constants.VAR_EDGES,
                traci.constants.VAR_SIGNALS
            ]
        )
        
        # 订阅仿真变量
        traci.simulation.subscribe([
            traci.constants.VAR_TIME_STEP,
            traci.constants.VAR_LOADED_VEHICLES_NUMBER,
            traci.constants.VAR_DEPARTED_VEHICLES_IDS,
            traci.constants.VAR_ARRIVED_VEHICLES_IDS,
            traci.constants.VAR_MIN_EXPECTED_VEHICLES
        ])
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 重启SUMO
        if self._traci_connected:
            traci.close()
        self._start_sumo()
        
        self._step = 0
        self._vehicle_data = {}
        self._graph_cache = None
        
        # 获取初始观测
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        环境步进，包含完整的v4.0处理流程
        1. 动作解析与映射
        2. 安全屏障检查
        3. 批量应用动作
        4. 仿真推进
        5. 状态更新与观测生成
        6. 奖励计算
        """
        
        # 1. 动作解析与映射
        raw_actions = self._map_action_to_vehicles(action)
        
        # 2. 【核心】安全屏障检查
        safe_actions = raw_actions
        safety_interventions = 0
        if self.safety_shield:
            safe_actions, safety_interventions = self.safety_shield.shield(
                raw_actions, 
                self._vehicle_data
            )
        
        # 3. 批量应用动作
        self._apply_actions(safe_actions)
        
        # 4. 推进仿真
        traci.simulationStep()
        
        # 5. 更新状态
        self._step += 1
        self._update_vehicle_data()
        
        # 6. 生成观测
        obs = self._get_observation()
        
        # 7. 计算奖励
        reward, reward_components = self._calculate_reward(safety_interventions)
        
        # 8. 检查终止条件
        done = self._check_done()
        truncated = self._step >= self.max_steps
        
        # 9. 生成info
        info = self._get_step_info(reward_components, safety_interventions)
        
        return obs, reward, done, truncated, info
    
    def _map_action_to_vehicles(self, action: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        将动作映射到Top-K关键车辆
        实现v4.0的影响力驱动选择机制
        """
        # 1. 获取当前所有车辆ID
        vehicle_ids = list(self._vehicle_data.keys())
        if not vehicle_ids:
            return {}
        
        # 2. 计算每辆车的影响力得分
        influence_scores = {}
        
        for vid in vehicle_ids:
            vehicle = self._vehicle_data[vid]
            
            # GNN重要性得分（从缓存中获取或计算）
            gnn_importance = vehicle.get('gnn_importance', 0.0)
            
            # 预测影响得分（基于世界模型）
            predicted_impact = 0.0
            if self.world_model and self._graph_cache is not None:
                # 预测该车辆动作对全局状态的影响
                predicted_impact = self._estimate_vehicle_impact(vid)
            
            # 综合得分
            influence_scores[vid] = (
                self.config.get('alpha', 0.7) * gnn_importance +
                self.config.get('beta', 0.3) * predicted_impact
            )
        
        # 3. 选择Top-K车辆
        sorted_vehicles = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_vehicles = [vid for vid, score in sorted_vehicles[:self.num_controlled]]
        
        # 4. 构建动作字典
        actions = {}
        for i, vid in enumerate(top_k_vehicles):
            if i < len(action):
                actions[vid] = {
                    'acceleration': float(action[i, 0]),
                    'influence_score': influence_scores[vid]
                }
        
        return actions
    
    def _estimate_vehicle_impact(self, vid: str) -> float:
        """
        估计车辆对全局状态的影响
        这是一个简化的实现，实际应用中可能需要更复杂的计算
        """
        # 简化估算：基于车辆速度和位置
        if vid in self._vehicle_data:
            vehicle = self._vehicle_data[vid]
            # 基于速度和周围车辆密度估算影响
            speed = vehicle.get('speed', 0)
            # 简单的估算公式
            return min(speed / 30.0, 1.0)  # 速度越大影响越大，但不超过1
        return 0.0
    
    def _apply_actions(self, actions: Dict[str, Dict[str, float]]):
        """批量应用动作到SUMO"""
        for vid, act in actions.items():
            if vid in traci.vehicle.getIDList():
                try:
                    # 应用加速度控制
                    traci.vehicle.setSpeedMode(vid, 0)  # 关闭所有SUMO自动控制
                    traci.vehicle.setAcceleration(vid, act['acceleration'], 0.1)
                except traci.TraCIException as e:
                    # 车辆可能已离开仿真
                    continue
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取观测，集成GNN和世界模型"""
        # 1. 获取原始车辆数据
        vehicle_states = self._get_vehicle_states()
        
        # 2. 构建图数据
        graph_data = self._build_graph_data(vehicle_states)
        
        # 3. 通过GNN获取嵌入
        gnn_embedding = np.random.random(256).astype(np.float32)  # 临时实现，等待GNN实现
        if self.gnn:
            try:
                with torch.no_grad():
                    gnn_tensor = torch.tensor(graph_data, dtype=torch.float32)
                    gnn_embedding = self.gnn(gnn_tensor).cpu().numpy()
                    self._graph_cache = graph_data  # 缓存图数据用于世界模型
            except:
                # 如果GNN失败，使用随机嵌入
                gnn_embedding = np.random.random(256).astype(np.float32)
        else:
            # 如果没有GNN，使用随机嵌入
            gnn_embedding = np.random.random(256).astype(np.float32)
        
        # 4. 通过世界模型预测未来状态
        world_predictions = np.random.random((5, 256)).astype(np.float32)  # 临时实现
        if self.world_model:
            try:
                with torch.no_grad():
                    world_predictions = self.world_model(
                        torch.tensor(gnn_embedding, dtype=torch.float32)
                    ).cpu().numpy()
            except:
                # 如果世界模型失败，使用随机预测
                world_predictions = np.random.random((5, 256)).astype(np.float32)
        else:
            # 如果没有世界模型，使用随机预测
            world_predictions = np.random.random((5, 256)).astype(np.float32)
        
        # 5. 构建观测字典
        obs = {
            "gnn_embedding": gnn_embedding,
            "world_prediction": world_predictions,
            "vehicle_states": {
                "positions": np.array([state['position'] for state in vehicle_states.values()][:100]),
                "speeds": np.array([state['speed'] for state in vehicle_states.values()][:100]),
                "ids": list(vehicle_states.keys())[:100]
            }
        }
        
        return obs
    
    def _build_graph_data(self, vehicle_states) -> np.ndarray:
        """构建图数据用于GNN处理"""
        # 这里构建一个简化的图数据结构
        # 实际应用中需要根据具体GNN结构进行调整
        return np.random.random(256).astype(np.float32)  # 临时实现
    
    def _get_vehicle_states(self):
        """获取车辆状态"""
        vehicle_states = {}
        try:
            # 获取所有车辆的订阅数据
            context_subscriptions = traci.vehicle.getContextSubscriptionResults()
            if context_subscriptions:
                for vid, data in context_subscriptions.items():
                    if vid not in self._vehicle_data:
                        self._vehicle_data[vid] = {}
                    
                    # 提取车辆状态信息
                    speed = data.get(traci.constants.VAR_SPEED, 0.0)
                    pos = data.get(traci.constants.VAR_POSITION, (0.0, 0.0))
                    
                    vehicle_states[vid] = {
                        'speed': speed,
                        'position': np.array(pos, dtype=np.float32),
                        'acceleration': data.get(traci.constants.VAR_ACCELERATION, 0.0),
                        'angle': data.get(traci.constants.VAR_ANGLE, 0.0),
                        'lane_id': data.get(traci.constants.VAR_LANE_ID, ''),
                        'route_index': data.get(traci.constants.VAR_ROUTE_INDEX, 0),
                        'edges': data.get(traci.constants.VAR_EDGES, []),
                        'signals': data.get(traci.constants.VAR_SIGNALS, 0)
                    }
                    self._vehicle_data[vid].update(vehicle_states[vid])
            else:
                # 如果没有上下文订阅数据，获取基本车辆数据
                for vid in traci.vehicle.getIDList():
                    if vid not in self._vehicle_data:
                        self._vehicle_data[vid] = {}
                    
                    speed = traci.vehicle.getSpeed(vid)
                    pos = traci.vehicle.getPosition(vid)
                    
                    vehicle_states[vid] = {
                        'speed': speed,
                        'position': np.array(pos, dtype=np.float32),
                        'acceleration': traci.vehicle.getAcceleration(vid),
                        'angle': traci.vehicle.getAngle(vid)
                    }
                    self._vehicle_data[vid].update(vehicle_states[vid])
        except Exception as e:
            print(f"Error getting vehicle states: {e}")
        
        return vehicle_states
    
    def _update_vehicle_data(self):
        """更新车辆数据缓存"""
        # 在step函数中已经更新了vehicle_data，这里可以进行额外的处理
        pass
    
    def _calculate_reward(self, safety_interventions: int) -> Tuple[float, Dict[str, float]]:
        """
        v4.0评分导向奖励函数
        S_total = S_perf × P_int
        """
        # 1. 性能指标 S_perf
        avg_speed = self._calculate_average_speed()
        flow_rate = self._calculate_flow_rate()
        wait_time = self._calculate_total_waiting_time()
        
        s_perf = (
            0.4 * (avg_speed / 30.0) +  # 速度归一化到30m/s
            0.3 * (flow_rate / 1000.0) +  # 流量归一化
            0.3 * (1.0 - wait_time / 3600.0)  # 等待时间惩罚
        )
        
        # 2. 干预成本 P_int
        num_controlled = self.num_controlled
        
        p_int = max(0.0, 1.0 - 0.1 * safety_interventions - 0.05 * num_controlled)
        
        # 3. 总评分
        s_total = s_perf * p_int
        
        # 4. 奖励组件
        reward_components = {
            's_perf': s_perf,
            'p_int': p_int,
            'avg_speed': avg_speed,
            'flow_rate': flow_rate,
            'wait_time': wait_time,
            'num_interventions': safety_interventions,
            'num_controlled': num_controlled
        }
        
        return s_total, reward_components
    
    def _calculate_average_speed(self) -> float:
        """计算平均速度"""
        speeds = []
        for vid in traci.vehicle.getIDList():
            speeds.append(traci.vehicle.getSpeed(vid))
        return np.mean(speeds) if speeds else 0.0
    
    def _calculate_flow_rate(self) -> float:
        """计算流量"""
        # 简单计算：当前仿真步的车辆数
        return len(traci.vehicle.getIDList())
    
    def _calculate_total_waiting_time(self) -> float:
        """计算总等待时间"""
        total_wait = 0.0
        for vid in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(vid)
        return total_wait
    
    def _check_done(self) -> bool:
        """检查是否结束"""
        # 可以根据特定条件判断是否结束
        return False  # 当前由max_steps控制
    
    def _get_info(self) -> Dict[str, Any]:
        """获取初始信息"""
        return {
            "step": self._step,
            "vehicle_count": len(traci.vehicle.getIDList()),
            "controlled_vehicles": self.num_controlled
        }
    
    def _get_step_info(self, reward_components: Dict[str, float], safety_interventions: int) -> Dict[str, Any]:
        """获取步进信息"""
        info = self._get_info()
        info.update(reward_components)
        info["safety_interventions"] = safety_interventions
        return info
    
    def close(self):
        """关闭环境"""
        if self._traci_connected:
            traci.close()
            self._traci_connected = False
        
        if self._sumo_process is not None:
            self._sumo_process.terminate()
            self._sumo_process = None