"""
SUMO-RL环境封装 - Gymnasium标准接口
集成TrafficController模型进行推理，支持LIBSUMO_AS_TRACI和批量订阅功能
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入SUMO-RL
try:
    from sumo_rl import SumoEnvironment as SUMOEnv
    SUMO_RL_AVAILABLE = True
except ImportError:
    logger.warning("SUMO-RL未安装，将使用基础TraCI接口")
    SUMO_RL_AVAILABLE = False

# 尝试导入TraCI
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    logger.error("TraCI未安装，请安装SUMO")
    TRACI_AVAILABLE = False

# 导入TrafficController
from neural_traffic_controller import TrafficController


class SUMOGymEnv(gym.Env):
    """
    SUMO-RL Gymnasium环境封装
    
    功能：
    - 继承gymnasium.Env标准接口
    - 集成TrafficController模型进行推理
    - 支持LIBSUMO_AS_TRACI加速
    - 支持批量订阅功能
    - 计算奖励和安全指标
    - 返回标准Gymnasium格式：observation, reward, done, truncated, info
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10
    }
    
    def __init__(
        self,
        sumo_cfg_path: str,
        port: Optional[int] = None,
        use_libsumo: bool = False,
        batch_subscribe: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 3600,
        use_gui: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        初始化SUMO Gymnasium环境
        
        Args:
            sumo_cfg_path: SUMO配置文件路径
            port: TraCI端口（None表示自动分配）
            use_libsumo: 是否启用LIBSUMO_AS_TRACI加速
            batch_subscribe: 是否启用批量订阅功能
            device: 计算设备（cuda/cpu）
            model_config: TrafficController模型配置
            max_steps: 最大仿真步数
            use_gui: 是否使用GUI
            seed: 随机种子
            **kwargs: 其他参数传递给SUMO-RL
        """
        super().__init__()
        
        # 基础配置
        self.sumo_cfg_path = sumo_cfg_path
        self.port = port
        self.use_libsumo = use_libsumo
        self.batch_subscribe = batch_subscribe
        self.device = device
        self.max_steps = max_steps
        self.use_gui = use_gui
        self.seed_val = seed
        self.kwargs = kwargs
        
        # 环境状态
        self.current_step = 0
        self.connected = False
        self.vehicle_ids = []
        self.sumo_env = None
        
        # 初始化TrafficController模型
        self.model_config = model_config or self._get_default_model_config()
        self.traffic_controller = TrafficController(self.model_config).to(self.device)
        self.traffic_controller.eval()  # 推理模式
        
        # 定义观察空间和动作空间
        self._define_spaces()
        
        # 统计信息
        self.total_reward = 0.0
        self.episode_rewards = []
        self.safety_metrics = {
            'ttc_violations': 0,
            'thw_violations': 0,
            'speed_violations': 0,
            'accel_violations': 0
        }
        
        # 批量订阅缓存
        self.subscription_cache = {}
        self.cache_timeout = 0.1
        
        logger.info(f"✅ SUMO Gymnasium环境初始化完成")
        logger.info(f"   配置文件: {sumo_cfg_path}")
        logger.info(f"   设备: {device}")
        logger.info(f"   LIBSUMO: {use_libsumo}")
        logger.info(f"   批量订阅: {batch_subscribe}")
        logger.info(f"   最大步数: {max_steps}")
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """
        获取默认的模型配置
        
        Returns:
            model_config: 默认模型配置字典
        """
        return {
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
            'ttc_threshold': 2.0,
            'thw_threshold': 1.5,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'emergency_decel': -5.0,
            'max_lane_change_speed': 5.0,
            'cost_limit': 0.1,
            'lambda_lr': 0.01
        }
    
    def _define_spaces(self):
        """
        定义观察空间和动作空间
        
        观察空间：包含车辆特征、边特征、全局指标
        动作空间：连续动作空间 [加速度, 换道概率]
        """
        # 观察空间 - 使用Dict空间
        self.observation_space = spaces.Dict({
            # 节点特征 [N, 9]
            'node_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(None, 9), dtype=np.float32
            ),
            # 边索引 [2, E]
            'edge_indices': spaces.Box(
                low=0, high=np.inf,
                shape=(2, None), dtype=np.int64
            ),
            # 边特征 [E, 4]
            'edge_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(None, 4), dtype=np.float32
            ),
            # 全局指标 [16]
            'global_metrics': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(16,), dtype=np.float32
            ),
            # 车辆ID列表
            'vehicle_ids': spaces.Box(
                low=0, high=np.inf,
                shape=(None,), dtype=np.object_
            ),
            # ICV标记
            'is_icv': spaces.Box(
                low=0, high=1,
                shape=(None,), dtype=np.bool_
            )
        })
        
        # 动作空间 - 连续空间 [加速度, 换道概率]
        # 加速度范围: [-5, 5] m/s²
        # 换道概率: [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-5.0, 0.0]),
            high=np.array([5.0, 1.0]),
            dtype=np.float32
        )
        
        logger.info(f"✅ 观察空间和动作空间定义完成")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        # 设置随机种子
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 关闭现有连接
        self._close_connection()
        
        # 重置状态
        self.current_step = 0
        self.total_reward = 0.0
        self.vehicle_ids = []
        self.safety_metrics = {
            'ttc_violations': 0,
            'thw_violations': 0,
            'speed_violations': 0,
            'accel_violations': 0
        }
        
        # 清空订阅缓存
        self.subscription_cache.clear()
        
        # 启动SUMO环境
        self._start_sumo()
        
        # 执行第一步以初始化车辆
        if TRACI_AVAILABLE:
            traci.simulationStep()
            self.current_step += 1
        
        # 获取初始观测
        observation = self._get_observation()
        
        # 构建info字典
        info = {
            'step': self.current_step,
            'vehicle_count': len(self.vehicle_ids),
            'safety_metrics': self.safety_metrics.copy()
        }
        
        logger.info(f"🚀 环境已重置，初始车辆数: {len(self.vehicle_ids)}")
        
        return observation, info
    
    def step(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一步仿真
        
        Args:
            action: 动作（可选，如果不提供则使用TrafficController推理）
            
        Returns:
            observation: 观测
            reward: 奖励
            done: 是否自然结束
            truncated: 是否被截断（达到最大步数）
            info: 额外信息
        """
        # 1. 调用TrafficController进行推理
        if action is None:
            # 使用TrafficController推理生成动作
            observation = self._get_observation()
            controller_output = self._run_controller_inference(observation)
            
            # 提取safe_actions
            safe_actions = controller_output.get('safe_actions', {})
            selected_vehicle_ids = controller_output.get('selected_vehicle_ids', [])
        else:
            # 使用提供的动作（用于外部控制）
            safe_actions = {'actions': [action]}
            selected_vehicle_ids = self.vehicle_ids[:1] if self.vehicle_ids else []
        
        # 2. 应用动作到SUMO环境
        self._apply_actions(selected_vehicle_ids, safe_actions)
        
        # 3. 推进仿真
        if TRACI_AVAILABLE:
            traci.simulationStep()
            self.current_step += 1
        
        # 4. 获取观测
        observation = self._get_observation()
        
        # 5. 计算奖励
        reward = self._calculate_reward(observation)
        self.total_reward += reward
        
        # 6. 计算安全指标
        safety_metrics = self._calculate_safety_metrics(observation)
        
        # 7. 检查终止条件
        done, truncated = self._check_termination()
        
        # 8. 构建info字典
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'vehicle_count': len(self.vehicle_ids),
            'safety_metrics': safety_metrics,
            'selected_vehicles': selected_vehicle_ids,
            'controller_output': controller_output if action is None else None
        }
        
        return observation, reward, done, truncated, info
    
    def close(self):
        """关闭环境"""
        self._close_connection()
        logger.info("✅ SUMO Gymnasium环境已关闭")
    
    def _start_sumo(self):
        """
        启动SUMO环境
        
        支持SUMO-RL和原生TraCI两种方式
        """
        if SUMO_RL_AVAILABLE and not self.use_libsumo:
            # 使用SUMO-RL
            try:
                self.sumo_env = SUMOEnv(
                    net_file=self._extract_net_file(),
                    route_file=self._extract_route_file(),
                    use_gui=self.use_gui,
                    num_seconds=self.max_steps,
                    max_steps=self.max_steps,
                    single_agent=True,
                    sumo_binary="sumo-gui" if self.use_gui else "sumo",
                    seed=self.seed_val,
                    **self.kwargs
                )
                self.sumo_env.reset()
                self.connected = True
                logger.info("✅ SUMO-RL环境已启动")
                return
            except Exception as e:
                logger.warning(f"SUMO-RL启动失败，降级到TraCI: {e}")
        
        # 使用原生TraCI
        if TRACI_AVAILABLE:
            self._start_traci()
        else:
            raise RuntimeError("无法启动SUMO环境：SUMO-RL和TraCI都不可用")
    
    def _start_traci(self):
        """
        使用TraCI启动SUMO
        支持LIBSUMO_AS_TRACI加速
        """
        # 构建SUMO命令
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]
        
        # 添加端口
        if self.port is not None:
            sumo_cmd.extend(["--remote-port", str(self.port)])
        
        # 添加随机种子
        if self.seed_val is not None:
            sumo_cmd.extend(["--seed", str(self.seed_val)])
        
        # LIBSUMO_AS_TRACI加速
        if self.use_libsumo:
            os.environ['LIBSUMO_AS_TRACI'] = '1'
            logger.info("✅ LIBSUMO_AS_TRACI已启用")
        
        try:
            traci.start(sumo_cmd)
            self.connected = True
            logger.info("✅ TraCI环境已启动")
        except Exception as e:
            logger.error(f"❌ TraCI启动失败: {e}")
            raise
    
    def _close_connection(self):
        """关闭SUMO连接"""
        if self.sumo_env is not None:
            try:
                self.sumo_env.close()
                self.sumo_env = None
            except Exception as e:
                logger.warning(f"关闭SUMO-RL时出错: {e}")
        
        if self.connected and TRACI_AVAILABLE:
            try:
                traci.close()
                self.connected = False
            except Exception as e:
                logger.warning(f"关闭TraCI时出错: {e}")
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        获取当前观测
        
        Returns:
            observation: 观测数据字典
        """
        if not TRACI_AVAILABLE:
            return self._get_empty_observation()
        
        # 获取所有车辆ID
        self.vehicle_ids = traci.vehicle.getIDList()
        
        if not self.vehicle_ids:
            return self._get_empty_observation()
        
        # 批量订阅车辆数据
        if self.batch_subscribe:
            vehicle_data = self._get_vehicle_data_batch()
        else:
            vehicle_data = self._get_vehicle_data_direct()
        
        # 构建图结构
        graph_data = self._build_graph(vehicle_data)
        
        # 计算全局指标
        global_metrics = self._compute_global_metrics(vehicle_data)
        
        # 构建观测
        observation = {
            'node_features': graph_data['node_features'],
            'edge_indices': graph_data['edge_indices'],
            'edge_features': graph_data['edge_features'],
            'global_metrics': global_metrics,
            'vehicle_ids': np.array(self.vehicle_ids, dtype=object),
            'is_icv': graph_data['is_icv'],
            'vehicle_data': vehicle_data
        }
        
        return observation
    
    def _get_empty_observation(self) -> Dict[str, Any]:
        """获取空观测（无车辆时）"""
        return {
            'node_features': np.zeros((0, 9), dtype=np.float32),
            'edge_indices': np.zeros((2, 0), dtype=np.int64),
            'edge_features': np.zeros((0, 4), dtype=np.float32),
            'global_metrics': np.zeros(16, dtype=np.float32),
            'vehicle_ids': np.array([], dtype=object),
            'is_icv': np.zeros(0, dtype=np.bool_),
            'vehicle_data': {}
        }
    
    def _get_vehicle_data_batch(self) -> Dict[str, Dict[str, Any]]:
        """
        使用批量订阅获取车辆数据
        
        Returns:
            vehicle_data: 车辆数据字典
        """
        vehicle_data = {}
        
        # 批量订阅车辆
        for veh_id in self.vehicle_ids:
            try:
                # 订阅常用变量
                traci.vehicle.subscribe(
                    veh_id,
                    [
                        traci.constants.VAR_LANEPOSITION,
                        traci.constants.VAR_SPEED,
                        traci.constants.VAR_ACCELERATION,
                        traci.constants.VAR_LANE_INDEX,
                        traci.constants.VAR_LANE_ID,
                        traci.constants.VAR_ROAD_ID,
                        traci.constants.VAR_VEHICLECLASS
                    ]
                )
            except Exception as e:
                logger.warning(f"订阅车辆 {veh_id} 失败: {e}")
        
        # 批量获取数据
        for veh_id in self.vehicle_ids:
            try:
                sub_results = traci.vehicle.getSubscriptionResults(veh_id)
                if sub_results:
                    vehicle_data[veh_id] = {
                        'position': sub_results.get(traci.constants.VAR_LANEPOSITION, 0.0),
                        'speed': sub_results.get(traci.constants.VAR_SPEED, 0.0),
                        'acceleration': sub_results.get(traci.constants.VAR_ACCELERATION, 0.0),
                        'lane_index': sub_results.get(traci.constants.VAR_LANE_INDEX, 0),
                        'lane_id': sub_results.get(traci.constants.VAR_LANE_ID, ''),
                        'road_id': sub_results.get(traci.constants.VAR_ROAD_ID, ''),
                        'is_icv': self._is_icv_vehicle(veh_id),
                        'id': veh_id
                    }
            except Exception as e:
                logger.warning(f"获取车辆 {veh_id} 数据失败: {e}")
        
        return vehicle_data
    
    def _get_vehicle_data_direct(self) -> Dict[str, Dict[str, Any]]:
        """
        直接获取车辆数据（不使用批量订阅）
        
        Returns:
            vehicle_data: 车辆数据字典
        """
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
                logger.warning(f"获取车辆 {veh_id} 数据失败: {e}")
        
        return vehicle_data
    
    def _build_graph(self, vehicle_data: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        构建图神经网络输入
        
        Args:
            vehicle_data: 车辆数据字典
            
        Returns:
            graph_data: 图数据字典
        """
        vehicle_ids = list(vehicle_data.keys())
        n_vehicles = len(vehicle_ids)
        
        if n_vehicles == 0:
            return {
                'node_features': np.zeros((0, 9), dtype=np.float32),
                'edge_indices': np.zeros((2, 0), dtype=np.int64),
                'edge_features': np.zeros((0, 4), dtype=np.float32),
                'is_icv': np.zeros(0, dtype=np.bool_)
            }
        
        # 1. 构建节点特征 [N, 9]
        node_features = []
        is_icv_list = []
        
        for veh_id in vehicle_ids:
            vehicle = vehicle_data[veh_id]
            
            # 节点特征: [位置, 速度, 加速度, 车道, 剩余距离, 完成率, 类型, 时间, 步长]
            features = [
                vehicle.get('position', 0.0),
                vehicle.get('speed', 0.0),
                vehicle.get('acceleration', 0.0),
                vehicle.get('lane_index', 0),
                1000.0,  # 剩余距离（简化）
                0.5,     # 完成率（简化）
                1.0 if vehicle.get('is_icv', False) else 0.0,
                self.current_step * 0.1,
                0.1
            ]
            
            node_features.append(features)
            is_icv_list.append(vehicle.get('is_icv', False))
        
        node_features = np.array(node_features, dtype=np.float32)
        is_icv = np.array(is_icv_list, dtype=np.bool_)
        
        # 2. 构建边索引和边特征
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
                if distance < 50:  # 50米内建立边
                    edge_indices.append([i, j])
                    
                    rel_distance = distance
                    rel_speed = abs(speed_i - speed_j)
                    
                    # 计算TTC和THW
                    ttc = rel_distance / max(rel_speed, 0.1) if rel_speed > 0 else 100
                    thw = rel_distance / max(speed_i, 0.1) if speed_i > 0 else 100
                    
                    edge_features.append([rel_distance, rel_speed, min(ttc, 10), min(thw, 10)])
        
        edge_indices = np.array(edge_indices, dtype=np.int64).T if edge_indices else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 4), dtype=np.float32)
        
        return {
            'node_features': node_features,
            'edge_indices': edge_indices,
            'edge_features': edge_features,
            'is_icv': is_icv
        }
    
    def _compute_global_metrics(self, vehicle_data: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        计算全局交通指标
        
        Args:
            vehicle_data: 车辆数据字典
            
        Returns:
            metrics: 16维全局指标
        """
        if not vehicle_data:
            return np.zeros(16, dtype=np.float32)
        
        vehicle_list = list(vehicle_data.values())
        speeds = [v['speed'] for v in vehicle_list]
        positions = [v['position'] for v in vehicle_list]
        accelerations = [v.get('acceleration', 0.0) for v in vehicle_list]
        
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        avg_accel = np.mean(np.abs(accelerations))
        vehicle_count = len(vehicle_list)
        
        # ICV统计
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
            self.current_step * 1.0,  # 当前时间
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
        
        return np.array(metrics, dtype=np.float32)
    
    def _run_controller_inference(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行TrafficController推理
        
        Args:
            observation: 观测数据
            
        Returns:
            controller_output: 控制器输出
        """
        # 构建批次数据
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
        
        # 运行推理
        with torch.no_grad():
            controller_output = self.traffic_controller(batch, self.current_step)
        
        return controller_output
    
    def _apply_actions(
        self,
        selected_vehicle_ids: List[str],
        safe_actions: Dict[str, Any]
    ):
        """
        应用控制动作到SUMO环境
        
        Args:
            selected_vehicle_ids: 选中的车辆ID列表
            safe_actions: 安全动作字典
        """
        if not TRACI_AVAILABLE or not selected_vehicle_ids:
            return
        
        actions = safe_actions.get('actions', [])
        
        for i, veh_id in enumerate(selected_vehicle_ids):
            if i >= len(actions):
                continue
            
            try:
                action_vec = actions[i]
                if isinstance(action_vec, torch.Tensor):
                    action_vec = action_vec.cpu().numpy()
                
                # 应用加速度
                accel_action = action_vec[0] if len(action_vec) > 0 else 0.0
                current_speed = traci.vehicle.getSpeed(veh_id)
                new_speed = max(0.0, current_speed + accel_action * 0.1)
                
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, new_speed)
                
                # 应用换道（如果有）
                if len(action_vec) > 1:
                    lane_change_prob = action_vec[1]
                    if lane_change_prob > 0.5:
                        # 尝试换道
                        current_lane = traci.vehicle.getLaneIndex(veh_id)
                        road_id = traci.vehicle.getRoadID(veh_id)
                        lane_count = traci.edge.getLaneNumber(road_id)
                        
                        if lane_count > 1:
                            target_lane = (current_lane + 1) % lane_count
                            traci.vehicle.changeLane(veh_id, target_lane, 1.0)
                
            except Exception as e:
                logger.warning(f"应用动作到车辆 {veh_id} 失败: {e}")
    
    def _calculate_reward(self, observation: Dict[str, Any]) -> float:
        """
        计算奖励 - 与train.py中的计算逻辑一致
        
        考虑：流量效率、安全、稳定性、控制成本
        
        Args:
            observation: 观测数据
            
        Returns:
            reward: 奖励值
        """
        vehicle_data = observation.get('vehicle_data', {})
        
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
    
    def _calculate_safety_metrics(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算安全指标（TTC、THW等）
        
        Args:
            observation: 观测数据
            
        Returns:
            safety_metrics: 安全指标字典
        """
        vehicle_data = observation.get('vehicle_data', {})
        
        if not vehicle_data:
            return self.safety_metrics.copy()
        
        ttc_threshold = self.model_config.get('ttc_threshold', 2.0)
        thw_threshold = self.model_config.get('thw_threshold', 1.5)
        
        ttc_violations = 0
        thw_violations = 0
        speed_violations = 0
        accel_violations = 0
        
        vehicle_ids = list(vehicle_data.keys())
        
        for i, veh_id_i in enumerate(vehicle_ids):
            vehicle_i = vehicle_data[veh_id_i]
            
            # 检查速度违规
            speed = vehicle_i.get('speed', 0.0)
            if speed > 35.0:
                speed_violations += 1
            
            # 检查加速度违规
            accel = vehicle_i.get('acceleration', 0.0)
            if accel < -4.0 or accel > 3.0:
                accel_violations += 1
            
            # 计算与其他车辆的TTC和THW
            for j, veh_id_j in enumerate(vehicle_ids):
                if i == j:
                    continue
                
                vehicle_j = vehicle_data[veh_id_j]
                
                pos_i = vehicle_i.get('position', 0.0)
                pos_j = vehicle_j.get('position', 0.0)
                speed_i = vehicle_i.get('speed', 0.0)
                speed_j = vehicle_j.get('speed', 0.0)
                
                distance = abs(pos_i - pos_j)
                rel_speed = abs(speed_i - speed_j)
                
                # TTC
                if rel_speed > 0:
                    ttc = distance / rel_speed
                    if ttc < ttc_threshold:
                        ttc_violations += 1
                
                # THW
                if speed_i > 0:
                    thw = distance / speed_i
                    if thw < thw_threshold:
                        thw_violations += 1
        
        # 更新累积指标
        self.safety_metrics['ttc_violations'] += ttc_violations
        self.safety_metrics['thw_violations'] += thw_violations
        self.safety_metrics['speed_violations'] += speed_violations
        self.safety_metrics['accel_violations'] += accel_violations
        
        return {
            'ttc_violations': ttc_violations,
            'thw_violations': thw_violations,
            'speed_violations': speed_violations,
            'accel_violations': accel_violations,
            'cumulative': self.safety_metrics.copy()
        }
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """
        检查终止条件
        
        Returns:
            done: 是否自然结束
            truncated: 是否被截断
        """
        done = False
        truncated = False
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            truncated = True
        
        # 检查是否没有车辆（仿真结束）
        if TRACI_AVAILABLE:
            min_expected = traci.simulation.getMinExpectedNumber()
            if min_expected <= 0 and self.current_step > 100:
                done = True
        
        return done, truncated
    
    def _is_icv_vehicle(self, veh_id: str) -> bool:
        """
        判断车辆是否为ICV（智能网联车）
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            is_icv: 是否为ICV
        """
        if not TRACI_AVAILABLE:
            return False
        
        # 方法1: 从车辆类型判断
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
        
        # 方法3: 使用确定性哈希（25% ICV渗透率）
        import hashlib
        hash_value = int(hashlib.md5(veh_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < 25
    
    def _extract_net_file(self) -> str:
        """从SUMO配置文件提取网络文件路径"""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(self.sumo_cfg_path)
            root = tree.getroot()
            config_dir = os.path.dirname(self.sumo_cfg_path)
            
            for input_elem in root.findall('.//input'):
                net_file = input_elem.find('net-file')
                if net_file is not None:
                    net_file_path = net_file.get('value')
                    if not os.path.isabs(net_file_path):
                        net_file_path = os.path.join(config_dir, net_file_path)
                    return net_file_path
        except Exception as e:
            logger.warning(f"提取网络文件失败: {e}")
        
        return ""
    
    def _extract_route_file(self) -> str:
        """从SUMO配置文件提取路径文件路径"""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(self.sumo_cfg_path)
            root = tree.getroot()
            config_dir = os.path.dirname(self.sumo_cfg_path)
            
            for input_elem in root.findall('.//input'):
                route_files = input_elem.find('route-files')
                if route_files is not None:
                    route_file_path = route_files.get('value')
                    if not os.path.isabs(route_file_path):
                        route_file_path = os.path.join(config_dir, route_file_path)
                    return route_file_path
        except Exception as e:
            logger.warning(f"提取路径文件失败: {e}")
        
        return ""
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        获取episode统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'total_steps': self.current_step,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.current_step, 1),
            'vehicle_count': len(self.vehicle_ids),
            'safety_metrics': self.safety_metrics.copy()
        }


def create_sumo_gym_env(
    sumo_cfg_path: str,
    **kwargs
) -> SUMOGymEnv:
    """
    创建SUMO Gymnasium环境的工厂函数
    
    Args:
        sumo_cfg_path: SUMO配置文件路径
        **kwargs: 其他参数
        
    Returns:
        env: SUMO Gymnasium环境实例
    """
    return SUMOGymEnv(sumo_cfg_path=sumo_cfg_path, **kwargs)


def main():
    """主函数 - 演示SUMO Gymnasium环境"""
    print("=" * 60)
    print("🚀 SUMO Gymnasium环境演示")
    print("=" * 60)
    
    # 创建环境
    env = create_sumo_gym_env(
        sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
        use_libsumo=False,
        batch_subscribe=True,
        device='cpu',
        max_steps=100,
        use_gui=False
    )
    
    try:
        # 重置环境
        observation, info = env.reset()
        
        print(f"\n初始观测:")
        print(f"  车辆数: {len(observation['vehicle_ids'])}")
        print(f"  节点特征形状: {observation['node_features'].shape}")
        print(f"  边索引形状: {observation['edge_indices'].shape}")
        print(f"  全局指标: {observation['global_metrics'][:4]}")
        
        # 运行几个步骤
        for step in range(10):
            # 执行一步（使用TrafficController推理）
            observation, reward, done, truncated, info = env.step()
            
            print(f"\n[Step {step+1}]")
            print(f"  奖励: {reward:.4f}")
            print(f"  车辆数: {info['vehicle_count']}")
            print(f"  总奖励: {info['total_reward']:.2f}")
            print(f"  安全指标: {info['safety_metrics']}")
            
            if done or truncated:
                print(f"\n环境结束: done={done}, truncated={truncated}")
                break
        
        # 打印统计信息
        stats = env.get_episode_statistics()
        print(f"\n{'='*60}")
        print("📊 Episode统计")
        print(f"{'='*60}")
        print(f"总步数: {stats['total_steps']}")
        print(f"总奖励: {stats['total_reward']:.2f}")
        print(f"平均奖励: {stats['avg_reward']:.4f}")
        print(f"安全指标: {stats['safety_metrics']}")
        print(f"{'='*60}")
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
