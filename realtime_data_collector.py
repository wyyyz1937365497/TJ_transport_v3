"""
实时SUMO数据收集系统
在SUMO仿真运行时在线收集交通数据，直接馈送到训练循环
支持主动车辆调度和控制干预
"""

import numpy as np
import traci
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from collections import deque
from threading import Thread, Lock
import time


class RealtimeDataCollector:
    """
    实时数据收集器
    在SUMO仿真运行时收集交通数据
    """
    
    def __init__(self, 
                 sumo_cfg_path: str,
                 max_buffer_size: int = 10000,
                 use_gui: bool = False):
        """
        初始化实时数据收集器
        
        Args:
            sumo_cfg_path: SUMO配置文件路径
            max_buffer_size: 数据缓冲区最大大小
            use_gui: 是否使用GUI
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.max_buffer_size = max_buffer_size
        self.use_gui = use_gui
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=max_buffer_size)
        self.buffer_lock = Lock()
        
        # 仿真状态
        self.connected = False
        self.current_step = 0
        self.vehicle_ids = []
        
        # 统计信息
        self.collected_samples = 0
        self.start_time = None
        
        # 车辆调度配置
        self.vehicle_schedule = {}
        self.control_interventions = {}
        
        # ICV配置
        self.icv_vehicles = set()
        self.icv_penetration_rate = 0.25
        
        print(f"✅ 实时数据收集器初始化完成")
        print(f"   配置文件: {sumo_cfg_path}")
        print(f"   缓冲区大小: {max_buffer_size}")
        print(f"   使用GUI: {False}")

    def is_ready_for_training(self) -> bool:
        """
        检查是否准备好进行训练
        
        Returns:
            ready: 是否准备好
        """
        # 使用最小样本数阈值，参考 OnlineTrainingDataGenerator 中的默认值
        min_samples_for_training = 1000
        with self.buffer_lock:
            return len(self.data_buffer) >= min_samples_for_training

    def connect(self):
        """连接到SUMO仿真"""
        if self.connected:
            return
        
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]
        
        try:
            traci.start(sumo_cmd)
            self.connected = True
            self.start_time = time.time()
            print(f"🚀 已连接到SUMO仿真")
        except Exception as e:
            print(f"❌ 连接SUMO失败: {e}")
            raise
    
    def disconnect(self):
        """断开SUMO连接"""
        if self.connected:
            try:
                traci.close()
                self.connected = False
                print(f"✅ 已断开SUMO连接")
            except Exception as e:
                print(f"⚠️  断开SUMO时出错: {e}")
    
    def collect_step(self, apply_interventions: bool = True) -> Optional[Dict[str, Any]]:
        """
        收集当前步的数据
        
        Args:
            apply_interventions: 是否应用控制干预
            
        Returns:
            step_data: 当前步的数据字典
        """
        if not self.connected:
            return None
        
        try:
            # 执行仿真步
            traci.simulationStep()
            self.current_step += 1
            
            # 获取车辆列表
            self.vehicle_ids = traci.vehicle.getIDList()
            
            if not self.vehicle_ids:
                return None
            
            # 收集车辆数据
            vehicle_data = {}
            for veh_id in self.vehicle_ids:
                try:
                    vehicle_data[veh_id] = self._collect_vehicle_data(veh_id)
                except Exception as e:
                    continue
            
            # 应用控制干预
            if apply_interventions:
                self._apply_control_interventions(vehicle_data)
            
            # 计算全局指标
            global_metrics = self._compute_global_metrics(vehicle_data)
            
            # 构建步数据
            step_data = {
                'vehicle_data': vehicle_data,
                'global_metrics': global_metrics,
                'vehicle_ids': list(vehicle_data.keys()),
                'step': self.current_step,
                'timestamp': time.time() - self.start_time if self.start_time else 0.0
            }
            
            # 添加到缓冲区
            with self.buffer_lock:
                self.data_buffer.append(step_data)
                self.collected_samples += 1
            
            return step_data
            
        except Exception as e:
            print(f"⚠️  收集步 {self.current_step} 数据时出错: {e}")
            return None
    
    def _collect_vehicle_data(self, veh_id: str) -> Dict[str, Any]:
        """
        收集单个车辆的数据
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            vehicle_data: 车辆数据字典
        """
        # 基础状态
        position = traci.vehicle.getLanePosition(veh_id)
        speed = traci.vehicle.getSpeed(veh_id)
        acceleration = traci.vehicle.getAcceleration(veh_id)
        lane_index = traci.vehicle.getLaneIndex(veh_id)
        lane_id = traci.vehicle.getLaneID(veh_id)
        road_id = traci.vehicle.getRoadID(veh_id)
        
        # 判断是否为ICV
        is_icv = self._is_icv_vehicle(veh_id)
        
        # 计算剩余距离和完成率
        route_length = traci.vehicle.getRouteLength(veh_id)
        remaining_distance = max(0.0, route_length - position)
        completion_rate = position / max(route_length, 1.0) if route_length > 0 else 0.0
        
        # 获取前车信息（用于TTC和THW计算）
        leader = self._find_leader(veh_id, lane_id, position)
        ttc, thw = self._calculate_safety_metrics(veh_id, leader, speed, position)
        
        return {
            'id': veh_id,
            'position': position,
            'speed': speed,
            'acceleration': acceleration,
            'lane_index': lane_index,
            'lane_id': lane_id,
            'road_id': road_id,
            'is_icv': is_icv,
            'remaining_distance': remaining_distance,
            'completion_rate': completion_rate,
            'ttc': ttc,
            'thw': thw,
            'leader_id': leader['id'] if leader else None
        }
    
    def _is_icv_vehicle(self, veh_id: str) -> bool:
        """
        判断车辆是否为ICV
        
        Args:
            veh_id: 车辆ID
            
        Returns:
            is_icv: 是否为ICV
        """
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
        
        # 方法3: 使用确定性哈希（用于演示）
        import hashlib
        hash_value = int(hashlib.md5(veh_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < self.icv_penetration_rate * 100
    
    def _find_leader(self, veh_id: str, lane_id: str, position: float) -> Optional[Dict[str, Any]]:
        """
        找到前车
        
        Args:
            veh_id: 自车ID
            lane_id: 车道ID
            position: 自车位置
            
        Returns:
            leader: 前车信息
        """
        min_distance = float('inf')
        leader = None
        
        for other_id in self.vehicle_ids:
            if other_id == veh_id:
                continue
            
            try:
                other_lane = traci.vehicle.getLaneID(other_id)
                if other_lane != lane_id:
                    continue
                
                other_pos = traci.vehicle.getLanePosition(other_id)
                if other_pos <= position:
                    continue
                
                distance = other_pos - position
                if distance < min_distance:
                    min_distance = distance
                    leader = {
                        'id': other_id,
                        'position': other_pos,
                        'distance': distance
                    }
            except:
                continue
        
        return leader if min_distance < 100 else None
    
    def _calculate_safety_metrics(self, veh_id: str, leader: Optional[Dict[str, Any]], 
                               speed: float, position: float) -> Tuple[float, float]:
        """
        计算安全指标（TTC和THW）
        
        Args:
            veh_id: 自车ID
            leader: 前车信息
            speed: 自车速度
            position: 自车位置
            
        Returns:
            ttc: 碰撞时间
            thw: 车头时距
        """
        if leader is None:
            return float('inf'), float('inf')
        
        leader_speed = traci.vehicle.getSpeed(leader['id'])
        distance = leader['distance']
        
        # 计算TTC
        relative_speed = speed - leader_speed
        if relative_speed > 0:
            ttc = distance / relative_speed
        else:
            ttc = float('inf')
        
        # 计算THW
        if speed > 0:
            thw = distance / speed
        else:
            thw = float('inf')
        
        return max(0.1, ttc), max(0.1, thw)
    
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
        
        speeds = [v['speed'] for v in vehicle_data.values()]
        accelerations = [v['acceleration'] for v in vehicle_data.values()]
        positions = [v['position'] for v in vehicle_data.values()]
        
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        avg_accel = np.mean(accelerations)
        vehicle_count = len(vehicle_data)
        
        # ICV统计
        icv_vehicles = [v for v in vehicle_data.values() if v['is_icv']]
        hv_vehicles = [v for v in vehicle_data.values() if not v['is_icv']]
        
        icv_count = len(icv_vehicles)
        hv_count = len(hv_vehicles)
        
        icv_total_speed = sum([v['speed'] for v in icv_vehicles])
        hv_total_speed = sum([v['speed'] for v in hv_vehicles])
        
        # 安全统计
        ttcs = [v['ttc'] for v in vehicle_data.values() if v['ttc'] < float('inf')]
        thws = [v['thw'] for v in vehicle_data.values() if v['thw'] < float('inf')]
        
        avg_ttc = np.mean(ttcs) if ttcs else float('inf')
        avg_thw = np.mean(thws) if thws else float('inf')
        
        # 16维全局指标
        metrics = [
            avg_speed,
            speed_std,
            avg_accel,
            float(vehicle_count),
            self.current_step * 0.1,  # 当前时间
            min(positions) if positions else 0.0,
            max(positions) if positions else 0.0,
            np.mean(positions) if positions else 0.0,
            float(icv_count),
            float(hv_count),
            icv_total_speed,
            hv_total_speed,
            avg_speed * vehicle_count,  # 总流量
            speed_std * vehicle_count,  # 总波动
            avg_accel * vehicle_count,  # 总加速度
            self.current_step % 100  # 周期性特征
        ]
        
        return metrics
    
    def _apply_control_interventions(self, vehicle_data: Dict[str, Any]):
        """
        应用控制干预
        
        Args:
            vehicle_data: 车辆数据字典
        """
        if not self.control_interventions:
            return
        
        step_key = str(self.current_step)
        if step_key not in self.control_interventions:
            return
        
        interventions = self.control_interventions[step_key]
        
        for veh_id, intervention in interventions.items():
            if veh_id not in vehicle_data:
                continue
            
            try:
                # 应用加速度干预
                if 'acceleration' in intervention:
                    target_accel = intervention['acceleration']
                    current_speed = vehicle_data[veh_id]['speed']
                    new_speed = max(0.0, current_speed + target_accel * 0.1)
                    
                    traci.vehicle.setSpeedMode(veh_id, 0)
                    traci.vehicle.setSpeed(veh_id, new_speed)
                
                # 应用换道干预
                if 'lane_change' in intervention and intervention['lane_change']:
                    current_lane = vehicle_data[veh_id]['lane_index']
                    direction = intervention.get('direction', 1)  # 1: 右, -1: 左
                    
                    try:
                        traci.vehicle.changeLane(veh_id, current_lane + direction, 0.1)
                    except:
                        pass
            except Exception as e:
                continue
    
    def set_control_intervention(self, step: int, veh_id: str, 
                             acceleration: float = None, lane_change: bool = False,
                             direction: int = 1):
        """
        设置控制干预
        
        Args:
            step: 步数
            veh_id: 车辆ID
            acceleration: 加速度干预
            lane_change: 是否换道
            direction: 换道方向
        """
        step_key = str(step)
        if step_key not in self.control_interventions:
            self.control_interventions[step_key] = {}
        
        self.control_interventions[step_key][veh_id] = {}
        
        if acceleration is not None:
            self.control_interventions[step_key][veh_id]['acceleration'] = acceleration
        
        if lane_change:
            self.control_interventions[step_key][veh_id]['lane_change'] = lane_change
            self.control_interventions[step_key][veh_id]['direction'] = direction
    
    def get_buffer_data(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """
        从缓冲区获取数据
        
        Args:
            num_samples: 获取的样本数，None表示全部
            
        Returns:
            data: 数据列表
        """
        with self.buffer_lock:
            if num_samples is None:
                return list(self.data_buffer)
            else:
                return list(self.data_buffer)[-num_samples:]
    
    def clear_buffer(self):
        """清空数据缓冲区"""
        with self.buffer_lock:
            self.data_buffer.clear()
            print(f"🗑️  数据缓冲区已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取收集统计信息
        
        Returns:
            stats: 统计信息字典
        """
        with self.buffer_lock:
            buffer_size = len(self.data_buffer)
        
        return {
            'collected_samples': self.collected_samples,
            'buffer_size': buffer_size,
            'current_step': self.current_step,
            'vehicle_count': len(self.vehicle_ids),
            'icv_count': len([v for v in self.vehicle_ids if self._is_icv_vehicle(v)]),
            'collection_time': time.time() - self.start_time if self.start_time else 0.0
        }
    
    def save_buffer(self, save_path: str):
        """
        保存缓冲区数据到文件
        
        Args:
            save_path: 保存路径
        """
        with self.buffer_lock:
            data = list(self.data_buffer)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存数据
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 已保存 {len(data)} 个样本到: {save_path}")


class OnlineTrainingDataGenerator:
    """
    在线训练数据生成器
    将实时数据收集器转换为训练数据格式
    """
    
    def __init__(self, data_collector: RealtimeDataCollector):
        """
        初始化在线训练数据生成器
        
        Args:
            data_collector: 实时数据收集器
        """
        self.data_collector = data_collector
        self.min_samples_for_training = 1000
        
        print(f"✅ 在线训练数据生成器初始化完成")
    
    def generate_training_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        生成训练批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 训练批次数据列表
        """
        buffer_data = self.data_collector.get_buffer_data()
        
        if len(buffer_data) < self.min_samples_for_training:
            print(f"⚠️  缓冲区数据不足: {len(buffer_data)} < {self.min_samples_for_training}")
            return []
        
        # 随机采样
        indices = np.random.choice(len(buffer_data), 
                               size=min(batch_size, len(buffer_data)),
                               replace=False)
        
        batch = [buffer_data[i] for i in indices]
        
        return batch
    
    def is_ready_for_training(self) -> bool:
        """
        检查是否准备好进行训练
        
        Returns:
            ready: 是否准备好
        """
        buffer_data = self.data_collector.get_buffer_data()
        return len(buffer_data) >= self.min_samples_for_training


def main():
    """主函数 - 演示实时数据收集"""
    print("🚀 实时SUMO数据收集系统演示")
    
    # 创建数据收集器
    collector = RealtimeDataCollector(
        sumo_cfg_path='仿真环境-初赛/sumo.sumocfg',
        max_buffer_size=10000,
        use_gui=False
    )
    
    # 连接到SUMO
    collector.connect()
    
    try:
        # 收集数据
        for step in range(1000):
            step_data = collector.collect_step(apply_interventions=False)
            
            if step_data is None:
                print(f"⚠️  步 {step}: 无车辆数据")
                continue
            
            # 每100步输出统计
            if step % 100 == 0:
                stats = collector.get_statistics()
                print(f"\n[Step {step}] 统计:")
                print(f"  收集样本: {stats['collected_samples']}")
                print(f"  缓冲区大小: {stats['buffer_size']}")
                print(f"  车辆数: {stats['vehicle_count']}")
                print(f"  ICV数: {stats['icv_count']}")
            
            # 测试控制干预
            if step == 500:
                print("\n🔧 测试控制干预...")
                # 为前3辆ICV设置干预
                icv_ids = [v for v in collector.vehicle_ids 
                           if collector._is_icv_vehicle(v)][:3]
                for i, veh_id in enumerate(icv_ids):
                    collector.set_control_intervention(
                        step=step + 1,
                        veh_id=veh_id,
                        acceleration=-2.0 if i == 0 else 1.0,
                        lane_change=(i == 1),
                        direction=1
                    )
                print(f"  已设置 {len(icv_ids)} 个控制干预")
        
        # 保存数据
        save_path = 'results/realtime_collected_data.json'
        collector.save_buffer(save_path)
        
        # 打印最终统计
        stats = collector.get_statistics()
        print(f"\n{'='*60}")
        print("📊 收集完成统计")
        print(f"{'='*60}")
        print(f"总收集样本: {stats['collected_samples']}")
        print(f"缓冲区大小: {stats['buffer_size']}")
        print(f"仿真步数: {stats['current_step']}")
        print(f"收集时间: {stats['collection_time']:.2f}s")
        print(f"{'='*60}")
        
    finally:
        # 断开连接
        collector.disconnect()


if __name__ == "__main__":
    main()
