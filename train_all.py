#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合训练脚本
支持多种训练模式：传统训练、Ray RLlib分布式训练、SUMO-RL多实例训练
"""

import os
import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="智能交通协同控制系统综合训练脚本")
    parser.add_argument("mode", type=str, choices=["traditional", "ray", "sumo-rl"], 
                        help="训练模式: traditional | ray | sumo-rl")
    parser.add_argument("--config", type=str, default="train_config.json",
                        help="训练配置文件路径 (default: train_config.json)")
    parser.add_argument("--sumo-config", type=str, default="仿真环境-初赛/sumo.sumocfg",
                        help="SUMO配置文件路径 (default: 仿真环境-初赛/sumo.sumocfg)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="检查点保存目录 (default: checkpoints)")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    if not os.path.exists(args.sumo_config):
        print(f"❌ SUMO配置文件不存在: {args.sumo_config}")
        return 1
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("智能交通协同控制系统 - 综合训练脚本")
    print("=" * 70)
    print(f"训练模式: {args.mode}")
    print(f"配置文件: {args.config}")
    print(f"SUMO配置: {args.sumo_config}")
    
    if args.mode == "ray" or args.mode == "sumo-rl":
        print(f"工作进程数: {config['parallel']['num_workers']}")
        print(f"每进程环境数: {config['parallel']['env_per_worker']}")
        print(f"总并行环境数: {config['parallel']['num_workers'] * config['parallel']['env_per_worker']}")
        print(f"基础端口: {config['parallel']['base_port']}")
    
    print("=" * 70)
    
    # 根据模式执行训练
    if args.mode == "traditional":
        from train import train_traffic_controller, save_model
        print("开始传统训练...")
        
        trained_model = train_traffic_controller(config)
        save_path = config['training']['save_path']
        save_model(trained_model, config, save_path)
        
    elif args.mode == "ray":
        try:
            import ray
            from ray_train import train_with_ray
            train_with_ray()
        except ImportError as e:
            print(f"❌ 未安装Ray RLlib: {e}")
            print("请先安装Ray RLlib: pip install ray[rllib]")
            return 1
            
    elif args.mode == "sumo-rl":
        try:
            import ray
            from sumo_rl_train import train_with_sumo_rl
            train_with_sumo_rl()
        except ImportError as e:
            print(f"❌ 未安装Ray RLlib: {e}")
            print("请先安装Ray RLlib: pip install ray[rllib]")
            return 1
    
    print("✅ 训练完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())