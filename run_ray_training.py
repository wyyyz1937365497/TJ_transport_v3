#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ray训练启动脚本
用于启动基于Ray RLlib的多SUMO并行训练
"""

import os
import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="启动Ray RLlib多SUMO并行训练")
    parser.add_argument("--config", type=str, default="train_config.json",
                        help="训练配置文件路径 (default: train_config.json)")
    parser.add_argument("--sumo-config", type=str, default="仿真环境-初赛/sumo.sumocfg",
                        help="SUMO配置文件路径 (default: 仿真环境-初赛/sumo.sumocfg)")
    
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
    
    print("=" * 60)
    print("智能交通协同控制系统 - Ray RLlib并行训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"SUMO配置: {args.sumo_config}")
    print(f"工作进程数: {config['parallel']['num_workers']}")
    print(f"每进程环境数: {config['parallel']['env_per_worker']}")
    print(f"总并行环境数: {config['parallel']['num_workers'] * config['parallel']['env_per_worker']}")
    print("=" * 60)
    
    # 导入Ray并开始训练
    try:
        import ray
        from ray_train import train_with_ray
        train_with_ray()
    except ImportError as e:
        print(f"❌ 未安装Ray RLlib: {e}")
        print("请先安装Ray RLlib: pip install ray[rllib]")
        return 1
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("✅ 训练完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())