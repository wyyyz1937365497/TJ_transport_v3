# 智能交通协同控制Agent v5.0

基于Ray RLlib的分布式训练架构，实现了三阶段课程学习方案。

## 架构概述

本项目采用分层架构设计，包含以下主要组件：

- **CustomSumoEnv**: 高性能SUMO环境封装，集成感知、预测和决策层
- **TrafficTorchModel**: 核心神经网络模型，实现v4.0架构
- **RiskSensitiveGNN**: 风险敏感图神经网络
- **ProgressiveWorldModel**: 渐进式世界模型
- **ConstrainedPPO**: 约束优化PPO算法

## 三阶段训练流程

### 阶段1：世界模型预训练
- 数据收集：使用随机策略和IDM策略收集交通数据
- 模型训练：学习交通环境的动力学模型

### 阶段2：安全RL训练
- 在预训练世界模型基础上训练RL策略
- 集成安全屏障机制

### 阶段3：约束优化训练
- 引入拉格朗日约束优化
- 实现性能与安全的平衡

## 项目结构

```
traffic_agent/
├── src/
│   ├── env/
│   │   └── custom_sumo_env.py      # 环境封装
│   ├── models/
│   │   ├── traffic_torch_model.py  # 核心模型
│   │   ├── risk_sensitive_gnn.py   # 风险敏感GNN
│   │   └── progressive_world_model.py  # 世界模型
│   ├── perception/
│   │   └── risk_sensitive_gnn.py   # 感知层
│   ├── prediction/
│   │   └── progressive_world_model.py  # 预测层
│   ├── decision/
│   │   └── dual_mode_safety_shield.py  # 安全屏障
│   ├── dataset/
│   │   ├── trajectory_buffer.py    # 数据缓冲区
│   │   └── data_collector.py       # 数据收集器
│   ├── policies/
│   │   └── constrained_ppo_policy.py  # 约束PPO策略
│   └── utils/
│       └── training_callbacks.py   # 训练回调
├── configs/
│   └── training_pipeline_config.json  # 训练配置
├── scripts/
│   ├── train_full_pipeline.py      # 完整训练管道
│   └── train_world_model.py        # 世界模型训练
├── main.py                         # 主训练脚本
├── requirements.txt                # 依赖包
└── README.md
```

## 安装说明

```bash
pip install -r requirements.txt
```

## 运行说明

### 运行完整三阶段训练

```bash
python main.py
```

### 运行特定阶段

```bash
# 运行第一阶段
python scripts/train_full_pipeline.py --phase 1

# 运行第二阶段
python scripts/train_full_pipeline.py --phase 2

# 运行第三阶段
python scripts/train_full_pipeline.py --phase 3
```

## 性能特点

- **高效分布式训练**: 基于Ray RLlib实现，支持多GPU和多节点训练
- **安全强化学习**: 集成双重安全屏障机制
- **约束优化**: 采用拉格朗日乘子法处理安全约束
- **课程学习**: 三阶段渐进式训练策略

## 技术规范

- Python >= 3.8
- PyTorch >= 1.13.0
- Ray RLlib >= 2.9.3
- SUMO >= 1.0.0

## 许可证

本项目仅供学术研究使用。