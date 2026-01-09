# 智能交通协同控制系统

本项目实现了一个基于深度学习的智能交通协同控制系统，采用了先进的图神经网络（GNN）、世界模型和强化学习技术，旨在优化交通流量并提高道路安全性。

## 项目结构

```
TJ_transport_v3/
├── neural_traffic_controller.py      # 主要的神经网络控制器实现
├── main.py                          # 改进的主程序，集成神经控制器
├── sumo_env_adapter.py              # SUMO环境适配器
├── train.py                         # 传统训练脚本
├── ray_train.py                     # Ray RLlib分布式训练脚本
├── sumo_rl_train.py                 # SUMO-RL多实例训练脚本
├── run_ray_training.py              # Ray训练启动脚本
├── train_config.json                # 训练配置文件
├── 仿真环境-初赛/                     # 原始仿真环境
│   ├── main.py
│   ├── net.xml
│   ├── readme.md
│   ├── routes.xml
│   └── sumo.sumocfg
├── 详细要求.md                       # 项目详细要求
└── 赛题.md                          # 竞赛题目
```

## 核心特性

### 1. 风险敏感图神经网络 (Risk-Sensitive GNN)
- 采用图神经网络处理车辆之间的复杂交互关系
- 在边特征中嵌入TTC（碰撞时间）和THW（车头时距）倒数
- 采用Biased Attention机制强化高风险交互的注意力权重

### 2. 渐进式世界模型 (Progressive World Model)
- 分两个阶段训练：
  - Phase 1：仅预测下一时刻车辆状态（位置、速度），学习基础动力学
  - Phase 2：冻结特征提取器，解耦输出为流演化与风险演化，联合优化轨迹MSE与冲突分类损失

### 3. 影响力驱动的Top-K稀疏控制机制
- 动态计算每辆车的影响力得分
- 仅对Top-K（如K=5）关键车辆执行强化学习控制
- 其余车辆使用IDM模型跟驰，控制25%的智能车辆

### 4. 双模态安全屏障
- Level 1：动作裁剪（加速度限幅、速度非负）
- Level 2：TTC < 2.0s时强制紧急制动，并在训练中施加巨大负奖励

## 使用方法

### 1. 运行仿真

```bash
python main.py
```

或者指定配置文件：

```bash
python main.py 仿真环境-初赛/sumo.sumocfg
```

### 2. 传统训练方式

```bash
python train.py
```

### 3. Ray RLlib分布式并行训练

首先安装Ray RLlib：

```bash
pip install ray[rllib]
```

然后运行并行训练：

```bash
python run_ray_training.py
```

或者指定配置文件：

```bash
python run_ray_training.py --config train_config.json --sumo-config 仿真环境-初赛/sumo.sumocfg
```

### 4. SUMO-RL多实例训练（推荐）

使用SUMO-RL实现多SUMO实例并行训练：

```bash
python sumo_rl_train.py
```

### 5. 配置参数

可以在[train_config.json](train_config.json)中修改训练参数：

- `phase1_epochs`: 第一阶段训练轮数
- `phase2_epochs`: 第二阶段训练轮数
- `phase3_epochs`: 第三阶段训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `top_k`: 同时控制的车辆数
- `device`: 计算设备（cpu或cuda）
- `parallel.num_workers`: 并行工作进程数
- `parallel.base_port`: SUMO实例基础端口
- `parallel.env_per_worker`: 每个工作进程的环境数

## 系统要求

- Python 3.7+
- SUMO仿真器
- PyTorch
- PyTorch Geometric
- Gymnasium
- Pandas
- NumPy
- Ray RLlib (用于并行训练)

## 技术细节

### 感知层
- 使用GATConv实现图注意力机制
- 节点特征包括位置、速度、加速度等9维特征
- 边特征包括相对距离、相对速度、TTC和THW

### 预测层
- LSTM网络预测未来状态
- 解耦流演化和风险演化
- 支持多步预测（默认5步）

### 决策层
- 融合GNN嵌入、世界模型预测和全局特征
- 计算每辆车的影响力得分
- 选择Top-K最具影响力的车辆进行控制

### 安全层
- 双层级安全机制确保驾驶安全
- Level 1: 动作裁剪和限幅
- Level 2: 紧急制动干预

## 并行训练架构

### Ray RLlib集成
- 支持多进程并行训练
- 每个进程可运行多个SUMO实例
- 动态端口分配避免冲突
- 高效的数据吞吐和模型更新

### SUMO-RL多实例训练
- 基于SUMO-RL框架实现多实例并行训练
- 通过不同端口运行多个SUMO实例
- 更高效地生成训练数据
- 显著加快训练速度

### 分布式训练优势
- 提高训练数据吞吐量
- 快速迭代实验
- 支持更大规模的仿真场景

## 评估指标

系统优化目标为最大化总评分：
```
S_total = S_perf × P_int
```
其中：
- `S_perf` 是性能评分
- `P_int` 是干预成本相关系数

## 注意事项

1. 本系统严格遵循竞赛规则，仅控制25%的智能车辆
2. 不修改原始OD路径，仅对车辆速度进行控制
3. 所有安全约束均得到满足
4. 模型设计考虑了实时性要求
5. 并行训练可显著加快模型收敛速度