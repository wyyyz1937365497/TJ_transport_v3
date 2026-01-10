# 智能交通协同控制系统

基于深度强化学习的智能交通协同控制系统，用于优化城市快速路的交通效率。

## 系统架构

本系统采用分层架构设计，包含以下核心组件：

### 1. 感知层：Risk-Sensitive GNN
- **功能**：风险敏感图神经网络，捕捉车辆间的交互关系
- **特点**：
  - 在边特征中嵌入TTC（碰撞时间）和THW（车头时距）倒数
  - 采用Biased Attention机制强化高风险交互的注意力权重
  - 输出256维全局嵌入向量

### 2. 预测层：Progressive World Model
- **功能**：渐进式世界模型，预测未来交通状态
- **训练阶段**：
  - Phase 1：仅预测下一时刻车辆状态，学习基础动力学
  - Phase 2：预测未来5步状态 + 冲突概率
- **输出**：
  - z_flow：流演化
  - z_risk：风险演化

### 3. 决策层：Influence-Driven Controller
- **功能**：影响力驱动的Top-K稀疏控制机制
- **核心算法**：
  - 动态计算每辆车的影响力得分：`Score_i = α·Importance_GNN + β·Impact_Predicted`
  - 仅对Top-K（如K=5）关键车辆执行强化学习控制
  - 其余车辆使用IDM模型跟驰

### 4. 安全层：Dual-mode Safety Shield
- **Level 1**：动作裁剪（加速度限幅、速度非负）
- **Level 2**：TTC < 2.0s时强制紧急制动
- **特点**：在训练中施加巨大负奖励以避免进入高危状态

## 文件结构

```
.
├── neural_traffic_controller.py   # 主控制器
├── risk_sensitive_gnn.py         # 风险敏感GNN
├── progressive_world_model.py     # 渐进式世界模型
├── influence_controller.py        # 影响力驱动控制器
├── safety_shield.py             # 双模态安全屏障
├── sumo_integration.py          # SUMO集成
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── config.json                 # 配置文件
├── requirements.txt            # 依赖包
└── README.md                   # 本文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

训练过程分为三个阶段：
- **Phase 1**：世界模型预训练（50 epochs）
- **Phase 2**：安全RL训练（200 epochs）
- **Phase 3**：约束优化（100 epochs）

### 2. 评估模型

```bash
python evaluate.py
```

评估结果将保存在 `results/evaluation_results.json` 中。

### 3. 使用预训练模型

```python
from sumo_integration import create_sumo_controller

# 创建控制器
controller = create_sumo_controller(config_path='config.json')

# 应用控制
control_results = controller.apply_control(vehicle_data, step)
```

## 配置说明

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `node_dim` | 节点特征维度 | 9 |
| `edge_dim` | 边特征维度 | 4 |
| `gnn_hidden_dim` | GNN隐藏层维度 | 64 |
| `gnn_output_dim` | GNN输出维度 | 256 |
| `gnn_layers` | GNN层数 | 3 |
| `gnn_heads` | GNN注意力头数 | 4 |
| `future_steps` | 预测步数 | 5 |
| `top_k` | 控制车辆数 | 5 |

### 安全参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ttc_threshold` | TTC阈值（秒） | 2.0 |
| `thw_threshold` | THW阈值（秒） | 1.5 |
| `max_accel` | 最大加速度 | 2.0 |
| `max_decel` | 最大减速度 | -3.0 |
| `emergency_decel` | 紧急减速度 | -5.0 |

### 约束参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cost_limit` | 成本限制 | 0.1 |
| `lambda_lr` | 拉格朗日乘子学习率 | 0.01 |
| `alpha` | GNN重要性权重 | 1.0 |
| `beta` | 预测影响力权重 | 5.0 |

## 核心算法

### 影响力计算

每辆车的影响力得分由两部分组成：

```
Score_i = α·Importance_GNN + β·Impact_Predicted
```

- `Importance_GNN`：基于GNN嵌入的重要性得分
- `Impact_Predicted`：基于世界模型预测的影响力得分
- `α` 和 `β`：可学习权重参数

### 安全屏障

双模态安全屏障确保所有控制指令的安全性：

1. **Level 1（软约束）**：
   - 动态调整加速度限制（基于速度）
   - 仅在低速时允许换道

2. **Level 2（硬约束）**：
   - TTC < 2.0s 或 THW < 1.5s 时强制紧急制动
   - 取消所有换道操作

## 评估指标

系统根据以下指标进行评估：

- **效率**：平均速度、吞吐量
- **稳定性**：速度标准差
- **安全性**：碰撞次数、紧急制动次数
- **干预成本**：受控车辆数、干预次数

最终得分：

```
S_total = S_perf × P_int
```

其中：
- `S_perf`：性能得分（效率 + 稳定性）
- `P_int`：干预成本惩罚因子

## 竞赛规则

本系统严格遵守竞赛规则：

1. **仅控制25%的智能车辆**：通过hash(veh_id) % 100 < 25确定ICV
2. **不修改OD路径**：仅控制车辆驾驶行为
3. **不使用外部数据**：所有训练数据在官方仿真环境中生成
4. **按需干预**：仅对Top-K最具影响力的车辆进行控制

## 性能优化

- **GNN缓存**：缓存GNN推理结果，减少重复计算
- **批量处理**：支持批量动作应用
- **事件触发**：仅在特定步数执行控制（默认每5步）

## 扩展性

系统设计支持未来扩展：

1. **复赛阶段**：可扩展至车路一体化协同控制
2. **多场景**：支持不同交通场景的配置
3. **分布式训练**：可与Ray RLlib集成

## 许可证

本项目用于智能交通协同控制竞赛。

## 联系方式

如有问题，请参考详细要求文档或联系竞赛组委会。
