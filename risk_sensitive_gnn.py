"""
感知层：风险敏感图神经网络 (Risk-Sensitive GNN)
在边特征中嵌入 TTC（碰撞时间）和 THW（车头时距）倒数
采用 Biased Attention 机制强化高风险交互的注意力权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class RiskSensitiveGNN(nn.Module):
    """
    风险敏感图神经网络
    输入：车辆节点特征(9维) + 交互边特征(4维)
    输出：256维全局嵌入
    """
    
    def __init__(self, node_dim: int = 9, edge_dim: int = 4, hidden_dim: int = 64, 
                 output_dim: int = 256, num_layers: int = 3, heads: int = 4):
        super().__init__()
        
        # 1. 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 3. 风险注意力机制
        self.risk_attention = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 4. GNN层（使用自定义的Graph Attention层）
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GraphAttentionLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=hidden_dim // 2,
                    concat=False
                )
            )
        
        # 5. 输出投影层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            graph: 包含x, edge_index, edge_attr的图数据字典
        Returns:
            global_embedding: [N, 256] 全局嵌入
        """
        # 1. 编码节点和边特征
        node_features = self.node_encoder(graph['x'])  # [N, 64]
        edge_features = self.edge_encoder(graph['edge_attr'])  # [E, 32]
        
        # 2. 计算风险注意力权重
        if edge_features.size(0) > 0:
            src_nodes = graph['edge_index'][0]
            risk_input = torch.cat([
                node_features[src_nodes],
                edge_features
            ], dim=1)  # [E, 96]
            risk_weights = self.risk_attention(risk_input)  # [E, 1]
        else:
            risk_weights = None
        
        # 3. GNN传播
        x = node_features
        for layer in self.gnn_layers:
            x = layer(x, graph['edge_index'], edge_attr=edge_features, attention_weights=risk_weights)
            x = F.relu(x)
        
        # 4. 输出投影
        global_embedding = self.output_layer(x)  # [N, 256]
        
        return global_embedding


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    支持边特征和风险感知的注意力机制
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 edge_dim: int = 32, concat: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.edge_dim = edge_dim
        
        # 线性变换
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # 注意力计算
        self.attention = nn.Linear(2 * out_channels + edge_dim, 1, bias=False)
        
        # 边特征变换
        self.edge_transform = nn.Linear(edge_dim, heads * out_channels, bias=False)
        
        # 输出变换
        if not concat:
            # 当concat=False时，多头结果会平均，所以输入维度是out_channels
            self.output_transform = nn.Linear(out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.xavier_uniform_(self.edge_transform.weight)
        if hasattr(self, 'output_transform'):
            nn.init.xavier_uniform_(self.output_transform.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [N, in_channels] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            attention_weights: [E, 1] 预计算的风险注意力权重
        Returns:
            out: [N, out_channels] 输出特征
        """
        N = x.size(0)
        
        # 1. 线性变换节点特征
        x = self.linear(x).view(N, self.heads, self.out_channels)  # [N, heads, out_channels]
        
        # 2. 边特征变换 - 保持原始边特征用于注意力计算
        if edge_attr is not None and edge_index.size(1) > 0:
            # 用于消息传递的边特征变换
            edge_attr_for_message = self.edge_transform(edge_attr).view(-1, self.heads, self.out_channels)
            # 用于注意力计算的原始边特征扩展
            edge_attr_for_attention = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)  # [E, heads, edge_dim]
        
        # 3. 计算注意力
        if edge_index.size(1) == 0:  # 没有边的情况
            # 直接返回变换后的节点特征
            out = x
            if self.concat:
                out = out.view(N, self.heads * self.out_channels)
            else:
                out = out.mean(dim=1)  # [N, out_channels]
                # 当concat=False时，直接使用out，不需要再展平
                out = self.output_transform(out)  # [N, out_channels]
            return out
        
        row, col = edge_index  # [E], [E]
        
        # 获取源节点和目标节点特征
        x_i = x[row]  # [E, heads, out_channels]
        x_j = x[col]  # [E, heads, out_channels]
        
        # 拼接特征用于注意力计算 - 使用原始边特征
        if edge_attr is not None:
            # 拼接: x_i (out_channels) + x_j (out_channels) + edge_attr (edge_dim)
            attention_input = torch.cat([x_i, x_j, edge_attr_for_attention], dim=-1)  # [E, heads, 2*out_channels + edge_dim]
        else:
            attention_input = torch.cat([x_i, x_j], dim=-1)  # [E, heads, 2*out_channels]
        
        # 重塑为二维张量以匹配attention层的期望输入
        E, heads, _ = attention_input.shape
        attention_input = attention_input.view(E * heads, -1)  # [E*heads, 2*out_channels + edge_dim] 或 [E*heads, 2*out_channels]
        
        # 计算注意力分数
        alpha = self.attention(attention_input).squeeze(-1)  # [E*heads]
        alpha = alpha.view(E, heads)  # 重新reshape为 [E, heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # 应用风险权重（如果提供）
        if attention_weights is not None:
            alpha = alpha * attention_weights.squeeze(-1).unsqueeze(-1)  # [E, heads]
        
        # Softmax归一化
        alpha = torch.softmax(alpha, dim=0)  # [E, heads]
        
        # 4. 聚合邻居信息
        # 使用scatter_add进行高效聚合
        out = torch.zeros_like(x)  # [N, heads, out_channels]
        
        # 为每个头分别聚合
        for head in range(self.heads):
            alpha_head = alpha[:, head].unsqueeze(-1)  # [E, 1]
            x_j_head = x_j[:, head, :]  # [E, out_channels]
            
            # 如果有边特征，将边特征加到消息中
            if edge_attr is not None:
                edge_attr_head = edge_attr_for_message[:, head, :]  # [E, out_channels]
                weighted_features = alpha_head * (x_j_head + edge_attr_head)  # [E, out_channels]
            else:
                weighted_features = alpha_head * x_j_head  # [E, out_channels]
            
            # 确保数据类型一致，然后聚合到目标节点
            target_dtype = out.dtype
            weighted_features = weighted_features.to(target_dtype)
            out[:, head, :].index_add_(0, col, weighted_features)
        
        # 5. 输出变换
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)  # [N, out_channels]
            # 当concat=False时，直接使用out不需要再展平
            out = self.output_transform(out)  # [N, out_channels]
        
        return out
