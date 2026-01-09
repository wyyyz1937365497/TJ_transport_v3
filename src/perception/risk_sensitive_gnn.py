import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class RiskSensitiveGNN(nn.Module):
    """
    风险敏感图神经网络
    用于交通场景中的感知和风险评估
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3, risk_sensitive=True):
        super(RiskSensitiveGNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.risk_sensitive = risk_sensitive
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # 第一层
        self.conv_layers.append(GATConv(node_dim, hidden_dim, edge_dim=edge_dim))
        self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        # 最后一层
        self.conv_layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
        self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        # 风险评估头
        if risk_sensitive:
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            batch: 批次索引 [num_nodes]
        Returns:
            输出嵌入 [output_dim] 或 [batch_size, output_dim]
        """
        # 如果没有提供批次信息，假设是单个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        h = x
        risk_scores = []
        
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            # 图卷积
            if edge_attr is not None:
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)
            
            # 归一化和激活
            h = norm(h)
            h = F.relu(h)
            
            # 如果启用了风险敏感机制，计算风险分数
            if self.risk_sensitive and i == len(self.conv_layers) - 1:
                # 计算每个节点的风险分数
                node_risks = self.risk_head(h)
                # 全局池化得到图级别的风险
                graph_risk = global_mean_pool(node_risks, batch)
                risk_scores.append(graph_risk)
        
        # 全局池化
        h_global = global_mean_pool(h, batch)
        
        # 输出投影
        output = self.output_proj(h_global)
        
        return output