# PyCode/model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class MOSSClassifier(torch.nn.Module):
    def __init__(self, in_channels=4096, hidden_channels=128, num_classes=44, num_iterations=10):
        super(MOSSClassifier, self).__init__()
        self.num_iterations = num_iterations
        
        # 1. LLM 4096维语义特征降维
        self.feature_proj = torch.nn.Linear(in_channels, hidden_channels)
        
        # 2. 两个独立的图注意力卷积核，分别负责前向和回环
        self.forward_conv = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.turnaround_conv = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False)
        
        # 3. 异步状态记忆门 (GRU)
        self.update_gate = torch.nn.GRUCell(hidden_channels, hidden_channels)
        
        # 4. 全局分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, edge_index, batch, is_dff):
        """
        注意输入多了一个 `is_dff`！
        它是一个 [N] 维的布尔张量 (True代表DFF，False代表普通逻辑门)。
        这是整个物理仿真系统的核心开关！
        """
        # 初始特征投影
        h = F.relu(self.feature_proj(x))
        
        # =================================================================
        # 绝对物理对齐的两相异步传播 (Two-Phase Asynchronous Propagation)
        # =================================================================
        for _ in range(self.num_iterations):
            
            # --- Phase 1: Forward (组合逻辑 -> 触发器) ---
            # 信息顺着导线流淌，计算所有的信息交互
            msg_forward = F.relu(self.forward_conv(h, edge_index))
            
            # 【核心隔离机制】！
            # 真实电路中，时钟沿没到来前，只有 DFF 才会锁存前向传播的结果！
            # 我们用 is_dff 掩码，强行只更新触发器的状态
            h_new = h.clone()
            h_new[is_dff] = self.update_gate(msg_forward[is_dff], h[is_dff])
            h = h_new
            
            # --- Phase 2: Turnaround (触发器 -> 组合逻辑) ---
            # 触发器的状态更新后，通过反馈环路将信息周转回前面的逻辑门
            msg_turnaround = F.relu(self.turnaround_conv(h, edge_index))
            
            # 【核心隔离机制】！
            # 这次周转，只更新组合逻辑的状态，触发器保持稳定！
            h_new = h.clone()
            h_new[~is_dff] = msg_turnaround[~is_dff]  # ~is_dff 表示非 DFF 节点
            h = h_new
            
        # =================================================================
        
        # 提取整个图的宏观特征
        graph_emb = global_mean_pool(h, batch)
        
        # 输出分类概率
        out = self.classifier(graph_emb)
        return out