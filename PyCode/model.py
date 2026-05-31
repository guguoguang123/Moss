"""
MOSS 模型定义文件
===================
本项目实现一个基于图神经网络 (GNN) 的电路 (Hardware Trojan) 检测模型 —— MOSS。
核心思想是将数字电路建模为有向图，其中：
  - 节点：组合逻辑门 (AND/OR/NOT/XOR/MUX) 和时序触发器 (DFF)
  - 边：导线连接关系
  - 节点特征：由 Yi-Coder 大语言模型生成的 4096 维语义嵌入向量

模型采用"两相异步传播"机制，物理对齐真实电路的时序行为：
  Phase 1 (Forward)：组合逻辑 -> 触发器，模拟时钟上升沿前的组合逻辑运算
  Phase 2 (Turnaround)：触发器 -> 组合逻辑，模拟时钟触发后状态的反馈传播

最终通过全局平均池化 + 全连接分类器输出 44 类电路判定结果。
"""

import torch
# torch.nn.functional: PyTorch 函数式 API，提供 relu、dropout 等无状态的激活/正则化函数
import torch.nn.functional as F
# GATv2Conv: 图注意力网络 v2 卷积层（修复了 GAT 的静态注意力问题，注意力分数计算更灵活）
# global_mean_pool: 全局平均池化，将图中所有节点的特征取平均，得到整个图的表示向量
from torch_geometric.nn import GATv2Conv, global_mean_pool


class MOSSClassifier(torch.nn.Module):
    """
    MOSS 图神经网络分类器
    ======================
    这是一个对硬件电路进行图级分类的 GNN 模型，输入一个电路的 PyG Data 对象，
    输出该电路属于各类电路的概率分布。

    架构组成:
      1. feature_proj: 线性投影层，将 LLM 生成的 4096 维特征降维到 128 维
      2. forward_conv: 前向图注意力卷积，模拟"组合逻辑→DFF"的信号传播
      3. turnaround_conv: 回环图注意力卷积，模拟"DFF→组合逻辑"的反馈传播
      4. update_gate: GRU 门控单元，只对 DFF 节点进行有记忆的状态更新
      5. classifier: 全连接分类器，将图的全局表示映射到各类别概率

    输入参数:
      - in_channels: 输入特征维度，默认 4096（与 Yi-Coder-9B 的 hidden_size 对齐）
      - hidden_channels: 隐藏层维度，默认 128
      - num_classes: 分类类别数，默认 44
      - num_iterations: 两相异步传播的迭代轮数，默认 10（论文推荐值）
    """

    def __init__(self, in_channels=4096, hidden_channels=128, num_classes=44, num_iterations=10):
        # 调用父类 torch.nn.Module 的构造函数，注册所有子模块和参数
        super(MOSSClassifier, self).__init__()
        # 保存异步传播的迭代次数，决定 Phase 1+Phase 2 循环执行多少轮
        self.num_iterations = num_iterations

        # =====================================================================
        # 1. LLM 特征降维层
        #    将 Yi-Coder 输出的 4096 维语义向量线性映射到 128 维隐空间
        #    weight 形状: [hidden_channels, in_channels] = [128, 4096]
        #    bias 形状: [hidden_channels] = [128]
        # =====================================================================
        self.feature_proj = torch.nn.Linear(in_channels, hidden_channels)

        # =====================================================================
        # 2. 两相独立的图注意力卷积核
        #    GATv2Conv: 对每个节点，计算其邻居的注意力权重并加权聚合
        #    heads=4: 使用 4 个注意力头并行计算，增强表达能力
        #    concat=False: 不拼接多头输出，而是取平均（避免维度膨胀）
        #    forward_conv: 负责 Phase 1 —— 前向传播（信号沿导线顺流）
        #    turnaround_conv: 负责 Phase 2 —— 回环传播（触发器反馈）
        # =====================================================================
        self.forward_conv = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.turnaround_conv = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False)

        # =====================================================================
        # 3. 异步状态记忆门 (GRU Cell)
        #    模拟触发器的锁存行为：只有当时钟沿到达时，DFF 才更新内部状态
        #    GRUCell 输入/输出维度均为 hidden_channels=128
        #    通过 is_dff 掩码精确控制哪些节点执行状态更新
        # =====================================================================
        self.update_gate = torch.nn.GRUCell(hidden_channels, hidden_channels)

        # =====================================================================
        # 4. 全局分类器
        #    Sequential 容器按顺序堆叠各层：
        #      Linear(128 -> 64) → ReLU → Dropout(0.3) → Linear(64 -> num_classes)
        #    Dropout 以 30% 概率随机丢弃神经元，缓解小样本过拟合
        # =====================================================================
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),  # 128 -> 64 降维
            torch.nn.ReLU(),                                         # 非线性激活
            torch.nn.Dropout(0.3),                                   # 正则化防过拟合
            torch.nn.Linear(hidden_channels // 2, num_classes)       # 64 -> 44 类别输出
        )

    def forward(self, x, edge_index, batch, is_dff):
        """
        前向传播 —— 两相异步传播 + 全局池化 + 分类

        参数:
          x:          节点特征矩阵，形状 [N, 4096]，N 为一个 batch 中所有图的节点总数
          edge_index: 边索引张量，形状 [2, E]，每列 (src, dst) 表示一条有向边
          batch:      批次索引向量，形状 [N]，batch[i] 表示节点 i 属于 batch 中的第几张图
          is_dff:     布尔张量，形状 [N]，True 表示该节点是 DFF 触发器（时序元件），
                      False 表示是组合逻辑门。这是物理仿真系统的核心开关！

        返回:
          out: 分类 logits，形状 [num_graphs_in_batch, num_classes]
        """
        # -----------------------------------------------------------------
        # 初始特征投影：4096维 → 128维，接 ReLU 激活
        # h 形状: [N, 128]
        # -----------------------------------------------------------------
        h = F.relu(self.feature_proj(x))

        # =================================================================
        # 绝对物理对齐的两相异步传播 (Two-Phase Asynchronous Propagation)
        # 每次迭代执行 Phase 1 + Phase 2，共迭代 num_iterations=10 轮
        # =================================================================
        for _ in range(self.num_iterations):

            # --- Phase 1: Forward (组合逻辑 -> 触发器) ---
            # 信息顺着导线流淌：前向卷积计算所有节点之间的信息交互
            # msg_forward 形状: [N, 128]
            msg_forward = F.relu(self.forward_conv(h, edge_index))

            # 【核心隔离机制】
            # 真实电路中，时钟沿到来前，只有 DFF 才会锁存前向传播的结果
            # is_dff 掩码确保只对 DFF 节点执行 GRU 状态更新，组合逻辑节点状态不变
            h_new = h.clone()                                                # 深拷贝当前状态，防止原地修改
            h_new[is_dff] = self.update_gate(msg_forward[is_dff], h[is_dff]) # 只更新 DFF 节点
            h = h_new                                                        # 替换为更新后的状态

            # --- Phase 2: Turnaround (触发器 -> 组合逻辑) ---
            # 触发器状态更新后，通过反馈环路将信息周转回前面的组合逻辑门
            # msg_turnaround 形状: [N, 128]
            msg_turnaround = F.relu(self.turnaround_conv(h, edge_index))

            # 【核心隔离机制】
            # 这次周转，只更新非 DFF 节点（组合逻辑），触发器保持稳定
            # ~is_dff 逻辑取反：True(DFF)变False跳过，False(组合逻辑)变True更新
            h_new = h.clone()                          # 再次深拷贝
            h_new[~is_dff] = msg_turnaround[~is_dff]   # 组合逻辑直接接受回环消息（无需 GRU）
            h = h_new                                  # 替换为更新后的状态

        # =================================================================
        # 全局图级池化：将每张图的所有节点特征取平均，得到固定维度的图表示
        # graph_emb 形状: [num_graphs_in_batch, 128]
        # =================================================================
        graph_emb = global_mean_pool(h, batch)

        # =================================================================
        # 分类输出：全连接层将图嵌入映射为各类别的 logits
        # out 形状: [num_graphs_in_batch, num_classes]
        # =================================================================
        out = self.classifier(graph_emb)
        return out