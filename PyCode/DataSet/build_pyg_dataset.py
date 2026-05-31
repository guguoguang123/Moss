"""
MOSS PyG 张量数据集构建器
==========================
功能: 将融合后的 NetworkX 图数据集 (.pkl) 转换为 PyTorch Geometric (PyG) 张量格式 (.pt)

转换流程:
  1. 从 .pkl 文件加载 NetworkX 图字典（每个模块对应一张有向图）
  2. 对每张图:
     a. 建立节点名 → 整数索引的映射（GPU 只能处理数字索引）
     b. 提取节点特征矩阵 X（形状 [N, 4096]）和 DFF 掩码 is_dff
     c. 提取边索引矩阵 edge_index（形状 [2, E]）
     d. 打包为 PyG Data 对象（含 x, edge_index, is_dff 三个核心字段）
  3. 将所有 Data 对象保存为 torch.save 格式的 .pt 文件

输出: data/DataSet/Graphs/moss_tensor_dataset.pt
"""

import os
import pickle
import torch
# Data: PyG 的核心数据类，用于表示一张图（节点特征 + 边 + 其他属性）
from torch_geometric.data import Data


class MOSSPyGDatasetBuilder:
    """
    PyG 张量数据集构建器

    职责: 将融合后的 NetworkX 图转为 GPU 可消费的 PyG 张量格式

    属性:
      fused_pkl_path: 输入 .pkl 文件路径（融合后的图字典）
      output_pt_path: 输出 .pt 文件路径（PyG 张量数据集）
    """

    def __init__(self, fused_pkl_path, output_pt_path):
        """
        初始化构建器

        参数:
          fused_pkl_path: str — 融合后的 NetworkX 图数据集 .pkl 路径
          output_pt_path: str — 输出 PyG 张量数据集 .pt 路径
        """
        self.fused_pkl_path = fused_pkl_path    # 输入路径
        self.output_pt_path = output_pt_path    # 输出路径

    def load_fused_graphs(self):
        """
        加载融合后的 NetworkX 图数据

        返回:
          dict[str, nx.DiGraph] — 键为模块名，值为对应的有向图对象
        """
        print(f"[*] 正在加载融合后的 NetworkX 图数据: {self.fused_pkl_path}")
        with open(self.fused_pkl_path, 'rb') as f:
            return pickle.load(f)  # 反序列化图字典

    def convert_to_tensor(self, G):
        """
        核心转换函数：将单个 NetworkX 有向图转为 PyG 的 Data 张量对象

        参数:
          G: nx.DiGraph — 单个电路模块的有向图（节点带 llm_embedding 和 node_type 属性）

        返回:
          data: torch_geometric.data.Data — 包含 x, edge_index, is_dff 的 PyG 数据对象
        """
        # -----------------------------------------------------------------
        # Step 1: 建立节点名 → 整数索引的映射
        #   GPU 不认识字符串名字（如 "counter_reg", "_AND_gate_5"），
        #   只认识 0, 1, 2, ... 这样的整数索引
        #   node_mapping: {"counter_reg": 0, "gate_1": 1, ...}
        # -----------------------------------------------------------------
        node_mapping = {node: i for i, node in enumerate(G.nodes())}

        # -----------------------------------------------------------------
        # Step 2: 提取节点特征矩阵 X 和 DFF 掩码 is_dff
        #   x_list:    存储每个节点的 4096 维 LLM 语义嵌入向量
        #   is_dff_list: 存储每个节点是否为 DFF 触发器（True=时序元件, False=组合逻辑）
        # -----------------------------------------------------------------
        x_list = []           # 节点特征列表，最终堆叠为形状 [N, 4096] 的张量
        is_dff_list = []      # DFF 掩码列表，最终堆叠为形状 [N] 的布尔张量

        # 遍历图中的每个节点
        for node in G.nodes():
            # 2.1 提取该节点的 LLM 语义嵌入向量（4096 维 float 列表）
            emb = G.nodes[node]['llm_embedding']
            x_list.append(emb)

            # 2.2 判断该节点的类型是否为 DFF 触发器
            #     node_type 取值: 'DFF' | 'AND_GATE' | 'OR_GATE' | 'NOT_GATE' | 'XOR_GATE' | 'MUX_GATE'
            node_type = G.nodes[node].get('node_type', '')  # 获取 node_type，默认空字符串
            is_dff_list.append(1 if node_type == 'DFF' else 0)  # DFF→1，其他→0

        # 将列表转为 PyTorch 张量
        x = torch.tensor(x_list, dtype=torch.float32)          # 特征: [N, 4096], float32
        is_dff = torch.tensor(is_dff_list, dtype=torch.bool)   # 掩码: [N], bool (True/False)

        # -----------------------------------------------------------------
        # Step 3: 提取边索引张量 edge_index
        #   格式: [[src1, src2, ...], [dst1, dst2, ...]]
        #   例如边 ('wire_A', 'DFF_B') 映射后变为 [0, 1]（一列 [0; 1]）
        #   最终形状: [2, E]，E 为边的总数
        # -----------------------------------------------------------------
        source_nodes = []   # 源节点索引列表（边的起点）
        target_nodes = []   # 目标节点索引列表（边的终点）
        for u, v in G.edges():                    # 遍历每条有向边 u → v
            source_nodes.append(node_mapping[u])  # 将源节点名 u 转为整数索引
            target_nodes.append(node_mapping[v])  # 将目标节点名 v 转为整数索引

        # 拼接为 [2, E] 的边索引张量，dtype=long 因为索引是整数
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # -----------------------------------------------------------------
        # Step 4: 打包为 PyG Data 对象
        #   Data 是 PyG 的核心数据结构，可被 GNN 模型和 DataLoader 直接消费
        # -----------------------------------------------------------------
        data = Data(x=x, edge_index=edge_index, is_dff=is_dff)
        return data

    def build_and_save(self):
        """
        完整构建流程: 加载 → 逐图转换 → 保存
        """
        # 加载融合后的 NetworkX 图字典
        dataset = self.load_fused_graphs()
        pyg_dataset = []  # 存放所有电路对应的 PyG Data 对象

        print("[*] 开始将图对象转换为 PyTorch 张量...")
        # 遍历每个模块（module_name 如 'timer', 'GCD', 'FIFO_FLUSH_1'...）
        for module_name, G in dataset.items():
            # 调用核心转换函数
            pyg_data = self.convert_to_tensor(G)
            pyg_dataset.append(pyg_data)  # 加入数据集列表

            # 打印转换日志：模块名、特征形状、边形状、DFF 掩码形状
            print(f"  -> [转换成功] {module_name:<15} | "
                  f"X 形状: {list(pyg_data.x.shape)} | "
                  f"Edge 形状: {list(pyg_data.edge_index.shape)} | "
                  f"DFF 掩码: {list(pyg_data.is_dff.shape)}")

        # 将 PyG Data 列表保存为 PyTorch 标准 .pt 文件
        # 使用 torch.save 因为 Data 对象内部全是张量，可被序列化
        torch.save(pyg_dataset, self.output_pt_path)
        print("-" * 50)
        print(f"[+] 数据已全部张量化，且物理状态掩码 (is_dff) 注入完毕！")
        print(f"[+] 张量数据集保存在: {self.output_pt_path}")


# =============================================================================
# 脚本入口：当直接运行此文件时执行完整构建流程
# =============================================================================
if __name__ == "__main__":
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 输入：融合后的 NetworkX 图数据集（由 rtl_to_llm_feature.py 生成）
    INPUT_PKL = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_fused_dataset.pkl")
    # 输出：PyG 张量数据集（供 main.py 训练使用）
    OUTPUT_PT = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_tensor_dataset.pt")

    # 创建构建器实例并执行转换
    builder = MOSSPyGDatasetBuilder(INPUT_PKL, OUTPUT_PT)
    builder.build_and_save()