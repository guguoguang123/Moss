import os
import pickle
import torch
from torch_geometric.data import Data

class MOSSPyGDatasetBuilder:
    def __init__(self, fused_pkl_path, output_pt_path):
        self.fused_pkl_path = fused_pkl_path
        self.output_pt_path = output_pt_path
        
    def load_fused_graphs(self):
        print(f"[*] 正在加载融合后的 NetworkX 图数据: {self.fused_pkl_path}")
        with open(self.fused_pkl_path, 'rb') as f:
            return pickle.load(f)

    def convert_to_tensor(self, G):
        """
        核心：将单个 NetworkX 图转为 PyG 的张量数据结构
        """
        # 1. 建立节点映射 (GPU不认识字符串名字，只认识 0, 1, 2... 这样的整数索引)
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        # 2. 提取特征张量 X (形状: [节点数, 4096]) 以及 DFF 掩码
        x_list = []
        is_dff_list = [] # 【新增】：用于记录每个节点是不是 DFF
        
        for node in G.nodes():
            # 2.1 抽出我们刚才辛辛苦苦注入的 4096 维特征
            emb = G.nodes[node]['llm_embedding'] 
            x_list.append(emb)
            
            # 2.2 【新增】：判断当前节点是否为 DFF 触发器
            node_type = G.nodes[node].get('node_type', '')
            is_dff_list.append(1 if node_type == 'DFF' else 0)
            
        x = torch.tensor(x_list, dtype=torch.float32)
        is_dff = torch.tensor(is_dff_list, dtype=torch.bool) # 【新增】：转为布尔张量 (True/False)
        
        # 3. 提取连线张量 Edge_Index (形状: [2, 边数])
        # 将 ('wire_A', 'DFF_B') 转换成数字索引，比如 [ [0], [1] ]
        source_nodes = []
        target_nodes = []
        for u, v in G.edges():
            source_nodes.append(node_mapping[u])
            target_nodes.append(node_mapping[v])
            
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # 4. 打包成 PyTorch Geometric 标准的数据对象
        # 【修改】：将 is_dff 也打包进 Data 对象中，让 GPU 知道谁是触发器
        data = Data(x=x, edge_index=edge_index, is_dff=is_dff) 
        return data

    def build_and_save(self):
        dataset = self.load_fused_graphs()
        pyg_dataset = []
        
        print("[*] 开始将图对象转换为 PyTorch 张量...")
        for module_name, G in dataset.items():
            pyg_data = self.convert_to_tensor(G)
            pyg_dataset.append(pyg_data)
            # 【修改】：在打印日志中加入 is_dff 的形状展示，确认注入成功
            print(f"  -> [转换成功] {module_name:<15} | X 形状: {list(pyg_data.x.shape)} | Edge 形状: {list(pyg_data.edge_index.shape)} | DFF 掩码: {list(pyg_data.is_dff.shape)}")

        # 将张量列表保存为 PyTorch 专用的 .pt 文件
        torch.save(pyg_dataset, self.output_pt_path)
        print("-" * 50)
        print(f"[+] 数据已全部张量化，且物理状态掩码 (is_dff) 注入完毕！")
        print(f"[+] 张量数据集保存在: {self.output_pt_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_PKL = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_fused_dataset.pkl")
    OUTPUT_PT = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_tensor_dataset.pt")
    
    builder = MOSSPyGDatasetBuilder(INPUT_PKL, OUTPUT_PT)
    builder.build_and_save()