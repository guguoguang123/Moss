import torch

# 1. 解冻 .pt 文件
dataset = torch.load("data/DataSet/Graphs/moss_tensor_dataset.pt", weights_only=False)

print(f"数据集中共有 {len(dataset)} 个电路的张量图。")

if len(dataset) > 0:
    # 获取第一个电路的图数据
    graph = dataset[0]
    
    print("\n--- [详细内容预览] ---")
    print(f"电路结构对象: {graph}")
    print(f"特征矩阵 x 的形状: {graph.x.shape}")
    print(f"连线矩阵 edge_index 的形状: {graph.edge_index.shape}")
    
    print("\n--- [LLM 特征向量 (前 5 个节点) 预览] ---")
    # 打印前 5 个节点的 4096 维特征前 10 个数字
    for i in range(min(5, graph.x.shape[0])):
        vector_preview = graph.x[i, :10] # 只打印前 10 个维度，防止刷屏
        print(f"节点 {i} 的特征向量预览 (前10维): {vector_preview.tolist()} ...")
        
    print("\n--- [连线预览 (前 5 条连线)] ---")
    print(f"前 5 条连线关系: \n{graph.edge_index[:, :5]}")