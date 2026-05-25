import pickle
import numpy as np

# 1. 加载融合后的 pkl 文件
pkl_path = "data/DataSet/Graphs/moss_fused_dataset.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"[*] pkl 文件加载成功，包含 {len(data)} 个模块。")

# 2. 随意选一个模块查看（以 'timer' 为例）
module_name = 'timer' 
if module_name in data:
    G = data[module_name]
    print(f"\n--- 正在查看模块: {module_name} ---")
    
    # 3. 遍历节点，寻找注入了特征的 DFF 节点
    dff_count = 0
    for node, attr in G.nodes(data=True):
        if attr.get('node_type') == 'DFF':
            dff_count += 1
            embedding = attr.get('llm_embedding')
            
            print(f"\n[DFF 节点]: {node}")
            # 检查特征是否注入（不是 None 且长度为 4096）
            if embedding is not None:
                print(f"  -> 特征已注入，维度: {len(embedding)}")
                print(f"  -> 前 5 位数值: {embedding[:5]}") # 打印前5位数值
            else:
                print("  -> 警告：特征未注入！")
            
            # 我们只看一个就够了，防止刷屏
            if dff_count >= 1:
                break
else:
    print(f"找不到模块 {module_name}，请检查数据集中有哪些模块。")
    print(f"当前数据集中的模块有: {list(data.keys())[:5]} ...")