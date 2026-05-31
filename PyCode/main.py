# PyCode/main.py
import torch
import pickle
import json
from torch_geometric.loader import DataLoader
import os

# 导入我们自己写的模块
from model import MOSSClassifier
from trainer import MOSSTrainer

def load_data_and_labels():
    print("[*] 正在加载张量特征与物理掩码...")
    pyg_dataset = torch.load("../data/DataSet/Graphs/moss_tensor_dataset.pt", weights_only=False)

    print("[*] 正在加载完美标签 JSON...")
    with open("../data/perfect_labels.json", "r") as f:
        label_data = json.load(f)
    labels_dict = label_data["labels"]

    print("[*] 正在对齐特征与标签...")
    with open("../data/DataSet/Graphs/moss_fused_dataset.pkl", "rb") as f:
        fused_graphs = pickle.load(f)
    module_names = list(fused_graphs.keys())

    valid_dataset = []
    max_label_id = -1 # 用于动态记录最大的合法 ID
    
    for i, data in enumerate(pyg_dataset):
        name = module_names[i]
        clean_name = name.replace("_netlist", "")
        
        if clean_name in labels_dict:
            label_id = labels_dict[clean_name]
            
            # 🛡️ 核心防御：直接踢掉所有的 -1 或非法标签！
            if label_id < 0:
                print(f"  [跳过] 发现坏标签 {clean_name}: {label_id}")
                continue
                
            # 打包合法的目标 Y
            data.y = torch.tensor([label_id], dtype=torch.long)
            valid_dataset.append(data)
            
            # 记录当前遇到的最大分类 ID
            if label_id > max_label_id:
                max_label_id = label_id
                
    # 动态推断出最安全的分类总数 (最大ID + 1)
    num_classes = max_label_id + 1
    print(f"[+] 数据就绪：共提取 {len(valid_dataset)} 个绝对安全的电路样本！(安全锁定为 {num_classes} 分类任务)")
    return valid_dataset, num_classes

def main():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 当前炼丹硬件: {device}")

    # 2. 准备数据加载器
    dataset, num_classes = load_data_and_labels()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True) # 因为电路图较大，batch_size设为4比较安全

    # 3. 实例化 MOSS 物理级对齐模型
    model = MOSSClassifier(
        in_channels=4096, 
        hidden_channels=128, 
        num_classes=num_classes,
        num_iterations=10 # 论文里的 10 次时序环路迭代
    ).to(device)
    
    # 4. 配置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. 移交大权给 Trainer
    trainer = MOSSTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir="../checkpoints" # 模型权重保存位置
    )
    
    # 6. 一键点火
    trainer.fit(epochs=40)

if __name__ == "__main__":
    # 确保相对路径正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()