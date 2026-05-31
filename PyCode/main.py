"""
MOSS 主入口文件
===============
本项目是硬件木马 (Hardware Trojan) 检测的主训练脚本，负责串联整个 MOSS 管线：

  数据管线流程:
    1. 加载预处理的 PyG 张量数据集 (.pt 文件)
    2. 加载人工标注的完美标签 (perfect_labels.json)
    3. 加载融合数据集 (.pkl) 以对齐模块名与标签
    4. 过滤掉非法标签 (-1，即无法确定类别的样本)
    5. 动态推断实际类别数（避免类别数写死导致的 bug）

  训练管线流程:
    1. 将 MOSSClassifier 模型部署到 GPU
    2. 配置 Adam 优化器和交叉熵损失函数
    3. 创建 MOSSTrainer 训练管理器
    4. 一键启动训练

运行方式:
    python PyCode/main.py
    确保在 PyCode 目录下运行（脚本会自动 chdir 到自身所在目录）
"""

import torch
import pickle
import json
# DataLoader: PyG 的数据加载器，自动做 mini-batch 图拼接（将多张小图拼成一张大图）
from torch_geometric.loader import DataLoader
import os

# 导入我们自己写的模块
from model import MOSSClassifier   # 图神经网络分类模型
from trainer import MOSSTrainer    # 训练管理器（封装训练循环、检查点保存）


def load_data_and_labels():
    """
    数据加载与标签对齐函数
    ========================
    功能: 依次加载三个数据源，将它们对齐，过滤坏样本，返回可用数据集和类别数。

    数据源说明:
      - moss_tensor_dataset.pt:  PyG Data 对象列表（节点特征 + 边 + DFF 掩码已完备）
      - perfect_labels.json:    人工标注的真值标签，格式 {"labels": {"module_name": label_id}}
      - moss_fused_dataset.pkl: 融合后的 NetworkX 图字典，用于获取模块名顺序

    返回:
      valid_dataset: List[Data] — 过滤后的合法数据样本列表（每个样本带 .y 标签）
      num_classes:   int       — 动态推断的类别总数 (max_label_id + 1)
    """
    # =========================================================================
    # Step 1: 加载 PyG 张量数据集
    #   weights_only=False: 允许加载非纯张量对象（PyG Data 对象含 dict 属性）
    # =========================================================================
    print("[*] 正在加载张量特征与物理掩码...")
    pyg_dataset = torch.load("../data/DataSet/Graphs/moss_tensor_dataset.pt", weights_only=False)

    # =========================================================================
    # Step 2: 加载完美标签 JSON
    #   perfect_labels.json 结构: {"labels": {"module_name": label_id, ...}}
    #   label_id 范围: 0~43 (共 44 类硬件木马)，-1 表示无法归类
    # =========================================================================
    print("[*] 正在加载完美标签 JSON...")
    with open("../data/perfect_labels.json", "r") as f:
        label_data = json.load(f)                # 解析 JSON 为 Python 字典
    labels_dict = label_data["labels"]            # 提取 labels 子字典: {模块名: 标签ID}

    # =========================================================================
    # Step 3: 对齐特征与标签
    #   pyg_dataset 和 fused_graphs 的模块顺序必须一致（由构建管线保证）
    #   通过 fused_graphs 的键获取模块名列表，用于去 netlist 后缀匹配 label
    # =========================================================================
    print("[*] 正在对齐特征与标签...")
    with open("../data/DataSet/Graphs/moss_fused_dataset.pkl", "rb") as f:
        fused_graphs = pickle.load(f)            # 反序列化 NetworkX 图字典
    module_names = list(fused_graphs.keys())      # 获取模块名列表（与 pyg_dataset 顺序一致）

    valid_dataset = []          # 存放过滤后的合法样本
    max_label_id = -1           # 记录遇到的最大合法标签 ID，用于动态推断 num_classes

    # 遍历 PyG 数据集中的每个电路样本（按编号 i 索引）
    for i, data in enumerate(pyg_dataset):
        name = module_names[i]                              # 获取第 i 个模块的原始名称
        clean_name = name.replace("_netlist", "")           # 去除 _netlist 后缀，与 labels_dict 键对齐

        if clean_name in labels_dict:                       # 该模块存在于标签字典中
            label_id = labels_dict[clean_name]              # 获取对应的类别标签 ID

            # 防御性过滤：跳过所有非法标签（-1 表示无法归类的坏样本）
            if label_id < 0:
                print(f"  [跳过] 发现坏标签 {clean_name}: {label_id}")
                continue

            # 将标签包装为 PyTorch 张量并赋值给 data.y（PyG 约定的标签字段）
            data.y = torch.tensor([label_id], dtype=torch.long)
            valid_dataset.append(data)                      # 加入合法数据集

            # 更新见过的最大标签 ID（用于后续动态计算类别总数）
            if label_id > max_label_id:
                max_label_id = label_id

    # 动态推断分类总数：最大合法标签 ID + 1（假设标签是 0-indexed 的连续整数）
    num_classes = max_label_id + 1
    print(f"[+] 数据就绪：共提取 {len(valid_dataset)} 个绝对安全的电路样本！(安全锁定为 {num_classes} 分类任务)")
    return valid_dataset, num_classes


def main():
    """
    主训练流程
    ==========
    依次执行: 设备检测 → 数据加载 → 模型构建 → 优化器配置 → 训练启动
    """
    # 1. 设备选择：优先使用 CUDA GPU，不可用时回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 当前炼丹硬件: {device}")

    # 2. 准备数据加载器
    #    batch_size=4: 每个 mini-batch 包含 4 张电路图（电路图节点较多，不宜设太大）
    #    shuffle=True:  每轮训练随机打乱样本顺序，防止模型记忆数据顺序
    dataset, num_classes = load_data_and_labels()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. 实例化 MOSS 图神经网络模型
    #    in_channels=4096:    与 Yi-Coder-9B 的 hidden_size 对齐
    #    hidden_channels=128: 隐层维度（经验值，平衡表达能力与计算开销）
    #    num_classes:         动态推断的实际类别数（而非写死的 44）
    #    num_iterations=10:   两相异步传播迭代轮数（论文推荐值）
    model = MOSSClassifier(
        in_channels=4096,
        hidden_channels=128,
        num_classes=num_classes,
        num_iterations=10
    ).to(device)

    # 4. 配置优化器和损失函数
    #    Adam 优化器: lr=0.005 为初始学习率（相对较大的起点，配合 GRU 加速收敛）
    #    CrossEntropyLoss: 交叉熵损失，内部集成 LogSoftmax + NLLLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. 创建训练管理器，负责训练循环、日志打印和最佳模型保存
    trainer = MOSSTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir="../checkpoints"  # 模型权重保存目录（相对路径，位于项目根目录下）
    )

    # 6. 启动训练：运行 40 个 epoch
    trainer.fit(epochs=40)


# 脚本入口：当直接运行此文件时（而非被 import），执行 main()
if __name__ == "__main__":
    # 将工作目录切换到脚本所在目录，确保所有相对路径正确解析
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()