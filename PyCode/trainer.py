"""
MOSS 训练管理器
===============
封装了完整的训练生命周期管理，包括:
  - 单 epoch 训练循环（前向传播 + 反向传播 + 指标统计）
  - 多 epoch 自动迭代
  - 训练日志打印（Loss + Accuracy）
  - 模型检查点（Checkpointing）：自动保存准确率最高的模型权重
"""

import torch
import os


class MOSSTrainer:
    """
    MOSS 模型训练管理器

    职责:
      1. 管理训练循环 (train_epoch) —— 遍历 DataLoader，执行前向/反向传播
      2. 管理训练生命周期 (fit) —— 控制 epoch 数量，保存最佳模型
      3. 自动创建权重保存目录

    属性:
      model:        MOSSClassifier 实例
      train_loader: PyG DataLoader，提供 mini-batch 图数据
      optimizer:    PyTorch 优化器（Adam）
      criterion:    损失函数（CrossEntropyLoss）
      device:       计算设备（cuda 或 cpu）
      save_dir:     模型权重保存目录
      best_acc:     当前记录的最高准确率（用于判断是否保存新最佳模型）
    """

    def __init__(self, model, train_loader, optimizer, criterion, device, save_dir="checkpoints"):
        """
        初始化训练管理器

        参数:
          model:        MOSSClassifier 图神经网络模型
          train_loader: PyG DataLoader，提供 mini-batch 图数据
          optimizer:    PyTorch 优化器实例
          criterion:    损失函数实例
          device:       torch.device('cuda') 或 torch.device('cpu')
          save_dir:     模型权重 (.pth) 保存目录，默认 "checkpoints"
        """
        self.model = model                # 待训练的 GNN 模型
        self.train_loader = train_loader  # mini-batch 数据迭代器
        self.optimizer = optimizer        # 优化器（含学习率等超参）
        self.criterion = criterion        # 损失函数
        self.device = device              # cuda 或 cpu
        self.save_dir = save_dir          # 权重保存路径

        # 记录历史最佳准确率，初始化为 0.0
        self.best_acc = 0.0

        # 如果保存目录不存在，递归创建（exist_ok 等效于 mkdir -p）
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train_epoch(self):
        """
        执行单个 Epoch 的完整训练

        流程:
          1. 将模型设为训练模式（启用 Dropout 等）
          2. 遍历 DataLoader 中的每个 mini-batch
          3. 对每个 batch: 梯度清零 → 前向传播 → 计算损失 → 反向传播 → 参数更新
          4. 累积统计损失和正确预测数
          5. 返回该 epoch 的平均损失和准确率

        返回:
          avg_loss: float — 该 epoch 的平均损失值
          accuracy: float — 该 epoch 的准确率 (0.0 ~ 1.0)
        """
        self.model.train()            # 切换为训练模式（影响 Dropout、BatchNorm 等层的行为）
        total_loss = 0.0              # 累积损失（用于计算平均值）
        correct = 0                   # 累积正确预测数
        total_samples = 0             # 累积总样本数（即图的数量）

        # 遍历每个 mini-batch（batch 是 PyG 将多张小图拼成的大图）
        for batch in self.train_loader:
            batch = batch.to(self.device)  # 将整个 batch 的数据转移到 GPU/CPU

            # Step 1: 将优化器中所有参数的梯度清零
            #         必须放在每次前向传播之前，否则梯度会累积
            self.optimizer.zero_grad()

            # Step 2: 前向传播
            #         输入: batch.x(节点特征), batch.edge_index(边), batch.batch(批次索引), batch.is_dff(DFF掩码)
            #         输出: out（形状 [num_graphs, num_classes] 的 logits）
            out = self.model(batch.x, batch.edge_index, batch.batch, batch.is_dff)

            # Step 3: 计算损失（交叉熵 = LogSoftmax + NLLLoss）
            loss = self.criterion(out, batch.y)

            # Step 4: 反向传播 — 计算所有参数关于 loss 的梯度
            loss.backward()
            # Step 5: 参数更新 — 优化器根据梯度更新模型权重
            self.optimizer.step()

            # Step 6: 累积统计量
            #   loss.item() 获取标量损失值；乘以 batch.num_graphs 加权（不同 batch 图数可能不同）
            total_loss += loss.item() * batch.num_graphs
            #   out.argmax(dim=1) 获取每张图预测概率最大的类别索引
            pred = out.argmax(dim=1)
            #   累加预测正确的图数量
            correct += int((pred == batch.y).sum())
            #   累加本 batch 中的总图数量
            total_samples += batch.num_graphs

        # 计算该 epoch 的平均损失和准确率
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def fit(self, epochs=30):
        """
        控制完整的训练生命周期

        流程:
          1. 打印训练开始信息
          2. 循环 epochs 次，每次调用 train_epoch()
          3. 每个 epoch 结束后打印 Loss 和 Accuracy
          4. 如果当前准确率超过历史最佳，自动保存模型权重

        参数:
          epochs: int — 训练的总 epoch 数，默认 30
        """
        print(f"[*] 🚀 炼丹炉已点火，准备进行 {epochs} 个 Epoch 的训练...")

        # 遍历每个 epoch
        for epoch in range(1, epochs + 1):
            # 执行一个 epoch 的训练并获取指标
            loss, acc = self.train_epoch()

            # 打印当前 epoch 的训练指标（epoch 编号 3 位补零对齐）
            print(f"Epoch [{epoch:03d}/{epochs}] | Loss: {loss:.4f} | 准确率: {acc*100:.2f}%")

            # 检查点保存 (Checkpointing)
            # 仅当当前准确率严格超过历史最佳时才保存（确保保存的是真正最优权重）
            if acc > self.best_acc:
                self.best_acc = acc                                                    # 更新历史最佳记录
                save_path = os.path.join(self.save_dir, "moss_best_model.pth")          # 拼接保存路径
                torch.save(self.model.state_dict(), save_path)                          # 序列化模型参数
                print(f"  -> 🌟 发现新巅峰！最佳准确率刷新为 {acc*100:.2f}%，模型已保存至 {save_path}")

        # 训练全部完成，输出最终结果
        print(f"\n[+] 🎉 训练圆满结束！全局最高准确率: {self.best_acc*100:.2f}%")