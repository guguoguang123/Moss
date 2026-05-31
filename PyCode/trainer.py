# PyCode/trainer.py
import torch
import os

class MOSSTrainer:
    def __init__(self, model, train_loader, optimizer, criterion, device, save_dir="checkpoints"):
        """
        MOSS 专属训练管理器
        """
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        
        # 记录最佳状态
        self.best_acc = 0.0
        
        # 确保保存权重的文件夹存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train_epoch(self):
        """运行单个 Epoch 的训练"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # 1. 梯度清零
            self.optimizer.zero_grad()
            
            # 2. 前向传播 (注意传入了我们独家的物理隔离掩码 is_dff)
            out = self.model(batch.x, batch.edge_index, batch.batch, batch.is_dff)
            
            # 3. 计算损失
            loss = self.criterion(out, batch.y)
            
            # 4. 反向传播与权重更新
            loss.backward()
            self.optimizer.step()
            
            # 5. 统计数据
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_samples += batch.num_graphs
            
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def fit(self, epochs=30):
        """控制完整的训练生命周期"""
        print(f"[*] 🚀 炼丹炉已点火，准备进行 {epochs} 个 Epoch 的训练...")
        
        for epoch in range(1, epochs + 1):
            # 执行一个 epoch
            loss, acc = self.train_epoch()
            
            print(f"Epoch [{epoch:03d}/{epochs}] | Loss: {loss:.4f} | 准确率: {acc*100:.2f}%")
            
            # 自动保存最佳模型 (Checkpointing)
            if acc > self.best_acc:
                self.best_acc = acc
                save_path = os.path.join(self.save_dir, "moss_best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> 🌟 发现新巅峰！最佳准确率刷新为 {acc*100:.2f}%，模型已保存至 {save_path}")
                
        print(f"\n[+] 🎉 训练圆满结束！全局最高准确率: {self.best_acc*100:.2f}%")