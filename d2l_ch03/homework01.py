# 练习题 1：脱离 d2l 库的手写 Softmax 回归
# 约束条件：
# 禁止 import d2l。
# 只允许使用 torch, torchvision (仅用于下载数据)。
# 不使用 torch.utils.data.DataLoader 以外的高级封装。
# 核心考点（书里帮你做了，现在你要自己做）：
# 数据预处理：自己写 transform 把图片转为 Tensor。
# 训练可视化：书里有动态画图，你不需要做那么复杂，但要求每隔 1 个 Epoch，在控制台打印出当前 Train Loss, Train Acc, Test Acc。
# 评估函数：你需要自己实现一个 evaluate_accuracy(net, data_iter)，能在 GPU/CPU 之间正确切换数据（如果用 GPU 的话）。
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


class SoftmaxTrainer:
    def __init__(self, batch_size=256, lr=0.1):
        # 1. 硬件探测 (Service Discovery)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        self.batch_size = batch_size
        self.lr = lr

        # 2. 构建模型 (Dependency Injection)
        # 建议：使用 nn.Flatten 和 nn.Linear
        self.net = self._build_model().to(self.device)

        # 3. 初始化权重 (Initialization)
        # 建议：使用 net.apply(...) 初始化为均值0，标准差0.01的正态分布
        self.net.apply(self._init_weights)

        # 4. 定义组件
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)

    def _build_model(self):
        return nn.Sequential(nn.Flatten(), nn.Linear(in_features=784, out_features=10))

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.01)

    def get_dataloader(self, train=True):
        # 数据转换流 (Middleware)
        trans = transforms.Compose([transforms.ToTensor()])
        # 下载数据 (Repository Layer)
        data = datasets.FashionMNIST(root="./data", train=train, transform=trans, download=True)
        # 封装为 Loader
        return DataLoader(data, batch_size=self.batch_size, shuffle=train, num_workers=2)

    def accuracy(self, y_hat, y):
        # 辅助函数：计算预测正确的数量
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def evaluate(self, data_iter):
        # 验证模式 (Read-Only Transaction)
        self.net.eval()  # 关闭 Dropout 等
        acc_sum, n = 0.0, 0
        with torch.no_grad():  # 关闭梯度记录 (节省显存)
            for X, y in data_iter:
                X, y = X.to(self.device), y.to(self.device)
                acc_sum += self.accuracy(self.net(X), y)
                n += y.numel()
        self.net.train()  # 恢复训练模式
        return acc_sum / n

    def fit(self, epochs=5):
        # 主循环 (Main Loop)
        train_iter = self.get_dataloader(train=True)
        test_iter = self.get_dataloader(train=False)

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0.0
            total_correct = 0.0
            total_samples = 0
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)

                # --- 核心训练步 (The Atomic Transaction) ---
                self.optimizer.zero_grad()
                y_hat = self.net(X)
                loss = self.loss_fn(y_hat, y)

                loss.backward()
                self.optimizer.step()

                batch_size_actual = y.numel()
                total_loss += loss.item()*batch_size_actual
                total_correct += self.accuracy(y_hat, y)
                total_samples += batch_size_actual

                # 提示：记得 metric 的累加

                # --- End Transaction ---

            # 计算 Epoch 级别的指标
            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            test_acc = self.evaluate(test_iter)

            print(
                f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | Time: {time.time() - start_time:.1f}s")


# --- Client Code ---
if __name__ == "__main__":
    trainer = SoftmaxTrainer(batch_size=256, lr=0.1)
    trainer.fit(epochs=5)