# 背景：
# 为了排查一个底层的数值溢出 Bug，你需要绕过 PyTorch 的 nn.CrossEntropyLoss 和 nn.Linear，完全使用 Tensor 操作来实现核心计算逻辑。

# 需求：
# 在不使用神经网络层（nn.Linear）和损失函数层（nn.CrossEntropyLoss）的前提下，完成一次前向传播和 Loss 计算。

# 验收标准：
# softmax 函数必须包含防止指数爆炸的数值稳定处理（即减去 max）。
# W.grad 不为 None 且形状正确。
# 计算结果需与 nn.CrossEntropyLoss 的结果误差在 1e-5 以内（可以用 PyTorch 标准版验证一下你的结果）。

import torch

# 1. 模拟数据 (Batch Size=2, Input Features=4, Classes=3)
X = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [0.5, 0.5, 0.5, 0.5]])
y = torch.tensor([0, 2])  # 真实标签

# 2. 初始化权重 (需要梯度)
W = torch.normal(mean=0, std=0.01, size=(4, 3), requires_grad=True)
b = torch.zeros(3, requires_grad=True)

def softmax(O):
    """
    手动实现 Softmax
    提示：为了数值稳定性，建议先对 O 做减去最大值的操作 (O - O.max(axis=1, keepdims=True))
    """
    o_max, _ = torch.max(O, dim=1, keepdim=True)
    o_without_max = O-o_max
    O_exp = torch.exp(o_without_max)
    partition = O_exp.sum(axis=1, keepdim=True)

    return O_exp/partition

def cross_entropy_loss(y_hat, y):
    """
    手动实现交叉熵
    提示：利用高级索引 (Advanced Indexing) 取出正确类别的概率，再求 -log
    """
    return -torch.log(y_hat[range(len(y_hat)), y])

def model(X):
    tmp = torch.matmul(X.reshape((-1, W.shape[0])), W)+b
    return softmax(tmp)

# 执行
y_hat = model(X)
loss = cross_entropy_loss(y_hat, y).mean()
loss.backward()

print("Loss:", loss.item())
print("W.grad:", W.grad)
