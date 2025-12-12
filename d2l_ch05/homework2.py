# 背景：
# 在某些编码器-解码器架构中，我们希望 Input Embedding 层和 Output Projection 层共享同一份权重矩阵（Tying Weights），以减少显存占用并强制特征对齐。

# 需求：
# 构建一个网络，其中包含两个完全独立的 nn.Linear 层对象：layer_a 和 layer_b。
# 强制绑定：使 layer_b 的权重 物理上指向 layer_a 的权重内存地址（修改 a 必须自动影响 b，反之亦然）。

# 验证：
# 通过代码证明它们是同一个对象。
# 对 layer_a 进行一次梯度更新（或者手动修改值），打印 layer_b 的权重，证明它也变了。

import torch
from torch import nn
import torch.nn.functional as F

shared_layer = nn.Linear(3, 3)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_a = shared_layer
        self.layer_b = shared_layer

    def forward(self, X):
        return self.layer_b(F.relu(self.layer_a(X)))

n = Net()
print(n.layer_a is n.layer_b)

# n.layer_a = torch.rand(size=(3, 3), requires_grad=True)
# print(n.layer_a.weight==n.layer_b.weight)
# print(n.layer_a.weight)
# print(n.layer_b.weight)