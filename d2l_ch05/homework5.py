# 背景：
# 你的模型代码将被部署到不同的环境：有的机器有 GPU，有的只有 CPU。你的 Forward 函数不能写死 cuda。

# 需求：
# 编写一个函数 robust_forward(net, x)：
# 检测 net 目前在哪个设备上（是 CPU 还是 GPU？）。
# 检查输入数据 x 在哪个设备上。

# 如果两者不一致，自动将 x 搬运到 net 所在的设备。
# 执行计算并返回结果。
# 测试：模拟一个场景：net 在 GPU 上，x 在 CPU 上，调用该函数，确保不报错（Expected device 错误）。

import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10).to('cuda:0')

    def forward(self, X):
        return self.layer(X)

net = Net()
x = torch.rand((1, 10), device='cpu')
def robust_forward(net, x):
    try:
        target_device = next(net.parameters()).device
    except StopIteration:
        target_device = x.device

    if target_device != x.device:
        x = x.to(target_device)
    return net(x)

robust_forward(net, x)

