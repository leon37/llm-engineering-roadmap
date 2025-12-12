# 背景：
# 普通的 nn.Sequential 就像一条直线的流水线。但在实际业务中，我们经常有“旁路逻辑”（Bypass/Skip Connection），比如 ResNet 的残差连接。
#
# 需求：
# 请编写一个名为 BypassBlock 的类（继承自 nn.Module），它包含两个子层：
# self.main_path: 一个 nn.Linear 层。
# self.bypass: 一个不含参数的自定义逻辑。
# Forward 逻辑：
# 输入 x。
# 如果 x 的均值大于 0，走主路：返回 self.main_path(x)。
# 如果 x 的均值小于等于 0，走旁路：返回 x 原样输出（Identity）。
#
# 挑战：打印出每次 forward 到底走了哪条路，验证动态流控是否生效。

import torch
from torch import nn

class BypassBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_path = nn.Linear(20, 10)

    def forward(self, X):
        if X.mean() > 0:
            print(f'X.mean() is {X.mean()}, main path')
            return self.main_path(X)
        else:
            print(f'X.mean() is {X.mean()}, bypass')
            return X

net = BypassBlock()
X = torch.rand((2, 20))
print(net(X))