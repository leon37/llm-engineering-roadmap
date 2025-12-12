# 背景：
# 现在的模型初始化策略太暴力了（全员 Xavier）。我们需要根据组件的“标签”做精细化配置。
#
# 需求：
# 编写一个 init_custom_weights(m) 函数，并用 net.apply() 应用到一个深层网络上。逻辑如下：
# 如果是 nn.Linear 层，且变量名（可以在 apply 外部通过命名访问，或者简单的根据层类型判断）是输出层（假设你定义的网络最后一名叫 output），则将其权重初始化为 全 1。
# 如果是其他的 nn.Linear 层，将其权重初始化为 均值为0，标准差为0.01的高斯分布。
#
# 主要考点：如何在 apply 循环中区分“这是哪个层”？（提示：apply 只传了 module 实例，没传名字，怎么解决这个问题？如果解决不了，有没有其他遍历参数的方法？）

import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 64)
        self.output = nn.Linear(64, 1)


def init_custom_weights(net):
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            if 'output' in name:
                nn.init.ones_(m)
            else:
                nn.init.normal_(m.weight, 0, 0.01)

net = Net()
net.apply(init_custom_weights)