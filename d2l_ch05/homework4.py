# 背景：
# 线上正在运行 Model_V1。现在你开发了 Model_V2，架构做了一点微调：
# Model_V1: Linear(20, 64) -> ReLU -> Linear(64, 1)
# Model_V2: Linear(20, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)
# 需求：
# 实例化 V1，保存其参数为 checkpoint_v1.pth。
# 实例化 V2。
# 任务：尝试将 checkpoint_v1.pth 加载到 V2 中。
# 直接加载会报错吗？为什么？
# 请编写代码，实现“能复用的尽量复用，不能复用的保持 V2 的随机初始化”的加载逻辑（Partial Loading）。

import torch
from torch import nn

class Model_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

class Model_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

model1 = Model_V1()
torch.save(model1.state_dict(), 'checkpoint_v1.pth')
print(model1.state_dict())
model2 = Model_V2()
param = torch.load('checkpoint_v1.pth')
# 方法1
# model2.load_state_dict(param, strict=False)

# 方法2
model2_state_dict = model2.state_dict()
filtered_state = {k: v for k, v in param.items() if k in model2_state_dict and v.shape == model2_state_dict[k].shape}
model2_state_dict.update(filtered_state)
model2.load_state_dict(model2_state_dict)