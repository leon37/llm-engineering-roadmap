# 【需求】
# 实现 CustomDropout 类。
# 【核心约束】
# 训练模式 (Train)：必须通过缩放 (Rescaling) 操作，确保输出的数学期望 (Expectation) 与输入保持一致。
# 预测模式 (Eval)：直接透传，不消耗计算资源。
# 禁止使用 nn.Dropout 或 F.dropout，只能用 torch.rand 等基础算子
import torch
from torch import nn

class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, X):
        if not self.training:
            return X
        if self.p == 1:
            return torch.zeros_like(X)
        elif self.p == 0:
            return X
        else:
            mask = (torch.rand(X.shape) > self.p).float()
            return X*mask/(1.0-self.p)

X = torch.ones((1000, 1000))
print(torch.sum(X))
d1 = CustomDropout(p = 0.3)
X_d1 = d1.forward(X)
print(torch.sum(X_d1))
d1.training=False
print(torch.sum(d1.forward(X)))