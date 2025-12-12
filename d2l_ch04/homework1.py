# 【背景】
# 书里的 MLP 都是硬编码的（Hardcoded），比如写死 nn.Linear(784, 256)。
# 但在实际工程中，模型的层数、每层的宽度通常是配置项（Config），存储在 YAML 或 JSON 里，需要动态生成。
# 【需求】
# 请编写一个函数 build_mlp，它接收一个配置列表，自动组装出一个多层的神经网络。
from torch import nn

def build_mlp(input_dim: int, output_dim: int, hidden_sizes: list[int], dropout_p: float = 0.0) -> nn.Sequential:
    """
    Args:
        input_dim: 输入维度 (如 784)
        output_dim: 输出维度 (如 10)
        hidden_sizes: 一个列表，定义中间每一层的宽度，如 [256, 128]
        dropout_p: 如果大于 0，则在每个激活函数后添加 Dropout 层

    Returns:
        一个构建好的 nn.Sequential 对象
    """
    # 你的代码写在这里
    # 禁止查阅书本，仅凭逻辑推导
    seq = nn.Sequential()
    pre_size = input_dim
    for hidden_size in hidden_sizes:
        seq.append(nn.Linear(pre_size, hidden_size))
        seq.append(nn.ReLU())
        if dropout_p > 0:
            seq.append(nn.Dropout(p=dropout_p))
        pre_size = hidden_size
    seq.append(nn.Linear(pre_size, output_dim))
    return seq

net = build_mlp(784, 10, [256, 64], dropout_p=0.5)
print(net)
