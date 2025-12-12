# ==========================================
# 输入数据：Synthetic Text (Safe Mode)
# 长度：约 600+ 字符
# ==========================================

RAW_TEXT = """
Alex was a backend engineer who loved writing Go code. 
He worked on distributed systems and high concurrency services. 
One day, he decided to learn deep learning. 
He opened the D2L book and started studying Recurrent Neural Networks. 
At first, the dimensions of tensors were confusing. 
Shape mismatch errors appeared everywhere. 
But Alex did not give up. He debugged the code line by line. 
He learned about batch size, sequence length, and hidden states. 
Finally, his loss function started to decrease. 
The model generated text that looked like real English. 
Alex was happy. He realized that AI was just another system to master.
""" * 10  # <--- 注意：我把它重复了 10 遍，增加数据量

# 你需要实现以下 4 个模块（Structs/Functions）：
# 1. 数据管道 (The Data Pipeline)
# 你需要把原始字符串转换成 PyTorch 能吃的 Batch。
# 输入：Raw Text String。
# 要求：
# 实现字符到索引的映射 (char_to_idx)。
# 构造数据迭代器。挑战：请使用 随机采样 (Random Sampling) 策略生成 Batch。
# 返回 X (Input) 和 Y (Target)。
# Hint: X 和 Y 的 Shape 应该是 (Batch, Seq_Len)。
# 2. 模型定义 (The Model Service)
# 定义一个继承自 nn.Module 的类。
# 组件：
# nn.Embedding (或者用 One-Hot 预处理，推荐尝试 Embedding，input_size 设为 256)。
# nn.RNN (尝试设为 num_layers=1, hidden_size=256, batch_first=True)。
# nn.Linear (输出层)。
# Forward 函数：
# 输入 (x, state)。
# 输出 (logits, state)。
# 注意：这里需要处理好 Linear 层的维度变换。
# 3. 推理逻辑 (The Inference Engine)
# 实现 predict(prefix, num_chars) 函数。
# 难点：
# 预热 (Warm-up)：先喂入 prefix 更新隐状态。
# 自回归 (Auto-regressive)：取最后一个时间步的输出 -> argmax -> 拿到字符 -> append 到结果 -> 把这个字符作为下一步的输入。
# 4. 训练循环 (The Training Loop) —— 最核心的考核点
# 编写 train() 函数。
# 要求：
# 使用 CrossEntropyLoss。
# 使用 SGD 或 Adam 优化器。
# 关键逻辑：在每个 Batch 开始时，如何初始化 State？（回忆刚才关于随机采样的讨论）。
# 监控：每隔 100 个 Epoch，打印一次当前的 Loss，并调用 predict 看看模型现在能写出什么鬼东西（一开始是乱码，慢慢变成单词）。

import torch
from torch import nn
import torch.nn.functional as F
import random

text = RAW_TEXT.replace('\n', ' ').lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
corpus_indices = [char_to_idx[char] for char in text]

def data_iter_random(corpus, batch_size, num_steps, device=None):
    num_examples = (len(corpus)-1) // num_steps
    num_batches = num_examples // batch_size

    initial_indices = list(range(0, num_examples*num_steps, num_steps))
    random.shuffle(initial_indices)

    for i in range(0, batch_size*num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i+batch_size]
        X_list=[]
        Y_list=[]

        for start_idx in initial_indices_per_batch:
            end_idx = start_idx+num_steps
            X_list.append(corpus[start_idx:end_idx])
            Y_list.append(corpus[start_idx+1:end_idx+1])

        X = torch.tensor(X_list, dtype=torch.long)
        Y = torch.tensor(Y_list, dtype=torch.long)

        if device:
            X = X.to(device)
            Y = Y.to(device)
        yield X, Y

class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.vocab_size = input_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, X, state):
        X = F.one_hot(X.long(), self.vocab_size).float()
        Y, state = self.rnn(X, state)

        output = self.linear(Y)
        return output, state

# 3. 推理逻辑 (The Inference Engine)
# 实现 predict(prefix, num_chars) 函数。
# 难点：
# 预热 (Warm-up)：先喂入 prefix 更新隐状态。
# 自回归 (Auto-regressive)：取最后一个时间步的输出 -> argmax -> 拿到字符 -> append 到结果 -> 把这个字符作为下一步的输入。

def predict_rnn(prefix, num_preds, model, device):
    """
    prefix: 提示字符串，如 "The time"
    num_preds: 续写长度，如 10
    """
    # 0. 切换到评估模式 (关闭 Dropout 等，虽然这里没用 Dropout)
    # 并且告诉 PyTorch 不需要算梯度 (节省内存)
    state = None
    output = [char_to_idx[prefix[0].lower()]]  # 记录结果的列表，先放入第一个字

    # 这里的逻辑是：
    # 比如 prefix 是 "ABC"，我们要先让模型读 "A"->"B"->"C"
    # 等读完 "C" 时，state 就包含了 "ABC" 的语义
    # 然后我们用 "C" 去预测 "D"

    with torch.no_grad():  # 推理模式，不开计算图
        # ==========================
        # 1. 预热 (Warm-up)
        # ==========================
        # 把 prefix 除了最后一个字之外，全部喂进去更新 state
        for i in range(len(prefix) - 1):
            # 制作输入 X: Shape (1, 1) -> (Batch=1, Seq=1)
            X = torch.tensor([[char_to_idx[prefix[i].lower()]]], device=device)

            # 只为了更新 state，output 我们不关心
            _, state = model(X, state)

            # 顺便把字符加到结果里
            output.append(char_to_idx[prefix[i + 1].lower()])

        # ==========================
        # 2. 生成 (Generation)
        # ==========================
        # 起始输入是 prefix 的最后一个字符
        last_char_idx = char_to_idx[prefix[-1].lower()]

        for _ in range(num_preds):
            # 制作输入 X: Shape (1, 1)
            X = torch.tensor([[last_char_idx]], device=device)

            # 扔进模型
            # Y_hat shape: (1, 1, vocab_size)
            # state: 更新后的状态
            Y_hat, state = model(X, state)

            # 贪婪采样 (Greedy Search): 选概率最大的那个字的索引
            # argmax(dim=2) 拿到 index
            # .item() 把 Tensor 里的数字取出来变成 Python int
            pred_idx = Y_hat.argmax(dim=2).item()

            # 存下来
            output.append(pred_idx)

            # 关键点！自回归！
            # 现在的输出，变成下一次循环的输入
            last_char_idx = pred_idx

    # 解码：把索引列表变回字符串
    return ''.join([idx_to_char[i] for i in output])


def train_rnn(model, data_iter_fn, corpus_indices, vocab_size,
              num_epochs, batch_size, num_steps, lr, device):
    # 1. 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()  # 开启训练模式

    print("开始训练...")

    for epoch in range(num_epochs):
        # 记录累积的 Loss 和 Token 数量，用来算这个 Epoch 的平均 PPL
        total_loss = 0.0
        total_tokens = 0

        # 获取数据迭代器 (每次随机采样)
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)

        for X, Y in data_iter:
            # A. 初始化 State (随机采样模式：每个Batch都要清零！)
            # 注意：state 不需要梯度，初始全0
            state = None

            # B. 前向传播
            # X: (Batch, Seq)
            # Y: (Batch, Seq)
            # output: (Batch, Seq, Vocab)
            output, state = model(X, state)

            # C. 算 Loss
            # CrossEntropy 需要 (N, C) 的输入
            # 所以我们要把 Batch 和 Seq 维度拍扁
            # output -> (Batch * Seq, Vocab)
            # Y      -> (Batch * Seq)
            l = loss_fn(output.reshape(-1, vocab_size), Y.reshape(-1))

            # D. 反向传播
            optimizer.zero_grad()
            l.backward()

            # (可选) 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # E. 累积统计 (为了展示)
            total_loss += l.item() * Y.numel()  # 乘回元素个数
            total_tokens += Y.numel()

        # End of Epoch
        if (epoch + 1) % 50 == 0:  # 每 50 轮打印一次
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)  # 困惑度
            print(f"Epoch {epoch + 1}: PPL {ppl:.1f}")

            # 顺便预测一下，看看效果
            print("生成文本: ", predict_rnn("The", 20, model, device))
            print("-" * 30)


import math  # 别忘了引入 math

# 简单的测试
if __name__ == '__main__':

    hidden_size = 256
    learning_rate = 0.01

    # 实例化模型
    # 注意：输入维度就是词表大小（因为是 One-Hot）
    model = RNNModule(input_size=vocab_size, hidden_size=hidden_size)

    # 放到 GPU (如果可用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"模型已加载到: {device}")

    # 造一个 dummy input 测一下
    # 假设 batch=2, seq_len=5
    dummy_X = torch.randint(0, vocab_size, (2, 5)).to(device)
    dummy_state = None  # 让 RNN 自己初始化全0

    # try:
    #     dummy_Y, _ = model(dummy_X, dummy_state)
    #     print(f"测试通过！输出 Shape: {dummy_Y.shape}")
    #     # 预期输出: torch.Size([2, 5, vocab_size])
    # except Exception as e:
    #     print(f"测试失败: {e}")

    # ========================
    # 单元测试：推理逻辑
    # ========================
    print("开始测试预测模块...")
    try:
        # 随便预测一下，看看能不能跑出 5 个字
        test_str = predict_rnn("The", 5, model, device)
        print(f"预测结果: {test_str}")
        # 预期输出: "The" 后面跟 5 个随机字符 (因为模型还没训练)
    except Exception as e:
        print(f"预测模块报错: {e}")