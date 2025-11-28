# 🚀 后端工程师转行 LLM 实战进阶指南

**文档说明**：
*   **目标读者**：拥有 4 年经验的服务器开发工程师（硕士）。
*   **核心策略**：以代码实战（Code-First）为主，理论直觉为辅。
*   **验收标准**：每个阶段必须产出可运行的代码或服务，拒绝“只看不练”。
*   **总耗时**：预计 3.5 - 4 个月（基于在职学习时间表）。

---

## 🛠️ 学习环境准备

1.  **代码仓库**：在 GitHub/GitLab 建立私有仓库 `llm-engineering-roadmap`。
2.  **IDE**：推荐 **Cursor** (AI 辅助编程) 或 VS Code/PyCharm + **GitHub Copilot**。
3.  **计算资源**：本地 Nvidia GPU (8G+) 或 Colab Pro / AutoDL。

---

## 📅 阶段一：PyTorch 破冰与深度学习基础 (Week 1-3)

**目标**：脱离文档写出 Training Loop，将后端思维转化为张量思维。

| 学习内容 (Input) | 重点知识点 |
| :--- | :--- |
| **李沐《动手学深度学习》**<br>- 第2章 (预备知识)<br>- 第3章 (线性神经网络)<br>- 第4章 (多层感知机) | - Tensor 维度变换 (View, Reshape, Permute)<br>- 广播机制 (Broadcasting)<br>- 自动求导 (Autograd) 原理<br>- `Dataset` 与 `DataLoader` 的封装 |

### ✅ 工程验收产出 (Definition of Done)

在仓库目录 `01_basics` 下提交以下代码：

**1. 任务 A：手撸线性回归 (No `nn.Linear`)**
*   **描述**：仅使用 Tensor 和 Autograd，拟合 $y = Xw + b$。
*   **验收标准**：
    *   [ ] 能够打印出每次迭代的 `w.grad`，确认梯度不为 0。
    *   [ ] 最终训练出的 $w$ 和 $b$ 与真实值误差小于 0.01。

**2. 任务 B：Fashion-MNIST 分类器**
*   **描述**：使用 `torch.nn` 搭建 MLP，完成图像分类。
*   **验收标准**：
    *   [ ] **可视化**：使用 `matplotlib` 或 `tensorboard` 画出 Loss 随 Epoch 下降的曲线（必须平滑下降）。
    *   [ ] **推理测试**：编写一个 `predict.py`，加载训练好的模型，输入一张图片，输出分类 Label。

> **👮 面试自测**：请在白板上写出一个标准的 PyTorch 训练循环（清零、前向、反向、更新）。

---

## 📅 阶段二：序列模型与文本处理 (Week 4-6)

**目标**：掌握变长序列处理，理解 NLP 的起源。

| 学习内容 (Input) | 重点知识点 |
| :--- | :--- |
| **李沐《动手学深度学习》**<br>- 第8章 (RNN)<br>- 第9章 (LSTM/GRU)<br>**吴恩达 Course 5** (听概念) | - 词嵌入 (Embedding) 的物理含义<br>- 隐状态 (Hidden State) 的传递<br>- 梯度裁剪 (Gradient Clipping)<br>- 困惑度 (Perplexity) 指标 |

### ✅ 工程验收产出 (Definition of Done)

在仓库目录 `02_rnn_char_gen` 下提交以下代码：

**任务：莎士比亚风格文本生成器 (Char-RNN)**
*   **描述**：基于 LSTM 模型，输入字符序列预测下一个字符。
*   **验收标准**：
    *   [ ] **Shape 对齐**：代码中需注释每一步 Tensor 的 Shape 变化 `(Batch, Seq, Hidden)`。
    *   [ ] **生成测试**：给定开头 "The King"，模型能生成一段虽然语法诡异但单词拼写正确的文本。
    *   [ ] **保存机制**：实现 Checkpoint 保存逻辑，支持中断后恢复训练。

> **👮 面试自测**：为什么 LSTM 能解决 RNN 的梯度消失问题？Input Gate 和 Forget Gate 分别起了什么作用？

---

## 📅 阶段三：Transformer 架构攻坚 (Week 7-10) 🔥 *关键期*

**目标**：彻底吃透 Transformer，不再把模型当黑盒。

| 学习内容 (Input) | 重点知识点 |
| :--- | :--- |
| **李沐 D2L 第10章**<br>**The Annotated Transformer** (GitHub)<br>**吴恩达 C5 Week 3-4** | - Self-Attention 计算公式 (Q, K, V)<br>- Multi-head 机制<br>- Positional Encoding (绝对 vs 相对)<br>- Mask 机制 (Padding Mask & Look-ahead Mask) |

### ✅ 工程验收产出 (Definition of Done)

在仓库目录 `03_transformer_scratch` 下提交以下代码：

**任务：从零实现 Mini-GPT**
*   **描述**：不使用 `nn.Transformer`，手动实现 `Attention` 类和 `Block` 类。
*   **验收标准**：
    *   [ ] **核心组件**：代码必须包含手写的 `ScaledDotProductAttention` 类。
    *   [ ] **过拟合测试**：只给模型喂一句话（如 "Server Engineer turns to AI"），训练 100 轮，模型必须能 100% 复述这句话（Loss 接近 0）。
    *   [ ] **Mask 验证**：打印 Attention 矩阵，确认下三角部分是有值的，上三角部分是被 Mask 掉的（负无穷）。

> **👮 面试自测**：Attention 公式里为什么要除以 $\sqrt{d_k}$？Softmax 是在哪个维度上做的？

---

## 📅 阶段四：Hugging Face 与 微调实战 (Week 11-14)

**目标**：掌握工业界标准流水线，解决显存限制问题。

| 学习内容 (Input) | 重点知识点 |
| :--- | :--- |
| **Hugging Face NLP Course**<br>(Focus: Fine-tuning)<br>学习 `PEFT` 库 | - Tokenizer (BPE/WordPiece)<br>- LoRA / QLoRA 原理<br>- 显存优化 (Gradient Checkpointing)<br>- 模型合并 (Merge LoRA weights) |

### ✅ 工程验收产出 (Definition of Done)

在仓库目录 `04_finetune_sft` 下提交以下代码：

**任务：让大模型进行“角色扮演” (LoRA 微调)**
*   **描述**：微调一个 7B 或 0.5B 的模型，使其学会特定说话风格（如海盗、鲁迅、客服）。
*   **验收标准**：
    *   [ ] **环境配置**：成功跑通 `bitsandbytes` 4bit 量化加载。
    *   [ ] **对比测试**：
        *   微调前：问“你好”，答“你好，我是AI助手”。
        *   微调后：问“你好”，答“俺是杰克船长，想找宝藏吗？”（风格明显变化）。
    *   [ ] **产物交付**：成功导出合并后的模型文件，并能用单独的脚本加载推理。

> **👮 面试自测**：在使用 LoRA 时，我们到底训练了模型的哪部分参数？Rank (r) 的大小对效果有什么影响？

---

## 📅 阶段五：RAG 系统与工程化落地 (持续进行)

**目标**：发挥后端优势，构建高并发 AI 应用。

| 学习内容 (Input) | 重点知识点 |
| :--- | :--- |
| **LangChain / LlamaIndex**<br>**Vector DB (Milvus/Chroma)**<br>**vLLM** | - 向量检索 (ANN Index)<br>- 文档切片 (Chunking Strategy)<br>- 流式响应 (SSE)<br>- 检索重排序 (Rerank) |

### ✅ 工程验收产出 (Definition of Done)

在仓库目录 `05_rag_backend` 下提交以下代码：

**任务：基于 PDF 的私有知识库问答 API**
*   **描述**：纯后端项目，FastAPI + VectorDB + LLM。
*   **验收标准**：
    *   [ ] **全链路跑通**：上传 PDF -> 解析切片 -> 存入向量库 -> 提问 -> 检索 TopK -> LLM 回答。
    *   [ ] **流式输出**：API 接口支持 SSE (Server-Sent Events)，模拟打字机效果。
    *   [ ] **幻觉测试**：问文档里没有的问题，模型应回答“文档未提及”，而非瞎编。
    *   [ ] **性能指标**：记录从提问到第一个字生成的延迟 (TTFT)，并尝试通过优化 Prompt 或检索参数来降低延迟。

> **👮 面试自测**：如果 RAG 检索回来的文档片段太多，超过了 LLM 的 Context Window，你怎么处理？

---

## 📝 总结

这份计划的核心在于**“Backend-Driven AI Learning”**。你不需要像科学家一样去推导所有数学公式，但你需要像资深工程师一样，保证每一行代码都是可运行、可测试、可部署的。

**Good Luck! 祝转型顺利！**