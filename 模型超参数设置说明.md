# BBC新闻文本分类 - 模型超参数设置说明

## 📊 超参数设置及理由

### 1️⃣ 通用超参数（所有模型保持一致）

| 参数 | 值 | 理由 |
|------|-----|------|
| `max_vocab_size` | 10,000 | 与CNN/LSTM保持一致，确保公平对比 |
| `max_len` | 100 | 与CNN/LSTM保持一致（Transformer使用128） |
| `embed_dim` | 100 | 与CNN保持一致，适中的词嵌入维度（Transformer使用256） |
| `batch_size` | 32/64 | RNN/CNN/LSTM/GRU使用32，Transformer/MLP使用64 |
| `learning_rate` | 1e-3 | 与其他模型完全一致 |
| `num_epochs` | 100 | 与其他模型保持一致 |
| `dropout` | 0.3-0.5 | 根据模型复杂度调整 |

---

### 2️⃣ RNN特定超参数设置

```python
# RNN配置
"hidden_dim": 256,         # 参考train.py的配置
"num_layers": 1,           # RNN使用单层（比LSTM/GRU简单）
"bidirectional": False,    # RNN使用单向（降低复杂度）
"dropout": 0.5,            # 标准dropout率
```

**设置理由：**
- **单层+单向**：RNN是最基础的循环网络，容易梯度消失，使用简单配置作为baseline
- 参考train.py第414-415行，RNN就是用单层单向配置
- 这样可以体现RNN的**基础性能**，便于与LSTM/GRU对比
- 适合作为序列模型的性能下限参考

---

### 3️⃣ LSTM特定超参数设置

```python
# LSTM配置
"hidden_dim": 128,         # 标准配置
"num_layers": 2,           # LSTM使用双层
"bidirectional": True,     # 使用双向LSTM
"dropout": 0.5,            # 标准dropout率
```

**设置理由：**
- **双层+双向**：LSTM通过门控机制解决梯度消失问题，能够捕捉长距离依赖
- 双向结构可以同时利用前向和后向的上下文信息
- 2层结构在性能和计算成本之间取得平衡
- 是序列模型中最常用的标准配置

---

### 4️⃣ GRU特定超参数设置

```python
# GRU配置
"hidden_dim": 256,         # 参考train.py的配置
"num_layers": 2,           # GRU使用双层（与LSTM对齐）
"bidirectional": True,     # GRU使用双向（与LSTM对齐）
"dropout": 0.5,            # 标准dropout率
```

**设置理由：**
- **双层+双向**：GRU是LSTM的变体，性能相近，使用相同配置便于直接对比
- 参考train.py第448-449行，GRU使用与LSTM相同的配置
- GRU比LSTM参数少约25%，但性能相近，这是其核心优势
- 通过对齐配置可以直接体现GRU的"轻量化"特点

---

### 5️⃣ CNN特定超参数设置

```python
# CNN配置
"embed_dim": 100,          # 词向量维度
"kernel_sizes": [3, 4, 5], # 多尺寸卷积核
"num_filters": 100,        # 每种卷积核数量
"dropout": 0.5,            # 标准dropout率
```

**设置理由：**
- **多尺寸卷积核**：捕捉不同长度的n-gram特征（trigram, 4-gram, 5-gram）
- 每种卷积核使用100个filter，总共300个特征
- CNN在文本分类中善于捕捉局部模式
- 不需要循环结构，训练速度快

---

### 6️⃣ Transformer特定超参数设置

```python
# Transformer配置
"max_vocab_size": 15000,   # 更大的词汇表
"max_len": 128,            # 更长的序列长度
"embed_dim": 256,          # 更大的词向量维度
"nhead": 8,                # 8个注意力头
"dim_feedforward": 512,    # 前馈网络维度
"num_encoder_layers": 4,   # 4层编码器
"dropout": 0.5,            # 标准dropout率
"batch_size": 64,          # 更大的批次
```

**设置理由：**
- **更大的容量**：Transformer需要更大的模型容量来充分发挥自注意力机制的优势
- **多头注意力**：8个注意力头可以从不同角度捕捉词间关系
- **更长序列**：Transformer善于处理长距离依赖，使用128长度
- **位置编码**：必须添加位置信息，因为自注意力是位置不敏感的
- 参数量较大，但并行度高，训练效率好

---

### 7️⃣ MLP特定超参数设置

```python
# MLP配置
"max_features": 5000,      # TF-IDF最大特征数
"hidden1": 512,            # 第一隐藏层维度
"hidden2": 128,            # 第二隐藏层维度
"dropout1": 0.4,           # 第一层dropout
"dropout2": 0.3,           # 第二层dropout
"batch_size": 64,          # 批次大小
```

**设置理由：**
- **TF-IDF特征**：MLP是前馈网络，不处理序列，使用传统的TF-IDF向量化
- **两层隐藏层**：512→128的降维结构，逐步提取抽象特征
- **较低的dropout**：TF-IDF特征已经比较稀疏，不需要太强的正则化
- 作为非序列模型的baseline

---

## 🎯 模型对比设置总结

| 模型 | 层数 | 方向 | Hidden Dim | Embed Dim | 参数特点 | 设计理念 |
|------|------|------|-----------|-----------|---------|----------|
| **MLP** | 2 | - | 512→128 | - | TF-IDF输入 | 传统前馈网络baseline |
| **RNN** | 1 | 单向 | 256 | 100 | 单层单向 | 最基础的序列模型 |
| **LSTM** | 2 | 双向 | 128 | 100 | 双层双向 | 解决梯度消失问题 |
| **GRU** | 2 | 双向 | 256 | 100 | 参数比LSTM少25% | LSTM的轻量化变体 |
| **CNN** | - | - | 100 filters×3 | 100 | 多尺寸卷积核 | 局部特征提取 |
| **Transformer** | 4层Encoder | - | 256 | 256 | 8头注意力 | 全局自注意力机制 |

---

## 🔍 为什么这样设计？

### 1. **RNN作为Baseline**
- 单层单向配置，体现vanilla RNN的基础性能
- 突出其梯度消失问题和序列建模能力限制
- 为LSTM/GRU提供性能对比基准

### 2. **GRU与LSTM对齐**
- 使用相同的层数和方向配置
- 便于直接对比：参数量少 vs 性能是否相近
- 体现GRU的"效率优势"（更少参数，相近性能）

### 3. **Hidden Dim差异**
- LSTM用128维（已有代码）
- RNN/GRU用256维（参考train.py）
- 这不影响公平性，因为我们关注的是**架构差异**而非参数量绝对值

### 4. **Transformer使用更大容量**
- 自注意力机制需要更多参数才能充分发挥优势
- 更大的词汇表和序列长度利用其长距离建模能力
- 体现现代深度学习模型的"大力出奇迹"特点

### 5. **MLP使用传统特征**
- TF-IDF是经典的文本表示方法
- 作为非序列模型的baseline
- 对比深度学习方法的优势

---

## ⚙️ 特殊处理说明

### 1. **文本预处理差异**

#### 序列模型（RNN/LSTM/GRU/CNN/Transformer）
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 使用Tokenizer进行分词和序列化
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text),
                            maxlen=max_len, padding='post', truncating='post')
```

**特点：**
- 保留词序信息
- 输出为整数索引序列
- 需要Embedding层将索引映射为向量

#### 非序列模型（MLP）
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text).toarray()
```

**特点：**
- 不保留词序（词袋模型）
- 输出为稀疏特征向量
- 直接输入网络，不需要Embedding层

---

### 2. **位置编码（仅Transformer）**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # ...
```

**为什么只有Transformer需要？**
- Transformer的自注意力机制是**位置不敏感**的
- 没有位置编码，"I love you"和"you love I"完全一样
- RNN/LSTM/GRU通过递归结构天然包含位置信息
- CNN通过卷积操作隐式编码局部位置关系

---

### 3. **Pack Padded Sequence（RNN/LSTM/GRU）**

```python
# 计算实际长度
lengths = (x != 0).sum(dim=1).cpu()

# Pack序列以忽略padding
packed_x = nn.utils.rnn.pack_padded_sequence(
    x, lengths, batch_first=True, enforce_sorted=False
)

# RNN/LSTM/GRU前向传播
packed_output, hidden = self.rnn(packed_x)
```

**为什么需要Pack？**
- 提高计算效率：跳过padding部分的计算
- 避免padding对梯度的影响
- 更准确地建模变长序列

**为什么CNN/Transformer不需要？**
- CNN：使用padding_idx在Embedding层处理
- Transformer：使用key_padding_mask在注意力层处理

---

### 4. **双向结构处理**

#### 双向LSTM/GRU
```python
if self.bidirectional:
    # 拼接forward和backward的最后隐藏状态
    hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim*2)
else:
    # 单向，取最后一层的隐藏状态
    hidden = h_n[-1]  # (batch, hidden_dim)
```

#### Transformer（不需要双向）
- Transformer通过自注意力机制同时看到所有位置
- 天然就是"全向"的，不需要显式双向结构

---

### 5. **梯度裁剪（所有模型一致）**

```python
# 梯度裁剪，防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**为什么重要？**
- RNN系列模型容易梯度爆炸
- 深度网络（Transformer）也需要梯度裁剪
- 统一设置max_norm=1.0确保训练稳定性

---

### 6. **学习率调度器（所有模型一致）**

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=PARAMS['num_epochs'],
    eta_min=1e-6
)

# 在每个epoch后调用
scheduler.step()
```

**为什么使用Cosine Annealing？**
- 学习率平滑衰减，训练更稳定
- 初期大学习率快速收敛
- 后期小学习率精细调整
- 被证明对各种模型都有效

---

## 📝 模型保存格式（统一）

所有模型都使用相同的checkpoint格式：

```python
torch.save({
    "model_state": model.state_dict(),
    "tokenizer_word_index": tokenizer.word_index,  # 或 vectorizer_vocabulary
    "params": PARAMS,
    "label_classes": le.classes_.tolist()
}, model_save_path)
```

**包含信息：**
- 模型权重
- 文本处理工具的词汇表
- 超参数配置
- 标签映射关系

**便于：**
- 模型部署和推理
- 实验复现
- 结果对比

---

## 🎓 总结

### 设计原则

1. **公平性**：相同任务使用相同的数据处理和训练流程
2. **可比性**：超参数设置合理，突出模型架构差异
3. **完整性**：每个模型都是独立可运行的完整脚本
4. **一致性**：代码风格、注释格式、打印输出完全统一

### 预期对比结果

| 维度 | RNN | LSTM | GRU | CNN | Transformer | MLP |
|------|-----|------|-----|-----|-------------|-----|
| **准确率** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **训练速度** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **参数量** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **长依赖建模** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |

---