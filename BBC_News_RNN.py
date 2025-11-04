# rnn_text_classification.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# =========================
# 超参数配置
# =========================
PARAMS = {
    "max_vocab_size": 10000,   # 最大词汇表
    "max_len": 400,            # 最大序列长度
    "embed_dim": 100,          # 词向量维度
    "hidden_dim": 256,         # RNN隐藏层维度
    "num_layers": 1,           # RNN层数
    "bidirectional": False,    # 是否使用双向RNN
    "dropout": 0.5,            # dropout概率
    "batch_size": 32,          # 批次大小
    "learning_rate": 1e-3,     # 学习率
    "num_epochs": 100,         # 训练轮次
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "rnn_text_cls.pth"
}

print("使用设备:", PARAMS["device"])

# =========================
# 1. 数据读取
# =========================
train_df = pd.read_csv('D:/NTU/EE6405/Group Project/BBC_News/data/train.csv')
val_df   = pd.read_csv('D:/NTU/EE6405/Group Project/BBC_News/data/val.csv')
test_df  = pd.read_csv('D:/NTU/EE6405/Group Project/BBC_News/data/test.csv')

# 使用第五列为文本输入，第二列为标签
X_train_text = train_df.iloc[:, 4].astype(str)
y_train_text = train_df.iloc[:, 1].astype(str)
X_val_text   = val_df.iloc[:, 4].astype(str)
y_val_text   = val_df.iloc[:, 1].astype(str)
X_test_text  = test_df.iloc[:, 4].astype(str)
y_test_text  = test_df.iloc[:, 1].astype(str)
print(X_train_text.head())

# =========================
# 2. 标签编码
# =========================
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_val   = le.transform(y_val_text)
y_test  = le.transform(y_test_text)
num_classes = len(le.classes_)
print("类别数:", num_classes, "Classes:", list(le.classes_))

# =========================
# 3. Tokenizer -> 序列化
# =========================
tokenizer = Tokenizer(num_words=PARAMS["max_vocab_size"], oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text.tolist())

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=PARAMS["max_len"], padding='post', truncating='post')
X_val_seq   = pad_sequences(tokenizer.texts_to_sequences(X_val_text),   maxlen=PARAMS["max_len"], padding='post', truncating='post')
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test_text),  maxlen=PARAMS["max_len"], padding='post', truncating='post')

vocab_size = min(len(tokenizer.word_index) + 1, PARAMS["max_vocab_size"])
print("词表大小:", vocab_size)

# 转为 tensor
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor   = torch.tensor(X_val_seq, dtype=torch.long)
y_val_tensor   = torch.tensor(y_val, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test_seq, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# Dataset & DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TextDataset(X_train_tensor, y_train_tensor), batch_size=PARAMS["batch_size"], shuffle=True)
val_loader   = DataLoader(TextDataset(X_val_tensor, y_val_tensor), batch_size=PARAMS["batch_size"], shuffle=False)
test_loader  = DataLoader(TextDataset(X_test_tensor, y_test_tensor), batch_size=PARAMS["batch_size"], shuffle=False)

# =========================
# 4. RNN 分类模型
# =========================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers, dropout, bidirectional, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding层：将词索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # RNN层：vanilla RNN，处理序列信息
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,  # 多层RNN才使用dropout
            batch_first=True
        )

        # 全连接层：将RNN输出映射到类别空间
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, num_classes)

        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, seq_len] - 输入文本的token索引
        Returns:
            logits: [batch_size, num_classes] - 分类logits
        """
        # x: (batch, seq_len)
        # 计算每个样本的实际长度（非padding部分）
        lengths = (x != 0).sum(dim=1).cpu()

        # Embedding + Dropout
        x = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)

        # Pack序列以忽略padding
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # RNN前向传播
        packed_output, h_n = self.rnn(packed_x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)

        # 使用最后一层的隐藏状态进行分类
        if self.bidirectional:
            # 拼接双向RNN的最后隐藏状态（forward和backward）
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim*2)
        else:
            # 单向RNN，取最后一层的隐藏状态
            hidden = h_n[-1]  # (batch, hidden_dim)

        # 应用dropout和全连接层
        out = self.dropout(hidden)
        logits = self.fc(out)

        return logits

# =========================
# 5. 初始化模型、损失、优化器
# =========================
device = torch.device(PARAMS["device"])
model = RNNClassifier(vocab_size=vocab_size,
                     embed_dim=PARAMS["embed_dim"],
                     hidden_dim=PARAMS["hidden_dim"],
                     num_classes=num_classes,
                     num_layers=PARAMS["num_layers"],
                     dropout=PARAMS["dropout"],
                     bidirectional=PARAMS["bidirectional"],
                     pad_idx=0).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PARAMS['num_epochs'], eta_min=1e-6)

# =========================
# 6. 训练与验证函数
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            p = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p)
            ys.extend(y_batch.numpy().tolist())
    acc = accuracy_score(ys, preds)
    report = classification_report(ys, preds, target_names=le.classes_, zero_division=0)
    return acc, report

# =========================
# 7. 训练主循环
# =========================
best_val_acc = 0.0
print("训练参数:", PARAMS)
for epoch in range(1, PARAMS["num_epochs"] + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    scheduler.step()
    val_acc, val_report = evaluate(model, val_loader, device)
    print(f"Epoch {epoch}/{PARAMS['num_epochs']} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "tokenizer_word_index": tokenizer.word_index,
            "params": PARAMS,
            "label_classes": le.classes_.tolist()
        }, PARAMS["model_save_path"])
        print("Saved best model ->", PARAMS["model_save_path"])

# =========================
# 8. 测试评估（加载最优模型）
# =========================
ckpt = torch.load(PARAMS["model_save_path"], map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
test_acc, test_report = evaluate(model, test_loader, device)
print("\nFinal Test Acc:", test_acc)
print("Test classification report:\n", test_report)