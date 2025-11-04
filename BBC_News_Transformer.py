# transformer_text_classification.py
import os
import math
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
    "max_vocab_size": 15000,
    "max_len": 512,
    "embed_dim": 256,            # Transformer 的 d_model
    "nhead": 8,
    "dim_feedforward": 512,
    "num_encoder_layers": 4,
    "dropout": 0.5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "transformer_text_cls.pth"
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
# 4. Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# =========================
# 5. Transformer 分类模型
# =========================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 nhead, num_layers, dim_feedforward, dropout, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=PARAMS["max_len"] + 10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x == 0)  # padding mask (batch, seq_len) where True indicates pad
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)  # (B, L, D)
        x = self.pos_enc(x)
        # transformer with key_padding_mask (True -> is padded and will be ignored)
        out = self.transformer(x, src_key_padding_mask=mask)
        # 池化：用 mask 做 mean pooling（忽略 pad）
        mask_inv = (~mask).unsqueeze(-1).float()  # (B, L, 1)
        summed = (out * mask_inv).sum(dim=1)      # (B, D)
        lengths = mask_inv.sum(dim=1).clamp(min=1.0)  # (B,1)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

# =========================
# 6. 初始化模型、损失、优化器
# =========================
device = torch.device(PARAMS["device"])
model = TransformerClassifier(vocab_size=vocab_size,
                              embed_dim=PARAMS["embed_dim"],
                              num_classes=num_classes,
                              nhead=PARAMS["nhead"],
                              num_layers=PARAMS["num_encoder_layers"],
                              dim_feedforward=PARAMS["dim_feedforward"],
                              dropout=PARAMS["dropout"],
                              pad_idx=0).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=PARAMS["learning_rate"], weight_decay=1e-5)

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PARAMS['num_epochs'], eta_min=1e-6)

# =========================
# 7. 训练与验证函数
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
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
# 8. 训练主循环
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
# 9. 测试评估（加载最优模型）
# =========================
ckpt = torch.load(PARAMS["model_save_path"], map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
test_acc, test_report = evaluate(model, test_loader, device)
print("\nFinal Test Acc:", test_acc)
print("Test classification report:\n", test_report)
