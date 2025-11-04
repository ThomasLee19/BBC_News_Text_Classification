# mlp_text_classification.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# =========================
# 超参数配置
# =========================
PARAMS = {
    "max_features": 5000,      # TF-IDF最大特征数
    "hidden1": 512,            # 第一隐藏层维度
    "hidden2": 128,            # 第二隐藏层维度
    "dropout1": 0.4,           # 第一层dropout概率
    "dropout2": 0.3,           # 第二层dropout概率
    "batch_size": 64,          # 批次大小
    "learning_rate": 1e-3,     # 学习率
    "num_epochs": 100,         # 训练轮次
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "mlp_text_cls.pth"
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
# 3. 文本向量化（TF-IDF）
# =========================
vectorizer = TfidfVectorizer(max_features=PARAMS["max_features"])
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_val   = vectorizer.transform(X_val_text).toarray()
X_test  = vectorizer.transform(X_test_text).toarray()

input_dim = X_train.shape[1]
print("TF-IDF特征维度:", input_dim)

# 转为 tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
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
# 4. MLP 分类模型
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout1, dropout2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        return logits

# =========================
# 5. 初始化模型、损失、优化器
# =========================
device = torch.device(PARAMS["device"])
model = MLPClassifier(input_dim=input_dim,
                     hidden1=PARAMS["hidden1"],
                     hidden2=PARAMS["hidden2"],
                     num_classes=num_classes,
                     dropout1=PARAMS["dropout1"],
                     dropout2=PARAMS["dropout2"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PARAMS['num_epochs'], eta_min=1e-6)

# =========================
# 6. 训练与验证函数
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
            "vectorizer_vocabulary": vectorizer.vocabulary_,
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
