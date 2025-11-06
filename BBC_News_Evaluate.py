# evaluate_all_models.py
"""
BBC News Text Classification 
Evaluate six models: MLP, RNN, LSTM, GRU, CNN, and Transformer, and generate comparison charts
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# =========================
# 配置
# =========================
PARAMS = {
    "test_data_path": "D:/NTU/EE6405/Group Project/BBC_News/data/test.csv",
    "models_dir": "D:/NTU/EE6405/Group Project/BBC_News/code/models",
    "figures_dir": "D:/NTU/EE6405/Group Project/BBC_News/code/figures",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 创建输出目录
os.makedirs(PARAMS["figures_dir"], exist_ok=True)

# 类别名称
CLASS_NAMES = ['business', 'entertainment', 'politics', 'sport', 'tech']

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("使用设备:", PARAMS["device"])

# =========================
# 数据集类（通用）
# =========================
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# 模型定义（需要与训练时一致）
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden1, hidden2, num_classes, dropout1, dropout2, pad_idx=0):
        super().__init__()
        # Embedding层：将词索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # MLP层
        self.fc1 = nn.Linear(embed_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # 平均池化：忽略padding位置
        mask = (x.sum(dim=2) != 0).float().unsqueeze(2)  # (batch, seq_len, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (batch, embed_dim)

        # MLP层
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        return logits

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers, dropout, bidirectional, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                         bidirectional=bidirectional,
                         dropout=dropout if num_layers > 1 else 0,
                         batch_first=True)
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu()
        x = self.dropout(self.embedding(x))
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.rnn(packed_x)
        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        out = self.dropout(hidden)
        logits = self.fc(out)
        return logits

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers, dropout, bidirectional, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu()
        x = self.dropout(self.embedding(x))
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.lstm(packed_x)
        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        out = self.dropout(hidden)
        logits = self.fc(out)
        return logits

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers, dropout, bidirectional, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                         bidirectional=bidirectional,
                         dropout=dropout if num_layers > 1 else 0,
                         batch_first=True)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(gru_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu()
        x = self.dropout(self.embedding(x))
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.gru(packed_x)
        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        out = self.dropout(hidden)
        logits = self.fc(out)
        return logits

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes, num_filters, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embed_dim))
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_outs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pool_outs = [torch.max(out, dim=2)[0] for out in conv_outs]
        out = torch.cat(pool_outs, dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 nhead, num_layers, dim_feedforward, dropout, max_len, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        mask = (x == 0)
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        out = self.transformer(x, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1).float()
        summed = (out * mask_inv).sum(dim=1)
        lengths = mask_inv.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

# =========================
# 加载模型和评估
# =========================
def load_and_evaluate_model(model_name):
    """加载模型并在测试集上评估"""
    print(f"\n{'='*60}")
    print(f"评估 {model_name} 模型")
    print(f"{'='*60}")

    # 加载checkpoint
    ckpt_path = os.path.join(PARAMS["models_dir"], f"{model_name.lower()}_text_cls.pth")
    ckpt = torch.load(ckpt_path, map_location=PARAMS["device"], weights_only=False)
    params = ckpt["params"]

    # 读取测试数据
    test_df = pd.read_csv(PARAMS["test_data_path"])
    X_test_text = test_df.iloc[:, 4].astype(str)
    y_test_text = test_df.iloc[:, 1].astype(str)

    # 标签编码
    le = LabelEncoder()
    le.classes_ = np.array(ckpt["label_classes"])
    y_test = le.transform(y_test_text)

    device = torch.device(PARAMS["device"])

    # 所有模型（包括MLP）都使用Tokenizer + Embedding
    tokenizer = Tokenizer(num_words=params["max_vocab_size"], oov_token="<OOV>")
    tokenizer.word_index = ckpt["tokenizer_word_index"]

    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text),
                               maxlen=params["max_len"], padding='post', truncating='post')

    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(TextDataset(X_test_tensor, y_test_tensor),
                            batch_size=32, shuffle=False)

    vocab_size = min(len(tokenizer.word_index) + 1, params["max_vocab_size"])

    # 根据不同模型创建
    if model_name == "mlp":
        model = MLPClassifier(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            hidden1=params["hidden1"],
            hidden2=params["hidden2"],
            num_classes=len(le.classes_),
            dropout1=params["dropout1"],
            dropout2=params["dropout2"]
        ).to(device)
    elif model_name == "rnn":
        model = RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            num_classes=len(le.classes_),
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"]
        ).to(device)
    elif model_name == "lstm":
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            num_classes=len(le.classes_),
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"]
        ).to(device)
    elif model_name == "gru":
        model = GRUClassifier(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            num_classes=len(le.classes_),
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"]
        ).to(device)
    elif model_name == "cnn":
        model = CNNClassifier(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            num_classes=len(le.classes_),
            kernel_sizes=params["kernel_sizes"],
            num_filters=params["num_filters"],
            dropout=params["dropout"]
        ).to(device)
    elif model_name == "transformer":
        model = TransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            num_classes=len(le.classes_),
            nhead=params["nhead"],
            num_layers=params["num_encoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            max_len=params["max_len"]
        ).to(device)

    # 加载权重
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 评估
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision_per_class': precision_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        'recall_per_class': recall_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        'f1_per_class': f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        'predictions': all_preds,
        'labels': all_labels
    }

    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1']*100:.2f}%")

    return metrics

# =========================
# 可视化函数
# =========================
def plot_confusion_matrix(y_true, y_pred, model_name, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')

    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {model_name.upper()} 混淆矩阵已保存: {save_path}")
    plt.close()

def plot_model_comparison(results, save_path):
    """绘制模型性能对比图 - 4个子图"""
    model_names = ['MLP', 'RNN', 'LSTM', 'GRU', 'CNN', 'Transformer']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # 准备数据
    data = {metric: [results[model.lower()][metric] * 100 for model in model_names]
            for metric in metrics}

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = ['#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1', '#95E1D3', '#A8E6CF']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # 绘制柱状图 - 调整宽度以适应6个模型
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, data[metric], color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1.2, width=0.7)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_ylabel(f'{label} (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=11)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')

        # 添加平均线
        avg_value = np.mean(data[metric])
        ax.axhline(y=avg_value, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Average: {avg_value:.2f}%')
        ax.legend(fontsize=10, loc='lower right')

    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 模型对比图已保存: {save_path}")
    plt.close()

def plot_per_class_metric(results, class_names, metric, metric_label, save_path):
    """绘制单个指标的各类别性能对比"""
    model_names = ['MLP', 'RNN', 'LSTM', 'GRU', 'CNN', 'Transformer']
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_classes)
    width = 0.13  # 6个模型，每个占0.13宽度

    colors = ['#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1', '#95E1D3', '#A8E6CF']

    for i, model_name in enumerate(model_names):
        values = [v * 100 for v in results[model_name.lower()][f'{metric}_per_class']]
        offset = width * (i - 2.5)  # 居中对齐
        bars = ax.bar(x + offset, values, width, label=model_name,
                     color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{metric_label} (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_label} per Class - All Models', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in class_names], fontsize=12)
    ax.legend(fontsize=11, loc='lower right', ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {metric_label} per class 图已保存: {save_path}")
    plt.close()

# =========================
# 主函数
# =========================
def main():
    print("="*60)
    print("BBC新闻文本分类 - 6模型统一评估")
    print("="*60)

    # 评估所有模型
    model_names = ['mlp', 'rnn', 'lstm', 'gru', 'cnn', 'transformer']
    results = {}

    for model_name in model_names:
        results[model_name] = load_and_evaluate_model(model_name)

    # 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)

    # 1. 混淆矩阵（6张）
    print("\n[1/5] 生成混淆矩阵...")
    for model_name in model_names:
        plot_confusion_matrix(
            results[model_name]['labels'],
            results[model_name]['predictions'],
            model_name,
            CLASS_NAMES,
            os.path.join(PARAMS["figures_dir"], f'confusion_matrix_{model_name}.png')
        )

    # 2. 模型性能对比（1张，4子图）
    print("\n[2/5] 生成模型性能对比图...")
    plot_model_comparison(
        results,
        os.path.join(PARAMS["figures_dir"], 'model_performance_comparison.png')
    )

    # 3. Precision per class
    print("\n[3/5] 生成Precision per class图...")
    plot_per_class_metric(
        results, CLASS_NAMES, 'precision', 'Precision',
        os.path.join(PARAMS["figures_dir"], 'precision_per_class.png')
    )

    # 4. Recall per class
    print("\n[4/5] 生成Recall per class图...")
    plot_per_class_metric(
        results, CLASS_NAMES, 'recall', 'Recall',
        os.path.join(PARAMS["figures_dir"], 'recall_per_class.png')
    )

    # 5. F1-Score per class
    print("\n[5/5] 生成F1-Score per class图...")
    plot_per_class_metric(
        results, CLASS_NAMES, 'f1', 'F1-Score',
        os.path.join(PARAMS["figures_dir"], 'f1_per_class.png')
    )

    # 保存评估结果
    eval_results = {}
    for model_name in model_names:
        eval_results[model_name] = {
            'accuracy': results[model_name]['accuracy'],
            'precision': results[model_name]['precision'],
            'recall': results[model_name]['recall'],
            'f1': results[model_name]['f1'],
            'precision_per_class': results[model_name]['precision_per_class'],
            'recall_per_class': results[model_name]['recall_per_class'],
            'f1_per_class': results[model_name]['f1_per_class']
        }

    with open(os.path.join(PARAMS["figures_dir"], 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)

    # 结果汇总
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)
    print(f"{'模型':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    for model_name in model_names:
        result = results[model_name]
        print(f"{model_name.upper():<15} {result['accuracy']*100:>6.2f}%     "
              f"{result['precision']*100:>6.2f}%      "
              f"{result['recall']*100:>6.2f}%     "
              f"{result['f1']*100:>6.2f}%")
    print("="*80)

    print("\n✓ 所有评估完成！")
    print(f"✓ 图表保存路径: {PARAMS['figures_dir']}")
    print("\n生成的图表文件:")
    print("  1. confusion_matrix_[model].png - 混淆矩阵 (6张)")
    print("  2. model_performance_comparison.png - 模型性能对比")
    print("  3. precision_per_class.png - Precision各类别对比")
    print("  4. recall_per_class.png - Recall各类别对比")
    print("  5. f1_per_class.png - F1-Score各类别对比")
    print("  6. evaluation_results.json - 评估结果JSON")

if __name__ == "__main__":
    main()