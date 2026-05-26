# ==============================================
# train.py ：模型训练脚本
# 功能：加载语料 → 构建词汇表 → 定义Transformer模型 → 训练 → 保存最优模型
# 包含：损失、PPL困惑度、验证集、最优模型保存
# ==============================================

import math
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- 1. 数据加载工具 ----------------------
def load_corpus(pattern="corpus.txt"):
    """加载语料文本文件"""
    with open(pattern, encoding="utf-8") as f:
        return f.read()

def build_vocab(text):
    """
    构建字符级词汇表
    输入：文本字符串
    输出：字符→索引、索引→字符 的字典
    """
    chars = sorted(set(text))
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for c,i in char2idx.items()}
    return char2idx, idx2char

# ---------------------- 2. 数据集定义 ----------------------
class CharDataset(Dataset):
    """自定义数据集，用于生成模型训练样本"""
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len  # 输入序列长度
        # 将文本字符转为索引
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        """数据集长度：总字符数 - 序列长度"""
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        """获取一条样本：输入x，目标y（下一个字符）"""
        x = self.data[idx: idx+self.seq_len]
        y = self.data[idx+1: idx+self.seq_len+1]
        return x, y

# ---------------------- 3. Transformer 位置编码 ----------------------
class PositionalEncoding(nn.Module):
    """位置编码：给序列加入位置信息"""
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 生成位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim,2) * (-math.log(10000.0)/embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """将位置编码加到词向量上"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ---------------------- 4. Transformer 模型定义 ----------------------
class TransformerLM(nn.Module):
    """Transformer 语言模型（因果掩码，自回归）"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        # Transformer Encoder 层
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            batch_first=True, dropout=dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        # 输出层：映射到词汇表
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        """
        生成因果掩码（下三角掩码）
        作用：防止模型看到未来的字符
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, x):
        """前向传播"""
        seq_len = x.size(1)
        # 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        # 加入位置编码
        x = self.pos_encoder(x.transpose(0,1)).transpose(0,1)
        # 因果掩码
        mask = self.generate_causal_mask(seq_len, x.device)
        # Transformer 编码
        x = self.encoder(x, mask=mask)
        # 输出预测
        return self.fc(x)

# ---------------------- 5. 训练/评估一轮 ----------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    训练 or 评估 一轮（一个epoch）
    返回：平均损失、PPL困惑度
    """
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        # 计算损失
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        return 999.0, 1e9

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)  # 困惑度 PPL
    return avg_loss, ppl

# ---------------------- 6. 主训练函数 ----------------------
def main():
    # 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--seq_len",    type=int,   default=32)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--num_heads",  type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--corpus",     default="corpus.txt")
    parser.add_argument("--save",       default="lm_model.pt")
    args = parser.parse_args()

    # 设备：GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载语料
    text = load_corpus(args.corpus)
    # 构建词汇表
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)

    # 训练集 & 验证集
    train_text = text
    val_text = text[:500]

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 初始化模型
    model = TransformerLM(
        vocab_size, args.embed_dim, args.hidden_dim,
        args.num_layers, args.num_heads, args.dropout
    ).to(device)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ppl = float('inf')

    # 打印信息
    print(f"设备: {device} | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>6}{'TrainLoss':>12}{'TrainPPL':>12}{'ValLoss':>12}{'ValPPL':>12}")
    print("-"*60)

    # 开始训练
    for epoch in range(1, args.epochs+1):
        # 训练一轮
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        # 验证一轮
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # 保存最优模型
        mark = " *" if va_ppl < best_ppl else ""
        if va_ppl < best_ppl:
            best_ppl = va_ppl
            torch.save({
                "model": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": args
            }, args.save)

        # 打印日志
        print(f"{epoch:6d}{tr_loss:12.4f}{tr_ppl:12.2f}{va_loss:12.4f}{va_ppl:12.2f}{mark}")

    print(f"\n训练完成！最优 PPL: {best_ppl:.2f} | 模型已保存")

if __name__ == "__main__":
    main()
