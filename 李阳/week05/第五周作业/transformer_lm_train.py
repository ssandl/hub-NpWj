"""
作者：深衷浅貌
日期：2026年05月21日--22:03
项目：NLP
文件名：transformer_lm_train.py
"""

"""
训练 Transformer 单向语言模型

用法:
    python transformer_lm_train.py --corpus "*.txt" --epochs 20
"""

import argparse
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob

from transformer_lm_model import TransformerLanguageModel


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    """读取所有txt文件"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """构建词表"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """字符级数据集"""

    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    return total_loss / total_tokens


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # 如果 loader 为空，直接返回无穷大
    if len(loader) == 0:
        return float('inf')

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description="训练 Transformer 语言模型")

    # 数据参数
    parser.add_argument("--corpus", default="*.txt", help="训练语料文件模式")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seq_len", type=int, default=32, help="序列长度")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)

    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 保存参数
    parser.add_argument("--save", default="transformer_lm.pt", help="模型保存路径")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模型参数: hidden_size={args.hidden_size}, layers={args.num_layers}, heads={args.num_heads}")

    # 加载数据
    print("加载语料...")
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件")
    print(f"语料字符数: {len(text):,}")

    # 构建词表
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 划分训练/验证集
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    # 创建数据集
    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"训练样本数: {len(train_ds):,}, 验证样本数: {len(val_ds):,}")

    # 创建模型
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Val PPL':>12}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        val_ppl = math.exp(val_loss)
        scheduler.step()

        marker = "  *" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "model_config": {
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "max_seq_len": args.max_seq_len,
                    "vocab_size": vocab_size,
                }
            }, args.save)

        print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>12.4f}  {val_ppl:>12.2f}{marker}")

    print(f"\n训练完成！最佳模型已保存至 {args.save}")
    print(f"最佳验证损失: {best_val_loss:.4f}, 困惑度: {math.exp(best_val_loss):.2f}")


if __name__ == "__main__":
    main()