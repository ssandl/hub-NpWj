"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        # self.seq_len: 每个样本的序列长度,同时也是截取原始文本的窗口大小
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 长度不足 seq_len 时, 不返回样本, 所以返回的样本数为 len(self.data) - self.seq_len + 1
        return max(0, len(self.data) - self.seq_len + 1)

    def __getitem__(self, idx):
        # x: 每个样本的输入序列, 即当前窗口内的字符索引
        x = self.data[idx: idx + self.seq_len]
        # y: 每个样本的标签, 即下一个字符的索引
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class RnnLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = TransformerEncoder(hidden_dim, num_layers, dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mask=None):
        e = self.drop(self.embed(x))
        out = self.transformer(e, mask)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits

# 自己实现的 Transformer 模型语言模型，实现到Decoder层和FeedForward层
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)   # 一次性算 Q K V
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        # [B, T, 3H] -> 3 个 [B, n_head, T, d_k]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # scaled dot-product
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = attn @ v                              # [B, n_head, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden, n_head, intermediate_dim):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))        # 残差 + LN
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden=768, n_layer=12, n_head=12, intermediate_dim=3072):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, intermediate_dim) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # 如果是Transformer模型，需要传递mask
        if isinstance(model, TransformerLM):
            # mask的大小是序列长度*序列长度，为下三角为1，其他为0
            # 1表示有效位置，0表示填充位置
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(device)
            logits = model(x, mask)
        else:
            logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="lstm", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    # drop_last=True 表示丢弃最后一个不完整批次
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    if args.model in ["rnn", "lstm"]:
        model = RnnLM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"未知模型名称: {args.model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 推理
    model.load_state_dict(torch.load(args.save)["model_state"])
    model.eval()
    print(generate_text(model, device, char2idx, idx2char, "你好"))


# 推理函数，实现生成文本的功能
def generate_text(model, device, char2idx, idx2char, start_text, max_len=100):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([char2idx[c] for c in start_text]).to(device)
        output = model(input_ids)
        for _ in range(max_len - len(start_text)):
            output = model(input_ids)
            output = output[:, -1, :]
            topv, topi = output.topk(1, dim=0)
            topi = topi.item()
            input_ids = torch.cat([input_ids, torch.tensor([topi]).to(device)], dim=0)
        return "".join([idx2char[i] for i in input_ids.tolist()])

if __name__ == "__main__":
    main()