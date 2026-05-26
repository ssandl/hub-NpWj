import random

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path


### 训练基于transformer的单向语言模型，并完成文本生成
### 预测下一个字符


# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
MAXLEN = 32
EMBED_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 3
TRAIN_RATIO = 0.8
NUM_HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 128

CHECKPOINT_PATH = Path(__file__).resolve().parent / "transformer_lm.pt"

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
def build_dataset():
    # 获取当前 my_llm.py 所在目录
    current_dir = Path(__file__).resolve().parent
    corpus_path = current_dir / "corpus.txt"
    # 读取文本
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    return corpus

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    # char_to_id: 字符 -> 数字
    char_to_id = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    for ch in data:
        if ch not in char_to_id:
            char_to_id[ch] = len(char_to_id)

    # id_to_char: 数字 -> 字符
    id_to_char = {}
    for ch, idx in char_to_id.items():
        id_to_char[idx] = ch

    return char_to_id, id_to_char

def encode(text, char_to_id):
    ids = []
    for ch in text:
        if ch in char_to_id:
            ids.append(char_to_id[ch])
        else:
            ids.append(char_to_id["<UNK>"])
    return ids


def decode(ids, id_to_char):
    chars = []
    for idx in ids:
        ch = id_to_char.get(int(idx), "<UNK>")
        # 这里先跳过特殊符号，避免生成文本里出现 <PAD>、<UNK>
        if ch in ["<PAD>", "<UNK>"]:
            continue

        chars.append(ch)

    return "".join(chars)

#
# # ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, token_ids, maxlen=MAXLEN):
        self.token_ids = token_ids
        self.maxlen = maxlen

    def __len__(self):
        # 每个样本需要 maxlen + 1 个 token
        return len(self.token_ids) - self.maxlen

    def __getitem__(self, i):
        chunk = self.token_ids[i : i + self.maxlen + 1]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


# # ─── 4. 模型定义 ────────────────────────────────────────────
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(MAXLEN, EMBED_DIM)
        self.transformer_layer = nn.ModuleList([
            MyTransformerEncoderLayer(
                hidden_size=EMBED_DIM,
                num_heads=NUM_HEADS,
                intermediate_size=INTERMEDIATE_SIZE,
            )
            for _ in range(NUM_LAYERS)
        ])
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        for layer in self.transformer_layer:
            x = layer(x)
        logits = self.lm_head(x)
        return logits


# # ─── 5. 训练与评估 ──────────────────────────────────────────
def generate_text(model, prompt, char_to_id, id_to_char, max_new_chars=50, temperature=0.8):
    model.eval()

    ids = encode(prompt, char_to_id)
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_chars):
            # 如果输入长度超过 MAXLEN，只保留最后 MAXLEN 个字符作为上下文
            context = input_ids[:, -MAXLEN:]

            logits = model(context)

            # 只取最后一个位置的输出，用它预测下一个字符
            next_logits = logits[:, -1, :]

            # greedy：取分数最高的字符
            next_id = sample_next_id(next_logits, temperature=temperature)

            # 拼到当前序列后面
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return decode(input_ids[0], id_to_char)


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_input_ids, batch_targets_ids in loader:
            logits = model(batch_input_ids)

            batch_size, seq_len, vvocab_size = logits.shape
            loss = criterion(
                logits.reshape(batch_size * seq_len, vvocab_size),
                batch_targets_ids.reshape(batch_size * seq_len),
            )
            total_loss += loss.item()
    return total_loss / len(loader)




def train():
    corpus_txt = build_dataset()
    char_to_id, id_to_char = build_vocab(corpus_txt)

    token_ids = encode(corpus_txt, char_to_id)
    dataset = TextDataset(token_ids, maxlen=MAXLEN)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerLanguageModel(vocab_size=len(char_to_id))

    total_params = sum(p.numel() for p in model.parameters())
    print("模型参数量:", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("语料长度:", len(corpus_txt))
    print("词表大小:", len(char_to_id))
    print("token 数量:", len(token_ids))
    print("样本数量:", len(dataset))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_input_ids, batch_target_ids in train_loader:
            logits = model(batch_input_ids)

            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(
                logits.reshape(batch_size * seq_len, vocab_size),
                batch_target_ids.reshape(batch_size * seq_len),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    return model, char_to_id, id_to_char

def generate(model, char_to_id, id_to_char):
    print("训练完成，开始生成文本：")

    for prompt in ["黄金", "投资", "市场", "中国证券报"]:
        print("=" * 40)
        print("prompt:", prompt)

        text = generate_text(
            model,
            prompt=prompt,
            char_to_id=char_to_id,
            id_to_char=id_to_char,
            max_new_chars=80,
            temperature=0.8,
        )

        print(text)


def sample_next_id(next_logits, temperature=1.0):
    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    next_logits = next_logits / temperature
    probs = torch.softmax(next_logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)



class MySelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MySelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # dim of head
        self.head_dim = hidden_size // num_heads

        #Q,K,V linear layer
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # output
        self.out_linear = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 经过线性层得到 Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Multi-Head
        # view() 用来改变形状，transpose(1, 2) 是把 num_heads 换到前面去方便计算
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # K.transpose(-1, -2) 是把最后两个维度反转，为了能乘起来
        # 公式: score = (Q @ K^T) / sqrt(d_k)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        # causal mask：禁止当前位置看到未来位置
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # 把分数变成概率 (Softmax)
        # softmax 会让每一行的分数加起来等于 1
        attn_weights = torch.softmax(scores, dim=-1)

        # 用概率给 V 乘上权重 (加权平均)
        # 公式: output = softmax(score) @ V
        attn_output = attn_weights @ V

        # 把多个头拼接回原来的形状
        # 形状变回 [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        # 最后经过一个线性层输出
        output = self.out_linear(attn_output)

        return output


class MyFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(MyFeedForward, self).__init__()
        # 两层全连接，中间通常用 GELU 或 ReLU 激活函数
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        # 自注意力机制
        self.attention = MySelfAttention(hidden_size, num_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        # 前馈神经网络
        self.fnn = MyFeedForward(hidden_size, intermediate_size)
        # 层归一化
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Attention 部分 + 残差连接 + 归一化
        attn_out = self.attention(x)
        # 残差连接: x + attn_out
        x = self.norm1(x + attn_out)

        # FFN 部分 + 残差连接 + 归一化
        fnn_out = self.fnn(x)
        # 残差连接: x + ffn_out
        x = self.norm2(x + fnn_out)

        return x



if __name__ == "__main__":
    # train
    model, char_to_id, id_to_char = train()
    generate(model, char_to_id, id_to_char)
