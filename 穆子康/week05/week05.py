# --------------------------
# 导入需要的库
# --------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --------------------------
# 1. 设置设备与超参数
# --------------------------
# 自动选择用GPU还是CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型与训练超参数（全部写清楚，方便你改）
batch_size = 8          # 每批次训练样本数
max_seq_len = 32         # 句子最大长度（超过截断，不足补0）
d_model = 128            # 词向量/模型特征维度
n_layers = 2             # Decoder层数
n_heads = 2              # 多头注意力头数
dim_feedforward = 256    # 前馈网络中间维度
lr = 1e-3                # 学习率
epochs = 50              # 训练轮数

# --------------------------
# 2. 构建语料与词典
# --------------------------
# 训练用的文本（你可以随便替换）
text_corpus = [
    "我喜欢学习自然语言处理",
    "深度学习非常有趣",
    "Transformer是强大的模型",
    "单向语言模型用于文本生成",
    "今天天气很好适合出门散步",
    "人工智能改变了世界",
    "我爱吃苹果和香蕉",
    "北京是中国的首都",
]

# 构建词汇表（把所有字去重后排序）
vocab = set()
for text in text_corpus:
    vocab.update(list(text))  # 把句子拆成单个字
vocab = sorted(vocab)
vocab_size = len(vocab)  # 词汇表大小

# 字 <-> 编号 互相映射
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# --------------------------
# 3. 构建数据集（用于批量训练）
# --------------------------
class TextDataset(Dataset):
    def __init__(self, texts, word2idx, max_len):
        self.texts = texts
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        """返回数据集总长度"""
        return len(self.texts)

    def __getitem__(self, idx):
        """取单条数据：输入x = 去掉最后一个字；标签y = 去掉第一个字"""
        # 把句子转成字列表
        tokens = list(self.texts[idx])
        # 把字转成编号
        token_ids = [self.word2idx[w] for w in tokens]

        # 语言模型任务：用前n个字预测第n+1个字
        x = token_ids[:-1]  # 输入：[1,2,3,4]
        y = token_ids[1:]   # 标签：[2,3,4,5]

        # 填充到固定长度 max_len
        x = x + [0] * (self.max_len - len(x))
        y = y + [0] * (self.max_len - len(y))

        # 转成张量返回
        return torch.tensor(x[:self.max_len]), torch.tensor(y[:self.max_len])

# 创建数据集和加载器
dataset = TextDataset(text_corpus, word2idx, max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------------
# 4. 位置编码（Transformer必须加顺序信息）
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # 偶数位用sin，奇数位用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为模型缓冲区（不参与训练，但随模型保存）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """把词向量 + 位置编码相加"""
        return x + self.pe[:, :x.size(1)]

# --------------------------
# 5. 单向Transformer语言模型（Decoder-only = GPT结构）
# --------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, dim_ff, n_layers, max_len):
        super().__init__()
        # 词嵌入层：把字编号转成向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 构建Decoder层（单向注意力核心）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,  # 数据形状 [batch, seq_len, dim]
            dropout=0.1
        )
        # 堆叠多层Decoder
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 最后输出层：把向量映射回词汇表（预测下一个字）
        self.fc = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len):
        """
        生成【上三角掩码】，实现单向注意力
        只能看到前面的字，看不到后面的字
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(device)

    def forward(self, x):
        """前向传播"""
        seq_len = x.shape[1]

        # 1. 词嵌入
        x = self.embedding(x)
        # 2. 加入位置编码
        x = self.pos_encoding(x)
        # 3. 生成单向掩码
        causal_mask = self.generate_causal_mask(seq_len)
        # 4. Decoder计算（因为是语言模型，memory=x 自己关注自己）
        out = self.decoder(x, memory=x, tgt_mask=causal_mask)
        # 5. 映射到词汇表概率
        out = self.fc(out)
        return out

# --------------------------
# 6. 初始化模型、损失、优化器
# --------------------------
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    dim_ff=dim_feedforward,
    n_layers=n_layers,
    max_len=max_seq_len
).to(device)

# 损失函数：交叉熵（分类任务）
criterion = nn.CrossEntropyLoss()
# 优化器：Adam
optimizer = optim.Adam(model.parameters(), lr=lr)

# --------------------------
# 7. 开始训练
# --------------------------
print("="*50)
print("开始训练单向Transformer语言模型...")
print("="*50)

model.train()  # 开启训练模式
for epoch in range(epochs):
    total_loss = 0

    for batch_x, batch_y in dataloader:
        # 把数据搬到GPU/CPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 前向传播：模型输出
        outputs = model(batch_x)

        # 把输出和标签展平，计算损失
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            batch_y.reshape(-1)
        )

        # 反向传播 + 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 每10轮打印一次损失
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | 平均损失: {avg_loss:.4f}")

# --------------------------
# 8. 文本生成函数（核心！）
# --------------------------
def generate_text(model, start_text, generate_length=10):
    """
    输入开头文字，自动生成后续文本
    :param start_text: 开头句子
    :param generate_length: 要生成多少个字
    """
    model.eval()  # 开启评估模式
    tokens = list(start_text)  # 把开头转成字列表

    with torch.no_grad():  # 不计算梯度
        for _ in range(generate_length):
            # 把当前字转成模型输入
            input_ids = [word2idx[w] for w in tokens]
            input_tensor = torch.tensor([input_ids]).to(device)

            # 模型预测
            outputs = model(input_tensor)

            # 取最后一个位置的预测结果（下一个字）
            next_token_id = outputs.argmax(-1)[:, -1].item()
            next_token = idx2word[next_token_id]

            # 把新字加入列表，继续循环生成
            tokens.append(next_token)

    return ''.join(tokens)

# --------------------------
# 9. 测试生成效果
# --------------------------
print("\n" + "="*50)
print("模型训练完成！开始文本生成")
print("="*50)

# 生成示例1
start1 = "我"
gen1 = generate_text(model, start1, generate_length=8)
print(f"\n开头：{start1}")
print(f"生成：{gen1}")

# 生成示例2
start2 = "深度"
gen2 = generate_text(model, start2, generate_length=8)
print(f"\n开头：{start2}")
print(f"生成：{gen2}")

# 生成示例3
start3 = "北京"
gen3 = generate_text(model, start3, generate_length=6)
print(f"\n开头：{start3}")
print(f"生成：{gen3}")
