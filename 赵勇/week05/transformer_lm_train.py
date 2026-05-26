import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# 超参数
# 模型训练迭代总轮数
EPOCHS = 8
# 单次送入模型训练的数据样本数量【调小防卡】
BATCH_SIZE = 16
# 模型输入文本序列固定长度【缩短序列最关键】
SEQ_LEN = 64
# 模型参数更新学习率
LR = 2e-4
# 字符向量嵌入维度大小
EMBED_DIM = 256
# Transformer编码器堆叠层数
NUM_LAYERS = 4
# 多头注意力机制头的数量
NUM_HEADS = 8
# 随机失活防止过拟合概率
DROPOUT = 0.1
# Top-K采样保留最高概率字符数量
TOP_K = 50
# Top-P核采样累积概率阈值
TOP_P = 0.9
# 生成文本温度系数，控制随机性
TEMPERATURE = 0.7
# 设备类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取本地中文训练语料文本
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

# 对语料中所有字符去重并排序
chars = sorted(set(corpus))
# 构建字符映射数字索引字典
char2idx = {c: i for i, c in enumerate(chars)}
# 构建数字索引映射字符字典
idx2char = {i: c for i, c in enumerate(chars)}
# 统计整体字符词汇表总数量
vocab_size = len(chars)
print(f"词汇表大小: {vocab_size}")
print(f"当前运行设备: {device}")


class TextDataset(Dataset):
    """
    文本数据集构建类，用于生成模型训练所需序列样本
    """
    def __init__(self, data, seq_len):
        # 赋值序列长度参数
        self.seq_len = seq_len
        # 赋值数字化后的完整文本数据
        self.data = data

    def __len__(self):
        # 计算数据集可生成的样本总数量，防止负数卡死
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # 根据索引截取输入序列数据
        x = self.data[idx:idx+self.seq_len]
        # 输入序列右移一位作为预测标签数据
        y = self.data[idx+1:idx+self.seq_len+1]
        # 转换为长整型张量返回训练数据与标签
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# 将全部文本字符转换为对应数字索引序列
data = [char2idx[c] for c in corpus]
# 按照9:1比例划分训练集与验证集
split_ratio = 0.9
split_idx = int(len(data) * split_ratio)
train_data = data[:split_idx]
val_data = data[split_idx:]

# 构建训练集数据迭代加载器，关闭打乱提速
train_loader = DataLoader(TextDataset(train_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
# 构建验证集数据迭代加载器
val_loader = DataLoader(TextDataset(val_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)


class MultiHeadAttention(nn.Module):
    """
    手写实现多头自注意力机制
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # 定义注意力头总数
        self.num_heads = num_heads
        # 计算单个注意力头对应的特征维度
        self.head_dim = embed_dim // num_heads
        # 单层线性网络同时生成Q、K、V三个特征矩阵
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # 多头特征拼接后的输出映射层
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # 获取输入数据的批次、序列长度、特征维度
        B, T, C = x.shape
        # 将QKV整体结果拆分为查询、键、值三个独立矩阵
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # 重塑查询矩阵维度，适配多头计算格式
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 重塑键矩阵维度，适配多头计算格式
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 重塑值矩阵维度，适配多头计算格式
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积计算注意力得分，防止维度过高数值溢出
        attn_score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # 判断是否传入掩码矩阵
        if mask is not None:
            # 将掩码无效位置注意力分值置为负无穷
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        # 对注意力得分进行归一化，得到注意力权重
        attn_score = F.softmax(attn_score, dim=-1)
        # 使用注意力权重加权融合值矩阵特征
        out = attn_score @ v
        # 还原数据维度顺序，拼接所有注意力头特征
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        # 通过输出层完成特征映射并返回结果
        return self.out(out)


class EncoderLayer(nn.Module):
    """
    Transformer 编码器基础模块
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # 引入自定义手写多头注意力模块
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        # 第一层特征归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        # 第二层特征归一化层
        self.norm2 = nn.LayerNorm(embed_dim)

        # 构建两层全连接前馈神经网络
        self.ffn = nn.Sequential(
            # 升维线性变换
            nn.Linear(embed_dim, embed_dim * 4),
            # 非线性激活函数
            nn.GELU(),
            # 降维线性变换
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # 定义随机失活层
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, mask=None):
        # 注意力子层计算+残差连接+归一化处理
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        # 前馈网络子层计算+残差连接+归一化处理
        x = self.norm2(x + self.drop(self.ffn(x)))
        # 返回单层编码器计算结果
        return x


class TransformerLM(nn.Module):
    """
    基于Transformer的单向字符级语言模型
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # 字符嵌入层，将数字索引转为向量特征
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 可学习位置编码层，补充文本序列位置信息
        self.pos_emb = nn.Embedding(SEQ_LEN, embed_dim)
        # 堆叠多层Transformer编码器
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        # 输出分类层，映射至词汇表维度
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # 获取输入数据批次与序列长度
        B, T = x.shape
        # 生成序列位置索引数组
        pos = torch.arange(min(T, SEQ_LEN), device=x.device).unsqueeze(0).expand(B, T)
        # 融合字符嵌入特征与位置编码特征
        x = self.embedding(x) + self.pos_emb(pos)

        # 生成下三角因果掩码矩阵
        mask = torch.tril(torch.ones((T, T))).to(x.device)
        # 扩展掩码维度适配多头注意力输入格式
        mask = mask.unsqueeze(0).unsqueeze(1)

        # 逐层完成特征提取运算
        for layer in self.layers:
            x = layer(x, mask)
        # 输出每个位置字符预测分值
        return self.fc(x)


# 实例化整体语言模型并移至设备
model = TransformerLM(vocab_size, EMBED_DIM, NUM_LAYERS, NUM_HEADS).to(device)
# 定义AdamW优化器，加入权重衰减抑制过拟合
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
# 定义多分类任务交叉熵损失函数
criterion = nn.CrossEntropyLoss()


def run_epoch(loader, train=True):
    """
    单轮训练或验证函数
    :param loader: 数据加载器
    :param train: 是否为训练模式
    :return: avg_loss, ppl
    """
    # 初始化累计损失数值
    total_loss = 0.0
    # 初始化统计文本字符总数量
    total_tokens = 0
    # 切换模型训练/评估运行模式
    model.train(train)

    # 根据运行模式开启或关闭梯度计算
    with torch.set_grad_enabled(train):
        # 遍历迭代所有批次数据
        for x, y in loader:
            # 数据移至对应设备
            x, y = x.to(device), y.to(device)
            # 模型前向传播获取预测结果
            logits = model(x)
            # 展平维度计算预测值与真实标签损失值
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            # 判断当前是否为训练模式
            if train:
                # 清空上一轮迭代累计梯度
                optimizer.zero_grad()
                # 反向传播计算网络参数梯度
                loss.backward()
                # 梯度裁剪操作，避免出现梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 优化器更新网络权重参数
                optimizer.step()

            # 累加当前批次整体损失值
            total_loss += loss.item() * y.numel()
            # 累加当前批次字符统计数量
            total_tokens += y.numel()

    # 计算单个字符平均损失值
    avg_loss = total_loss / total_tokens
    # 通过平均损失计算语言模型困惑度PPL
    ppl = math.exp(avg_loss)
    # 返回平均损失与困惑度
    return avg_loss, ppl


# 打印训练日志表头信息
print("\n===== 手写 Transformer 单向语言模型 =====")
print(f"{'Epoch':<5}{'Train Loss':<12}{'Train PPL':<12}{'Val Loss':<12}{'Val PPL':<12}")
print("-" * 55)

# 初始化最优困惑度为无穷大
best_ppl = float('inf')
# 循环执行所有训练轮次
for epoch in range(1, EPOCHS + 1):
    # 执行训练集单轮迭代训练
    tr_loss, tr_ppl = run_epoch(train_loader, train=True)
    # 执行验证集单轮效果评估
    val_loss, val_ppl = run_epoch(val_loader, train=False)

    # 判断当前验证集效果是否为最优
    if val_ppl < best_ppl:
        # 更新最优困惑度数值
        best_ppl = val_ppl
        # 保存当前最优模型权重文件
        torch.save(model.state_dict(), "best_model.pt")

    # 打印当前轮次训练与验证指标数据
    print(f"{epoch:<5}{tr_loss:<12.3f}{tr_ppl:<12.2f}{val_loss:<12.3f}{val_ppl:<12.2f}")

# 打印训练完成提示与最优困惑度
print(f"\n训练完成！最佳验证 PPL: {best_ppl:.2f}")


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1e9):
    """
    联合Top-K与Top-P概率筛选
    :param logits: 模型预测原始分值
    :param top_k: 保留最高概率字符数量
    :param top_p: 核采样累积概率阈值
    :param filter_value: 屏蔽无效字符分值
    :return: 筛选后预测分值
    """
    # 判断是否开启Top-K采样筛选
    if top_k > 0:
        # 提取前K个最高概率预测分值
        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # 屏蔽低于第K位分值的所有预测结果
        logits[logits < top_k_vals[..., -1, None]] = filter_value

    # 判断是否开启Top-P核采样筛选
    if top_p > 0.0:
        # 按照概率从大到小排序预测分值与对应索引
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 计算排序后预测概率的累加值
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 标记超出累积概率阈值的预测位置
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保证至少保留第一个最高概率字符
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # 获取所有需要屏蔽的字符索引
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # 屏蔽超出概率阈值的预测分值
        logits[:, indices_to_remove] = filter_value

    # 返回经过双层筛选后的预测分值
    return logits


def generate(prompt="人工智能", max_len=200):
    """
    文本生成函数，使用温度+Top-K+Top-P联合采样
    :param prompt: 输入提示文本
    :param max_len: 生成最大长度
    :return: 生成文本字符串
    """
    # 加载训练完成后的最优模型权重
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    # 切换模型为推理评估模式
    model.eval()
    # 将输入提示文本转换为数字索引序列
    ids = [char2idx.get(c, 0) for c in prompt]

    # 关闭梯度计算，节省推理运算资源
    with torch.no_grad():
        # 循环逐字生成文本内容
        for _ in range(max_len - len(prompt)):
            # 将已有生成序列转为模型输入张量
            x = torch.tensor([ids], device=device)
            if x.size(1) > SEQ_LEN:
                x = x[:, -SEQ_LEN:]
            # 仅提取最后一个字符位置的预测分值
            logits = model(x)[:, -1, :]
            # 通过温度系数缩放调整概率分布
            logits = logits / TEMPERATURE
            # 同时执行Top-K与Top-P双层候选筛选
            logits = top_k_top_p_filtering(logits, top_k=TOP_K, top_p=TOP_P)
            # 归一化处理得到标准字符预测概率
            probs = F.softmax(logits, dim=-1)
            # 根据概率分布随机采样获取下一个字符索引
            next_id = torch.multinomial(probs, num_samples=1).item()
            # 将新生成字符索引加入序列列表
            ids.append(next_id)

    # 将数字索引序列还原为自然中文文本
    return ''.join(idx2char[i] for i in ids)


# 调用生成函数输出最终文本结果
print("\n【生成结果】")
print(generate())

""" 运行结果
词汇表大小: 1057
当前运行设备: cpu

===== 手写 Transformer 单向语言模型 =====
EpochTrain Loss  Train PPL   Val Loss    Val PPL     
-------------------------------------------------------
1    5.329       206.31      4.591       98.60       
2    4.008       55.02       4.151       63.48       
3    3.231       25.31       3.916       50.22       
4    2.653       14.20       3.881       48.49       
5    2.153       8.61        4.050       57.40       
6    1.698       5.47        4.047       57.20       
7    1.253       3.50        4.254       70.42       
8    0.857       2.36        4.290       72.97       

训练完成！最佳验证 PPL: 48.49

【生成结果】
人工智能、言行一致、实一处一、成共创造美好奇心、不断。
、无坚定力是处理可靠，独立自我、言行一处、达成共创新、温暖安心、达成就感。
成就感是内心，心，自然和友、终获成就、终获成就、温暖人、终身份、温暖安心、终身成就。
民情感是人格力量，从容应对、明辨是人、引领，自、达成就自我、言行一处、达成就自我。
得认可靠心、达成就自我。
公信是内心感是团结情感，自我认知、终达成就、勇于担当、达成就自我、不。
"""