# ==============================================
# generate.py ：文本生成脚本（推理专用）
# 功能：加载训练好的模型文件 → 输入开头文字 → 自动续写文本
# 架构：Transformer + 因果掩码 + 自回归生成
# ==============================================

import torch
import torch.nn.functional as F
import math
import torch.nn as nn

# --------------------------- 位置编码模块 ---------------------------
# 作用：给序列加入位置信息，让 Transformer 知道字符的先后顺序
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        # dropout层，防止过拟合
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵（正弦余弦位置编码）
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        # 初始化位置编码
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        self.register_buffer('pe', pe)  # 固定参数，不参与训练

    def forward(self, x):
        # 把位置编码加到词向量上
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --------------------------- Transformer 模型定义 ---------------------------
# 注意：模型结构必须 和 train.py 训练时 完全一样，否则无法加载
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim  # 词向量维度

        # 1. 词嵌入层：把字符索引转为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. 位置编码层
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # 3. Transformer Encoder 层
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,        # 向量维度
            nhead=num_heads,          # 注意力头数
            dim_feedforward=hidden_dim,  # 前馈网络大小
            batch_first=True,         # 形状 [batch, seq, dim]
            dropout=dropout,          # 随机失活
            activation="gelu"         # 激活函数
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)  # 堆叠N层

        # 4. 输出层：把向量映射回字符概率
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        """
        生成因果掩码（下三角mask）
        作用：生成文本时，模型只能看前面的字符，不能看未来的字符
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))  # 掩码位置设为负无穷

    def forward(self, x):
        seq_len = x.size(1)  # 输入序列长度

        # 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.embed_dim)

        # 加入位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # 生成因果mask
        mask = self.generate_causal_mask(seq_len, x.device)

        # Transformer 编码
        x = self.encoder(x, mask=mask)

        # 输出每个位置的下一个字符预测
        return self.fc(x)

# --------------------------- 文本生成核心函数 ---------------------------
def generate_text(model_path="lm_model.pt", prompt="人工智能", max_len=100, temperature=0.5):
    """
    模型文本生成函数
    :param model_path: 训练好的模型文件路径
    :param prompt: 输入的开头文本
    :param max_len: 最大生成多少字符
    :param temperature: 温度系数（越小越保守，越大越随机）
    :return: 生成完成的句子
    """
    # 1. 设置设备：GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型文件（weights_only=False 解决新版本PyTorch报错）
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 3. 从模型文件中读取 词汇表、训练配置
    char2idx = ckpt["char2idx"]    # 字符 → 索引
    idx2char = ckpt["idx2char"]    # 索引 → 字符
    args = ckpt["args"]            # 训练时的超参数

    # 4. 重建模型（结构必须和训练一致）
    model = TransformerLM(
        len(char2idx),
        args.embed_dim,
        args.hidden_dim,
        args.num_layers,
        args.num_heads,
        args.dropout
    ).to(device)

    # 5. 加载训练好的权重
    model.load_state_dict(ckpt["model"])

    # 6. 切换为推理模式（关闭dropout、batchnorm等）
    model.eval()

    # 7. 把输入的开头文字 转为 模型能识别的索引序列
    seq = [char2idx[c] for c in prompt if c in char2idx]

    # 8. 开始自回归生成（循环生成下一个字符）
    with torch.no_grad():  # 推理时不计算梯度
        for _ in range(max_len):
            # 把当前序列送入模型
            x = torch.tensor([seq]).to(device)

            # 模型预测下一个字符的概率分布
            logits = model(x)[:, -1, :] / temperature  # 用温度调节随机性

            # softmax 转为概率
            prob = F.softmax(logits, dim=-1)

            # 根据概率采样下一个字符索引
            idx = torch.multinomial(prob, 1).item()

            # 添加到序列中，继续生成
            seq.append(idx)

            # 如果生成了 。或换行 就停止
            if idx2char[idx] in "。\n":
                break

    # 9. 把索引序列转回文字
    return "".join(idx2char[i] for i in seq)

# --------------------------- 运行生成 ---------------------------
if __name__ == "__main__":
    print("=== Transformer 文本生成 ===")
    # 测试用的开头
    prompts = ["我", "人工智能", "Transformer", "今天", "学习"]

    # 逐个生成
    for p in prompts:
        result = generate_text(prompt=p)
        print(f"\n输入开头：{p}")
        print(f"生成结果：{result}")
