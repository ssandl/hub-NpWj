"""
作者：深衷浅貌
日期：2026年05月13日--22:00
项目：NLP
文件名：single_transformer
"""
# class DiyTransformer:
#     def __init__(self, hidden_size, batch_size, num_attention_heads):
#
# hidden_sizae = 768  # 词向量维度
# words_num = 4   # 句子包含单词数量
# num_attention_heads = 4     # 注意力头数
# x = torch.randn(words_num, hidden_sizae)
#
# attention_head_size = hidden_sizae/num_attention_heads  # 单头注意力长度
#
#
#
# def single_transformer(x):

import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    """
    单个 Transformer 编码器层，逻辑与 diy_bert.py 中的 single_transformer_layer_forward 完全一致。
    包含：
        - 多头自注意力（MHA）（Q/K/V 投影 + 注意力计算 + 输出投影）
        - 残差连接 + LayerNorm
        - 前馈网络（FFN）（两个线性层 + GELU）
        - 残差连接 + LayerNorm
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 intermediate_size: int = None,
                 attention_dropout: float = 0.0,
                 hidden_dropout: float = 0.0,
                 layer_norm_eps: float = 1e-12):
        """
        参数:
            hidden_size: 隐藏层维度（例如 BERT-base 为 768）
            num_attention_heads: 多头注意力头数（例如 12）
            intermediate_size: FFN 中间层维度，默认为 4 * hidden_size（BERT 标准）
            attention_dropout: 注意力权重矩阵的 Dropout 概率
            hidden_dropout: FFN 输出和注意力输出后的 Dropout 概率
            layer_norm_eps: LayerNorm 的 epsilon
        """
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads   # 单注意力头长度
        assert hidden_size % num_attention_heads == 0, "hidden_size 必须能被 num_attention_heads 整除"

        # ========== 多头自注意力子层 ==========
        # Q, K, V 投影（无 bias？原代码有 bias，这里保持有 bias）
        # 投影直接使用PyTorch 提供的全连接层（线性变换）模块，nn.Linear
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)

        # 注意力输出投影
        self.attention_output = nn.Linear(hidden_size, hidden_size, bias=True)

        # 注意力部分的 Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

        # LayerNorm (先残差后 Norm，所以放在 MHA 和 FFN 之后)
        # 残差连接层：上层输入输出相加再归一化（LayerNorm）
        # 归一化，直接使用PyTorch 提供的归一化模块，nn.LayerNorm
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # ========== 前馈网络子层 ==========
        # 两层线性层，加上GELU激活
        self.intermediate = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.output = nn.Linear(intermediate_size, hidden_size, bias=True)

        # FFN 部分的 Dropout
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        self.ff_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # 激活函数使用 GELU（与原代码 gelu 函数一致）
        """
        激活函数的作用是引入非线性。如果没有激活函数，多层线性变换最终等价于一次线性变换，无法拟合复杂的数据分布。
        GELU 的特性：
        平滑性：GELU 在整个实数域上光滑可导（ReLU 在 0 点不可导），利于梯度传播。
        随机正则化效应：GELU 可视为 ReLU、Dropout 的组合变体（以输入值的大小决定“激活”概率），在 Transformer 模型（BERT、GPT 等）中普遍使用，往往比 ReLU 表现更好。
        公式：GELU(x) = x · Φ(x)，其中 Φ(x) 是标准正态分布的累积分布函数。相比 ReLU，负数区域不是直接截断为零，而是有一个平滑的过渡。
        """
        self.gelu = nn.GELU()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        这里是代码实现的难点，需要慢慢消化
        将 [batch_size, seq_len, hidden_size] 转换为 [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        # 交换维度，得到 [batch_size, num_heads, seq_len, head_size]
        return x.permute(0, 2, 1, 3)

    def self_attention(self,
                       hidden_states: torch.Tensor,
                       attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        自注意力核心计算，完全模仿 diy_bert.py 中的 self_attention 方法
        torch.matmul：矩阵乘法
        合并多头是代码实现难点：context_layer.permute， context_layer.view

        """
        # 1. 线性变换得到 Q, K, V
        q = self.query(hidden_states)   # [batch, seq_len, hidden]
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # 2. 切分为多头
        q = self.transpose_for_scores(q)   # [batch, num_heads, seq_len, head_size]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 3. 计算注意力分数 (QK^T) / sqrt(d_k)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))   # [batch, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 4. 应用注意力 mask（可选，原代码无 mask，此处预留）
        if attention_mask is not None:
            # attention_mask 形状通常为 [batch, 1, 1, seq_len] 或 [batch, 1, seq_len, seq_len]
            attention_scores = attention_scores + attention_mask

        # 5. softmax 归一化 + dropout
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # 6. 加权求和得到上下文向量
        context_layer = torch.matmul(attention_probs, v)   # [batch, num_heads, seq_len, head_size]

        # 7. 合并多头： [batch, seq_len, num_heads, head_size] -> [batch, seq_len, hidden]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.hidden_size)

        # 8. 输出投影
        attention_output = self.attention_output(context_layer)

        return attention_output

    def feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前馈网络：线性 + GELU + 线性"""
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.gelu(intermediate_output)
        ff_output = self.output(intermediate_output)
        ff_output = self.hidden_dropout(ff_output)
        return ff_output

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播，完全按照 diy_bert.py 的顺序：
            attention_output = self_attention(x)
            x = LayerNorm(x + attention_output)
            ff_output = feed_forward(x)
            x = LayerNorm(x + ff_output)
        """
        # 保存残差连接用的输入
        residual = hidden_states

        # 多头自注意力
        attention_output = self.self_attention(hidden_states, attention_mask)

        # 第一次残差 + LayerNorm
        hidden_states = self.attention_layer_norm(residual + attention_output)

        # 前馈网络
        residual = hidden_states
        ff_output = self.feed_forward(hidden_states)

        # 第二次残差 + LayerNorm
        hidden_states = self.ff_layer_norm(residual + ff_output)

        return hidden_states

# 参数与 BERT-base 一致
hidden_size = 768
num_heads = 12
intermediate_size = 3072

# 创建 Transformer 层
transformer_layer = TransformerBlock(
    hidden_size=hidden_size,
    num_attention_heads=num_heads,
    intermediate_size=intermediate_size,
    attention_dropout=0.1,
    hidden_dropout=0.1
)

# 模拟输入 [batch_size=2, seq_len=4, hidden_size=768]
batch_size, seq_len = 2, 4
x = torch.randn(batch_size, seq_len, hidden_size)

# 前向传播
out = transformer_layer(x)
print(out.shape)  # torch.Size([2, 4, 768])

