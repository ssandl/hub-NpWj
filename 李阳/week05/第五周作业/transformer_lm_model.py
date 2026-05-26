"""
作者：深衷浅貌
日期：2026年05月21日--22:02
项目：NLP
文件名：transformer_lm_model.py
"""

"""
Transformer 语言模型定义（训练和生成共用）
"""

import math
import torch
import torch.nn as nn


class MultiHeadCausalAttention(nn.Module):
    """
    多头因果自注意力
    特点：只能看到当前位置之前的信息，不能看到未来
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size 必须能被 num_attention_heads 整除"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # Q, K, V 投影（三个独立的线性层，便于理解）
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)

        # 注意力输出投影
        self.attention_output = nn.Linear(hidden_size, hidden_size, bias=True)

        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

        # 缓存因果掩码（causal mask）
        self.register_buffer("causal_mask", None)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """将 [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_size]"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码（下三角矩阵）
        对于位置 i，只能看到 j <= i 的位置
        """
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # 创建下三角矩阵，当前位置可看到自己和之前的位置
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            # 将 mask 中的 0（未来位置）替换为 -inf，1 替换为 0
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, 0.0)
            self.causal_mask = mask
        return self.causal_mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用因果掩码实现单向语言模型

        参数:
            hidden_states: [batch, seq_len, hidden]

        返回:
            attention_output: [batch, seq_len, hidden]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. 线性变换得到 Q, K, V
        q = self.query(hidden_states)  # [batch, seq_len, hidden]
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # 2. 切分为多头
        q = self.transpose_for_scores(q)  # [batch, num_heads, seq_len, head_size]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 3. 计算注意力分数 (QK^T) / sqrt(d_k)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 4. 应用因果掩码（关键！确保未来不可见）
        causal_mask = self.get_causal_mask(seq_len, device)
        attention_scores = attention_scores + causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展到 [1,1,seq,seq]

        # 5. softmax 归一化 + dropout
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # 6. 加权求和得到上下文向量
        context_layer = torch.matmul(attention_probs, v)  # [batch, num_heads, seq_len, head_size]

        # 7. 合并多头: [batch, seq_len, hidden]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)

        # 8. 输出投影
        attention_output = self.attention_output(context_layer)

        return attention_output


class TransformerDecoderBlock(nn.Module):
    """
    单个 Transformer 解码器块（GPT风格）
    包含：
        - 多头因果自注意力（MHA）
        - 残差连接 + LayerNorm
        - 前馈网络（FFN）
        - 残差连接 + LayerNorm
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 intermediate_size: int = None,
                 attention_dropout: float = 0.1,
                 hidden_dropout: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        # ========== 因果自注意力子层 ==========
        self.self_attention = MultiHeadCausalAttention(
            hidden_size, num_attention_heads, attention_dropout
        )
        self.attention_dropout = nn.Dropout(hidden_dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # ========== 前馈网络子层 ==========
        self.intermediate = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.output = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # 激活函数
        self.gelu = nn.GELU()

    def feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前馈网络：线性 -> GELU -> 线性 -> dropout"""
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.gelu(intermediate_output)
        ff_output = self.output(intermediate_output)
        ff_output = self.hidden_dropout(ff_output)
        return ff_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播，残差连接贯穿整个过程

        顺序：自注意力 -> 残差+LN -> FFN -> 残差+LN
        """
        # 多头因果自注意力 + 残差 + LayerNorm
        residual = hidden_states
        attention_output = self.self_attention(hidden_states)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_layer_norm(residual + attention_output)

        # 前馈网络 + 残差 + LayerNorm
        residual = hidden_states
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.ff_layer_norm(residual + ff_output)

        return hidden_states


class TransformerLanguageModel(nn.Module):
    """
    基于 Transformer 的自回归语言模型（GPT风格）

    结构：词嵌入 + 位置嵌入 + N个Transformer解码器块 + 输出层
    """

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 num_attention_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # 位置嵌入层（可学习的位置编码）
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer 解码器块堆叠
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=4 * hidden_size,
                attention_dropout=dropout,
                hidden_dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 最终的 LayerNorm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # 输出层（将隐藏状态映射到词表大小）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 权重共享：词嵌入和输出层共享权重
        self.lm_head.weight = self.token_embedding.weight

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用正态分布初始化
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: [batch, seq_len] 输入字符索引

        返回:
            logits: [batch, seq_len, vocab_size] 输出概率
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. 获取词嵌入 [batch, seq_len, hidden]
        token_embeds = self.token_embedding(input_ids)

        # 2. 获取位置嵌入 [seq_len, hidden] 并扩展到 batch 维度
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)

        # 3. 组合词嵌入和位置嵌入
        hidden_states = self.dropout(token_embeds + pos_embeds)

        # 4. 通过 N 个 Transformer 块
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # 5. 最后的 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        # 6. 输出层，得到 logits
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]

        return logits

    def generate(self,
                 prompt: str,
                 char2idx: dict,
                 idx2char: dict,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = None,
                 device: str = "cpu") -> str:
        """
        自回归生成文本

        参数:
            prompt: 起始提示词
            char2idx: 字符到索引的映射
            idx2char: 索引到字符的映射
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（>1 更随机，<1 更确定）
            top_k: 只从概率最高的 k 个 token 中采样
            device: 设备

        返回:
            生成的完整文本
        """
        self.eval()

        # 将 prompt 转换为索引
        input_ids = [char2idx.get(c, 0) for c in prompt]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = prompt

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 只使用最近的 max_seq_len 个 token
                if input_ids.size(1) > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]

                # 前向传播
                logits = self(input_ids)  # [1, seq_len, vocab_size]

                # 只取最后一个位置的 logits
                next_token_logits = logits[0, -1, :] / temperature

                # 应用 top_k 采样
                if top_k is not None:
                    # 只保留概率最高的 k 个 token
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    mask.scatter_(0, top_k_indices, top_k_values)
                    next_token_logits = mask

                # 计算概率并采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # 转换为字符
                next_char = idx2char.get(next_token, '')
                generated += next_char

                # 更新输入
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

                # 遇到换行符或句号可以提前终止（可选）
                if next_char in '\n':
                    break

        return generated