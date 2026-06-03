"""
GPT 语言模型定义（Transformer Decoder-only 架构）

教学重点：
  1. 多头自注意力（Multi-Head Self-Attention）：Q/K/V 投影 + 缩放点积
  2. 因果掩码（Causal Mask）：预训练时每个位置只能看到自己及左侧 token
  3. 前馈网络（FFN）：两层线性 + GELU，参数量约占模型总量的 2/3
  4. 位置编码（Learned Positional Embedding）：可学习 vs 固定 sin/cos
  5. 语言模型头（LM Head）：最后一层映射到 vocab_size，权重与 embedding 共享

默认配置（Mini GPT，~25M 参数，适合 8G 显存）：
  vocab_size=21128, seq_len=256, d_model=384, n_heads=6, n_layers=6, d_ff=1536
"""

import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    因果多头自注意力

    "因果"的含义：位置 i 的输出只依赖位置 0..i 的输入，
    通过在注意力分数上加下三角掩码实现，保证自回归生成时不作弊。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        # 注意力权重和残差连接的 dropout （训练时开启，推理时关闭）防止过拟合，提升泛化能力
        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, self.n_heads, 3 * self.d_head).transpose(1, 2)  # (B, n_heads, T, 3*d_head)
        q, k, v = qkv.chunk(3, dim=-1)  # 各 (B, n_heads, T, d_head)

        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_heads, T, T)

        # 添加因果掩码
        # 因果掩码：上三角（不含对角线）置为 -inf，softmax 后趋近于 0
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))  # (B, n_heads, T, T)

        atten_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)
        atten_weights = self.atten_dropout(atten_weights)

        # 计算注意力输出
        attn_output = torch.matmul(atten_weights, v)  # (B, n_heads, T, d_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.resid_dropout(self.o_proj(attn_output))  # (B, T, C)

class FeedForward(nn.Module):
    """位置独立的前馈网络：Linear → GELU → Linear"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Transformerblock(nn.Module):
    """Transformer 块：包含一个自注意力层和一个前馈网络，均有残差连接和 LayerNorm"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 注意力 + 残差连接
        x = x + self.ffn(self.ln2(x))  # 前馈网络 + 残差连接
        return x
    
class MiniGPT(nn.Module):
    """
    Mini GPT 语言模型：Transformer Decoder-only 架构
    Decoder-only GPT 语言模型
    参数规模（默认配置）：
      Token Embedding:  21128 × 384 = 8.1M
      Position Embed:     256 × 384 = 0.1M
      6 × TransformerBlock:
        Attention QKV+Out: 4 × 384² ≈ 0.6M/层 × 6 = 3.5M
        FFN (384→1536→384): 2 × 384×1536 ≈ 1.2M/层 × 6 = 7.1M
        LayerNorm × 2:      negligible
      LM Head: 共享 Token Embedding 权重，不额外计参数
    总计：~25M 参数
    """
    def __init__(self, vocab_size:int=21128, seq_len:int=256, d_model:int=384, n_heads:int=6, n_layers:int=6, d_ff:int=1536, dropout:float=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Transformerblock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  #    共享 Token Embedding 权重
        self.lm_head.weight = self.token_embed.weight  # 权重共享
        self.apply(self._init_weights)  # 权重初始化

    def _init_weights(self, module:nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x:torch.Tensor):
        """
        input_ids: (B, T)  long tensor，T <= seq_len
        返回 logits: (B, T, vocab_size)
        """
        B,T = x.size()
        token_embeddings = self.token_embed(x)  # (B, T, d_model)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        position_embeddings = self.pos_embed(position_ids)  # (1, T, d_model)
        h = token_embeddings + position_embeddings  # (B, T, d_model)

        for block in self.blocks:
            h = block(h)  # (B, T, d_model)

        h = self.ln_f(h)  # (B, T, d_model)
        logits = self.lm_head(h)  # (B, T, vocab_size)
        return logits
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(config):
    """
    根据配置构建 MiniGPT 模型
    config: dict 包含模型超参数，如 vocab_size, seq_len, d_model, n_heads, n_layers, d_ff, dropout
    返回：MiniGPT 模型实例
    """
    model = MiniGPT(
        vocab_size=config.get('vocab_size', 21128),
        seq_len=config.get('seq_len', 256),
        d_model=config.get('d_model', 384),
        n_heads=config.get('n_heads', 6),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1536),
        dropout=config.get('dropout', 0.1)
    )
    logger.info(f'Model built with {model.count_parameters()/1e6:.2f}M parameters')
    return model 

if __name__ == "__main__":
    # 测试模型构建
    config = {
        'vocab_size': 21128,
        'seq_len': 256,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 6,
        'd_ff': 1536,
        'dropout': 0.1
    }
    model = build_model(config)
    n_params = model.count_parameters()
    logger.info(f'Total parameters: {n_params/1e6:.2f}M')

    # 测试前向传播
    batch_size = 4
    seq_len = config['seq_len']
    dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len), dtype=torch.long)
    dummy_output = model(dummy_input)
    logger.info(f'Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}')
