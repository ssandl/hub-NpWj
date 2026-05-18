import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    参数:
        Q: [batch_size, n_heads, seq_len, d_k]
        K: [batch_size, n_heads, seq_len, d_k]
        V: [batch_size, n_heads, seq_len, d_v]
        mask: [batch_size, 1, seq_len, seq_len] 或 [batch_size, n_heads, seq_len, seq_len]
    """
    d_k = Q.size(-1)
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. 计算softmax得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attn_weights = dropout(attn_weights)
        
    # 4. 注意力权重与 V 相乘得到输出
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V 的线性映射层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # 输出线性映射层
        self.fc = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性映射并拆分为多头
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        
        # 3. 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 过最后的线性层
        output = self.fc(attn_output)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        # 1. 多头自注意力 + 残差 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. 前馈神经网络 + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. 带掩码的自注意力 + 残差 + Norm
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. 交叉注意力 (Q=x, K=enc_output, V=enc_output) + 残差 + Norm
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_output))
        
        # 3. 前馈神经网络 + 残差 + Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x

if __name__ == "__main__":
    # 定义超参数
    batch_size = 2
    seq_len = 5
    d_model = 512
    n_heads = 8
    d_ff = 2048
    
    # 1. 测试 Encoder Layer
    print("--- Testing Encoder Layer ---")
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
    
    # 模拟输入 [batch_size, seq_len, d_model]
    enc_input = torch.randn(batch_size, seq_len, d_model)
    
    # Encoder不需要前瞻掩码，通常只需要Padding Mask，这里为了简单先用None
    enc_output = encoder_layer(enc_input, src_mask=None)
    print(f"Encoder Input Shape:  {enc_input.shape}")
    print(f"Encoder Output Shape: {enc_output.shape}")
    
    # 2. 测试 Decoder Layer
    print("\n--- Testing Decoder Layer ---")
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
    
    # 模拟Decoder输入和Encoder的最终输出
    dec_input = torch.randn(batch_size, seq_len, d_model)
    
    # 生成一个下三角的掩码用于Decoder的自注意力 [1, 1, seq_len, seq_len]
    # 这样可以防止当前词看到未来的词
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    dec_output = decoder_layer(dec_input, enc_output, src_mask=None, tgt_mask=tgt_mask)
    print(f"Decoder Input Shape:  {dec_input.shape}")
    print(f"Decoder Output Shape: {dec_output.shape}")
