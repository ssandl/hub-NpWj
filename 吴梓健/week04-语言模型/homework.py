import torch
import torch.nn as nn



class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = MyTransformerBlock(embed_dim, num_heads=12, intermediate_size=3072)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        e = self.embed(x)
        out = self.transformer(e)
        logits = self.fc(out)
        return logits


class MyTransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4,intermediate_size=3072):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.ffn1 = nn.Linear(embed_dim, intermediate_size)
        self.ffn2 = nn.Linear(intermediate_size, embed_dim)

    def multihead_self_attention(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        # 线性投影
        q = self.wq(x)  # (batch_size, seq_len, embed_dim)
        k = self.wk(x)  # (batch_size, seq_len, embed_dim)
        v = self.wv(x)  # (batch_size, seq_len, embed_dim)
        # 分头
        head_dim = embed_dim // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        attn_scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)          # (batch_size, num_heads, seq_len, seq_len)
        out = attn_weights @ v                            # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        # 多头拼接完成后，再通过一层线性层
        out = self.wo(out)  # (batch_size, seq_len, embed_dim)
        return out

    def feed_forward(self, x):
        x = self.ffn1(x)
        # 激活函数为GELU
        x = torch.nn.functional.gelu(x)
        x = self.ffn2(x)
        return x

    def forward(self, x):
        res1 = self.multihead_self_attention(x)
        print("经过multihead self attention的形状:", res1.shape)  # 应该是 (batch_size, seq_len, embed_dim)
        # add & norm
        x = x + res1
        x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        res2 = self.feed_forward(x)
        # add & norm
        x = x + res2
        x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        return x
    
if __name__ == "__main__":
    # 测试模型
    vocab_size = 30522
    embed_dim = 768
    batch_size = 2
    seq_len = 512

    model = LM(vocab_size, embed_dim)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print("输入x的形状:", x.shape)  # 应该是 (batch_size, seq_len)
    logits = model(x)
    print("logits的形状:", logits.shape)  # 应该是 (batch_size, seq_len, vocab_size)