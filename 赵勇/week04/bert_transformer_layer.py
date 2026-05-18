import torch
import torch.nn as nn


# 定义BERT嵌入层类，包含词嵌入、位置嵌入、归一化和Dropout
class BertEmbedding(nn.Module):
    # 初始化函数，定义层结构和参数
    def __init__(self, vocab_size, d_model=768, max_len=512, dropout_rate=0.1):
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 定义词嵌入层：将词汇表索引映射为d_model维向量
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        # 定义位置嵌入层：将位置索引映射为d_model维向量
        self.position_embeddings = nn.Embedding(max_len, d_model)
        # 定义层归一化，稳定模型训练
        self.layer_norm = nn.LayerNorm(d_model)
        # 定义Dropout层，防止模型过拟合
        self.dropout = nn.Dropout(dropout_rate)

    # 前向传播函数，定义数据流向
    def forward(self, input_ids):
        # 从输入张量中获取批次大小和序列长度
        batch_size, seq_len = input_ids.shape

        # 将输入的词ID转换为对应的词嵌入向量
        word_emb = self.word_embeddings(input_ids)
        # 生成从0到序列长度的位置索引张量
        position_ids = torch.arange(0, seq_len, dtype=torch.long)
        # 将位置索引转换为对应的位置嵌入向量
        pos_emb = self.position_embeddings(position_ids)

        # 词嵌入与位置嵌入逐元素相加
        emb = word_emb + pos_emb
        # 对融合后的嵌入进行层归一化
        emb = self.layer_norm(emb)
        # 对嵌入结果应用Dropout
        emb = self.dropout(emb)

        # 打印嵌入层输出张量的形状
        print(f"[Embedding] 输出形状: {emb.shape}")
        # 返回最终的嵌入结果
        return emb


# 手写多头自注意力类
class MultiHeadAttention(nn.Module):
    # 初始化函数，定义注意力相关的线性层和参数
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        # 调用父类初始化
        super().__init__()
        # 保存注意力头的数量
        self.num_heads = num_heads
        # 计算每个注意力头对应的维度
        self.head_dim = d_model // num_heads
        # 保存模型总特征维度
        self.d_model = d_model

        # 定义Q查询向量的线性变换层
        self.wq = nn.Linear(d_model, d_model)
        # 定义K键向量的线性变换层
        self.wk = nn.Linear(d_model, d_model)
        # 定义V值向量的线性变换层
        self.wv = nn.Linear(d_model, d_model)
        # 定义注意力输出的线性投影层
        self.wo = nn.Linear(d_model, d_model)

        # 定义注意力权重的Dropout层
        self.attn_dropout = nn.Dropout(dropout_rate)

    # 前向传播函数，实现多头注意力计算
    def forward(self, x):
        # 获取输入张量的批次大小、序列长度、特征维度
        batch_size, seq_len, d_model = x.shape

        # 线性变换生成Q查询向量
        q = self.wq(x)
        # 线性变换生成K键向量
        k = self.wk(x)
        # 线性变换生成V值向量
        v = self.wv(x)

        # 对Q进行分头并调整维度顺序：[B, L, D] -> [B, H, L, HD]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 对K进行分头并调整维度顺序
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 对V进行分头并调整维度顺序
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算Q和K的相似度得分，并进行缩放
        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        # 对得分进行Softmax，得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # 对注意力权重应用Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 使用注意力权重对V进行加权求和
        out = attn_weights @ v
        # 拼接多头结果，恢复原始形状
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        # 对拼接结果进行线性投影
        out = self.wo(out)

        # 打印多头注意力输出形状
        print(f"[MultiHeadAttention] 输出形状: {out.shape}")
        # 返回注意力计算结果
        return out


# 定义前馈神经网络类
class FeedForward(nn.Module):
    # 初始化函数，定义前馈网络的线性层、激活和Dropout
    def __init__(self, d_model, intermediate_size, dropout_rate=0.1):
        # 调用父类初始化
        super().__init__()
        # 第一个线性层：特征升维
        self.ffn1 = nn.Linear(d_model, intermediate_size)
        # 第二个线性层：特征降维回原始维度
        self.ffn2 = nn.Linear(intermediate_size, d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # GELU激活函数
        self.gelu = nn.GELU()

    # 前向传播函数
    def forward(self, x):
        # 输入经过第一个线性层升维
        x = self.ffn1(x)
        # 经过GELU激活函数
        x = self.gelu(x)
        # 应用Dropout
        x = self.dropout(x)
        # 经过第二个线性层降维
        x = self.ffn2(x)

        # 打印前馈网络输出形状
        print(f"[FeedForward] 输出形状: {x.shape}")
        # 返回前馈计算结果
        return x


# 定义单个Transformer块，包含注意力、残差、归一化、前馈
class TransformerBlock(nn.Module):
    # 初始化函数，组装注意力、前馈、归一化、Dropout
    def __init__(self, d_model=768, num_heads=12, intermediate_size=3072, dropout_rate=0.1):
        # 调用父类初始化
        super().__init__()
        # 实例化多头自注意力模块
        self.attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        # 实例化前馈网络模块
        self.ffn = FeedForward(d_model, intermediate_size, dropout_rate)

        # 注意力模块后的层归一化
        self.norm1 = nn.LayerNorm(d_model)
        # 前馈模块后的层归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 注意力残差的Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        # 前馈残差的Dropout
        self.dropout2 = nn.Dropout(dropout_rate)

    # 前向传播函数，实现完整Transformer块计算
    def forward(self, x):
        # 保存输入作为残差分支
        residual = x
        # 先进行层归一化（Pre-LN结构）
        x = self.norm1(x)
        # 进入多头自注意力计算
        attn_out = self.attn(x)
        # 残差连接 + Dropout
        x = residual + self.dropout1(attn_out)
        # 再次归一化
        x = self.norm1(x)

        # 打印注意力+残差+归一化后的输出形状
        print(f"[Block - 注意力+残差+归一化] 形状: {x.shape}")

        # 保存新的残差分支
        residual = x
        # 进入前馈网络计算
        ffn_out = self.ffn(x)
        # 残差连接 + Dropout
        x = residual + self.dropout2(ffn_out)
        # 最终归一化
        x = self.norm2(x)

        # 打印前馈+残差+归一化后的输出形状
        print(f"[Block - 前馈+残差+归一化] 形状: {x.shape}")
        # 返回Transformer块输出
        return x


# 定义最终语言模型LM
class LM(nn.Module):
    # 初始化函数，组装整个语言模型
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, intermediate_size=3072):
        # 调用父类初始化
        super().__init__()
        # 实例化嵌入层
        self.embed = BertEmbedding(vocab_size, embed_dim)
        # 实例化Transformer块
        self.transformer = TransformerBlock(embed_dim, num_heads, intermediate_size)
        # 定义线性分类头，映射到词汇表大小
        self.fc = nn.Linear(embed_dim, vocab_size)

    # 前向传播函数，定义语言模型整体流程
    def forward(self, x):
        # 打印模型输入形状
        print(f"[LM] 输入形状: {x.shape}")

        # 将输入ID转换为嵌入向量
        e = self.embed(x)

        # 将嵌入向量送入Transformer块
        out = self.transformer(e)

        # 通过线性层得到最终预测分数
        logits = self.fc(out)

        # 打印模型最终输出形状
        print(f"[LM] 最终 logits 形状: {logits.shape}")
        # 返回预测结果
        return logits


if __name__ == "__main__":
    # 词汇表大小
    vocab_size = 30522
    # 词向量/模型维度
    embed_dim = 768
    # 批次大小
    batch_size = 2
    # 序列长度
    seq_len = 20

    # 实例化语言模型
    model = LM(vocab_size, embed_dim)
    # 生成随机词ID作为模型输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 运行模型前向传播，得到输出
    logits = model(x)