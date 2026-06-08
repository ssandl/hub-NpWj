"""
中文NER（命名实体识别）
BiLSTM序列标注模型

数据格式：

{
    "tokens":["海","钓","比","赛"],
    "ner_tags":["O","O","O","O"]
}

作者：课程作业版
"""

import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# =====================================================
# 1. 读取数据
# =====================================================

with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("validation.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

with open("label_names.json", "r", encoding="utf-8") as f:
    label_names = json.load(f)

print("训练集数量:", len(train_data))
print("验证集数量:", len(val_data))
print("标签:", label_names)


# =====================================================
# 2. 构建词表
# =====================================================
#
# <PAD> 用于补齐
# <UNK> 用于未知字符
#
# 例如：
#
# 海 -> 2
# 钓 -> 3
# 比 -> 4
# 赛 -> 5
#
# =====================================================

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for sample in train_data:

    for token in sample["tokens"]:

        if token not in vocab:
            vocab[token] = len(vocab)

print("词表大小:", len(vocab))


# =====================================================
# 3. 标签映射
# =====================================================
#
# O -> 0
# B-PER -> 1
# I-PER -> 2
# ...
#
# =====================================================

label2id = {
    label: idx
    for idx, label in enumerate(label_names)
}

id2label = {
    idx: label
    for idx, label in enumerate(label_names)
}

print(label2id)


# =====================================================
# 4. Dataset
# =====================================================

MAX_LEN = 128


class NERDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        tokens = sample["tokens"]
        tags = sample["ner_tags"]

        # -------------------------
        # 字符转ID
        # -------------------------
        input_ids = [
            vocab.get(token, 1)
            for token in tokens
        ]

        # -------------------------
        # 标签转ID
        # -------------------------
        labels = [
            label2id[tag]
            for tag in tags
        ]

        # -------------------------
        # 截断
        # -------------------------
        input_ids = input_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]

        # -------------------------
        # Padding
        # -------------------------
        pad_len = MAX_LEN - len(input_ids)

        input_ids += [0] * pad_len

        # -100表示忽略计算loss
        labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }


# =====================================================
# 5. 创建Dataset
# =====================================================

train_dataset = NERDataset(train_data)

val_dataset = NERDataset(val_data)

# =====================================================
# 6. DataLoader
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32
)


# =====================================================
# 7. BiLSTM模型
# =====================================================

class BiLSTMNER(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_labels
    ):
        super().__init__()

        # 字向量
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 分类层
        self.fc = nn.Linear(
            hidden_dim * 2,
            num_labels
        )

    def forward(self, input_ids):

        # (batch,seq_len)
        x = self.embedding(input_ids)

        # (batch,seq_len,embed_dim)
        output, _ = self.lstm(x)

        # (batch,seq_len,hidden*2)
        logits = self.fc(output)

        # (batch,seq_len,num_labels)
        return logits


# =====================================================
# 8. 设备
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device =", device)


# =====================================================
# 9. 创建模型
# =====================================================

model = BiLSTMNER(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=128,
    num_labels=len(label_names)
)

model = model.to(device)


# =====================================================
# 10. Loss
# =====================================================
#
# ignore_index=-100
#
# 忽略PAD位置
#
# =====================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


# =====================================================
# 11. Optimizer
# =====================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# =====================================================
# 12. 训练
# =====================================================

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)

        labels = batch["labels"].to(device)

        # 前向传播
        logits = model(input_ids)

        # logits:
        # (batch,seq_len,num_labels)

        loss = criterion(
            logits.view(
                -1,
                len(label_names)
            ),
            labels.view(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={avg_loss:.4f}"
    )


# =====================================================
# 13. 保存模型
# =====================================================

torch.save(
    model.state_dict(),
    "bilstm_ner.pth"
)

print("模型保存成功")
