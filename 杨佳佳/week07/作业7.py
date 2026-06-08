# -*- coding:utf-8 -*-
# 第七周作业：BiLSTM 人民日报NER序列标注（peoples_daily数据集）
# 任务：识别人名PER、地名LOC、机构ORG，BIO序列标注
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========1 加载本地人民日报数据集=========
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("validation.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
with open("label_names.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)

print(f"训练集条数：{len(train_data)}，验证集条数：{len(val_data)}")
print("全部标签：", label_list)

# ==========2 构建字符词表、标签映射=========
vocab = {"<PAD>": 0, "<UNK>": 1}
# 遍历训练集所有字符生成词典
for sent in train_data:
    for char in sent["tokens"]:
        if char not in vocab:
            vocab[char] = len(vocab)

label2id = {lab: idx for idx, lab in enumerate(label_list)}
id2label = {idx: lab for idx, lab in enumerate(label_list)}
MAX_LEN = 128

# ==========3 自定义数据集=========
class PeopleNERDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"][:MAX_LEN]
        tags = item["ner_tags"][:MAX_LEN]
        input_ids = [vocab.get(c, vocab["<UNK>"]) for c in tokens]
        label_ids = [label2id[t] for t in tags]
        pad_len = MAX_LEN - len(input_ids)
        input_ids += [vocab["<PAD>"]] * pad_len
        label_ids += [-100] * pad_len
        return torch.tensor(input_ids), torch.tensor(label_ids)

train_set = PeopleNERDataset(train_data)
val_set = PeopleNERDataset(val_data)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# ==========4 BiLSTM模型定义=========
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, out_label_num):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, out_label_num)
    def forward(self, x):
        emb_out = self.embedding(x)
        lstm_out, _ = self.bilstm(emb_out)
        out = self.linear(lstm_out)
        return out

# ==========5 训练初始化=========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_NER(len(vocab), embed_dim=128, hidden_dim=128, out_label_num=len(label_list)).to(device)
loss_func = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_num = 5

# ==========6 模型训练=========
for epoch in range(epoch_num):
    model.train()
    total_loss = 0.0
    for inp, lab in train_loader:
        inp, lab = inp.to(device), lab.to(device)
        pred = model(inp)
        loss = loss_func(pred.reshape(-1, len(label_list)), lab.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"第{epoch+1}轮 | 训练平均损失：{avg_loss:.4f}")

# 保存训练模型
torch.save(model.state_dict(), "bilstm_people_ner.pth")
print("模型已保存：bilstm_people_ner.pth")
