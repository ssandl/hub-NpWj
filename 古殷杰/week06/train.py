import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from torch.utils.data import DataLoader

# =========================
# 1. 数据
# =========================

dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 128
BATCH_SIZE = 8


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )


dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

train_loader = DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset["test"],
    batch_size=BATCH_SIZE
)

# =========================
# 2. 自定义BERT分类器
# =========================

class BertClassifier(nn.Module):

    def __init__(
        self,
        use_token=True,
        use_position=True,
        use_segment=True
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.use_token = use_token
        self.use_position = use_position
        self.use_segment = use_segment

        hidden = 768

        self.fc = nn.Linear(hidden, 2)

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        batch_size, seq_len = input_ids.shape

        # -------------------------
        # token embedding
        # -------------------------
        if self.use_token:
            token_embeds = \
                self.bert.embeddings.word_embeddings(
                    input_ids
                )
        else:
            token_embeds = torch.zeros(
                batch_size,
                seq_len,
                768,
                device=input_ids.device
            )

        # -------------------------
        # position embedding
        # -------------------------
        if self.use_position:

            position_ids = torch.arange(
                seq_len,
                device=input_ids.device
            ).unsqueeze(0)

            position_embeds = \
                self.bert.embeddings.position_embeddings(
                    position_ids
                )

        else:

            position_embeds = torch.zeros(
                batch_size,
                seq_len,
                768,
                device=input_ids.device
            )

        # -------------------------
        # segment embedding
        # -------------------------
        if self.use_segment:

            token_type_ids = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.long,
                device=input_ids.device
            )

            segment_embeds = \
                self.bert.embeddings.token_type_embeddings(
                    token_type_ids
                )

        else:

            segment_embeds = torch.zeros(
                batch_size,
                seq_len,
                768,
                device=input_ids.device
            )

        # =========================
        # 三类向量相加
        # =========================

        embeddings = (
            token_embeds
            + position_embeds
            + segment_embeds
        )

        embeddings = self.bert.embeddings.LayerNorm(
            embeddings
        )

        embeddings = self.bert.embeddings.dropout(
            embeddings
        )

        # =========================
        # Encoder
        # =========================

        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask[:, None, None, :]
        )

        sequence_output = encoder_outputs.last_hidden_state

        cls_output = sequence_output[:, 0]

        logits = self.fc(cls_output)

        return logits

# =========================
# 3. 训练函数
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, epochs=1):

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids,
                attention_mask
            )

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"loss = {total_loss:.4f}")

# =========================
# 4. 测试函数
# =========================

def evaluate(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    acc = correct / total

    return acc

# =========================
# 5. 四组实验
# =========================

configs = {
    "Full BERT":
        (True, True, True),

    "No Segment":
        (True, True, False),

    "No Position":
        (True, False, True),

    "Only Token":
        (True, False, False),
}

results = {}

for name, config in configs.items():

    print(f"\n======== {name} ========")

    model = BertClassifier(
        use_token=config[0],
        use_position=config[1],
        use_segment=config[2]
    )

    train_model(model)

    acc = evaluate(model)

    results[name] = acc

    print(f"{name} ACC = {acc:.4f}")

# =========================
# 6. 输出结果
# =========================

print("\nFinal Results:")

for k, v in results.items():
    print(f"{k}: {v:.4f}")
