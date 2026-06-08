import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数据。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 应该是一个 JSON list")

    return data


def normalize_example(example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """统一样本字段，支持 ner_tags 或 labels。"""
    if "tokens" not in example:
        raise KeyError("样本缺少 tokens 字段")

    tokens = example["tokens"]
    tags = example.get("ner_tags", example.get("labels", None))

    if tags is None:
        raise KeyError("样本缺少 ner_tags 或 labels 字段")

    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise TypeError("tokens 和 ner_tags/labels 都必须是 list")

    if len(tokens) != len(tags):
        raise ValueError(
            f"tokens 与 tags 长度不一致: len(tokens)={len(tokens)}, len(tags)={len(tags)}"
        )

    return tokens, tags


def build_label_mappings(*splits: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从 train/dev/test 中构建标签映射，保证测试集中出现的标签也可识别。"""
    label_set = set()

    for split in splits:
        for item in split:
            _, tags = normalize_example(item)
            label_set.update(tags)

    # O 放在第一位；其他标签按实体类型和 B/I 排序，便于阅读。
    labels = ["O"] if "O" in label_set else []
    other_labels = sorted([x for x in label_set if x != "O"])
    labels.extend(other_labels)

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def bio_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
    """
    将 BIO 标签转成实体列表。
    这个函数主要用于保存可读预测结果，不参与训练。
    """
    entities = []
    cur_type = None
    cur_tokens = []

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur_type is not None:
                entities.append({"text": "".join(cur_tokens), "type": cur_type})
            cur_type = tag[2:]
            cur_tokens = [token]

        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if cur_type == tag_type:
                cur_tokens.append(token)
            else:
                # 非法 I 标签：为了鲁棒性，按一个新实体开始处理。
                if cur_type is not None:
                    entities.append({"text": "".join(cur_tokens), "type": cur_type})
                cur_type = tag_type
                cur_tokens = [token]

        else:
            if cur_type is not None:
                entities.append({"text": "".join(cur_tokens), "type": cur_type})
            cur_type = None
            cur_tokens = []

    if cur_type is not None:
        entities.append({"text": "".join(cur_tokens), "type": cur_type})

    return entities


def label_to_i(label: str) -> str:
    """把 B-XXX 转成 I-XXX，用于 label_all_tokens 场景。"""
    if label.startswith("B-"):
        return "I-" + label[2:]
    return label


class NerDataset(Dataset):
    """NER 序列标注数据集。"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
        label_all_tokens: bool = False,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        tokens, tags = normalize_example(self.data[index])

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        word_ids = encoding.word_ids()
        labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_id:
                labels.append(self.label2id[tags[word_id]])
            else:
                if self.label_all_tokens:
                    labels.append(self.label2id[label_to_i(tags[word_id])])
                else:
                    labels.append(-100)
            previous_word_id = word_id

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item


def decode_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将模型输出 logits 和 labels 转成 seqeval 需要的 label 序列。
    忽略 label == -100 的位置。
    """
    preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    golds = labels.detach().cpu().tolist()

    pred_list = []
    gold_list = []

    for pred_seq, gold_seq in zip(preds, golds):
        cur_preds = []
        cur_golds = []

        for pred_id, gold_id in zip(pred_seq, gold_seq):
            if gold_id == -100:
                continue
            cur_preds.append(id2label[int(pred_id)])
            cur_golds.append(id2label[int(gold_id)])

        pred_list.append(cur_preds)
        gold_list.append(cur_golds)

    return pred_list, gold_list


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """在验证集或测试集上评估。"""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_golds = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        batch_preds, batch_golds = decode_predictions(logits, batch["labels"], id2label)
        all_preds.extend(batch_preds)
        all_golds.extend(batch_golds)

    avg_loss = total_loss / max(len(dataloader), 1)
    precision = precision_score(all_golds, all_preds)
    recall = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)
    report = classification_report(all_golds, all_preds, digits=4)

    return {
        "loss": round(avg_loss, 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "classification_report": report,
    }


@torch.no_grad()
def predict_samples(
    model,
    raw_data: List[Dict[str, Any]],
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int,
    label_all_tokens: bool,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """保存测试集逐条预测结果。"""
    dataset = NerDataset(
        raw_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
        label_all_tokens=label_all_tokens,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_pred_tags = []

    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        batch_preds, _ = decode_predictions(
            outputs.logits,
            batch["labels"],
            id2label,
        )
        all_pred_tags.extend(batch_preds)

    results = []
    for item, pred_tags in zip(raw_data, all_pred_tags):
        tokens, gold_tags = normalize_example(item)

        # 如果样本被 max_length 截断，预测长度可能短于原始 tokens。
        valid_len = min(len(tokens), len(pred_tags))
        cut_tokens = tokens[:valid_len]
        cut_gold_tags = gold_tags[:valid_len]
        cut_pred_tags = pred_tags[:valid_len]

        results.append(
            {
                "text": "".join(cut_tokens),
                "tokens": cut_tokens,
                "gold_tags": cut_gold_tags,
                "pred_tags": cut_pred_tags,
                "gold_entities": bio_to_entities(cut_tokens, cut_gold_tags),
                "pred_entities": bio_to_entities(cut_tokens, cut_pred_tags),
            }
        )

    return results


def save_json(obj: Any, path: Path) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Any, path: Path) -> None:
    """追加保存 JSONL 日志。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a token-classification NER model."
    )

    parser.add_argument("--data_dir", type=str, required=True, help="包含 train/dev/test.json 的数据目录")
    parser.add_argument("--model_name_or_path", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output_dir", type=str, default="outputs/ner_token_cls")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--label_all_tokens", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="CUDA 可用时启用混合精度训练")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_json(data_dir / "train.json")
    dev_data = load_json(data_dir / "dev.json")
    test_data = load_json(data_dir / "test.json")

    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(dev_data)}")
    print(f"测试集: {len(test_data)}")

    label2id, id2label = build_label_mappings(train_data, dev_data, test_data)
    num_labels = len(label2id)

    print(f"标签数量: {num_labels}")
    print("标签列表:", list(label2id.keys()))

    save_json(label2id, output_dir / "label2id.json")
    save_json({str(k): v for k, v in id2label.items()}, output_dir / "id2label.json")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    if not tokenizer.is_fast:
        raise ValueError("需要 fast tokenizer 才能使用 word_ids() 完成标签对齐")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model.to(device)

    train_dataset = NerDataset(
        train_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
    )
    dev_dataset = NerDataset(
        dev_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
    )
    test_dataset = NerDataset(
        test_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dev_f1 = -1.0
    best_model_dir = output_dir / "best_model"
    train_log_path = output_dir / "train_log.jsonl"

    if train_log_path.exists():
        train_log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        dev_metrics = evaluate(model, dev_loader, id2label, device)

        log_item = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "dev_loss": dev_metrics["loss"],
            "dev_precision": dev_metrics["precision"],
            "dev_recall": dev_metrics["recall"],
            "dev_f1": dev_metrics["f1"],
        }
        append_jsonl(log_item, train_log_path)

        print("\n" + "=" * 70)
        print(f"Epoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(
            f"Dev Precision: {dev_metrics['precision']:.4f} | "
            f"Dev Recall: {dev_metrics['recall']:.4f} | "
            f"Dev F1: {dev_metrics['f1']:.4f}"
        )
        print("=" * 70)

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"保存当前最佳模型到: {best_model_dir}")

    print("\n加载验证集最优模型进行最终测试...")
    best_model = AutoModelForTokenClassification.from_pretrained(
        best_model_dir,
        trust_remote_code=True,
    ).to(device)

    dev_results = evaluate(best_model, dev_loader, id2label, device)
    test_results = evaluate(best_model, test_loader, id2label, device)

    save_json(dev_results, output_dir / "dev_results.json")
    save_json(test_results, output_dir / "test_results.json")

    predictions = predict_samples(
        best_model,
        raw_data=test_data,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        device=device,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
        batch_size=args.batch_size,
    )
    save_json(predictions, output_dir / "test_predictions.json")

    print("\n" + "=" * 70)
    print("最终测试集结果")
    print("=" * 70)
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall:    {test_results['recall']:.4f}")
    print(f"F1:        {test_results['f1']:.4f}")
    print("\n逐类型结果：")
    print(test_results["classification_report"])
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
