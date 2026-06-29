"""
多数据集文本匹配实验工具函数。

这些函数刻意保持为纯 Python 小工具，方便测试，也方便后续把快速实验切换为全量实验。
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


METHOD_LABELS = {
    "biencoder_cosine": "BiEncoder + CosineEmbeddingLoss",
    "biencoder_triplet": "BiEncoder + TripletLoss",
    "crossencoder": "CrossEncoder + CrossEntropyLoss",
}

METHOD_LOG_FILES = {
    "biencoder_cosine": "biencoder_cosine_log.json",
    "biencoder_triplet": "biencoder_triplet_log.json",
    "crossencoder": "crossencoder_log.json",
}

METHOD_CKPT_FILES = {
    "biencoder_cosine": "biencoder_cosine_best.pt",
    "biencoder_triplet": "biencoder_triplet_best.pt",
    "crossencoder": "crossencoder_best.pt",
}


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict]) -> dict:
    labels = Counter(r.get("label") for r in rows)
    lengths = []
    for row in rows:
        if "sentence1" in row:
            lengths.append(len(row["sentence1"]))
        if "sentence2" in row:
            lengths.append(len(row["sentence2"]))
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "total": len(rows),
        "labels": dict(sorted(labels.items(), key=lambda kv: str(kv[0]))),
        "avg_char_len": avg_len,
    }


def summarize_dataset(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    summary = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            summary[split] = summarize_rows(read_jsonl(path))
    return summary


def _sample_label(rows: list[dict], label: int, limit: int, rng: random.Random) -> list[dict]:
    selected = [row for row in rows if row.get("label") == label]
    rng.shuffle(selected)
    return selected[: min(limit, len(selected))]


def prepare_balanced_subset(
    source_dir: str | Path,
    target_dir: str | Path,
    per_label: dict[str, int],
    seed: int = 42,
) -> dict:
    """
    为每个 split 按 label=0/1 分别抽样，生成轻量但类别均衡的实验子集。

    返回目标子集的统计信息。若某个 split 不存在，则跳过。
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for split, limit in per_label.items():
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.exists():
            continue
        rows = read_jsonl(source_path)
        rng = random.Random(f"{seed}:{source_dir.name}:{split}")
        subset = _sample_label(rows, 0, limit, rng) + _sample_label(rows, 1, limit, rng)
        rng.shuffle(subset)
        write_jsonl(target_dir / f"{split}.jsonl", subset)
        stats[split] = summarize_rows(subset)

    return stats


def build_training_command(
    method_key: str,
    bert_path: str,
    data_dir: str | Path,
    epochs: int,
    batch_size: int,
    num_hidden_layers: int,
    project_root: str | Path,
    python_executable: str | None = None,
    max_length_biencoder: int = 64,
    max_length_crossencoder: int = 128,
) -> list[str]:
    """根据方法名生成可直接传给 subprocess 的训练命令。"""
    if method_key not in METHOD_LABELS:
        raise ValueError(f"未知方法: {method_key}")

    common = [
        "--bert_path",
        str(bert_path),
        "--data_dir",
        str(Path(data_dir).resolve()),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--num_hidden_layers",
        str(num_hidden_layers),
    ]

    if method_key.startswith("biencoder"):
        loss = "cosine" if method_key == "biencoder_cosine" else "triplet"
        return [
            python_executable or sys.executable,
            "train_biencoder.py",
            *common,
            "--loss",
            loss,
            "--max_length",
            str(max_length_biencoder),
        ]

    return [
        python_executable or sys.executable,
        "train_crossencoder.py",
        *common,
        "--max_length",
        str(max_length_crossencoder),
    ]


def extract_best_epoch(log_path: str | Path) -> dict:
    records = json.loads(Path(log_path).read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"日志为空: {log_path}")
    best = max(records, key=lambda row: row.get("val_f1", row.get("f1", -1.0)))
    return {
        "epoch": best.get("epoch"),
        "val_acc": best.get("val_acc", best.get("accuracy")),
        "val_f1": best.get("val_f1", best.get("f1")),
        "threshold": best.get("threshold"),
        "elapsed_s": best.get("elapsed_s"),
    }


def archive_method_outputs(
    project_root: str | Path,
    method_key: str,
    destination_dir: str | Path,
    stdout_log: str | Path | None = None,
) -> dict:
    """把固定输出文件归档到单个方法目录，避免下一次训练覆盖。"""
    project_root = Path(project_root)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    log_src = project_root / "outputs" / "logs" / METHOD_LOG_FILES[method_key]
    ckpt_src = project_root / "outputs" / "checkpoints" / METHOD_CKPT_FILES[method_key]

    if log_src.exists():
        log_dst = destination_dir / METHOD_LOG_FILES[method_key]
        shutil.copy2(log_src, log_dst)
        copied["log_path"] = log_dst
    if ckpt_src.exists():
        ckpt_dst = destination_dir / METHOD_CKPT_FILES[method_key]
        shutil.copy2(ckpt_src, ckpt_dst)
        copied["ckpt_path"] = ckpt_dst
    if stdout_log is not None and Path(stdout_log).exists():
        stdout_dst = destination_dir / "train_stdout.log"
        shutil.copy2(stdout_log, stdout_dst)
        copied["stdout_path"] = stdout_dst
    return copied
