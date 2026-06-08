"""NER 工具函数（PeoplesDaily 格式）—— 供 llm_ner / train_sft / evaluate_sft 共享"""

import json
import re

ENTITY_TYPES = ["PER", "ORG", "LOC"]

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地名",
}

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def bio_to_spans(tokens: list[str], ner_tags: list[str]) -> set[tuple[str, str, int, int]]:
    """将 BIO 标注转为 span 集合，格式 (text, type, start, end)。"""
    spans = set()
    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(ner_tags) and ner_tags[i] == f"I-{etype}":
                i += 1
            end = i - 1
            surface = "".join(tokens[start:end + 1])
            spans.add((surface, etype, start, end))
        else:
            i += 1
    return spans


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    """从 peopls_daily 记录中提取 gold spans。"""
    return bio_to_spans(record["tokens"], record["ner_tags"])


def record_to_target(record: dict) -> str:
    """将记录转为 SFT 目标 JSON 字符串。"""
    spans = gold_spans_from_record(record)
    entities = [{"text": s, "type": t} for s, t, _, _ in spans]
    # 按出现顺序去重 (text, type)，与 pred_spans_from_output 的 text.find 行为对齐
    seen = set()
    unique = []
    for ent in entities:
        key = (ent["text"], ent["type"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return json.dumps({"entities": unique}, ensure_ascii=False)


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level 精确率、召回率、F1。"""
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


def pred_spans_from_output(text: str, raw_output: str) -> set[tuple[str, str, int, int]]:
    """从 LLM 生成的 JSON 输出中解析 spans。"""
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return set()
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()
    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()
    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))
    return spans
