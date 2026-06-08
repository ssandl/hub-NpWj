"""
评估 LLM SFT NER 模型（Peoples Daily 数据集）

使用方式：
  python evaluate_sft_pd.py
"""

import os
import json
import argparse
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "peoples_daily"
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地点）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def bio_to_entities(tokens: list, ner_tags: list) -> list:
    """将 token-level BIO 标签转换为实体列表。"""
    entities = []
    current_entity = None
    current_text = []
    
    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append({
                    "text": "".join(current_text),
                    "type": current_entity
                })
            current_entity = tag[2:]
            current_text = [token]
        elif tag.startswith("I-"):
            if current_entity is not None:
                current_text.append(token)
        else:
            if current_entity is not None:
                entities.append({
                    "text": "".join(current_text),
                    "type": current_entity
                })
                current_entity = None
                current_text = []
    
    if current_entity is not None:
        entities.append({
            "text": "".join(current_text),
            "type": current_entity
        })
    
    return entities


def entities_to_bio(text: str, entities: list, tokens: list) -> list:
    """将实体列表转换为 BIO 标签序列。"""
    bio = ["O"] * len(tokens)
    
    for entity in entities:
        entity_text = entity["text"]
        entity_type = entity["type"]
        
        start_idx = 0
        while start_idx < len(text):
            idx = text.find(entity_text, start_idx)
            if idx == -1:
                break
            
            # 检查是否在 token 边界上
            token_start = idx
            token_end = idx + len(entity_text) - 1
            
            if token_start < len(bio):
                bio[token_start] = f"B-{entity_type}"
                for i in range(token_start + 1, min(token_end + 1, len(bio))):
                    bio[i] = f"I-{entity_type}"
            
            start_idx = idx + 1
    
    return bio


def extract_json_from_response(response: str) -> dict:
    """从模型响应中提取 JSON。"""
    # 尝试匹配 JSON
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, response, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"entities": []}
    
    return {"entities": []}


def main():
    parser = argparse.ArgumentParser(description="评估 LLM SFT NER 模型（Peoples Daily）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--adapter_dir", default=str(OUTPUT_DIR / "sft_adapter_pd"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载测试数据
    with open(DATA_DIR / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"测试集大小: {len(test_data)}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    print(f"\n加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # 加载 LoRA adapter
    adapter_dir = Path(args.adapter_dir)
    if adapter_dir.exists() and PEFT_AVAILABLE:
        print(f"加载 LoRA adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    elif adapter_dir.exists():
        print(f"adapter 目录存在但 peft 不可用，尝试直接加载")
    
    model = model.to(device)
    model.eval()

    # 推理并收集结果
    all_preds = []
    all_golds = []
    
    print("\n正在推理...")
    for item in test_data:
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        text = "".join(tokens)
        
        # 构建 prompt
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 生成
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt_text):].strip()
        
        # 解析输出
        result = extract_json_from_response(response)
        pred_entities = result.get("entities", [])
        
        # 转换为 BIO 标签
        pred_bio = entities_to_bio(text, pred_entities, tokens)
        gold_bio = ner_tags
        
        all_preds.append(pred_bio)
        all_golds.append(gold_bio)

    # 计算指标
    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)

    print("\n" + "=" * 70)
    print(f"模型：LLM SFT (Qwen2-0.5B-Instruct) | 数据集：Peoples Daily")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    print(classification_report(all_golds, all_preds, digits=4))

    # 保存结果
    result = {
        "model": "LLM SFT (Qwen2-0.5B-Instruct)",
        "dataset": "Peoples Daily",
        "split": "test",
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
    }
    
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "eval_sft_pd.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存 → {log_path}")


if __name__ == "__main__":
    main()
