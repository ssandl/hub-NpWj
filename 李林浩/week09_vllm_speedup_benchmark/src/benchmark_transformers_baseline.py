"""Sequential local Transformers baseline benchmark.

This script is intentionally simple: it runs local generation requests one by one,
which is used as the baseline for comparing vLLM service throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEFAULT_HF_MODEL_NAME, PROMPTS


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * q))
    return float(sorted_values[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_HF_MODEL_NAME)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="outputs/transformers_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    request_results: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    for i in range(args.requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        started = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        completion_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
        request_results.append(
            {
                "ok": True,
                "latency_s": latency,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        )

    wall_time = time.perf_counter() - wall_started
    latencies = [item["latency_s"] for item in request_results]
    total_completion_tokens = sum(item["completion_tokens"] for item in request_results)
    summary = {
        "engine": "transformers_sequential_baseline",
        "model": args.model,
        "device": device,
        "num_requests": args.requests,
        "concurrency": 1,
        "max_tokens": args.max_tokens,
        "success_requests": len(request_results),
        "success_rate": 1.0,
        "wall_time_s": round(wall_time, 4),
        "total_completion_tokens": total_completion_tokens,
        "throughput_tokens_per_s": round(total_completion_tokens / max(wall_time, 1e-9), 4),
        "requests_per_s": round(len(request_results) / max(wall_time, 1e-9), 4),
        "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.50), 4),
        "p95_latency_s": round(percentile(latencies, 0.95), 4),
        "errors": [],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "requests": request_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
