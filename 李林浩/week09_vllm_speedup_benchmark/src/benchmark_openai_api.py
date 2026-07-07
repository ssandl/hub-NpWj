"""Benchmark an OpenAI-compatible vLLM chat completion service.

Example:
    python src/benchmark_openai_api.py \
      --base-url http://127.0.0.1:8000/v1 \
      --model week09-qwen \
      --requests 32 \
      --concurrency 8 \
      --max-tokens 128 \
      --output outputs/vllm_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from config import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME, PROMPTS


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * q))
    return float(sorted_values[index])


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            latency = time.perf_counter() - started
            usage = data.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or 0)
            if completion_tokens <= 0:
                # Fallback when usage is unavailable.
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                completion_tokens = max(1, len(content) // 2)
            return {
                "ok": True,
                "latency_s": latency,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "latency_s": time.perf_counter() - started,
                "completion_tokens": 0,
                "error": repr(exc),
            }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=args.timeout, write=30.0, pool=args.timeout)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [send_one(client, url, args.model, p, args.max_tokens, semaphore) for p in prompts]
        results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - started

    ok_results = [item for item in results if item["ok"]]
    latencies = [item["latency_s"] for item in ok_results]
    total_completion_tokens = sum(item["completion_tokens"] for item in ok_results)

    summary = {
        "engine": "vllm_openai_api",
        "model": args.model,
        "base_url": args.base_url,
        "num_requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "success_requests": len(ok_results),
        "success_rate": round(len(ok_results) / max(args.requests, 1), 4),
        "wall_time_s": round(wall_time, 4),
        "total_completion_tokens": total_completion_tokens,
        "throughput_tokens_per_s": round(total_completion_tokens / max(wall_time, 1e-9), 4),
        "requests_per_s": round(len(ok_results) / max(wall_time, 1e-9), 4),
        "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.50), 4),
        "p95_latency_s": round(percentile(latencies, 0.95), 4),
        "errors": [item["error"] for item in results if not item["ok"]][:5],
    }
    return {"summary": summary, "requests": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="outputs/vllm_results.json")
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
