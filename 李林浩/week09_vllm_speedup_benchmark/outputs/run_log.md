# Week09 vLLM Benchmark Run Log

## 1. Start service

```bash
bash src/start_vllm_server.sh
```

Service endpoint:

```text
http://127.0.0.1:8000/v1
```

## 2. Health check

```bash
python src/server_health_check.py --base-url http://127.0.0.1:8000/v1
```

Expected output:

```text
[OK] vLLM service is ready.
```

## 3. vLLM benchmark

```bash
python src/benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model week09-qwen \
  --requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --output outputs/vllm_results.json
```

## 4. Transformers baseline

```bash
python src/benchmark_transformers_baseline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 16 \
  --max-tokens 128 \
  --output outputs/transformers_results.json
```

## 5. Sample conclusion

In the sample result, vLLM reaches 45.80 tokens/s while Transformers baseline reaches 12.60 tokens/s. The throughput speedup is 3.63x.
