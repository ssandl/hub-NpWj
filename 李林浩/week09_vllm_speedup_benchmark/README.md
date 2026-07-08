# Week09 作业：vLLM 大模型服务部署与速度提升验证

## 1. 作业目标

本项目对应第九周作业要求：**尝试部署一个 vLLM 大模型服务，验证速度提升**。

项目以 OpenAI-Compatible API 方式启动 vLLM 服务，并通过同一批提示词对比两类推理方式：

- **Transformers 本地逐条推理**：作为传统基线方案。
- **vLLM 服务化批量推理**：使用 vLLM 的 PagedAttention 和连续批处理能力，提高吞吐量。

最终输出吞吐量、平均延迟、P95 延迟、请求成功率等指标，并生成对比图。

---

## 2. 项目结构

```text
week09_vllm_speedup_benchmark/
├── README.md
├── requirements.txt
├── report.md
├── src/
│   ├── start_vllm_server.sh
│   ├── server_health_check.py
│   ├── chat_client.py
│   ├── benchmark_openai_api.py
│   ├── benchmark_transformers_baseline.py
│   ├── plot_results.py
│   └── config.py
└── outputs/
    ├── benchmark_results.json
    ├── benchmark_summary.csv
    ├── run_log.md
    └── speedup_comparison.png
```

---

## 3. 环境准备

建议使用 Python 3.10+，并在具备 NVIDIA GPU/CUDA 的 Linux 环境中运行。

```bash
conda create -n week09-vllm python=3.10 -y
conda activate week09-vllm
pip install -r requirements.txt
```

> 说明：vLLM 通常需要 GPU 环境。如果本地没有 GPU，可以只阅读代码与样例输出，或将服务部署到云端 GPU 机器上执行。

---

## 4. 启动 vLLM 服务

默认模型使用体积较小的 `Qwen/Qwen2.5-0.5B-Instruct`，便于课程作业演示。也可以通过环境变量替换为其他模型。

```bash
cd src
bash start_vllm_server.sh
```

脚本默认参数如下：

```bash
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
SERVED_MODEL_NAME=week09-qwen
HOST=0.0.0.0
PORT=8000
```

启动成功后，服务地址为：

```text
http://127.0.0.1:8000/v1
```

---

## 5. 健康检查

```bash
python src/server_health_check.py --base-url http://127.0.0.1:8000/v1
```

如果返回模型列表，说明服务可用。

---

## 6. 单次调用验证

```bash
python src/chat_client.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model week09-qwen \
  --prompt "请用三句话解释 vLLM 为什么能提升大模型推理吞吐量。"
```

---

## 7. vLLM 服务吞吐量测试

```bash
python src/benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model week09-qwen \
  --requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --output outputs/vllm_results.json
```

---

## 8. Transformers 基线测试

```bash
python src/benchmark_transformers_baseline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 16 \
  --max-tokens 128 \
  --output outputs/transformers_results.json
```

---

## 9. 生成对比图

```bash
python src/plot_results.py \
  --input outputs/benchmark_results.json \
  --output outputs/speedup_comparison.png
```

---

## 10. 样例结论

本项目 `outputs/` 目录中已经放置一组课程作业样例结果，用于说明输出格式。样例中：

| 推理方式 | 输出吞吐量 tokens/s | 平均延迟 s | P95 延迟 s |
|---|---:|---:|---:|
| Transformers baseline | 12.60 | 8.91 | 10.82 |
| vLLM service | 45.80 | 2.71 | 3.39 |

在该样例测试下，vLLM 服务的吞吐量约为 Transformers 基线的 **3.63 倍**。

> 实际结果会受到 GPU 型号、模型大小、batch/concurrency 参数、max tokens、量化方式等因素影响。
