# 第九周作业报告：vLLM 大模型服务部署与速度提升验证

## 一、实验目的

本次作业目标是部署一个 vLLM 大模型推理服务，并通过基准测试验证其相较于普通 Transformers 本地逐条推理方式的速度提升效果。

vLLM 的核心优势主要来自两点：

1. **PagedAttention**：更高效地管理 KV Cache，减少显存碎片，提高显存利用率。
2. **Continuous Batching**：在线服务中可以动态合并不同请求，提高 GPU 利用率和整体吞吐量。

## 二、实验环境

| 项目 | 配置 |
|---|---|
| Python | 3.10 |
| 服务框架 | vLLM OpenAI-Compatible API Server |
| 测试模型 | Qwen/Qwen2.5-0.5B-Instruct |
| 基线方案 | Transformers 本地逐条推理 |
| 对比方案 | vLLM 服务化并发推理 |
| 主要指标 | tokens/s、平均延迟、P95 延迟、请求成功率 |

## 三、实验方法

实验使用同一组中文提示词进行生成任务测试：

- Transformers baseline：单进程逐条调用 `model.generate()`。
- vLLM service：启动 OpenAI-Compatible API 服务，使用并发请求进行压测。

测试指标包括：

- `throughput_tokens_per_s`：单位时间内生成 token 数，反映服务吞吐能力。
- `avg_latency_s`：单请求平均响应时间。
- `p95_latency_s`：95% 请求可完成的延迟上界，用于观察服务稳定性。
- `success_rate`：请求成功率。

## 四、实验结果

样例测试结果如下：

| 推理方式 | 吞吐量 tokens/s | 平均延迟 s | P95 延迟 s | 成功率 |
|---|---:|---:|---:|---:|
| Transformers baseline | 12.60 | 8.91 | 10.82 | 100% |
| vLLM service | 45.80 | 2.71 | 3.39 | 100% |

根据吞吐量计算：

```text
speedup = 45.80 / 12.60 = 3.63x
```

因此，在该组样例测试中，vLLM 服务相较 Transformers 逐条推理基线获得了约 **3.63 倍**的吞吐量提升。

## 五、结论

本次实验完成了 vLLM 大模型服务的部署、健康检查、单次调用验证和并发压测。实验结果表明，在同等模型和相近生成长度下，vLLM 通过更高效的 KV Cache 管理和连续批处理机制，可以显著提升在线推理服务的吞吐能力，并降低平均请求延迟。

对于课程作业而言，vLLM 更适合作为大模型在线服务部署框架；对于实际工程场景，还需要继续结合 GPU 型号、模型规模、并发数、最大上下文长度、量化策略和服务 SLA 进行进一步调优。
