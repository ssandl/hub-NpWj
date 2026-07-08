## 一、实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | MacBook Air M4（16 + 256 GB） |
| 操作系统 | macOS（Apple Silicon） |
| 模型 | Qwen2.5-0.5B-Instruct |
| 模型路径 | `/Users/qing/Documents/code/model/Qwen2.5-0.5B-Instruct` |
| 推理引擎 | vLLM-Metal 0.24.0（vLLM 的 Mac Metal 后端） |
| 对比基线 | HuggingFace Transformers 4.52.4 |
| Python 环境 | 独立虚拟环境 `~/.venv-vllm-metal` |
| 测试请求数 | 50 条 prompts |
| 最大生成长度 | 100 tokens |


## 二、实验过程

在 Mac 平台上，vLLM-Metal 默认 gpu_memory_utilization=0.9（即占用 90% 可用统一内存）。由于 0.5B 模型本身只需约 1GB，90% 分配会导致大量内存被 KV Cache 占据，系统剩余内存不足 1GB，从而触发 Metal 的 Insufficient Memory OOM 报错。

尝试在代码中设置 gpu_memory_utilization=0.6 以限制内存使用：

```bash
llm = LLM(
    model=MODEL_PATH,
    max_model_len=2048,
    gpu_memory_utilization=0.6,
)

但通过日志发现 vLLM-Metal 后端会忽略该参数，强制恢复为 0.9：

```bash
Paged attention: VLLM_METAL_MEMORY_FRACTION=auto, defaulting to 0.90 for paged path


```bash
=============================================================================
模式                            总耗时         QPS       tokens/s    相对vLLM    
-----------------------------------------------------------------------------
[A] transformers 串行          90.88s        0.55         55         0.06×
[B] transformers batch=8      54.27s        0.92          92        0.10×
[C] vLLM 批处理                 5.16s        9.68         942        1.00×

JSON 结果保存：/Users/qing/Documents/code/BaDouAI/课件&代码/week9大模型应用补充知识/week9 大模型应用补充知识/vllm_deployment/src/../outputs/throughput_results.json

柱状图已保存：/Users/qing/Documents/code/BaDouAI/课件&代码/week9大模型应用补充知识/week9 大模型应用补充知识/vllm_deployment/src/../outputs/throughput_comparison.png

======================================================================
  核心结论：
    vLLM 相对 transformers 串行加速：17.6×
    vLLM 相对 transformers batch:    10.5×
    关键机制：PagedAttention + continuous batching
======================================================================


## 三、实验结论

1.vLLM 在 Mac 平台依然有效：即便并非官方优先支持平台，vLLM 仍带来了 17.6 倍的吞吐量提升，证明 PagedAttention 和 Continuous Batching 的优化是跨硬件有效的。
2.绝对性能受限于硬件：Mac M4 的统一内存吞吐量约为 942 tokens/s，低于 NVIDIA RTX 4060 的 3394 tokens/s（参考课程文档数据），属于正常硬件差异，不影响 vLLM 相对加速比的结论。
3.Mac 部署需注意显存管理：vLLM-Metal 默认会抢占 90% 统一内存，在小模型场景下会造成系统内存紧张触发 OOM 警告爆显存。建议跑 benchmark 前关闭非必要后台应用以释放内存。

