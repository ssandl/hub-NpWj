"""Shared configuration for week09 vLLM benchmark."""

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_NAME = "week09-qwen"
DEFAULT_HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "请用三句话解释 vLLM 为什么能提升大模型推理吞吐量。",
    "请总结大模型服务部署时需要关注的显存、并发和延迟指标。",
    "请说明 PagedAttention 与传统 KV Cache 管理方式的区别。",
    "请写一个简短的课程作业结论，主题是 vLLM 服务验证。",
    "请解释连续批处理 continuous batching 对在线推理服务的意义。",
    "请给出三条优化大模型推理服务性能的工程建议。",
    "请说明吞吐量 tokens/s 和请求延迟 latency 的区别。",
    "请从工程角度说明为什么要把模型推理封装成 API 服务。",
]
