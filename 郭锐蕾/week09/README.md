# 三、各脚本使用方法

所有脚本位于 `src/`，运行前确保已激活 venv 且 server 已启动（`bench_throughput.py` 除外）。

### 3.1 demo_guided_choice.py — 枚举约束

```bash
cd src/
python demo_guided_choice.py
```

**场景**：金融问答意图路由（查股价 / 查财报 / 查新闻 / 对比分析 / 其他）

**内部流程**：
1. 对每个测试问题，调用 server 两次：裸 prompt + `extra_body={"guided_choice": ...}`
2. 对比两种模式的输出合法率和分类准确率

**预期输出**（关键行）：
```
输出合法（在枚举内）   10/12 (83%)    12/12 (100%)
预测正确            3/12 (25%)     3/12 (25%)
```

**教学要点**：guided_choice 100% 保证输出合法，分类正确率决定于模型本身能力。

---

### 3.2 demo_guided_regex.py — 正则约束

```bash
python demo_guided_regex.py
```

**场景**：日期标准化（→ YYYY-MM-DD）、股票代码抽取（→ 6 位数字）

**教学要点**：凡下游有严格解析器的字段，正则约束能把"模型说对但格式错"的问题一次根治。

---

### 3.3 demo_guided_json.py — JSON Schema 基础

```bash
python demo_guided_json.py
```

**场景**：财报问答意图抽取（公司/年度/指标三元组）

**三种模式对比**：
- 裸 prompt：靠指令和 few-shot
- `response_format={"type": "json_object"}`：OpenAI 标准，保证是 JSON
- `guided_json=schema`：vLLM 扩展，保证完全符合 Schema

**关键看点**："22 年" 这类输入下，裸 prompt 和 response_format 可能输出 `year: 22`（违反 `minimum: 2015`），只有 guided_json 能强制修正为 2022。

---

### 3.4 demo_response_format.py — OpenAI 标准方式

```bash
python demo_response_format.py
```

**场景**：新闻情感分类 + 置信度 + 关键词

**教学要点**：`response_format={"type": "json_object"}` 是 OpenAI/Azure/vLLM 都兼容的**可移植方案**。相比 `guided_json` 它跨厂商可用但约束更弱。选型时权衡：
- 跨厂商部署 → response_format
- 单一 vLLM 部署 + 严格解析 → guided_json

---

### 3.5 demo_function_call.py ★ 核心

```bash
# 跑两个工具共 100 个用例
python demo_function_call.py

# 只跑一个
python demo_function_call.py --tool stock
python demo_function_call.py --tool order
```

**两个工具**：
- `get_stock_quote`：金融股价查询，schema 含 string+enum+regex+array+minItems
- `create_order`：电商下单，schema 含 integer 范围+手机号正则+多枚举

**每个工具 50 条测试**，三种模式对比，产出：
- 终端表格：JSON 合法率 / 必选字段率 / schema 完全通过率
- 典型失败案例（前 3 条）
- `outputs/function_call_results.json`：详细数据（可用于后续分析）

**预期结果**：
| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 合法 | ~90% | 100% | 100% |
| 字段齐全 | ~90% | 100% | 100% |
| **完整 schema 通过** | **40-60%** | **40-70%** | **100%** |

**核心教学点**：`response_format` 和 `guided_json` 之间的 30~50 个百分点差距就是约束解码的工程价值——`response_format` 只管语法，不管字段值是否合法。

---
