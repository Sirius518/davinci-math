# CanonicalRecord Schema

本文档描述 `CanonicalRecord` 的完整字段设计，覆盖 Stage 1–6 数据处理 pipeline 的全部产出。

## 设计原则

1. **只丢弃不改写** — `question` 就是当前版本，不合格的数据直接丢弃，不保留 raw/cleaned 双版本
2. **分流与过滤合并** — `training_phase` + `filter_tag` 两个字段同时完成路由和过滤原因记录
3. **rollout 与难度合并** — `verification` 嵌套结构统一承载 rollout 采样、GT 验证、通过率、难度判定
4. **CoT 蒸馏独立** — `distillation` 嵌套结构支持多模型蒸馏，每个模型独立记录质量状态
5. **去污染完整证据链** — `decontamination` 嵌套结构覆盖 embedding 召回 → LLM 精判 → 最终决策

---

## 顶层字段

```python
@dataclass
class CanonicalRecord:
    # 身份
    record_id: str
    dataset_name: str           # 如 "DeepMath-103K"、"numinamath"、"OpenR1-math"

    # 核心内容
    question: str               # 题面（通过清洗的版本；不合格的已被丢弃）
    raw_dataset_answer: str     # 原始数据集中给出的答案
    confirmed_answer: str       # 经 rollout 验证后最终确认的答案

    # 分流 + 过滤
    training_phase: str         # "posttrain" / "midtrain" / "drop"
    filter_tag: str             # posttrain 为空；midtrain/drop 填命中原因（如 `low_confidence`）

    # 嵌套结构（Parquet 中以 JSON 字符串存储）
    verification: dict          # rollout + GT 验证 + 难度判定
    distillation: dict          # CoT 蒸馏结果（多模型）
    decontamination: dict       # benchmark 去污染证据链

    # 扩展
    meta: dict                  # 仅用于实验性标签
    trace: list[TraceEvent]     # 处理轨迹
```

---

## `verification` — rollout + GT 验证 + 难度

对应 Stage 4（正确性验证）和 Stage 6（难度/通过率过滤）。

```json
{
    "model": "gpt-oss-120b-medium",
    "fallback_model": "gpt-oss-120b-high",
    "used_fallback": false,
    "num_samples": 8,
    "majority_answer": "x = 3",
    "majority_count": 6,

    "raw_dataset_answer": "x = 3",
    "majority_matches_gt": true,
    "confirmed_answer": "x = 3",
    "llm_judge_gt_quality": "clear",
    "llm_judge_equivalence": "equivalent",
    "llm_judge_model": "gpt-oss-120b",

    "all_correct": false,
    "pass_ratio": 0.875,
    "difficulty_bucket": "hard",

    "samples": [
        {
            "index": 0,
            "answer": "x = 3",
            "solution": "Let x + 1 = 4, so x = 3",
            "reasoning": "<full CoT reasoning text>",
            "success": true,
            "stop_reason": "natural",
            "token_count": {"qwen2.5": 1234},
            "judge_correct": true
        }
    ]
}
```

字段说明：

- `model` / `fallback_model` — 主模型和副模型。副模型仅在主模型 majority 与 GT 不一致时启用
- `used_fallback` — 是否实际启用了副模型
- `num_samples` — rollout 采样次数
- `majority_answer` / `majority_count` — 多数投票结果及票数；当 `majority_count <= 1` 时，说明模型对题目缺乏共识，记录会被降级为 `training_phase="midtrain"` 且 `filter_tag="low_confidence"`
- `raw_dataset_answer` — 原始数据集 GT（与顶层字段相同，冗余存储方便独立使用）
- `majority_matches_gt` — majority 是否与 GT 一致
- `confirmed_answer` — 最终确认的答案（同步写到顶层 `confirmed_answer`）
- `llm_judge_gt_quality` — 当 GT 与 majority 初步不一致时，LLM 对 GT 答案清晰度的判断：`clear` / `unclear`
- `llm_judge_equivalence` — 当 GT 清晰时，LLM 对 GT 与 majority 是否等价的判断：`equivalent` / `not_equivalent` / `not_applicable`
- `llm_judge_model` — 执行上述判断所使用的模型
- `all_correct` — 是否全部 sample 正确（用于判定 easy → midtrain）
- `pass_ratio` — 通过率。rollout 阶段会先按规则匹配写入 `samples[].judge_correct=true` 的样本；后续逐 sample judge 会仅补判剩余未确认样本，并根据最终的 `samples[].judge_correct` 结果重新计算（如 7/8 = 0.875）。注意 `training_phase` 的降级不依赖该字段，而依赖 `majority_count`：若 `majority_count <= 1`，则视为低置信度样本并转入 `midtrain`
- `difficulty_bucket` — 难度分桶：`easy` / `medium` / `hard`
- `samples[].answer` / `solution` / `reasoning` — 每次 rollout 的完整三件套
- `samples[].success` — 该次请求是否成功
- `samples[].stop_reason` — `"natural"`（自然停止）/ `"truncated"`（被截断）/ `"error"`（请求失败）
- `samples[].token_count` — 以 `<think>{reasoning}</think>{solution}` 格式拼接后的 token 数，按 tokenizer 名称为 key 的 dict。如 `{"qwen2.5": 1234}`
- `samples[].judge_correct` — 该次 rollout 的答案是否应视为正确。`true` 表示规则已判定与 GT 一致，或后续被 20B 小模型判为等价；`false` 表示 20B 小模型判为不等价；`null` / 缺失表示 rollout 规则未命中，等待后续逐 sample judge 补判

---

## `distillation` — CoT 蒸馏

对应 Stage 5。一个题目可能用多个模型蒸馏，每个模型只蒸馏一次。外层 key 为模型名。

```json
{
    "gpt-oss-120b-high": {
        "model": "gpt-oss-120b-high",
        "solution": "<distilled solution text>",
        "reasoning": "<distilled reasoning/CoT text>",
        "quality_status": "valid",
        "quality_tags": []
    },
    "qwen-72b": {
        "model": "qwen-72b",
        "solution": "...",
        "reasoning": "...",
        "quality_status": "invalid",
        "quality_tags": ["truncated"]
    }
}
```

字段说明：

- `model` — 蒸馏所用模型
- `solution` — 蒸馏得到的解答
- `reasoning` — 蒸馏得到的推理过程（CoT）
- `quality_status` — `"valid"` / `"invalid"`，下游直接用于过滤
- `quality_tags` — 不合格原因列表，如 `["truncated", "non_english", "no_final_answer"]`；合格时为空

与 `verification.samples` 的区别：蒸馏是"一次性出一条高质量 CoT"，而非"多次采样投票"。

---

## `decontamination` — 去污染

对应 Stage 3。记录 embedding 召回 → LLM 精判 → 最终决策的完整证据链。

```json
{
    "status": "clean",

    "embedding_model": "BGE-large-en-v1.5",
    "similarity_threshold": null,

    "llm_judge_model": "gpt-4o",
    "top_k": 5,

    "candidates": [
        {
            "benchmark_id": "gsm8k_train_0042",
            "benchmark_source": "GSM8K",
            "similarity": 0.92,
            "llm_decision": "yes"
        },
        {
            "benchmark_id": "math_test_1137",
            "benchmark_source": "MATH",
            "similarity": 0.87,
            "llm_decision": "no"
        }
    ],

    "contamination_source_id": null,
    "contamination_source_set": null
}
```

字段说明：

- `status` — 最终决策：`"clean"` / `"contaminated"` / `"not_checked"`。**唯一的下游过滤依据**
- `embedding_model` — embedding 模型名
- `similarity_threshold` — 相似度阈值。`null` 表示不设门槛，直接取 top-k 全部送 LLM
- `llm_judge_model` — LLM 判定模型名
- `top_k` — 取 top-k 候选送 LLM
- `candidates[]` — 候选证据列表（最多 top_k 条）
  - `benchmark_id` — benchmark 样本 ID
  - `benchmark_source` — 所属 benchmark 集（如 GSM8K、MATH）
  - `similarity` — cosine similarity
  - `llm_decision` — `"yes"` / `"no"` / `null`（未送 LLM 时）
- `contamination_source_id` / `contamination_source_set` — 确认污染时指向的 benchmark 样本及所属集；`status` 为 `"clean"` 时为 `null`

配置字段记录在每条 record 内，因为不同批次可能使用不同配置。embedding 向量本身不存在 record 中。

---

## `filter_tag` 枚举

`filter_tag` 记录样本被标记为 midtrain 或 drop 的原因。posttrain 样本的 `filter_tag` 为空字符串。

| Stage | Tag | 含义 |
|-------|-----|------|
| Stage 1 | `boolean_and_mcq` | GT 是 true/false/yes/no 或选择题无法转为开放式 |
| Stage 1 | `non_english` | 含非英文字符 |
| Stage 1 | `short_question` | 过短（单词 < 5 且字符 < 20） |
| Stage 1 | `rule_verify_failed` | math_verify 验证失败 |
| Stage 3 | `contaminated` | benchmark 去污染确认 |
| Stage 4 | `gt_suspect` | majority 与 GT 不一致，疑似 GT 错误 |
| Stage 4 | `correctness_failed` | 不满足 require_correct / require_majority |
| Stage 4 | `answer_mismatch` | GT 清晰，但与 majority 不一致，以 GT 为准降级到 midtrain |
| Stage 4 | `gt_unclear` | GT 不清晰，但 majority 高置信度一致，降级到 midtrain 并标记 |
| Stage 4 | `answer_uncertain` | GT 不清晰且 majority 不统一，无法确认答案 |
| Stage 5 | `cot_truncated` | CoT 被截断 |
| Stage 5 | `cot_non_english` | CoT 含非英文 |
| Stage 6 | `too_easy` | pass_ratio 过高 |
| Stage 6 | `too_hard` | pass_ratio 过低 |
| Stage 6 | `all_correct_easy` | rollout 全对，降级为 midtrain |

---

## 数据流

```
RawData
  → Stage 1: Clean & Filter → drop(filter_tag) / keep
  → Stage 2: Dedup
  → Stage 3: Decontamination → drop(contaminated) / clean
  → Stage 4: Verification (rollout + GT check)  → drop(gt_suspect) / keep
  → Stage 5: CoT Distillation
  → Stage 6: Pass Ratio Filter → drop(too_easy, too_hard) / keep
  → Route: posttrain / midtrain
```

## Parquet 存储

- 顶层标量字段直接映射为 Arrow 列
- `verification`、`distillation`、`decontamination` 以 `pa.large_string()` 存储 JSON 序列化结果
- `meta` 以 `pa.large_string()` 存储 JSON
- `trace` 以 `pa.large_string()` 存储 JSON 数组
