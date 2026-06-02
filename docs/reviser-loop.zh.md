# Reviser 修订循环

本文档讲解 `reviser/` 模块的工作机制：在同一任务上执行多次 attempt，每次基于上一次的轨迹 + 教程让 VLM 修订 skills，并产出便于做 ablation 的双视图数据。

> 阅读前置：先读 [`execution-flow.md`](execution-flow.md) 了解单次 attempt 内的 step 循环。

---

## 1. 这是什么

「Reviser」是夹在多次 attempt 之间的两阶段管线：

1. attempt_j 跑完，`traj.jsonl` 落盘。
2. **Phase 1 — `ReviserAnalyzer`**：把 traj 按 `chunk_size` 切片，逐段输出滚动 `<summary>`，最后一段输出 `<root_cause>`，整体写入 `attempt_{j+1}/root_cause.xml`。
3. **Phase 2 — `ReviserRefiner`**：以 4 路输入产新 skills，写入 `attempt_{j+1}/skills/`：
   - **当前 root_cause**（刚由 attempt_j 的 traj 分析得到）
   - **旧 skills**（attempt_j 用的那一份）
   - **教程原文** + 教程图片（受 `tutorial_image_cap` 限）
   - **历次 root_cause 链**：attempt_2..attempt_j 的 `root_cause.xml`，避免 refiner 把之前的修复又改回去。详见 §3。
4. attempt_{j+1} 用新 skills 重跑，循环直到 `max_attempts`。

设计目标有两层：
- **修订 skills**：让后续 attempt 不再重复同样错误。
- **产出 ablation 数据**：early-stop 标记让一次完整的 `max_attempts` 跑同时回答两个问题——「按 early-stop 策略我能拿几分」和「再多跑几轮分数会不会回升或抖动」。

---

## 2. 何时启用

由 `configs/config.yaml: reviser.max_attempts` 控制：

| `max_attempts` | 行为 |
|---|---|
| `1`（默认） | 不启用 reviser；只跑一次 attempt_1，写入 canonical bucket。 |
| `≥ 2` | 启用 reviser；attempt_1 写 canonical bucket，attempt_2..N 写 reviser bucket。 |

`agent_mode in {vanilla, vanilla_tutorial}` 时 `_effective_max_attempts()`（`runner.py`，常量 `_NO_REVISER_MODES`）会强制夹回 1 —— 这两种模式都没有 skills 可改。

---

## 3. Refine 阶段的输入：当前 root_cause + 历次 root_cause 链

每次 refine 都会拿到「当前轮 root_cause」+「之前所有轮的 root_cause」一起喂给 VLM。这一节解释为什么、是什么、可以怎么关。

### 为什么需要历史

只看当前轮 root_cause，refiner 容易把上一轮刚加进去的修复又改掉 —— 比如 attempt_2 加了「先点击 Activities 再搜索」的步骤把 attempt_1 的失败修了，但 attempt_2 又因为别的原因失败；如果 refiner 只看 attempt_2 的 root_cause，它可能把 Activities 那步当多余删掉，attempt_3 就退化回 attempt_1 的错误。喂历史 root_cause 等于告诉 refiner「这些修复已经验证过了，别动」。

### 历史的具体内容

`_collect_history(target_attempt=N)`（`reviser_runner.py:506`）按顺序读 `attempt_2/root_cause.xml`、`attempt_3/root_cause.xml`、…、`attempt_{N-1}/root_cause.xml`。

- `attempt_1` 没有 `root_cause.xml`（它前面没有任何 attempt），所以历史从 `attempt_2` 起。
- 当前轮自己的 root_cause 不在 history 里，是另一个参数单独传。

### 具体例子

跑 `max_attempts=4`，每一轮 refine 看到的是：

| 产生的 attempt | 当前 root_cause（来源） | history（=列表） |
|---|---|---|
| `attempt_2` | 分析 `attempt_1/traj.jsonl` | `[]` |
| `attempt_3` | 分析 `attempt_2/traj.jsonl` | `[attempt_2/root_cause.xml]` |
| `attempt_4` | 分析 `attempt_3/traj.jsonl` | `[attempt_2/root_cause.xml, attempt_3/root_cause.xml]` |

注意 `attempt_2/root_cause.xml` 实际是「分析 attempt_1 traj 得到的 root_cause」—— 它存在 attempt_2 目录下，是因为这份 rc **就是用来产生 attempt_2 skills 的**。所以 history 链反映的是「之前每一轮 refine 各自看到了什么问题、做了什么修订」。

### Prompt 中的呈现

`refiner.py:_format_history_block()` 把 history 渲染成 `<previous_root_cause attempt="2">...</previous_root_cause>` 这样的 XML 块塞进 user prompt，与当前轮的 `<root_cause>` 并列。VLM 可以横向对比哪些 issue 是反复出现的、哪些已经被修过。

### 关掉历史（ablation）

设 `reviser.history_in_refine=false` 即可让 `_collect_history()` 直接 `return []`（`reviser_runner.py:520`）。这是消融实验用的开关，默认开。

> 注意：analyzer 的滚动 `<summary>` 是 chunk 之间的，**不属于** 这里说的「历次 root_cause」；那是同一个 attempt 内部的中间产物。

---

## 4. 双桶 result 布局

`runner.py:_compute_run_names()` 决定两个 bucket 名：

```
results/{tutorial_type}/a2s-{mode}-{agent_model}/{benchmark}/{domain}/{task_id}/
  task.json
  attempt_1/                         ← 永远在 canonical bucket
    traj.jsonl, result.txt, runtime.log, recording.mp4, step_N_*/
    skills/, meta.json

results/{tutorial_type}/a2s-{mode}-{agent_model}-r_{reviser_model}/{benchmark}/{domain}/{task_id}/
  task.json
  attempt_2/                         ← attempt_2+ 都在 reviser bucket
    root_cause.xml                   ← analyzer 产出，决定本 attempt 用什么 skills
    skills/                          ← refiner 产出
    traj.jsonl, result.txt, runtime.log, recording.mp4, step_N_*/
    meta.json
  attempt_3/  ...
```

设计要点：

- **canonical bucket 永远是裸 agent 的 baseline**，跨 reviser 实验复用。一旦写完只读，`reviser_runner.py` 在 resume 时会跳过 canonical 的清理逻辑（`reviser_runner.py:236`）。
- **reviser bucket 按 reviser model 区分**。即使 `reviser.model == agent.model`，只要 `max_attempts > 1`，仍会产生独立的 `-r_{reviser_model}` 目录，不会污染 baseline。
- 当 `max_attempts == 1` 两个 bucket 名相同（`_compute_run_names()` 直接 return 同一个 canonical 名）。

---

## 5. `meta.json` 字段

每个 `attempt_N/meta.json`：

```json
{
  "score": 0.0,
  "steps_taken": 12,
  "skills_count": 4,
  "early_stop_triggered": false,
  "completed": true
}
```

`early_stop_triggered` 是这次循环最重要的字段。它标记当前配置下第一个满足 early-stop signal 的 attempt。默认 `reviser.early_stop_signal=likely_success`，即 analyzer 的 `outcome_assessment == "likely_success"`。

**关键点**：循环不会真的因为这个标记而 break。代码只把标记打到当前 attempt 的 `meta.json` 上，然后 force-continue 到 `max_attempts`。这样一次跑同时产生两份视图：

- **Early-stop 视图**：取 `early_stop_triggered=True` 的那个 attempt 的 score（如果不存在则取最后一个 attempt 的 score）。
- **Full-run 视图**：所有 attempt 的 score 序列（看是否抖动 / 回升 / 恶化）。

---

## 6. early_stop 触发条件

由 `ReviserAnalyzer` 在 attempt_j 之后被调用，输出 `RootCauseAnalysis(issues=[...], outcome_assessment=...)`：

- 默认 `likely_success` 策略：`outcome_assessment == "likely_success"` 时触发，对应评估中使用的 analyzer-selected early-stop 视图。
- 兼容 `no_issue` 策略：`issues == []` 时触发，可通过 `reviser.early_stop_signal=no_issue` 显式切换。
- 触发时把 `early_stop_triggered=True` 写到 attempt_j（**不是** attempt_{j+1}）的 `meta.json`。
- `already_marked` flag 保证一次循环里只标第一次，避免重复回写。

崩溃恢复路径：如果一个 attempt 已经 saved meta（默认 `early_stop_triggered=False`）但在 flip 之前 worker 挂了，下一次启动 `_recover_skills_for_attempt()` 在解析 `attempt_j/root_cause.xml` 时会调用 `_backfill_early_stop_if_first()`（`reviser_runner.py:469`）补打标记。

---

## 7. 配置项

定义在 `configs/config.yaml` 顶层 `reviser.*`，可以被 `configs/benchmark/<name>.yaml` 覆盖：

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_attempts` | `1` | 总尝试次数。`1` = 关闭 reviser。 |
| `model` / `base_url` / `api_key` | `null` | reviser 用的 VLM。`null` = 复用 agent 的 client。 |
| `analysis_temperature` | `null` | analyzer 温度。`null` 走 reasoning endpoint 兼容配置。 |
| `refine_temperature` | `null` | refiner 温度，同上。 |
| `analysis_max_tokens` | `32768` | analyzer 单次响应上限。 |
| `refine_max_tokens` | `32768` | refiner 单次响应上限。 |
| `response_char_limit` | `32768` | trajectory 中 agent response 渲染时截断长度。 |
| `rolling_summary_char_limit` | `32768` | 滚动 `<summary>` 字符上限，超过会被压缩。 |
| `tutorial_image_cap` | `20` | refiner 注入的最多教程图片数。 |
| `include_tutorial_in_refine` | `true` | 关掉后 refiner 只看 root_cause + 旧 skills。 |
| `early_stop_signal` | `likely_success` | early-stop 标记信号。可设为 `no_issue` 使用旧语义。 |
| `history_in_refine` | `true` | 把历次 `root_cause.xml` 也喂给 refiner，避免覆盖之前的修复。详见 §3。 |
| `chunk_size` | 各 benchmark YAML | OSWorld / Minecraft = 15，RLCard = 30。analyzer 的 trajectory 切片大小。 |

---

## 8. Resume 与崩溃恢复

`scan_attempts(canonical_dir, reviser_dir)` 在每次 `run_with_reviser()` 启动时扫两个 bucket，找出 `last_complete`。下一轮从 `last_complete + 1` 接着跑：

- `attempt_j/skills/` 已存在 → 直接拿来跑（不重复 refine）。
- `attempt_j/root_cause.xml` 存在但 `skills/` 缺失 → 跳过 analyze，只跑 refine。
- 都不存在 → 读 `attempt_{j-1}/traj.jsonl`，跑完整 analyze + refine。

清理策略（`reviser_runner.py:236`）：执行类 artifact（`traj.jsonl`、`step_*/`、`result.txt`、`response.log`、`reviser_chunks/`）会被清；持久 artifact（`skills/`、`root_cause.xml`、`meta.json`）保留。canonical attempt_1 永不被清。

---

## 9. 分析工具

- **`anything2skill/metrics/tracker.py`** —— `ExperimentTracker.add_attempt()` 把 `attempt_N/meta.json` 聚合到 task 维度。
- **`results/{tutorial_type}/{run_name}/{benchmark}/experiment_results.json`** —— 每个 attempt 完成后由 `_rebuild_experiment_results()` 重写一次。reviser bucket 的 `experiment_results.json` 会**主动并入** canonical 的 `attempt_1`（按设计：reviser 实验需要看到 baseline）。
- **`notebooks/analyze_one_model.ipynb`** —— per-model 逐 attempt 分析，输出双视图（early-stop / full-run）的 success_rate、avg_score、avg_steps，导出到 `RUN_DIR/analysis_one_model.json`。

---

## 10. 关键文件索引

| 文件 | 职责 |
|---|---|
| `anything2skill/reviser/reviser_runner.py` | `ReviserRunner.run_with_reviser()` —— 主循环、resume、early_stop 标记、experiment_results 刷新；`_collect_history()` 拉历次 root_cause |
| `anything2skill/reviser/refiner.py:_format_history_block` | 把 `list[RootCauseAnalysis]` 渲染成 prompt 里的 `<previous_root_cause>` XML 块 |
| `anything2skill/reviser/analyzer.py` | `ReviserAnalyzer.analyze()` —— Phase 1，trajectory → `<root_cause>` XML |
| `anything2skill/reviser/refiner.py` | `ReviserRefiner.refine()` —— Phase 2，XML + skills + tutorial → 新 Skills |
| `anything2skill/reviser/data_types.py` | `RootCauseAnalysis` 数据类 |
| `anything2skill/reviser/trajectory.py` | trajectory 分段 / 渲染辅助 |
| `anything2skill/reviser/skills_io.py` | `Skills` 落盘 / 读取（`SKILL.md` 格式） |
| `anything2skill/reviser/prompts.py` | analyzer / refiner 的 prompt 模板（与 `agent/prompts.py` 平行） |
| `anything2skill/runner.py:_compute_run_names` | canonical / reviser bucket 名计算 |
| `anything2skill/runner.py:_effective_max_attempts` | `vanilla` / `vanilla_tutorial` 模式的 max_attempts 强制夹 1（见 `_NO_REVISER_MODES`） |
| `anything2skill/runner.py:_make_reviser_vlm_factory` | reviser VLMClient 懒构造（同模型时复用 agent vlm） |
| `anything2skill/metrics/tracker.py` | `ExperimentTracker` / `AttemptResult` / `TaskResult` |
