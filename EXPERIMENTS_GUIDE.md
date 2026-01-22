# PDEBench 实验运行指南

本指南介绍如何运行 PDEBench 的 6 个核心实验，使用 **CodePDE** 作为 Code Agent。

---

## 🎯 实验概览

| 实验 | 名称 | 目的 | 关键参数 |
|------|------|------|----------|
| 1.1 | 纯 LLM 零样本 | 测试 LLM 代码生成能力 | `--agent gpt-4o` (单次) |
| 1.2 | Code Agent 单轮 | 测试 Agent 单次表现 | `--agent codepde` (单次) |
| 2.1 | 多轮迭代 | 测试从错误中学习 | `--max-attempts 3` |
| 4.1 | Gate 通过率分析 | 分析通过率门槛 | 自动计算 |
| 4.5 | 错误分析 | 分类失败原因 | 自动分析 |
| 4.6 | 成本效益分析 | 追踪成本与性能 | 自动计算 |

---

## 📦 环境准备

### 1. 激活环境

```bash
conda activate pdebench
cd /Users/yusan/agent/pdebench
```

### 2. 设置 API Keys

```bash
# OpenAI (用于 CodePDE)
export OPENAI_API_KEY="your_openai_api_key"

# 可选：其他 LLM
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
```

### 3. 确认 CodePDE 路径

检查配置文件 `pdebench/configs/codepde.json`:
```json
{
  "codepde_path": "/Users/yusan/agent/CodePDE",
  "model": "gpt-4o"
}
```

---

## 🧪 实验 1.1 - 纯 LLM 零样本测试

**目的**: 测试 LLM 在无外部工具情况下的代码生成能力

### 运行命令

```bash
# 测试单个 case
python scripts/run_benchmark.py \
  --agent gpt-4o \
  --cases poisson_basic \
  --output results/exp1.1_gpt4o_test

# 完整评测 (所有 cases)
python scripts/run_benchmark.py \
  --agent gpt-4o gpt-4o-mini claude-3-5-sonnet \
  --output results/exp1.1_llm_zeroshot

# 按方程类型评测
python scripts/run_benchmark.py \
  --agent gpt-4o \
  --equation-types poisson heat \
  --output results/exp1.1_basic_pdes
```

### 预期输出

- `results/exp1.1_*/summary.json`: 汇总统计
- `results/exp1.1_*/{case_id}/`:
  - `prompt.md`: 输入 prompt
  - `llm_response.txt`: LLM 原始响应
  - `solver.py`: 生成的代码
  - `result.json`: 评测结果

### 生成报告

```bash
python scripts/generate_reports.py \
  --results results/exp1.1_llm_zeroshot \
  --output reports/exp1.1

# 输出:
# - figure1_pass_rate_comparison.png
# - table1_detailed_results.md
```

---

## 🤖 实验 1.2 - Code Agent 单轮测试

**目的**: 测试 Code Agent (CodePDE) 的单次表现

### 运行命令

```bash
# 测试 CodePDE
python scripts/run_benchmark.py \
  --agent codepde \
  --cases poisson_basic heat_basic stokes_basic \
  --output results/exp1.2_codepde

# 对比 LLM vs CodePDE
python scripts/run_benchmark.py \
  --agent gpt-4o codepde \
  --output results/exp1.2_llm_vs_agent
```

### CodePDE 配置

编辑 `pdebench/configs/codepde.json`:
```json
{
  "codepde_path": "/Users/yusan/agent/CodePDE",
  "model": "gpt-4o",          # 或 "claude-3-5-sonnet", "gpt-4o-mini"
  "temperature": 0.7,
  "max_tokens": 4096
}
```

### 预期输出

- `results/exp1.2_*/codepde/{case_id}/`:
  - `agent_response.txt`: CodePDE 响应
  - `solver.py`: 生成的代码
  - `result.json`: 评测结果（包含 token 和成本）

### 生成报告

```bash
python scripts/generate_reports.py \
  --results results/exp1.2_llm_vs_agent \
  --output reports/exp1.2

# 输出:
# - figure2_agent_vs_llm_pass_rate.png
# - table2_agent_framework_comparison.md
```

---

## 🔄 实验 2.1 - 多轮迭代测试

**目的**: 测试 Agent 从错误中学习的能力（最多 3 次尝试）

### 运行命令

```bash
# 单次尝试 (baseline)
python scripts/run_benchmark.py \
  --agent codepde \
  --cases poisson_basic heat_basic \
  --max-attempts 1 \
  --output results/exp2.1_codepde_1attempt

# 多轮尝试 (3次)
python scripts/run_benchmark.py \
  --agent codepde \
  --cases poisson_basic heat_basic \
  --max-attempts 3 \
  --output results/exp2.1_codepde_3attempts

# 完整评测
python scripts/run_benchmark.py \
  --agent codepde \
  --max-attempts 3 \
  --output results/exp2.1_multi_attempt
```

### 多轮迭代工作流程

1. **第1次尝试**: 使用原始 prompt
2. **第2次尝试**: 如果失败，添加错误反馈到 prompt
3. **第3次尝试**: 如果仍失败，再次添加反馈

### 预期输出

- `results/exp2.1_*/{case_id}/`:
  - `attempts_history.json`: 所有尝试的详细记录
  - `solver_attempt_1.py`, `solver_attempt_2.py`, ...
  - `feedback_prompt_attempt_2.md`, `feedback_prompt_attempt_3.md`
  - `result.json`: 包含改进分析

### 改进分析指标

- `num_attempts`: 使用的尝试次数
- `improved`: 是否有改进
- `error_reduction_pct`: 误差下降百分比
- `error_trajectory`: 误差变化轨迹
- `status_trajectory`: 状态变化轨迹

### 生成报告

```bash
python scripts/generate_reports.py \
  --results results/exp2.1_multi_attempt \
  --output reports/exp2.1

# 输出:
# - figure3_pass_rate_vs_attempt_number.png (学习曲线)
# - table3_improvement_statistics.md
```

---

## 📊 实验 4.1 - Gate 通过率分析

**目的**: 分析 case-level 通过率（0/1 判定）

### 运行命令

```bash
# Gate 分析自动运行，无需额外参数
python scripts/run_benchmark.py \
  --agent codepde \
  --output results/exp4.1_gate_analysis

# 查看 summary.json 中的 gate_statistics
cat results/exp4.1_gate_analysis/codepde/summary.json | jq '.gate_statistics'
```

### Gate 定义

1. **Exec Valid** (执行有效性): 代码能否成功执行
2. **Accuracy Pass** (精度门槛): `error ≤ target_error`
3. **Time Pass** (时间门槛): `time ≤ target_time`
4. **Final Pass** (最终通过): 所有门槛都通过

### 通过率计算

```
exec_valid_rate = exec_valid_count / total_cases
accuracy_pass_rate = accuracy_pass_count / total_cases
time_pass_rate = time_pass_count / total_cases
final_pass_rate = final_pass_count / total_cases
```

### 预期输出

`result.json` 中的 `gate_breakdown`:
```json
{
  "exec_valid": true,
  "accuracy_pass": true,
  "time_pass": false,
  "final_pass": false,
  "failure_stage": "time",
  "failure_reason": "TIME_FAIL: time=5.32s > target=2.00s"
}
```

### 生成报告

```bash
python scripts/generate_reports.py \
  --results results/exp4.1_gate_analysis \
  --output reports/exp4.1

# 输出:
# - figure6_gate_breakdown.png (堆叠条形图)
# - table4_case_level_pass_rate.md
```

---

## 🐛 实验 4.5 - 错误分析

**目的**: 自动分类失败原因

### 运行命令

```bash
# 错误分类自动运行
python scripts/run_benchmark.py \
  --agent codepde \
  --output results/exp4.5_error_analysis
```

### 错误分类

| 类别 | 描述 | 示例 |
|------|------|------|
| `syntax_error` | Python/DOLFINx 语法错误 | SyntaxError, IndentationError |
| `api_error` | DOLFINx API 使用错误 | AttributeError, TypeError |
| `import_error` | 导入错误 | ModuleNotFoundError |
| `math_error` | PDE 离散化/数值错误 | 奇异矩阵, NaN, Inf |
| `convergence_error` | 求解器不收敛 | KSP_DIVERGED, SNES_DIVERGED |
| `parameter_error` | 参数选择错误 | 网格太粗/细 |
| `stabilization_missing` | 缺少稳定化 | 需要 SUPG 但未使用 |
| `timeout` | 算法效率低 | 超时 |
| `other` | 其他错误 | - |

### 使用 ErrorClassifier

```python
from pdebench.analysis import ErrorClassifier

classifier = ErrorClassifier()

# 分类单个结果
error_category = classifier.classify(result, case)
print(f"Error type: {error_category}")
print(f"Description: {classifier.get_error_description(error_category)}")

# 批量分析
all_results = [...]  # 从 summary.json 加载
analysis = classifier.analyze_errors_batch(all_results)
print(f"Most common error: {analysis['most_common']}")
print(f"Error distribution: {analysis['error_distribution']}")
```

### 生成报告

```bash
python scripts/generate_reports.py \
  --results results/exp4.5_error_analysis \
  --output reports/exp4.5

# 输出:
# - figure11_failure_mode_distribution.png (饼图)
# - table7_error_analysis_matrix.md
```

---

## 💰 实验 4.6 - 成本效益分析

**目的**: 追踪 API 调用、token 消耗、推理时间和货币成本

### 运行命令

```bash
# 成本追踪自动运行
python scripts/run_benchmark.py \
  --agent codepde gpt-4o gpt-4o-mini \
  --output results/exp4.6_cost_analysis
```

### 成本指标

`summary.json` 中的 `cost_analysis`:
```json
{
  "total_cost_usd": 2.45,
  "total_tokens": 125000,
  "avg_llm_latency_sec": 12.3,
  "cost_per_case_usd": 0.49,
  "cost_per_pass_usd": 0.82,
  "tokens_per_case": 25000
}
```

### 每个 case 的成本

`result.json` 中的 `llm_usage`:
```json
{
  "input_tokens": 450,
  "output_tokens": 1200,
  "total_tokens": 1650,
  "latency_sec": 13.5,
  "cost_usd": 0.0234
}
```

### 成本估算

CodePDE 使用的 LLM 定价（2026 估算）：
- **GPT-4o**: $5/1M input, $15/1M output
- **GPT-4o-mini**: $0.15/1M input, $0.60/1M output
- **Claude-3.5-Sonnet**: $3/1M input, $15/1M output

### 生成报告

```bash
python scripts/generate_reports.py \
  --results results/exp4.6_cost_analysis \
  --output reports/exp4.6

# 输出:
# - figure12_cost_performance_scatter.png
# - table8_cost_benefit_analysis.md
```

---

## 📈 生成完整报告

### 运行所有实验

```bash
# 脚本化运行所有实验
./scripts/run_all_experiments.sh
```

或手动运行：

```bash
# 实验 1.1
python scripts/run_benchmark.py --agent gpt-4o --output results/exp1.1

# 实验 1.2
python scripts/run_benchmark.py --agent codepde --output results/exp1.2

# 实验 2.1
python scripts/run_benchmark.py --agent codepde --max-attempts 3 --output results/exp2.1

# 实验 4.1, 4.5, 4.6 (自动)
# 已包含在上述实验中
```

### 生成所有报告

```bash
for exp in exp1.1 exp1.2 exp2.1 exp4.1 exp4.5 exp4.6; do
  python scripts/generate_reports.py \
    --results results/$exp \
    --output reports/$exp
done
```

---

## 🔍 调试技巧

### 查看详细输出

```bash
# 查看单个 case 的结果
cat results/exp*/agent_name/case_id/result.json | jq '.'

# 查看汇总统计
cat results/exp*/agent_name/summary.json | jq '.pass_rate, .gate_statistics, .cost_analysis'

# 查看失败原因
cat results/exp*/agent_name/summary.json | jq '.results[] | select(.status != "PASS") | {case_id, status, fail_reason}'
```

### 重新运行失败的 cases

```bash
# 列出失败的 cases
python -c "
import json
with open('results/exp1.2/codepde/summary.json') as f:
    data = json.load(f)
    failed = [r['case_id'] for r in data['results'] if r['status'] != 'PASS']
    print(' '.join(failed))
"

# 只重新运行失败的
python scripts/run_benchmark.py \
  --agent codepde \
  --cases poisson_basic heat_basic \
  --output results/exp1.2_retry
```

### 跳过代码生成（使用已有代码）

```bash
# 只重新执行和评测，不调用 LLM
python scripts/run_benchmark.py \
  --agent codepde \
  --skip-generation \
  --output results/exp1.2_codepde
```

---

## 📁 结果目录结构

```
results/
├── exp1.1_llm_zeroshot/
│   ├── gpt-4o/
│   │   ├── summary.json
│   │   ├── poisson_basic/
│   │   │   ├── prompt.md
│   │   │   ├── llm_response.txt
│   │   │   ├── solver.py
│   │   │   ├── agent_output/
│   │   │   │   └── solution.npz
│   │   │   └── result.json
│   │   └── ...
│   └── gpt-4o-mini/
├── exp1.2_codepde/
│   └── codepde/
│       ├── summary.json
│       └── ...
├── exp2.1_multi_attempt/
│   └── codepde/
│       ├── summary.json
│       ├── poisson_basic/
│       │   ├── attempts_history.json
│       │   ├── solver_attempt_1.py
│       │   ├── solver_attempt_2.py
│       │   ├── feedback_prompt_attempt_2.md
│       │   └── result.json
│       └── ...
└── .oracle_cache/
    ├── poisson_basic.json
    └── ...

reports/
├── exp1.1/
│   ├── figure1_pass_rate_comparison.png
│   └── table1_detailed_results.md
├── exp1.2/
│   ├── figure2_agent_vs_llm_pass_rate.png
│   └── table2_agent_framework_comparison.md
├── exp2.1/
│   ├── figure3_pass_rate_vs_attempt_number.png
│   └── table3_improvement_statistics.md
├── exp4.1/
│   ├── figure6_gate_breakdown.png
│   └── table4_case_level_pass_rate.md
├── exp4.5/
│   ├── figure11_failure_mode_distribution.png
│   └── table7_error_analysis_matrix.md
└── exp4.6/
    ├── figure12_cost_performance_scatter.png
    └── table8_cost_benefit_analysis.md
```

---

## ⚠️ 注意事项

### 1. API 成本控制

- 先用少量 cases 测试: `--cases poisson_basic heat_basic`
- 使用更便宜的模型测试: `gpt-4o-mini`
- 每个实验的估算成本:
  - 实验 1.1/1.2 (50 cases): ~$5-10
  - 实验 2.1 (50 cases, 3 attempts): ~$15-30

### 2. 时间预算

- 单个 case: 1-5 分钟（包括 LLM 调用和执行）
- 完整实验 (50 cases): 1-4 小时
- 多轮迭代 (3 attempts): 3-12 小时

### 3. 资源要求

- **内存**: 8GB+ (DOLFINx 求解器)
- **存储**: 2GB+ (结果和缓存)
- **网络**: 稳定连接（API 调用）

---

## 🎯 快速开始示例

```bash
# 1. 完整流程示例（5-10分钟）
conda activate pdebench
export OPENAI_API_KEY="your_key"

# 测试 2 个简单 cases
python scripts/run_benchmark.py \
  --agent codepde \
  --cases poisson_basic heat_basic \
  --max-attempts 3 \
  --output results/quick_test

# 生成报告
python scripts/generate_reports.py \
  --results results/quick_test \
  --output reports/quick_test

# 查看结果
cat results/quick_test/codepde/summary.json | jq '.pass_rate, .gate_statistics, .cost_analysis'
open reports/quick_test/*.png
```

---

## 📞 支持

遇到问题？
1. 检查 `results/*/summary.json` 中的错误统计
2. 查看详细的 `result.json` 和 `stderr`
3. 使用 `ErrorClassifier` 分析失败原因
4. 参考 `PHASE2_CODE_AGENT_GUIDE.md` 了解 CodePDE 配置

---

