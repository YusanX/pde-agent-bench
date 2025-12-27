# PDEBench Data 目录

这是 PDEBench 的唯一真实数据源。所有的 case 定义都存储在此目录中。

## 📁 文件说明

- `benchmark.jsonl`: 完整的 benchmark 数据集（所有 cases）
- `schema.json`: JSON Schema 定义（可选）
- `README.md`: 本文件

## 📊 数据格式

每一行是一个 JSON 对象，表示一个 case。

### 关键字段

```json
{
  "id": "case_identifier",
  "pde_classification": {
    "equation_type": "poisson",    // Oracle Solver 选择器
    "math_type": ["elliptic"]      // Metrics 选择器
  },
  "oracle_config": { ... },        // Oracle 物理配置
  "evaluation_config": { ... },    // 评测策略（不含具体数值）
  "tags": { ... }                  // 排行榜多维分析标签
}
```

### 重要说明

⚠️ **此数据不包含任何运行时基准值** (如 `target_error` 的具体数值、`baseline`、`difficulty_tiers`)。

这些值将由 `scripts/build_cases.py` **动态生成**：
1. 运行 Oracle 求解器
2. 获取基准性能 (error_ref, time_ref)
3. 计算难度分级
4. 生成最终的 `cases/*/config.json`

## 🚀 如何使用

### 1. 从旧数据迁移 (首次使用)

```bash
python scripts/migrate_to_data.py
```

这将合并 `cases` 和 `datasets/full_benchmark.jsonl`，生成 `data/benchmark.jsonl`。

### 2. 构建 Cases 目录

```bash
# 完整构建（会运行 Oracle，较慢）
python scripts/build_cases.py

# 快速构建（跳过 Oracle，使用默认基准）
python scripts/build_cases.py --skip-oracle

# 仅构建特定 cases
python scripts/build_cases.py --cases poisson_simple heat_simple
```

生成的 `cases/` 目录包含：
- `config.json`: 完整配置（含动态计算的基准值）
- `description.md`: Agent 任务描述（动态生成）
- `test_*.py`: 测试脚本

### 3. Git 管理

```
# Git 跟踪
data/benchmark.jsonl      ← 唯一真实数据源

# Git 忽略
cases/                    ← 生成产物
```

## 📝 数据维护

### 添加新 Case

1. 在 `benchmark.jsonl` 末尾添加新行：

```json
{"id": "new_case", "pde_classification": {...}, "oracle_config": {...}, ...}
```

2. 重新构建：

```bash
python scripts/build_cases.py --cases new_case
```

### 修改现有 Case

1. 编辑 `benchmark.jsonl` 中对应的行
2. 重新构建该 case：

```bash
python scripts/build_cases.py --cases case_id
```

## 🔗 相关文件

- `scripts/build_cases.py`: Cases 构建器
- `pdebench/templates/prompts.py`: Prompt 生成模板
- `pdebench/templates/scripts.py`: 测试脚本模板



