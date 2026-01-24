# PDEBench: AI-Driven PDE Solver Code Generation Benchmark

**世界首个评估大型语言模型和AI Agent端到端PDE求解代码生成能力的基准测试系统。**

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.10.0-orange.svg)]()

## 🎯 项目愿景

PDEBench 评估 AI Agent 是否能"像计算科学家一样思考"：

- **从物理到代码**：给定自然语言描述的 PDE 问题，Agent 需生成完整的 FEniCSx/dolfinx 求解代码
- **数值稳定性意识**：高对流问题需要SUPG稳定化，Agent 能否识别？
- **网格无关验证**：Agent 和 Oracle 可能使用不同网格，系统通过插值进行公平比较
- **精度-速度权衡**：支持 fix_accuracy（比速度）和 fix_time（比精度）两种评测模式

## 🚀 快速开始

### 1. 安装环境

```bash
# 创建 conda 环境并安装 FEniCSx
conda create -n pdebench python=3.11
conda activate pdebench
conda install -c conda-forge fenics-dolfinx=0.10.0 mpich petsc4py

# 安装 PDEBench
cd pde-agent-bench
pip install -e .
```

### 2. 评测你的 Agent

假设你的 Agent 已经生成了求解器代码，目录结构如下：

```
results/my_agent/
├── poisson_simple/
│   └── solver.py
├── heat_simple/
│   └── solver.py
└── ...
```

运行统一评测入口：

```bash
python scripts/run_evaluation.py \
    --agent-name my_agent \
    --agent-dir results/my_agent \
    --cases-dir cases \
    --output results/my_agent/evaluation.json \
    --modes fix_accuracy fix_time \
    --timeout 300
```

### 3. 生成排行榜

```bash
python scripts/generate_leaderboard_v2.py \
    --results results/*/evaluation.json \
    --output leaderboard_v2.html
```

在浏览器中打开 `leaderboard_v2.html` 查看：
- 总体排行榜（速度和精度）
- PDE类型子榜单（椭圆型、抛物型等）
- 专用指标分析（效率、能量衰减、CFL数等）

## 📖 Agent 开发指南

### Agent 任务说明

详见 [`AGENT_TASK.md`](AGENT_TASK.md)

**核心要求：**
1. 读取 `cases/{case_id}/description.md` 了解问题
2. 生成符合规范的 `solver.py`
3. 保存到 `results/{agent_name}/{case_id}/solver.py`

### Solver 接口规范

**命令行参数：**
```bash
python solver.py \
    --resolution 128 \
    --degree 2 \
    --outdir output/ \
    [--dt 0.01]  # 时间相关PDE需要
```

**输出文件：**

1. `solution.npz` (必需)：
```python
np.savez('output/solution.npz',
    x=x_grid,  # 1D array
    y=y_grid,  # 1D array
    u=u_grid   # 2D array (ny, nx)
)
```

2. `meta.json` (必需)：
```json
{
  "wall_time_sec": 1.23,
  "solver_info": {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "iterations": 45
  }
}
```

## 📁 项目结构（2025-12重构）

```
pdebench/
├── pdebench/                    # Python 包
│   ├── harness/                # 🆕 评测编排器
│   │   ├── case_runner.py      # 单case运行器
│   │   └── batch_evaluator.py  # 批量评测器
│   │
│   ├── metrics/                # 🆕 指标计算模块
│   │   ├── scoring.py          # 通用评分逻辑
│   │   ├── tier_levels.py      # 等级判定
│   │   └── specialized/        # PDE专用指标
│   │       ├── elliptic.py     # 椭圆型指标
│   │       ├── parabolic.py    # 抛物型指标
│   │       └── hyperbolic.py   # 双曲型指标
│   │
│   ├── oracle/                 # Oracle系统（Ground Truth生成）
│   │   ├── core/              # generate/solve/evaluate
│   │   └── solvers/           # PDE求解器
│   │
│   ├── sandbox/               # 执行沙箱
│   │   └── executor.py        # 隔离执行Agent代码
│   │
│   └── evaluation/            # 网格无关验证器
│       └── validator.py       # 插值 + 误差计算
│
├── cases/                  # 测试案例（11个）
│   ├── poisson_simple/
│   │   ├── config.json        # 案例配置
│   │   └── description.md     # 问题描述（给Agent看）
│   │ 
│   ├── heat_simple/
│   └── ...
│
├── scripts/                   # 工具脚本
│   ├── run_evaluation.py      # 🆕 统一评测入口
│   ├── generate_leaderboard_v2.py  # 排行榜生成器
│   └── build_cases.py         # 从Oracle生成cases
│
├── results/                   # 评测结果
│   ├── gemini-2.5/
│   │   ├── poisson_simple/solver.py
│   │   └── evaluation.json
│   └── gpt-4/
│       └── ...
│
└── README.md                  # 本文件
```

## 🎯 评测模式

### Mode 1: Fix Accuracy（固定精度，比速度）

目标：在满足精度要求的前提下，越快越好。

**评分公式：**
```
score = 100 × min(time_budget / runtime, 1.0)
```

**满分条件：** 既快又准

### Mode 2: Fix Time（固定时间，比精度）

目标：在时间预算内，精度越高越好。

**评分公式：**
```
score = 100 × max(0, 1 - error / target_error)
```

**满分条件：** 误差接近零

### 三级难度（Tier Levels）

每个case有3个难度等级：
- **Level 1 (Easy)**: 基础目标
- **Level 2 (Medium)**: 标准目标
- **Level 3 (Hard)**: 困难目标

## 📊 PDE类型与专用指标

### Elliptic（椭圆型）
- **典型问题**: Poisson、Helmholtz
- **专用指标**: DOF/s效率、求解器迭代次数、条件数估计

### Parabolic（抛物型）
- **典型问题**: Heat equation、扩散方程
- **专用指标**: WorkRate、能量衰减率、CFL数、最大值原理验证

### Hyperbolic（双曲型）
- **典型问题**: 波动方程、对流方程
- **专用指标**: 激波分辨率、TV范数、振荡检测

## 🔧 高级用法

### 评测特定cases

```bash
python scripts/run_evaluation.py \
    --agent-name my_agent \
    --agent-dir results/my_agent \
    --cases poisson_simple heat_simple \
    --output results/my_agent/partial.json
```

### 仅运行accuracy模式

```bash
python scripts/run_evaluation.py \
    --agent-name my_agent \
    --agent-dir results/my_agent \
    --modes fix_accuracy \
    --output results/my_agent/accuracy_only.json
```

### 自定义超时

```bash
python scripts/run_evaluation.py \
    --agent-name my_agent \
    --agent-dir results/my_agent \
    --timeout 600 \
    --output results/my_agent/evaluation.json
```

## 🧪 Oracle系统（开发者）

Oracle系统用于生成Ground Truth和创建新的测试案例：

### 生成新的测试案例

```bash
# 从data/benchmark.jsonl生成完整的cases/目录
python scripts/build_cases.py \
    --data data/benchmark.jsonl \
    --output cases \
    --skip-oracle  # 快速模式（使用默认baseline）

# 或运行完整Oracle（较慢但更准确）
python scripts/build_cases.py \
    --data data/benchmark.jsonl \
    --output cases
```

### 预生成Oracle缓存

为了加速评测，建议预先生成所有Oracle缓存：

```bash
# 对每个case运行一次Oracle
for case_dir in cases/*/; do
    case_id=$(basename "$case_dir")
    python -m pdebench.cli run "cases/$case_id/config.json" \
        --outdir "cases/$case_id/oracle_cache"
done
```

## 📈 评测结果格式

### 单case结果 (`test_output/result_fix_accuracy.json`)

```json
{
  "case_id": "poisson_simple",
  "test_mode": "fix_accuracy",
  "status": "PASSED",
  "runtime_sec": 2.345,
  "error": 0.000123,
  "score": 85.3,
  "tier_levels": {
    "passed": [1, 2],
    "total": 3
  },
  "specialized_metrics": {
    "dof": 16384,
    "efficiency_dof_per_sec": 6985.4,
    "linear_iterations_mean": 45.2
  }
}
```

### 批量评测结果 (`evaluation.json`)

```json
{
  "agent_name": "gemini-2.5",
  "evaluation_date": "2025-12-27T...",
  "summary": {
    "fix_accuracy": {
      "total_cases": 11,
      "passed": 10,
      "pass_rate": 0.909,
      "avg_score": 76.5
    },
    "fix_time": {...}
  },
  "results": {...}
}
```

## 🌟 架构特点

### 文件协议（File Protocol）

借鉴 SWE-bench 设计，Agent 和评测系统通过标准化文件交互：
- ✅ **无需适配器**：任何能生成规范 `solver.py` 的系统都可使用
- ✅ **Agent解耦**：支持 OpenHands、SWE-agent、MiniMax、LLM直接输出等
- ✅ **可复现**：所有结果保存为结构化JSON

### 模块化设计

- **`harness/`**: 流程编排，不关心PDE细节
- **`metrics/`**: 纯函数式评分，易于测试
- **`specialized/`**: PDE专用逻辑，易于扩展
- **`oracle/`**: Ground Truth生成，与评测解耦


## 🔮 未来工作

- [ ] 添加更多PDE类型（Stokes、Navier-Stokes）
- [ ] 支持3D问题
- [ ] 自适应网格细化
- [ ] Docker完全隔离
- [ ] 多Agent对比分析工具
- [ ] 在线评测平台

## 🎓 研究应用

本系统适用于：
- 评估LLM的科学代码生成能力
- 研究AI Agent在数值计算中的表现
- 构建科学计算训练数据集
- 分析数值稳定性的AI理解能力

## 📄 许可证

本项目用于 AI 科学编程能力评估研究。

## 🙏 致谢

- FEniCSx 团队提供优秀的有限元框架
- PETSc 提供强大的线性代数工具
- SWE-bench 提供评测系统设计灵感

---

**重要更新**：本项目已于2025年12月完成重构，实现统一评测入口，支持任意Agent评测。详见 [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
