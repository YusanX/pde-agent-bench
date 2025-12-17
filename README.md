# PDEBench: AI-Driven Scientific Coding Benchmark

**世界首个评估大型语言模型端到端科学建模与仿真能力的基准测试系统。**

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.6.0+-orange.svg)]()

## 🎯 项目愿景

PDEBench 不是传统的求解器性能测试，而是评估 AI Agent 是否能"像计算物理学家一样思考"：

- **从物理到代码**：给定自然语言描述的 PDE 问题，Agent 需生成完整的 FEniCSx 求解代码
- **数值稳定性意识**：高 Péclet 数对流扩散问题需要 SUPG 稳定化，Agent 能否识别？
- **网格无关验证**：Agent 和 Oracle 可能使用不同网格，系统通过插值进行公平比较
- **Pareto 前沿评估**：同时考虑精度和计算成本，支持多分辨率测试

## 📊 任务层级

### Level 2.1: 基础（线性对称）
- **Poisson 方程**：`-∇·(κ ∇u) = f`
- **Heat 方程**：`∂u/∂t - ∇·(κ ∇u) = f`（向后 Euler）
- **测试点**：基本语法、边界条件、时间离散

### Level 2.2: 稳定性（数值挑战）
- **对流扩散方程**：高 Péclet 数需要 SUPG
- **Stokes 方程**：混合元素空间（未来）
- **测试点**：数值稳定性意识、预条件器选择

### Level 2.3: 复杂（非线性瞬态）
- **Navier-Stokes 方程**：算子分裂、收敛处理（未来）

## 🚀 快速开始

### 1. 安装

```bash
# 创建 conda 环境并安装 FEniCSx
conda create -n pdebench python=3.10
conda activate pdebench
conda install -c conda-forge fenics-dolfinx mpich petsc4py

# 安装 PDEBench
cd pdebench
pip install -e ".[dev]"
```

### 2. 生成数据集

从 Oracle 案例生成基准测试数据集：

```bash
# 生成 Level 2.1（基础）数据集
python scripts/build_dataset.py \
    --cases-dir cases/demo \
    --output datasets/level_2_1_basic.jsonl \
    --filter-level "2.1"

# 生成 Level 2.2（稳定性）数据集
python scripts/build_dataset.py \
    --output datasets/level_2_2_stability.jsonl \
    --filter-level "2.2"

# 生成完整数据集
python scripts/build_dataset.py \
    --output datasets/full_benchmark.jsonl
```

### 3. 使用 Mock Agent 测试系统

```bash
# 使用 Mock Agent（使用 Oracle 求解器）验证评估流程
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --mock-agent \
    --outdir results/mock_test \
    --limit 3
```

预期输出：
```
[1/3] Case: poisson_simple
============================================================
Evaluating: poisson_simple
============================================================
  Agent execution: ✓ Success
  Wall time: 1.23s
  Validation: ✓ Pass
  rel_L2_grid=4.718e-14 ≤ target=1.000e-02
```

### 4. 评估真实 Agent

```bash
# 使用你的 Agent 生成的求解器脚本
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --agent-script path/to/your_agent_solver.py \
    --outdir results/agent_run_001
```

## 📁 项目结构（重构后）

```
pdebench/
├── pdebench/                    # Python 包
│   ├── oracle/                  # Oracle 系统（生成 Ground Truth）
│   │   ├── core/               # generate/solve/evaluate
│   │   ├── solvers/            # PDE 求解器（poisson/heat/convdiff）
│   │   └── linsolve/           # 线性求解器（baseline/reference）
│   │
│   ├── datasets/               # 数据集模块
│   │   └── schema.py           # JSONL 数据集格式定义
│   │
│   ├── sandbox/                # 执行沙箱
│   │   └── executor.py         # 隔离执行 Agent 代码
│   │
│   ├── evaluation/             # 网格无关验证器
│   │   └── validator.py        # 插值 + 误差计算
│   │
│   ├── cli.py                  # Oracle CLI（旧接口）
│   └── benchmark_cli.py        # Benchmark CLI（新接口）
│
├── datasets/                   # 生成的 JSONL 数据集
│   ├── level_2_1_basic.jsonl
│   ├── level_2_2_stability.jsonl
│   └── full_benchmark.jsonl
│
├── cases/                      # Oracle 案例配置
│   ├── demo/                   # 14 个预配置案例
│   └── schema.case.json
│
├── scripts/                    # 工具脚本
│   ├── build_dataset.py        # 数据集生成器
│   ├── evaluate_agent.py       # 完整评估流程
│   └── make_demo_cases.py      # 生成 demo cases
│
└── tests/                      # 测试套件
```

## 🔬 数据集格式

每个数据集条目是一个 JSON 对象（存储为 JSONL）：

```json
{
  "id": "poisson_simple",
  "level": "2.1",
  "prompt": "Solve the Poisson equation on a unit square...\n\n**Requirements:**\n1. Use dolfinx...",
  "requirements": [
    "Use dolfinx (FEniCSx) for FEM implementation",
    "Accept CLI arguments: --resolution N, --degree P",
    "Save solution to solution.npz with fields: x, y, u",
    "Save metadata to meta.json"
  ],
  "oracle_config": { ... },  // 用于生成 Ground Truth（对 Agent 隐藏）
  "evaluation_config": {
    "target_metric": "rel_L2_grid",
    "target_error": 0.01,
    "timeout_sec": 300,
    "memory_limit_mb": 4096
  }
}
```

## 🎓 Agent 接口规范

Agent 生成的脚本必须：

### 命令行接口
```bash
python agent_solver.py --resolution N --degree P --outdir OUTPUT_DIR
```

### 输出文件

**solution.npz**（必需）：
```python
{
    'x': np.ndarray,  # 1D 数组，x 坐标
    'y': np.ndarray,  # 1D 数组，y 坐标
    'u': np.ndarray,  # 2D 数组，解场 (ny, nx)
}
```

**meta.json**（必需）：
```json
{
  "wall_time_sec": 1.23,
  "solver_info": {
    "ksp_type": "cg",
    "pc_type": "jacobi",
    "iters": 42
  }
}
```

## 📈 评估指标

### 精度指标
- **rel_L2_error**：相对 L2 误差（与 Oracle 参考解比较）
- **rel_Linf_error**：相对 L∞ 误差
- **abs_L2_error**：绝对 L2 误差

### 达标判定
每个案例定义目标阈值，例如：
```json
{
  "target_metric": "rel_L2_grid",
  "target_error": 0.01
}
```

Agent 解必须满足 `rel_L2_error ≤ 0.01` 才算通过。

## 🔧 使用场景

### 场景 1：测试新的 LLM Agent

```bash
# 1. 让 Agent 读取问题描述
cat datasets/level_2_1_basic.jsonl | jq -r '.prompt' | head -1

# 2. Agent 生成求解器代码
# your_agent.py -> outputs my_solver.py

# 3. 评估 Agent 性能
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --agent-script my_solver.py \
    --outdir results/llm_gpt4
```

### 场景 2：生成训练数据

```bash
# 从现有案例生成提示-代码对
python scripts/build_dataset.py --output datasets/training.jsonl

# 使用 Oracle 代码作为标准答案
# (oracle_config 中包含完整的求解器配置)
```

### 场景 3：Curriculum Learning

```bash
# 从简单到困难逐步训练
python scripts/evaluate_agent.py --dataset datasets/level_2_1_basic.jsonl ...
python scripts/evaluate_agent.py --dataset datasets/level_2_2_stability.jsonl ...
```

## 🧪 Oracle 系统（仅用于生成参考解）

Oracle 系统保留了原有功能，用于生成 Ground Truth：

```bash
# 生成 Oracle 案例
python scripts/make_demo_cases.py

# 运行单个案例（使用 Oracle CLI）
python -m pdebench.cli run cases/demo/poisson_simple.json \
    --outdir artifacts/poisson_simple
```

**重要**：Agent 代码不应导入 `pdebench.oracle` 模块。

## 📊 评估报告

运行评估后，生成 `summary.json`：

```json
{
  "summary": {
    "total_cases": 11,
    "successful_cases": 10,
    "failed_cases": 1,
    "success_rate": 0.909
  },
  "accuracy_statistics": {
    "avg_rel_L2_error": 0.0023,
    "min_rel_L2_error": 4.718e-14,
    "max_rel_L2_error": 0.0089
  },
  "cases": [...]
}
```

## 🌟 为什么选择 PDEBench？

| 特性 | PDEBench | 传统 PDE Benchmark |
|------|----------|-------------------|
| **评估对象** | AI Agent 代码生成 | 求解器性能 |
| **输入** | 自然语言描述 | 已有代码 |
| **输出** | 完整求解脚本 | 数值解 |
| **难点** | 数值稳定性意识 | 计算效率 |
| **验证** | 网格无关插值 | 固定网格 |

## 🔮 未来工作

- [ ] 添加 Stokes 方程（混合元素空间）
- [ ] 实现 Navier-Stokes（非线性迭代）
- [ ] 支持自适应网格细化
- [ ] 添加逆问题（参数估计）
- [ ] 集成 Docker 沙箱（完全隔离）
- [ ] Pareto 前沿可视化工具

## 📄 许可证

本项目用于 AI 科学编程能力评估研究。

## 🙏 致谢

- FEniCSx 团队提供优秀的有限元框架
- PETSc 提供强大的线性代数工具

## 📞 联系方式

如有问题或建议，欢迎提交 Issue。

---

**重要提示**：本项目已完成从"求解器自动调优"到"物理到代码生成"的战略转型（2024年12月）。
