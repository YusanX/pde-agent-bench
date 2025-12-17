# PDEBench

最小可运行的 PDE benchmark 系统，基于 FEniCSx/dolfinx 实现。

## 项目状态

✅ **所有测试通过**：20/20 tests passed (10 validation + 10 smoke tests)

## 功能特性

- **支持的 PDE**：
  - Poisson 方程：`-div(κ ∇u) = f`
  - Heat 方程：`∂u/∂t - div(κ ∇u) = f`（backward Euler）

- **完整的工作流**：
  1. **generate**：构建离散系统、生成参考解（direct LU）
  2. **solve**：使用 baseline Krylov 求解器求解
  3. **evaluate**：计算精度、残差、代价等多维度指标
  4. **run**：一键执行完整流程

- **灵活的配置**：JSON schema 定义的 case 格式，支持 manufactured solution 自动推导源项

## 安装

### 前置依赖

首先需要通过 conda 安装 FEniCSx (dolfinx >= 0.6.0)：

```bash
conda create -n pdebench python=3.10
conda activate pdebench
conda install -c conda-forge fenics-dolfinx mpich petsc4py
```

### 安装 PDEBench

然后在 pdebench 环境中安装本包：

```bash
cd pdebench
pip install -e .
```

或安装开发依赖：

```bash
pip install -e ".[dev]"
```

## 快速开始

### 1. 生成 demo cases

```bash
python pdebench/scripts/make_demo_cases.py
```

这将在 `cases/demo/` 目录下生成 10 个预配置的 demo case：
- **Poisson 方程** (5个): `poisson_simple`, `poisson_p2`, `poisson_quad`, `poisson_varied`, `poisson_grid_target`
- **Heat 方程** (5个): `heat_simple`, `heat_longer`, `heat_p2`, `heat_quad`, `heat_grid_target`

### 2. 运行单个 case

```bash
python -m pdebench.cli run cases/demo/poisson_simple.json --outdir artifacts/poisson_simple
```

### 3. 分步执行

```bash
# 仅生成参考解
python -m pdebench.cli generate cases/demo/poisson_simple.json --outdir artifacts/poisson_simple

# 使用 baseline solver 求解
python -m pdebench.cli solve cases/demo/poisson_simple.json --outdir artifacts/poisson_simple

# 评估结果
python -m pdebench.cli evaluate cases/demo/poisson_simple.json --outdir artifacts/poisson_simple
```

### 4. 自定义求解器参数

```bash
python -m pdebench.cli run cases/demo/poisson_simple.json --outdir artifacts/test \
    --ksp-type gmres --pc-type ilu --ksp-rtol 1e-12
```

## Case 格式说明

每个 case 是一个 JSON 文件，完整的 schema 定义见 `cases/schema.case.json`。

### 主要字段

- **id**: case 唯一标识符
- **pde**: PDE 定义
  - `type`: `"poisson"` 或 `"heat"`
  - `coefficients.kappa`: 扩散系数（目前支持 constant）
  - `manufactured_solution.u`: 精确解表达式（用于自动推导 f 和边界条件）
  - `source_term.f`: 源项（若未提供且有 manufactured_solution，则自动推导）
  - `time`: 时间参数（仅 heat 方程）
- **domain**: 区域类型（目前仅 `"unit_square"`）
- **mesh**: 网格参数
  - `resolution`: 网格分辨率
  - `cell_type`: `"triangle"` 或 `"quadrilateral"`
- **fem**: 有限元设置
  - `family`: `"Lagrange"`
  - `degree`: 多项式阶数（1 或 2）
- **bc**: 边界条件
  - `dirichlet.on`: `"all"` 表示所有边界
  - `dirichlet.value`: 边界值表达式（可为 `"u"` 表示使用 manufactured solution）
- **targets**: 达标要求
  - `target_error`: 目标误差阈值
  - `metric`: 达标依据的指标（`rel_L2_grid`, `rel_L2_fe`, `rel_H1_semi_fe`）
- **expose_parameters**: 需要在输出中报告的参数列表
- **output**: 输出配置
  - `format`: `"npz"`
  - `grid`: 采样网格配置（bbox, nx, ny）

### Manufactured Solution 示例

```json
{
  "pde": {
    "type": "poisson",
    "manufactured_solution": {
      "u": "sin(pi*x)*sin(pi*y)"
    },
    "coefficients": {
      "kappa": {"type": "constant", "value": 1.0}
    }
  },
  "bc": {
    "dirichlet": {"on": "all", "value": "u"}
  }
}
```

系统会自动计算 `f = -div(kappa * grad(u))`，并使用精确解作为边界条件。

对于 Heat 方程，使用含时间的表达式：

```json
{
  "manufactured_solution": {
    "u": "exp(-t)*sin(pi*x)*sin(pi*y)"
  }
}
```

## 输出文件

每次运行会在 `--outdir` 生成以下文件：

### 主要输出

- **solution.npz**: 求解结果
  - `x`: x 坐标数组 (nx,)
  - `y`: y 坐标数组 (ny,)
  - `u`: 解场 (ny, nx)
  
- **meta.json**: 求解元数据
  - `wall_time_sec`: 求解耗时
  - `peak_rss_mb`: 峰值内存
  - `solver_info`: 求解器信息（ksp_type, pc_type, rtol, iters）
  - `exposed_parameters`: case 中指定的暴露参数

- **metrics.json**: 评估指标
  - `validity`: 达标情况（pass/fail + reason）
  - `rel_L2_grid`: 网格上的相对 L2 误差（与 reference 比较）
  - `rel_L2_fe`: FE 空间的相对 L2 误差（与 exact 比较，如有）
  - `rel_H1_semi_fe`: FE 空间的相对 H1 半范数误差（与 exact 比较，如有）
  - `rel_res`: 相对残差 `||b - Au||/||b||`
  - `rel_lin_err_M`: 相对线性误差 `||u - u_star||_M / ||u_star||_M`
  - `cost`: 代价指标（wall_time_sec, iters）

### 中间文件

- **reference.npz**: 参考解（direct LU）
- **exact.npz**: 精确解（如有 manufactured solution）
- **system_A.dat, system_b.dat**: 离散线性系统（PETSc 格式）
- **reference_u_star.dat, solution_u.dat**: FE 向量（PETSc 格式）
- **problem_info.json, generate_meta.json**: 中间元数据

## 指标解释

### 精度指标

- **rel_L2_grid**: 在固定观测网格上的离散 L2 相对误差。与 reference solution（同网格、同 FE、direct LU）比较。
- **rel_L2_fe**: FE 空间中的 L2 相对误差（通过 `assemble_scalar(inner(e,e)*dx)` 计算）。与 exact solution 比较（需要 manufactured solution）。
- **rel_H1_semi_fe**: FE 空间中的 H1 半范数相对误差（`assemble_scalar(inner(grad(e),grad(e))*dx)`）。与 exact solution 比较。

### 离散/线性指标

- **rel_res**: 相对残差范数 `||b - A*u|| / ||b||`，衡量离散方程的满足程度。
- **rel_lin_err_M**: 相对线性误差 `||u - u_star||_M / ||u_star||_M`，其中 `u_star` 是 direct LU 得到的 reference solution，`||·||_M` 为 M-范数（L2 内积）。

### 代价指标

- **wall_time_sec**: 求解耗时（秒）
- **iters**: Krylov 迭代次数

## 测试

运行所有测试：

```bash
cd /path/to/pdebench
pytest -q
```

预期输出：
```
....................                                          [100%]
20 passed in 5.81s
```

测试包括：
- **test_case_validation.py**: 验证所有 demo case 符合 JSON schema (10 tests)
- **test_smoke_demo.py**: 对每个 demo case 执行完整流程并检查达标情况 (10 tests)

运行特定测试：

```bash
# 只运行 validation 测试
pytest tests/test_case_validation.py -v

# 只测试单个 case
pytest tests/test_smoke_demo.py::test_run_demo_case[poisson_simple] -v
```

## 扩展到更多 PDE

要添加新的 PDE 类型：

1. 在 `pdebench/solvers/` 下创建新的求解器模块（参考 `poisson.py` 和 `heat.py`）
2. 实现以下函数：
   - `setup_<pde>_problem(msh, V, case_spec, ...)` → 返回 `(A, b, bcs, u_exact_func)`
   - `solve_<pde>(...)` → 调用线性求解器
3. 在 `pdebench/core/generate.py`, `solve.py` 中添加对应的分支
4. 更新 `cases/schema.case.json` 的 `pde.type` 枚举
5. 创建 demo cases 并添加测试

## 求解器替换

Baseline 求解器实现在 `pdebench/linsolve/baseline.py`：

```python
def solve_linear(A, b, ksp_params) -> (x, info)
```

可以通过以下方式替换：
- 修改 `baseline.py` 实现不同的求解策略
- 通过 CLI 参数覆盖 KSP 设置（`--ksp-type`, `--pc-type`, `--ksp-rtol`）
- 在 case JSON 中预设默认参数

Reference 求解器（direct LU）实现在 `solve_linear_direct()`，不应被修改以保证一致的参考基准。

## 依赖

- **FEniCSx (dolfinx)**: 有限元框架（通过 conda 安装）
- **PETSc**: 线性代数和求解器（通过 conda 安装）
- **numpy, scipy**: 数值计算
- **sympy**: 符号计算（用于 manufactured solution 推导）
- **jsonschema**: case 格式校验
- **psutil**: 内存监控
- **pytest**: 测试框架

## 项目结构

```
pdebench/
├── pdebench/              # Python 包
│   ├── cli.py            # CLI 入口
│   ├── core/             # 核心流程（generate/solve/evaluate）
│   ├── solvers/          # PDE 求解器（poisson/heat）
│   └── linsolve/         # 线性求解器（baseline/reference）
├── cases/                # Case 定义
│   ├── schema.case.json  # JSON Schema
│   └── demo/             # 10 个 demo cases
├── scripts/              # 辅助脚本
│   └── make_demo_cases.py
├── tests/                # 测试套件
│   ├── test_case_validation.py
│   └── test_smoke_demo.py
├── pyproject.toml        # 包配置
└── README.md             # 本文档
```

## 注意事项

1. **环境要求**：必须在 conda 环境中安装 dolfinx，不能通过 pip 安装
2. **精度设置**：Heat 方程由于时间离散误差，精度目标需要设置得比 Poisson 方程宽松
3. **网格分辨率**：demo cases 使用较小的分辨率以保证测试快速完成
4. **Krylov 收敛**：默认使用 CG + Jacobi，对于复杂问题可能需要调整求解器参数
5. **参考解生成**：reference solution 使用 direct LU 求解，与 baseline Krylov 使用相同的离散系统

## 许可

本项目仅用于 benchmark 演示。

