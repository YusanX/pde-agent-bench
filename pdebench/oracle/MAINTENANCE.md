维护说明（oracle）
===================

目标
----
本目录用于提供各方程类型的 ground truth 求解器。新增或扩展方程类型时，应保证：
- 统一入口 `OracleSolver` 能正确分发到对应 solver。
- 输出包含 `baseline_error`、`baseline_time`、`reference`（网格解）、`solver_info`、`num_dofs`。
- 兼容 dolfinx 0.10.0 API（`petsc_options_prefix` 必填等）。

目录结构
--------
- `oracle.py`: 统一入口与分发。
- `common.py`: 网格、解析表达式、采样、误差计算等通用工具。
- `<pde>.py`: 每个方程类型一个 solver（如 `poisson.py`、`heat.py`、`stokes.py`、`navier_stokes.py`）。

已实现的求解器
------------
- `poisson.py`: Poisson 方程（标量椭圆问题）
- `darcy.py`: Darcy 流（椭圆压力形式与混合形式 RT×DG）
- `reaction_diffusion.py`: Reaction-Diffusion（稳态/可选瞬态；含非线性反应项，Newton/SNES）
- `heat.py`: 热传导方程（时间依赖抛物问题）
- `convection_diffusion.py`: 对流扩散方程
- `stokes.py`: Stokes 方程（稳态不可压缩流动，线性问题）
- `navier_stokes.py`: Navier-Stokes 方程（稳态不可压缩流动，**非线性问题**）
  - 使用 Newton 迭代求解非线性对流项 (u·∇)u
  - 支持 manufactured solution 和 reference solution
  - Taylor-Hood 混合元（P2/P1 或 P3/P2）保证 inf-sup 稳定性
  - 压力零空间固定（点约束或均值约束）

新增方程类型流程
--------------
1) 新建 solver 文件  
   例如 `reaction_diffusion.py`，实现 `class ReactionDiffusionSolver`，暴露 `solve(case_spec) -> OracleResult`。

2) 在 `oracle.py` 中注册  
   在 `OracleSolver.solve()` 的分发分支里加入：
   - `elif pde_type == "reaction_diffusion": return ReactionDiffusionSolver().solve(case_spec)`

3) 解析解与数值参考策略  
   - 若 `case_spec['pde']['manufactured_solution']` 存在，则使用解析解评估误差。
   - 若无解析解，则使用 `reference_config` 指定更高精度数值解，计算网格误差作为 baseline。

4) 输出约定  
   `OracleResult.reference` 必须是**评测 grid**上的 `u_grid (nx, ny)`，以便后续误差计算与子榜指标。

输入字段规范（case_spec）
------------------------
- `pde.type`: 方程类型（字符串）
- `pde.description`: 描述
- `pde.manufactured_solution`: 可选，解析解（字符串或向量/压力）
- `pde.source_term`: 可选，无解析解时的 f(x) 表达式
- `pde.initial_condition`: 可选，时间依赖问题初值
- `pde.time`: 可选，时间步配置（t0/t_end/dt/scheme）
- `domain`: 域类型（`unit_square` / `unit_cube`）
- `mesh`: 网格（`resolution`、`cell_type`）
- `fem`: 有限元空间（`family`、`degree` 或 `degree_u/degree_p`）
- `bc`: 边界条件（目前使用 `dirichlet.on=all`）
- `output.grid`: 评测 grid（bbox/nx/ny）
- `oracle_solver`: 求解器参数（ksp_type/pc_type/rtol 等）
- `reference_config`: 无解析解时的高精度参考配置

建议约束
--------
- 对解析解使用 `sympy` → UFL 转换，不要用 Python callable 直接传给 `fem.Expression`。
- 时间依赖问题的 Dirichlet 边界应随时间更新。
- 对混合问题（如 Stokes/NS）应处理压力零空间（固定一点或去均值）。
- **非线性问题（如 NS）**应使用 Newton 迭代：
  - 用 Stokes 解作为初始猜测（提高收敛性）
  - 使用 `ufl.derivative()` 自动计算 Jacobian 矩阵
  - 支持 relaxation 参数（线搜索）提高鲁棒性
  - oracle_solver 配置应包含 `rtol`、`atol`、`max_it` 参数
  - manufactured solution 的源项需包含非线性项（手动推导）

调试建议
--------
- 先在单 case 上验证 `oracle` 输出 `reference` 形状正确。
- 运行 `run_benchmark.py --cases <id>` 快速验证。
