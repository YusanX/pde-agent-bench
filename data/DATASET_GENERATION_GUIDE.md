# 如何高效生成高质量 PDEBench 数据集（benchmark.jsonl）

本指南面向后续“快速扩充方程类型 + 提升数据质量 + 保持可比性/可复现性”的数据维护工作。

---

## 核心原则（先定规则再加数量）

- **可比性优先**：同一方程类型下，**评测输出 grid 固定**（`oracle_config.output.grid`），避免模型通过选择输出分辨率投机。
- **变化要“正交”**：case 之间的差异要来自“数值难点维度”（系数/源项/边界/时间/离散/稳定化），不要只改网格分辨率凑数量。
- **可解释性**：每个 case 都要有明确“考察点”（例如：高对比系数、长时间积分、边界层、强非对称等）。
- **可复现性**：无解析解 case 必须提供 `reference_config`（更高精度数值参考解的生成方案），确保误差定义稳定。
- **防 prompt 锚定**：`agent_knobs` 只描述“可调参数的作用”，不要给“推荐范围/数字”。

---

## 数据结构速览（每行一个 case）

`benchmark.jsonl` 每一行是一个 JSON 对象（不可换行），建议最少包含：

- **id**：唯一标识（建议带类型前缀）
- **pde_classification**
  - `equation_type`：方程类型（poisson/heat/...）
  - `math_type`：子榜类型数组（例如 `["elliptic"]`、`["parabolic"]`、`["mixed_type"]`、`["incompressible_flow"]`）
- **oracle_config**：oracle/评测所需全部配置（见下文模板）
- **evaluation_config**
  - `target_metric`：目前建议 `rel_L2_grid`
  - `timeout_sec`
  - `tolerance`：pass/fail 容忍倍数（如 1.2）
- **agent_knobs**：提示 agent 应该“选择/优化”的参数（仅描述作用）

---

## 生成高质量 case 的流程（推荐 6 步）

### Step 1：为一个方程类型制定“case 设计矩阵”

以每个 equation type 做到 20 个为例，推荐先定义 4~6 个**变化维度**（每个维度 2~4 个离散水平），然后抽样组合：

- **系数维度**（kappa/nu/epsilon 等）：常数 / 平滑变系数 / 强对比 / 各向异性（可扩展）
- **源项与解特征**：平滑 / 高频 / 局部层 / 近奇异（可扩展）
- **边界/初值**：全 Dirichlet / 混合边界（未来）/ 非零初值（parabolic）
- **时间维度**（parabolic）：短时 / 长时 / 大 dt / 小 dt（注意稳定性）
- **离散维度**：tri vs quad、P1 vs P2（尽量不要单独靠它扩充数量）
- **稳定化维度**（mixed_type 等）：无稳定化 / SUPG / upwind（视方程而定）

**目标**：不同 case 的差异能对应到清晰的“子榜解释”，而不是随机噪声。

### Step 2：决定“解析解 vs 数值参考解”

- **优先用解析解（manufactured solution）**：便于严格定义误差，最稳定。
  - 使用 `pde.manufactured_solution`（如 `{"u": "..."}` 或流体问题的 `{"u": [...], "p": "..."}`）
- **无解析解 case**：必须提供
  - `pde.source_term`（必要时还有 `pde.initial_condition`）
  - `reference_config`（更高精度参考解配置）

### Step 3：写 oracle_config（保持字段一致）

建议模板（可按方程扩展 `pde_params`）：

```json
{
  "pde": {
    "type": "poisson",
    "description": "...",
    "coefficients": {"kappa": {"type": "constant", "value": 1.0}},
    "manufactured_solution": {"u": "sin(pi*x)*sin(pi*y)"},
    "source_term": "1.0",
    "initial_condition": "sin(pi*x)*sin(pi*y)",
    "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
    "pde_params": {}
  },
  "domain": {"type": "unit_square"},
  "mesh": {"resolution": 100, "cell_type": "triangle"},
  "fem": {"family": "Lagrange", "degree": 1},
  "bc": {"dirichlet": {"on": "all", "value": "u"}},
  "output": {"format": "npz", "field": "scalar", "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50}},
  "oracle_solver": {"ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
  "reference_config": {
    "mesh": {"resolution": 200, "cell_type": "triangle"},
    "fem": {"family": "Lagrange", "degree": 2},
    "time": {"dt": 0.005},
    "oracle_solver": {"ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-12}
  }
}
```

说明：
- **有解析解**时：`manufactured_solution` 存在即可，`source_term/reference_config` 可省略。
- **无解析解**时：`source_term` + `reference_config` 必填；parabolic 还需要 `initial_condition`。

### Step 4：写 agent_knobs（只写“作用”，不写“范围/数字”）

推荐写法（避免锚定）：

```json
[
  {"name": "mesh_resolution", "description": "Control spatial discretization fidelity"},
  {"name": "element_degree", "description": "Polynomial degree of the finite element space"},
  {"name": "dt", "description": "Time step size controlling accuracy vs runtime"},
  {"name": "ksp_type", "description": "PETSc Krylov solver choice"},
  {"name": "pc_type", "description": "Preconditioner choice"},
  {"name": "rtol", "description": "Linear solver relative tolerance"}
]
```

### Step 5：质量门禁（新增 case 前必须过）

建议至少做以下“静态检查”（可以后续写脚本自动化）：

- **唯一性**：`id` 不重复
- **完整性**：必须字段存在（见“数据结构速览”）
- **一致性**：
  - `oracle_config.pde.type` 与 `pde_classification.equation_type` 一致
  - `output.grid` 必须给 bbox/nx/ny 且 nx,ny>0
- **无解析解 case**：
  - 必有 `source_term`
  - 必有 `reference_config`（并且分辨率/阶次/容差更严格）
- **表达式可解析**：`manufactured_solution/source_term/initial_condition/kappa expr` 均为 sympy/ufl 可解析表达式（避免使用 Python 逻辑表达式如 `x>0.5`，优先用平滑表达式）

### Step 6：小规模试跑（保证“可运行 + 可判定”）

每新增一个 case，建议至少跑：

- 只跑 oracle：`run_benchmark.py --agent gpt-4o --cases <id> --skip-generation`
- 再跑一个 LLM：`run_benchmark.py --agent gpt-4o --cases <id>`

观察：
- oracle 能否生成 reference（且时间不异常）
- agent 是否能正确返回 `solve()` 与 `solver_info`
- 误差计算是否稳定（无 NaN）
- 子榜指标是否能读到 `solver_info`（例如 elliptic 的 resolution/degree）

---

## “一个方程类型做 20 个”是否合理？

**合理**，但建议满足：
- 20 个 case 分布在多个变化维度上（至少 4 个维度，每个维度 2~4 个水平）
- 同类型 case 不要让某一维度（比如仅 mesh）占比过大
- 解析解与无解析解案例比例建议先以 **70/30** 左右起步（稳定性优先）

---

## 命名建议

- `poisson_basic / poisson_variable_kappa / poisson_high_contrast_no_exact`
- `heat_basic / heat_longer / heat_no_exact / heat_variable_kappa`
- `convdiff_high_pe / convdiff_low_pe / convdiff_no_exact`

命名要直接体现“变化点”（variable_kappa / high_contrast / no_exact / longer 等）。

---

## 常见坑（数据侧）

- 在 `agent_knobs` 里写范围（会锚定模型）
- 用不可解析表达式（例如 `x > 0.5` 这类逻辑表达式）
- 无解析解 case 没有 `reference_config`
- 仅靠“改 mesh 分辨率”扩充数量（信息增益低）

