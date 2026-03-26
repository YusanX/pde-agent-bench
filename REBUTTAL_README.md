# PDEBench Rebuttal — P2 API Decoupling Experiment

**目的**：回应 ICML 2026 四位审稿人对"benchmark 混淆 DOLFINx API 熟练度与数值推理能力"的共同质疑（K4KR Q1、eVPk W2、iw9Y、3Q3e）。

eVPk 在 Key Questions Q2 中明确提出：
> "Can you disentangle DOLFINx API proficiency from numerical reasoning? For example, by providing a correct DOLFINx template that handles mesh creation, function space setup, and solver invocation, and asking models to fill in only the numerical parameters."

本次改动实现了这一对照实验。

---

## 实验设计

### 两种 Prompt 变体（Condition A vs B）

| | Condition A — `standard` | Condition B — `template_guided` |
|---|---|---|
| **模型任务** | 从零写完整 DOLFINx solver | 只填写数学内容 + 参数选择 |
| **API 部分** | 模型自己写（mesh、FunctionSpace、采样、BC setup 等） | 全部预写好，模型不得修改 |
| **数学部分** | 模型自己推导 | 模型必须填写（`a = ...`, `L = ...`） |
| **数值参数** | 模型自己选，无提示 | 模型自己选，无默认值（`N = None`） |
| **PDE 数据** | 从 prompt 文字描述推断 | 直接内联在代码中（kappa、制造解等） |

### 核心结论逻辑

- 若 **template_guided pass-rate ≈ standard pass-rate**：API 不是瓶颈，数值推理才是 → benchmark 测的是数学能力
- 若 **template_guided pass-rate > standard pass-rate**：当前 benchmark 因 API 噪声低估了模型能力 → 结论更保守、对论文更有利
- 两种结果都支持"benchmark 测量的是数值推理"这一核心论点

---

## 新增文件

```
pdebench/
├── pdebench/core/
│   └── template_prompt_builder.py        ← 新增：P2 实验 prompt 生成器
```

---

## 修改文件

### `scripts/run_benchmark.py`

1. **新增 `--prompt-variant` CLI 参数**
   ```bash
   --prompt-variant {standard,template_guided}   # 默认 standard
   ```

2. **`run_single_case()` 接入 variant 逻辑**：根据 PDE 类型自动选择对应 template；若该类型尚无 template，自动 fallback 到 standard 并打印提示。

3. **所有结果 JSON 新增 `prompt_variant` 字段**：方便后续 `analyze_p2_results.py` 对比分析。

---

## Template 设计原则（`template_prompt_builder.py`）

### 已支持 PDE 类型

目前完整实现 **Poisson** 和 **Heat**（rebuttal 主力），其余 8 种（convection-diffusion、Stokes、Navier-Stokes、Helmholtz、biharmonic、linear elasticity、Darcy、reaction-diffusion）已有骨架，供 NeurIPS 版本使用。

### Poisson / Heat 的关键设计决策

**1. 数值参数不给默认值**

```python
# ❌ 旧版（有锚定）
N        = 64       # mesh resolution
degree   = 1        # FE polynomial degree

# ✅ 新版（无锚定）
N        = None  # ← choose mesh resolution (integer)
degree   = None  # ← choose FE polynomial degree: 1, 2, or 3
```

模型必须主动选择，否则代码报错（`None` 不能传给整数参数）。这样测量的才是模型的数值判断能力。

**2. PDE 数学数据在 prompt 生成时内联，不暴露 oracle_config 结构**

```python
# ❌ 旧版（暴露 oracle 路径，存在参数泄露风险）
pde_cfg    = case_spec["oracle_config"]["pde"]       # 可顺手读 oracle_config.mesh.resolution
kappa_spec = pde_cfg["coefficients"]["kappa"]

# ✅ 新版（prompt 生成时直接内联，模型运行时不访问 case_spec）
kappa   = _kappa_from_spec(msh, {'type': 'constant', 'value': 1.0})    # 直接写死
f_h, bc = _manufactured_f_and_bc(
    msh, V,
    {'manufactured_solution': {'u': 'sin(pi*x)*sin(pi*y)'}},           # 精确解字符串
    {'type': 'constant', 'value': 1.0}
)
```

避免模型通过 `case_spec["oracle_config"]["mesh"]["resolution"]`（= 120）或 `case_spec["oracle_config"]["fem"]["degree"]`（= 1）偷看 oracle 的数值参数选择。

**3. API 样板全部预写，模型只填数学内容**

```python
# 模型只需填这两行：
a = ...   # ← YOUR CODE: bilinear form a(u, v)
L = ...   # ← YOUR CODE: linear form L(v)

# 其余全部 "PROVIDED — do not modify"：
# mesh 创建、FunctionSpace、LinearProblem 调用、输出采样
```

**4. 提供完整 API 工具函数**

Template 内置以下工具（模型可直接调用，无需了解 DOLFINx API 细节）：

| 函数 | 功能 |
|---|---|
| `_sample_scalar(u_h, nx, ny)` | 均匀网格采样（避免模型手写 bb_tree 几何查询） |
| `_sample_vector_magnitude(u_h, nx, ny)` | 向量场模值采样 |
| `_all_boundary_dofs(msh, V)` | 全边界 DOF 定位 |
| `_bc_from_str(msh, V, expr_str)` | 字符串表达式 → DirichletBC |
| `_kappa_from_spec(msh, kappa_spec)` | 系数规格 → UFL 表达式 |
| `_f_from_str(msh, expr_str)` | 字符串 → UFL/fem 源项 |
| `_manufactured_f_and_bc(msh, V, pde_cfg, kappa_spec)` | 自动推导制造解的 f 和 BC |

---

## 运行方法


```bash
cd pdebench

export OPENAI_API_KEY=$your_OPENAI_api_key$ 
# Condition A：standard（对照组，用已有 results2/ 数据即可）
# Condition B：template_guided（实验组）
python scripts/run_benchmark.py \
  --agent gpt-5.4 \
  --equation-types poisson \
  --prompt-variant template_guided \
  --output results_p2/gpt-5.4_template_guided

```


