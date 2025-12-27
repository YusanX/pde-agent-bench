"""
动态 Prompt 生成器

根据 case 配置动态生成给 Agent 看的问题描述
"""

from typing import Dict, Any


# 方程类型模板
EQUATION_TEMPLATES = {
    "poisson": {
        "title": "Poisson equation",
        "equation": "-∇·(κ ∇u) = f   in Ω\n  u = g           on ∂Ω",
        "description": "This is an elliptic boundary value problem."
    },
    "heat": {
        "title": "transient Heat equation",
        "equation": "∂u/∂t - ∇·(κ ∇u) = f   in Ω × (0, T]\n  u = g                    on ∂Ω × (0, T]\n  u(x, y, 0) = u₀(x, y)    in Ω",
        "description": "This is a parabolic evolution problem requiring time-stepping."
    },
    "convection_diffusion": {
        "title": "Convection-Diffusion equation",
        "equation": "-ε ∇²u + β·∇u = f   in Ω\n  u = g                on ∂Ω",
        "description": "This is a convection-diffusion problem that may require stabilization."
    }
}


# Domain 模板
DOMAIN_TEMPLATES = {
    "unit_square": "[0,1]×[0,1]"
}


def format_coefficient(coeff: Dict) -> str:
    """格式化系数配置"""
    coeff_type = coeff.get('type', 'constant')
    
    if coeff_type == 'constant':
        return f"{coeff['value']}"
    elif coeff_type == 'piecewise_x':
        left = coeff['left']
        right = coeff['right']
        x_split = coeff.get('x_split', 0.5)
        return f"{left} (x < {x_split}), {right} (x ≥ {x_split})"
    elif coeff_type == 'expr':
        return coeff['expr']
    else:
        return str(coeff)


def generate_peclet_warning(pde_config: Dict) -> str:
    """生成对流扩散方程的稳定性警告"""
    if 'pde_params' in pde_config:
        params = pde_config['pde_params']
        epsilon = params.get('epsilon', 1.0)
        beta = params.get('beta', [1.0, 1.0])
        
        if isinstance(beta, list):
            beta_norm = (beta[0]**2 + beta[1]**2)**0.5
        else:
            beta_norm = beta
            
        # 估算 Péclet 数 (假设特征长度 L=1)
        peclet = beta_norm / epsilon if epsilon > 0 else float('inf')
        
        if peclet > 10:
            return f"""
**Physical Context:**
This is a convection-dominated problem (Péclet number Pe ≈ {peclet:.1f}).
⚠️ Standard Galerkin may produce oscillations. Consider stabilization techniques (SUPG, streamline diffusion, or upwinding).
"""
        else:
            return f"""
**Physical Context:**
This is a balanced convection-diffusion problem (Péclet number Pe ≈ {peclet:.1f}).
Standard Galerkin should be adequate.
"""
    return ""


def generate_time_discretization(pde_config: Dict) -> str:
    """生成时间离散描述"""
    if 'time' not in pde_config:
        return ""
    
    time_cfg = pde_config['time']
    t_end = time_cfg.get('t_end', 1.0)
    dt = time_cfg.get('dt', 0.01)
    scheme = time_cfg.get('scheme', 'backward_euler')
    
    scheme_name = {
        'backward_euler': 'backward Euler scheme',
        'crank_nicolson': 'Crank-Nicolson scheme',
        'bdf2': 'BDF2 scheme'
    }.get(scheme, scheme)
    
    return f"""
**Time Discretization:**
- Final time T = {t_end}
- Time step Δt = {dt}
- Use {scheme_name}
"""


def generate_prompt(config: Dict[str, Any], target_error: float = None, timeout_sec: int = 300) -> str:
    """
    动态生成 Agent 任务描述
    
    Args:
        config: Case 配置（从 benchmark.jsonl 读取）
        target_error: 目标误差（由 build 脚本计算）
        timeout_sec: 超时限制
    """
    
    case_id = config['id']
    equation_type = config['pde_classification']['equation_type']
    oracle_config = config['oracle_config']
    pde_config = oracle_config['pde']
    domain_config = oracle_config['domain']
    output_config = oracle_config['output']
    
    # 获取方程模板
    eq_template = EQUATION_TEMPLATES.get(equation_type, EQUATION_TEMPLATES['poisson'])
    domain_desc = DOMAIN_TEMPLATES.get(domain_config['type'], 'custom domain')
    
    # 构建 Prompt
    prompt = f"""# Case: {case_id}

## Problem Description

Solve the {eq_template['title']} on a unit square domain {domain_desc}:

  {eq_template['equation']}

{eq_template['description']}

**Problem Parameters:**
- Manufactured solution: u = {pde_config.get('manufactured_solution', {}).get('u', 'provided in config')}
- Source term f and boundary data g are derived from the manufactured solution
"""
    
    # 添加系数描述
    if 'coefficients' in pde_config:
        for coeff_name, coeff_data in pde_config['coefficients'].items():
            prompt += f"- {coeff_name.capitalize()} coefficient: κ = {format_coefficient(coeff_data)}\n"
    
    # 添加时间离散（如果是时间相关）
    time_desc = generate_time_discretization(pde_config)
    if time_desc:
        prompt += time_desc
    
    # 添加对流扩散警告
    if equation_type == 'convection_diffusion':
        prompt += generate_peclet_warning(pde_config)
    
    # 边界条件
    bc_config = oracle_config.get('bc', {})
    if 'dirichlet' in bc_config:
        prompt += f"""
**Boundary Conditions:**
- Dirichlet BC on all boundaries: u = u_exact (from manufactured solution)
"""
    
    # 输出网格
    grid = output_config.get('grid', {})
    nx = grid.get('nx', 50)
    ny = grid.get('ny', 50)
    
    prompt += f"""
**Requirements:**
Your implementation must:
1. Use `dolfinx` (FEniCSx) for finite element assembly and solving
2. Accept command-line arguments: `--resolution N` (mesh resolution) and `--degree P` (polynomial degree)
3. Save the solution to `solution.npz` with fields: `x` (1D array), `y` (1D array), `u` (2D array)
4. Save solver metadata to `meta.json` with fields: `wall_time_sec`, `solver_info` (dict with `ksp_type`, `pc_type`, `iters`)

**Output Grid:**
Sample the solution on a uniform {nx}×{ny} grid spanning the domain.

---

## Testing Modes

This case supports two testing modes:

### Mode 1: Fix Accuracy, Optimize Speed
**Goal:** Achieve target error within minimum time.

```bash
python test_fix_accuracy.py --agent-script your_solver.py
```

**Scoring:**
- Must achieve `error` ≤ {target_error if target_error else 'target_error'}
- Score = 100 × (time_budget / your_runtime)
- Faster = Higher score

### Mode 2: Fix Time Budget, Optimize Accuracy
**Goal:** Achieve minimum error within time budget.

```bash
python test_fix_time.py --agent-script your_solver.py
```

**Scoring:**
- Must finish within time_budget
- Score = 100 × max(0, 1 - error/target_error)
- Lower error = Higher score

---

## Target Metrics

- Target Error: {target_error if target_error else 'To be determined by Oracle baseline'}
- Timeout: {timeout_sec} seconds
"""
    
    return prompt


def generate_description_md(config: Dict[str, Any], target_error: float, 
                            difficulty_tiers: Dict[str, Any]) -> str:
    """
    生成完整的 description.md 文件
    
    包含 prompt + 难度分级信息
    """
    
    prompt = generate_prompt(config, target_error)
    
    # 添加难度分级说明
    accuracy_tiers = difficulty_tiers.get('accuracy', {})
    speed_tiers = difficulty_tiers.get('speed', {})
    
    prompt += f"""
---

## Difficulty Tiers

This case provides multiple difficulty levels for comprehensive analysis:

### Accuracy Levels (Fix Time Budget, Optimize Accuracy)
- **Level 1 (Easy)**: Target error ≤ {accuracy_tiers.get('level_1', {}).get('target_error', 'N/A'):.2e}
- **Level 2 (Medium)**: Target error ≤ {accuracy_tiers.get('level_2', {}).get('target_error', 'N/A'):.2e} (Oracle baseline)
- **Level 3 (Hard)**: Target error ≤ {accuracy_tiers.get('level_3', {}).get('target_error', 'N/A'):.2e}

### Speed Levels (Fix Accuracy, Optimize Speed)
- **Fast (Real-time)**: Time budget = {speed_tiers.get('fast', {}).get('time_budget', 'N/A'):.3f}s (10× faster than Oracle)
- **Medium (Interactive)**: Time budget = {speed_tiers.get('medium', {}).get('time_budget', 'N/A'):.3f}s (Oracle baseline)
- **Slow (Batch)**: Time budget = {speed_tiers.get('slow', {}).get('time_budget', 'N/A'):.3f}s (10× slower than Oracle)

Your solver will be evaluated against the medium level by default, but achieving harder/faster levels demonstrates superior numerical methods.
"""
    
    return prompt

