"""
Prompt生成器 - 从benchmark.jsonl配置生成给LLM的prompt
"""

from typing import Dict, Any, Optional
from pathlib import Path


# 方程类型模板
EQUATION_TEMPLATES = {
    "poisson": {
        "title": "Poisson Equation",
        "equation": "-∇·(κ ∇u) = f   in Ω\n  u = g           on ∂Ω",
        "description": "Elliptic boundary value problem."
    },
    "heat": {
        "title": "Heat Equation (Transient)",
        "equation": "∂u/∂t - ∇·(κ ∇u) = f   in Ω × (0, T]\n  u = g                    on ∂Ω\n  u(x,0) = u₀(x)           in Ω",
        "description": "Parabolic evolution problem requiring time-stepping."
    },
    "convection_diffusion": {
        "title": "Convection-Diffusion Equation",
        "equation": "-ε ∇²u + β·∇u = f   in Ω\n  u = g                on ∂Ω",
        "description": "May require stabilization (SUPG) for high Péclet numbers."
    },
    "stokes": {
        "title": "Stokes Flow (Incompressible)",
        "equation": "-ν ∇²u + ∇p = f   in Ω\n  ∇·u = 0             in Ω\n  u = g               on ∂Ω",
        "description": "Steady incompressible flow; use Taylor-Hood mixed elements."
    },
    "navier_stokes_incompressible": {
        "title": "Navier-Stokes (Incompressible, Steady)",
        "equation": "u·∇u - ν ∇²u + ∇p = f   in Ω\n  ∇·u = 0               in Ω\n  u = g                 on ∂Ω",
        "description": "Nonlinear steady incompressible flow; Newton/Picard is acceptable."
    }
}


def format_coefficient(coeff: Dict) -> str:
    """格式化系数配置"""
    coeff_type = coeff.get('type', 'constant')
    
    if coeff_type == 'constant':
        return str(coeff['value'])
    elif coeff_type == 'piecewise_x':
        return f"{coeff['left']} (x < {coeff.get('x_split', 0.5)}), {coeff['right']} otherwise"
    else:
        return str(coeff)


def generate_prompt(case: Dict, oracle_info: Optional[Dict] = None) -> str:
    """
    为case生成完整的prompt
    
    Args:
        case: benchmark.jsonl中的case配置
        oracle_info: oracle参考信息 {'error': float, 'time': float}
    
    Returns:
        给LLM的完整prompt字符串
    """
    case_id = case['id']
    pde_type = case['oracle_config']['pde']['type']
    pde_config = case['oracle_config']['pde']
    
    # 获取方程模板
    eq_template = EQUATION_TEMPLATES.get(pde_type, EQUATION_TEMPLATES['poisson'])
    
    # 构建prompt
    prompt = f"""# Task: Solve {eq_template['title']}

## Problem Description

{eq_template['equation']}

{eq_template['description']}

**Case ID:** {case_id}
"""

    math_type = case.get("pde_classification", {}).get("math_type", [])
    if math_type:
        prompt += f"\n**Math Type:** {', '.join(math_type)}\n"

    # 添加manufactured solution
    manufactured = pde_config.get('manufactured_solution', {})
    if 'u' in manufactured:
        prompt += f"""
**Manufactured Solution:** u = {manufactured['u']}
(Source term f and boundary data are derived from this exact solution)
"""
        if pde_type in ["stokes", "navier_stokes_incompressible"]:
            prompt += f"**Manufactured Pressure:** p = {manufactured.get('p', 'N/A')}\n"
    else:
        source_term = pde_config.get('source_term')
        if source_term:
            prompt += f"\n**Source Term:** f = {source_term}\n"
        initial_condition = pde_config.get('initial_condition')
        if initial_condition:
            prompt += f"**Initial Condition:** u0 = {initial_condition}\n"

    # 添加系数
    coefficients = pde_config.get('coefficients', {})
    if coefficients:
        prompt += "\n**Coefficients:**\n"
        for name, coeff in coefficients.items():
            prompt += f"- κ = {format_coefficient(coeff)}\n"

    # 对流扩散特有参数
    if pde_type == 'convection_diffusion':
        params = pde_config.get('pde_params', {})
        epsilon = params.get('epsilon', 0.01)
        beta = params.get('beta', [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5 if isinstance(beta, list) else beta
        peclet = beta_norm / epsilon if epsilon > 0 else float('inf')
        
        prompt += f"""
**Convection-Diffusion Parameters:**
- ε (diffusion) = {epsilon}
- β (velocity) = {beta}
- Péclet number ≈ {peclet:.1f}
"""
        if peclet > 10:
            prompt += "⚠️ High Péclet number - consider SUPG stabilization!\n"
    
    if pde_type in ['stokes', 'navier_stokes_incompressible']:
        params = pde_config.get('pde_params', {})
        nu = params.get('nu', 1.0)
        prompt += f"\n**Viscosity:** ν = {nu}\n"

    # 时间相关参数
    if 'time' in pde_config:
        time_cfg = pde_config['time']
        prompt += f"""
**Time Parameters:**
- t_end = {time_cfg.get('t_end', 1.0)}
- dt (suggested) = {time_cfg.get('dt', 0.01)}
- scheme: {time_cfg.get('scheme', 'backward_euler')}
"""

    # 网格和输出配置
    mesh_cfg = case['oracle_config']['mesh']
    output_cfg = case['oracle_config']['output']
    grid_cfg = output_cfg.get('grid', {})
    output_field = output_cfg.get('field', 'scalar')
    
    prompt += f"""
**Domain:** [0,1] × [0,1] (unit square)

**Output Requirements (handled by evaluator):**
- Evaluator will sample solution on a {grid_cfg.get('nx', 50)} × {grid_cfg.get('ny', 50)} uniform grid
- Output field: {output_field}

---

## Implementation Requirements

Write a Python module using **dolfinx** (FEniCSx) that exposes:

```python
def solve(case_spec: dict) -> dict:
    \"\"\"
    Return a dict with:
    - "u": u_grid, numpy array with shape (nx, ny)
    - "solver_info": required fields:
        mesh_resolution, element_degree, ksp_type, pc_type, rtol
    \"\"\"
```

Notes:
1. Do NOT write files (no solution.npz / meta.json).
2. Evaluator will time your solve() call and write outputs.
3. You decide mesh resolution, element degree, solver, etc., but must report them in solver_info.
"""

    # 添加Agent参数暴露
    agent_knobs = case.get("agent_knobs", [])
    if agent_knobs:
        prompt += "\n**Agent-Selectable Parameters:**\n"
        for knob in agent_knobs:
            desc = knob.get('description', '')
            if desc:
                # Remove range hints in parentheses to avoid anchoring models.
                desc = desc.split('(')[0].strip()
            prompt += f"- {knob.get('name')}: {desc}\n"

    # 添加Oracle参考信息
    if oracle_info:
        tolerance = case.get("evaluation_config", {}).get("tolerance", 1.2)
        target_error = oracle_info.get("error", 0.0) * tolerance
        target_time = oracle_info.get("time", 0.0) * tolerance
        prompt += f"""
---

## Oracle Reference

The oracle baseline achieved:
- **Reference Error:** {oracle_info.get('error', 0):.2e}
- **Reference Time:** {oracle_info.get('time', 0):.3f}s

**Pass/Fail Criteria (single tier):**
- Accuracy: error ≤ {target_error:.2e} (tolerance {tolerance:.2f}×)
- Time: wall_time_sec ≤ {target_time:.3f}s (tolerance {tolerance:.2f}×)
"""

    prompt += """
---

**Output only the complete, runnable Python code.** No explanations needed.
"""

    # 附加 DOLFINX 0.10.0 指南（若存在）
    guide_path = Path(__file__).resolve().parents[2] / "DOLFINX_GUIDE.md"
    if guide_path.exists():
        guide_text = guide_path.read_text()
        prompt += f"""

---

## DOLFINX 0.10.0 Guide

{guide_text}
"""

    return prompt
