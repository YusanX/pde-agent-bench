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


def generate_pde_specific_requirements(pde_config: Dict, tags: Dict = None) -> str:
    """生成PDE类型特定的meta.json要求"""
    pde_type = pde_config.get('type', 'poisson')
    
    # 从tags获取主要PDE类型（更准确）
    if tags:
        pde_types = tags.get('pde_type', [])
        if pde_types:
            primary_type = pde_types[0]
        else:
            primary_type = 'elliptic'
    else:
        primary_type = 'elliptic'
    
    # 1. Elliptic (Poisson, Helmholtz等) - 只需基本参数
    if primary_type == 'elliptic' and 'time' not in pde_config:
        return ""  # 基本参数已经足够
    
    # 2. Parabolic (Heat, Diffusion等) - 需要时间步进
    elif primary_type == 'parabolic' or pde_type == 'heat' or 'time' in pde_config:
        time_config = pde_config.get('time', {})
        t_end = time_config.get('t_end', 1.0)
        dt_default = time_config.get('dt', 0.01)
        
        return f"""
**Time-Dependent Problem (Parabolic) Requirements:**
Your meta.json must include time-stepping information:
```json
{{
  "solver_info": {{
    "pde_specific": {{
      "time_stepping": {{
        "dt": {dt_default},           // Your chosen time step
        "n_steps": {int(t_end/dt_default)},        // Total number of steps
        "scheme": "backward_euler"  // Your time integration scheme
      }}
    }}
  }}
}}
```

**Time Step Selection Guide:**
- CFL condition: dt ≤ 0.5 * h² / κ (for explicit schemes)
- Implicit schemes (backward Euler, Crank-Nicolson) are unconditionally stable
- Smaller dt → more accurate, slower
"""
    
    # 3. Hyperbolic (Wave, Advection等) - 需要时间步进+稳定化
    elif primary_type == 'hyperbolic':
        return """
**Hyperbolic Problem Requirements:**
Your meta.json must include:
```json
{
  "solver_info": {
    "pde_specific": {
      "time_stepping": {
        "dt": 0.001,
        "n_steps": 1000,
        "scheme": "runge_kutta"  // or "explicit_euler", "ssp_rk3"
      },
      "stabilization": {
        "method": "TVD",         // or "WENO", "Upwind", "Lax-Friedrichs"
        "limiter": "minmod"      // or "superbee", "van_leer"
      }
    }
  }
}
```

**Key Considerations:**
- CFL condition critical: dt ≤ h / |wave_speed|
- TVD/WENO schemes prevent spurious oscillations
- Flux limiters maintain monotonicity
"""
    
    # 4. Mixed-Type (Convection-Diffusion等)
    elif primary_type == 'mixed_type' or pde_type == 'convection_diffusion':
        params = pde_config.get('pde_params', {})
        epsilon = params.get('epsilon', 0.01)
        beta = params.get('beta', [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5 if isinstance(beta, list) else beta
        peclet = beta_norm / epsilon if epsilon > 0 else float('inf')
        
        if peclet > 10:
            return f"""
**Convection-Diffusion Problem (Pe ≈ {peclet:.1f}) Requirements:**
⚠️ High Péclet number - stabilization likely needed!

Your meta.json should include:
```json
{{
  "solver_info": {{
    "pde_specific": {{
      "convection_diffusion": {{
        "stabilization": "SUPG",      // Your stabilization method
        "tau_parameter": 0.05         // Stabilization parameter
      }}
    }}
  }}
}}
```

**Stabilization Options:**
- SUPG (Streamline Upwind Petrov-Galerkin)
- GLS (Galerkin Least Squares)
- Upwind scheme
"""
        else:
            return """
**Convection-Diffusion Problem (Balanced Pe) Requirements:**
Standard Galerkin should work, but you may add stabilization info if used.
"""
    
    # 5. Incompressible Flow (Stokes, Navier-Stokes等)
    elif primary_type == 'incompressible_flow':
        return """
**Incompressible Flow Problem Requirements:**
Your meta.json must include mixed finite element information:
```json
{
  "solver_info": {
    "pde_specific": {
      "stokes_ns": {
        "velocity_degree": 2,      // Polynomial degree for velocity
        "pressure_degree": 1,      // Polynomial degree for pressure
        "velocity_space": "Lagrange",
        "pressure_space": "Lagrange"
      }
    }
  }
}
```

**Key Considerations:**
- Taylor-Hood (P2-P1) satisfies inf-sup condition
- Other stable pairs: P2-P0, P3-P2, mini element
- Block preconditioners recommended for efficiency
"""
    
    # 6. Reaction-Diffusion (Allen-Cahn, Cahn-Hilliard等)
    elif primary_type == 'reaction_diffusion':
        return """
**Reaction-Diffusion Problem Requirements:**
Your meta.json must include time-stepping and nonlinear solver info:
```json
{
  "solver_info": {
    "pde_specific": {
      "time_stepping": {
        "dt": 0.01,
        "n_steps": 100,
        "scheme": "backward_euler"
      },
      "nonlinear_solver": {
        "method": "newton",        // or "picard"
        "max_iterations": 10,
        "tolerance": 1e-6,
        "line_search": true        // Helps with convergence
      }
    }
  }
}
```

**Key Considerations:**
- Reaction terms can be stiff → implicit schemes
- Newton's method for strong nonlinearity
- Adaptive time-stepping may help
"""
    
    # 7. Dispersive (Schrödinger, KdV等)
    elif primary_type == 'dispersive':
        return """
**Dispersive PDE Problem Requirements:**
Your meta.json must include time-stepping info:
```json
{
  "solver_info": {
    "pde_specific": {
      "time_stepping": {
        "dt": 0.0001,
        "n_steps": 10000,
        "scheme": "crank_nicolson"  // Recommended for dispersion
      }
    }
  }
}
```

**Key Considerations:**
- High-order time integration for phase accuracy
- Fine time steps to resolve dispersive waves
- Conservation of L2 norm (for Schrödinger)
"""
    
    # 8. Compressible Flow (Euler, Navier-Stokes等)
    elif primary_type == 'compressible_flow':
        return """
**Compressible Flow Problem Requirements:**
Your meta.json must include:
```json
{
  "solver_info": {
    "pde_specific": {
      "time_stepping": {
        "dt": 0.0001,
        "n_steps": 1000,
        "scheme": "runge_kutta"
      },
      "stabilization": {
        "method": "WENO",          // or "TVD", "flux_limiter"
        "order": 5                 // WENO order
      },
      "flux_scheme": "roe"         // or "hll", "hllc", "lax_friedrichs"
    }
  }
}
```

**Key Considerations:**
- Shock-capturing schemes essential
- Positivity-preserving for density
- Entropy-stable schemes preferred
"""
    
    # 9. Kinetic (Boltzmann, Vlasov等)
    elif primary_type == 'kinetic':
        return """
**Kinetic Equation Problem Requirements:**
Your meta.json must include phase-space discretization:
```json
{
  "solver_info": {
    "pde_specific": {
      "time_stepping": {
        "dt": 0.001,
        "n_steps": 1000,
        "scheme": "strang_splitting"  // Operator splitting
      },
      "phase_space": {
        "x_resolution": 64,      // Physical space
        "v_resolution": 32,      // Velocity space
        "v_max": 5.0             // Velocity domain
      }
    }
  }
}
```

**Key Considerations:**
- High-dimensional problem (d_x + d_v)
- Operator splitting for efficiency
- Conservation of mass/momentum/energy
"""
    
    # 10. Fractional (Fractional Laplacian, Caputo derivative等)
    elif primary_type == 'fractional':
        return """
**Fractional PDE Problem Requirements:**
Your meta.json must include fractional order:
```json
{
  "solver_info": {
    "pde_specific": {
      "fractional": {
        "alpha": 0.5,              // Fractional order (0 < α < 1)
        "approximation": "finite_difference"  // or "spectral"
      }
    }
  }
}
```

**Key Considerations:**
- Non-local operators → dense matrices
- Memory-intensive
- Special quadrature rules needed
"""
    
    # 11. Stochastic (SPDE等)
    elif primary_type == 'stochastic':
        return """
**Stochastic PDE Problem Requirements:**
Your meta.json must include Monte Carlo parameters:
```json
{
  "solver_info": {
    "pde_specific": {
      "stochastic": {
        "n_samples": 1000,         // Number of MC samples
        "random_seed": 42,         // For reproducibility
        "noise_type": "white"      // or "colored"
      },
      "time_stepping": {
        "dt": 0.01,
        "n_steps": 100,
        "scheme": "euler_maruyama"  // Stochastic integrator
      }
    }
  }
}
```

**Key Considerations:**
- Convergence rate ~ 1/√N
- Strong vs weak convergence
- Reproducibility critical
"""
    
    # 12. Multiphysics (Coupled problems)
    elif primary_type == 'multiphysics':
        return """
**Multiphysics Problem Requirements:**
Your meta.json must include coupling strategy:
```json
{
  "solver_info": {
    "pde_specific": {
      "multiphysics": {
        "coupling_method": "monolithic",  // or "segregated", "staggered"
        "physics_fields": ["heat", "mechanics"],
        "convergence_tolerance": 1e-6,
        "max_coupling_iterations": 10
      }
    }
  }
}
```

**Key Considerations:**
- Monolithic: expensive but robust
- Segregated: efficient but may need relaxation
- Interface coupling conditions critical
"""
    
    return ""


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
    
    # 添加PDE类型特定的meta.json要求
    tags = config.get('tags', {})
    pde_specific_reqs = generate_pde_specific_requirements(pde_config, tags)
    if pde_specific_reqs:
        prompt += pde_specific_reqs
    
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
2. **YOU decide the optimal discretization parameters** (mesh resolution, element degree, etc.)
3. Accept command-line argument: `--outdir DIR` (output directory)
4. Save the solution to `solution.npz` with fields: `x` (1D array), `y` (1D array), `u` (2D array)
5. Save solver metadata to `meta.json` with fields:
   - `wall_time_sec`: Your solver runtime (required)
   - `solver_info`: Dictionary with (required):
     * `mesh_resolution`: The resolution you chose (e.g., 200)
     * `element_degree`: The polynomial degree you chose (e.g., 1 or 2)
     * `ksp_type`: Linear solver type (e.g., 'cg', 'gmres')
     * `pc_type`: Preconditioner type (e.g., 'hypre', 'ilu')
     * `rationale`: (optional) Brief explanation of your parameter choices

**Output Grid:**
Sample the solution on a uniform {nx}×{ny} grid spanning the domain.

**Parameter Selection Strategy:**
You must balance accuracy and speed:
- Higher resolution → better accuracy, slower runtime
- Higher element degree → better accuracy per DOF, more expensive assembly
- Choose wisely based on the target error and time budget!

---

## Testing Modes

This case supports two testing modes:

### Mode 1: Fix Accuracy, Optimize Speed
**Goal:** Meet accuracy requirement in minimum time.

```bash
python test_fix_accuracy.py --agent-script your_solver.py
```

**Challenge:** Choose resolution/degree to achieve the target error as fast as possible.

**Scoring:**
- Must achieve `error` ≤ {target_error if target_error else 'target_error'}
- Score = 100 × (time_budget / your_runtime)
- Faster = Higher score
- Smart parameter choices = Better scores!

### Mode 2: Fix Time Budget, Optimize Accuracy
**Goal:** Achieve best accuracy within time budget.

```bash
python test_fix_time.py --agent-script your_solver.py
```

**Challenge:** Choose resolution/degree to minimize error while staying within time budget.

**Scoring:**
- Must finish within time_budget
- Score = 100 × max(0, 1 - error/target_error)
- Lower error = Higher score
- Efficient solvers = Better scores!

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

This case provides **3-level evaluation** to comprehensively assess your solver's quality:

### Accuracy Levels (Fix Time Budget, Optimize Accuracy)
- **Level 1 (Engineering Grade)**: Target error ≤ {accuracy_tiers.get('level_1', {}).get('target_error', 'N/A'):.2e}
  * Relaxed requirement (100× Oracle baseline)
  * Suitable for rough estimates
  
- **Level 2 (Standard Grade)**: Target error ≤ {accuracy_tiers.get('level_2', {}).get('target_error', 'N/A'):.2e} ⭐ **Oracle baseline**
  * Default requirement
  * Represents good numerical practice
  
- **Level 3 (Scientific Grade)**: Target error ≤ {accuracy_tiers.get('level_3', {}).get('target_error', 'N/A'):.2e}
  * Strict requirement (0.01× Oracle baseline)
  * Requires refined meshes or high-order elements

### Speed Levels (Fix Accuracy, Optimize Speed)
- **Level 3 (Real-time)**: Time budget = {speed_tiers.get('fast', {}).get('time_budget', 'N/A'):.3f}s (0.1× Oracle)
  * Extremely fast (requires optimal algorithms)
  
- **Level 2 (Interactive)**: Time budget = {speed_tiers.get('medium', {}).get('time_budget', 'N/A'):.3f}s ⭐ **Oracle baseline**
  * Default requirement
  * Standard solver performance
  
- **Level 1 (Batch)**: Time budget = {speed_tiers.get('slow', {}).get('time_budget', 'N/A'):.3f}s (10× Oracle)
  * Relaxed requirement (suitable for direct solvers)

**Evaluation Strategy:**
Your solver will be scored based on how many levels you pass:
- Passing Level 2 = Good solver (meets standard requirements)
- Passing Level 3 = Excellent solver (superior numerical methods or parameter choices)

**Oracle Reference:**
The Oracle baseline was obtained with optimized parameters. Your goal is to match or exceed this performance through smart parameter selection and efficient algorithms.
"""
    
    return prompt

