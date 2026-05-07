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
    "convection_diffusion_transient": {
        "title": "Convection-Diffusion Equation (Transient)",
        "equation": "∂u/∂t - ε ∇²u + β·∇u = f   in Ω × (0, T]\n  u = g                    on ∂Ω\n  u(x,0) = u₀(x)           in Ω",
        "description": "Time-dependent convection-diffusion requiring time-stepping; stabilization may be needed at high Péclet numbers."
    },
    "stokes": {
        "title": "Stokes Flow (Incompressible)",
        "equation": "-ν ∇²u + ∇p = f   in Ω\n  ∇·u = 0             in Ω\n  u = g               on ∂Ω",
        "description": "Steady incompressible flow; use Taylor-Hood mixed elements."
    },
    "navier_stokes": {
        "title": "Navier-Stokes (Incompressible, Steady)",
        "equation": "u·∇u - ν ∇²u + ∇p = f   in Ω\n  ∇·u = 0               in Ω\n  u = g                 on ∂Ω",
        "description": "Nonlinear steady incompressible flow; Newton/Picard is acceptable."
    },
    "darcy": {
        "title": "Darcy Flow (Steady)",
        "equation": "Elliptic (pressure) form:\n  -∇·(κ ∇p) = f   in Ω\n  p = g           on ∂Ω\n\nMixed (flux-pressure) form:\n  u + κ ∇p = 0     in Ω\n  ∇·u = f          in Ω\n  (boundary data depends on formulation)",
        "description": "Steady porous-media flow. Elliptic pressure formulation and a stable mixed RT×DG formulation are both acceptable; report what you solve and what field you output."
    },
    "reaction_diffusion": {
        "title": "Reaction-Diffusion Equation (Steady or Transient)",
        "equation": "Steady:\n  -ε ∇²u + R(u) = f    in Ω\n  u = g              on ∂Ω\n\nTransient (if time params provided):\n  ∂u/∂t - ε ∇²u + R(u) = f   in Ω × (0,T]\n  u = g                    on ∂Ω\n  u(x,0) = u₀(x)           in Ω",
        "description": "Scalar diffusion with (possibly nonlinear) reaction term. Newton/Picard/time-stepping are acceptable depending on R(u) and whether time dependence is present."
    },
    "helmholtz": {
        "title": "Helmholtz Equation",
        "equation": "-∇²u - k² u = f   in Ω\n  u = g          on ∂Ω",
        "description": "Indefinite elliptic problem (can be challenging at large k); GMRES+ILU or a direct solver is acceptable."
    },
    "biharmonic": {
        "title": "Biharmonic Equation",
        "equation": "Δ²u = f   in Ω\n  u = g   on ∂Ω",
        "description": "Fourth-order elliptic problem; a mixed formulation (two Poisson solves) is acceptable."
    },
    "linear_elasticity": {
        "title": "Linear Elasticity (2D, Small Strain)",
        "equation": "-∇·σ(u) = f   in Ω\n  u = g        on ∂Ω\n  σ(u) = 2μ ε(u) + λ tr(ε(u)) I,   ε(u)=sym(∇u)",
        "description": "Vector-valued elliptic system; use a conforming vector FE space. CG+AMG or GMRES+AMG/direct is acceptable depending on conditioning."
    },
    "wave": {
        "title": "Wave Equation (2D, Transient)",
        "equation": "∂²u/∂t² - c² Δu = f   in Ω × (0,T]\n  u = g                  on ∂Ω × (0,T]\n  u(x,0)      = u₀(x)   in Ω\n  ∂u/∂t(x,0) = v₀(x)   in Ω",
        "description": "Second-order hyperbolic equation; use a second-order-in-time scheme such as Newmark-β (β=1/4, γ=1/2) or leap-frog. The Newmark average-acceleration scheme (θ=1/4) is unconditionally stable."
    },
    "burgers": {
        "title": "Burgers' Equation (2D, Transient, Nonlinear)",
        "equation": "∂u/∂t + u·∇u - ν Δu = f   in Ω × (0,T]\n  u = g                      on ∂Ω × (0,T]\n  u(x,0) = u₀(x)             in Ω",
        "description": "Nonlinear parabolic equation; semi-implicit linearization (treat the convection term u_n·∇u explicitly, diffusion implicitly) is recommended. Small ν may require stabilization."
    }
}


def format_domain(domain_cfg: Dict) -> str:
    """根据 oracle_config.domain 生成人类可读的域描述字符串。
    与 oracle/common.py 和 oracle/firedrake_oracle/common.py 的 create_mesh() 保持一致。
    """
    domain_type = domain_cfg.get("type", "unit_square")
    params = domain_cfg.get("geometry_params", {})

    if domain_type == "unit_square":
        return "[0,1] × [0,1] (unit square)"

    if domain_type == "unit_cube":
        return "[0,1] × [0,1] × [0,1] (unit cube)"

    if domain_type == "l_shape":
        verts = params.get("vertices")
        if verts:
            vstr = ", ".join(f"({v[0]},{v[1]})" for v in verts)
            return f"L-shaped polygon, vertices: [{vstr}]"
        # 旧格式兼容
        ob = domain_cfg.get("outer_bbox", [0, 1, 0, 1])
        cb = domain_cfg.get("cutout_bbox", [0.5, 1, 0.5, 1])
        return (f"L-shaped domain: outer [{ob[0]},{ob[1]}]×[{ob[2]},{ob[3]}], "
                f"top-right cutout [{cb[0]},{cb[1]}]×[{cb[2]},{cb[3]}]")

    if domain_type == "circle":
        c, r = params.get("center", [0.5, 0.5]), params.get("radius", 0.5)
        return f"Circular domain: center=({c[0]},{c[1]}), radius={r}"

    if domain_type == "annulus":
        c = params.get("center", [0, 0])
        r_i, r_o = params.get("inner_r", 0.5), params.get("outer_r", 1.0)
        return f"Annular (ring) domain: center=({c[0]},{c[1]}), inner_r={r_i}, outer_r={r_o}"

    if domain_type == "square_with_hole":
        out = params.get("outer", [0, 1, 0, 1])
        ostr = f"[{out[0]},{out[1]}]×[{out[2]},{out[3]}]"
        ih = params.get("inner_hole", {})
        ht = ih.get("type", "circle")
        if ht == "circle":
            c, r = ih.get("center", [0.5, 0.5]), ih.get("radius", 0.2)
            return f"Square {ostr} with circular hole: center=({c[0]},{c[1]}), radius={r}"
        if ht == "rect":
            b = ih.get("bbox", [0.4, 0.6, 0.4, 0.6])
            return f"Square {ostr} with rectangular hole: [{b[0]},{b[1]}]×[{b[2]},{b[3]}]"
        # polygon
        verts = ih.get("vertices", [])
        vstr = ", ".join(f"({v[0]},{v[1]})" for v in verts)
        return f"Square {ostr} with polygonal hole: [{vstr}]"

    if domain_type == "multi_hole":
        out = params.get("outer", [0, 1, 0, 1])
        ostr = f"[{out[0]},{out[1]}]×[{out[2]},{out[3]}]"
        holes = params.get("holes", [])
        hstr = "; ".join(f"center=({h['c'][0]},{h['c'][1]}), r={h['r']}" for h in holes)
        return f"Square {ostr} with {len(holes)} circular hole(s): [{hstr}]"

    if domain_type == "t_junction":
        h = params.get("horizontal_rect", [0.0, 1.0, 0.4, 0.6])
        v = params.get("vertical_rect", [0.4, 0.6, 0.0, 0.5])
        return (f"T-junction domain: horizontal [{h[0]},{h[1]}]×[{h[2]},{h[3]}], "
                f"vertical [{v[0]},{v[1]}]×[{v[2]},{v[3]}]")

    if domain_type == "sector":
        c, r = params.get("center", [0, 0]), params.get("radius", 1.0)
        ang = params.get("angle", 90)
        return f"Circular sector: center=({c[0]},{c[1]}), radius={r}, angle={ang}°"

    if domain_type in ("star", "star_shape"):
        c = params.get("center", [0, 0])
        n = params.get("points", 5)
        r_i, r_o = params.get("inner_r", 0.3), params.get("outer_r", 0.7)
        return f"{n}-point star domain: center=({c[0]},{c[1]}), inner_r={r_i}, outer_r={r_o}"

    if domain_type == "gear":
        c = params.get("center", [0, 0])
        teeth = params.get("teeth", 8)
        b_r, t_h = params.get("base_r", 0.5), params.get("tooth_h", 0.2)
        return f"Gear domain: center=({c[0]},{c[1]}), {teeth} teeth, base_r={b_r}, tooth_h={t_h}"

    if domain_type == "eccentric_annulus":
        oc = params.get("outer_circle", {"c": [0, 0], "r": 1.0})
        ic = params.get("inner_circle", {"c": [0.2, 0], "r": 0.4})
        return (f"Eccentric annulus: outer circle center=({oc['c'][0]},{oc['c'][1]}), r={oc['r']}; "
                f"inner circle center=({ic['c'][0]},{ic['c'][1]}), r={ic['r']}")

    if domain_type == "dumbbell":
        # oracle 同时支持 left_center/right_center 和 left_circle/right_circle 两种 key
        lc = params.get("left_circle", params.get("left_center", {"c": [0.25, 0.5], "r": 0.25}))
        rc = params.get("right_circle", params.get("right_center", {"c": [0.75, 0.5], "r": 0.25}))
        if isinstance(lc, dict) and "c" in lc:
            lpos, lr = lc["c"], lc.get("r", 0.25)
        else:
            lpos, lr = lc, params.get("radius", 0.2)
        if isinstance(rc, dict) and "c" in rc:
            rpos, rr = rc["c"], rc.get("r", 0.25)
        else:
            rpos, rr = rc, params.get("radius", 0.2)
        bridge = params.get("bridge", {})
        bstr = (f"bridge x=[{bridge.get('x_min', lpos[0])},{bridge.get('x_max', rpos[0])}], "
                f"y=[{bridge.get('y_min', 0.4)},{bridge.get('y_max', 0.6)}]"
                if bridge else f"bar_width={params.get('bar_width', 0.2)}")
        return (f"Dumbbell domain: left circle center=({lpos[0]},{lpos[1]}), r={lr}; "
                f"right circle center=({rpos[0]},{rpos[1]}), r={rr}; {bstr}")

    if domain_type == "periodic_square":
        bounds = params.get("bounds", params.get("extents", [0, 1, 0, 1]))
        return f"Periodic square: [{bounds[0]},{bounds[1]}]×[{bounds[2]},{bounds[3]}] (periodic BCs on all sides)"

    return f"{domain_type} domain (see oracle_config.domain for geometry parameters)"


def generate_nl_description(case: Dict) -> str:
    """Generate a concise natural-language problem description for the prompt header.

    Format (one clause per line):
        Solve <the [steady-state] PDE name>
        on <domain description>
        with <coefficient description>
        and <boundary condition description>.
    """
    pde_config = case['oracle_config']['pde']
    pde_type = pde_config.get('type', 'poisson')
    domain_cfg = case['oracle_config'].get('domain', {'type': 'unit_square'})
    bc_cfg = case['oracle_config'].get('bc', {})

    # ── PDE name ──────────────────────────────────────────────────────────────
    has_time = 'time' in pde_config
    _pde_base = {
        'poisson':              'Poisson equation',
        'heat':                 'heat equation',
        'convection_diffusion': 'convection-diffusion equation',
        'stokes':               'Stokes flow problem',
        'navier_stokes':        'Navier-Stokes flow problem',
        'helmholtz':            'Helmholtz equation',
        'biharmonic':           'biharmonic equation',
        'linear_elasticity':    'linear elasticity problem',
        'reaction_diffusion':   'reaction-diffusion equation',
        'wave':                 'wave equation',
        'burgers':              "Burgers' equation",
    }
    pde_base = _pde_base.get(pde_type, f'{pde_type.replace("_", "-")} equation')
    _steady_types = {
        'poisson', 'stokes', 'navier_stokes', 'helmholtz', 'biharmonic', 'linear_elasticity',
    }
    if pde_type in _steady_types:
        full_pde = f'the steady-state {pde_base}'
    elif has_time:
        full_pde = f'the transient {pde_base}'
    else:
        full_pde = f'the {pde_base}'

    # ── Domain ────────────────────────────────────────────────────────────────
    domain_type = domain_cfg.get('type', 'unit_square')
    _domain_nl = {
        'unit_square':       'a unit square domain',
        'unit_cube':         'a 3D unit cube domain',
        'l_shape':           'an L-shaped domain',
        'circle':            'a circular domain',
        'annulus':           'an annular (ring) domain',
        'square_with_hole':  'a square domain with a hole',
        'multi_hole':        'a square domain with multiple holes',
        't_junction':        'a T-junction domain',
        'sector':            'a circular sector domain',
        'star':              'a star-shaped domain',
        'star_shape':        'a star-shaped domain',
        'gear':              'a gear-shaped domain',
        'eccentric_annulus': 'an eccentric annular domain',
        'dumbbell':          'a dumbbell-shaped domain',
        'periodic_square':   'a periodic square domain',
    }
    domain_nl = _domain_nl.get(domain_type, f'a {domain_type.replace("_", "-")} domain')

    # ── Coefficients ──────────────────────────────────────────────────────────
    coefficients = pde_config.get('coefficients', {})
    coeff_desc = ''
    if coefficients:
        has_variable = any(
            (isinstance(v, dict) and v.get('type', 'constant') != 'constant')
            for v in coefficients.values()
        )
        if pde_type == 'linear_elasticity':
            coeff_desc = ('heterogeneous material properties'
                          if has_variable else 'homogeneous material properties')
        elif pde_type in ('stokes', 'navier_stokes'):
            coeff_desc = 'variable viscosity' if has_variable else 'constant viscosity'
        else:
            coeff_desc = ('heterogeneous conductivity'
                          if has_variable else 'homogeneous conductivity')

    # Helmholtz: always mention wave number as the key parameter
    if pde_type == 'helmholtz':
        k = pde_config.get('pde_params', {}).get('k',
              pde_config.get('pde_params', {}).get('wave_number', None))
        coeff_desc = f'wave number k={k}' if k is not None else 'prescribed wave number'

    # Convection-diffusion: mention Péclet regime
    if pde_type == 'convection_diffusion':
        params = pde_config.get('pde_params', {})
        eps = params.get('epsilon', 0.01)
        beta = params.get('beta', [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5 if isinstance(beta, list) else float(beta)
        pe = beta_norm / eps if eps > 0 else float('inf')
        coeff_desc = f'high-Péclet convection (Pe≈{pe:.0f})' if pe > 10 else f'low-Péclet diffusion (Pe≈{pe:.1f})'

    # ── Boundary conditions ───────────────────────────────────────────────────
    has_dirichlet = bool(bc_cfg.get('dirichlet', {}))
    has_neumann = bool(bc_cfg.get('neumann', {}))
    if has_dirichlet and has_neumann:
        bc_desc = 'mixed (Dirichlet + Neumann) boundary conditions'
    elif has_neumann:
        bc_desc = 'Neumann boundary conditions'
    else:
        bc_desc = 'Dirichlet boundary conditions'

    # ── Assemble ──────────────────────────────────────────────────────────────
    lines = [f'Solve {full_pde}', f'on {domain_nl}']
    if coeff_desc:
        lines.append(f'with {coeff_desc}')
        lines.append(f'and {bc_desc}.')
    else:
        lines.append(f'with {bc_desc}.')

    return '\n'.join(lines)


def format_coefficient(coeff: Dict) -> str:
    """格式化系数配置"""
    coeff_type = coeff.get('type', 'constant')
    
    if coeff_type == 'constant':
        return str(coeff['value'])
    elif coeff_type == 'piecewise_x':
        return f"{coeff['left']} (x < {coeff.get('x_split', 0.5)}), {coeff['right']} otherwise"
    else:
        return str(coeff)


def generate_prompt(
    case: Dict,
    oracle_info: Optional[Dict] = None,
    solver_library: str = "dolfinx",
) -> str:
    """
    为case生成完整的prompt

    Args:
        case: benchmark.jsonl中的case配置
        oracle_info: oracle参考信息 {'error': float, 'time': float}
        solver_library: 'dolfinx' (default) | 'firedrake'

    Returns:
        给LLM的完整prompt字符串
    """
    case_id = case['id']
    pde_type = case['oracle_config']['pde']['type']
    pde_config = case['oracle_config']['pde']
    
    # 获取方程模板（对流扩散：如果有 time 字段，使用 transient 模板）
    if pde_type == "convection_diffusion" and "time" in pde_config:
        eq_template = EQUATION_TEMPLATES["convection_diffusion_transient"]
    else:
        eq_template = EQUATION_TEMPLATES.get(pde_type, EQUATION_TEMPLATES['poisson'])
    
    # 自然语言描述（首行）
    nl_desc = generate_nl_description(case)

    # 构建prompt
    prompt = f"""{nl_desc}

# Task: Solve {eq_template['title']}

## Problem Description

{eq_template['equation']}

{eq_template['description']}

**Case ID:** {case_id}
"""

    math_type = case.get("pde_classification", {}).get("math_type", [])
    if math_type:
        prompt += f"\n**Math Type:** {', '.join(math_type)}\n"

    # ── Source term ──────────────────────────────────────────────────────────
    # manufactured_solution 已离线预计算为 source_term / initial_condition 等字段，
    # 此处直接展示预计算结果，不再向 agent 暴露解析解表达式。
    source_term = pde_config.get('source_term')
    if source_term:
        if isinstance(source_term, list):
            # 向量 PDE（Stokes / NS / LE）：逐分量展示
            for idx, comp in enumerate(source_term):
                comp_label = ["x", "y", "z"][idx] if idx < 3 else str(idx)
                prompt += f"\n**Source Term f_{comp_label}:** {comp}\n"
        else:
            prompt += f"\n**Source Term:** f = {source_term}\n"

    # ── Initial / boundary conditions ────────────────────────────────────────
    initial_condition = pde_config.get('initial_condition')
    if initial_condition:
        prompt += f"**Initial Condition:** u₀ = {initial_condition}\n"

    # Boundary conditions（Dirichlet value 已由预计算脚本替换为显式表达式）
    bc_cfg = case['oracle_config'].get('bc', {})
    dirichlet = bc_cfg.get('dirichlet', {})
    if dirichlet:
        prompt += "\n**Boundary Conditions (Dirichlet):**\n"
        entries = dirichlet if isinstance(dirichlet, list) else [dirichlet]
        for entry in entries:
            bc_on = entry.get('on', 'all')
            bc_value = entry.get('value', '0.0')
            if isinstance(bc_value, list):
                bc_value_str = f"[{', '.join(str(v) for v in bc_value)}]"
            else:
                bc_value_str = str(bc_value)
            prompt += f"- u = {bc_value_str}   on {bc_on}\n"
    neumann = bc_cfg.get('neumann', {})
    if neumann:
        prompt += "\n**Boundary Conditions (Neumann):**\n"
        entries = neumann if isinstance(neumann, list) else [neumann]
        for entry in entries:
            nm_on = entry.get('on', 'part')
            nm_value = entry.get('value', '0.0')
            if isinstance(nm_value, list):
                nm_value_str = f"[{', '.join(str(v) for v in nm_value)}]"
            else:
                nm_value_str = str(nm_value)
            prompt += f"- ∂u/∂n = {nm_value_str}   on {nm_on}\n"

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
    
    if pde_type in ['stokes', 'navier_stokes']:
        params = pde_config.get('pde_params', {})
        nu = params.get('nu', 1.0)
        prompt += f"\n**Viscosity:** ν = {nu}\n"

    if pde_type == 'helmholtz':
        params = pde_config.get('pde_params', {})
        k = params.get('k', params.get('wave_number', 10.0))
        prompt += f"\n**Wavenumber:** k = {k}\n"

    if pde_type == 'linear_elasticity':
        params = pde_config.get('pde_params', {})
        E = params.get('E', None)
        nu = params.get('nu', None)
        lam = params.get('lambda', None)
        mu = params.get('mu', None)
        if E is not None and nu is not None:
            prompt += f"\n**Material Parameters:** E = {E}, ν = {nu}\n"
        elif lam is not None and mu is not None:
            prompt += f"\n**Material Parameters:** λ = {lam}, μ = {mu}\n"

    if pde_type == 'wave':
        params = pde_config.get('pde_params', {})
        c = params.get('c', 1.0)
        prompt += f"\n**Wave Speed:** c = {c}\n"
        # initial_condition 已在通用段渲染；此处仅补充 initial_velocity（若未被通用段覆盖）
        iv = pde_config.get('initial_velocity')
        if iv:
            prompt += f"**Initial Velocity v₀:** {iv}\n"
        prompt += "⚠️ Implement a **second-order** time scheme (e.g. Newmark-β with β=1/4, γ=1/2).\n"

    if pde_type == 'burgers':
        params = pde_config.get('pde_params', {})
        nu_val = params.get('nu', 0.01)
        t_final = pde_config.get('t_final', 0.1)
        dt_val = pde_config.get('dt', 0.01)
        prompt += f"""
**Burgers Parameters:**
- ν (viscosity) = {nu_val}
- T_final = {t_final}
- dt (suggested) = {dt_val}
"""
        if nu_val < 0.05:
            prompt += "⚠️ Low viscosity — consider SUPG stabilization or sufficiently fine mesh.\n"
        prompt += "Use **semi-implicit linearization**: treat u_n·∇u explicitly, diffusion implicitly.\n"

    # 时间相关参数（适用于 heat / convection_diffusion_transient / wave 等含 time 字典的方程）
    if 'time' in pde_config:
        time_cfg = pde_config['time']
        t0 = time_cfg.get('t0', 0.0)
        t_end = time_cfg.get('t_end', 1.0)
        dt = time_cfg.get('dt', 0.01)
        scheme = time_cfg.get('scheme', 'backward_euler' if pde_type != 'wave' else 'newmark_beta')
        prompt += f"""
**Time Parameters:**
- t0 = {t0}
- t_end = {t_end}
- dt (suggested) = {dt}
- scheme: {scheme}
"""

    # 网格和输出配置
    output_cfg = case['oracle_config']['output']
    output_field = output_cfg.get('field', 'scalar')
    eq_type = case.get('pde_classification', {}).get('equation_type', '')

    # 向量场附加说明（仅线弹性等向量值PDE）
    vector_field_note = ""
    if eq_type == "linear_elasticity" or "displacement" in output_field:
        vector_field_note = (
            "\n- ⚠️  **Vector-valued problem**: your FEM space must be a **vector** Lagrange space "
            "`(shape=(gdim,))`. The evaluated quantity is the **displacement magnitude** "
            "`‖u‖ = √(u₁² + u₂²)`, not individual components. "
            "For near-incompressible materials (ν > 0.4), use **P2 or higher** to avoid volumetric locking."
        )

    if solver_library == "dealii":
        lib_name = "**deal.II** (https://www.dealii.org, C++ FEM library)"
    elif solver_library == "firedrake":
        lib_name = "**Firedrake** (https://www.firedrakeproject.org)"
    else:
        lib_name = "**dolfinx** (FEniCSx)"

    # 域描述
    domain_cfg = case['oracle_config'].get('domain', {'type': 'unit_square'})
    domain_desc = format_domain(domain_cfg)
    is_complex_domain = domain_cfg.get('type', 'unit_square') not in ('unit_square', 'unit_cube')

    # 复杂域 NaN 采样说明（按库分支）
    complex_domain_note = ""
    if is_complex_domain:
        domain_type = domain_cfg.get("type", "")
        if solver_library == "firedrake":
            complex_domain_note = f"""
- ⚠️  **Non-rectangular domain** (`{domain_type}`): the output grid `bbox` may extend \
beyond Ω. Build your mesh from `case_spec["domain"]` (type + geometry_params). \
Use `u.at(coords, dont_raise=True)` to evaluate — Firedrake returns `None` for \
points outside the mesh; convert to `np.nan` so the NaN-safe error metric ignores them:
  ```python
  domain = case_spec["domain"]  # type, geometry_params
  raw = u_h.at(coords, dont_raise=True)
  values = np.array([float(v) if v is not None else np.nan for v in raw], dtype=float)
  ```"""
        elif solver_library != "dealii":
            complex_domain_note = f"""
- ⚠️  **Non-rectangular domain** (`{domain_type}`): the output grid `bbox` may extend \
beyond Ω. Build your mesh from `case_spec["domain"]` (type + geometry_params). \
Use dolfinx collision detection so exterior points become `np.nan` \
(the NaN-safe error metric ignores them):
  ```python
  domain = case_spec["domain"]  # type, geometry_params
  from dolfinx import geometry
  bb_tree = geometry.bb_tree(msh, msh.topology.dim)
  cell_candidates = geometry.compute_collisions_points(bb_tree, points)
  colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
  values = np.full(len(points), np.nan)
  for i, links in enumerate(colliding_cells):
      if len(links) > 0:
          values[i] = float(u_h.eval(points[i:i+1], links[:1]))
  ```"""

    # 检测 3D
    grid_cfg = output_cfg.get('grid', {})
    is_3d = 'nz' in grid_cfg and len(grid_cfg.get('bbox', [])) == 6
    if is_3d:
        shape_desc = "`(nz, ny, nx)`"
        grid_keys = '`nx`, `ny`, `nz`'
        bbox_desc = '`[xmin, xmax, ymin, ymax, zmin, zmax]`'
    else:
        shape_desc = "`(ny, nx)`"
        grid_keys = '`nx`, `ny`'
        bbox_desc = '`[xmin, xmax, ymin, ymax]`'

    prompt += f"""
**Domain:** {domain_desc}

**Output Requirements:**
- You must sample your solution onto the uniform grid defined by `case_spec["output"]["grid"]` ({grid_keys} and `bbox` = {bbox_desc}) and return it as a numpy array of shape {shape_desc}.
- The output shape MUST match exactly; mismatched shapes will fail evaluation (no interpolation or resampling is applied by the evaluator).
- Output field: {output_field}{vector_field_note}{complex_domain_note}

---

## Implementation Requirements
"""

    # ── deal.II C++ 接口（与 Python 接口不同）──────────────────────────────
    if solver_library == "dealii":
        if is_3d:
            grid_comment = (
                '    //      nx = case_spec["output"]["grid"]["nx"]  (int)\n'
                '    //      ny = case_spec["output"]["grid"]["ny"]  (int)\n'
                '    //      nz = case_spec["output"]["grid"]["nz"]  (int)\n'
                '    //      bbox = case_spec["output"]["grid"]["bbox"]  ([xmin,xmax,ymin,ymax,zmin,zmax])'
            )
            bin_shape = "[nz, ny, nx]"
            ordering = (
                "- Row-major order: outer loop = z (plane k), middle = y (row j), inner = x (col i)\n"
                "- `value[k*ny*nx + j*nx + i]` = u at point (x_lin[i], y_lin[j], z_lin[k])\n"
                "- `x_lin = linspace(bbox[0], bbox[1], nx)`\n"
                "- `y_lin = linspace(bbox[2], bbox[3], ny)`\n"
                "- `z_lin = linspace(bbox[4], bbox[5], nz)`"
            )
            npz_shape = "(nz, ny, nx)"
            meta_nz = '\n  "nz": <int>,'
        else:
            grid_comment = (
                '    //      nx = case_spec["output"]["grid"]["nx"]  (int)\n'
                '    //      ny = case_spec["output"]["grid"]["ny"]  (int)\n'
                '    //      bbox = case_spec["output"]["grid"]["bbox"]  ([xmin,xmax,ymin,ymax])'
            )
            bin_shape = "[ny, nx]"
            ordering = (
                "- Row-major order: outer loop = y (row j), inner loop = x (col i)\n"
                "- `value[j*nx + i]` = u at point (x_lin[i], y_lin[j])\n"
                "- `x_lin = linspace(bbox[0], bbox[1], nx)`\n"
                "- `y_lin = linspace(bbox[2], bbox[3], ny)`"
            )
            npz_shape = "(ny, nx)"
            meta_nz = ''

        prompt += f"""
Write a **C++** program using {lib_name} that:

```cpp
// Required interface:
// argv[1]: path to case_spec.json  (contains the full case specification)
// argv[2]: output directory        (already exists; write your output here)

int main(int argc, char* argv[]) {{
    // 1. Read case_spec.json with nlohmann/json
    // 2. Build mesh, FE space, assemble, solve
    // 3. Sample solution on uniform grid:
{grid_comment}
    // 4. Write output files:
    //      argv[2]/solution_grid.bin  (float64, row-major {bin_shape})
    //      argv[2]/meta.json          (see below)
}}
```

**meta.json must contain:**
```json
{{
  "nx": <int>,
  "ny": <int>,{meta_nz}
  "wall_time_sec": <float>,
  "solver_info": {{
    "mesh_resolution": <int>,
    "element_degree":  <int>,
    "ksp_type":        "<str>",
    "pc_type":         "<str>",
    "rtol":            <float>
  }}
}}
```

**Grid ordering convention** (must match):
- `solution_grid.bin` is a raw binary array of float64 values shaped {bin_shape}
{ordering}

**Alternatively**, you may write `solution.npz` (numpy format) with field `"u"` of shape {npz_shape}.

The evaluator provides:
- `nlohmann/json` (header-only, `#include <nlohmann/json.hpp>`)
- deal.II ≥ 9.3 (linked via CMake `deal_ii_setup_target`)
"""
    else:
        # ── Python 接口（dolfinx / firedrake）─────────────────────────────
        prompt += f"""
Write a Python module using {lib_name} that exposes:

**Key requirements (read before the code template):**
1. You decide mesh resolution, element degree, solver type, etc. — but you MUST sample your final FEM solution onto the prescribed evaluation grid and return it with the exact required shape.
2. Read the grid specification from `case_spec["output"]["grid"]`: it contains {grid_keys}, `bbox` = {bbox_desc}. Build the uniform grid from these, evaluate your FEM solution on every grid point, and return the result as a numpy array of shape {shape_desc}.
3. Do NOT write any files (no solution.npz / meta.json). The evaluator handles file I/O.
4. Report your solver choices in `solver_info` (see template below).
"""
        if is_3d:
            prompt += """
```python
def solve(case_spec: dict) -> dict:
    \"\"\"
    Return a dict with:
    - "u": numpy array of shape **(nz, ny, nx)** sampled on the uniform 3-D grid:
           nx   = case_spec["output"]["grid"]["nx"]
           ny   = case_spec["output"]["grid"]["ny"]
           nz   = case_spec["output"]["grid"]["nz"]
           bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax, zmin, zmax]
         Use your FEM solution's eval() to interpolate onto these nx*ny*nz points.
         ⚠️  Output shape MUST be exactly (nz, ny, nx); wrong shape will fail evaluation.
"""
        else:
            prompt += """
```python
def solve(case_spec: dict) -> dict:
    \"\"\"
    Return a dict with:
    - "u": numpy array of shape **(ny, nx)** sampled on the uniform 2-D grid:
           nx   = case_spec["output"]["grid"]["nx"]
           ny   = case_spec["output"]["grid"]["ny"]
           bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
         Use your FEM solution's eval() to interpolate onto these nx*ny points.
         ⚠️  Output shape MUST be exactly (ny, nx); wrong shape will fail evaluation.
"""
        prompt += """    - "solver_info": dict with fields organized by PDE type:
    
      ALWAYS REQUIRED (all PDEs):
        - mesh_resolution (int): spatial mesh resolution (e.g., 64, 128)
        - element_degree (int): polynomial degree (1, 2, 3, ...)
        - ksp_type (str): linear solver type (e.g., 'cg', 'gmres')
        - pc_type (str): preconditioner type (e.g., 'jacobi', 'ilu', 'hypre')
        - rtol (float): relative tolerance for linear solver
      
      REQUIRED if you perform LINEAR solves (record actual solver behavior):
        - iterations (int): total linear solver iterations across all solves
      
      REQUIRED if PDE contains TIME (check case_spec['pde']['time']):
        - dt (float): time step size you used (e.g., 0.01)
        - n_steps (int): number of time steps you actually computed (e.g., 50)
        - time_scheme (str): time integrator you used ('backward_euler', 'crank_nicolson', or 'bdf2')
        
        Example for transient PDE:
          "solver_info": {{
            "mesh_resolution": 120, "element_degree": 1,
            "ksp_type": "gmres", "pc_type": "ilu", "rtol": 1e-8,
            "iterations": 450,  # sum of all linear iterations
            "dt": 0.01, "n_steps": 50, "time_scheme": "backward_euler"
          }}
      
      REQUIRED if PDE is NONLINEAR (e.g., reaction terms like u^3 or u(1-u)):
        - nonlinear_iterations (list of int): Newton iterations per time step
          (for steady: single value in list; for transient: one per time step)
        
        Example for nonlinear transient:
          "nonlinear_iterations": [5, 4, 4, 3, ...]  # one per time step
    
    ADDITIONALLY for time-dependent PDEs (highly recommended for analysis):
    - "u_initial": initial condition array, same shape as u (enables front propagation tracking)
    \"\"\"
```
"""

    # 添加Agent参数暴露（过滤 manufactured_solution 相关 knob，避免泄露实现细节）
    _HIDDEN_KNOB_NAMES = {"manufactured_solution", "manufactured_u", "exact_solution"}
    agent_knobs = [k for k in case.get("agent_knobs", [])
                   if k.get("name") not in _HIDDEN_KNOB_NAMES]
    if agent_knobs:
        prompt += "\n**Agent-Selectable Parameters:**\n"
        for knob in agent_knobs:
            desc = knob.get('description', '')
            if desc:
                # Remove range hints in parentheses to avoid anchoring models.
                desc = desc.split('(')[0].strip()
            prompt += f"- {knob.get('name')}: {desc}\n"

    # 添加评测标准（不展示Oracle参考信息）
    if oracle_info:
        eval_cfg = case.get("evaluation_config", {})
        legacy_tolerance = eval_cfg.get("tolerance", 1.2)
        accuracy_tolerance = eval_cfg.get("accuracy_tolerance", legacy_tolerance)
        time_tolerance = eval_cfg.get("time_tolerance", legacy_tolerance)
        # 与主链路一致：误差阈值有最小下限，时间阈值不设最小值
        min_error_threshold = 1e-6
        target_error = max(oracle_info.get("error", 0.0) * accuracy_tolerance, min_error_threshold)
        target_time = oracle_info.get("time", 0.0) * time_tolerance
        prompt += f"""
---

**Pass/Fail Criteria (single tier):**
- Accuracy: error ≤ {target_error:.2e}
- Time: wall_time_sec ≤ {target_time:.3f}s
"""

    if solver_library == "dealii":
        prompt += """
---

**Output only the complete, runnable C++ code.** No explanations needed.
"""
    else:
        prompt += """
---

**Output only the complete, runnable Python code.** No explanations needed.
"""

    # 附加对应库的参考指南（若存在）
    guide_root = Path(__file__).resolve().parents[2]
    if solver_library == "dealii":
        guide_path = guide_root / "DEALII_GUIDE.md"
        guide_title = "deal.II 9.x C++ API Reference Guide"
    elif solver_library == "firedrake":
        guide_path = guide_root / "FIREDRAKE_GUIDE.md"
        guide_title = "Firedrake API Reference Guide"
    else:
        guide_path = guide_root / "DOLFINX_GUIDE.md"
        guide_title = "DOLFINX 0.10.0 Guide"

    if guide_path.exists():
        guide_text = guide_path.read_text()
        prompt += f"""

---

## {guide_title}

{guide_text}
"""

    return prompt
