"""
Template-guided prompt builder for P2 (API Decoupling) experiment.

PURPOSE
-------
Address ICML reviewer concern (eVPk W2, K4KR Q1):
  "The benchmark conflates DOLFINx API proficiency with numerical reasoning ability."

DESIGN
------
This prompt variant provides a complete DOLFINx skeleton with all API boilerplate
pre-filled (mesh creation, function space setup, point-sampling output, solver call).
The model only needs to fill in:
  1. Variational form  — a(u,v) and L(v)  [numerical reasoning / PDE math]
  2. Boundary conditions — BC values from case_spec  [mathematical understanding]
  3. Numerical parameters — mesh resolution, element degree, ksp/pc type, tolerance  [numerical judgment]

INTERPRETATION
--------------
If template_guided pass-rate ≈ standard pass-rate
  → API knowledge is NOT the bottleneck; math/numerics are
  → Benchmark reliably measures numerical reasoning

If template_guided pass-rate > standard pass-rate
  → Standard prompt under-estimates models by imposing API overhead
  → Benchmark is conservative (stronger result for authors)
"""

from typing import Dict, Any, Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Utility code injected at the top of every skeleton
# This provides the "hard API" parts so the model can focus on mathematics
# ─────────────────────────────────────────────────────────────────────────────
_UTILITY_CODE = '''
# ════════════════════════════════════════════════════════════════════════
#  PROVIDED UTILITIES  —  do NOT modify these functions
# ════════════════════════════════════════════════════════════════════════
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as _dmesh, fem, geometry as _geo
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import ufl

def _sample_scalar(u_h, nx: int, ny: int) -> np.ndarray:
    """Sample a scalar DOLFINx Function onto an (ny, nx) uniform grid [0,1]²."""
    msh = u_h.function_space.mesh
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = _geo.bb_tree(msh, msh.topology.dim)
    cands = _geo.compute_collisions_points(tree, pts)
    colliding = _geo.compute_colliding_cells(msh, cands, pts)
    found_pts, cells, idx = [], [], []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links):
            found_pts.append(pts[i]); cells.append(links[0]); idx.append(i)
    vals = np.full(nx * ny, np.nan)
    if found_pts:
        ev = u_h.eval(np.array(found_pts), np.array(cells, dtype=np.int32))
        vals[idx] = ev[:, 0]
    return vals.reshape(ny, nx)


def _sample_vector_magnitude(u_h, nx: int, ny: int) -> np.ndarray:
    """Sample ‖u‖₂ of a vector DOLFINx Function onto an (ny, nx) uniform grid."""
    msh = u_h.function_space.mesh
    gdim = msh.geometry.dim
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = _geo.bb_tree(msh, msh.topology.dim)
    cands = _geo.compute_collisions_points(tree, pts)
    colliding = _geo.compute_colliding_cells(msh, cands, pts)
    found_pts, cells, idx = [], [], []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links):
            found_pts.append(pts[i]); cells.append(links[0]); idx.append(i)
    vals = np.full((nx * ny, gdim), np.nan)
    if found_pts:
        ev = u_h.eval(np.array(found_pts), np.array(cells, dtype=np.int32))
        vals[idx] = ev
    return np.linalg.norm(vals, axis=1).reshape(ny, nx)


def _all_boundary_dofs(msh, V):
    """Return DOF indices on the entire boundary ∂Ω."""
    fdim = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    return fem.locate_dofs_topological(V, fdim, facets)


def _bc_from_str(msh, V, expr_str: str) -> fem.DirichletBC:
    """Build a Dirichlet BC u=g on ∂Ω from a string expression g(x,y).
    Handles constants (e.g. '0.0') and symbolic expressions (e.g. 'sin(pi*x)*sin(pi*y)').
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    try:
        val = float(expr_str)
        g = fem.Function(V)
        g.x.array[:] = val
    except ValueError:
        sym = sp.sympify(expr_str, locals={"x": sx, "y": sy, "pi": sp.pi})
        g_np = sp.lambdify((sx, sy), sym, modules="numpy")
        g = fem.Function(V)
        g.interpolate(lambda pts: g_np(pts[0], pts[1]).astype(np.float64))
    dofs = _all_boundary_dofs(msh, V)
    return fem.dirichletbc(g, dofs)


def _bc_vec_zero(msh, V) -> fem.DirichletBC:
    """Homogeneous Dirichlet BC for a vector function space."""
    fdim = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    g = fem.Function(V)
    g.x.array[:] = 0.0
    return fem.dirichletbc(g, dofs)


def _kappa_from_spec(msh, kappa_spec: dict):
    """Parse a kappa coefficient spec into a UFL scalar expression.
    Spec examples:
      {'type': 'constant', 'value': 1.0}
      {'type': 'expr', 'expr': '1 + 0.5*sin(2*pi*x)*sin(2*pi*y)'}
    """
    import sympy as sp
    ktype = kappa_spec.get("type", "constant")
    x = ufl.SpatialCoordinate(msh)
    if ktype == "constant":
        return ufl.as_ufl(float(kappa_spec["value"]))
    elif ktype == "piecewise_x":
        left  = float(kappa_spec["left"])
        right = float(kappa_spec["right"])
        split = float(kappa_spec.get("x_split", 0.5))
        return ufl.conditional(x[0] < split, ufl.as_ufl(left), ufl.as_ufl(right))
    elif ktype == "expr":
        sx, sy = sp.symbols("x y", real=True)
        sym = sp.sympify(kappa_spec["expr"],
                         locals={"x": sx, "y": sy, "pi": sp.pi})
        kappa_fn = sp.lambdify((sx, sy), sym, modules="numpy")
        kappa_h = fem.Function(fem.functionspace(msh, ("DG", 0)))
        kappa_h.interpolate(lambda pts: kappa_fn(pts[0], pts[1]).astype(np.float64))
        return kappa_h
    else:
        return ufl.as_ufl(1.0)


def _f_from_str(msh, expr_str: str):
    """Parse a source-term string to a UFL expression or fem.Function."""
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    try:
        return ufl.as_ufl(float(expr_str))
    except (ValueError, TypeError):
        pass
    sym = sp.sympify(expr_str, locals={"x": sx, "y": sy, "pi": sp.pi})
    x = ufl.SpatialCoordinate(msh)
    # Build a DG0 interpolation for general expressions
    f_np = sp.lambdify((sx, sy), sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))
    return f_h


def _manufactured_f_and_bc(msh, V, pde_cfg: dict, kappa_spec: dict):
    """From a manufactured_solution spec, derive f = -div(κ ∇u_exact) symbolically
    and return (f_expr_ufl, bc) for use in the variational form.
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    u_str = pde_cfg["manufactured_solution"]["u"]
    u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "pi": sp.pi})

    ktype = kappa_spec.get("type", "constant")
    if ktype == "constant":
        kap_sym = sp.sympify(kappa_spec.get("value", 1.0))
    elif ktype == "expr":
        kap_sym = sp.sympify(kappa_spec["expr"],
                             locals={"x": sx, "y": sy, "pi": sp.pi})
    else:
        kap_sym = sp.sympify(1.0)

    f_sym = -(sp.diff(kap_sym * sp.diff(u_sym, sx), sx)
              + sp.diff(kap_sym * sp.diff(u_sym, sy), sy))
    f_sym = sp.simplify(f_sym)

    f_np = sp.lambdify((sx, sy), f_sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))

    g_np = sp.lambdify((sx, sy), u_sym, modules="numpy")
    g_h = fem.Function(V)
    g_h.interpolate(lambda pts: g_np(pts[0], pts[1]).astype(np.float64))
    dofs = _all_boundary_dofs(msh, V)
    bc = fem.dirichletbc(g_h, dofs)
    return f_h, bc
# ════════════════════════════════════════════════════════════════════════
#  END OF PROVIDED UTILITIES
# ════════════════════════════════════════════════════════════════════════
'''

# ─────────────────────────────────────────────────────────────────────────────
# Poisson: case-specific dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_poisson_skeleton(case: Dict) -> str:
    """
    Generate a Poisson skeleton with ALL PDE math data pre-filled.
    The model only writes: variational form a, L + numerical parameter choices.
    No case_spec access needed at runtime.
    """
    pde_cfg    = case["oracle_config"]["pde"]
    kappa_spec = pde_cfg.get("coefficients", {}).get("kappa",
                                                     {"type": "constant", "value": 1.0})
    manufactured = pde_cfg.get("manufactured_solution", {})
    out_cfg    = case["oracle_config"]["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    # Cell type for mesh creation
    cell_type_str = case["oracle_config"]["mesh"].get("cell_type", "triangle")
    if cell_type_str == "quadrilateral":
        mesh_call = "    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N,\n" \
                    "               cell_type=_dmesh.CellType.quadrilateral)"
    else:
        mesh_call = "    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)"

    # PDE data section — fully pre-filled, no case_spec needed
    kappa_repr = repr(kappa_spec)
    if manufactured.get("u"):
        u_str = manufactured["u"]
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa   = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h, bc = _manufactured_f_and_bc(\n"
            f"        msh, V,\n"
            f"        {{'manufactured_solution': {{'u': {repr(u_str)}}}}},\n"
            f"        {kappa_repr}\n"
            f"    )"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h   = _f_from_str(msh, {repr(f_str)})\n"
            f"    bc    = _bc_from_str(msh, V, {repr(bc_val)})"
        )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    N        = None  # ← choose mesh resolution (integer)
    degree   = None  # ← choose FE polynomial degree: 1, 2, or 3
    ksp_type = None  # ← choose linear solver: "cg" or "gmres"
    pc_type  = None  # ← choose preconditioner: "hypre", "ilu", or "jacobi"
    rtol     = None  # ← choose relative tolerance

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
{mesh_call}
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE:  -∇·(κ ∇u) = f  in Ω,   u = g on ∂Ω
    # Weak form:  find u ∈ V s.t.  ∫ κ ∇u·∇v dx = ∫ f v dx   ∀ v ∈ V
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form a(u, v)
    L = ...   # ← YOUR CODE: linear form L(v)

    # ── SOLVE  (PROVIDED — do not modify) ────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc],
                        petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                        petsc_options_prefix="p2_poisson_").solve()

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Heat: case-specific dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_heat_skeleton(case: Dict) -> str:
    """
    Generate a Heat skeleton with ALL PDE math data pre-filled.
    The model writes: variational form a, L + time-loop + numerical choices.
    No case_spec access needed at runtime.
    """
    pde_cfg    = case["oracle_config"]["pde"]
    kappa_spec = pde_cfg.get("coefficients", {}).get("kappa",
                                                     {"type": "constant", "value": 1.0})
    manufactured = pde_cfg.get("manufactured_solution", {})
    time_cfg   = pde_cfg.get("time", {})
    t_end      = float(time_cfg.get("t_end", 1.0))
    ic_str     = pde_cfg.get("initial_condition", "0.0")
    out_cfg    = case["oracle_config"]["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    kappa_repr = repr(kappa_spec)
    if manufactured.get("u"):
        u_str = manufactured["u"]
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa   = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h, bc = _manufactured_f_and_bc(\n"
            f"        msh, V,\n"
            f"        {{'manufactured_solution': {{'u': {repr(u_str)}}}}},\n"
            f"        {kappa_repr}\n"
            f"    )"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h   = _f_from_str(msh, {repr(f_str)})\n"
            f"    bc    = _bc_from_str(msh, V, {repr(bc_val)})"
        )

    # Initial condition
    ic_code = (
        f"    u_n = fem.Function(V)\n"
        f"    u_n.interpolate(lambda pts: __import__('sympy').lambdify(\n"
        f"        (__import__('sympy').symbols('x y')),\n"
        f"        __import__('sympy').sympify({repr(ic_str)},\n"
        f"            locals={{'x': __import__('sympy').Symbol('x'),\n"
        f"                     'y': __import__('sympy').Symbol('y'),\n"
        f"                     'pi': __import__('sympy').pi}}),\n"
        f"        modules='numpy')(pts[0], pts[1]).astype(float))"
    )
    # Simplified: just use _bc_from_str logic for IC
    ic_code = (
        f"    # Initial condition (pre-filled — do not modify):\n"
        f"    u_n = fem.Function(V)\n"
        f"    _ic = _bc_from_str(msh, V, {repr(ic_str)})  # reuse BC helper for interpolation\n"
        f"    u_n.x.array[:] = _ic.g.x.array[:]  # copy interpolated values"
        if ic_str != "0.0" else
        f"    # Initial condition (pre-filled — do not modify):\n"
        f"    u_n = fem.Function(V)  # zero initial condition"
    )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    N           = None  # ← choose mesh resolution (integer)
    degree      = None  # ← choose FE polynomial degree: 1, 2, or 3
    dt          = None  # ← choose time step (float); t_end = {t_end}
    time_scheme = None  # ← choose: "backward_euler" or "crank_nicolson"
    ksp_type    = None  # ← choose linear solver: "cg" or "gmres"
    pc_type     = None  # ← choose preconditioner: "hypre", "ilu", or "jacobi"
    rtol        = None  # ← choose relative tolerance

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

{ic_code}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE:  ∂u/∂t - ∇·(κ ∇u) = f   in Ω × (0, {t_end}]
    # Discretise time with your chosen scheme.
    # For Backward Euler weak form:
    #   ∫ u v dx + dt ∫ κ ∇u·∇v dx = ∫ u_n v dx + dt ∫ f v dx   ∀ v
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form (must include mass term for time-stepping)
    L = ...   # ← YOUR CODE: linear form   (must include u_n term)

    # ── TIME LOOP  (PROVIDED structure — fill in the solve call) ─────────
    n_steps = max(1, round({t_end} / dt))
    u_h = fem.Function(V)

    for _ in range(n_steps):
        u_h = LinearProblem(a, L, bcs=[bc],
                            petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                            petsc_options_prefix="p2_heat_").solve()
        u_n.x.array[:] = u_h.x.array  # update previous step

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol,
                        "dt": dt, "n_steps": n_steps, "time_scheme": time_scheme}},
    }}
'''

_CONVDIFF_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 64
    degree   = 1
    ksp_type = "gmres"
    pc_type  = "ilu"
    rtol     = 1e-8
    use_supg = True    # set True for high Péclet (Pe > 1); harmless otherwise

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # NOTE: case_spec["oracle_config"]["pde"]["pde_params"] contains:
    #   'epsilon': diffusion coefficient ε
    #   'beta': convection velocity [β_x, β_y]
    # Check case_spec["oracle_config"]["pde"].get("time") for transient variant.
    #
    # Helpers: _manufactured_f_and_bc, _f_from_str, _bc_from_str

    pde_cfg = case_spec["oracle_config"]["pde"]
    params  = pde_cfg.get("pde_params", {})
    epsilon = float(params.get("epsilon", 0.01))
    beta_v  = params.get("beta", [1.0, 1.0])
    beta    = ufl.as_vector([float(beta_v[0]), float(beta_v[1])])

    # TODO: define f_h and bc
    f_h = ...   # TODO
    bc  = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # PDE:  -ε ∇²u + β·∇u = f   in Ω,   u = g on ∂Ω
    # Standard weak form:  ε ∫ ∇u·∇v dx + ∫ (β·∇u) v dx = ∫ f v dx
    # SUPG stabilization (for high Pe): adds τ_SUPG * (β·∇u) * (β·∇v) terms
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # TODO: bilinear form (with optional SUPG)
    L = ...   # TODO: linear form

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["oracle_config"]["output"]["grid"]
    return {
        "u": _sample_scalar(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_STOKES_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 32       # mesh resolution (Stokes is expensive; 32–64 typical)
    deg_u    = 2        # velocity element degree  — MUST be > deg_p for inf-sup
    deg_p    = 1        # pressure element degree
    ksp_type = "gmres"
    pc_type  = "lu"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    from basix.ufl import element as _bel, mixed_element as _bmix
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    # Taylor-Hood: P{deg_u}/P{deg_p}
    W = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, deg_u, shape=(gdim,)),
        _bel("Lagrange", cell, deg_p),
    ]))
    V_col, _ = W.sub(0).collapse()
    x = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # case_spec['pde']['pde_params']['nu'] — kinematic viscosity
    # case_spec['pde']['manufactured_solution'] — {'u': [...], 'p': ...} (may be present)
    # case_spec['pde']['source_term'] — body force as list ['fx', 'fy'] (may be present)
    #
    # For manufactured Stokes:
    #   f = -ν ∇²u_exact + ∇p_exact  (compute symbolically from strings)

    pde_cfg = case_spec["pde"]
    nu = float(pde_cfg.get("pde_params", {}).get("nu", 1.0))

    # TODO: define body force f as ufl.as_vector([fx, fy])
    f_body = ...   # TODO: e.g. ufl.as_vector([0.0, 0.0])

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Stokes:  -ν ∇²u + ∇p = f,  ∇·u = 0
    # Weak form (mixed):
    #   ν ∫ ∇u:∇v dx - ∫ p ∇·v dx + ∫ q ∇·u dx = ∫ f·v dx
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q)     = ufl.TestFunctions(W)

    a = ...   # TODO: bilinear form
    L = ...   # TODO: linear form

    # ── BOUNDARY CONDITIONS  (YOUR CODE) ─────────────────────────────────
    # No-slip on walls: u = 0 (or manufactured u_exact)
    # Hint: use _bc_vec_zero(msh, V_col) for homogeneous BC then lift to W.sub(0)
    fdim  = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, facets)
    u_bc   = fem.Function(V_col); u_bc.x.array[:] = 0.0  # TODO: replace with exact BC if available
    bc_u   = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    bcs    = [bc_u]
    # TODO: add pressure pin or manufactured pressure BC if needed

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    wh = LinearProblem(a, L, bcs=bcs, petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()
    u_h = wh.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_vector_magnitude(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": deg_u,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_NS_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N              = 32
    deg_u          = 2
    deg_p          = 1
    newton_rtol    = 1e-8
    newton_max_it  = 30

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    from basix.ufl import element as _bel, mixed_element as _bmix
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    W = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, deg_u, shape=(gdim,)),
        _bel("Lagrange", cell, deg_p),
    ]))
    V_col, _ = W.sub(0).collapse()
    x = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    pde_cfg = case_spec["pde"]
    nu = float(pde_cfg.get("pde_params", {}).get("nu", 1.0))
    # TODO: define body force f_body as ufl.as_vector([fx, fy])
    f_body = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Navier-Stokes (steady):  (u·∇)u - ν ∇²u + ∇p = f,  ∇·u = 0
    # Newton linearisation: F(w) = 0, J = dF/dw
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = ...   # TODO: nonlinear residual form
    J = ufl.derivative(F, w)   # Jacobian (automatic differentiation — provided)

    # ── BOUNDARY CONDITIONS  (YOUR CODE) ─────────────────────────────────
    fdim   = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, facets)
    u_bc   = fem.Function(V_col); u_bc.x.array[:] = 0.0  # TODO: set exact BC
    bcs    = [fem.dirichletbc(u_bc, dofs_u, W.sub(0))]

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    from dolfinx.nls.petsc import NewtonSolver
    from dolfinx.fem.petsc import NonlinearProblem as _NLP
    problem = _NLP(F, w, bcs=bcs, J=J)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = newton_rtol
    solver.max_it = newton_max_it
    solver.solve(w)
    w.x.scatter_forward()
    u_h = w.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_vector_magnitude(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": deg_u,
                        "ksp_type": "gmres", "pc_type": "lu", "rtol": newton_rtol},
    }
'''

_HELMHOLTZ_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 64       # need ~10+ points per wavelength: N ≳ 10*k/π
    degree   = 2        # higher degree recommended for Helmholtz
    ksp_type = "gmres"
    pc_type  = "lu"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    pde_cfg = case_spec["pde"]
    params  = pde_cfg.get("pde_params", {})
    k       = float(params.get("k", params.get("wave_number", 10.0)))

    # TODO: define f_h (source term) and bc
    f_h = ...   # TODO
    bc  = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # PDE:  -∇²u - k² u = f   in Ω,   u = g on ∂Ω
    # Weak form:  ∫ ∇u·∇v dx - k² ∫ u v dx = ∫ f v dx   ∀ v
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # TODO: ∫ ∇u·∇v dx - k² ∫ u v dx
    L = ...   # TODO: ∫ f v dx

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_scalar(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_BIHARMONIC_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    # Mixed formulation: Δ²u = f  →  (−Δu = w,  −Δw = f)
    N        = 64
    degree   = 2        # P2 minimum for 4th-order problems
    ksp_type = "cg"
    pc_type  = "hypre"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    from basix.ufl import element as _bel, mixed_element as _bmix
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    # Mixed space for Ciarlet-Raviart formulation: W = V × V
    W = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, degree),
        _bel("Lagrange", cell, degree),
    ]))
    V_col, _ = W.sub(0).collapse()
    x = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    pde_cfg = case_spec["pde"]
    # TODO: define f_h (source term) and boundary conditions
    f_h = ...   # TODO
    bc  = ...   # TODO: u = 0 and w = 0 on ∂Ω (both components)

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Biharmonic Δ²u = f via mixed: find (u,w) in W×W s.t.
    #   ∫ ∇u·∇v dx + ∫ w v dx = 0           [eq 1: −Δu = w]
    #   ∫ ∇w·∇z dx             = ∫ f z dx   [eq 2: −Δw = f]
    (u_t, w_t) = ufl.TrialFunctions(W)
    (v, z)     = ufl.TestFunctions(W)

    a = ...   # TODO: coupled bilinear form for the mixed system
    L = ...   # TODO: right-hand side

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    wh  = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()
    u_h = wh.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_scalar(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_ELASTICITY_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 64
    degree   = 2        # P2 avoids locking for most ν; increase if ν → 0.5
    ksp_type = "cg"
    pc_type  = "hypre"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    gdim = msh.geometry.dim
    V    = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))
    x    = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # case_spec['pde']['pde_params']:
    #   'E', 'nu'  (Young's modulus + Poisson ratio)  or  'lambda', 'mu' (Lamé)
    # case_spec['pde']['source_term']: list of 2 strings ['fx', 'fy'] or single string
    # case_spec['pde']['manufactured_solution']['u']: list ['ux','uy'] or string

    pde_cfg = case_spec["pde"]
    params  = pde_cfg.get("pde_params", {})
    E_val   = params.get("E",  None)
    nu_val  = params.get("nu", None)
    lam_val = params.get("lambda", None)
    mu_val  = params.get("mu",     None)

    # TODO: compute Lamé constants mu, lam from E,nu  or  use lam,mu directly
    mu  = ...   # TODO
    lam = ...   # TODO

    # TODO: define body force f as ufl.as_vector([...])
    f_body = ...   # TODO

    # TODO: define Dirichlet BC (displacement on ∂Ω)
    # Hint: for homogeneous  →  _bc_vec_zero(msh, V)
    bc = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # -∇·σ(u) = f,   σ = 2μ ε(u) + λ tr(ε(u)) I,   ε(u) = sym(∇u)
    # Weak form:  ∫ σ(u):ε(v) dx = ∫ f·v dx   ∀ v
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    def eps(w): return ufl.sym(ufl.grad(w))
    def sigma(w): return ...   # TODO: 2μ ε(w) + λ tr(ε(w)) I

    a = ...   # TODO: ∫ σ(u):ε(v) dx
    L = ...   # TODO: ∫ f·v dx

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
        "pc_hypre_type": "boomeramg",
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_vector_magnitude(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_DARCY_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    # Use pressure formulation (standard Poisson): -∇·(κ ∇p) = f
    N        = 64
    degree   = 1
    ksp_type = "cg"
    pc_type  = "hypre"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    pde_cfg    = case_spec["pde"]
    kappa_spec = pde_cfg.get("coefficients", {}).get("kappa", {"type": "constant", "value": 1.0})
    kappa      = _kappa_from_spec(msh, kappa_spec)

    if "manufactured_solution" in pde_cfg and "u" in pde_cfg["manufactured_solution"]:
        f_h, bc = _manufactured_f_and_bc(msh, V, pde_cfg, kappa_spec)
    else:
        f_str = pde_cfg.get("source_term", "0.0")
        bc_str = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        f_h = _f_from_str(msh, f_str)
        bc  = _bc_from_str(msh, V, bc_str)

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Pressure form: -∇·(κ ∇p) = f
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # TODO: identical to Poisson but for pressure p
    L = ...   # TODO

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_scalar(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_RXNDIFF_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N             = 64
    degree        = 1
    newton_rtol   = 1e-8
    newton_max_it = 25
    ksp_type      = "gmres"
    pc_type       = "ilu"

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # case_spec['pde']['pde_params']:
    #   'epsilon': diffusion coefficient ε
    #   'reaction': reaction term string, e.g. 'u*(1-u)' or 'u**3'
    # If time parameters present, add time-stepping.
    pde_cfg = case_spec["pde"]
    params  = pde_cfg.get("pde_params", {})
    epsilon = float(params.get("epsilon", 1.0))
    rxn_str = str(params.get("reaction", "0"))

    # TODO: define f_h (source term) and bc
    f_h = ...   # TODO
    bc  = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # PDE:  -ε ∇²u + R(u) = f   in Ω
    # R(u) may be nonlinear (e.g. u³). Use Newton if so.
    # Nonlinear residual form:  F(u) = ε ∫ ∇u·∇v dx + ∫ R(u) v dx - ∫ f v dx = 0
    uh = fem.Function(V)    # iterate
    v  = ufl.TestFunction(V)

    # TODO: define reaction R_u as a UFL expression in uh
    # e.g. for rxn_str == 'u*(1-u)': R_u = uh * (1 - uh)
    R_u = ...   # TODO

    F = ...   # TODO: nonlinear residual
    J = ufl.derivative(F, uh)   # Jacobian (provided — automatic)

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    from dolfinx.nls.petsc import NewtonSolver
    from dolfinx.fem.petsc import NonlinearProblem as _NLP
    problem = _NLP(F, uh, bcs=[bc], J=J)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol   = newton_rtol
    solver.max_it = newton_max_it
    solver.solve(uh)

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_scalar(uh, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": newton_rtol},
    }
'''

# Map PDE type → skeleton
# Static skeletons (other PDE types — still use case_spec at runtime)
_SKELETONS_STATIC = {
    "convection_diffusion": _CONVDIFF_SKELETON,
    "stokes":               _STOKES_SKELETON,
    "navier_stokes":        _NS_SKELETON,
    "helmholtz":            _HELMHOLTZ_SKELETON,
    "biharmonic":           _BIHARMONIC_SKELETON,
    "linear_elasticity":    _ELASTICITY_SKELETON,
    "darcy":                _DARCY_SKELETON,
    "reaction_diffusion":   _RXNDIFF_SKELETON,
}

# Dynamic skeleton builders (Poisson and Heat — PDE data inlined at generation time)
_SKELETON_BUILDERS = {
    "poisson": _build_poisson_skeleton,
    "heat":    _build_heat_skeleton,
}

# All supported PDE types
_SUPPORTED_PDE_TYPES = set(_SKELETONS_STATIC) | set(_SKELETON_BUILDERS)


def _get_skeleton(pde_type: str, case: Dict) -> str:
    """Return the skeleton code for a given PDE type and case."""
    if pde_type in _SKELETON_BUILDERS:
        return _SKELETON_BUILDERS[pde_type](case)
    return _SKELETONS_STATIC.get(pde_type, _build_poisson_skeleton(case))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_template_prompt(case: Dict, oracle_info: Optional[Dict] = None) -> str:
    """
    Generate a template-guided prompt for the P2 (API Decoupling) experiment.

    The prompt contains:
      1. Problem statement (same PDE description as standard prompt)
      2. Provided utility functions (sampling, BC helpers, expression parsing)
      3. A PDE-specific skeleton with:
           - Provided sections: mesh/space setup, solver call, output
           - TODO sections:     variational form, BCs, PDE data extraction
         The model fills in ONLY the TODO sections.

    This isolates numerical reasoning ability from DOLFINx API knowledge.
    """
    case_id  = case["id"]
    pde_cfg  = case["oracle_config"]["pde"]
    pde_type = pde_cfg["type"]

    # ── Header: problem statement ──────────────────────────────────────
    from pdebench.core.prompt_builder import EQUATION_TEMPLATES, format_coefficient
    if pde_type == "convection_diffusion" and "time" in pde_cfg:
        eq_tmpl = EQUATION_TEMPLATES.get("convection_diffusion_transient",
                                         EQUATION_TEMPLATES["convection_diffusion"])
    else:
        eq_tmpl = EQUATION_TEMPLATES.get(pde_type, EQUATION_TEMPLATES["poisson"])

    prompt = f"""# Task: Solve {eq_tmpl['title']}  [Template-Guided Mode]

## Problem Description

{eq_tmpl['equation']}

{eq_tmpl['description']}

**Case ID:** {case_id}
"""

    # PDE math metadata
    math_type = case.get("pde_classification", {}).get("math_type", [])
    if math_type:
        prompt += f"\n**Math Type:** {', '.join(math_type)}\n"

    manufactured = pde_cfg.get("manufactured_solution", {})
    if "u" in manufactured:
        prompt += f"\n**Manufactured Solution:** u = {manufactured['u']}\n"
        if pde_type in ["stokes", "navier_stokes"]:
            prompt += f"**Manufactured Pressure:** p = {manufactured.get('p', 'N/A')}\n"
    else:
        src = pde_cfg.get("source_term")
        if src:
            prompt += f"\n**Source Term:** f = {src}\n"
        ic = pde_cfg.get("initial_condition")
        if ic:
            prompt += f"**Initial Condition:** u₀ = {ic}\n"

    coefficients = pde_cfg.get("coefficients", {})
    if coefficients:
        prompt += "\n**Coefficients:**\n"
        for name, coeff in coefficients.items():
            prompt += f"- κ = {format_coefficient(coeff)}\n"

    if pde_type in ["convection_diffusion"]:
        params = pde_cfg.get("pde_params", {})
        epsilon = params.get("epsilon", 0.01)
        beta = params.get("beta", [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5 if isinstance(beta, list) else float(beta)
        peclet = beta_norm / epsilon if epsilon > 0 else float("inf")
        prompt += f"\n**Convection-Diffusion Parameters:**\n- ε = {epsilon}\n- β = {beta}\n- Péclet ≈ {peclet:.1f}\n"
        if peclet > 10:
            prompt += "⚠️  High Péclet — SUPG stabilization strongly recommended.\n"

    if pde_type in ["stokes", "navier_stokes"]:
        nu = pde_cfg.get("pde_params", {}).get("nu", 1.0)
        prompt += f"\n**Viscosity:** ν = {nu}\n"

    if pde_type == "helmholtz":
        k = pde_cfg.get("pde_params", {}).get("k",
            pde_cfg.get("pde_params", {}).get("wave_number", 10.0))
        prompt += f"\n**Wavenumber:** k = {k}\n"

    if pde_type == "linear_elasticity":
        params = pde_cfg.get("pde_params", {})
        E  = params.get("E")
        nu = params.get("nu")
        lam = params.get("lambda")
        mu  = params.get("mu")
        if E is not None and nu is not None:
            prompt += f"\n**Material Parameters:** E = {E}, ν = {nu}\n"
        elif lam is not None and mu is not None:
            prompt += f"\n**Material Parameters:** λ = {lam}, μ = {mu}\n"

    if "time" in pde_cfg:
        tc = pde_cfg["time"]
        prompt += (f"\n**Time Parameters:** t_end={tc.get('t_end',1.0)}, "
                   f"dt_suggested={tc.get('dt',0.01)}, "
                   f"scheme={tc.get('scheme','backward_euler')}\n")

    if oracle_info:
        eval_cfg = case.get("evaluation_config", {})
        at = eval_cfg.get("accuracy_tolerance", eval_cfg.get("tolerance", 1.2))
        tt = eval_cfg.get("time_tolerance",     eval_cfg.get("tolerance", 1.2))
        min_err = 1e-6
        target_err  = max(oracle_info.get("error", 0.0) * at, min_err)
        target_time = oracle_info.get("time", 0.0) * tt
        prompt += f"""
---

**Pass/Fail Criteria:**
- Accuracy: error ≤ {target_err:.2e}
- Time: wall_time_sec ≤ {target_time:.3f}s
"""

    # ── Experiment framing ─────────────────────────────────────────────
    prompt += """
---

## Your Task (Template-Guided Mode)

**All DOLFINx API boilerplate is provided below** — mesh creation, function space
setup, point-sampling output, and solver invocation are already written for you.

You only need to fill in the sections marked `# TODO`:
1. **Variational form** — the bilinear form `a(u,v)` and linear form `L(v)`
2. **Boundary conditions** — read BC values from `case_spec` using the provided helpers
3. **Numerical parameters** — choose mesh resolution, element degree, solver type, tolerance

This focuses purely on mathematical / numerical reasoning, not API syntax.

**Return the complete, runnable Python code (fill in all TODOs).**
"""

    # ── Utility code + skeleton ────────────────────────────────────────
    skeleton = _get_skeleton(pde_type, case)
    prompt += f"""
---

## Provided Utilities + Code Skeleton

```python
{_UTILITY_CODE}
{skeleton}
```
"""

    return prompt


def is_template_supported(pde_type: str) -> bool:
    """Return True if a template exists for this PDE type."""
    return pde_type in _SUPPORTED_PDE_TYPES
