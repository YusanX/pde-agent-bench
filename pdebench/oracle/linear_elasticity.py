"""Linear elasticity oracle solver (elliptic, vector-valued).

We solve the 2D linear elasticity (small strain) problem on the unit square:

    -div(sigma(u)) = f    in Ω
                 u = g    on ∂Ω

where
    eps(u)   = sym(grad(u))
    sigma(u) = 2 μ eps(u) + λ tr(eps(u)) I

Material parameters:
  - Provide either (E, nu) in pde_params, or (lambda, mu).
  - Defaults: E=1.0, nu=0.3 (plane strain interpretation for λ formula).

Output:
  - Sample displacement magnitude ||u|| on a fixed grid (consistent with existing benchmarks).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem

from .common import (
    OracleResult,
    compute_rel_L2_grid,
    create_mesh,
    create_vector_space,
    interpolate_expression,
    parse_expression,
    parse_vector_expression,
    sample_vector_magnitude_on_grid,
    _sample_vector_mag_grid,
    _mms_local_dict,
    _mms_coords,
    _eval_exact_vec_mag_on_grid,
)


def _lame_from_params(params: Dict[str, Any]) -> tuple[float, float]:
    """Return (lambda, mu) from pde_params."""
    if "lambda" in params and "mu" in params:
        lam = float(params["lambda"])
        mu = float(params["mu"])
        return lam, mu
    E = float(params.get("E", 1.0))
    nu = float(params.get("nu", 0.3))
    # Plane strain Lamé parameters
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(lam), float(mu)


def _normalize_dirichlet_cfg(bc_cfg: Any) -> list[Dict[str, Any]]:
    if bc_cfg is None:
        return []
    if isinstance(bc_cfg, dict):
        return [bc_cfg]
    if isinstance(bc_cfg, list):
        return bc_cfg
    raise ValueError("bc.dirichlet must be a dict or a list of dicts")


def _boundary_selector(on: str, dim: int):
    key = on.lower()
    if key in {"all", "*"}:
        return lambda x: np.ones(x.shape[1], dtype=bool)
    if key in {"x0", "xmin"}:
        return lambda x: np.isclose(x[0], 0.0)
    if key in {"x1", "xmax"}:
        return lambda x: np.isclose(x[0], 1.0)
    if dim >= 2:
        if key in {"y0", "ymin"}:
            return lambda x: np.isclose(x[1], 0.0)
        if key in {"y1", "ymax"}:
            return lambda x: np.isclose(x[1], 1.0)
    if dim >= 3:
        if key in {"z0", "zmin"}:
            return lambda x: np.isclose(x[2], 0.0)
        if key in {"z1", "zmax"}:
            return lambda x: np.isclose(x[2], 1.0)
    raise ValueError(f"Unknown boundary selector: {on}")


def _rhs_vector(msh, source_expr: Any, dim: int):
    """Build robust vector RHS.

    - If components are numeric constants, use fem.Constant(mesh, tuple).
    - Otherwise parse expressions into a UFL vector.
    """
    if source_expr is None:
        return fem.Constant(msh, tuple([0.0] * dim))
    if isinstance(source_expr, (list, tuple)):
        if len(source_expr) != dim:
            raise ValueError("source_term dimension mismatch")
        expr_list = list(source_expr)
    else:
        expr_list = [source_expr] * dim
    try:
        const_values = [float(sp.sympify(expr)) for expr in expr_list]
        return fem.Constant(msh, tuple(const_values))
    except Exception:
        x = ufl.SpatialCoordinate(msh)
        comps = [parse_expression(expr, x) for expr in expr_list]
        return ufl.as_vector(comps)


def _build_dirichlet_bcs(msh, V, bc_cfg: Any, u_exact: fem.Function | None, dim: int):
    bcs = []
    dirichlet_cfgs = _normalize_dirichlet_cfg(bc_cfg)
    fdim = msh.topology.dim - 1
    for cfg in dirichlet_cfgs:
        on = cfg.get("on", "all")
        if on.lower() in {"all", "*"}:
            # Topological approach: locate boundary facets first, then DOFs on those facets.
            # locate_dofs_geometrical with np.ones(...) would incorrectly select ALL interior
            # DOFs as Dirichlet, bypassing the FEM solve entirely.
            facets = dmesh.locate_entities_boundary(
                msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
            )
            boundary_dofs = fem.locate_dofs_topological(V, fdim, facets)
        else:
            selector = _boundary_selector(on, dim)
            boundary_dofs = fem.locate_dofs_geometrical(V, selector)
        value = cfg.get("value", "0.0")
        if isinstance(value, str) and value in {"u", "u_exact"}:
            if u_exact is None:
                raise ValueError("Dirichlet value 'u' requires manufactured_solution")
            bc_func = u_exact
        else:
            if isinstance(value, (list, tuple)):
                if len(value) != dim:
                    raise ValueError("Dirichlet vector value dimension mismatch")
                expr_list = list(value)
            else:
                expr_list = [value] * dim

            # constants fast path
            try:
                const_values = [float(sp.sympify(expr)) for expr in expr_list]
                is_constant = True
            except Exception:
                is_constant = False

            bc_func = fem.Function(V)
            if is_constant:
                bc_func.interpolate(
                    lambda x: np.array([[v] * x.shape[1] for v in const_values])
                )
            else:
                x = ufl.SpatialCoordinate(msh)
                bc_expr = ufl.as_vector([parse_expression(expr, x) for expr in expr_list])
                interpolate_expression(bc_func, bc_expr)
        bcs.append(fem.dirichletbc(bc_func, boundary_dofs))
    return bcs


class LinearElasticitySolver:
    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        # ⏱️ 开始计时整个求解流程
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        dim = msh.geometry.dim

        V = create_vector_space(
            msh, case_spec["fem"]["family"], case_spec["fem"]["degree"]
        )

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        lam, mu = _lame_from_params(params)

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")

        f_expr = None
        u_exact = None

        if "u" in manufactured:
            u_sym_list = manufactured["u"]
            if not isinstance(u_sym_list, (list, tuple)) or len(u_sym_list) != dim:
                raise ValueError(
                    f"manufactured_solution.u must be a {dim}-component list for linear_elasticity"
                )

            local_dict = _mms_local_dict(dim)
            coords = _mms_coords(dim)

            u_comps = [sp.sympify(u_sym_list[i], locals=local_dict) for i in range(dim)]

            lam_s = sp.sympify(lam)
            mu_s = sp.sympify(mu)

            # Build full strain tensor eps[i][j] = sym(grad u)[i][j]
            eps = [[sp.Rational(1, 2) * (sp.diff(u_comps[i], coords[j]) + sp.diff(u_comps[j], coords[i]))
                    for j in range(dim)] for i in range(dim)]
            tr_eps = sum(eps[i][i] for i in range(dim))

            # sigma[i][j] = 2*mu*eps[i][j] + lam*tr(eps)*delta[i][j]
            sigma = [[2 * mu_s * eps[i][j] + (lam_s * tr_eps if i == j else 0)
                      for j in range(dim)] for i in range(dim)]

            # f[i] = -div(sigma)[i] = -sum_j d sigma[i][j] / d x_j
            f_comps = [-(sum(sp.diff(sigma[i][j], coords[j]) for j in range(dim)))
                       for i in range(dim)]

            x = ufl.SpatialCoordinate(msh)
            f_expr = parse_vector_expression(f_comps, x)

            u_exact_expr = parse_vector_expression(u_comps, x)
            u_exact = fem.Function(V)
            interpolate_expression(u_exact, u_exact_expr)
        else:
            f_expr = _rhs_vector(msh, source_expr, dim)

        def eps(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(dim)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma(u), eps(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx

        bc_cfg = case_spec.get("bc", {}).get("dirichlet")
        if bc_cfg is None:
            bc_cfg = {"on": "all", "value": "u" if u_exact is not None else ["0.0", "0.0"]}
        bcs = _build_dirichlet_bcs(msh, V, bc_cfg, u_exact, dim)
        if not bcs:
            raise ValueError("linear_elasticity requires Dirichlet boundary conditions")

        solver_params = case_spec.get("oracle_solver", {})
        petsc_options: Dict[str, Any] = {
            "ksp_type": solver_params.get("ksp_type", "cg"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
            "ksp_atol": solver_params.get("atol", 1e-12),
        }
        if "pc_factor_mat_solver_type" in solver_params:
            petsc_options["pc_factor_mat_solver_type"] = solver_params["pc_factor_mat_solver_type"]

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="oracle_linear_elasticity_",
        )

        u_h = problem.solve()

        grid_cfg = case_spec["output"]["grid"]
        u_mag = _sample_vector_mag_grid(u_h, grid_cfg)

        baseline_error = 0.0
        solver_info: Dict[str, Any] = {
            "lambda": lam,
            "mu": mu,
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
            "mesh_resolution": case_spec["mesh"].get("resolution"),
            "cell_type": case_spec["mesh"].get("cell_type", "tetrahedron" if dim == 3 else "triangle"),
        }

        if u_exact is not None:
            # 直接在格点上计算位移模长，避免 FEM 投影误差
            u_exact_mag = _eval_exact_vec_mag_on_grid(
                u_comps, coords, grid_cfg
            )
            baseline_error = compute_rel_L2_grid(u_mag, u_exact_mag)
            u_mag = u_exact_mag
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_dim = ref_msh.geometry.dim
            ref_V = create_vector_space(
                ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"]
            )
            ref_f_expr = _rhs_vector(ref_msh, source_expr, ref_dim)

            ref_u = ufl.TrialFunction(ref_V)
            ref_v = ufl.TestFunction(ref_V)
            ref_a = ufl.inner(
                2.0 * mu * ufl.sym(ufl.grad(ref_u))
                + lam * ufl.tr(ufl.sym(ufl.grad(ref_u))) * ufl.Identity(ref_dim),
                ufl.sym(ufl.grad(ref_v)),
            ) * ufl.dx
            ref_L = ufl.inner(ref_f_expr, ref_v) * ufl.dx

            ref_bcs = _build_dirichlet_bcs(ref_msh, ref_V, bc_cfg, None, ref_dim)
            ref_petsc = {
                "ksp_type": ref_solver.get("ksp_type", petsc_options["ksp_type"]),
                "pc_type": ref_solver.get("pc_type", petsc_options["pc_type"]),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
                "ksp_atol": ref_solver.get("atol", 1e-14),
            }
            if "pc_factor_mat_solver_type" in ref_solver:
                ref_petsc["pc_factor_mat_solver_type"] = ref_solver["pc_factor_mat_solver_type"]

            ref_problem = LinearProblem(
                ref_a,
                ref_L,
                bcs=ref_bcs,
                petsc_options=ref_petsc,
                petsc_options_prefix="oracle_linear_elasticity_ref_",
            )
            ref_u_h = ref_problem.solve()
            ref_mag = _sample_vector_mag_grid(ref_u_h, grid_cfg)
            baseline_error = compute_rel_L2_grid(u_mag, ref_mag)
            u_mag = ref_mag
            solver_info["reference_resolution"] = ref_mesh_spec.get("resolution")
            solver_info["reference_degree"] = ref_fem_spec.get("degree")

        # ⏱️ 结束计时（包含完整流程）
        baseline_time = time.perf_counter() - t_start_total

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_mag,
            solver_info=solver_info,
            num_dofs=V.dofmap.index_map.size_global,
        )

