"""Firedrake Linear Elasticity oracle (vector-valued, 2D small strain)."""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl

from firedrake import (
    Function, TrialFunction, TestFunction, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, dx, solve,
    sym, Identity, tr, as_vector, as_tensor,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_vector_space,
    parse_expression, parse_vector_expression,
    sample_vector_magnitude_on_grid,
    _scalar_solver_params,
)


_NAMED_BOUNDARY_MAP = {
    "y0": 1, "ymin": 1,
    "x1": 2, "xmax": 2,
    "y1": 3, "ymax": 3,
    "x0": 4, "xmin": 4,
}


def _lame(params: Dict[str, Any]):
    if "lambda" in params and "mu" in params:
        return float(params["lambda"]), float(params["mu"])
    E = float(params.get("E", 1.0))
    nu = float(params.get("nu", 0.3))
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(lam), float(mu)


def _to_firedrake_boundary(on: str):
    """Convert benchmark boundary label to Firedrake boundary marker."""
    key = str(on).lower()
    if key in {"all", "*"}:
        return "on_boundary"
    if key in _NAMED_BOUNDARY_MAP:
        return _NAMED_BOUNDARY_MAP[key]
    try:
        return int(on)
    except (ValueError, TypeError):
        raise ValueError(f"Unknown boundary label '{on}'")


def _build_vector_rhs(V, value, x):
    """Build a mesh-bound vector RHS expression for elasticity."""
    dim = V.value_size
    rhs_fn = Function(V)
    if value is None:
        return rhs_fn

    if isinstance(value, (list, tuple)):
        rhs_fn.interpolate(parse_vector_expression(value, x))
        return rhs_fn

    rhs_fn.interpolate(as_vector([parse_expression(value, x)] * dim))
    return rhs_fn


def _build_vector_bcs(V, bc_cfg, x):
    """Build vector DirichletBCs with safe Constant/Function values."""
    dim = V.value_size
    dirichlet = bc_cfg if bc_cfg is not None else {"on": "all", "value": [0.0] * dim}
    if isinstance(dirichlet, dict):
        dirichlet = [dirichlet]

    bcs = []
    for cfg in dirichlet:
        boundary = _to_firedrake_boundary(cfg.get("on", "all"))
        value = cfg.get("value", [0.0] * dim)

        if isinstance(value, (list, tuple)):
            float_vals = []
            all_const = True
            for v_ in value:
                try:
                    float_vals.append(float(sp.sympify(str(v_))))
                except Exception:
                    all_const = False
                    break

            if all_const:
                bc_val = Constant(float_vals)
            else:
                bc_fn = Function(V)
                bc_fn.interpolate(parse_vector_expression(value, x))
                bc_val = bc_fn
        else:
            try:
                c = float(sp.sympify(str(value)))
                bc_val = Constant([c] * dim)
            except Exception:
                bc_fn = Function(V)
                bc_fn.interpolate(as_vector([parse_expression(value, x)] * dim))
                bc_val = bc_fn

        bcs.append(DirichletBC(V, bc_val, boundary))
    return bcs


class FiredrakeLinearElasticitySolver:
    """2D linear elasticity oracle using Firedrake."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        degree = case_spec["fem"].get("degree", 1)
        family = case_spec["fem"].get("family", "Lagrange")
        V = create_vector_space(msh, family, degree)
        x = SpatialCoordinate(msh)

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        lam, mu = _lame(params)

        def eps(u):
            return sym(grad(u))

        def sigma(u):
            return 2.0 * mu * eps(u) + lam * tr(eps(u)) * Identity(2)

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact_fn = None
        f_ufl = None

        if "u" in manufactured:
            sx, sy = sp.symbols("x y", real=True)
            u_sym_vec = [sp.sympify(s, locals={"x": sx, "y": sy, "pi": sp.pi})
                         for s in manufactured["u"]]
            lam_s, mu_s = sp.sympify(lam), sp.sympify(mu)

            def eps_sym(uvec):
                return [[sp.diff(uvec[i], [sx, sy][j]) + sp.diff(uvec[j], [sx, sy][i])
                         for j in range(2)] for i in range(2)]

            e = eps_sym(u_sym_vec)
            div_e = [(sp.diff(e[i][0], sx) + sp.diff(e[i][1], sy)) for i in range(2)]
            trace = e[0][0] + e[1][1]
            grad_trace = [sp.diff(trace, sx), sp.diff(trace, sy)]
            f_sym = [
                -(2.0 * mu_s * div_e[i] + lam_s * grad_trace[i])
                for i in range(2)
            ]
            f_ufl = parse_vector_expression(f_sym, x)
            u_exact_fn = Function(V)
            u_exact_fn.interpolate(parse_vector_expression(u_sym_vec, x))
        elif source_expr is not None:
            f_ufl = _build_vector_rhs(V, source_expr, x)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(sigma(u), eps(v)) * dx
        if f_ufl is not None:
            L = inner(f_ufl, v) * dx
        else:
            L = inner(Function(V), v) * dx

        if u_exact_fn is not None:
            bcs = [DirichletBC(V, u_exact_fn, "on_boundary")]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet")
            bcs = _build_vector_bcs(V, bc_cfg, x)

        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = {
            "ksp_type": solver_params.get("ksp_type", "cg"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
        }

        uh = Function(V)
        solve(a == L, uh, bcs=bcs, solver_parameters=sp_dict)

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_vector_magnitude_on_grid(uh, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

        baseline_error = 0.0
        if u_exact_fn is not None:
            _, _, u_exact_grid = sample_vector_magnitude_on_grid(
                u_exact_fn, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver_cfg = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_vector_space(ref_msh, ref_fem_spec.get("family", family), ref_fem_spec.get("degree", degree))
            ref_x = SpatialCoordinate(ref_msh)

            if source_expr is not None:
                ref_f_ufl = _build_vector_rhs(ref_V, source_expr, ref_x)
            else:
                ref_f_ufl = None

            ref_u = TrialFunction(ref_V)
            ref_v = TestFunction(ref_V)
            ref_a = inner(sigma(ref_u), eps(ref_v)) * dx
            if ref_f_ufl is not None:
                ref_L = inner(ref_f_ufl, ref_v) * dx
            else:
                ref_L = inner(Function(ref_V), ref_v) * dx

            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet")
            ref_bcs = _build_vector_bcs(ref_V, ref_bc_cfg, ref_x)

            ref_sp = {
                "ksp_type": ref_solver_cfg.get("ksp_type", sp_dict["ksp_type"]),
                "pc_type": ref_solver_cfg.get("pc_type", sp_dict["pc_type"]),
                "ksp_rtol": ref_solver_cfg.get("rtol", 1e-12),
            }
            ref_uh = Function(ref_V)
            solve(ref_a == ref_L, ref_uh, bcs=ref_bcs, solver_parameters=ref_sp)

            _, _, ref_grid = sample_vector_magnitude_on_grid(ref_uh, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        baseline_time = time.perf_counter() - t_start
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info={
                "ksp_type": sp_dict["ksp_type"],
                "pc_type": sp_dict["pc_type"],
                "rtol": sp_dict["ksp_rtol"],
                "lam": lam,
                "mu": mu,
            },
            num_dofs=V.dof_count,
        )
