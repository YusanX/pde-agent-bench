"""Firedrake Poisson oracle solver."""
from __future__ import annotations

import time
from typing import Any, Dict

import sympy as sp
import ufl

from firedrake import (
    Function, TrialFunction, TestFunction, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, dx, solve,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_scalar_space,
    parse_expression,
    build_scalar_bc,
    sample_scalar_on_grid,
    _scalar_solver_params,
    _eval_exact_sym_on_grid,
)


class FiredrakePoissonSolver:
    """Poisson equation oracle using Firedrake."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        x = SpatialCoordinate(msh)
        dim = msh.geometric_dimension()

        pde_cfg = case_spec["pde"]
        coeffs = pde_cfg.get("coefficients", {})
        kappa_spec = coeffs.get("kappa", {"type": "constant", "value": 1.0})

        def _mms_symbols():
            sx, sy, sz = sp.symbols("x y z", real=True)
            locals_dict = {"x": sx, "y": sy, "pi": sp.pi}
            coords = [sx, sy]
            if dim >= 3:
                locals_dict["z"] = sz
                coords.append(sz)
            return locals_dict, tuple(coords)

        # Build kappa field
        if kappa_spec["type"] == "constant":
            kappa = Constant(float(kappa_spec["value"]))
        else:
            local_dict, _ = _mms_symbols()
            kappa_sym = sp.sympify(kappa_spec["expr"], locals=local_dict)
            kappa_ufl = parse_expression(kappa_sym, x)
            kappa_fn = Function(V)
            kappa_fn.interpolate(kappa_ufl)
            kappa = kappa_fn

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact_fn = None
        f_ufl = None

        if "u" in manufactured:
            local_dict, coords = _mms_symbols()
            u_sym = sp.sympify(manufactured["u"], locals=local_dict)
            if kappa_spec["type"] == "expr":
                kappa_sym = sp.sympify(kappa_spec["expr"], locals=local_dict)
            else:
                kappa_sym = sp.sympify(kappa_spec.get("value", 1.0))
            f_sym = -sum(sp.diff(kappa_sym * sp.diff(u_sym, c), c) for c in coords)
            f_ufl = parse_expression(f_sym, x)
            u_exact_fn = Function(V)
            u_exact_fn.interpolate(parse_expression(u_sym, x))
        elif source_expr is not None:
            try:
                f_ufl = Constant(float(sp.sympify(source_expr)))
            except Exception:
                f_ufl = parse_expression(source_expr, x)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(kappa * grad(u), grad(v)) * dx
        L = (f_ufl if f_ufl is not None else Constant(0.0)) * v * dx

        # Boundary conditions
        if u_exact_fn is not None:
            bcs = [DirichletBC(V, u_exact_fn, "on_boundary")]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x)]

        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _scalar_solver_params(solver_params)

        uh = Function(V)
        solve(a == L, uh, bcs=bcs, solver_parameters=sp_dict)

        grid_cfg = case_spec["output"]["grid"]
        sample_args = (grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg.get("nz"))
        *_, u_grid = sample_scalar_on_grid(uh, *sample_args)

        baseline_error = 0.0
        if u_exact_fn is not None:
            u_exact_grid = _eval_exact_sym_on_grid(u_sym, coords, grid_cfg)
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_x = SpatialCoordinate(ref_msh)

            if kappa_spec["type"] == "constant":
                ref_kappa = Constant(float(kappa_spec["value"]))
            else:
                ref_kappa_fn = Function(ref_V)
                ref_kappa_fn.interpolate(parse_expression(kappa_sym, ref_x))
                ref_kappa = ref_kappa_fn

            if source_expr is not None:
                try:
                    ref_f = Constant(float(sp.sympify(source_expr)))
                except Exception:
                    ref_f = parse_expression(source_expr, ref_x)
            else:
                ref_f = Constant(0.0)

            ref_u = TrialFunction(ref_V)
            ref_v = TestFunction(ref_V)
            ref_a = inner(ref_kappa * grad(ref_u), grad(ref_v)) * dx
            ref_L = ref_f * ref_v * dx
            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            ref_bcs = [build_scalar_bc(ref_V, ref_bc_cfg.get("value", "0.0"), ref_x)]
            ref_uh = Function(ref_V)
            ref_sp = _scalar_solver_params(ref_solver)
            ref_sp["ksp_rtol"] = ref_solver.get("rtol", 1e-12)
            solve(ref_a == ref_L, ref_uh, bcs=ref_bcs, solver_parameters=ref_sp)

            *_, ref_grid = sample_scalar_on_grid(ref_uh, *sample_args)
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
            },
            num_dofs=V.dof_count,
        )
