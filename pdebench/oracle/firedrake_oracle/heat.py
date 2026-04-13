"""Firedrake Heat equation oracle solver (backward Euler)."""
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


class FiredrakeHeatSolver:
    """Backward-Euler heat equation oracle using Firedrake."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        x = SpatialCoordinate(msh)
        dim = msh.geometric_dimension()

        pde_cfg = case_spec["pde"]
        coeffs = pde_cfg.get("coefficients", {})
        kappa_spec = coeffs.get("kappa", {"type": "constant", "value": 1.0})

        def _mms_symbols(with_time: bool = False):
            sx, sy, sz, st = sp.symbols("x y z t", real=True)
            locals_dict = {"x": sx, "y": sy, "pi": sp.pi}
            coords = [sx, sy]
            if dim >= 3:
                locals_dict["z"] = sz
                coords.append(sz)
            if with_time:
                locals_dict["t"] = st
            return locals_dict, tuple(coords), st

        if kappa_spec["type"] == "constant":
            kappa = Constant(float(kappa_spec["value"]))
        else:
            local_dict, _, _ = _mms_symbols()
            kappa_sym = sp.sympify(kappa_spec["expr"], locals=local_dict)
            kappa_fn = Function(V)
            kappa_fn.interpolate(parse_expression(kappa_sym, x))
            kappa = kappa_fn

        time_cfg = pde_cfg["time"]
        t0 = time_cfg.get("t0", 0.0)
        t_end = time_cfg["t_end"]
        dt_val = time_cfg.get("dt", 0.01)
        num_steps = int((t_end - t0) / dt_val + 0.999999)
        dt = Constant(dt_val)

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        initial_expr = pde_cfg.get("initial_condition")
        u_exact_sym = None
        f_sym = None

        if "u" in manufactured:
            local_dict, coords, st = _mms_symbols(with_time=True)
            u_sym = sp.sympify(manufactured["u"], locals=local_dict)
            u_t = sp.diff(u_sym, st)
            if kappa_spec["type"] == "expr":
                ks = sp.sympify(kappa_spec["expr"], locals=local_dict)
            else:
                ks = sp.sympify(kappa_spec.get("value", 1.0))
            f_sym = u_t - sum(sp.diff(ks * sp.diff(u_sym, c), c) for c in coords)
            u_exact_sym = u_sym
        elif source_expr is not None:
            local_dict, _, _ = _mms_symbols()
            f_sym = sp.sympify(source_expr, locals=local_dict)

        # Initial condition
        u_prev = Function(V)
        if u_exact_sym is not None:
            u_prev.interpolate(parse_expression(u_exact_sym, x, t=t0))
        elif initial_expr is not None:
            u_prev.interpolate(parse_expression(initial_expr, x, t=t0))

        # Build time-stepping forms
        u = TrialFunction(V)
        v = TestFunction(V)
        a = (u * v + dt * inner(kappa * grad(u), grad(v))) * dx

        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _scalar_solver_params(solver_params)

        uh = Function(V)
        t_cur = t0
        for _ in range(num_steps):
            t_cur += dt_val
            if f_sym is not None:
                f_ufl = parse_expression(f_sym, x, t=t_cur)
            else:
                f_ufl = Constant(0.0)
            L = (u_prev * v + dt * f_ufl * v) * dx

            # Update BC for time-dependent exact solution
            if u_exact_sym is not None:
                bc_fn = Function(V)
                bc_fn.interpolate(parse_expression(u_exact_sym, x, t=t_cur))
                bcs = [DirichletBC(V, bc_fn, "on_boundary")]
            else:
                bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x, t=t_cur)]

            solve(a == L, uh, bcs=bcs, solver_parameters=sp_dict)
            u_prev.assign(uh)

        grid_cfg = case_spec["output"]["grid"]
        sample_args = (grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg.get("nz"))
        *_, u_grid = sample_scalar_on_grid(u_prev, *sample_args)

        baseline_error = 0.0
        if u_exact_sym is not None:
            u_exact_grid = _eval_exact_sym_on_grid(u_sym, coords, grid_cfg, t=t_cur, t_sym=st)
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver_cfg = ref_cfg.get("oracle_solver", {})
            ref_time_cfg = ref_cfg.get("time", {})
            ref_dt_val = ref_time_cfg.get("dt", dt_val * 0.5)

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_x = SpatialCoordinate(ref_msh)

            if kappa_spec["type"] == "constant":
                ref_kappa = Constant(float(kappa_spec["value"]))
            else:
                ref_kappa_fn = Function(ref_V)
                ref_kappa_fn.interpolate(parse_expression(kappa_sym, ref_x))
                ref_kappa = ref_kappa_fn

            ref_u_prev = Function(ref_V)
            if initial_expr is not None:
                ref_u_prev.interpolate(parse_expression(initial_expr, ref_x, t=t0))

            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bc_value = bc_cfg.get("value", "0.0")

            ref_u_trial = TrialFunction(ref_V)
            ref_v_test = TestFunction(ref_V)
            ref_dt = Constant(ref_dt_val)
            ref_a = (ref_u_trial * ref_v_test + ref_dt * inner(ref_kappa * grad(ref_u_trial), grad(ref_v_test))) * dx

            ref_sp = _scalar_solver_params(ref_solver_cfg)
            ref_sp["ksp_rtol"] = ref_solver_cfg.get("rtol", 1e-12)
            ref_sp["ksp_atol"] = ref_solver_cfg.get("atol", 1e-14)

            ref_uh = Function(ref_V)
            ref_t = t0
            ref_num_steps = int((t_end - t0) / ref_dt_val + 0.999999)
            ref_bcs = [build_scalar_bc(ref_V, bc_value, ref_x, t=t0)]
            for _ in range(ref_num_steps):
                ref_t += ref_dt_val
                ref_f_ufl = parse_expression(f_sym, ref_x, t=ref_t) if f_sym is not None else Constant(0.0)
                ref_L = (ref_u_prev * ref_v_test + ref_dt * ref_f_ufl * ref_v_test) * dx
                solve(ref_a == ref_L, ref_uh, bcs=ref_bcs, solver_parameters=ref_sp)
                ref_u_prev.assign(ref_uh)

            *_, ref_grid = sample_scalar_on_grid(ref_u_prev, *sample_args)
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
                "num_timesteps": num_steps,
                "dt": dt_val,
            },
            num_dofs=V.dof_count,
        )
