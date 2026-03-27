"""Firedrake Reaction-Diffusion oracle (steady + transient, linear + nonlinear)."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import sympy as sp
import ufl

from firedrake import (
    Function, TrialFunction, TestFunction, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, dx, solve,
    derivative,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_scalar_space,
    parse_expression,
    build_scalar_bc,
    sample_scalar_on_grid,
    _scalar_solver_params,
)


def _reaction_ufl(u, reaction: Dict[str, Any]) -> Tuple[ufl.core.expr.Expr, bool]:
    rtype = str(reaction.get("type", "linear")).lower()
    if rtype == "linear":
        alpha = float(reaction.get("alpha", 0.0))
        return alpha * u, False
    if rtype in {"cubic", "poly3"}:
        alpha = float(reaction.get("alpha", 0.0))
        beta = float(reaction.get("beta", 1.0))
        return alpha * u + beta * u**3, True
    if rtype in {"allen_cahn", "allen-cahn"}:
        lam = float(reaction.get("lambda", reaction.get("lam", 1.0)))
        return lam * (u**3 - u), True
    if rtype in {"logistic", "fisher_kpp", "fisher-kpp"}:
        rho = float(reaction.get("rho", 1.0))
        return rho * u * (1 - u), True
    raise ValueError(f"Unsupported reaction type: {rtype}")


def _reaction_sym(u_s, reaction: Dict[str, Any]) -> Tuple[sp.Expr, bool]:
    rtype = str(reaction.get("type", "linear")).lower()
    if rtype == "linear":
        alpha = sp.sympify(reaction.get("alpha", 0.0))
        return alpha * u_s, False
    if rtype in {"cubic", "poly3"}:
        alpha = sp.sympify(reaction.get("alpha", 0.0))
        beta = sp.sympify(reaction.get("beta", 1.0))
        return alpha * u_s + beta * u_s**3, True
    if rtype in {"allen_cahn", "allen-cahn"}:
        lam = sp.sympify(reaction.get("lambda", reaction.get("lam", 1.0)))
        return lam * (u_s**3 - u_s), True
    if rtype in {"logistic", "fisher_kpp", "fisher-kpp"}:
        rho = sp.sympify(reaction.get("rho", 1.0))
        return rho * u_s * (1 - u_s), True
    raise ValueError(f"Unsupported reaction type: {rtype}")


class FiredrakeReactionDiffusionSolver:
    """Reaction-diffusion oracle (steady and transient) using Firedrake."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        x = SpatialCoordinate(msh)

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        epsilon = float(params.get("epsilon", 1.0))
        reaction = params.get("reaction", {"type": "linear", "alpha": 0.0})
        source_expr = pde_cfg.get("source_term")
        time_cfg = pde_cfg.get("time")

        manufactured = pde_cfg.get("manufactured_solution", {})
        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _scalar_solver_params(solver_params)

        # ── STEADY ──────────────────────────────────────────────────────────
        if time_cfg is None:
            u_exact_fn = None
            f_ufl = None
            u_exact_sym = None

            if "u" in manufactured:
                sx, sy = sp.symbols("x y", real=True)
                u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy})
                R_sym, is_nonlin = _reaction_sym(u_sym, reaction)
                f_sym = -epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)) + R_sym
                if source_expr is not None:
                    src = sp.sympify(source_expr, locals={"x": sx, "y": sy})
                    f_sym = f_sym - src
                f_ufl = parse_expression(f_sym, x)
                u_exact_fn = Function(V)
                u_exact_fn.interpolate(parse_expression(u_sym, x))
                u_exact_sym = u_sym
            elif source_expr is not None:
                try:
                    f_ufl = Constant(float(sp.sympify(source_expr)))
                except Exception:
                    f_ufl = parse_expression(source_expr, x)

            v = TestFunction(V)
            R_ufl, is_nonlinear = _reaction_ufl(Function(V), reaction)  # probe nonlinearity
            is_nonlinear = is_nonlinear or (reaction.get("type", "linear") != "linear")

            if not is_nonlinear:
                # Linear problem
                u = TrialFunction(V)
                R_lin, _ = _reaction_ufl(u, reaction)
                a = (epsilon * inner(grad(u), grad(v)) + R_lin * v) * dx
                L = (f_ufl if f_ufl is not None else Constant(0.0)) * v * dx
                if u_exact_fn is not None:
                    bcs = [DirichletBC(V, u_exact_fn, "on_boundary")]
                else:
                    bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                    bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x)]
                uh = Function(V)
                solve(a == L, uh, bcs=bcs, solver_parameters=sp_dict)
            else:
                # Nonlinear problem
                uh = Function(V)
                if u_exact_fn is not None:
                    uh.interpolate(u_exact_fn)  # good initial guess
                R_nl, _ = _reaction_ufl(uh, reaction)
                F = (epsilon * inner(grad(uh), grad(v)) + R_nl * v) * dx
                if f_ufl is not None:
                    F -= f_ufl * v * dx
                if u_exact_fn is not None:
                    bcs = [DirichletBC(V, u_exact_fn, "on_boundary")]
                else:
                    bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                    bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x)]
                nl_params = {
                    "snes_type": "newtonls",
                    "snes_rtol": solver_params.get("rtol", 1e-10),
                    "snes_atol": solver_params.get("atol", 1e-12),
                    "snes_max_it": solver_params.get("max_it", 50),
                    "ksp_type": solver_params.get("ksp_type", "gmres"),
                    "pc_type": solver_params.get("pc_type", "ilu"),
                    "ksp_rtol": solver_params.get("ksp_rtol", 1e-10),
                }
                problem = NonlinearVariationalProblem(F, uh, bcs=bcs)
                solver = NonlinearVariationalSolver(problem, solver_parameters=nl_params)
                solver.solve()

        # ── TRANSIENT (Backward Euler) ────────────────────────────────────
        else:
            t0 = time_cfg.get("t0", 0.0)
            t_end = time_cfg["t_end"]
            dt_val = time_cfg.get("dt", 0.01)
            num_steps = int((t_end - t0) / dt_val + 0.999999)
            dt = Constant(dt_val)

            u_exact_sym = None
            f_sym = None

            if "u" in manufactured:
                sx, sy, st = sp.symbols("x y t", real=True)
                u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "t": st})
                u_t = sp.diff(u_sym, st)
                R_sym, _ = _reaction_sym(u_sym, reaction)
                f_sym = u_t - epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)) + R_sym
                u_exact_sym = u_sym
            elif source_expr is not None:
                sx, sy = sp.symbols("x y", real=True)
                f_sym = sp.sympify(source_expr, locals={"x": sx, "y": sy})

            u_prev = Function(V)
            if u_exact_sym is not None:
                u_prev.interpolate(parse_expression(u_exact_sym, x, t=t0))
            elif pde_cfg.get("initial_condition"):
                u_prev.interpolate(parse_expression(pde_cfg["initial_condition"], x, t=t0))

            _, is_nonlinear = _reaction_ufl(u_prev, reaction)

            uh = Function(V)
            t_cur = t0
            for _ in range(num_steps):
                t_cur += dt_val
                f_ufl = parse_expression(f_sym, x, t=t_cur) if f_sym is not None else Constant(0.0)
                v = TestFunction(V)
                if u_exact_sym is not None:
                    bc_fn = Function(V)
                    bc_fn.interpolate(parse_expression(u_exact_sym, x, t=t_cur))
                    bcs = [DirichletBC(V, bc_fn, "on_boundary")]
                else:
                    bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                    bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x, t=t_cur)]

                if not is_nonlinear:
                    u = TrialFunction(V)
                    R_lin, _ = _reaction_ufl(u, reaction)
                    a = (u * v + dt * epsilon * inner(grad(u), grad(v)) + dt * R_lin * v) * dx
                    L = (u_prev * v + dt * f_ufl * v) * dx
                    solve(a == L, uh, bcs=bcs, solver_parameters=sp_dict)
                else:
                    uh.assign(u_prev)
                    R_nl, _ = _reaction_ufl(uh, reaction)
                    F = (
                        (uh - u_prev) * v / dt * dx
                        + epsilon * inner(grad(uh), grad(v)) * dx
                        + R_nl * v * dx
                        - f_ufl * v * dx
                    )
                    nl_params = {
                        "snes_type": "newtonls",
                        "snes_rtol": solver_params.get("rtol", 1e-10),
                        "snes_max_it": solver_params.get("max_it", 20),
                        "ksp_type": solver_params.get("ksp_type", "gmres"),
                        "pc_type": solver_params.get("pc_type", "ilu"),
                    }
                    problem = NonlinearVariationalProblem(F, uh, bcs=bcs)
                    solver = NonlinearVariationalSolver(problem, solver_parameters=nl_params)
                    solver.solve()
                u_prev.assign(uh)
            uh = u_prev

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(uh, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

        # u_exact_fn is only defined in the steady branch; default to None for transient.
        _u_exact_fn = locals().get("u_exact_fn", None)

        baseline_error = 0.0
        if _u_exact_fn is not None:
            _, _, u_exact_grid = sample_scalar_on_grid(
                _u_exact_fn, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        elif u_exact_sym is not None:
            u_ex = Function(V)
            u_ex.interpolate(parse_expression(u_exact_sym, x, t=t_cur))
            _, _, u_exact_grid = sample_scalar_on_grid(
                u_ex, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            # No exact solution: run a higher-resolution reference solver.
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver_cfg = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
            ref_x = SpatialCoordinate(ref_msh)

            if source_expr is not None:
                try:
                    ref_f_ufl = Constant(float(sp.sympify(source_expr)))
                except Exception:
                    ref_f_ufl = parse_expression(source_expr, ref_x)
            else:
                ref_f_ufl = Constant(0.0)

            ref_sp = _scalar_solver_params(ref_solver_cfg)
            ref_sp["ksp_rtol"] = ref_solver_cfg.get("rtol", 1e-12)

            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            ref_bcs = [build_scalar_bc(ref_V, ref_bc_cfg.get("value", "0.0"), ref_x)]

            if time_cfg is None:
                # Steady reference solve
                ref_v = TestFunction(ref_V)
                _, ref_is_nl = _reaction_ufl(Function(ref_V), reaction)
                if not ref_is_nl:
                    ref_u = TrialFunction(ref_V)
                    ref_R, _ = _reaction_ufl(ref_u, reaction)
                    ref_a = (epsilon * inner(grad(ref_u), grad(ref_v)) + ref_R * ref_v) * dx
                    ref_L = ref_f_ufl * ref_v * dx
                    ref_uh = Function(ref_V)
                    solve(ref_a == ref_L, ref_uh, bcs=ref_bcs, solver_parameters=ref_sp)
                else:
                    ref_uh = Function(ref_V)
                    ref_R_nl, _ = _reaction_ufl(ref_uh, reaction)
                    ref_F = (epsilon * inner(grad(ref_uh), grad(ref_v)) + ref_R_nl * ref_v) * dx
                    ref_F -= ref_f_ufl * ref_v * dx
                    nl_params = {
                        "snes_type": "newtonls",
                        "snes_rtol": ref_solver_cfg.get("rtol", 1e-12),
                        "snes_atol": ref_solver_cfg.get("atol", 1e-14),
                        "snes_max_it": solver_params.get("max_it", 50),
                        "ksp_type": ref_solver_cfg.get("ksp_type", "gmres"),
                        "pc_type": ref_solver_cfg.get("pc_type", "ilu"),
                        "ksp_rtol": ref_solver_cfg.get("ksp_rtol", 1e-12),
                    }
                    ref_problem = NonlinearVariationalProblem(ref_F, ref_uh, bcs=ref_bcs)
                    ref_solver = NonlinearVariationalSolver(ref_problem, solver_parameters=nl_params)
                    ref_solver.solve()
            else:
                # Transient reference solve
                ref_time_cfg = ref_cfg.get("time", {})
                ref_dt_val = ref_time_cfg.get("dt", time_cfg.get("dt", 0.01) * 0.5)
                ref_t0 = time_cfg.get("t0", 0.0)
                ref_t_end = time_cfg["t_end"]
                ref_num_steps = int((ref_t_end - ref_t0) / ref_dt_val + 0.999999)
                ref_dt = Constant(ref_dt_val)

                ref_u_prev = Function(ref_V)
                if pde_cfg.get("initial_condition"):
                    ref_u_prev.interpolate(parse_expression(pde_cfg["initial_condition"], ref_x, t=ref_t0))

                _, ref_is_nl = _reaction_ufl(ref_u_prev, reaction)
                ref_uh = Function(ref_V)
                for _ in range(ref_num_steps):
                    ref_v = TestFunction(ref_V)
                    if not ref_is_nl:
                        ref_u = TrialFunction(ref_V)
                        ref_R_lin, _ = _reaction_ufl(ref_u, reaction)
                        ref_a = (ref_u * ref_v + ref_dt * epsilon * inner(grad(ref_u), grad(ref_v)) + ref_dt * ref_R_lin * ref_v) * dx
                        ref_L = (ref_u_prev * ref_v + ref_dt * ref_f_ufl * ref_v) * dx
                        solve(ref_a == ref_L, ref_uh, bcs=ref_bcs, solver_parameters=ref_sp)
                    else:
                        ref_uh.assign(ref_u_prev)
                        ref_R_nl, _ = _reaction_ufl(ref_uh, reaction)
                        ref_F = (
                            (ref_uh - ref_u_prev) * ref_v / ref_dt * dx
                            + epsilon * inner(grad(ref_uh), grad(ref_v)) * dx
                            + ref_R_nl * ref_v * dx
                            - ref_f_ufl * ref_v * dx
                        )
                        nl_params = {
                            "snes_type": "newtonls",
                            "snes_rtol": ref_solver_cfg.get("rtol", 1e-12),
                            "snes_max_it": solver_params.get("max_it", 20),
                            "ksp_type": ref_solver_cfg.get("ksp_type", "gmres"),
                            "pc_type": ref_solver_cfg.get("pc_type", "ilu"),
                        }
                        ref_problem = NonlinearVariationalProblem(ref_F, ref_uh, bcs=ref_bcs)
                        ref_nl_solver = NonlinearVariationalSolver(ref_problem, solver_parameters=nl_params)
                        ref_nl_solver.solve()
                    ref_u_prev.assign(ref_uh)

            _, _, ref_grid = sample_scalar_on_grid(ref_uh, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])
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
                "epsilon": epsilon,
            },
            num_dofs=V.dof_count,
        )
