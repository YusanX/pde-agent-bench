"""Firedrake Convection-Diffusion oracle (steady + transient, optional SUPG)."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import sympy as sp
import ufl

from firedrake import (
    Function, TrialFunction, TestFunction, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, div, dot, dx, solve,
    as_vector, CellDiameter, sqrt,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_scalar_space,
    parse_expression,
    build_scalar_bc,
    sample_scalar_on_grid,
    _scalar_solver_params,
)


def _is_linear_solve_failure(exc: Exception) -> bool:
    """Return True when Firedrake/PETSc failed in the linear solve stage."""
    msg = str(exc)
    return (
        "DIVERGED_LINEAR_SOLVE" in msg
        or "Linear solve failed to converge" in msg
        or "Nonlinear solve failed to converge after 0 nonlinear iterations" in msg
    )


def _build_cd_solver_parameters(
    solver_params: Dict[str, Any],
    *,
    default_ksp: str = "cg",
    default_pc: str = "hypre",
    default_rtol: float = 1e-10,
    default_atol: float = 1e-12,
) -> Dict[str, Any]:
    """Build PETSc options for convection-diffusion solves."""
    return {
        "ksp_type": solver_params.get("ksp_type", default_ksp),
        "pc_type": solver_params.get("pc_type", default_pc),
        "ksp_rtol": solver_params.get("rtol", default_rtol),
        "ksp_atol": solver_params.get("atol", default_atol),
    }


def _fallback_cd_solver_parameters(sp_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate more robust solver fallbacks for difficult transport solves."""
    fallbacks: List[Dict[str, Any]] = []

    if sp_dict.get("pc_type") == "ilu":
        gmres_hypre = dict(sp_dict)
        gmres_hypre["pc_type"] = "hypre"
        fallbacks.append(gmres_hypre)

    if not (sp_dict.get("ksp_type") == "preonly" and sp_dict.get("pc_type") == "lu"):
        lu_direct = dict(sp_dict)
        lu_direct["ksp_type"] = "preonly"
        lu_direct["pc_type"] = "lu"
        fallbacks.append(lu_direct)

    return fallbacks


def _solve_linear_problem(a, L, V, bcs, solver_parameters: Dict[str, Any], *, allow_fallback: bool = False):
    """Solve a linear scalar problem, retrying with safer settings if needed."""
    attempts = [solver_parameters]
    if allow_fallback:
        attempts.extend(_fallback_cd_solver_parameters(solver_parameters))

    last_exc = None
    for params in attempts:
        uh = Function(V)
        try:
            solve(a == L, uh, bcs=bcs, solver_parameters=params)
            return uh, params
        except Exception as exc:
            last_exc = exc
            if not allow_fallback or not _is_linear_solve_failure(exc):
                raise

    raise last_exc


class FiredrakeConvectionDiffusionSolver:
    """Convection-diffusion oracle with optional SUPG stabilization."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        x = SpatialCoordinate(msh)

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        epsilon = float(params.get("epsilon", 0.01))
        beta = params.get("beta", [1.0, 1.0])
        beta_vec = as_vector(beta)
        source_expr = pde_cfg.get("source_term")
        time_cfg = pde_cfg.get("time")

        manufactured = pde_cfg.get("manufactured_solution", {})
        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _build_cd_solver_parameters(
            solver_params,
            default_ksp=solver_params.get("ksp_type", "cg"),
            default_pc=solver_params.get("pc_type", "hypre"),
        )
        stabilization = solver_params.get("stabilization", params.get("stabilization"))

        # ── STEADY ──────────────────────────────────────────────────────────
        if time_cfg is None:
            u_exact_fn = None
            f_ufl = None

            if "u" in manufactured:
                sx, sy = sp.symbols("x y", real=True)
                u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy})
                bx, by = beta
                f_sym = (
                    -epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2))
                    + bx * sp.diff(u_sym, sx)
                    + by * sp.diff(u_sym, sy)
                )
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
            a = (epsilon * inner(grad(u), grad(v)) + dot(beta_vec, grad(u)) * v) * dx
            L = (f_ufl if f_ufl is not None else Constant(0.0)) * v * dx

            if stabilization == "supg":
                h = CellDiameter(msh)
                beta_norm = sqrt(dot(beta_vec, beta_vec))
                upwind_param = float(solver_params.get("upwind_parameter", 1.0))
                tau = upwind_param * h / (2.0 * beta_norm + Constant(1e-12))
                a += tau * dot(beta_vec, grad(v)) * (dot(beta_vec, grad(u)) - epsilon * div(grad(u))) * dx
                if f_ufl is not None:
                    L += tau * dot(beta_vec, grad(v)) * f_ufl * dx

            if u_exact_fn is not None:
                bcs = [DirichletBC(V, u_exact_fn, "on_boundary")]
            else:
                bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x)]

            uh, used_sp_dict = _solve_linear_problem(
                a, L, V, bcs, sp_dict, allow_fallback=True
            )

            grid_cfg = case_spec["output"]["grid"]
            _, _, u_grid = sample_scalar_on_grid(uh, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

            baseline_error = 0.0
            if u_exact_fn is not None:
                _, _, u_exact_grid = sample_scalar_on_grid(
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
                ref_V = create_scalar_space(ref_msh, ref_fem_spec["family"], ref_fem_spec["degree"])
                ref_x = SpatialCoordinate(ref_msh)

                if source_expr is not None:
                    try:
                        ref_f = Constant(float(sp.sympify(source_expr)))
                    except Exception:
                        ref_f = parse_expression(source_expr, ref_x)
                else:
                    ref_f = Constant(0.0)

                ref_u = TrialFunction(ref_V)
                ref_v = TestFunction(ref_V)
                ref_a = (epsilon * inner(grad(ref_u), grad(ref_v)) + dot(beta_vec, grad(ref_u)) * ref_v) * dx
                ref_L = ref_f * ref_v * dx

                if stabilization == "supg":
                    ref_h = CellDiameter(ref_msh)
                    beta_norm = sqrt(dot(beta_vec, beta_vec))
                    upwind_param = float(solver_params.get("upwind_parameter", 1.0))
                    ref_tau = upwind_param * ref_h / (2.0 * beta_norm + Constant(1e-12))
                    ref_a += ref_tau * dot(beta_vec, grad(ref_v)) * (dot(beta_vec, grad(ref_u)) - epsilon * div(grad(ref_u))) * dx
                    if source_expr is not None:
                        ref_L += ref_tau * dot(beta_vec, grad(ref_v)) * ref_f * dx

                ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                ref_bcs = [build_scalar_bc(ref_V, ref_bc_cfg.get("value", "0.0"), ref_x)]
                ref_sp = _build_cd_solver_parameters(
                    ref_solver_cfg,
                    default_ksp=sp_dict["ksp_type"],
                    default_pc=sp_dict["pc_type"],
                    default_rtol=1e-12,
                    default_atol=sp_dict["ksp_atol"],
                )
                ref_uh, _ = _solve_linear_problem(
                    ref_a, ref_L, ref_V, ref_bcs, ref_sp, allow_fallback=True
                )

                _, _, ref_grid = sample_scalar_on_grid(ref_uh, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])
                baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
                u_grid = ref_grid

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
                bx, by = beta
                f_sym = (
                    u_t
                    - epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2))
                    + bx * sp.diff(u_sym, sx)
                    + by * sp.diff(u_sym, sy)
                )
                u_exact_sym = u_sym
            elif source_expr is not None:
                sx, sy, st = sp.symbols("x y t", real=True)
                f_sym = sp.sympify(source_expr, locals={"x": sx, "y": sy, "t": st})

            u_prev = Function(V)
            if u_exact_sym is not None:
                u_prev.interpolate(parse_expression(u_exact_sym, x, t=t0))
            elif pde_cfg.get("initial_condition"):
                u_prev.interpolate(parse_expression(pde_cfg["initial_condition"], x, t=t0))

            u = TrialFunction(V)
            v = TestFunction(V)
            a = (u * v + dt * epsilon * inner(grad(u), grad(v)) + dt * dot(beta_vec, grad(u)) * v) * dx

            uh = Function(V)
            t_cur = t0
            for _ in range(num_steps):
                t_cur += dt_val
                f_ufl = parse_expression(f_sym, x, t=t_cur) if f_sym is not None else Constant(0.0)
                L = (u_prev * v + dt * f_ufl * v) * dx
                if u_exact_sym is not None:
                    bc_fn = Function(V)
                    bc_fn.interpolate(parse_expression(u_exact_sym, x, t=t_cur))
                    bcs = [DirichletBC(V, bc_fn, "on_boundary")]
                else:
                    bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                    bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x, t=t_cur)]
                uh, used_sp_dict = _solve_linear_problem(
                    a, L, V, bcs, sp_dict, allow_fallback=True
                )
                u_prev.assign(uh)

            grid_cfg = case_spec["output"]["grid"]
            _, _, u_grid = sample_scalar_on_grid(u_prev, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

            baseline_error = 0.0
            if u_exact_sym is not None:
                u_exact_fn = Function(V)
                u_exact_fn.interpolate(parse_expression(u_exact_sym, x, t=t_cur))
                _, _, u_exact_grid = sample_scalar_on_grid(
                    u_exact_fn, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
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

                ref_u_prev = Function(ref_V)
                if pde_cfg.get("initial_condition"):
                    ref_u_prev.interpolate(parse_expression(pde_cfg["initial_condition"], ref_x, t=t0))

                ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bc_value = ref_bc_cfg.get("value", "0.0")

                ref_u = TrialFunction(ref_V)
                ref_v = TestFunction(ref_V)
                ref_dt = Constant(ref_dt_val)
                ref_a = (ref_u * ref_v + ref_dt * epsilon * inner(grad(ref_u), grad(ref_v)) + ref_dt * dot(beta_vec, grad(ref_u)) * ref_v) * dx

                ref_sp = _build_cd_solver_parameters(
                    ref_solver_cfg,
                    default_ksp=sp_dict["ksp_type"],
                    default_pc=sp_dict["pc_type"],
                    default_rtol=1e-12,
                    default_atol=1e-14,
                )

                ref_t = t0
                ref_num_steps = int((t_end - t0) / ref_dt_val + 0.999999)
                ref_bcs = [build_scalar_bc(ref_V, bc_value, ref_x, t=t0)]
                for _ in range(ref_num_steps):
                    ref_t += ref_dt_val
                    ref_f_ufl = parse_expression(f_sym, ref_x, t=ref_t) if f_sym is not None else Constant(0.0)
                    ref_L = (ref_u_prev * ref_v + ref_dt * ref_f_ufl * ref_v) * dx
                    ref_uh, _ = _solve_linear_problem(
                        ref_a, ref_L, ref_V, ref_bcs, ref_sp, allow_fallback=True
                    )
                    ref_u_prev.assign(ref_uh)

                _, _, ref_grid = sample_scalar_on_grid(ref_u_prev, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])
                baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
                u_grid = ref_grid

        baseline_time = time.perf_counter() - t_start
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info={
                "ksp_type": used_sp_dict["ksp_type"] if "used_sp_dict" in locals() else sp_dict["ksp_type"],
                "pc_type": used_sp_dict["pc_type"] if "used_sp_dict" in locals() else sp_dict["pc_type"],
                "rtol": used_sp_dict["ksp_rtol"] if "used_sp_dict" in locals() else sp_dict["ksp_rtol"],
                "epsilon": epsilon,
                "beta": beta,
                "stabilization": stabilization,
            },
            num_dofs=V.dof_count,
        )
