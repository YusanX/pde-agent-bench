"""Firedrake Navier-Stokes oracle (steady incompressible, Newton iteration)."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import sympy as sp
import ufl

from firedrake import (
    Function, TestFunctions, TrialFunctions, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, div, dot, dx,
    as_vector, split, nabla_grad,
    MixedVectorSpaceBasis, VectorSpaceBasis,
    NonlinearVariationalProblem, NonlinearVariationalSolver, derivative,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_mixed_space,
    parse_expression, parse_vector_expression,
    sample_vector_magnitude_on_grid,
    _apply_domain_mask,
)
from .stokes import (
    _build_velocity_bcs,
    _build_stokes_solver_parameters,
    _solve_linear_stokes,
)


def _is_linear_solve_failure(exc: Exception) -> bool:
    """Return True when Firedrake/PETSc failed in the linear solve stage."""
    msg = str(exc)
    return (
        "DIVERGED_LINEAR_SOLVE" in msg
        or "Linear solve failed to converge" in msg
        or "Nonlinear solve failed to converge after 0 nonlinear iterations" in msg
    )


def _is_nonlinear_convergence_failure(exc: Exception) -> bool:
    """Return True when SNES failed after entering nonlinear iterations."""
    msg = str(exc)
    return (
        "Nonlinear solve failed to converge" in msg
        or "DIVERGED_LINE_SEARCH" in msg
        or "DIVERGED_MAX_IT" in msg
    )


def _build_vector_rhs(W, source_expr, x, dim):
    """Build a mesh-bound velocity-space forcing for Navier-Stokes."""
    V_vel = W.sub(0).collapse()
    f_ufl = Function(V_vel)
    if source_expr is None:
        return f_ufl

    if isinstance(source_expr, (list, tuple)):
        f_ufl.interpolate(parse_vector_expression(source_expr, x))
        return f_ufl

    try:
        c = float(sp.sympify(source_expr))
        f_ufl.assign(Constant((c,) * dim))
    except Exception:
        f_ufl.interpolate(as_vector([parse_expression(source_expr, x)] * dim))
    return f_ufl


def _build_ns_solver_parameters(
    solver_params: Dict[str, Any],
    *,
    default_pc: str = "hypre",
) -> Dict[str, Any]:
    """Build PETSc options for Firedrake Navier-Stokes Newton solves."""
    return {
        "snes_type": "newtonls",
        "snes_linesearch_type": solver_params.get("linesearch", "bt"),
        "snes_rtol": solver_params.get("rtol", 1e-10),
        "snes_atol": solver_params.get("atol", 1e-12),
        "snes_max_it": solver_params.get("max_it", 50),
        "ksp_type": solver_params.get("ksp_type", "gmres"),
        "pc_type": solver_params.get("pc_type", default_pc),
        "ksp_rtol": solver_params.get("ksp_rtol", solver_params.get("rtol", 1e-10)),
        "ksp_atol": solver_params.get("ksp_atol", solver_params.get("atol", 1e-12)),
    }


def _fallback_ns_solver_parameters(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate more robust fallback linear solver choices for Newton steps."""
    fallbacks: List[Dict[str, Any]] = []

    if params.get("pc_type") == "lu":
        hypre_params = dict(params)
        hypre_params["pc_type"] = "hypre"
        fallbacks.append(hypre_params)

    if params.get("pc_type") != "fieldsplit":
        fieldsplit_params = dict(params)
        fieldsplit_params.update({
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "lu",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "lu",
        })
        fallbacks.append(fieldsplit_params)

    return fallbacks


def _fallback_ns_nonlinear_parameters(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate nonlinear fallback strategies once the linear solve is stable."""
    fallbacks: List[Dict[str, Any]] = []

    if params.get("snes_type") == "newtonls":
        for ls in ("basic", "l2"):
            if params.get("snes_linesearch_type") == ls:
                continue
            alt = dict(params)
            alt["snes_linesearch_type"] = ls
            alt["snes_max_it"] = max(int(params.get("snes_max_it", 50)), 160)
            fallbacks.append(alt)

    return fallbacks


def _initialize_guess(W, f_ufl, nu, bcs, solver_params, u_exact_fn=None, p_exact_fn=None):
    """Create a robust initial guess for Newton."""
    init_mode = solver_params.get("init", "stokes")
    w0 = Function(W)

    if init_mode == "zero":
        return w0

    if init_mode == "exact":
        if u_exact_fn is None:
            raise ValueError("init='exact' requires manufactured_solution.u")
        w0.subfunctions[0].assign(u_exact_fn)
        if p_exact_fn is not None:
            w0.subfunctions[1].assign(p_exact_fn)
        return w0

    if init_mode == "continuation" and u_exact_fn is not None:
        # Manufactured cases have the exact velocity available; use it as a
        # robust warm start instead of emulating the full DOLFINx continuation path.
        w0.subfunctions[0].assign(u_exact_fn)
        if p_exact_fn is not None:
            w0.subfunctions[1].assign(p_exact_fn)
        return w0

    if init_mode in {"stokes", "continuation"}:
        (u_s, p_s) = TrialFunctions(W)
        (v_s, q_s) = TestFunctions(W)
        a_stokes = (
            nu * inner(grad(u_s), grad(v_s)) - div(v_s) * p_s - q_s * div(u_s)
        ) * dx
        L_stokes = inner(f_ufl, v_s) * dx
        stokes_sp = _build_stokes_solver_parameters(
            {
                "ksp_type": solver_params.get("stokes_ksp_type", "minres"),
                "pc_type": solver_params.get("stokes_pc_type", "hypre"),
                "rtol": solver_params.get("stokes_ksp_rtol", 1e-10),
                "max_it": solver_params.get("stokes_max_it", 500),
            },
            default_max_it=500,
        )
        stokes_w, _ = _solve_linear_stokes(
            a_stokes, L_stokes, W, bcs, stokes_sp, allow_fallback=True
        )
        w0.assign(stokes_w)
        return w0

    raise ValueError(f"Unsupported init mode: {init_mode}")


class FiredrakeNavierStokesOracle:
    """Steady incompressible Navier-Stokes oracle using Firedrake."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        degree_u = case_spec["fem"].get("degree_u", 2)
        degree_p = case_spec["fem"].get("degree_p", 1)
        W = create_mixed_space(msh, degree_u, degree_p)
        x = SpatialCoordinate(msh)
        dim = msh.geometric_dimension

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        nu = float(params.get("nu", 1.0))

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact_fn = None
        p_exact_fn = None
        f_ufl = None

        if "u" in manufactured and "p" in manufactured:
            sx, sy, sz = sp.symbols("x y z", real=True)
            local_d = {"x": sx, "y": sy, "z": sz}
            u_sym = manufactured["u"]
            p_sym = sp.sympify(manufactured["p"], locals=local_d)
            u_sym_vec = [sp.sympify(u_sym[i], locals=local_d) for i in range(dim)]
            coords = [sx, sy, sz][:dim]
            f_sym = []
            for i, ui in enumerate(u_sym_vec):
                conv = sum(u_sym_vec[j] * sp.diff(ui, coords[j]) for j in range(dim))
                lap = sum(sp.diff(ui, c, 2) for c in coords)
                grad_p = sp.diff(p_sym, coords[i])
                f_sym.append(conv - nu * lap + grad_p)
            f_ufl = parse_vector_expression(f_sym, x)
            V_sub = W.sub(0).collapse()  # Firedrake: returns space only (not a tuple)
            Q_sub = W.sub(1).collapse()
            u_exact_fn = Function(V_sub)
            p_exact_fn = Function(Q_sub)
            u_exact_fn.interpolate(parse_vector_expression(u_sym_vec, x))
            p_exact_fn.interpolate(parse_expression(p_sym, x))
        elif source_expr is not None:
            f_ufl = _build_vector_rhs(W, source_expr, x, dim)

        if f_ufl is None:
            f_ufl = _build_vector_rhs(W, None, x, dim)

        bc_cfg = case_spec.get("bc", {}).get("dirichlet")
        bcs = _build_velocity_bcs(W, u_exact_fn, bc_cfg, x, dim)

        solver_params = case_spec.get("oracle_solver", {})

        # Solve via Newton
        w_h = self._solve_newton(
            W, f_ufl, nu, bcs, solver_params, dim, u_exact_fn, p_exact_fn
        )
        u_h, p_h = w_h.subfunctions

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_vector_magnitude_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        if u_exact_fn is not None:
            # Evaluate exact solution analytically on the output grid
            bbox = grid_cfg["bbox"]
            nx_g, ny_g = grid_cfg["nx"], grid_cfg["ny"]
            x_lin = np.linspace(bbox[0], bbox[1], nx_g)
            y_lin = np.linspace(bbox[2], bbox[3], ny_g)
            xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
            u_sym_strs = manufactured.get("u", [])
            sx, sy = sp.symbols("x y", real=True)
            try:
                uvec = [sp.sympify(s, locals={"x": sx, "y": sy}) for s in u_sym_strs]
                u1_fn = sp.lambdify((sx, sy), uvec[0], "numpy")
                u2_fn = sp.lambdify((sx, sy), uvec[1], "numpy")
                u_mag_exact = np.sqrt(u1_fn(xx, yy)**2 + u2_fn(xx, yy)**2)
                u_exact_grid = u_mag_exact
            except Exception:
                _, _, u_exact_grid = sample_vector_magnitude_on_grid(
                    u_exact_fn, grid_cfg["bbox"], nx_g, ny_g
                )
            u_exact_grid = _apply_domain_mask(u_grid, u_exact_grid)
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver_cfg = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_deg_u = ref_fem_spec.get("degree_u", degree_u)
            ref_deg_p = ref_fem_spec.get("degree_p", degree_p)
            ref_W = create_mixed_space(ref_msh, ref_deg_u, ref_deg_p)
            ref_x = SpatialCoordinate(ref_msh)

            ref_f_ufl = _build_vector_rhs(ref_W, source_expr, ref_x, dim)

            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet")
            ref_bcs = _build_velocity_bcs(ref_W, None, ref_bc_cfg, ref_x, dim)

            ref_solver_params = dict(ref_solver_cfg)
            ref_w_h = self._solve_newton(
                ref_W, ref_f_ufl, nu, ref_bcs, ref_solver_params, dim
            )
            ref_u_h, _ = ref_w_h.subfunctions

            _, _, ref_grid = sample_vector_magnitude_on_grid(ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        baseline_time = time.perf_counter() - t_start
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info={
                "ksp_type": solver_params.get("ksp_type", "gmres"),
                "pc_type": solver_params.get("pc_type", "hypre"),
                "rtol": solver_params.get("rtol", 1e-10),
                "nu": nu,
                "degree_u": degree_u,
                "degree_p": degree_p,
            },
            num_dofs=W.dof_count,
        )

    def _solve_newton(self, W, f_ufl, nu, bcs, solver_params, dim, u_exact_fn=None, p_exact_fn=None):
        w_init = _initialize_guess(
            W, f_ufl, nu, bcs, solver_params, u_exact_fn=u_exact_fn, p_exact_fn=p_exact_fn
        )
        base_params = _build_ns_solver_parameters(solver_params)
        attempts = [base_params]
        attempts.extend(_fallback_ns_solver_parameters(base_params))
        attempts.extend(_fallback_ns_nonlinear_parameters(base_params))
        nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

        last_exc = None
        for nl_params in attempts:
            w = Function(W)
            w.assign(w_init)
            u, p = split(w)
            v, q = TestFunctions(W)

            F = (
                nu * inner(grad(u), grad(v)) * dx
                + inner(dot(grad(u), u), v) * dx
                - p * div(v) * dx
                - q * div(u) * dx
                - inner(f_ufl, v) * dx
            )
            problem = NonlinearVariationalProblem(F, w, bcs=bcs)
            solver = NonlinearVariationalSolver(
                problem, solver_parameters=nl_params, nullspace=nullspace
            )
            try:
                solver.solve()
                return w
            except Exception as exc:
                last_exc = exc
                if not (_is_linear_solve_failure(exc) or _is_nonlinear_convergence_failure(exc)):
                    raise

        raise last_exc
