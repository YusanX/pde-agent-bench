"""Firedrake Stokes oracle (Taylor-Hood mixed elements, P2-P1)."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import sympy as sp
import ufl

from firedrake import (
    Function, TrialFunctions, TestFunctions, SpatialCoordinate,
    DirichletBC, Constant, interpolate, inner, grad, div, dx, solve,
    as_vector, split,
    MixedVectorSpaceBasis, VectorSpaceBasis,
)

from .common import (
    OracleResult, compute_rel_L2_grid,
    create_mesh, create_mixed_space,
    parse_expression, parse_vector_expression,
    sample_vector_magnitude_on_grid,
    _eval_exact_vec_mag_on_grid,
    _apply_domain_mask,
)


# Firedrake UnitSquareMesh boundary markers:
#   1 = bottom (y=0),  2 = right (x=1),  3 = top (y=1),  4 = left (x=0)
_NAMED_BOUNDARY_MAP = {
    "y0": 1, "ymin": 1,
    "x1": 2, "xmax": 2,
    "y1": 3, "ymax": 3,
    "x0": 4, "xmin": 4,
    "z0": 5, "zmin": 5,
    "z1": 6, "zmax": 6,
}


def _to_firedrake_boundary(on: str):
    """Convert boundary label string to Firedrake boundary ID."""
    key = str(on).lower()
    if key in {"all", "*"}:
        return "on_boundary"
    if key in _NAMED_BOUNDARY_MAP:
        return _NAMED_BOUNDARY_MAP[key]
    try:
        return int(on)
    except (ValueError, TypeError):
        raise ValueError(f"Unknown boundary label '{on}' (expected x0/x1/y0/y1/z0/z1 or integer)")


def _build_velocity_bcs(W, u_exact_fn, bc_cfg, x, dim):
    """Build velocity DirichletBCs for the mixed space W = V × Q.

    BC values must be either a Firedrake Constant or a Function (not a plain
    UFL ComponentTensor like as_vector([Constant, Constant])).  A domain-free
    UFL expression passed to DirichletBC triggers Firedrake's interpolation
    path which requires a mesh domain and raises "missing integration domain".
    """
    dirichlet = bc_cfg if bc_cfg is not None else {"on": "all", "value": "0.0"}
    if isinstance(dirichlet, dict):
        dirichlet = [dirichlet]

    # Collapsed velocity space for creating domain-bound Function BC values.
    V_vel = W.sub(0).collapse()

    bcs = []
    for cfg in dirichlet:
        on = cfg.get("on", "all")
        boundary = _to_firedrake_boundary(on)
        value = cfg.get("value", "0.0")

        if isinstance(value, str) and value in {"u", "u_exact"}:
            bc_val = u_exact_fn
        elif isinstance(value, (list, tuple)):
            # Try to parse every component as a float constant first.
            float_vals = []
            all_const = True
            for v_ in value:
                try:
                    float_vals.append(float(sp.sympify(str(v_))))
                except Exception:
                    all_const = False
                    break

            if all_const:
                # Use a Firedrake vector Constant – handled without domain lookup.
                bc_val = Constant(float_vals)
            else:
                # At least one component is a symbolic expression; interpolate
                # into a Function (domain-bound) so DirichletBC never needs to
                # evaluate an integral.
                bc_fn = Function(V_vel)
                expr_parts = []
                for v_ in value:
                    try:
                        c = float(sp.sympify(str(v_)))
                        expr_parts.append(parse_expression(str(c), x))
                    except Exception:
                        expr_parts.append(parse_expression(str(v_), x))
                bc_fn.interpolate(as_vector(expr_parts))
                bc_val = bc_fn
        else:
            try:
                c = float(sp.sympify(value))
                bc_val = Constant([c] * dim)
            except Exception:
                bc_fn = Function(V_vel)
                bc_fn.interpolate(as_vector([parse_expression(value, x)] * dim))
                bc_val = bc_fn

        bcs.append(DirichletBC(W.sub(0), bc_val, boundary))
    return bcs


def _is_linear_solve_failure(exc: Exception) -> bool:
    """Return True when Firedrake/PETSc failed in the linear solve stage."""
    msg = str(exc)
    return (
        "DIVERGED_LINEAR_SOLVE" in msg
        or "Linear solve failed to converge" in msg
        or "Nonlinear solve failed to converge after 0 nonlinear iterations" in msg
    )


def _build_stokes_solver_parameters(
    solver_params: Dict[str, Any],
    *,
    default_ksp: str = "minres",
    default_pc: str = "hypre",
    default_rtol: float = 1e-10,
    default_max_it: int | None = None,
) -> Dict[str, Any]:
    """Build Firedrake PETSc options for linear Stokes solves."""
    ksp_type = solver_params.get("ksp_type", default_ksp)
    pc_type = solver_params.get("pc_type", default_pc)

    sp_dict = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": solver_params.get("rtol", default_rtol),
    }
    if default_max_it is not None or "max_it" in solver_params:
        sp_dict["ksp_max_it"] = solver_params.get("max_it", default_max_it)

    if "pc_factor_mat_solver_type" in solver_params:
        sp_dict["pc_factor_mat_solver_type"] = solver_params["pc_factor_mat_solver_type"]

    if pc_type == "fieldsplit":
        sp_dict.update({
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "lu",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "lu",
        })
    return sp_dict


def _fallback_stokes_solver_parameters(sp_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate more robust solver fallbacks for difficult reference solves."""
    fallbacks: List[Dict[str, Any]] = []
    max_it = int(sp_dict.get("ksp_max_it") or 0)
    relaxed_max_it = max(max_it, 2000)

    if sp_dict.get("ksp_type") == "minres" and sp_dict.get("pc_type") == "hypre":
        gmres_hypre = dict(sp_dict)
        gmres_hypre["ksp_type"] = "gmres"
        gmres_hypre["ksp_max_it"] = relaxed_max_it
        fallbacks.append(gmres_hypre)

    if sp_dict.get("pc_type") != "fieldsplit":
        fallbacks.append({
            "ksp_type": "gmres",
            "pc_type": "fieldsplit",
            "ksp_rtol": sp_dict["ksp_rtol"],
            "ksp_max_it": relaxed_max_it,
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "lu",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "lu",
        })

    return fallbacks


def _solve_linear_stokes(
    a,
    L,
    W,
    bcs,
    solver_parameters: Dict[str, Any],
    *,
    allow_fallback: bool = False,
):
    """Solve a linear Stokes system, retrying with safer PETSc settings if needed."""
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    attempts = [solver_parameters]
    if allow_fallback:
        attempts.extend(_fallback_stokes_solver_parameters(solver_parameters))

    last_exc = None
    for params in attempts:
        w_h = Function(W)
        try:
            solve(a == L, w_h, bcs=bcs, nullspace=nullspace, solver_parameters=params)
            return w_h, params
        except Exception as exc:
            last_exc = exc
            if not allow_fallback or not _is_linear_solve_failure(exc):
                raise

    raise last_exc


class FiredrakeStokesOracle:
    """Stokes oracle using Firedrake Taylor-Hood (CG2/CG1)."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        degree_u = case_spec["fem"].get("degree_u", 2)
        degree_p = case_spec["fem"].get("degree_p", 1)
        W = create_mixed_space(msh, degree_u, degree_p)
        x = SpatialCoordinate(msh)
        dim = msh.geometric_dimension()

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        nu = float(params.get("nu", 1.0))

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact_fn = None
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
                lap = sum(sp.diff(ui, c, 2) for c in coords)
                grad_p = sp.diff(p_sym, coords[i])
                f_sym.append(-nu * lap + grad_p)
            f_ufl = parse_vector_expression(f_sym, x)
            V_sub = W.sub(0).collapse()  # Firedrake: returns space only (not a tuple)
            u_exact_fn = Function(V_sub)
            u_exact_fn.interpolate(parse_vector_expression(u_sym_vec, x))
        elif source_expr is not None:
            V_vel = W.sub(0).collapse()
            if isinstance(source_expr, (list, tuple)):
                # Interpolate into a mesh-bound Function so constant vectors like
                # ["0.0", "0.0"] do not collapse to a domain-free UFL zero.
                f_ufl = Function(V_vel)
                f_ufl.interpolate(parse_vector_expression(source_expr, x))
            else:
                try:
                    c = float(sp.sympify(source_expr))
                    f_ufl = Function(V_vel)
                    f_ufl.assign(Constant((c,) * dim))
                except Exception:
                    f_ufl = Function(V_vel)
                    f_ufl.interpolate(as_vector([parse_expression(source_expr, x)] * dim))

        if f_ufl is None:
            # Use a zero Function (domain-bound) to avoid "missing integration domain" error.
            # as_vector([Constant(0.0)] * dim) has no mesh domain and causes UFL errors.
            V_vel = W.sub(0).collapse()
            f_ufl = Function(V_vel)  # zero-initialized vector function

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
        a = (nu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * dx
        L = inner(f_ufl, v) * dx

        bc_cfg = case_spec.get("bc", {}).get("dirichlet")
        bcs = _build_velocity_bcs(W, u_exact_fn, bc_cfg, x, dim)

        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _build_stokes_solver_parameters(solver_params)
        w_h, used_sp_dict = _solve_linear_stokes(a, L, W, bcs, sp_dict)

        u_h, p_h = w_h.subfunctions

        grid_cfg = case_spec["output"]["grid"]
        sample_args = (grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg.get("nz"))
        *_, u_grid = sample_vector_magnitude_on_grid(u_h, *sample_args)

        baseline_error = 0.0
        if u_exact_fn is not None:
            u_exact_grid = _apply_domain_mask(u_grid, _eval_exact_vec_mag_on_grid(u_sym_vec, tuple(coords), grid_cfg))
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

            if source_expr is not None:
                ref_V_vel = ref_W.sub(0).collapse()
                if isinstance(source_expr, (list, tuple)):
                    ref_f_ufl = Function(ref_V_vel)
                    ref_f_ufl.interpolate(parse_vector_expression(source_expr, ref_x))
                else:
                    try:
                        c = float(sp.sympify(source_expr))
                        ref_f_ufl = Function(ref_V_vel)
                        ref_f_ufl.assign(Constant((c,) * dim))
                    except Exception:
                        ref_f_ufl = Function(ref_V_vel)
                        ref_f_ufl.interpolate(as_vector([parse_expression(source_expr, ref_x)] * dim))
            else:
                # Zero-initialized Function is domain-bound; avoids "missing integration domain".
                ref_V_vel = ref_W.sub(0).collapse()
                ref_f_ufl = Function(ref_V_vel)

            (ref_u, ref_p) = TrialFunctions(ref_W)
            (ref_v, ref_q) = TestFunctions(ref_W)
            ref_a = (nu * inner(grad(ref_u), grad(ref_v)) - div(ref_v) * ref_p - ref_q * div(ref_u)) * dx
            ref_L = inner(ref_f_ufl, ref_v) * dx

            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet")
            ref_bcs = _build_velocity_bcs(ref_W, None, ref_bc_cfg, ref_x, dim)

            # Start from the configured solver, but retry with more robust
            # Krylov/preconditioner settings when large open-boundary reference
            # solves diverge at the linear stage.
            ref_sp = _build_stokes_solver_parameters(
                ref_solver_cfg,
                default_ksp="minres",
                default_pc="hypre",
                default_rtol=1e-10,
                default_max_it=500,
            )
            ref_w_h, _ = _solve_linear_stokes(
                ref_a, ref_L, ref_W, ref_bcs, ref_sp, allow_fallback=True
            )
            ref_u_h, _ = ref_w_h.subfunctions

            *_, ref_grid = sample_vector_magnitude_on_grid(ref_u_h, *sample_args)
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        baseline_time = time.perf_counter() - t_start
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info={
                "ksp_type": used_sp_dict["ksp_type"],
                "pc_type": used_sp_dict["pc_type"],
                "rtol": used_sp_dict["ksp_rtol"],
                "degree_u": degree_u,
                "degree_p": degree_p,
            },
            num_dofs=W.dof_count,
        )
