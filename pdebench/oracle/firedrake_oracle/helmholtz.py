"""Firedrake Helmholtz oracle solver."""
from __future__ import annotations

import time
from typing import Any, Dict, List

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


def _is_linear_solve_failure(exc: Exception) -> bool:
    """Return True when Firedrake/PETSc failed in the linear solve stage."""
    msg = str(exc)
    return (
        "DIVERGED_LINEAR_SOLVE" in msg
        or "Linear solve failed to converge" in msg
        or "Nonlinear solve failed to converge after 0 nonlinear iterations" in msg
    )


def _build_helmholtz_solver_parameters(
    solver_params: Dict[str, Any],
    *,
    default_ksp: str = "gmres",
    default_pc: str = "ilu",
    default_rtol: float = 1e-10,
    default_atol: float = 1e-12,
) -> Dict[str, Any]:
    """Build PETSc options for Helmholtz solves."""
    return {
        "ksp_type": solver_params.get("ksp_type", default_ksp),
        "pc_type": solver_params.get("pc_type", default_pc),
        "ksp_rtol": solver_params.get("rtol", default_rtol),
        "ksp_atol": solver_params.get("atol", default_atol),
    }


def _fallback_helmholtz_solver_parameters(sp_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate more robust fallbacks for indefinite Helmholtz solves."""
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


def _solve_linear_helmholtz(a, L, V, bcs, solver_parameters: Dict[str, Any], *, allow_fallback: bool = False):
    """Solve a Helmholtz system, retrying with safer PETSc settings if needed."""
    attempts = [solver_parameters]
    if allow_fallback:
        attempts.extend(_fallback_helmholtz_solver_parameters(solver_parameters))

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


class FiredrakeHelmholtzSolver:
    """Helmholtz equation oracle: -Δu - k²u = f using Firedrake."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start = time.perf_counter()

        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        x = SpatialCoordinate(msh)
        dim = msh.geometric_dimension()

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        k = float(params.get("k", params.get("wave_number", 10.0)))

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact_fn = None
        f_ufl = None

        def _mms_symbols():
            sx, sy, sz = sp.symbols("x y z", real=True)
            locals_dict = {"x": sx, "y": sy, "pi": sp.pi}
            coords = [sx, sy]
            if dim >= 3:
                locals_dict["z"] = sz
                coords.append(sz)
            return locals_dict, tuple(coords)

        if "u" in manufactured:
            local_dict, coords = _mms_symbols()
            u_sym = sp.sympify(manufactured["u"], locals=local_dict)
            k_sym = sp.sympify(k)
            f_sym = -sum(sp.diff(u_sym, c, 2) for c in coords) - k_sym**2 * u_sym
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
        a = (inner(grad(u), grad(v)) - k**2 * u * v) * dx
        rhs = f_ufl if f_ufl is not None else Constant(0.0)
        L = rhs * v * dx

        if u_exact_fn is not None:
            bcs = [DirichletBC(V, u_exact_fn, "on_boundary")]
        else:
            bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            bcs = [build_scalar_bc(V, bc_cfg.get("value", "0.0"), x)]

        solver_params = case_spec.get("oracle_solver", {})
        sp_dict = _build_helmholtz_solver_parameters(solver_params)
        uh, used_sp_dict = _solve_linear_helmholtz(a, L, V, bcs, sp_dict, allow_fallback=True)

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
            if source_expr is not None:
                try:
                    ref_f = Constant(float(sp.sympify(source_expr)))
                except Exception:
                    ref_f = parse_expression(source_expr, ref_x)
            else:
                ref_f = Constant(0.0)
            ref_u = TrialFunction(ref_V)
            ref_v = TestFunction(ref_V)
            ref_a = (inner(grad(ref_u), grad(ref_v)) - k**2 * ref_u * ref_v) * dx
            ref_L = ref_f * ref_v * dx
            ref_bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
            ref_bcs = [build_scalar_bc(ref_V, ref_bc_cfg.get("value", "0.0"), ref_x)]
            ref_sp = _build_helmholtz_solver_parameters(
                ref_solver,
                default_ksp=sp_dict["ksp_type"],
                default_pc=sp_dict["pc_type"],
                default_rtol=1e-12,
                default_atol=sp_dict["ksp_atol"],
            )
            ref_uh, _ = _solve_linear_helmholtz(
                ref_a, ref_L, ref_V, ref_bcs, ref_sp, allow_fallback=True
            )
            *_, ref_grid = sample_scalar_on_grid(ref_uh, *sample_args)
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        baseline_time = time.perf_counter() - t_start
        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info={
                "k": k,
                "ksp_type": used_sp_dict["ksp_type"],
                "pc_type": used_sp_dict["pc_type"],
                "rtol": used_sp_dict["ksp_rtol"],
            },
            num_dofs=V.dof_count,
        )
