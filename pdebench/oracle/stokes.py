"""Stokes oracle solver (steady incompressible flow)."""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

from .common import (
    OracleResult,
    compute_rel_L2_grid,
    create_mesh,
    create_mixed_space,
    interpolate_expression,
    parse_expression,
    parse_vector_expression,
    sample_vector_magnitude_on_grid,
)


def _normalize_dirichlet_cfg(bc_cfg: Any) -> list[Dict[str, Any]]:
    """Normalize bc.dirichlet to always be a list of dicts."""
    if bc_cfg is None:
        return []
    if isinstance(bc_cfg, dict):
        return [bc_cfg]
    if isinstance(bc_cfg, list):
        return bc_cfg
    raise ValueError("bc.dirichlet must be a dict or a list of dicts")


def _boundary_selector(on: str, dim: int):
    """Create a lambda for boundary selection based on string identifier."""
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


def _ensure_domain_scalar(expr, x):
    """Ensure expression is bound to domain (has integration measure)."""
    if isinstance(expr, (int, float)):
        return ufl.as_ufl(expr) + 0.0 * x[0]
    # Check if expr is a pure UFL constant without domain
    if hasattr(expr, 'ufl_domain') and expr.ufl_domain() is None:
        return expr + 0.0 * x[0]
    return expr


def _build_dirichlet_bcs(
    msh, W, bc_cfg: Any, u_exact: fem.Function | None, dim: int
) -> list[fem.DirichletBC]:
    """Build Dirichlet BCs from configuration."""
    bcs = []
    dirichlet_cfgs = _normalize_dirichlet_cfg(bc_cfg)
    V, _ = W.sub(0).collapse()
    for cfg in dirichlet_cfgs:
        on = cfg.get("on", "all")
        selector = _boundary_selector(on, dim)
        boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V), selector)
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
            
            # Check if all components are constants
            try:
                const_values = [float(expr) for expr in expr_list]
                is_constant = True
            except (ValueError, TypeError):
                is_constant = False
            
            bc_func = fem.Function(V)
            if is_constant:
                # Use lambda for pure constants (more robust)
                bc_func.interpolate(lambda x: np.array([[v] * x.shape[1] for v in const_values]))
            else:
                # Use UFL for expressions
                x = ufl.SpatialCoordinate(msh)
                bc_components = [
                    _ensure_domain_scalar(parse_expression(expr, x), x)
                    for expr in expr_list
                ]
                bc_expr = ufl.as_vector(bc_components)
                interpolate_expression(bc_func, bc_expr)
        bcs.append(fem.dirichletbc(bc_func, boundary_dofs, W.sub(0)))
    return bcs


class StokesSolver:
    """Taylor-Hood mixed solver for Stokes."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        # ⏱️ 开始计时整个求解流程
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        dim = msh.geometry.dim
        degree_u = case_spec["fem"].get("degree_u", 2)
        degree_p = case_spec["fem"].get("degree_p", 1)
        W = create_mixed_space(msh, degree_u, degree_p)

        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        nu = float(params.get("nu", 1.0))

        x = ufl.SpatialCoordinate(msh)
        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        u_exact = None
        p_exact = None
        f_expr = None

        if "u" in manufactured and "p" in manufactured:
            sx, sy, sz = sp.symbols("x y z", real=True)
            local_dict = {"x": sx, "y": sy, "z": sz}
            u_sym = manufactured["u"]
            if len(u_sym) != dim:
                raise ValueError("manufactured_solution.u dimension mismatch")
            p_sym = sp.sympify(manufactured["p"], locals=local_dict)
            u_sym_vec = [sp.sympify(u_sym[i], locals=local_dict) for i in range(dim)]

            coords = [sx, sy, sz][:dim]
            f_sym = []
            for i, ui in enumerate(u_sym_vec):
                lap = sum(sp.diff(ui, c, 2) for c in coords)
                grad_p = sp.diff(p_sym, coords[i])
                f_sym.append(-nu * lap + grad_p)
            f_expr = parse_vector_expression(f_sym, x)

            u_exact_expr = parse_vector_expression(u_sym_vec, x)
            p_exact_expr = parse_expression(p_sym, x)

            V, _ = W.sub(0).collapse()
            Q, _ = W.sub(1).collapse()
            u_exact = fem.Function(V)
            p_exact = fem.Function(Q)
            interpolate_expression(u_exact, u_exact_expr)
            interpolate_expression(p_exact, p_exact_expr)
        elif source_expr is not None:
            if isinstance(source_expr, (list, tuple)):
                if len(source_expr) != dim:
                    raise ValueError("source_term dimension mismatch")
                expr_list = list(source_expr)
            else:
                expr_list = [source_expr] * dim
            
            # Check if all components are pure constants
            try:
                const_values = [float(sp.sympify(expr)) for expr in expr_list]
                # All are constants, use fem.Constant (DOLFINx canonical way)
                f_expr = fem.Constant(msh, tuple(const_values))
            except (ValueError, TypeError, AttributeError, Exception):
                # Has symbolic expressions, use UFL parsing
                f_components = [
                    _ensure_domain_scalar(parse_expression(expr, x), x)
                    for expr in expr_list
                ]
                f_expr = ufl.as_vector(f_components)

        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        a = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.div(v) * p * ufl.dx
            - q * ufl.div(u) * ufl.dx
        )
        # Use fem.Constant for zero vector (DOLFINx canonical way)
        if f_expr is None:
            f_expr = fem.Constant(msh, tuple([0.0] * dim))
        L = ufl.inner(f_expr, v) * ufl.dx

        solver_params = case_spec.get("oracle_solver", {})

        bcs = []
        bc_cfg = case_spec.get("bc", {})
        dirichlet_cfg = bc_cfg.get("dirichlet")
        if dirichlet_cfg is None:
            if u_exact is not None:
                bcs = _build_dirichlet_bcs(
                    msh, W, {"on": "all", "value": "u"}, u_exact, dim
                )
            else:
                bcs = _build_dirichlet_bcs(
                    msh, W, {"on": "all", "value": "0.0"}, None, dim
                )
        else:
            bcs = _build_dirichlet_bcs(msh, W, dirichlet_cfg, u_exact, dim)

        if not bcs:
            raise ValueError("Stokes requires at least one Dirichlet boundary condition")

        pressure_fixing = solver_params.get("pressure_fixing", "point")
        if pressure_fixing != "none":
            Q, _ = W.sub(1).collapse()
            if dim == 2:
                p_dofs = fem.locate_dofs_geometrical(
                    (W.sub(1), Q),
                    lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
                )
            else:
                p_dofs = fem.locate_dofs_geometrical(
                    (W.sub(1), Q),
                    lambda x: np.isclose(x[0], 0.0)
                    & np.isclose(x[1], 0.0)
                    & np.isclose(x[2], 0.0),
                )
            if len(p_dofs) > 0:
                p0 = fem.Function(Q)
                p0.x.array[:] = 0.0
                bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

        petsc_options = {
            "ksp_type": solver_params.get("ksp_type", "minres"),
            "pc_type": solver_params.get("pc_type", "hypre"),
            "ksp_rtol": solver_params.get("rtol", 1e-10),
        }

        problem = LinearProblem(
            a, L, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix="oracle_stokes_"
        )
        w_h = problem.solve()

        u_h = w_h.sub(0).collapse()
        p_h = w_h.sub(1).collapse()

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_vector_magnitude_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        if u_exact is not None:
            _, _, u_exact_grid = sample_vector_magnitude_on_grid(
                u_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
            # Use exact grid as reference for evaluation alignment.
            u_grid = u_exact_grid
        else:
            ref_cfg = case_spec.get("reference_config", {})
            ref_mesh_spec = ref_cfg.get("mesh", case_spec["mesh"])
            ref_fem_spec = ref_cfg.get("fem", case_spec["fem"])
            ref_solver = ref_cfg.get("oracle_solver", {})

            ref_msh = create_mesh(case_spec["domain"], ref_mesh_spec)
            ref_dim = ref_msh.geometry.dim
            ref_W = create_mixed_space(
                ref_msh,
                ref_fem_spec.get("degree_u", degree_u),
                ref_fem_spec.get("degree_p", degree_p),
            )
            ref_x = ufl.SpatialCoordinate(ref_msh)
            if source_expr is not None:
                if isinstance(source_expr, (list, tuple)):
                    if len(source_expr) != ref_dim:
                        raise ValueError("reference source_term dimension mismatch")
                    ref_expr_list = list(source_expr)
                else:
                    ref_expr_list = [source_expr] * ref_dim
                
                # Check if all components are pure constants
                try:
                    ref_const_values = [float(sp.sympify(expr)) for expr in ref_expr_list]
                    ref_f_expr = fem.Constant(ref_msh, tuple(ref_const_values))
                except (ValueError, TypeError, AttributeError, Exception):
                    ref_f_expr = parse_vector_expression(ref_expr_list, ref_x)
            else:
                ref_f_expr = fem.Constant(ref_msh, tuple([0.0] * ref_dim))

            ref_u, ref_p = ufl.TrialFunctions(ref_W)
            ref_v, ref_q = ufl.TestFunctions(ref_W)
            ref_nu = float(pde_cfg.get("pde_params", {}).get("nu", 1.0))
            ref_a = (
                ref_nu * ufl.inner(ufl.grad(ref_u), ufl.grad(ref_v)) * ufl.dx
                - ufl.div(ref_v) * ref_p * ufl.dx
                - ref_q * ufl.div(ref_u) * ufl.dx
            )
            ref_L = ufl.inner(ref_f_expr, ref_v) * ufl.dx
            ref_bcs = _build_dirichlet_bcs(
                ref_msh, ref_W, dirichlet_cfg or {"on": "all", "value": "0.0"}, None, ref_dim
            )
            if not ref_bcs:
                raise ValueError("Reference Stokes requires Dirichlet boundary conditions")

            ref_pressure_fixing = ref_solver.get("pressure_fixing", pressure_fixing)
            if ref_pressure_fixing != "none":
                ref_Q, _ = ref_W.sub(1).collapse()
                if ref_dim == 2:
                    ref_p_dofs = fem.locate_dofs_geometrical(
                        (ref_W.sub(1), ref_Q),
                        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
                    )
                else:
                    ref_p_dofs = fem.locate_dofs_geometrical(
                        (ref_W.sub(1), ref_Q),
                        lambda x: np.isclose(x[0], 0.0)
                        & np.isclose(x[1], 0.0)
                        & np.isclose(x[2], 0.0),
                    )
                if len(ref_p_dofs) > 0:
                    ref_p0 = fem.Function(ref_Q)
                    ref_p0.x.array[:] = 0.0
                    ref_bcs.append(fem.dirichletbc(ref_p0, ref_p_dofs, ref_W.sub(1)))

            ref_petsc_options = {
                "ksp_type": ref_solver.get("ksp_type", petsc_options["ksp_type"]),
                "pc_type": ref_solver.get("pc_type", petsc_options["pc_type"]),
                "ksp_rtol": ref_solver.get("rtol", 1e-12),
            }
            ref_problem = LinearProblem(
                ref_a,
                ref_L,
                bcs=ref_bcs,
                petsc_options=ref_petsc_options,
                petsc_options_prefix="oracle_stokes_ref_",
            )
            ref_w = ref_problem.solve()
            ref_u_h = ref_w.sub(0).collapse()
            _, _, ref_grid = sample_vector_magnitude_on_grid(
                ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        solver_info = {
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["ksp_rtol"],
            "pressure_fixing": pressure_fixing,
        }
        if u_exact is None:
            solver_info["reference_resolution"] = ref_mesh_spec.get("resolution")
            solver_info["reference_degree_u"] = ref_fem_spec.get("degree_u", degree_u)
            solver_info["reference_degree_p"] = ref_fem_spec.get("degree_p", degree_p)

        # ⏱️ 结束计时（包含完整流程）
        baseline_time = time.perf_counter() - t_start_total

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=W.dofmap.index_map.size_global,
        )
