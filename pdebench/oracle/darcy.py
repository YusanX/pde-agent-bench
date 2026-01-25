"""Darcy flow oracle solver (elliptic and mixed formulations).

We support two mathematically equivalent steady Darcy models on Ω:

Elliptic (pressure) form:
  -∇·(κ ∇p) = f  in Ω,    p = g on ∂Ω

Mixed (flux-pressure) form:
  u + κ ∇p = 0    in Ω
  ∇·u      = f    in Ω

Notes (explicit assumptions):
- κ ("permeability") is treated as a scalar, strictly positive coefficient field.
- Manufactured-solution cases derive f from (κ, p) exactly using SymPy.
- For the mixed formulation, we impose flux Dirichlet data on u at the boundary
  using the manufactured u = -κ ∇p, and fix the pressure nullspace by a single
  point constraint (cell-based DG dof) when requested.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

from .common import (
    OracleResult,
    compute_rel_L2_grid,
    create_kappa_field,
    create_mesh,
    create_scalar_space,
    interpolate_expression,
    locate_all_boundary_dofs,
    parse_expression,
    parse_vector_expression,
    sample_scalar_on_grid,
    sample_vector_magnitude_on_grid,
)


def _create_darcy_mixed_space(
    msh: mesh.Mesh, degree_u: int, degree_p: int
) -> fem.FunctionSpace:
    """Create H(div)×L2 mixed space for Darcy: RT_k × DG_l."""
    from basix.ufl import element as basix_element
    from basix.ufl import mixed_element as basix_mixed_element

    cell = msh.topology.cell_name()
    # RT is inherently H(div)-conforming and vector-valued on the reference cell.
    # In Basix, RT already carries its value shape; do NOT wrap it with an extra
    # `shape=(dim,)`, which would conceptually create a tensor-valued field.
    u_el = basix_element("RT", cell, degree_u)
    p_el = basix_element("DG", cell, degree_p)
    W_el = basix_mixed_element([u_el, p_el])
    return fem.functionspace(msh, W_el)


def _ensure_positive_kappa(kappa_spec: Dict[str, Any]) -> None:
    """Conservative sanity-check for κ positivity on manufactured cases."""
    if kappa_spec.get("type", "constant") == "constant":
        if float(kappa_spec.get("value", 1.0)) <= 0.0:
            raise ValueError("Permeability κ must be strictly positive.")
    # For expression κ(x), we cannot guarantee positivity symbolically here.
    # We rely on benchmark design to keep κ(x) > 0 on Ω.


def _manufactured_from_pressure(
    msh: mesh.Mesh, kappa_spec: Dict[str, Any], p_expr_str: str
) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr, ufl.core.expr.Expr]:
    """Given κ and manufactured pressure p, derive f and flux u."""
    x = ufl.SpatialCoordinate(msh)
    dim = msh.geometry.dim
    if dim != 2:
        raise ValueError("Darcy oracle currently supports 2D manufactured cases only.")

    sx, sy = sp.symbols("x y", real=True)
    local = {"x": sx, "y": sy, "pi": sp.pi}
    p_sym = sp.sympify(p_expr_str, locals=local)

    if kappa_spec.get("type", "constant") == "expr":
        k_sym = sp.sympify(kappa_spec["expr"], locals=local)
    else:
        k_sym = sp.sympify(kappa_spec.get("value", 1.0))

    dp_dx = sp.diff(p_sym, sx)
    dp_dy = sp.diff(p_sym, sy)

    # u = -κ ∇p
    u_sym = [-k_sym * dp_dx, -k_sym * dp_dy]

    # f = ∇·u = -∇·(κ ∇p)
    f_sym = sp.diff(u_sym[0], sx) + sp.diff(u_sym[1], sy)

    u_expr = parse_vector_expression(u_sym, x)
    p_expr = parse_expression(p_sym, x)
    f_expr = parse_expression(f_sym, x)
    return u_expr, p_expr, f_expr


def _pressure_point_fix_bc(W: fem.FunctionSpace) -> fem.DirichletBC | None:
    """Fix pressure constant mode by pinning one DG dof.

    Important: for DG spaces, dofs are typically cell-based (not located on
    vertices), therefore a geometrical lookup at (0,0) may return an empty set.
    We therefore fall back to pinning dofs on a (local) cell entity.
    """
    Q, submap = W.sub(1).collapse()
    # First try geometrical lookup (works for nodal spaces, sometimes for DG)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) == 0:
        # Robust fallback: pin the first available pressure dof using the collapse map.
        # This avoids relying on geometric point queries, which are brittle for DG spaces.
        if submap is None or len(submap) == 0:
            return None
        p_dofs = np.array([int(submap[0])], dtype=np.int32)
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    return fem.dirichletbc(p0, p_dofs, W.sub(1))


class DarcySolver:
    """Oracle solver for Darcy flow."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        # ⏱️ 开始计时整个求解流程
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        if msh.geometry.dim != 2:
            raise ValueError("Darcy oracle currently supports 2D unit_square cases only.")

        pde_cfg = case_spec["pde"]
        coeffs = pde_cfg.get("coefficients", {})
        kappa_spec = coeffs.get("kappa", {"type": "constant", "value": 1.0})
        _ensure_positive_kappa(kappa_spec)
        kappa = create_kappa_field(msh, kappa_spec)

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        formulation = pde_cfg.get("formulation", "elliptic").lower()

        # Output selection
        output_field = case_spec.get("output", {}).get("field", "pressure")
        grid_cfg = case_spec["output"]["grid"]

        solver_params = case_spec.get("oracle_solver", {})

        # ---------------------------------------------------------------------
        # Elliptic pressure formulation: -div(κ grad p) = f, p = g on boundary.
        # ---------------------------------------------------------------------
        if formulation == "elliptic":
            fem_spec = case_spec["fem"]
            V = create_scalar_space(msh, fem_spec.get("family", "Lagrange"), fem_spec.get("degree", 1))

            p_exact = None
            f_expr = None
            if "p" in manufactured:
                u_expr, p_expr, f_expr = _manufactured_from_pressure(
                    msh, kappa_spec, manufactured["p"]
                )
                p_exact = fem.Function(V)
                interpolate_expression(p_exact, p_expr)
            elif source_expr is not None:
                x = ufl.SpatialCoordinate(msh)
                f_expr = parse_expression(source_expr, x)

            p = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            a = ufl.inner(kappa * ufl.grad(p), ufl.grad(v)) * ufl.dx
            L = (f_expr if f_expr is not None else 0.0) * v * ufl.dx

            # BC: manufactured pressure if provided, else configured scalar expression
            bcs = []
            if p_exact is not None:
                boundary_dofs = locate_all_boundary_dofs(msh, V)
                bcs = [fem.dirichletbc(p_exact, boundary_dofs)]
            else:
                bc_cfg = case_spec.get("bc", {}).get("dirichlet", {})
                bc_val = bc_cfg.get("value", "0.0")
                bc_fun = fem.Function(V)
                interpolate_expression(bc_fun, parse_expression(bc_val, ufl.SpatialCoordinate(msh)))
                boundary_dofs = locate_all_boundary_dofs(msh, V)
                bcs = [fem.dirichletbc(bc_fun, boundary_dofs)]

            petsc_options = {
                "ksp_type": solver_params.get("ksp_type", "cg"),
                "pc_type": solver_params.get("pc_type", "hypre"),
                "ksp_rtol": solver_params.get("rtol", 1e-10),
                "ksp_atol": solver_params.get("atol", 1e-12),
            }
            problem = LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options=petsc_options,
                petsc_options_prefix="oracle_darcy_elliptic_",
            )
            p_h = problem.solve()

            if output_field == "pressure":
                _, _, out_grid = sample_scalar_on_grid(
                    p_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
            elif output_field == "flux_magnitude":
                # Compute u = -κ ∇p in a DG0 vector space for evaluation robustness
                Vdg = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
                u_fun = fem.Function(Vdg)
                u_expr = -kappa * ufl.grad(p_h)
                interpolate_expression(u_fun, u_expr)
                _, _, out_grid = sample_vector_magnitude_on_grid(
                    u_fun, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
            else:
                raise ValueError(f"Unsupported Darcy output field: {output_field}")

            baseline_error = 0.0
            if p_exact is not None and output_field == "pressure":
                _, _, p_exact_grid = sample_scalar_on_grid(
                    p_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                baseline_error = compute_rel_L2_grid(out_grid, p_exact_grid)
                out_grid = p_exact_grid

            # ⏱️ 结束计时（椭圆型情况）
            baseline_time = time.perf_counter() - t_start_total

            solver_info = {
                "formulation": "elliptic",
                "ksp_type": petsc_options["ksp_type"],
                "pc_type": petsc_options["pc_type"],
                "rtol": petsc_options["ksp_rtol"],
                "kappa_type": kappa_spec.get("type", "constant"),
                "output_field": output_field,
                "element_family": fem_spec.get("family", "Lagrange"),
                "element_degree": fem_spec.get("degree", 1),
            }
            return OracleResult(
                baseline_error=float(baseline_error),
                baseline_time=float(baseline_time),
                reference=out_grid,
                solver_info=solver_info,
                num_dofs=V.dofmap.index_map.size_global,
            )

        # ---------------------------------------------------------------------
        # Mixed formulation: RT×DG with flux boundary from manufactured solution.
        # ---------------------------------------------------------------------
        if formulation == "mixed":
            degree_u = int(case_spec["fem"].get("degree_u", 1))
            degree_p = int(case_spec["fem"].get("degree_p", max(0, degree_u - 1)))
            W = _create_darcy_mixed_space(msh, degree_u, degree_p)
            V, _ = W.sub(0).collapse()
            Q, _ = W.sub(1).collapse()

            if "p" not in manufactured:
                raise ValueError("Darcy mixed formulation requires manufactured_solution.p")

            u_expr, p_expr, f_expr = _manufactured_from_pressure(
                msh, kappa_spec, manufactured["p"]
            )
            u_exact = fem.Function(V)
            p_exact = fem.Function(Q)
            interpolate_expression(u_exact, u_expr)
            interpolate_expression(p_exact, p_expr)

            # Build flux Dirichlet BC on u at the boundary using u_exact.
            boundary_facets = mesh.locate_entities_boundary(
                msh, msh.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool)
            )
            u_dofs = fem.locate_dofs_topological((W.sub(0), V), msh.topology.dim - 1, boundary_facets)
            bcs: list[fem.DirichletBC] = [fem.dirichletbc(u_exact, u_dofs, W.sub(0))]

            pressure_fixing = solver_params.get("pressure_fixing", "point")
            if pressure_fixing == "point":
                p_bc = _pressure_point_fix_bc(W)
                if p_bc is not None:
                    bcs.append(p_bc)
            elif pressure_fixing == "none":
                pass
            else:
                raise ValueError(f"Unsupported pressure_fixing: {pressure_fixing}")

            (u, p) = ufl.TrialFunctions(W)
            (v, q) = ufl.TestFunctions(W)
            a = (
                ufl.inner((1.0 / kappa) * u, v) * ufl.dx
                - p * ufl.div(v) * ufl.dx
                + q * ufl.div(u) * ufl.dx
            )
            L = f_expr * q * ufl.dx

            # Mixed RT×DG leads to a symmetric-indefinite saddle-point system.
            # AMG (e.g. hypre) is generally not a safe black-box PC for this structure.
            # For oracle robustness/accuracy we default to a direct factorization unless overridden.
            petsc_options = {
                "ksp_type": solver_params.get("ksp_type", "preonly"),
                "pc_type": solver_params.get("pc_type", "lu"),
                "ksp_rtol": solver_params.get("rtol", 1e-10),
                "ksp_atol": solver_params.get("atol", 1e-12),
            }
            problem = LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options=petsc_options,
                petsc_options_prefix="oracle_darcy_mixed_",
            )
            w_h = problem.solve()

            # Enforce solver success explicitly (otherwise we may silently return a garbage vector).
            if problem.solver.getConvergedReason() <= 0:
                raise RuntimeError(
                    f"Darcy mixed linear solve did not converge (reason={problem.solver.getConvergedReason()})."
                )

            u_h = w_h.sub(0).collapse()
            p_h = w_h.sub(1).collapse()

            if output_field == "pressure":
                _, _, out_grid = sample_scalar_on_grid(
                    p_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                _, _, p_exact_grid = sample_scalar_on_grid(
                    p_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                baseline_error = compute_rel_L2_grid(out_grid, p_exact_grid)
                out_grid = p_exact_grid
            elif output_field == "flux_magnitude":
                _, _, out_grid = sample_vector_magnitude_on_grid(
                    u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                _, _, u_exact_grid = sample_vector_magnitude_on_grid(
                    u_exact, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
                )
                baseline_error = compute_rel_L2_grid(out_grid, u_exact_grid)
                out_grid = u_exact_grid
            else:
                raise ValueError(f"Unsupported Darcy output field: {output_field}")

            # ⏱️ 结束计时（混合形式）
            baseline_time = time.perf_counter() - t_start_total

            solver_info = {
                "formulation": "mixed",
                "ksp_type": petsc_options["ksp_type"],
                "pc_type": petsc_options["pc_type"],
                "rtol": petsc_options["ksp_rtol"],
                "pressure_fixing": pressure_fixing,
                "degree_u": degree_u,
                "degree_p": degree_p,
                "kappa_type": kappa_spec.get("type", "constant"),
                "output_field": output_field,
            }
            return OracleResult(
                baseline_error=float(baseline_error),
                baseline_time=float(baseline_time),
                reference=out_grid,
                solver_info=solver_info,
                num_dofs=W.dofmap.index_map.size_global,
            )

        raise ValueError(f"Unsupported Darcy formulation: {formulation}")

