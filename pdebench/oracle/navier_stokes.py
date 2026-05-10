"""Navier-Stokes oracle solver (steady incompressible flow, nonlinear)."""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
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
    if hasattr(expr, "ufl_domain") and expr.ufl_domain() is None:
        return expr + 0.0 * x[0]
    return expr


def _build_dirichlet_bcs(
    msh, W, bc_cfg: Any, u_exact: fem.Function | None, dim: int
) -> list[fem.DirichletBC]:
    """Build Dirichlet BCs from configuration."""
    bcs = []
    dirichlet_cfgs = _normalize_dirichlet_cfg(bc_cfg)
    V, _ = W.sub(0).collapse()
    fdim = msh.topology.dim - 1
    for cfg in dirichlet_cfgs:
        on = cfg.get("on", "all")
        if on.lower() in {"all", "*"}:
            # Topological approach: locate boundary facets first, then DOFs on those facets.
            # locate_dofs_geometrical with np.ones(...) would incorrectly select ALL interior
            # DOFs as Dirichlet, bypassing the FEM solve entirely.
            facets = dmesh.locate_entities_boundary(
                msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
            )
            boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        else:
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

            try:
                const_values = [float(expr) for expr in expr_list]
                is_constant = True
            except (ValueError, TypeError):
                is_constant = False

            bc_func = fem.Function(V)
            if is_constant:
                bc_func.interpolate(
                    lambda x: np.array([[v] * x.shape[1] for v in const_values])
                )
            else:
                x = ufl.SpatialCoordinate(msh)
                bc_components = [
                    _ensure_domain_scalar(parse_expression(expr, x), x)
                    for expr in expr_list
                ]
                bc_expr = ufl.as_vector(bc_components)
                interpolate_expression(bc_func, bc_expr)
        bcs.append(fem.dirichletbc(bc_func, boundary_dofs, W.sub(0)))
    return bcs


def _compute_manufactured_forcing(
    msh,
    manufactured: Dict[str, Any],
    nu: float,
    dim: int,
    V: fem.FunctionSpace,
    Q: fem.FunctionSpace,
):
    x = ufl.SpatialCoordinate(msh)
    sx, sy, sz = sp.symbols("x y z", real=True)
    local_dict = {"x": sx, "y": sy, "z": sz}
    u_sym = manufactured["u"]
    if len(u_sym) != dim:
        raise ValueError("manufactured_solution.u dimension mismatch")
    p_sym = sp.sympify(manufactured["p"], locals=local_dict)
    u_sym_vec = [sp.sympify(u_sym[i], locals=local_dict) for i in range(dim)]

    coords = [sx, sy, sz][:dim]
    div_sym = sum(sp.diff(u_sym_vec[i], coords[i]) for i in range(dim))
    div_simplified = sp.simplify(div_sym)
    if not div_simplified.equals(0):
        raise ValueError("manufactured_solution.u is not divergence-free")

    f_sym = []
    for i, ui in enumerate(u_sym_vec):
        conv = sum(u_sym_vec[j] * sp.diff(ui, coords[j]) for j in range(dim))
        lap = sum(sp.diff(ui, c, 2) for c in coords)
        grad_p = sp.diff(p_sym, coords[i])
        f_sym.append(conv - nu * lap + grad_p)

    f_expr = parse_vector_expression(f_sym, x)
    u_exact_expr = parse_vector_expression(u_sym_vec, x)
    p_exact_expr = parse_expression(p_sym, x)

    u_exact = fem.Function(V)
    p_exact = fem.Function(Q)
    interpolate_expression(u_exact, u_exact_expr)
    interpolate_expression(p_exact, p_exact_expr)

    return f_expr, u_exact, p_exact


def _pressure_point_bc(W, dim: int) -> fem.DirichletBC | None:
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
    if len(p_dofs) == 0:
        return None
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    return fem.dirichletbc(p0, p_dofs, W.sub(1))


def _solve_navier_stokes(
    msh,
    W,
    f_expr,
    nu: float,
    bcs: list[fem.DirichletBC],
    solver_params: Dict[str, Any],
    init_mode: str,
    u_exact: "fem.Function | None" = None,
) -> tuple[fem.Function, Dict[str, Any]]:
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    J = ufl.derivative(F, w)
    if init_mode == "stokes":
        (u_s, p_s) = ufl.TrialFunctions(W)
        (v_s, q_s) = ufl.TestFunctions(W)
        a_stokes = (
            nu * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
            - ufl.div(v_s) * p_s * ufl.dx
            - q_s * ufl.div(u_s) * ufl.dx
        )
        L_stokes = ufl.inner(f_expr, v_s) * ufl.dx
        stokes_options = {
            "ksp_type": solver_params.get("stokes_ksp_type", "minres"),
            "pc_type": solver_params.get("stokes_pc_type", "hypre"),
            "ksp_rtol": solver_params.get("stokes_ksp_rtol", 1e-10),
        }
        stokes_problem = LinearProblem(
            a_stokes,
            L_stokes,
            bcs=bcs,
            petsc_options=stokes_options,
            petsc_options_prefix="oracle_navier_stokes_init_",
        )
        w0 = stokes_problem.solve()
        w.x.array[:] = w0.x.array
    elif init_mode == "zero":
        w.x.array[:] = 0.0
    elif init_mode == "exact":
        # Initialize from interpolated exact solution — guarantees Newton starts
        # in the correct basin of attraction. Only valid for manufactured cases.
        if u_exact is None:
            raise ValueError("init='exact' requires a manufactured solution (u_exact)")
        V, _ = W.sub(0).collapse()
        from .common import interpolate_expression
        import ufl as _ufl
        x_coord = _ufl.SpatialCoordinate(msh)
        # u_exact is already a fem.Function in V; scatter its DOFs into W.sub(0)
        u_init = fem.Function(V)
        u_init.x.array[:] = u_exact.x.array
        # Also zero-initialize pressure (will be corrected by Newton)
        w.x.array[:] = 0.0
        # Inject u_exact values into the velocity sub-space of w
        W.sub(0).collapse()
        try:
            # Map DOFs from collapsed V to W.sub(0)
            dof_map = W.sub(0).dofmap
            w_arr = w.x.array
            # Use dolfinx interpolation into sub-space
            w_u = w.sub(0)
            w_u.interpolate(u_init)
            w.x.scatter_forward()
        except Exception:
            # Fallback: set via petsc BC enforcement using u_exact as the BC value
            fem.petsc.set_bc(w.x.petsc_vec, bcs)
            w.x.scatter_forward()
    elif init_mode == "continuation":
        # Re-continuation: solve from high nu → target nu, using each solution
        # as warm-start for the next step. Robust for moderate/high Re cases.
        nu_start = solver_params.get("continuation_nu_start", 1.0)
        n_steps = solver_params.get("continuation_steps", 8)
        nu_values = [
            nu_start * (nu / nu_start) ** (k / n_steps)
            for k in range(1, n_steps + 1)
        ]
        # step 0: Stokes init at nu_start
        (u_s, p_s) = ufl.TrialFunctions(W)
        (v_s, q_s) = ufl.TestFunctions(W)
        a_stokes_c = (
            nu_start * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
            - ufl.div(v_s) * p_s * ufl.dx
            - q_s * ufl.div(u_s) * ufl.dx
        )
        L_stokes_c = ufl.inner(f_expr, v_s) * ufl.dx
        stokes_c = LinearProblem(
            a_stokes_c, L_stokes_c, bcs=bcs,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre", "ksp_rtol": 1e-10},
            petsc_options_prefix="oracle_ns_cont_init_",
        )
        w0 = stokes_c.solve()
        w.x.array[:] = w0.x.array
        # continuation loop
        cont_opts = {
            "snes_type": "newtonls",
            "snes_linesearch_type": solver_params.get("linesearch", "bt"),
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": solver_params.get("max_it", 50),
            "ksp_type": solver_params.get("ksp_type", "gmres"),
            "pc_type": solver_params.get("pc_type", "lu"),
        }
        for step_nu in nu_values:
            w_step = fem.Function(W)
            u_step, p_step = ufl.split(w_step)
            v_step, q_step = ufl.TestFunctions(W)
            F_step = (
                ufl.inner(ufl.dot(ufl.grad(u_step), u_step), v_step) * ufl.dx
                + step_nu * ufl.inner(ufl.grad(u_step), ufl.grad(v_step)) * ufl.dx
                - p_step * ufl.div(v_step) * ufl.dx
                - q_step * ufl.div(u_step) * ufl.dx
                - ufl.inner(f_expr, v_step) * ufl.dx
            )
            J_step = ufl.derivative(F_step, w_step)
            w_step.x.array[:] = w.x.array
            prob_step = NonlinearProblem(
                F_step, w_step, bcs=bcs, J=J_step,
                petsc_options_prefix=f"oracle_ns_cont_step_",
                petsc_options=cont_opts,
            )
            w_step = prob_step.solve()
            w.x.array[:] = w_step.x.array
    else:
        raise ValueError(f"Unsupported init mode: {init_mode}")

    snes_rtol = solver_params.get("rtol", 1e-10)
    snes_atol = solver_params.get("atol", 1e-12)
    snes_max_it = solver_params.get("max_it", 50)
    line_search = solver_params.get("linesearch", "bt")
    ksp_type = solver_params.get("ksp_type", "gmres")
    pc_type = solver_params.get("pc_type", "lu")
    ksp_rtol = solver_params.get("ksp_rtol", 1e-10)
    ksp_atol = solver_params.get("ksp_atol", 1e-12)
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": line_search,
        "snes_rtol": snes_rtol,
        "snes_atol": snes_atol,
        "snes_max_it": snes_max_it,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": ksp_rtol,
        "ksp_atol": ksp_atol,
    }
    if "pc_factor_mat_solver_type" in solver_params:
        petsc_options["pc_factor_mat_solver_type"] = solver_params[
            "pc_factor_mat_solver_type"
        ]

    problem = NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="oracle_navier_stokes_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()

    solver_info = {
        "snes_type": petsc_options["snes_type"],
        "snes_linesearch_type": line_search,
        "snes_rtol": snes_rtol,
        "snes_atol": snes_atol,
        "snes_max_it": snes_max_it,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": ksp_rtol,
        "ksp_atol": ksp_atol,
        "init_mode": init_mode,
    }
    return w_h, solver_info


class NavierStokesSolver:
    """Taylor-Hood mixed solver for steady incompressible Navier-Stokes."""

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

        manufactured = pde_cfg.get("manufactured_solution", {})
        source_expr = pde_cfg.get("source_term")
        f_expr = None
        u_exact = None
        p_exact = None

        if "u" in manufactured and "p" in manufactured:
            V, _ = W.sub(0).collapse()
            Q, _ = W.sub(1).collapse()
            f_expr, u_exact, p_exact = _compute_manufactured_forcing(
                msh, manufactured, nu, dim, V, Q
            )
        elif source_expr is not None:
            x = ufl.SpatialCoordinate(msh)
            if isinstance(source_expr, (list, tuple)):
                if len(source_expr) != dim:
                    raise ValueError("source_term dimension mismatch")
                expr_list = list(source_expr)
            else:
                expr_list = [source_expr] * dim
            try:
                const_values = [float(sp.sympify(expr)) for expr in expr_list]
                f_expr = fem.Constant(msh, tuple(const_values))
            except (ValueError, TypeError, AttributeError, Exception):
                f_components = [
                    _ensure_domain_scalar(parse_expression(expr, x), x)
                    for expr in expr_list
                ]
                f_expr = ufl.as_vector(f_components)
        else:
            f_expr = fem.Constant(msh, tuple([0.0] * dim))

        solver_params = case_spec.get("oracle_solver", {})

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
            raise ValueError("Navier-Stokes requires Dirichlet boundary conditions")

        pressure_fixing = solver_params.get("pressure_fixing", "point")
        if pressure_fixing == "point":
            p_bc = _pressure_point_bc(W, dim)
            if p_bc is not None:
                bcs.append(p_bc)
        elif pressure_fixing == "none":
            pass
        else:
            raise ValueError(f"Unsupported pressure_fixing: {pressure_fixing}")

        init_mode = solver_params.get("init", "stokes")

        baseline_time = time.time()
        w_h, solver_info = _solve_navier_stokes(
            msh, W, f_expr, nu, bcs, solver_params, init_mode,
            u_exact=u_exact,
        )
        baseline_time = time.time() - baseline_time

        u_h = w_h.sub(0).collapse()

        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_vector_magnitude_on_grid(
            u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
        )

        baseline_error = 0.0
        if u_exact is not None:
            # Evaluate the EXACT analytical solution directly on the output grid
            # (not via FEM interpolation) so that oracle_error reflects the true
            # FEM discretisation error.  This avoids the pathological case where
            # FEM init from u_exact makes oracle_error identically 0.
            bbox = grid_cfg["bbox"]
            nx_g, ny_g = grid_cfg["nx"], grid_cfg["ny"]
            x_lin = np.linspace(bbox[0], bbox[1], nx_g)
            y_lin = np.linspace(bbox[2], bbox[3], ny_g)
            xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
            manufactured = pde_cfg.get("manufactured_solution", {})
            u_sym_strs = manufactured.get("u", [])
            sx, sy = sp.symbols("x y", real=True)
            local_d = {"x": sx, "y": sy}
            try:
                u_sym_vec = [sp.sympify(s, locals=local_d) for s in u_sym_strs]
                u1_fn = sp.lambdify((sx, sy), u_sym_vec[0], "numpy")
                u2_fn = sp.lambdify((sx, sy), u_sym_vec[1], "numpy")
                u_mag_exact = np.sqrt(u1_fn(xx, yy) ** 2 + u2_fn(xx, yy) ** 2)
                u_exact_grid = u_mag_exact  # shape (ny, nx)
            except Exception:
                # Fallback: use FEM interpolant (old behaviour)
                _, _, u_exact_grid = sample_vector_magnitude_on_grid(
                    u_exact, bbox, nx_g, ny_g
                )
            baseline_error = compute_rel_L2_grid(u_grid, u_exact_grid)
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
            if source_expr is not None:
                ref_x = ufl.SpatialCoordinate(ref_msh)
                if isinstance(source_expr, (list, tuple)):
                    if len(source_expr) != ref_dim:
                        raise ValueError("reference source_term dimension mismatch")
                    ref_expr_list = list(source_expr)
                else:
                    ref_expr_list = [source_expr] * ref_dim
                try:
                    ref_const_values = [float(sp.sympify(expr)) for expr in ref_expr_list]
                    ref_f_expr = fem.Constant(ref_msh, tuple(ref_const_values))
                except (ValueError, TypeError, AttributeError, Exception):
                    ref_f_components = [
                        _ensure_domain_scalar(parse_expression(expr, ref_x), ref_x)
                        for expr in ref_expr_list
                    ]
                    ref_f_expr = ufl.as_vector(ref_f_components)
            else:
                ref_f_expr = fem.Constant(ref_msh, tuple([0.0] * ref_dim))

            ref_dirichlet_cfg = dirichlet_cfg or {"on": "all", "value": "0.0"}
            ref_bcs = _build_dirichlet_bcs(
                ref_msh, ref_W, ref_dirichlet_cfg, None, ref_dim
            )
            if not ref_bcs:
                raise ValueError("Reference Navier-Stokes requires Dirichlet BCs")

            ref_pressure_fixing = ref_solver.get("pressure_fixing", pressure_fixing)
            if ref_pressure_fixing == "point":
                ref_p_bc = _pressure_point_bc(ref_W, ref_dim)
                if ref_p_bc is not None:
                    ref_bcs.append(ref_p_bc)
            elif ref_pressure_fixing != "none":
                raise ValueError(f"Unsupported reference pressure_fixing: {ref_pressure_fixing}")

            ref_init = ref_solver.get("init", init_mode)
            ref_w, _ = _solve_navier_stokes(
                ref_msh, ref_W, ref_f_expr, nu, ref_bcs, ref_solver, ref_init
            )
            ref_u_h = ref_w.sub(0).collapse()
            _, _, ref_grid = sample_vector_magnitude_on_grid(
                ref_u_h, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"]
            )
            baseline_error = compute_rel_L2_grid(u_grid, ref_grid)
            u_grid = ref_grid

        solver_info.update(
            {
                "nu": nu,
                "pressure_fixing": pressure_fixing,
                "degree_u": degree_u,
                "degree_p": degree_p,
                "mesh_resolution": case_spec["mesh"].get("resolution"),
                "cell_type": case_spec["mesh"].get("cell_type", "triangle"),
            }
        )

        # ⏱️ 结束计时（包含完整流程）
        baseline_time = time.perf_counter() - t_start_total

        return OracleResult(
            baseline_error=float(baseline_error),
            baseline_time=float(baseline_time),
            reference=u_grid,
            solver_info=solver_info,
            num_dofs=W.dofmap.index_map.size_global,
        )