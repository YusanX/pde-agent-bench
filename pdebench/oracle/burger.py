"""Burgers' equation oracle solver (parabolic, linearized).

Equation: du/dt + u*(du/dx + du/dy) - nu*Laplacian(u) = f
Linearized as: (u - u_n)/dt + u_n*(du/dx + du/dy) - nu*Laplacian(u) = f
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from .common import (
    OracleResult,
    compute_rel_L2_grid,
    create_mesh,
    create_scalar_space,
    locate_all_boundary_dofs,
    parse_expression,
    sample_scalar_on_grid,
)

class BurgersSolver:
    """Oracle solver using Semi-implicit Linearization with bulletproof compatibility."""

    def _safe_interpolate(self, func: fem.Function, ufl_expr):
        """Robust interpolation bypassing common.py bugs."""
        V = func.function_space
        # 处理常数
        try:
            func.x.array[:] = float(ufl_expr)
            return
        except Exception:
            pass

        # 兼容不同 dolfinx 版本的插值点获取方式
        try:
            pts = V.element.interpolation_points()
        except TypeError:
            pts = V.element.interpolation_points

        # 兼容不同版本的 Expression 编译
        try:
            expr_c = fem.Expression(ufl_expr, pts)
        except Exception:
            expr_c = fem.Expression(ufl_expr, pts, comm=V.mesh.comm)
            
        func.interpolate(expr_c)

    def _build_safe_bc(self, msh, V, bc_val_str, x_ufl, t_c):
        """Robust boundary condition builder."""
        bc_func = fem.Function(V)
        try:
            # 纯数字
            bc_func.x.array[:] = float(bc_val_str)
        except ValueError:
            # UFL 表达式
            expr = parse_expression(bc_val_str, x_ufl, t=t_c)
            self._safe_interpolate(bc_func, expr)
            
        bdofs = locate_all_boundary_dofs(msh, V)
        return fem.dirichletbc(bc_func, bdofs)


    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        t_start_total = time.perf_counter()
        
        msh = create_mesh(case_spec["domain"], case_spec["mesh"])
        V = create_scalar_space(msh, case_spec["fem"]["family"], case_spec["fem"]["degree"])
        dx = ufl.Measure("dx", domain=msh)
        
        pde_cfg = case_spec["pde"]
        params = pde_cfg.get("pde_params", {})
        nu = fem.Constant(msh, float(params.get("nu", 0.01)))
        
        t_val = 0.0
        T_final = float(pde_cfg.get("t_final", 0.1))
        dt_val = float(pde_cfg.get("dt", 0.01))
        dt_c = fem.Constant(msh, dt_val)
        t_c = fem.Constant(msh, t_val) 
        
        x_ufl = ufl.SpatialCoordinate(msh)
        
        # 1. 表达式解析
        manufactured = pde_cfg.get("manufactured_solution", {})
        if "u" in manufactured:
            sx, sy, st = sp.symbols("x y t", real=True)
            u_sym = sp.sympify(manufactured["u"], locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
            
            # f = u_t + u*(u_x + u_y) - nu*Lap(u)
            u_t = sp.diff(u_sym, st)
            u_x = sp.diff(u_sym, sx)
            u_y = sp.diff(u_sym, sy)
            lap_u = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
            f_sym = u_t + u_sym * (u_x + u_y) - float(params.get("nu", 0.01)) * lap_u
            
            f_ufl = parse_expression(f_sym, x_ufl, t=t_c)
            u_exact_ufl = parse_expression(u_sym, x_ufl, t=t_c)
        else:
            f_ufl = parse_expression(pde_cfg.get("source_term", "0.0"), x_ufl, t=t_c)
            u_exact_ufl = None

        # 2. 状态变量
        u_n = fem.Function(V)
        u_h_step = fem.Function(V)  # 当前时间步的解（专门用于 LinearProblem 接收参数）
        
        t_c.value = 0.0
        if u_exact_ufl:
            self._safe_interpolate(u_n, u_exact_ufl)
        
        # 3. 定义线性变分形式
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        convection = u_n * (u.dx(0) + u.dx(1))
        a = (u / dt_c) * v * dx + convection * v * dx + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        L = (u_n / dt_c) * v * dx + f_ufl * v * dx

        safe_petsc_options = {
            "ksp_type": "gmres",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-8,
            "ksp_atol": 1e-10,
        }

        # 4. 时间步迭代
        num_steps = max(1, int(T_final / dt_val))
        
        for step in range(num_steps):
            t_val += dt_val
            t_c.value = t_val 
            
            # 安全边界条件
            if u_exact_ufl:
                bc_func = fem.Function(V)
                self._safe_interpolate(bc_func, u_exact_ufl)
                bcs = [fem.dirichletbc(bc_func, locate_all_boundary_dofs(msh, V))]
            else:
                bc_val_str = str(case_spec.get("bc", {}).get("dirichlet", {}).get("value", "0.0"))
                bcs = [self._build_safe_bc(msh, V, bc_val_str, x_ufl, t_c)]

            # 严格按照 dolfinx 版本规范传入参数，补齐 u 和 petsc_options_prefix
            problem = LinearProblem(
                a, 
                L, 
                bcs=bcs, 
                u=u_h_step, 
                petsc_options=safe_petsc_options, 
                petsc_options_prefix="burgers_"
            )
            
            # 执行求解
            problem.solve()
            
            # 步进更新
            u_n.x.array[:] = u_h_step.x.array[:]

        # 5. 结果采样
        grid_cfg = case_spec["output"]["grid"]
        _, _, u_grid = sample_scalar_on_grid(u_n, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])

        error = 0.0
        if u_exact_ufl:
            t_c.value = t_val
            u_ex_final = fem.Function(V)
            self._safe_interpolate(u_ex_final, u_exact_ufl)
            _, _, ex_grid = sample_scalar_on_grid(u_ex_final, grid_cfg["bbox"], grid_cfg["nx"], grid_cfg["ny"])
            error = compute_rel_L2_grid(u_grid, ex_grid)
            u_grid = ex_grid

        if msh.comm.rank == 0:
            return OracleResult(
                float(error),
                time.perf_counter() - t_start_total,
                u_grid,
                {"steps": num_steps, "dt": dt_val},
                V.dofmap.index_map.size_global
            )
        return OracleResult(0.0, 0.0, np.array([]), {}, 0)