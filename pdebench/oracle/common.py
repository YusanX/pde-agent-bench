"""Common utilities for oracle solvers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem, mesh
from dolfinx.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc


@dataclass
class OracleResult:
    baseline_error: float
    baseline_time: float
    reference: np.ndarray
    solver_info: Dict[str, Any]
    num_dofs: int


def create_mesh(domain_spec: Dict[str, Any], mesh_spec: Dict[str, Any]) -> mesh.Mesh:
    domain_type = domain_spec["type"]
    resolution = mesh_spec["resolution"]
    cell_type_str = mesh_spec.get("cell_type", "triangle")

    if domain_type == "unit_square":
        if cell_type_str == "triangle":
            cell_type = CellType.triangle
        elif cell_type_str == "quadrilateral":
            cell_type = CellType.quadrilateral
        else:
            raise ValueError(f"Unknown 2D cell type: {cell_type_str}")
        return mesh.create_unit_square(MPI.COMM_WORLD, resolution, resolution, cell_type)

    if domain_type == "unit_cube":
        if cell_type_str == "tetrahedron":
            cell_type = CellType.tetrahedron
        elif cell_type_str == "hexahedron":
            cell_type = CellType.hexahedron
        else:
            raise ValueError(f"Unknown 3D cell type: {cell_type_str}")
        return mesh.create_unit_cube(
            MPI.COMM_WORLD, resolution, resolution, resolution, cell_type
        )

    raise ValueError(f"Unknown domain type: {domain_type}")


def create_scalar_space(msh: mesh.Mesh, family: str, degree: int) -> fem.FunctionSpace:
    return fem.functionspace(msh, (family, degree))


def create_vector_space(
    msh: mesh.Mesh, family: str, degree: int
) -> fem.FunctionSpace:
    return fem.functionspace(msh, (family, degree, (msh.geometry.dim,)))


def create_mixed_space(
    msh: mesh.Mesh, degree_u: int, degree_p: int
) -> fem.FunctionSpace:
    vel_el = ufl.VectorElement("Lagrange", msh.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    return fem.FunctionSpace(msh, mixed_el)


def locate_all_boundary_dofs(
    msh: mesh.Mesh, V: fem.FunctionSpace
) -> np.ndarray:
    def boundary(x):
        return np.ones(x.shape[1], dtype=bool)

    boundary_facets = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, boundary
    )
    return fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)


def build_dirichlet_bc(
    msh: mesh.Mesh,
    V: fem.FunctionSpace,
    value_expr: str,
    t: Optional[float] = None,
) -> fem.DirichletBC:
    x = ufl.SpatialCoordinate(msh)
    expr = parse_expression(value_expr, x, t=t)
    bc_func = fem.Function(V)
    interpolate_expression(bc_func, expr)
    boundary_dofs = locate_all_boundary_dofs(msh, V)
    return fem.dirichletbc(bc_func, boundary_dofs)


def parse_expression(
    expr_str: Union[str, sp.Expr],
    x: ufl.SpatialCoordinate,
    t: Optional[float] = None,
) -> ufl.core.expr.Expr:
    if isinstance(expr_str, sp.Expr):
        expr_sympy = expr_str
    else:
        sx, sy, sz, st = sp.symbols("x y z t", real=True)
        local_dict = {"x": sx, "y": sy, "z": sz}
        if t is not None:
            local_dict["t"] = st
        expr_sympy = sp.sympify(expr_str, locals=local_dict)

    sx, sy, sz, st = sp.symbols("x y z t", real=True)

    def sympy_to_ufl(expr):
        if expr.is_Number:
            return float(expr)
        if expr.is_Symbol:
            if expr == sx:
                return x[0]
            if expr == sy:
                return x[1]
            if expr == sz:
                return x[2] if x.ufl_shape[0] > 2 else 0.0
            if expr == st:
                return t if t is not None else 0.0
            raise ValueError(f"Unknown symbol: {expr}")
        if expr.func == sp.sin:
            return ufl.sin(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.cos:
            return ufl.cos(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.exp:
            return ufl.exp(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.sqrt:
            return ufl.sqrt(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.log:
            return ufl.ln(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.tan:
            return ufl.tan(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.sinh:
            return ufl.sinh(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.cosh:
            return ufl.cosh(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.tanh:
            return ufl.tanh(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.Abs:
            arg = sympy_to_ufl(expr.args[0])
            return ufl.conditional(ufl.gt(arg, 0), arg, -arg)
        if expr.func == sp.Add:
            result = sympy_to_ufl(expr.args[0])
            for arg in expr.args[1:]:
                result = result + sympy_to_ufl(arg)
            return result
        if expr.func == sp.Mul:
            result = sympy_to_ufl(expr.args[0])
            for arg in expr.args[1:]:
                result = result * sympy_to_ufl(arg)
            return result
        if expr.func == sp.Pow:
            base = sympy_to_ufl(expr.args[0])
            exp_val = sympy_to_ufl(expr.args[1])
            return base**exp_val
        if expr == sp.pi:
            return math.pi
        raise NotImplementedError(f"Unsupported sympy function: {expr.func}")

    return sympy_to_ufl(expr_sympy)


def parse_vector_expression(
    expr_list: Iterable[Union[str, sp.Expr]],
    x: ufl.SpatialCoordinate,
    t: Optional[float] = None,
) -> ufl.core.expr.Expr:
    return ufl.as_vector([parse_expression(expr, x, t=t) for expr in expr_list])


def interpolate_expression(func: fem.Function, expr: ufl.core.expr.Expr) -> None:
    interp_points = func.function_space.element.interpolation_points
    expr_compiled = fem.Expression(expr, interp_points)
    func.interpolate(expr_compiled)


def create_kappa_field(
    msh: mesh.Mesh, kappa_spec: Dict[str, Any]
) -> Union[fem.Constant, fem.Function]:
    from dolfinx import default_scalar_type

    if kappa_spec["type"] == "constant":
        return fem.Constant(msh, default_scalar_type(kappa_spec["value"]))
    if kappa_spec["type"] == "expr":
        x = ufl.SpatialCoordinate(msh)
        kappa_expr = parse_expression(kappa_spec["expr"], x)
        V_dg = fem.functionspace(msh, ("DG", 0))
        kappa_func = fem.Function(V_dg)
        interp_points = V_dg.element.interpolation_points
        expr_compiled = fem.Expression(kappa_expr, interp_points)
        kappa_func.interpolate(expr_compiled)
        return kappa_func
    raise ValueError(f"Unknown kappa type: {kappa_spec['type']}")


def compute_L2_error(u_h: fem.Function, u_exact: fem.Function) -> float:
    e = u_h - u_exact
    L2_e_squared = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    L2_exact_squared = fem.assemble_scalar(
        fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    )
    if L2_exact_squared < 1e-15:
        return float(math.sqrt(L2_e_squared))
    return float(math.sqrt(L2_e_squared) / math.sqrt(L2_exact_squared))


def compute_rel_L2_grid(u1: np.ndarray, u2: np.ndarray) -> float:
    mask = ~(np.isnan(u1) | np.isnan(u2))
    diff = (u1 - u2)[mask]
    ref = u2[mask]
    if diff.size == 0:
        return float("nan")
    l2_diff = math.sqrt(float(np.sum(diff**2)))
    l2_ref = math.sqrt(float(np.sum(ref**2)))
    if l2_ref < 1e-15:
        return l2_diff
    return l2_diff / l2_ref


def _eval_on_grid(
    msh: mesh.Mesh,
    eval_fn,
    bbox: List[float],
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from dolfinx import geometry

    xmin, xmax, ymin, ymax = bbox
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")

    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    values = np.full(points.shape[0], np.nan)
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells_on_proc.append(colliding_cells.links(i)[0])
            eval_map.append(i)

    if points_on_proc:
        values_eval = eval_fn(np.array(points_on_proc), np.array(cells_on_proc))
        values[eval_map] = values_eval

    return x_grid, y_grid, values.reshape(nx, ny)


def sample_scalar_on_grid(
    u_fem: fem.Function, bbox: List[float], nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    msh = u_fem.function_space.mesh
    x_grid, y_grid, u_grid = _eval_on_grid(
        msh,
        lambda pts, cells: u_fem.eval(pts, cells).flatten(),
        bbox,
        nx,
        ny,
    )
    return x_grid, y_grid, u_grid


def sample_vector_magnitude_on_grid(
    u_vec: fem.Function, bbox: List[float], nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    msh = u_vec.function_space.mesh

    def eval_fn(pts, cells):
        values = u_vec.eval(pts, cells)
        return np.linalg.norm(values, axis=1)

    x_grid, y_grid, u_mag = _eval_on_grid(msh, eval_fn, bbox, nx, ny)
    return x_grid, y_grid, u_mag
