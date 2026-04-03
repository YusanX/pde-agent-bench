"""Common utilities for oracle solvers."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
import ufl
from dolfinx import fem, mesh
from dolfinx.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc
import os
from typing import Dict, Any
import numpy as np
import pygmsh
import meshio
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType

from ._types import OracleResult, compute_rel_L2_grid  # noqa: F401  re-export


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
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")

    points = np.zeros((ny * nx, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    # 1. 寻找本地碰撞
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    # 2. 准备本地数据：域外初始化为 0
    # 注意：使用 0 而不是 NaN，因为 MPI.Reduce(SUM) 不好处理 NaN
    local_values = np.zeros(points.shape[0], dtype=PETSc.ScalarType)
    
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(points.shape[0]):
        # 只处理位于本进程网格内的点
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(colliding_cells.links(i)[0])
            eval_map.append(i)

    if points_on_proc:
        values_eval = eval_fn(np.array(points_on_proc), np.array(cells_on_proc))
        local_values[eval_map] = values_eval.flatten()

    # 3. 使用 MPI Reduce 汇总所有进程的贡献
    # 因为每个点只会被一个进程“认领”并填充非零值，SUM 即可聚合
    total_values = np.zeros_like(local_values)
    msh.comm.Reduce(local_values, total_values, op=MPI.SUM, root=0)

    if msh.comm.rank == 0:
        return x_grid, y_grid, total_values.reshape(ny, nx)
    else:
        return x_grid, y_grid, np.zeros((ny, nx))

def create_mesh(domain_spec: Dict[str, Any], mesh_spec: Dict[str, Any]) -> mesh.Mesh:
    resolution = mesh_spec.get("resolution", 16)
    char_length = 1.0 / float(resolution)
    domain_type = domain_spec["type"]
    params = domain_spec.get("geometry_params", {})

    def finalize_mesh(geom, dim=2):
        geom.characteristic_length_max = char_length
        mesh_data = geom.generate_mesh()
        cell_key = "triangle" if dim == 2 else "tetra"
        points = mesh_data.points[:, :2] if dim == 2 else mesh_data.points
        out_mesh = meshio.Mesh(points=points, cells={cell_key: mesh_data.cells_dict[cell_key]})
        fname = f"tmp_mesh_{MPI.COMM_WORLD.rank}_{os.getpid()}"
        meshio.write(f"{fname}.xdmf", out_mesh)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            d_mesh = xdmf.read_mesh(name="Grid")
        if MPI.COMM_WORLD.rank == 0:
            for ext in [".xdmf", ".h5"]:
                if os.path.exists(fname + ext): os.remove(fname + ext)
        return d_mesh

    if domain_type == "unit_square":
        return mesh.create_unit_square(MPI.COMM_WORLD, resolution, resolution)

    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_max = char_length

        if domain_type == "l_shape":
            # 动态读取顶点
            v = params.get("vertices", [[0,0], [1,0], [1,0.5], [0.5,0.5], [0.5,1], [0,1]])
            geom.add_polygon([[p[0], p[1], 0] for p in v])

        elif domain_type == "circle":
            c = params.get("center", [0.5, 0.5])
            r = params.get("radius", 0.5)
            geom.add_disk([c[0], c[1], 0], r)

        elif domain_type == "annulus":
            c = params.get("center", [0, 0])
            r_in = params.get("inner_r", 0.5)
            r_out = params.get("outer_r", 1.0)
            c1 = geom.add_disk([c[0], c[1], 0], r_out)
            c2 = geom.add_disk([c[0], c[1], 0], r_in)
            geom.boolean_difference(c1, c2)

        elif domain_type == "square_with_hole":
            # 根据 JSON 结构处理 outer 和 inner_hole
            out = params.get("outer", [0, 1, 0, 1])
            rect = geom.add_rectangle([out[0], out[2], 0], out[1]-out[0], out[3]-out[2])
            ih = params.get("inner_hole", {})
            if ih.get("type") == "circle":
                c, r = ih.get("center", [0.5, 0.5]), ih.get("radius", 0.2)
                hole = geom.add_disk([c[0], c[1], 0], r)
            elif ih.get("type") == "rect":
                b = ih.get("bbox", [0.4, 0.6, 0.4, 0.6])
                hole = geom.add_rectangle([b[0], b[2], 0], b[1]-b[0], b[3]-b[2])
            elif ih.get("type") == "polygon":
                v = ih.get("vertices", [])
                hole = geom.add_polygon([[p[0], p[1], 0] for p in v])
            geom.boolean_difference(rect, hole)

        elif domain_type == "multi_hole":
            out = params.get("outer", [0, 1, 0, 1])
            rect = geom.add_rectangle([out[0], out[2], 0], out[1]-out[0], out[3]-out[2])
            holes = []
            for h in params.get("holes", []):
                c, r = h.get("c", [0,0]), h.get("r", 0.1)
                holes.append(geom.add_disk([c[0], c[1], 0], r))
            geom.boolean_difference(rect, holes)

        elif domain_type == "sector":
            c = params.get("center", [0, 0])
            r = params.get("radius", 1.0)
            ang = math.radians(params.get("angle", 90))
            # 改进的扇形生成逻辑：包含圆心
            pts = [[c[0], c[1], 0]]
            num_arc_pts = 20
            for a in np.linspace(0, ang, num_arc_pts):
                pts.append([c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0])
            geom.add_polygon(pts)

        elif domain_type in ["star", "star_shape"]:
            n = params.get("points", 5)
            r_in = params.get("inner_r", 0.3)
            r_out = params.get("outer_r", 0.7)
            pts = []
            for i in range(2 * n):
                angle = i * math.pi / n - math.pi/2
                r = r_out if i % 2 == 0 else r_in
                pts.append([r * math.cos(angle), r * math.sin(angle), 0])
            geom.add_polygon(pts)

        elif domain_type == "gear":
            n = params.get("teeth", 8)
            r_base = params.get("base_r", 0.5)
            h = params.get("tooth_h", 0.2)
            pts = []
            for i in range(2 * n):
                angle = i * math.pi / n
                r = r_base + h if i % 2 == 0 else r_base
                pts.append([r * math.cos(angle), r * math.sin(angle), 0])
            geom.add_polygon(pts)

        return finalize_mesh(geom)

def create_scalar_space(msh: mesh.Mesh, family: str, degree: int) -> fem.FunctionSpace:
    return fem.functionspace(msh, (family, degree))


def create_vector_space(
    msh: mesh.Mesh, family: str, degree: int
) -> fem.FunctionSpace:
    return fem.functionspace(msh, (family, degree, (msh.geometry.dim,)))


def create_mixed_space(
    msh: mesh.Mesh, degree_u: int, degree_p: int
) -> fem.FunctionSpace:
    from basix.ufl import element as basix_element
    from basix.ufl import mixed_element as basix_mixed_element

    vel_el = basix_element(
        "Lagrange",
        msh.topology.cell_name(),
        degree_u,
        shape=(msh.geometry.dim,),
    )
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    mixed_el = basix_mixed_element([vel_el, pres_el])
    return fem.functionspace(msh, mixed_el)


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
    if isinstance(expr_str, (int, float, np.number)):
        return ufl.as_ufl(float(expr_str))
        
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    local_dict = {"x": sx, "y": sy, "z": sz}
    if t is not None:
        local_dict["t"] = st
    
    expr_sympy = sp.sympify(expr_str, locals=local_dict) if isinstance(expr_str, str) else expr_str

    def sympy_to_ufl(expr):
        if expr.is_Number:
            return ufl.as_ufl(float(expr))
        if expr.is_Symbol:
            if expr == sx: return x[0]
            if expr == sy: return x[1]
            if expr == sz: return x[2] if x.ufl_shape[0] > 2 else 0.0
            if expr == st: return t if t is not None else 0.0
        
        # Functions
        if expr.func == sp.Add: return sum(sympy_to_ufl(a) for a in expr.args)
        if expr.func == sp.Mul: 
            res = sympy_to_ufl(expr.args[0])
            for a in expr.args[1:]: res *= sympy_to_ufl(a)
            return res
        if expr.func == sp.Pow: return sympy_to_ufl(expr.args[0])**sympy_to_ufl(expr.args[1])
        if expr.func == sp.sin: return ufl.sin(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.cos: return ufl.cos(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.exp: return ufl.exp(sympy_to_ufl(expr.args[0]))
        if expr.func == sp.sqrt: return ufl.sqrt(sympy_to_ufl(expr.args[0]))
        if expr == sp.pi: return math.pi
        return ufl.as_ufl(float(expr.evalf()))

    return sympy_to_ufl(expr_sympy)

def parse_vector_expression(
    expr_list: Iterable[Union[str, sp.Expr]],
    x: ufl.SpatialCoordinate,
    t: Optional[float] = None,
) -> ufl.core.expr.Expr:
    return ufl.as_vector([parse_expression(expr, x, t=t) for expr in expr_list])


def interpolate_expression(func: fem.Function, expr: ufl.core.expr.Expr) -> None:
    """Interpolate a UFL expression into a FEM function with explicit communicator."""
    msh = func.function_space.mesh
    # Ensure expression is interpolated at the correct points
    # Explicitly pass comm to avoid "Could not extract MPI communicator" error
    try:
        points = func.function_space.element.interpolation_points
        expr_compiled = fem.Expression(expr, points, comm=msh.comm)
        func.interpolate(expr_compiled)
    except Exception:
        # Fallback for ultra-simple expressions that fem.Expression might still fail on
        if isinstance(expr, (int, float, np.number)):
            func.x.array[:] = float(expr)
        else:
            # Handle cases where expr might be a simple UFL constant
            func.interpolate(lambda x: np.full(x.shape[1], float(ufl.assemble(expr * ufl.dx(msh))/ufl.assemble(1.0 * ufl.dx(msh)))))


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
    # indexing="xy": xx[i,j]=x_grid[j], yy[i,j]=y_grid[i]
    # result[i,j] = f(x_grid[j], y_grid[i])  →  row=y, col=x  (standard image convention)
    # Models sample with the same "xy" convention, so oracle and agent grids are aligned.
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")

    points = np.zeros((ny * nx, 3))
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

    return x_grid, y_grid, values.reshape(ny, nx)


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
