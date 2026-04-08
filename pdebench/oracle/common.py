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


def create_mesh(domain_spec: Dict[str, Any], mesh_spec: Dict[str, Any]) -> mesh.Mesh:
    """根据 JSON 配置生成网格，支持 10+ 种复杂几何。"""
    resolution = mesh_spec.get("resolution", 16)
    char_length = 1.0 / float(resolution)
    domain_type = domain_spec["type"]
    params = domain_spec.get("geometry_params", {})

    def finalize_mesh(geom, dim=2):
        """将 pygmsh 对象转为 dolfinx 网格，包含 KeyError 保护。"""
        geom.characteristic_length_max = char_length
        mesh_data = geom.generate_mesh()

        cell_key = "triangle" if dim == 2 else "tetra"
        pts = mesh_data.points[:, :dim] if dim < mesh_data.points.shape[1] else mesh_data.points
        out_mesh = meshio.Mesh(points=pts, cells={cell_key: mesh_data.cells_dict[cell_key]})
        fname = f"tmp_mesh_{MPI.COMM_WORLD.rank}_{os.getpid()}"
        meshio.write(f"{fname}.xdmf", out_mesh)
        
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            d_mesh = xdmf.read_mesh(name="Grid")
        
        if MPI.COMM_WORLD.rank == 0:
            for ext in [".xdmf", ".h5"]:
                if os.path.exists(fname + ext): os.remove(fname + ext)
        return d_mesh

    # 内置基础形状
    if domain_type == "unit_square":
        return mesh.create_unit_square(MPI.COMM_WORLD, resolution, resolution)

    # 复杂几何 (OpenCASCADE 内核)
    with pygmsh.occ.Geometry() as geom:
        import gmsh
        gmsh.option.setNumber("General.Verbosity", 0)
        geom.characteristic_length_max = char_length

        if domain_type == "l_shape":
            v = params.get("vertices", [[0,0], [1,0], [1,0.5], [0.5,0.5], [0.5,1], [0,1]])
            geom.add_polygon([[p[0], p[1], 0] for p in v])

        elif domain_type == "circle":
            c, r = params.get("center", [0.5, 0.5]), params.get("radius", 0.5)
            geom.add_disk([c[0], c[1], 0], r)

        elif domain_type == "annulus":
            c = params.get("center", [0, 0])
            r_in, r_out = params.get("inner_r", 0.5), params.get("outer_r", 1.0)
            c1 = geom.add_disk([c[0], c[1], 0], r_out)
            c2 = geom.add_disk([c[0], c[1], 0], r_in)
            geom.boolean_difference(c1, c2)

        elif domain_type in ["square_with_hole", "multi_hole"]:
            out = params.get("outer", [0, 1, 0, 1])
            rect = geom.add_rectangle([out[0], out[2], 0], out[1]-out[0], out[3]-out[2])
            if domain_type == "square_with_hole":
                ih = params.get("inner_hole", {})
                if ih.get("type") == "circle":
                    c, r = ih.get("center", [0.5, 0.5]), ih.get("radius", 0.2)
                    hole = geom.add_disk([c[0], c[1], 0], r)
                elif ih.get("type") == "rect":
                    b = ih.get("bbox", [0.4, 0.6, 0.4, 0.6])
                    hole = geom.add_rectangle([b[0], b[2], 0], b[1]-b[0], b[3]-b[2])
                else: # polygon
                    v = ih.get("vertices", [[0.4,0.4], [0.6,0.4], [0.5,0.7]])
                    hole = geom.add_polygon([[p[0], p[1], 0] for p in v])
                geom.boolean_difference(rect, hole)
            else: # multi_hole
                holes = []
                for h in params.get("holes", []):
                    c, r = h.get("c", [0,0]), h.get("r", 0.1)
                    holes.append(geom.add_disk([c[0], c[1], 0], r))
                geom.boolean_difference(rect, holes)

        elif domain_type == "t_junction":
            # 自动适配 rd_tjunction 的两种矩形参数
            h = params.get("horizontal_rect", [0.0, 1.0, 0.4, 0.6])
            v = params.get("vertical_rect", [0.4, 0.6, 0.0, 0.5])
            r1 = geom.add_rectangle([h[0], h[2], 0], h[1]-h[0], h[3]-h[2])
            r2 = geom.add_rectangle([v[0], v[2], 0], v[1]-v[0], v[3]-v[2])
            geom.boolean_union([r1, r2])

        elif domain_type == "sector":
            c, r = params.get("center", [0, 0]), params.get("radius", 1.0)
            ang = math.radians(params.get("angle", 90))
            pts = [[c[0], c[1], 0]]
            for a in np.linspace(0, ang, 15):
                pts.append([c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0])
            geom.add_polygon(pts)

        elif domain_type in ["star", "star_shape"]:
            n = params.get("points", 5)
            r_in, r_out = params.get("inner_r", 0.3), params.get("outer_r", 0.7)
            pts = []
            for i in range(2 * n):
                angle = i * math.pi / n - math.pi/2
                r = r_out if i % 2 == 0 else r_in
                pts.append([r * math.cos(angle), r * math.sin(angle), 0])
            geom.add_polygon(pts)

        elif domain_type == "gear":
            n = params.get("teeth", 8)
            r_base, h = params.get("base_r", 0.5), params.get("tooth_h", 0.2)
            pts = []
            for i in range(2 * n):
                angle = i * math.pi / n
                r = r_base + h if i % 2 == 0 else r_base
                pts.append([r * math.cos(angle), r * math.sin(angle), 0])
            geom.add_polygon(pts)

        elif domain_type == "dumbbell":
            w = params.get("bar_width", 0.2)
            c1, c2 = params.get("left_center", [0.2, 0.5]), params.get("right_center", [0.8, 0.5])
            r = params.get("radius", 0.2)
            d1 = geom.add_disk([c1[0], c1[1], 0], r)
            d2 = geom.add_disk([c2[0], c2[1], 0], r)
            bar = geom.add_rectangle([c1[0], 0.5-w/2, 0], c2[0]-c1[0], w)
            geom.boolean_union([d1, d2, bar])
        
        elif domain_type == "eccentric_annulus":
            outer = params.get("outer_circle", {"c": [0, 0], "r": 1.0})
            inner = params.get("inner_circle", {"c": [0.2, 0], "r": 0.4})
            oc, o_r = outer["c"], outer["r"]
            ic, i_r = inner["c"], inner["r"]
            d1 = geom.add_disk([oc[0], oc[1], 0], o_r)
            d2 = geom.add_disk([ic[0], ic[1], 0], i_r)
            geom.boolean_difference(d1, d2)

        elif domain_type == "periodic_square":
            # 几何上是一个矩形，周期性配对逻辑由 Solver 在创建 FunctionSpace 时调用 create_periodic_map 处理
            out = params.get("extents", [0.0, 1.0, 0.0, 1.0]) # [xmin, xmax, ymin, ymax]
            geom.add_rectangle([out[0], out[2], 0], out[1]-out[0], out[3]-out[2])

        else:
            return mesh.create_unit_square(MPI.COMM_WORLD, resolution, resolution)

        return finalize_mesh(geom)

# --- 采样与并行聚合逻辑 ---

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


def parse_expression(expr_str, x, t=None):
    if isinstance(expr_str, (int, float)): return ufl.as_ufl(float(expr_str))
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    local_dict = {"x": sx, "y": sy, "z": sz, "t": st}
    expr_sympy = sp.sympify(expr_str, locals=local_dict) if isinstance(expr_str, str) else expr_str

    def to_ufl(e):
        if e.is_Number: return ufl.as_ufl(float(e))
        if e.is_Symbol:
            if e == sx: return x[0]
            if e == sy: return x[1]
            if e == sz: return x[2] if x.ufl_shape[0]>2 else 0.0
            if e == st: return t if t is not None else 0.0
        if e.func == sp.Add: return sum(to_ufl(a) for a in e.args)
        if e.func == sp.Mul: 
            res = to_ufl(e.args[0])
            for a in e.args[1:]: res *= to_ufl(a)
            return res
        if e.func == sp.Pow: return to_ufl(e.args[0])**to_ufl(e.args[1])
        if e.func == sp.sin: return ufl.sin(to_ufl(e.args[0]))
        if e.func == sp.cos: return ufl.cos(to_ufl(e.args[0]))
        if e.func == sp.exp: return ufl.exp(to_ufl(e.args[0]))
        return ufl.as_ufl(float(e.evalf()))
    return to_ufl(expr_sympy)

def parse_vector_expression(
    expr_list: Iterable[Union[str, sp.Expr]],
    x: ufl.SpatialCoordinate,
    t: Optional[float] = None,
) -> ufl.core.expr.Expr:
    return ufl.as_vector([parse_expression(expr, x, t=t) for expr in expr_list])


# 修改 common.py 中的这个函数
def interpolate_expression(func: fem.Function, expr: ufl.core.expr.Expr) -> None:
    msh = func.function_space.mesh
    try:
        # 尝试标准插值
        points = func.function_space.element.interpolation_points
        expr_compiled = fem.Expression(expr, points, comm=msh.comm)
        func.interpolate(expr_compiled)
    except Exception:
        # 兜底方案：处理常数或简单的 UFL 表达式
        try:
            # 检查是否为纯数字常数
            val = float(expr)
            func.x.array[:] = val
        except Exception:
            # 如果是 UFL 表达式但插值失败，尝试通过积分计算平均值
            dx_m = ufl.Measure("dx", domain=msh)
            try:
                num = fem.assemble_scalar(fem.form(expr * dx_m))
                den = fem.assemble_scalar(fem.form(1.0 * dx_m))
                func.x.array[:] = float(num / den)
            except:
                func.x.array[:] = 0.0

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


# 修改前：
# L2_e_squared = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
# 修改后：
def compute_L2_error(u_h: fem.Function, u_exact: fem.Function) -> float:
    msh = u_h.function_space.mesh
    dx_m = ufl.Measure("dx", domain=msh) # 显式绑定网格
    e = u_h - u_exact
    # 使用绑定了 domain 的 dx
    L2_e_squared = fem.assemble_scalar(fem.form(ufl.inner(e, e) * dx_m))
    L2_exact_squared = fem.assemble_scalar(
        fem.form(ufl.inner(u_exact, u_exact) * dx_m)
    )
    if L2_exact_squared < 1e-15:
        return float(math.sqrt(L2_e_squared))
    return float(math.sqrt(L2_e_squared) / math.sqrt(L2_exact_squared))

def _eval_on_grid(msh: mesh.Mesh, eval_fn, bbox: List[float], nx: int, ny: int):
    from dolfinx import geometry
    x_grid = np.linspace(bbox[0], bbox[1], nx)
    y_grid = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    points = np.zeros((ny * nx, 3))
    points[:, 0], points[:, 1] = xx.ravel(), yy.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_values = np.zeros(points.shape[0], dtype=PETSc.ScalarType)
    # 使用 NaN 初始化以便后续判断
    local_values[:] = np.nan 

    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(points.shape[0]):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(colliding_cells.links(i)[0])
            eval_map.append(i)

    if points_on_proc:
        values_eval = eval_fn(np.array(points_on_proc), np.array(cells_on_proc))
        local_values[eval_map] = values_eval.flatten()

    # 并行聚合：收集所有进程的数据
    total_values = np.zeros_like(local_values)
    # 用 MPI_MAX 因为有值的点是非 nan，没值的点是 nan (处理 NaN 需要谨慎，这里简化逻辑)
    # 更稳妥的做法是先用 0，最后再处理
    local_values_zero = np.nan_to_num(local_values, nan=0.0)
    msh.comm.Reduce(local_values_zero, total_values, op=MPI.SUM, root=0)

    if msh.comm.rank == 0:
        return x_grid, y_grid, total_values.reshape(ny, nx)
    else:
        return x_grid, y_grid, None
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

def create_periodic_map(extents: List[float], direction: str = "both"):
    """
    为 dolfinx.fem.functionspace 生成周期性边界映射。
    extents: [xmin, xmax, ymin, ymax]
    direction: "x", "y", 或 "both"
    """
    xmin, xmax, ymin, ymax = extents
    def periodic_map(x):
        out_x = np.copy(x)
        if direction in ["x", "both"]:
            is_right = np.isclose(x[0], xmax)
            out_x[0][is_right] = xmin
        if direction in ["y", "both"]:
            is_top = np.isclose(x[1], ymax)
            out_x[1][is_top] = ymin
        return out_x
    return periodic_map