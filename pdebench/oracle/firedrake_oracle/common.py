"""Common utilities for Firedrake oracle solvers.

Firedrake API differences from DOLFINx:
- Mesh:        UnitSquareMesh(nx, ny) / UnitSquareMesh(nx, ny, quadrilateral=True)
- Spaces:      FunctionSpace(mesh, "CG", degree) / VectorFunctionSpace / W = V * Q
- BCs:         DirichletBC(V, value, "on_boundary")  (no explicit DOF location needed)
- Solve:       solve(a == L, uh, bcs=bcs, solver_parameters={...})
- Nonlinear:   NonlinearVariationalSolver with derivative(F, w)
- Sampling:    Function.at(coords) for point evaluation on regular grids (preserves input order)
- Interpolate: Function(V).interpolate(expr)  or  interpolate(expr, V)
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

# Firedrake re-exports UFL; we import everything from firedrake for convenience.
from firedrake import (
    UnitSquareMesh,
    UnitCubeMesh,
    Mesh,
    PeriodicRectangleMesh,
    FunctionSpace,
    VectorFunctionSpace,
    MixedFunctionSpace,
    Function,
    TrialFunction,
    TestFunction,
    TrialFunctions,
    TestFunctions,
    SpatialCoordinate,
    DirichletBC,
    Constant,
    interpolate,
    split,
    solve,
    inner,
    grad,
    div,
    dot,
    dx,
    CellDiameter,
    FacetNormal,
    nabla_grad,
    as_vector,
    as_ufl,
    sqrt,
    sin,
    cos,
    exp,
    ln,
    tan,
    sinh,
    cosh,
    tanh,
    conditional,
    gt,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    LinearVariationalProblem,
    LinearVariationalSolver,
    MixedVectorSpaceBasis,
    VectorSpaceBasis,
)
import ufl

# Re-export OracleResult from the parent common module (library-agnostic dataclass).
from .._types import OracleResult, compute_rel_L2_grid


# =============================================================================
# Mesh creation helpers
# =============================================================================

def _build_pygmsh_mesh_file(domain_spec: Dict[str, Any], char_length: float) -> str:
    """
    使用 pygmsh 生成复杂几何网格，写成 gmsh v2 ASCII 格式 (.msh)，返回临时文件路径。
    Firedrake 的 Mesh() 可直接读取 gmsh v2/v4 格式。

    支持的 domain 类型：
        l_shape, circle, annulus, square_with_hole, multi_hole,
        t_junction, sector, star / star_shape, gear, dumbbell,
        eccentric_annulus, periodic_square
    """
    import pygmsh
    import meshio
    import gmsh

    domain_type = domain_spec["type"]
    params = domain_spec.get("geometry_params", {})

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("General.Verbosity", 0)
        geom.characteristic_length_max = char_length

        if domain_type == "l_shape":
            v = params.get("vertices", [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]])
            geom.add_polygon([[p[0], p[1], 0] for p in v])

        elif domain_type == "circle":
            c, r = params.get("center", [0.5, 0.5]), params.get("radius", 0.5)
            geom.add_disk([c[0], c[1], 0], r)

        elif domain_type == "annulus":
            c = params.get("center", [0, 0])
            r_in = params.get("inner_r", 0.5)
            r_out = params.get("outer_r", 1.0)
            d1 = geom.add_disk([c[0], c[1], 0], r_out)
            d2 = geom.add_disk([c[0], c[1], 0], r_in)
            geom.boolean_difference(d1, d2)

        elif domain_type in ("square_with_hole", "multi_hole"):
            out = params.get("outer", [0, 1, 0, 1])
            rect = geom.add_rectangle([out[0], out[2], 0], out[1] - out[0], out[3] - out[2])
            if domain_type == "square_with_hole":
                ih = params.get("inner_hole", {})
                if ih.get("type") == "circle":
                    c, r = ih.get("center", [0.5, 0.5]), ih.get("radius", 0.2)
                    hole = geom.add_disk([c[0], c[1], 0], r)
                elif ih.get("type") == "rect":
                    b = ih.get("bbox", [0.4, 0.6, 0.4, 0.6])
                    hole = geom.add_rectangle([b[0], b[2], 0], b[1] - b[0], b[3] - b[2])
                else:
                    v = ih.get("vertices", [[0.4, 0.4], [0.6, 0.4], [0.5, 0.7]])
                    hole = geom.add_polygon([[p[0], p[1], 0] for p in v])
                geom.boolean_difference(rect, hole)
            else:  # multi_hole
                holes = []
                for h in params.get("holes", []):
                    c, r = h.get("c", [0, 0]), h.get("r", 0.1)
                    holes.append(geom.add_disk([c[0], c[1], 0], r))
                geom.boolean_difference(rect, holes)

        elif domain_type == "t_junction":
            h = params.get("horizontal_rect", [0.0, 1.0, 0.4, 0.6])
            v = params.get("vertical_rect", [0.4, 0.6, 0.0, 0.5])
            r1 = geom.add_rectangle([h[0], h[2], 0], h[1] - h[0], h[3] - h[2])
            r2 = geom.add_rectangle([v[0], v[2], 0], v[1] - v[0], v[3] - v[2])
            geom.boolean_union([r1, r2])

        elif domain_type == "sector":
            c, r = params.get("center", [0, 0]), params.get("radius", 1.0)
            ang = math.radians(params.get("angle", 90))
            pts = [[c[0], c[1], 0]]
            for a in np.linspace(0, ang, 15):
                pts.append([c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0])
            geom.add_polygon(pts)

        elif domain_type in ("star", "star_shape"):
            n = params.get("points", 5)
            r_in = params.get("inner_r", 0.3)
            r_out = params.get("outer_r", 0.7)
            pts = []
            for i in range(2 * n):
                angle = i * math.pi / n - math.pi / 2
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

        elif domain_type == "dumbbell":
            w = params.get("bar_width", 0.2)
            c1 = params.get("left_center", [0.2, 0.5])
            c2 = params.get("right_center", [0.8, 0.5])
            r = params.get("radius", 0.2)
            d1 = geom.add_disk([c1[0], c1[1], 0], r)
            d2 = geom.add_disk([c2[0], c2[1], 0], r)
            bar = geom.add_rectangle([c1[0], 0.5 - w / 2, 0], c2[0] - c1[0], w)
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
            out = params.get("extents", [0.0, 1.0, 0.0, 1.0])
            geom.add_rectangle([out[0], out[2], 0], out[1] - out[0], out[3] - out[2])

        else:
            raise ValueError(f"Unsupported domain type for Firedrake: {domain_type}")

        mesh_data = geom.generate_mesh()

    # 提取 2D 三角形网格并写成 gmsh v2 ASCII 格式（Firedrake 兼容）
    pts = mesh_data.points[:, :2]
    triangles = mesh_data.cells_dict.get("triangle")
    if triangles is None:
        raise RuntimeError(
            f"pygmsh generated no triangle cells for domain type '{domain_type}'. "
            "Check geometry parameters."
        )
    out_mesh = meshio.Mesh(points=pts, cells={"triangle": triangles})

    fd, tmp_path = tempfile.mkstemp(suffix=".msh", prefix="firedrake_mesh_")
    os.close(fd)
    meshio.write(tmp_path, out_mesh, file_format="gmsh22")
    return tmp_path


# =============================================================================
# Mesh creation
# =============================================================================

def create_mesh(domain_spec: Dict[str, Any], mesh_spec: Dict[str, Any]):
    """Create a Firedrake mesh from domain/mesh configuration.

    Supports all domain types in benchmark_v2.jsonl:
      unit_square, unit_cube  – built-in Firedrake meshes
      periodic_square         – PeriodicRectangleMesh (geometry identical to unit_square,
                                periodic pairing handled at function-space level)
      everything else         – pygmsh + gmsh v2 .msh file → Mesh()
    """
    domain_type = domain_spec["type"]
    resolution = mesh_spec["resolution"]
    cell_type = mesh_spec.get("cell_type", "triangle")
    char_length = 1.0 / float(resolution)

    if domain_type == "unit_square":
        quad = (cell_type == "quadrilateral")
        return UnitSquareMesh(resolution, resolution, quadrilateral=quad)

    if domain_type == "unit_cube":
        hexahedral = (cell_type == "hexahedron")
        return UnitCubeMesh(resolution, resolution, resolution, hexahedral=hexahedral)

    if domain_type == "periodic_square":
        # Firedrake 的周期性网格需要专门的 PeriodicRectangleMesh。
        # 几何形状与 unit_square 相同；周期性配对由 Firedrake 内核处理。
        params = domain_spec.get("geometry_params", {})
        extents = params.get("extents", [0.0, 1.0, 0.0, 1.0])
        Lx = extents[1] - extents[0]
        Ly = extents[3] - extents[2]
        direction = params.get("direction", "both")
        if direction == "x":
            return PeriodicRectangleMesh(resolution, resolution, Lx, Ly,
                                         direction="x", quadrilateral=(cell_type == "quadrilateral"))
        elif direction == "y":
            return PeriodicRectangleMesh(resolution, resolution, Lx, Ly,
                                         direction="y", quadrilateral=(cell_type == "quadrilateral"))
        else:  # "both"
            return PeriodicRectangleMesh(resolution, resolution, Lx, Ly,
                                         quadrilateral=(cell_type == "quadrilateral"))

    # 复杂几何：通过 pygmsh 生成 .msh，Firedrake 的 Mesh() 直接加载
    msh_file = _build_pygmsh_mesh_file(domain_spec, char_length)
    try:
        msh = Mesh(msh_file)
    finally:
        # Firedrake 在 Mesh() 返回时已将网格读入内存，临时文件可立即删除
        try:
            os.remove(msh_file)
        except OSError:
            pass
    return msh


# =============================================================================
# Function space creation
# =============================================================================

def create_scalar_space(msh, family: str, degree: int):
    """Scalar CG (Lagrange) function space."""
    fd_family = "CG" if family in ("Lagrange", "CG") else family
    return FunctionSpace(msh, fd_family, degree)


def create_vector_space(msh, family: str, degree: int):
    """Vector CG function space for elasticity / velocity fields."""
    fd_family = "CG" if family in ("Lagrange", "CG") else family
    return VectorFunctionSpace(msh, fd_family, degree)


def create_mixed_space(msh, degree_u: int = 2, degree_p: int = 1):
    """Taylor-Hood mixed space V × Q for Stokes / Navier-Stokes."""
    V = VectorFunctionSpace(msh, "CG", degree_u)
    Q = FunctionSpace(msh, "CG", degree_p)
    return V * Q


# =============================================================================
# Expression parsing  (sympy → UFL, library-agnostic)
# =============================================================================

def parse_expression(
    expr_str: Union[str, sp.Expr],
    x,          # Firedrake SpatialCoordinate output
    t: Optional[float] = None,
) -> ufl.core.expr.Expr:
    """Convert a sympy / string expression to a UFL expression suitable for Firedrake."""
    if isinstance(expr_str, sp.Expr):
        expr_sympy = expr_str
    else:
        sx, sy, sz, st = sp.symbols("x y z t", real=True)
        local_dict = {"x": sx, "y": sy, "z": sz, "pi": sp.pi}
        if t is not None:
            local_dict["t"] = st
        expr_sympy = sp.sympify(expr_str, locals=local_dict)

    sx, sy, sz, st = sp.symbols("x y z t", real=True)

    def sympy_to_ufl(expr):
        if expr.is_Number:
            val = float(expr)
            # Bind to domain so integration measure is available
            return ufl.as_ufl(val) * (1.0 + 0.0 * x[0])
        if expr.is_Symbol:
            if expr == sx:
                return x[0]
            if expr == sy:
                return x[1]
            if expr == sz:
                return x[2] if hasattr(x, '__len__') and len(x) > 2 else 0.0
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
            return base ** exp_val
        if expr == sp.pi:
            return math.pi
        raise NotImplementedError(f"Unsupported sympy function: {expr.func}")

    return sympy_to_ufl(expr_sympy)


def parse_vector_expression(
    expr_list: Iterable[Union[str, sp.Expr]],
    x,
    t: Optional[float] = None,
) -> ufl.core.expr.Expr:
    return ufl.as_vector([parse_expression(expr, x, t=t) for expr in expr_list])


# =============================================================================
# Boundary conditions helpers
# =============================================================================

def _parse_bc_value_scalar(value, V, x, t=None):
    """Return a Firedrake Function or Constant suitable for DirichletBC."""
    if isinstance(value, (int, float)):
        return Constant(float(value))
    if isinstance(value, str):
        try:
            c = float(sp.sympify(value))
            return Constant(c)
        except Exception:
            expr = parse_expression(value, x, t=t)
            f = Function(V)
            f.interpolate(expr)
            return f
    raise TypeError(f"Unsupported BC value type: {type(value)}")


def build_scalar_bc(V, value, x, t=None, boundary="on_boundary"):
    """Build a Dirichlet BC for a scalar space."""
    val = _parse_bc_value_scalar(value, V, x, t=t)
    return DirichletBC(V, val, boundary)


def build_scalar_bc_from_function(V, func, boundary="on_boundary"):
    """Build a Dirichlet BC from an existing Function."""
    return DirichletBC(V, func, boundary)


def update_scalar_bc(bc, value, V, x, t):
    """Re-interpolate an existing time-dependent BC at time t."""
    try:
        c = float(sp.sympify(value))
        bc.function_arg.assign(Constant(c))
    except Exception:
        expr = parse_expression(value, x, t=t)
        f = Function(V)
        f.interpolate(expr)
        bc.function_arg = f


# =============================================================================
# Grid sampling via Function.at()
# NOTE: VertexOnlyMesh is NOT used here because its dat.data_ro is in internal
#       mesh-cell ordering, NOT input-coordinate ordering.  Function.at(coords)
#       is the correct API for ordered point evaluation.
# =============================================================================

def _make_grid_coords(bbox: List[float], nx: int, ny: int):
    """Build (ny×nx) uniform grid coordinates.

    Uses indexing='xy' to match DOLFINx oracle convention:
      result[i,j] = value at (x=x_lin[j], y=y_lin[i])
    """
    xmin, xmax, ymin, ymax = bbox
    x_lin = np.linspace(xmin, xmax, nx)
    y_lin = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return x_lin, y_lin, coords, (ny, nx)


def _make_grid_coords_3d(bbox: List[float], nx: int, ny: int, nz: int):
    """Build a uniform 3-D grid with shape (nz, ny, nx)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    x_lin = np.linspace(xmin, xmax, nx)
    y_lin = np.linspace(ymin, ymax, ny)
    z_lin = np.linspace(zmin, zmax, nz)
    zz, yy, xx = np.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    return x_lin, y_lin, z_lin, coords, (nz, ny, nx)


def _is_3d_grid(bbox: List[float], nz: Optional[int] = None) -> bool:
    return len(bbox) == 6 and nz is not None


def _at_scalar_safe(u_h: Function, coords: np.ndarray) -> np.ndarray:
    """Evaluate a scalar Function at coords, returning NaN for out-of-domain points.

    Uses ``dont_raise=True`` so Firedrake returns None instead of raising
    PointNotInDomainError for points outside non-convex / irregular meshes
    (e.g. l_shape, annulus, dumbbell …).

    Out-of-domain points are NaN so that error computation ignores them via
    the NaN-safe ``compute_rel_L2_grid``.
    """
    raw = u_h.at(coords, dont_raise=True)
    return np.array(
        [float(v) if v is not None else np.nan for v in raw],
        dtype=float,
    )


def _at_vector_mag_safe(u_h: Function, coords: np.ndarray) -> np.ndarray:
    """Evaluate ||u|| at coords, returning NaN for out-of-domain points."""
    raw = u_h.at(coords, dont_raise=True)
    result = np.empty(len(raw), dtype=float)
    for i, v in enumerate(raw):
        if v is None:
            result[i] = np.nan
        else:
            arr = np.asarray(v, dtype=float)
            result[i] = float(np.linalg.norm(arr)) if arr.ndim > 0 else float(abs(arr))
    return result


def _apply_domain_mask(
    u_fem_grid: Optional[np.ndarray],
    u_exact_grid: np.ndarray,
) -> np.ndarray:
    """将 FEM 采样的域内掩码应用到精确解网格。

    域外点在 FEM 采样结果中为 NaN（``_at_scalar_safe`` / ``_at_vector_mag_safe``
    的返回值）。将相同位置在精确解中也设为 NaN，使误差计算只覆盖域内点。

    对简单（矩形）域无域外点时为空操作，不影响现有结果。
    """
    if u_fem_grid is None or not np.any(np.isnan(u_fem_grid)):
        return u_exact_grid
    masked = u_exact_grid.copy()
    masked[np.isnan(u_fem_grid)] = np.nan
    return masked


def sample_scalar_on_grid(
    u_h: Function, bbox: List[float], nx: int, ny: int, nz: Optional[int] = None
) -> Tuple[np.ndarray, ...]:
    """Sample a scalar Function on a uniform grid.

    Uses ``Function.at(coords, dont_raise=True)`` so points that fall outside
    the mesh (common for non-rectangular domains like l_shape, annulus …) are
    filled with NaN instead of raising PointNotInDomainError.
    """
    if _is_3d_grid(bbox, nz):
        x_lin, y_lin, z_lin, coords, shape = _make_grid_coords_3d(bbox, nx, ny, int(nz))
        values = _at_scalar_safe(u_h, coords)
        return x_lin, y_lin, z_lin, values.reshape(shape)

    x_lin, y_lin, coords, shape = _make_grid_coords(bbox, nx, ny)
    values = _at_scalar_safe(u_h, coords)
    return x_lin, y_lin, values.reshape(shape)


def sample_vector_magnitude_on_grid(
    u_h: Function, bbox: List[float], nx: int, ny: int, nz: Optional[int] = None
) -> Tuple[np.ndarray, ...]:
    """Sample ||u|| of a vector Function on a uniform grid.

    Out-of-domain points are filled with NaN.
    """
    if _is_3d_grid(bbox, nz):
        x_lin, y_lin, z_lin, coords, shape = _make_grid_coords_3d(bbox, nx, ny, int(nz))
        magnitudes = _at_vector_mag_safe(u_h, coords)
        return x_lin, y_lin, z_lin, magnitudes.reshape(shape)

    x_lin, y_lin, coords, shape = _make_grid_coords(bbox, nx, ny)
    magnitudes = _at_vector_mag_safe(u_h, coords)
    return x_lin, y_lin, magnitudes.reshape(shape)


def _eval_exact_sym_on_grid(
    u_sym,
    spatial_coords: Tuple,
    grid_cfg: Dict[str, Any],
    t: float = None,
    t_sym=None,
) -> np.ndarray:
    """Evaluate a sympy scalar expression directly on a 2-D uniform grid.

    Uses numpy lambdify — no FEM projection, machine-precision accuracy.
    Returns shape (ny, nx) matching the sample_scalar_on_grid convention.
    """
    bbox = grid_cfg["bbox"]
    nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    expr = u_sym.subs(t_sym, t) if (t is not None and t_sym is not None) else u_sym
    u_func = sp.lambdify(spatial_coords, expr, modules="numpy")
    result = u_func(X, Y)

    if np.isscalar(result):
        result = np.full((ny, nx), float(result))
    return np.asarray(result, dtype=float)


def _eval_exact_vec_mag_on_grid(
    u_sym_vec: List,
    spatial_coords: Tuple,
    grid_cfg: Dict[str, Any],
    t: float = None,
    t_sym=None,
) -> np.ndarray:
    """Evaluate vector magnitude sqrt(sum(ui^2)) directly on a 2-D uniform grid.

    Returns shape (ny, nx) matching sample_vector_magnitude_on_grid convention.
    """
    bbox = grid_cfg["bbox"]
    nx, ny = int(grid_cfg["nx"]), int(grid_cfg["ny"])
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    mag_sq = np.zeros((ny, nx), dtype=float)
    for comp_sym in u_sym_vec:
        expr = comp_sym.subs(t_sym, t) if (t is not None and t_sym is not None) else comp_sym
        comp_func = sp.lambdify(spatial_coords, expr, modules="numpy")
        vals = comp_func(X, Y)
        if np.isscalar(vals):
            vals = np.full((ny, nx), float(vals))
        mag_sq += np.asarray(vals, dtype=float) ** 2
    return np.sqrt(mag_sq)


# =============================================================================
# Solver parameter helpers
# =============================================================================

def _scalar_solver_params(solver_params: Dict[str, Any]) -> Dict[str, Any]:
    """Build Firedrake solver_parameters dict from oracle_solver config."""
    return {
        "ksp_type": solver_params.get("ksp_type", "cg"),
        "pc_type":  solver_params.get("pc_type", "hypre"),
        "ksp_rtol": solver_params.get("rtol", 1e-10),
        "ksp_atol": solver_params.get("atol", 1e-12),
    }
