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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

# Firedrake re-exports UFL; we import everything from firedrake for convenience.
from firedrake import (
    UnitSquareMesh,
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
# Mesh creation
# =============================================================================

def create_mesh(domain_spec: Dict[str, Any], mesh_spec: Dict[str, Any]):
    """Create a Firedrake mesh from domain/mesh configuration."""
    domain_type = domain_spec["type"]
    resolution = mesh_spec["resolution"]
    cell_type = mesh_spec.get("cell_type", "triangle")

    if domain_type == "unit_square":
        quad = (cell_type == "quadrilateral")
        return UnitSquareMesh(resolution, resolution, quadrilateral=quad)

    raise ValueError(f"Unsupported domain type for Firedrake: {domain_type}")


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


def sample_scalar_on_grid(
    u_h: Function, bbox: List[float], nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a scalar Function on a uniform grid.

    Uses ``Function.at(coords)`` which is guaranteed to return values in the
    same order as the input coordinates, avoiding the VertexOnlyMesh internal
    reordering issue that corrupts grid-point values.
    """
    x_lin, y_lin, coords, shape = _make_grid_coords(bbox, nx, ny)
    values = np.array(u_h.at(coords))   # shape (ny*nx,), input-order guaranteed
    return x_lin, y_lin, values.reshape(shape)


def sample_vector_magnitude_on_grid(
    u_h: Function, bbox: List[float], nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ||u|| of a vector Function on a uniform grid."""
    x_lin, y_lin, coords, shape = _make_grid_coords(bbox, nx, ny)
    vals = np.array(u_h.at(coords))   # shape (ny*nx, gdim), input-order guaranteed
    if vals.ndim == 1:
        # scalar-valued vector space with dim=1
        magnitudes = np.abs(vals)
    else:
        magnitudes = np.linalg.norm(vals, axis=1)
    return x_lin, y_lin, magnitudes.reshape(shape)


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
