"""Base classes and utilities for PDE solvers."""
import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc
import ufl


def create_mesh(domain_type, resolution, cell_type_str="triangle"):
    """Create mesh based on domain specification."""
    if domain_type == "unit_square":
        if cell_type_str == "triangle":
            cell_type = CellType.triangle
        elif cell_type_str == "quadrilateral":
            cell_type = CellType.quadrilateral
        else:
            raise ValueError(f"Unknown cell type: {cell_type_str}")
        
        return mesh.create_unit_square(
            MPI.COMM_WORLD, resolution, resolution, cell_type
        )
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")


def create_function_space(msh, family, degree):
    """Create finite element function space."""
    element = (family, degree)
    return fem.functionspace(msh, element)


def parse_expression(expr_str, x, y, t=None):
    """Parse string expression to UFL expression."""
    import sympy as sp
    
    # Create sympy symbols
    sx, sy = sp.symbols('x y', real=True)
    st = sp.Symbol('t', real=True) if t is not None else None
    
    # Parse expression using sympy
    if t is not None:
        # Include t in the expression
        expr_sympy = sp.sympify(expr_str, locals={'x': sx, 'y': sy, 't': st})
    else:
        expr_sympy = sp.sympify(expr_str, locals={'x': sx, 'y': sy})
    
    # Now substitute sympy variables with UFL spatial coordinates
    # We do this by replacing sympy symbols with appropriate UFL expressions
    subs_dict = {
        sx: x[0],
        sy: x[1],
    }
    if st is not None:
        subs_dict[st] = t
    
    # Use sympy's lambdify to create a numerical function, but we'll build UFL directly
    # Convert sympy expression to UFL by replacing functions
    def sympy_to_ufl(expr):
        if expr.is_Number:
            return float(expr)
        elif expr.is_Symbol:
            if expr == sx:
                return x[0]
            elif expr == sy:
                return x[1]
            elif expr == st:
                return t
            else:
                return expr
        elif expr.func == sp.sin:
            return ufl.sin(sympy_to_ufl(expr.args[0]))
        elif expr.func == sp.cos:
            return ufl.cos(sympy_to_ufl(expr.args[0]))
        elif expr.func == sp.exp:
            return ufl.exp(sympy_to_ufl(expr.args[0]))
        elif expr.func == sp.sqrt:
            return ufl.sqrt(sympy_to_ufl(expr.args[0]))
        elif expr.func == sp.Add:
            return sum(sympy_to_ufl(arg) for arg in expr.args)
        elif expr.func == sp.Mul:
            result = sympy_to_ufl(expr.args[0])
            for arg in expr.args[1:]:
                result = result * sympy_to_ufl(arg)
            return result
        elif expr.func == sp.Pow:
            base = sympy_to_ufl(expr.args[0])
            exp_val = sympy_to_ufl(expr.args[1])
            return base ** exp_val
        elif expr == sp.pi:
            return np.pi
        else:
            # For other functions, try to evaluate
            raise NotImplementedError(f"Unsupported sympy function: {expr.func}")
    
    return sympy_to_ufl(expr_sympy)


def create_kappa_field(msh, kappa_spec):
    """Create kappa coefficient field from specification."""
    if kappa_spec['type'] == 'constant':
        from dolfinx import default_scalar_type
        return fem.Constant(msh, default_scalar_type(kappa_spec['value']))
    elif kappa_spec['type'] == 'expr':
        # TODO: implement expression-based kappa
        raise NotImplementedError("Expression-based kappa not yet implemented")
    else:
        raise ValueError(f"Unknown kappa type: {kappa_spec['type']}")


def interpolate_ufl_expression(func, expr):
    """
    Interpolate a UFL expression into a Function.
    
    Args:
        func: fem.Function to interpolate into
        expr: UFL expression to evaluate
    """
    # Create Expression with interpolation points from the element
    V = func.function_space
    
    # Get interpolation points for the element (it's a property, not a method)
    interp_points = V.element.interpolation_points
    
    # Create dolfinx Expression from UFL expression
    expr_compiled = fem.Expression(expr, interp_points)
    
    # Interpolate into the function
    func.interpolate(expr_compiled)


def sample_on_grid(u_fem, bbox, nx, ny):
    """
    Sample FE function on regular grid.
    
    Returns:
        x_grid: (nx,) array
        y_grid: (ny,) array
        u_grid: (ny, nx) array
    """
    xmin, xmax, ymin, ymax = bbox
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    
    u_grid = np.zeros((ny, nx))
    
    for j, yval in enumerate(y_grid):
        for i, xval in enumerate(x_grid):
            point = np.array([[xval, yval, 0.0]])
            try:
                u_grid[j, i] = u_fem.eval(point, [0])[0]
            except:
                # Point outside domain or other eval error
                u_grid[j, i] = np.nan
    
    return x_grid, y_grid, u_grid

