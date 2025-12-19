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
    elif kappa_spec['type'] == 'piecewise_x':
        # Create a piecewise constant function based on x coordinate
        # kappa = left if x < x_split else right
        
        left_val = kappa_spec['left']
        right_val = kappa_spec['right']
        x_split = kappa_spec['x_split']
        
        # Use DG0 (piecewise constant) for coefficient field
        # Use fem.functionspace (lowercase) as per project convention
        V_dg = fem.functionspace(msh, ("DG", 0))
        kappa_func = fem.Function(V_dg)
        
        # Define piecewise function: left if x < x_split, else right
        def piecewise_expr(x):
            values = np.full(x.shape[1], right_val, dtype=np.float64)
            left_mask = x[0] < x_split
            values[left_mask] = left_val
            return values
            
        kappa_func.interpolate(piecewise_expr)
        return kappa_func
        
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
    Sample FE function on regular grid using correct cell lookup.
    
    This function properly identifies which cell each grid point belongs to
    before evaluating the FEM solution at that point.
    
    Returns:
        x_grid: (nx,) array
        y_grid: (ny,) array
        u_grid: (ny, nx) array
    """
    from dolfinx import geometry
    
    xmin, xmax, ymin, ymax = bbox
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    
    # Create meshgrid points (note: indexing='ij' for row-major ordering)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Flatten to (N, 3) array for dolfinx
    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    # points[:, 2] is already 0.0
    
    # Find cells for all points using bounding box tree
    msh = u_fem.function_space.mesh
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    
    # Evaluate at each point with its correct cell
    u_grid = np.full(points.shape[0], np.nan)
    points_on_proc, cells_on_proc, eval_map = [], [], []
    
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells_on_proc.append(colliding_cells.links(i)[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        u_values = u_fem.eval(np.array(points_on_proc), 
                              np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = u_values.flatten()
    
    # Reshape to (nx, ny) for output
    u_grid = u_grid.reshape(nx, ny)
    
    return x_grid, y_grid, u_grid

