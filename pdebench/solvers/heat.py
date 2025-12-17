"""Heat equation solver: u_t - div(kappa * grad u) = f."""
import numpy as np
from dolfinx import fem, default_scalar_type, mesh as dfx_mesh
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc


def derive_source_from_manufactured(msh, V, u_exact_expr, kappa, t_val):
    """
    Derive source term f from manufactured solution u at time t:
    f = u_t - div(kappa * grad(u_exact))
    
    For simplicity, we compute numerical time derivative if needed.
    """
    import sympy as sp
    
    x = ufl.SpatialCoordinate(msh)
    
    # Parse u_exact with time
    sx, sy, st = sp.symbols('x y t', real=True)
    u_sympy = sp.sympify(u_exact_expr)
    
    # Compute u_t symbolically
    u_t_sympy = sp.diff(u_sympy, st)
    
    # Convert to UFL
    from .base import parse_expression
    u_t_ufl = parse_expression(str(u_t_sympy), x, x, t=t_val)
    u_exact_ufl = parse_expression(u_exact_expr, x, x, t=t_val)
    
    # f = u_t - div(kappa * grad(u))
    f_ufl = u_t_ufl - ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    return f_ufl, u_exact_ufl


def setup_heat_problem(msh, V, case_spec, u_prev, dt, t_current):
    """
    Set up backward Euler heat equation system at time t_current:
    (u - u_prev)/dt - div(kappa * grad u) = f
    
    Weak form:
    (u * v + dt * kappa * grad(u) * grad(v)) dx = (u_prev * v + dt * f * v) dx
    
    Returns:
        A, b, bcs, u_exact_func (or None)
    """
    from .base import create_kappa_field, parse_expression
    
    kappa_spec = case_spec['pde'].get('coefficients', {}).get('kappa', {'type': 'constant', 'value': 1.0})
    kappa = create_kappa_field(msh, kappa_spec)
    
    manufactured = case_spec['pde'].get('manufactured_solution', {})
    
    x = ufl.SpatialCoordinate(msh)
    u_exact_func = None
    
    if 'u' in manufactured:
        # Derive f from manufactured solution
        f_ufl, u_exact_ufl = derive_source_from_manufactured(
            msh, V, manufactured['u'], kappa, t_current
        )
        # Create exact solution function at t_current
        u_exact_func = fem.Function(V)
        from .base import interpolate_ufl_expression
        u_exact_expr = parse_expression(manufactured['u'], x, x, t=t_current)
        interpolate_ufl_expression(u_exact_func, u_exact_expr)
    else:
        # Use provided source term
        source_spec = case_spec['pde'].get('source_term', {})
        if 'f' in source_spec:
            f_ufl = parse_expression(source_spec['f'], x, x, t=t_current)
        else:
            f_ufl = fem.Constant(msh, default_scalar_type(0.0))
    
    # Weak form (backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(msh, default_scalar_type(dt))
    
    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_prev * v + dt_const * f_ufl * v) * ufl.dx
    
    # Boundary conditions at t_current
    bc_spec = case_spec['bc']['dirichlet']
    
    def boundary(x):
        return np.ones(x.shape[1], dtype=bool)
    
    boundary_facets = dfx_mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
    boundary_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
    
    bc_value_str = bc_spec['value']
    if bc_value_str == 'u' and u_exact_func is not None:
        bc_func = u_exact_func
    else:
        from .base import interpolate_ufl_expression
        bc_expr = parse_expression(bc_value_str, x, x, t=t_current)
        bc_func = fem.Function(V)
        interpolate_ufl_expression(bc_func, bc_expr)
    
    bc = fem.dirichletbc(bc_func, boundary_dofs)
    
    # Assemble
    A = fem_petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    
    b = fem_petsc.assemble_vector(fem.form(L))
    fem_petsc.apply_lifting(b, [fem.form(a)], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])
    
    return A, b, [bc], u_exact_func


def solve_heat_timestep(A, b, bcs, solver_func):
    """Solve one heat equation timestep."""
    u_vec, info = solver_func(A, b)
    
    for bc in bcs:
        fem_petsc.set_bc(u_vec, [bc])
    
    return u_vec, info

