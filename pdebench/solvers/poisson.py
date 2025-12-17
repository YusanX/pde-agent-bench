"""Poisson equation solver: -div(kappa * grad u) = f."""
import numpy as np
from dolfinx import fem, default_scalar_type, mesh as dfx_mesh
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc


def derive_source_from_manufactured(msh, V, u_exact_expr, kappa):
    """
    Derive source term f from manufactured solution u using:
    f = -div(kappa * grad(u_exact))
    """
    x = ufl.SpatialCoordinate(msh)
    
    # Parse u_exact
    from .base import parse_expression
    u_exact_ufl = parse_expression(u_exact_expr, x, x)
    
    # Compute f = -div(kappa * grad(u_exact))
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    return f_ufl, u_exact_ufl


def setup_poisson_problem(msh, V, case_spec):
    """
    Set up Poisson problem: assemble matrix and RHS.
    
    Returns:
        A, b, bc_dofs, u_exact_func (or None if no manufactured solution)
    """
    # Get kappa
    from .base import create_kappa_field
    kappa_spec = case_spec['pde'].get('coefficients', {}).get('kappa', {'type': 'constant', 'value': 1.0})
    kappa = create_kappa_field(msh, kappa_spec)
    
    # Get source term
    manufactured = case_spec['pde'].get('manufactured_solution', {})
    
    x = ufl.SpatialCoordinate(msh)
    u_exact_func = None
    
    if 'u' in manufactured:
        # Derive f from manufactured solution
        f_ufl, u_exact_ufl = derive_source_from_manufactured(
            msh, V, manufactured['u'], kappa
        )
        # Create exact solution function
        u_exact_func = fem.Function(V)
        from .base import parse_expression, interpolate_ufl_expression
        u_exact_expr_eval = parse_expression(manufactured['u'], x, x)
        interpolate_ufl_expression(u_exact_func, u_exact_expr_eval)
    else:
        # Use provided source term
        source_spec = case_spec['pde'].get('source_term', {})
        if 'f' in source_spec:
            from .base import parse_expression
            f_ufl = parse_expression(source_spec['f'], x, x)
        else:
            f_ufl = fem.Constant(msh, default_scalar_type(0.0))
    
    # Weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx
    
    # Boundary conditions
    bc_spec = case_spec['bc']['dirichlet']
    
    # Find boundary DOFs
    def boundary(x):
        return np.ones(x.shape[1], dtype=bool)  # All boundaries
    
    boundary_facets = dfx_mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
    boundary_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
    
    # Get BC value
    bc_value_str = bc_spec['value']
    if bc_value_str == 'u' and u_exact_func is not None:
        # Use manufactured solution on boundary
        bc_func = u_exact_func
    else:
        # Parse BC expression
        from .base import parse_expression, interpolate_ufl_expression
        bc_expr = parse_expression(bc_value_str, x, x)
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


def solve_poisson(A, b, bcs, solver_func):
    """
    Solve Poisson system using provided solver function.
    
    Args:
        A: PETSc Mat
        b: PETSc Vec
        bcs: list of Dirichlet BCs
        solver_func: function(A, b) -> (x, info)
    
    Returns:
        u_vec, info
    """
    u_vec, info = solver_func(A, b)
    
    # Apply BC (ensure BC values are set correctly)
    for bc in bcs:
        fem_petsc.set_bc(u_vec, [bc])
    
    return u_vec, info

