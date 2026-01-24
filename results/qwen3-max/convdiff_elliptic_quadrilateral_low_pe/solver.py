import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Problem parameters
    eps = 0.25
    beta = [1.0, 0.5]
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions
    def u_exact(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact)
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.full_like(x[0], True))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution in UFL
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = -eps*div(grad(u)) + beta.grad(u)
    grad_u = ufl.grad(u_ex)
    laplacian_u = ufl.div(grad_u)
    conv_term = beta[0] * ufl.Dx(u_ex, 0) + beta[1] * ufl.Dx(u_ex, 1)
    f_expr = -eps * laplacian_u + conv_term
    
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Bilinear and linear forms
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        (beta[0] * ufl.Dx(u, 0) + beta[1] * ufl.Dx(u, 1)) * v * ufl.dx
    L = f_func * v * ufl.dx
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 1000
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree and find colliding cells
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Evaluate solution at points
    u_values = np.full(points.shape[1], np.nan)
    for i in range(points.shape[1]):
        if len(colliding_cells.links(i)) > 0:
            u_values[i] = u_sol.eval(points[:, i:i+1], colliding_cells.links(i)[0:1])
    
    u_grid = u_values.reshape((nx, ny))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }