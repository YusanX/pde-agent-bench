import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    eps = 0.01
    beta = [0.0, 15.0]
    
    # Mesh resolution and element degree
    mesh_resolution = 64
    element_degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f derived from exact solution
    grad_u = ufl.grad(u_exact)
    laplacian_u = ufl.div(grad_u)
    f = -eps * laplacian_u + beta[0] * grad_u[0] + beta[1] * grad_u[1]
    
    # Boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.full_like(x[0], True))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_standard = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_standard += (beta[0] * ufl.Dx(u, 0) + beta[1] * ufl.Dx(u, 1)) * v * ufl.dx
    L_standard = f * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    Pe = ufl.sqrt(beta[0]**2 + beta[1]**2) * h / (2.0 * eps)
    tau = ufl.conditional(ufl.gt(Pe, 1.0), h / (2.0 * ufl.sqrt(beta[0]**2 + beta[1]**2)) * (1.0 - 1.0/Pe), 0.0)
    
    # Residual of the strong form
    residual = -eps * ufl.div(ufl.grad(u)) + beta[0] * ufl.Dx(u, 0) + beta[1] * ufl.Dx(u, 1) - f
    
    # SUPG terms
    a_supg = tau * (beta[0] * ufl.Dx(u, 0) + beta[1] * ufl.Dx(u, 1)) * (beta[0] * ufl.Dx(v, 0) + beta[1] * ufl.Dx(v, 1)) * ufl.dx
    L_supg = tau * f * (beta[0] * ufl.Dx(v, 0) + beta[1] * ufl.Dx(v, 1)) * ufl.dx
    
    # Total forms
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-8
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree and evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8
        }
    }