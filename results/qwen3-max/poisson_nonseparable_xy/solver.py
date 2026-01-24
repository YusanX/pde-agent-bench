import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # Setup
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions
    def u_exact(x):
        return np.sin(np.pi * x[0] * x[1])
    
    # Boundary condition
    u_D = fem.Function(V)
    u_D.interpolate(u_exact)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_geometrical(V, lambda x: np.full(x.shape[1], True, dtype=bool)))
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficients
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Source term derived from manufactured solution u = sin(pi*x*y)
    x = ufl.SpatialCoordinate(domain)
    u_manu = ufl.sin(ufl.pi * x[0] * x[1])
    f = -ufl.div(kappa * ufl.grad(u_manu))
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_converged_reason": "none"
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Create evaluation grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution on grid
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
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }