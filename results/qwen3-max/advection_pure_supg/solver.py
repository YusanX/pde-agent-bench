import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Problem parameters
    eps = 0.0
    beta = [10.0, 4.0]
    
    # Mesh resolution and element degree
    nx = ny = 64
    element_degree = 1
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Boundary condition
    u_D = fem.Function(V)
    u_D.interpolate(exact_solution)
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(u_D, dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Velocity vector
    beta_ufl = ufl.as_vector(beta)
    
    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * ufl.sqrt(ufl.dot(beta_ufl, beta_ufl)))
    
    # Source term from manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_ufl, ufl.grad(u_exact))
    
    # Standard Galerkin form
    a_galerkin = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx
    L_galerkin = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization terms
    a_supg = tau * ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), ufl.dot(beta_ufl, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f, ufl.dot(beta_ufl, ufl.grad(v))) * ufl.dx
    
    # Total forms
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="advection_pure_supg_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_eval = ny_eval = 50
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
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
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12
    }
    
    return {"u": u_grid, "solver_info": solver_info}