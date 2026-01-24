import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec with defaults
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    degree_u = case_spec.get("degree_u", 2)
    degree_p = case_spec.get("degree_p", 1)
    newton_max_it = case_spec.get("newton_max_it", 20)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Define function spaces (Taylor-Hood elements)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Define trial and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Parameters
    nu = ScalarType(0.18)
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.as_vector([
        6*(1 - ufl.tanh(6*(x[1]-0.5))**2)*ufl.sin(ufl.pi*x[0]),
        -ufl.pi*ufl.tanh(6*(x[1]-0.5))*ufl.cos(ufl.pi*x[0])
    ])
    p_exact_expr = ufl.cos(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1])
    
    # Compute source term f from the exact solution
    # f = -nu*div(grad(u)) + (u·grad)u + grad(p)
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2*nu*epsilon(u) - p*ufl.Identity(len(u))
    
    # Compute the residual of the exact solution
    # R = -div(sigma(u_exact, p_exact)) + dot(grad(u_exact), u_exact)
    R = -ufl.div(sigma(u_exact_expr, p_exact_expr)) + ufl.dot(ufl.grad(u_exact_expr), u_exact_expr)
    f_expr = R
    
    # Boundary conditions (Dirichlet on all boundaries)
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.vstack((
        6*(1 - np.tanh(6*(x[1]-0.5))**2)*np.sin(np.pi*x[0]),
        -np.pi*np.tanh(6*(x[1]-0.5))*np.cos(np.pi*x[0])
    )))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_D, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Weak form residual
    F = (
        ufl.inner(sigma(u, p), epsilon(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Initial guess: interpolate exact solution
    w.sub(0).interpolate(lambda x: np.vstack((
        6*(1 - np.tanh(6*(x[1]-0.5))**2)*np.sin(np.pi*x[0]),
        -np.pi*np.tanh(6*(x[1]-0.5))*np.cos(np.pi*x[0])
    )))
    w.sub(1).interpolate(lambda x: np.cos(np.pi*x[0])*np.cos(np.pi*x[1]))
    
    # Nonlinear problem setup
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = newton_max_it
    
    # Configure linear solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n} iterations"
    w.x.scatter_forward()
    
    # Extract velocity component
    u_sol = w.sub(0).collapse()
    
    # Create evaluation grid (50x50)
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate velocity magnitude on the grid
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute velocity magnitude
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": {"velocity": degree_u, "pressure": degree_p},
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8
        }
    }