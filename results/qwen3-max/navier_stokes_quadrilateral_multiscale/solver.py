import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = case_spec.get("mesh_resolution", 32)
    degree_u = case_spec.get("degree_u", 2)
    degree_p = case_spec.get("degree_p", 1)
    newton_max_it = case_spec.get("newton_max_it", 20)
    
    # Create quadrilateral mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function spaces (Taylor-Hood)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Trial and test functions
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]) + 
        ufl.pi * ufl.cos(4 * ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) - 
        (ufl.pi / 2) * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    ])
    p_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Viscosity
    nu = 0.1
    
    # Source term f derived from exact solution
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(domain.geometry.dim)
    
    # Compute f from exact solution
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]) + 
        ufl.pi * ufl.cos(4 * ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) - 
        (ufl.pi / 2) * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    ])
    p_exact = ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Compute residual of momentum equation with exact solution
    f_expr = (
        -ufl.div(sigma(u_exact, p_exact)) 
        + ufl.grad(u_exact) * u_exact
    )
    
    # Weak form
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Boundary conditions (Dirichlet on entire boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Interpolate exact velocity for BC
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) + 
        np.pi * np.cos(4 * np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) - 
        (np.pi / 2) * np.cos(2 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess: interpolate exact solution
    w.sub(0).interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) + 
        np.pi * np.cos(4 * np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) - 
        (np.pi / 2) * np.cos(2 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])
    ]))
    w.sub(1).interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    
    # Nonlinear solver setup
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = newton_max_it
    solver.relaxation_parameter = 1.0
    
    # Linear solver settings
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n} iterations"
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Create evaluation grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate velocity magnitude
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