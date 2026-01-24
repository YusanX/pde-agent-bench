import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    degree_u = case_spec.get("degree_u", 2)
    degree_p = case_spec.get("degree_p", 1)
    newton_rtol = case_spec.get("newton_rtol", 1e-8)
    newton_max_it = case_spec.get("newton_max_it", 30)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
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
    u_exact_expr = ufl.as_vector([x[0]**2*(1-x[0])**2*(1-2*x[1]), -2*x[0]*(1-x[0])*(1-2*x[0])*x[1]*(1-x[1])])
    p_exact_expr = x[0] + x[1]
    
    # Viscosity
    nu = 0.22
    
    # Source term f derived from exact solution
    # Compute -nu*div(grad(u)) + (u·grad)u + grad(p)
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2*nu*eps(u) - p*ufl.Identity(len(u))
    
    # Compute the source term by substituting exact solution into NS equation
    u_exact = u_exact_expr
    p_exact = p_exact_expr
    
    # Laplacian term
    laplacian_term = -nu * ufl.div(ufl.grad(u_exact))
    
    # Convection term
    convection_term = ufl.dot(u_exact, ufl.nabla_grad(u_exact))
    
    # Pressure gradient
    pressure_grad = ufl.grad(p_exact)
    
    f_expr = laplacian_term + convection_term + pressure_grad
    
    # Weak form residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.dot(u, ufl.nabla_grad(u)), v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Boundary conditions (Dirichlet on entire boundary)
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.vstack((
        x[0]**2 * (1-x[0])**2 * (1-2*x[1]),
        -2 * x[0] * (1-x[0]) * (1-2*x[0]) * x[1] * (1-x[1])
    )))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_D, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess (use exact solution for better convergence)
    w.sub(0).interpolate(lambda x: np.vstack((
        x[0]**2 * (1-x[0])**2 * (1-2*x[1]),
        -2 * x[0] * (1-x[0]) * (1-2*x[0]) * x[1] * (1-x[1])
    )))
    w.sub(1).interpolate(lambda x: x[0] + x[1])
    
    # Nonlinear problem setup
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.max_it = newton_max_it
    solver.error_on_nonconvergence = True
    
    # Linear solver settings
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(w)
    assert converged, "Newton solver did not converge"
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
    
    # Evaluate velocity magnitude on grid
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
            "rtol": newton_rtol
        }
    }