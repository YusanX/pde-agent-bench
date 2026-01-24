import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec with defaults
    mesh_resolution = case_spec.get("mesh_resolution", 32)
    degree_u = case_spec.get("degree_u", 2)
    degree_p = case_spec.get("degree_p", 1)
    newton_rtol = case_spec.get("newton_rtol", 1e-6)
    newton_max_it = case_spec.get("newton_max_it", 20)
    
    # Problem parameters
    nu = 0.12
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Define function spaces (Taylor-Hood elements)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Define trial and test functions
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    # Define boundary conditions
    # Channel flow: inflow on left (x=0), outflow on right (x=1), no-slip on top/bottom
    def inflow_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def outflow_boundary(x):
        return np.isclose(x[0], 1.0)
    
    def walls_boundary(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    
    # Inflow velocity profile (parabolic)
    def inflow_velocity(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = 4.0 * x[1] * (1.0 - x[1])  # parabolic profile
        return values
    
    # Create boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Inflow BC
    inflow_facets = mesh.locate_entities_boundary(domain, fdim, inflow_boundary)
    u_inflow = fem.Function(V)
    u_inflow.interpolate(inflow_velocity)
    dofs_inflow = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
    bc_inflow = fem.dirichletbc(u_inflow, dofs_inflow, W.sub(0))
    
    # Walls BC (no-slip)
    walls_facets = mesh.locate_entities_boundary(domain, fdim, walls_boundary)
    u_walls = fem.Function(V)
    u_walls.x.array[:] = 0.0
    dofs_walls = fem.locate_dofs_topological((W.sub(0), V), fdim, walls_facets)
    bc_walls = fem.dirichletbc(u_walls, dofs_walls, W.sub(0))
    
    bcs = [bc_inflow, bc_walls]
    
    # Source term
    f = ufl.as_vector((0.0, 0.0))
    
    # Weak form (steady Navier-Stokes)
    # Using the standard form: (u·∇)u
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Set initial guess (Stokes solution would be better, but zero works for this case)
    w.x.array[:] = 0.0
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.max_it = newton_max_it
    solver.error_on_nonconvergence = False
    
    # Configure linear solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(w)
    w.x.scatter_forward()
    
    # Extract velocity solution
    u_sol = w.sub(0).collapse()
    
    # Create evaluation grid (50x50)
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute velocity magnitude
    u_magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_magnitude.reshape((nx, ny))
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": newton_rtol
        }
    }