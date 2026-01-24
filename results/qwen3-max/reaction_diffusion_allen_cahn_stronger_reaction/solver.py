import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters
    epsilon = case_spec.get("epsilon", 0.01)
    reaction_lambda = case_spec.get("reaction_lambda", 100.0)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    element_degree = 2
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions and initial condition
    def u_exact(x):
        return 0.15 + 0.12 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Boundary condition
    u_D = fem.Function(V)
    u_D.interpolate(u_exact)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution in UFL
    u_ex_ufl = 0.15 + 0.12 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Reaction term R(u) = lambda * (u^3 - u) for Allen-Cahn
    def R(u):
        return reaction_lambda * (u**3 - u)
    
    # Source term f = -epsilon * Laplacian(u_exact) + R(u_exact)
    grad_u = ufl.grad(u_ex_ufl)
    laplacian_u = ufl.div(grad_u)
    f = -epsilon * laplacian_u + R(u_ex_ufl)
    
    # Nonlinear problem: -epsilon * Laplacian(u) + R(u) = f
    uh = fem.Function(V)
    uh.interpolate(u_exact)  # Good initial guess
    
    F = epsilon * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + ufl.inner(R(uh), v) * ufl.dx - ufl.inner(f, v) * ufl.dx
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, uh, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    # Configure Krylov solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    
    n, converged = solver.solve(uh)
    assert converged, "Newton solver did not converge"
    uh.x.scatter_forward()
    
    # Evaluate on 75x75 grid
    nx, ny = 75, 75
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution at points
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
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