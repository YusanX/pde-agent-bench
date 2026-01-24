import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    mesh_resolution = case_spec.get("mesh_resolution", 60)
    element_degree = case_spec.get("element_degree", 2)
    epsilon = case_spec.get("epsilon", 1.0)
    newton_rtol = case_spec.get("newton_rtol", 1e-8)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution for boundary conditions and initial condition
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = 0.2 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Create boundary condition
    def boundary_marker(x):
        return np.full_like(x[0], True, dtype=bool)
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define trial and test functions
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Set initial guess (for nonlinear solve)
    u.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    
    # Define reaction term R(u) = u*(1-u) for logistic reaction
    R_u = u * (1 - u)
    
    # Compute source term f from manufactured solution
    # For steady: -ε ∇²u + R(u) = f
    # So f = -ε ∇²u_exact + R(u_exact)
    u_exact = u_exact_expr
    laplacian_u_exact = -2.0 * ufl.pi**2 * 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    R_u_exact = u_exact * (1 - u_exact)
    f_expr = -epsilon * laplacian_u_exact + R_u_exact
    
    # Define the residual form
    F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(R_u, v) * ufl.dx - ufl.inner(f_expr, v) * ufl.dx
    
    # Set up nonlinear problem
    problem = petsc.NonlinearProblem(F, u, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.atol = 1e-10
    solver.max_it = 50
    
    # Configure linear solver within Newton
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(u)
    assert converged, "Newton solver did not converge"
    u.x.scatter_forward()
    
    # Create evaluation grid (60x60)
    nx, ny = 60, 60
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
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": newton_rtol
        }
    }