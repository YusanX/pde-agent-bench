import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec with defaults
    mesh_resolution = case_spec.get("mesh_resolution", 60)
    element_degree = case_spec.get("element_degree", 2)
    epsilon = case_spec.get("epsilon", 0.01)
    nonlinear_method = case_spec.get("nonlinear_method", "Newton")
    newton_rtol = case_spec.get("newton_rtol", 1e-8)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary condition (u = 0 on boundary)
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Source term: f = 5*exp(-180*((x-0.35)**2 + (y-0.55)**2))
    x = ufl.SpatialCoordinate(domain)
    f_expr = 5 * ufl.exp(-180 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Reaction term: R(u) = u^3 - u (Allen-Cahn type)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_sol = fem.Function(V)
    
    # For steady state: -ε ∇²u + R(u) = f
    # Weak form: ε ∇u·∇v + (u^3 - u)v = f v
    F = epsilon * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx + \
        (u_sol**3 - u_sol) * v * ufl.dx - f_func * v * ufl.dx
    
    # Nonlinear solver setup
    problem = petsc.NonlinearProblem(F, u_sol, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.max_it = 50
    solver.error_on_nonconvergence = False
    
    # Configure Krylov solver for linearized problems
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    
    # Solve
    n, converged = solver.solve(u_sol)
    
    # Prepare evaluation grid (60x60)
    nx, ny = 60, 60
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    for i in range(points.shape[1]):
        point = points[:, i]
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:] = u_eval.flatten()
    
    # Reshape to 2D grid
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": newton_rtol
        }
    }