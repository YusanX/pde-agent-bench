import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters
    reaction_rho = case_spec.get("reaction_rho", 100.0)
    epsilon = case_spec.get("epsilon", 0.01)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    element_degree = 2
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions and initial condition
    def u_exact(x):
        return 0.35 + 0.1 * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Exact solution as UFL expression
    x = ufl.SpatialCoordinate(domain)
    u_ex_ufl = 0.35 + 0.1 * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Reaction term R(u) = rho * u * (1 - u) for logistic
    R_u = reaction_rho * u_ex_ufl * (1 - u_ex_ufl)
    
    # Compute source term f from exact solution
    laplacian_u = ufl.div(ufl.grad(u_ex_ufl))
    f_expr = -epsilon * laplacian_u + R_u
    
    # Convert to dolfinx constants/functions
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Weak form for steady problem
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction_rho * u * (1 - u_ex_ufl) * v * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx + reaction_rho * u_ex_ufl * (1 - u_ex_ufl) * v * ufl.dx
    
    # Nonlinear problem setup
    uh = fem.Function(V)
    uh.interpolate(u_exact)
    
    F = epsilon * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + reaction_rho * uh * (1 - uh) * v * ufl.dx - ufl.inner(f_func, v) * ufl.dx
    J = ufl.derivative(F, uh)
    
    problem = petsc.NonlinearProblem(F, uh, bcs=[bc], J=J)
    solver = petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 20
    
    # Configure linear solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(uh)
    assert converged, "Newton solver did not converge"
    uh.x.scatter_forward()
    
    # Evaluate on 65x65 grid
    nx, ny = 65, 65
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Use bounding box tree for evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    for i in range(points.shape[1]):
        point = points[:, i]
        candidate_cells = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, candidate_cells, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        eval_points = np.array(points_on_proc).T
        eval_cells = np.array(cells, dtype=np.int32)
        u_eval = uh.eval(eval_points, eval_cells)
        u_values[:] = u_eval.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }