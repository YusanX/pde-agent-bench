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
    pc_type = case_spec.get("pc_type", "lu")
    newton_max_it = case_spec.get("newton_max_it", 20)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution for manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = 0.25 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term f for the Allen-Cahn equation
    # The reaction term is R(u) = u^3 - u (standard Allen-Cahn)
    # So the PDE is: -ε ∇²u + u^3 - u = f
    # We need to compute f from the exact solution
    epsilon = 0.01  # Standard value for Allen-Cahn
    
    # Compute Laplacian of exact solution
    u_exact = u_exact_expr
    laplacian_u = ufl.div(ufl.grad(u_exact))
    
    # Reaction term R(u) = u^3 - u
    R_u = u_exact**3 - u_exact
    
    # Source term f = -ε ∇²u + R(u)
    f_expr = -epsilon * laplacian_u + R_u
    
    # Convert to dolfinx constants/functions
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Boundary condition from exact solution
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Create boundary condition
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # For nonlinear problem, we need to use a Function for the solution
    uh = fem.Function(V)
    
    # Nonlinear residual: F(u; v) = ε ∇u·∇v + (u^3 - u)v - fv
    F = epsilon * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + \
        (uh**3 - uh) * v * ufl.dx - f_func * v * ufl.dx
    
    # Set up nonlinear problem
    problem = petsc.NonlinearProblem(F, uh, bcs=[bc])
    solver = petsc.nls.newton.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.max_it = newton_max_it
    
    # Configure linear solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(uh)
    assert converged, f"Newton solver did not converge after {n} iterations"
    uh.x.scatter_forward()
    
    # Evaluate on 60x60 grid
    nx, ny = 60, 60
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    points[2] = np.zeros(nx * ny)
    
    # Create bounding box tree and evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full((nx * ny,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": pc_type,
            "rtol": 1e-8
        }
    }