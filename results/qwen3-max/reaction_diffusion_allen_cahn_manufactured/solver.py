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
    epsilon = case_spec.get("epsilon", 0.01)
    newton_rtol = case_spec.get("newton_rtol", 1e-8)
    pc_type = case_spec.get("pc_type", "lu")
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution for manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = 0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Define boundary condition (from exact solution)
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    dofs_geo = fem.locate_dofs_geometrical(V, boundary_marker)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: 0.3 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs_geo)
    
    # Define trial and test functions
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Define reaction term R(u) = u^3 - u (Allen-Cahn type)
    R_u = u**3 - u
    
    # Compute source term f from the exact solution
    # For steady: -ε ∇²u + R(u) = f
    # So f = -ε ∇²u_exact + R(u_exact)
    laplacian_u_exact = -0.3 * 2 * (ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    R_u_exact = (0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))**3 - (0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    f_expr = -epsilon * laplacian_u_exact + R_u_exact
    
    # Define variational form (nonlinear residual)
    F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(R_u, v) * ufl.dx - ufl.inner(f_expr, v) * ufl.dx
    
    # Set up nonlinear problem
    problem = petsc.NonlinearProblem(F, u, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.max_it = 50
    
    # Configure linear solver within Newton
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(u)
    assert converged, "Newton solver did not converge"
    u.x.scatter_forward()
    
    # Evaluate on 70x70 grid
    nx, ny = 70, 70
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Use bounding box tree to evaluate
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
            "pc_type": pc_type,
            "rtol": newton_rtol
        }
    }