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
    epsilon = case_spec.get("epsilon", 1.0)
    newton_rtol = case_spec.get("newton_rtol", 1e-8)
    pc_type = case_spec.get("pc_type", "lu")
    max_it = case_spec.get("max_it", 20)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Boundary condition (u = 0 on boundary)
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Source term f = sin(6*pi*x)*sin(5*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    
    # Reaction term R(u) = u^3 (cubic reaction)
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Weak form for steady problem: -ε ∇²u + u^3 = f
    F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u**3 * v * ufl.dx - f * v * ufl.dx
    
    # Nonlinear problem setup
    problem = petsc.NonlinearProblem(F, u, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.max_it = max_it
    
    # Configure Krylov solver for linearized problems
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
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    
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
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": newton_rtol
        }
    }