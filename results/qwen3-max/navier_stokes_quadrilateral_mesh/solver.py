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
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_resolution, mesh_resolution], 
                                   cell_type=mesh.CellType.quadrilateral)
    
    # Function spaces (Taylor-Hood)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Exact solution for velocity
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact_expr = ufl.Constant(domain, ScalarType(0.0))
    
    # Source term f = -nu*Δu + (u·∇)u + ∇p
    nu = 0.1
    grad_u = ufl.grad(u_exact_expr)
    laplacian_u = ufl.div(grad_u)
    conv_term = ufl.dot(grad_u, u_exact_expr)
    f_expr = -nu * laplacian_u + conv_term + ufl.grad(p_exact_expr)
    
    # Boundary condition (Dirichlet everywhere)
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.vstack((
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    )))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_D, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Weak form residual
    f = fem.Expression(f_expr, W.element.interpolation_points)
    f_func = fem.Function(W.sub(0).collapse()[0])
    f_func.interpolate(f)
    
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - ufl.inner(f_func, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Initial guess from exact solution
    w.sub(0).interpolate(u_D)
    w.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    
    # Nonlinear solver setup
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = newton_max_it
    solver.relaxation_parameter = 1.0
    
    # Linear solver configuration
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
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_eval = np.linspace(0.0, 1.0, nx)
    y_eval = np.linspace(0.0, 1.0, ny)
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
    
    u_magnitude = np.linalg.norm(u_values, axis=1).reshape((nx, ny))
    
    return {
        "u": u_magnitude,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8
        }
    }