import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters from case_spec or defaults
    stabilization = case_spec.get("stabilization", "supg")
    upwind_parameter = case_spec.get("upwind_parameter", 0.5)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    ksp_type = case_spec.get("ksp_type", "gmres")
    pc_type = case_spec.get("pc_type", "ilu")
    rtol = case_spec.get("rtol", 1e-8)
    
    # Problem parameters
    eps = 0.01
    beta = [14.0, 6.0]
    domain = mesh.create_unit_square(MPI.COMM_SELF, mesh_resolution, mesh_resolution, 
                                     cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary conditions (entire boundary)
    u_D = fem.Function(V)
    u_D.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, dofs)
    
    # Source term derived from exact solution
    grad_u_exact = ufl.grad(u_exact_expr)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_expr))
    f_expr = -eps * laplacian_u_exact + beta[0] * grad_u_exact[0] + beta[1] * grad_u_exact[1]
    f = fem.Constant(domain, PETSc.ScalarType(1.0))  # Placeholder, will use Expression
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin form
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        (beta[0] * ufl.Dx(u, 0) + beta[1] * ufl.Dx(u, 1)) * v * ufl.dx
    L = f_expr * v * ufl.dx
    
    # SUPG stabilization
    if stabilization == "supg":
        h = ufl.CellDiameter(domain)
        Pe = ufl.sqrt(beta[0]**2 + beta[1]**2) * h / (2.0 * eps)
        tau = ufl.conditional(ufl.gt(Pe, 1.0), 
                             upwind_parameter * h / (2.0 * ufl.sqrt(beta[0]**2 + beta[1]**2)) * (1.0 - 1.0/Pe),
                             0.0)
        beta_vec = ufl.as_vector(beta)
        residual = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - f_expr
        a += tau * (ufl.dot(beta_vec, ufl.grad(v))) * residual * ufl.dx
        L += tau * (ufl.dot(beta_vec, ufl.grad(v))) * f_expr * ufl.dx
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_error_if_not_converged": True
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Create evaluation grid
    nx, ny = 50, 50
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": 1,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }