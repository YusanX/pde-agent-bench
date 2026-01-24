import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    epsilon = 0.01
    beta = [12.0, 0.0]
    
    # Mesh resolution and element degree
    mesh_resolution = 64
    element_degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(3*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # Boundary conditions
    def boundary_marker(x):
        return np.full_like(x[0], True)  # All boundaries
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(3*x[0]) * np.sin(np.pi*x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Source term f derived from exact solution
    grad_u = ufl.grad(u_exact_expr)
    laplacian_u = ufl.div(grad_u)
    f_expr = -epsilon * laplacian_u + beta[0] * ufl.Dx(u_exact_expr, 0) + beta[1] * ufl.Dx(u_exact_expr, 1)
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin form
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.as_vector(beta), ufl.grad(u)) * v * ufl.dx
    L = f_func * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    Pe = ufl.sqrt(ufl.dot(ufl.as_vector(beta), ufl.as_vector(beta))) * h / (2 * epsilon)
    tau = ufl.conditional(ufl.gt(Pe, 1.0), h / (2 * ufl.sqrt(ufl.dot(ufl.as_vector(beta), ufl.as_vector(beta)))) * (1 - ufl.exp(-2 * Pe)) / (1 + ufl.exp(-2 * Pe)), 0.0)
    
    # Residual for SUPG
    residual = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(ufl.as_vector(beta), ufl.grad(u)) - f_func
    a_supg = tau * ufl.dot(ufl.as_vector(beta), ufl.grad(v)) * residual * ufl.dx
    L_supg = tau * ufl.dot(ufl.as_vector(beta), ufl.grad(v)) * f_func * ufl.dx
    
    # Add SUPG terms
    a_total = a + a_supg
    L_total = L + L_supg
    
    # Solver options
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 1000
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree and evaluate
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }