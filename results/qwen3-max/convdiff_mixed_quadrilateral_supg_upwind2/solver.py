import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters from case_spec or defaults
    stabilization = case_spec.get("stabilization", "supg")
    upwind_parameter = case_spec.get("upwind_parameter", 1.0)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    ksp_type = case_spec.get("ksp_type", "gmres")
    pc_type = case_spec.get("pc_type", "ilu")
    rtol = case_spec.get("rtol", 1e-8)
    
    # Problem parameters
    eps = 0.005
    beta = [18.0, 6.0]
    
    # Create quadrilateral mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Exact solution for boundary conditions
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Boundary condition
    u_D = fem.Function(V)
    u_D.interpolate(exact_solution)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_geometrical(V, lambda x: np.full_like(x[0], True)))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Velocity vector
    beta_ufl = ufl.as_vector(beta)
    
    # Source term derived from exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_ufl, ufl.grad(u_exact))
    
    # Standard Galerkin form
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx
    
    # SUPG stabilization
    if stabilization == "supg":
        h = ufl.CellDiameter(domain)
        Pe = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl)) * h / (2.0 * eps)
        tau = ufl.conditional(
            ufl.gt(Pe, 1.0),
            h / (2.0 * ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe),
            h * h / (12.0 * eps)
        )
        tau = upwind_parameter * tau
        r = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u)) - f
        a += tau * ufl.dot(beta_ufl, ufl.grad(v)) * r * ufl.dx
        L += tau * ufl.dot(beta_ufl, ufl.grad(v)) * f * ufl.dx
    
    # Solve linear problem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 1000
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
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }