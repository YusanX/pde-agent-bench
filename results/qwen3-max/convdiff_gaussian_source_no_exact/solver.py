import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    epsilon = 0.02
    beta = [8.0, 3.0]
    
    # Agent-selectable parameters
    mesh_resolution = 64
    element_degree = 1
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-6
    stabilization = True  # Enable SUPG
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary condition (homogeneous Dirichlet)
    u_D = fem.Function(V)
    u_D.x.array[:] = 0.0
    
    # Locate boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, dofs)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-250 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Convection vector
    beta_ufl = ufl.as_vector(beta)
    
    # Standard Galerkin form
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L = f_func * v * ufl.dx
    
    # SUPG stabilization
    if stabilization:
        h = ufl.CellDiameter(domain)
        Pe = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl)) * h / (2.0 * epsilon)
        tau = ufl.conditional(
            ufl.gt(Pe, 1.0),
            h / (2.0 * ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))) * (1.0 - ufl.exp(-2.0 * Pe) / (1.0 - ufl.exp(-2.0))),
            h**2 / (12.0 * epsilon)
        )
        # Residual of the strong form
        res = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u)) - f_func
        a += tau * ufl.dot(beta_ufl, ufl.grad(v)) * res * ufl.dx
        L += tau * ufl.dot(beta_ufl, ufl.grad(v)) * f_func * ufl.dx
    
    # Solve
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
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build evaluation mapping
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