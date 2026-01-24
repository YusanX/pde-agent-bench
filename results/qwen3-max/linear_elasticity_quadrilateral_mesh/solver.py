import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 32
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.quadrilateral)
    
    # Material parameters
    E = 1.0
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree, (domain.geometry.dim,)))
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.as_vector([
        ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1]),
        ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    ])
    
    # Boundary condition
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.vstack((
        np.sin(2 * np.pi * x[0]) * np.cos(3 * np.pi * x[1]),
        np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    )))
    
    # Apply Dirichlet BC on entire boundary
    def boundary_marker(x):
        return np.full(x.shape[1], True)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, dofs)
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(domain.geometry.dim)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term from manufactured solution
    f_expr = -ufl.div(sigma(u_exact_expr))
    
    # Variational form
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "pc_hYPRE_type": "boomeramg"
        },
        petsc_options_prefix="elasticity_"
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
    cells = []
    points_on_proc = []
    for i, point in enumerate(points.T):
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.zeros((nx * ny, 2))
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(points_on_proc, cells)
        u_values[:len(u_eval)] = u_eval
    
    # Compute displacement magnitude
    u_magnitude = np.linalg.norm(u_values, axis=1).reshape((nx, ny))
    
    return {
        "u": u_magnitude,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }