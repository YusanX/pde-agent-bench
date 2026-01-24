import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
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
        ufl.tanh(6*(x[1]-0.5))*ufl.sin(ufl.pi*x[0]),
        0.1*ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    ])
    
    # Boundary condition: apply exact solution on all boundaries
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))
    
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
            "pc_hYPRE_type": "boomeramg",
            "ksp_max_it": 1000
        },
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_grid = np.linspace(0, 1, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree and evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    for i in range(points.shape[1]):
        point = points[:, i]
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.zeros((nx * ny, 2))
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:len(points_on_proc)] = u_eval
    
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