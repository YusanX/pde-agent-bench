import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Problem parameters
    eps = 0.05
    beta = [2.0, 1.0]
    
    # Create mesh
    comm = MPI.COMM_WORLD
    mesh_resolution = 64
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    element_degree = 1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary condition (u = 0 on all boundaries)
    def boundary_marker(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
        )
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-250 * ((x[0] - 0.35)**2 + (y_expr := (x[1] - 0.65))**2))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Convection vector
    beta_ufl = ufl.as_vector(beta)
    
    # Standard Galerkin form
    a_standard = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L_standard = ufl.inner(f_func, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    Pe = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl)) * h / (2.0 * eps)
    tau = ufl.conditional(ufl.gt(Pe, 1.0), h / (2.0 * ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))) * (1.0 - 1.0/Pe), 0.0)
    
    # Residual of the strong form
    R = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u)) - f_func
    
    # SUPG terms
    a_supg = tau * ufl.dot(beta_ufl, ufl.grad(v)) * (-eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))) * ufl.dx
    L_supg = tau * ufl.dot(beta_ufl, ufl.grad(v)) * f_func * ufl.dx
    
    # Total forms
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # Solver configuration
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
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
    
    # Evaluate solution on 50x50 grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Find cells for evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }