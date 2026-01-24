import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Problem parameters
    eps = 0.03
    beta = [5.0, 2.0]
    
    # Create mesh
    mesh_resolution = 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    element_degree = 2
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term f derived from exact solution
    grad_u = ufl.grad(u_exact)
    laplacian_u = ufl.div(grad_u)
    f = -eps * laplacian_u + beta[0] * grad_u[0] + beta[1] * grad_u[1]
    
    # Boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Convection vector
    beta_vec = ufl.as_vector(beta)
    
    # Standard Galerkin form
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_vec, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx
    
    # Compute mesh size for SUPG stabilization
    h = ufl.CellDiameter(domain)
    Pe = ufl.sqrt(beta_vec[0]**2 + beta_vec[1]**2) * h / (2 * eps)
    tau = ufl.conditional(ufl.gt(Pe, 1), h / (2 * ufl.sqrt(beta_vec[0]**2 + beta_vec[1]**2)) * (1 - ufl.exp(-2 * Pe) / (1 - ufl.exp(-2 * Pe))), 0)
    
    # SUPG stabilization
    a_supg = tau * ufl.inner(beta_vec, ufl.grad(v)) * (eps * ufl.div(ufl.grad(u)) - ufl.inner(beta_vec, ufl.grad(u)) + f) * ufl.dx
    L_supg = tau * ufl.inner(beta_vec, ufl.grad(v)) * f * ufl.dx
    
    a_total = a + a_supg
    L_total = L + L_supg
    
    # Solve with direct solver
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
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