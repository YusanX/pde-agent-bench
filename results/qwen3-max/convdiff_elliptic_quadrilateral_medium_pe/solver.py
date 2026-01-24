import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Problem parameters
    epsilon = 0.05
    beta = [4.0, 2.0]
    
    # Create mesh (quadrilateral elements for better accuracy with SUPG)
    comm = MPI.COMM_WORLD
    mesh_resolution = 32
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [mesh_resolution, mesh_resolution], 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    element_degree = 2
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for boundary conditions and source term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # Source term f = -ε∇²u + β·∇u
    grad_u = ufl.grad(u_exact)
    laplacian_u = ufl.div(grad_u)
    f_expr = -epsilon * laplacian_u + beta[0] * grad_u[0] + beta[1] * grad_u[1]
    
    # Boundary conditions (Dirichlet everywhere)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin form
    a_galerkin = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_galerkin += (beta[0] * u.dx(0) + beta[1] * u.dx(1)) * v * ufl.dx
    L_galerkin = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(beta[0]**2 + beta[1]**2)
    tau = h / (2 * beta_mag) * (ufl.sqrt(1 + (4 * epsilon * beta_mag**2) / (h**2 * beta_mag**2)) - 1)
    
    # Residual of the PDE
    res = -epsilon * ufl.div(ufl.grad(u)) + beta[0] * u.dx(0) + beta[1] * u.dx(1) - f_expr
    
    # SUPG terms
    a_supg = tau * (beta[0] * u.dx(0) + beta[1] * u.dx(1)) * (beta[0] * v.dx(0) + beta[1] * v.dx(1)) * ufl.dx
    L_supg = tau * f_expr * (beta[0] * v.dx(0) + beta[1] * v.dx(1)) * ufl.dx
    
    # Combined forms
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
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
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Find cells containing points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Map points to cells
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate solution
    u_values = np.full((nx * ny,), np.nan)
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