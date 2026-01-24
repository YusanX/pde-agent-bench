import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters
    mesh_resolution = case_spec.get("mesh_resolution", 32)
    element_degree = case_spec.get("element_degree", 2)
    ksp_type = case_spec.get("ksp_type", "preonly")
    pc_type = case_spec.get("pc_type", "lu")
    rtol = case_spec.get("rtol", 1e-8)
    
    # Set up
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Create quadrilateral mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Reaction term R(u) - we'll use R(u) = u for simplicity (linear case)
    # For the manufactured solution, we need to compute f accordingly
    epsilon = 1.0  # diffusion coefficient
    
    # Compute source term f from the exact solution
    # Steady: -ε ∇²u + R(u) = f
    # We choose R(u) = u, so f = -ε ∇²u + u
    grad_u = ufl.grad(u_exact)
    laplacian_u = ufl.div(grad_u)
    f_expr = -epsilon * laplacian_u + u_exact
    
    # Boundary condition (Dirichlet) from exact solution
    def boundary_marker(x):
        return np.full_like(x[0], True)  # all boundaries
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Bilinear and linear forms for steady problem
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="reaction_diffusion_"
    )
    u_sol = problem.solve()
    
    # Create evaluation grid (60x60)
    nx, ny = 60, 60
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }