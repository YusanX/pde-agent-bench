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
    pc_type = "gamg"
    rtol = 1e-6
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed function space for mixed formulation: V = H1 x H1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    W = V * V  # (w, v) where w = -Δu, so Δw = f
    
    # Trial and test functions
    (w, u) = ufl.TrialFunctions(W)
    (q, v) = ufl.TestFunctions(W)
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(3*ufl.pi*x[0]) + ufl.cos(2*ufl.pi*x[1])
    
    # Source term f = Δ²u
    f_expr = ufl.div(ufl.div(ufl.grad(ufl.grad(u_exact_expr))))
    
    # Define forms for mixed formulation:
    # -Δw = f  => ∫∇w·∇q = ∫f q
    # -Δu = w  => ∫∇u·∇v = ∫w v
    a = (ufl.inner(ufl.grad(w), ufl.grad(q)) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = ufl.inner(f_expr, q) * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω
    def boundary(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Create function for exact boundary values
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.sin(3*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
    
    # Locate boundary DOFs for u component
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs_u = fem.locate_dofs_topological((W.sub(1), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_D, dofs_u, W.sub(1))
    
    # Also need boundary condition for w? In mixed formulation, w doesn't have BCs
    # Only u has Dirichlet BCs
    bcs = [bc_u]
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 1000
        },
        petsc_options_prefix="biharmonic_"
    )
    w_u_sol = problem.solve()
    
    # Extract u component
    u_sol = w_u_sol.sub(1).collapse()
    
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }