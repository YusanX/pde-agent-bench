import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed function space for mixed formulation: V1 for w, V2 for u
    V1 = fem.functionspace(domain, ("Lagrange", element_degree))
    V2 = fem.functionspace(domain, ("Lagrange", element_degree))
    W = V1 * V2
    
    # Trial and test functions
    (w, u) = ufl.TrialFunctions(W)
    (v1, v2) = ufl.TestFunctions(W)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(8 * ufl.pi * x[0]) * ufl.cos(6 * ufl.pi * x[1])
    
    # Mixed variational form: 
    # -Δu = w  => ∫∇u·∇v2 = ∫w v2
    # -Δw = f  => ∫∇w·∇v1 = ∫f v1
    a = (ufl.inner(ufl.grad(w), ufl.grad(v1)) + ufl.inner(ufl.grad(u), ufl.grad(v2)) - ufl.inner(w, v2)) * ufl.dx
    L = ufl.inner(f, v1) * ufl.dx
    
    # Boundary conditions: u = 0 on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    
    # Locate DOFs for u component
    dofs_u = fem.locate_dofs_topological((W.sub(1), V2), fdim, boundary_facets)
    u_bc = fem.Function(V2)
    u_bc.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(1))
    
    # Also need w = 0 on boundary for well-posedness of the mixed system
    dofs_w = fem.locate_dofs_topological((W.sub(0), V1), fdim, boundary_facets)
    w_bc = fem.Function(V1)
    w_bc.x.array[:] = 0.0
    bc_w = fem.dirichletbc(w_bc, dofs_w, W.sub(0))
    
    bcs = [bc_u, bc_w]
    
    # Solve the linear system
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
    w_sol = problem.solve()
    
    # Extract u component
    u_sol = w_sol.sub(1).collapse()
    
    # Create evaluation points on 50x50 grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution at points
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
    
    # Reshape to 2D grid
    u_grid = np.full((nx, ny), np.nan)
    for idx, (i, j) in enumerate(np.ndindex(nx, ny)):
        u_grid[i, j] = u_values[idx]
    
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