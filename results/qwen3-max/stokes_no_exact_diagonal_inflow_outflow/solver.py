import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 32
    degree_u = 2
    degree_p = 1
    ksp_type = "minres"
    pc_type = "jacobi"
    rtol = 1e-6
    
    # Setup
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function spaces (Taylor-Hood)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Trial and test functions
    w = ufl.TrialFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    # Source term
    f = ufl.as_vector((0.0, 0.0))
    
    # Viscosity
    nu = 0.8
    
    # Bilinear and linear forms
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p + ufl.div(u) * q) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions (u = g on ∂Ω)
    # For this problem, we assume g = 0 on all boundaries
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_converged_reason": "none"
        },
        petsc_options_prefix="stokes_"
    )
    solution = problem.solve()
    u_sol, p_sol = solution.split()
    
    # Create evaluation grid
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate velocity magnitude
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    velocity_magnitude = np.linalg.norm(u_values, axis=1).reshape((nx, ny))
    
    return {
        "u": velocity_magnitude,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }