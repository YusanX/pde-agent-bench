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
    nu = 0.2
    
    # Bilinear and linear forms
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p + ufl.div(u) * q) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions (lid-driven cavity: top wall moves, others fixed)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Top boundary (y=1): u = (1, 0)
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    # Other boundaries: u = (0, 0)
    def other_boundaries(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0)
        ))
    
    # Top boundary condition
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))
    
    # Other boundaries
    other_facets = mesh.locate_entities_boundary(domain, fdim, other_boundaries)
    dofs_other = fem.locate_dofs_topological((W.sub(0), V), fdim, other_facets)
    u_other = fem.Function(V)
    u_other.interpolate(lambda x: np.zeros((2, x.shape[1])))
    bc_other = fem.dirichletbc(u_other, dofs_other, W.sub(0))
    
    bcs = [bc_top, bc_other]
    
    # Solve linear system
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 1000
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
    
    # Evaluate velocity magnitude on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    for i in range(points.shape[1]):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(colliding_cells.links(i)[0])
    
    u_values = np.zeros((nx * ny, 2))
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[:len(points_on_proc)] = u_eval
    
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