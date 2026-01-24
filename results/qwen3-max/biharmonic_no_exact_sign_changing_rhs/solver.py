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
    rtol = 1e-6
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed function space for biharmonic: V = H1 x H1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    W = V * V  # (u, v) where v = -Δu
    
    # Trial and test functions
    (u, v) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)
    
    # Source term f = cos(4*pi*x)*sin(3*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f = ufl.cos(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    # Mixed variational form for Δ²u = f
    # System: -Δu = v, -Δv = f
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(phi)) * ufl.dx +
        ufl.inner(ufl.grad(v), ufl.grad(psi)) * ufl.dx -
        ufl.inner(v, phi) * ufl.dx
    )
    L = ufl.inner(f, psi) * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω (g=0 for this case)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.full_like(x[0], True)  # Entire boundary
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    
    # BC for u component
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # BC for v component (natural BC, no constraint needed for v)
    bcs = [bc_u]
    
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
    u_sol = w_sol.sub(0).collapse()
    
    # Create evaluation grid
    nx, ny = 50, 50
    x_eval = np.linspace(0.0, 1.0, nx)
    y_eval = np.linspace(0.0, 1.0, ny)
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