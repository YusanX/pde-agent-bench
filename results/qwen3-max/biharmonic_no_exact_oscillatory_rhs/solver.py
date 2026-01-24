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
    
    # Mixed formulation for biharmonic: Δ²u = f
    # Let w = -Δu, then we have:
    # -Δw = f in Ω
    # w = -Δu in Ω
    # u = g on ∂Ω (g=0 in this case)
    # For the second equation, we need boundary conditions for w.
    # Since u = 0 on ∂Ω, we can use natural BC for w: ∂u/∂n = 0 (clamped plate assumption)
    # However, for simplicity and since no exact solution is given, we use a mixed formulation with two Poisson equations
    # with homogeneous Dirichlet BCs for both u and w.
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # First solve: -Δw = f with w = 0 on ∂Ω
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f, v) * ufl.dx
    
    # Boundary condition for w (homogeneous Dirichlet)
    def boundary(x):
        return np.full_like(x[0], True)  # All boundary points
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(ScalarType(0.0), dofs_w, V)
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="biharmonic_"
    )
    w_sol = problem1.solve()
    
    # Second solve: -Δu = w with u = 0 on ∂Ω
    u = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_sol, v) * ufl.dx
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(ScalarType(0.0), dofs_u, V)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="biharmonic2_"
    )
    u_sol = problem2.solve()
    
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