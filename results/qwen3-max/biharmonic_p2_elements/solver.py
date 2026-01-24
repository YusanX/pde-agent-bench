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
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed function space for biharmonic: V = H1 x H1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    W = V * V  # (u, v) where v = -Δu
    
    # Trial and test functions
    (u, v) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)
    
    # Exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    f = 16 * ufl.pi**4 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Mixed variational form: 
    # -Δv = f  => ∫∇v·∇psi = ∫f*psi
    # -Δu = v  => ∫∇u·∇phi = ∫v*phi
    a = (ufl.inner(ufl.grad(v), ufl.grad(psi)) + ufl.inner(ufl.grad(u), ufl.grad(phi)) - ufl.inner(v, phi)) * ufl.dx
    L = ufl.inner(f, psi) * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω
    def boundary(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    
    # Dirichlet BC for u component
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Dirichlet BC for v component (natural BC for biharmonic, but we set to exact for stability)
    v_bc = fem.Function(V)
    v_exact_expr = 4 * ufl.pi**2 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    v_bc_expr = fem.Expression(v_exact_expr, V.element.interpolation_points)
    v_bc.interpolate(v_bc_expr)
    dofs_v = fem.locate_dofs_topological((W.sub(1), V), fdim, boundary_facets)
    bc_v = fem.dirichletbc(v_bc, dofs_v, W.sub(1))
    
    bcs = [bc_u, bc_v]
    
    # Solve the mixed problem
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
    u_sol, v_sol = w_sol.split()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree and evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    for i in range(points.shape[1]):
        point = points[:, i]
        candidate_cells = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, candidate_cells, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:] = u_eval.flatten()
    
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