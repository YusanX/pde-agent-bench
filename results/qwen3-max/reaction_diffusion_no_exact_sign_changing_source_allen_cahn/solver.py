import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    source_term = case_spec.get("source_term", "3*cos(3*pi*x[0])*sin(2*pi*x[1])")
    epsilon = case_spec.get("epsilon", 0.01)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    
    # Allen-Cahn reaction term: R(u) = u^3 - u
    def reaction_term(u):
        return u**3 - u
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    element_degree = 2
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary condition (homogeneous Dirichlet)
    def boundary_marker(x):
        return np.full_like(x[0], True)  # All boundaries
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define variational problem
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Source term f = 3*cos(3*pi*x)*sin(2*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_expr = 3 * ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f = fem.Constant(domain, ScalarType(0.0))  # Will be updated if needed, but we use UFL expression directly
    
    # Weak form for steady Allen-Cahn: -ε ∇²u + (u³ - u) = f
    F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction_term(u) * v * ufl.dx - f_expr * v * ufl.dx
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, u, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    
    # Configure linear solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    
    # Solve
    n, converged = solver.solve(u)
    assert converged, "Newton solver did not converge"
    u.x.scatter_forward()
    
    # Evaluate solution on 70x70 grid
    nx, ny = 70, 70
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
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8
        }
    }