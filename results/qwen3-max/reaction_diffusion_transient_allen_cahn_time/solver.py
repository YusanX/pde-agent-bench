import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters
    dt = case_spec.get("dt", 0.02)
    t_end = case_spec.get("t_end", 0.3)
    epsilon = case_spec.get("epsilon", 1.0)
    reaction_lambda = case_spec.get("reaction_lambda", 1.0)
    
    # Mesh and function space
    comm = MPI.COMM_WORLD
    nx = ny = 64
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    t = 0.0
    u_exact_expr = 0.2 * ufl.exp(-0.5 * t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: 0.2 * np.exp(-0.5 * 0.0) * np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Boundary condition
    def boundary_marker(x):
        return np.full_like(x[0], True, dtype=bool)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-dependent boundary condition function
    u_bc = fem.Function(V)
    
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Reaction term for Allen-Cahn: R(u) = lambda * (u^3 - u)
    u_k = fem.Function(V)
    u_k.x.array[:] = u_n.x.array
    
    # Weak form for backward Euler
    F = (u - u_n) / dt * v * ufl.dx + epsilon * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        reaction_lambda * (u_k**3 - u_k) * v * ufl.dx
    
    # Source term derived from manufactured solution
    # Compute f = du/dt - epsilon*laplacian(u) + reaction_lambda*(u^3 - u)
    t_sym = ufl.variable(ufl.real(ufl.t))
    u_man = 0.2 * ufl.exp(-0.5 * t_sym) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_t = ufl.diff(u_man, t_sym)
    laplacian_u = ufl.div(ufl.grad(u_man))
    f_expr = u_t - epsilon * laplacian_u + reaction_lambda * (u_man**3 - u_man)
    
    f = fem.Function(V)
    
    # Nonlinear problem setup
    problem = petsc.NonlinearProblem(F, u_k, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.max_it = 20
    
    # Set linear solver options
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    # Time-stepping
    t = 0.0
    for n in range(num_steps):
        t += dt
        
        # Update boundary condition
        u_bc.interpolate(lambda x: 0.2 * np.exp(-0.5 * t) * np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        # Update source term
        f.interpolate(fem.Expression(f_expr, V.element.interpolation_points, t=t))
        
        # Update the reaction term coefficient
        # Solve nonlinear problem
        n_iter, converged = solver.solve(u_k)
        assert converged, f"Newton solver did not converge at time step {n}"
        
        # Update previous solution
        u_n.x.array[:] = u_k.x.array
    
    # Prepare output grid
    nx_out = ny_out = 65
    x_vals = np.linspace(0, 1, nx_out)
    y_vals = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
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
        vals = u_k.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": [nx, ny],
            "element_degree": 1,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8
        }
    }