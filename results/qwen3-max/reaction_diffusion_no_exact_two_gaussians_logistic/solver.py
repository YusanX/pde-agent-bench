import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec with defaults
    mesh_resolution = case_spec.get("mesh_resolution", 60)
    epsilon = case_spec.get("epsilon", 0.01)
    newton_max_it = case_spec.get("newton_max_it", 20)
    pc_type = case_spec.get("pc_type", "lu")
    ksp_type = case_spec.get("ksp_type", "preonly")
    rtol = case_spec.get("rtol", 1e-8)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Define boundary condition (u = 0 on boundary)
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Define source term f
    x = ufl.SpatialCoordinate(domain)
    f_expr = 6 * (ufl.exp(-160 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) + 
                  0.8 * ufl.exp(-160 * ((x[0] - 0.75)**2 + (x[1] - 0.35)**2)))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Define reaction term R(u) = u*(1-u) (logistic reaction)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fem.Function(V)
    
    # Steady reaction-diffusion equation: -ε ∇²u + R(u) = f
    # Weak form: ε ∇u·∇v + R(u)v = f v
    F = epsilon * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + \
        (uh * (1 - uh) - f_func) * v * ufl.dx
    
    # Nonlinear problem setup
    problem = petsc.NonlinearProblem(F, uh, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = rtol
    solver.max_it = newton_max_it
    solver.krylov_solver.setType(ksp_type)
    solver.krylov_solver.getPC().setType(pc_type)
    
    # Solve
    n, converged = solver.solve(uh)
    assert converged, "Newton solver did not converge"
    uh.x.scatter_forward()
    
    # Create evaluation grid (70x70)
    nx, ny = 70, 70
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": 1,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }