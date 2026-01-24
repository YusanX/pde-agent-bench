import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.1
    dt = 0.01
    nx = ny = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Coefficients
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1 + 0.8 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    kappa = fem.Expression(kappa_expr, V.element.interpolation_points)
    kappa_func = fem.Function(V)
    kappa_func.interpolate(kappa)
    
    # Exact solution for boundary and initial conditions
    def exact_solution(x, t):
        return np.exp(-t) * np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: exact_solution(x, 0.0))
    
    # Boundary condition
    u_D = fem.Function(V)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_geometrical(V, lambda x: np.full(x.shape[1], True, dtype=bool)))
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_expr = ( -ufl.exp(-x[2]) * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
              + kappa_expr * ufl.exp(-x[2]) * (9*ufl.pi**2 + 4*ufl.pi**2) * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
              - ufl.exp(-x[2]) * ufl.dot(ufl.grad(kappa_expr), ufl.grad(ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1]))) )
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    
    a = u * v * ufl.dx + dt * ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f_func) * v * ufl.dx
    
    # Prepare linear solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.rtol = rtol
    
    # Time-stepping
    t = 0.0
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    
    for n in range(num_steps):
        t += dt
        # Update boundary condition
        u_D.interpolate(lambda x: exact_solution(x, t))
        # Update source term
        f_func.interpolate(f)
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array
    
    # Create evaluation grid
    nx_grid, ny_grid = 50, 50
    x_eval = np.linspace(0, 1, nx_grid)
    y_eval = np.linspace(0, 1, ny_grid)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx_grid * ny_grid))
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_grid, ny_grid))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol
    }
    
    return {"u": u_grid, "solver_info": solver_info}