import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.1
    dt = 0.02
    nx = ny = 32
    element_degree = 1
    
    # Create mesh and function space
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define time variables
    t = 0.0
    
    # Define source term f = 1 + sin(2*pi*x)*cos(2*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_expr = 1.0 + ufl.sin(2*ufl.pi*x[0])*ufl.cos(2*ufl.pi*x[1])
    f = fem.Expression(f_expr, V.element.interpolation_points)
    
    # Define coefficient kappa = 1 + 0.6*sin(2*pi*x)*sin(2*pi*y)
    kappa_expr = 1.0 + 0.6*ufl.sin(2*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    kappa = fem.Expression(kappa_expr, V.element.interpolation_points)
    
    # Initial condition u0 = sin(pi*x)*sin(pi*y)
    u0_expr = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    u0 = fem.Expression(u0_expr, V.element.interpolation_points)
    
    # Create functions
    u_n = fem.Function(V)
    u_n.interpolate(u0)
    u = fem.Function(V)
    
    # Boundary condition (Dirichlet on entire boundary)
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    g = fem.Function(V)
    g.interpolate(lambda x: np.full_like(x[0], 0.0))
    bc = fem.dirichletbc(g, dofs)
    
    # Time-stepping parameters
    num_steps = int(t_end / dt)
    
    # Variational problem
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    
    # Interpolate kappa to function for use in form
    kappa_func = fem.Function(V)
    kappa_func.interpolate(kappa)
    
    # Backward Euler scheme: (u - u_n)/dt - div(kappa*grad(u)) = f
    a = u_trial * v * ufl.dx + dt * ufl.inner(kappa_func * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f_expr) * v * ufl.dx
    
    # Create forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-8)
    
    # Time-stepping loop
    for _ in range(num_steps):
        t += dt
        
        # Update f and kappa if needed (they are time-independent here)
        f_func = fem.Function(V)
        f_func.interpolate(f)
        
        # Reassemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve linear system
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
    
    # Create evaluation grid (50x50)
    nx_eval = ny_eval = 50
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    for i in range(points.shape[1]):
        point = points[:, i]
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        u_eval = u.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:len(u_eval)] = u_eval.flatten()
    
    # Reshape to 2D grid
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    # Return results
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8
        }
    }