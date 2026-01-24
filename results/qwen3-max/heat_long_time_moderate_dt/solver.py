import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.2
    dt = 0.01
    kappa = 0.5
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Initial condition (u_n)
    u_n = fem.Function(V)
    u_exact_expr = lambda x: np.exp(-2*0.0) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
    u_n.interpolate(u_exact_expr)
    
    # Coefficient kappa
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # Source term f derived from manufactured solution u = exp(-2*t)*sin(pi*x)*sin(pi*y)
    # Compute f = du/dt - div(kappa*grad(u))
    # du/dt = -2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    # grad(u) = [pi*exp(-2*t)*cos(pi*x)*sin(pi*y), pi*exp(-2*t)*sin(pi*x)*cos(pi*y)]
    # div(grad(u)) = -2*pi^2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    # So f = -2*exp(-2*t)*sin(pi*x)*sin(pi*y) - kappa*(-2*pi^2*exp(-2*t)*sin(pi*x)*sin(pi*y))
    #     = exp(-2*t)*sin(pi*x)*sin(pi*y)*(-2 + 2*kappa*pi^2)
    t = 0.0
    f_expr = lambda x, t_val: np.exp(-2*t_val) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]) * (-2 + 2*kappa*np.pi**2)
    
    # Boundary condition (Dirichlet) - exact solution on boundary
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    dofs_geo = fem.locate_dofs_geometrical(V, boundary_marker)
    
    # Create boundary condition function
    u_bc = fem.Function(V)
    
    # Variational problem (Backward Euler)
    a = u * v * ufl.dx + dt * kappa_const * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt * fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx  # f will be updated in loop
    
    # Prepare linear problem
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = fem.petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Solver setup
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    # Time-stepping
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Update source term
        f_t = fem.Constant(domain, PETSc.ScalarType(0.0))
        # Instead of updating f_t, we'll directly use the expression in L
        # But since we have time-dependent BCs, we need to update them too
        
        # Update boundary condition
        u_bc_expr = lambda x: np.exp(-2*t) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, dofs_geo)
        
        # Update L with current f
        f_current = f_expr
        f_func = fem.Function(V)
        f_func.interpolate(lambda x: f_current(x, t))
        L_updated = u_n * v * ufl.dx + dt * f_func * v * ufl.dx
        L_form_updated = fem.form(L_updated)
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, L_form_updated)
        
        # Apply lifting for non-zero BCs
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.zeros((3, nx*ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Use interpolation for evaluation
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
    
    u_values = np.full((nx*ny,), np.nan)
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