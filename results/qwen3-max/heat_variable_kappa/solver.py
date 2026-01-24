import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 32
    element_degree = 2
    dt = 0.01
    t_end = 0.1
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define coefficients
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1 + 0.3 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    kappa = fem.Expression(kappa_expr, V.element.interpolation_points)
    kappa_func = fem.Function(V)
    kappa_func.interpolate(kappa)
    
    # Exact solution for boundary and initial conditions
    def exact_solution(x, t):
        return np.exp(-t) * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Boundary condition
    u_D = fem.Function(V)
    def boundary_marker(x):
        return np.full_like(x[0], True, dtype=bool)
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(u_D, dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: exact_solution(x, 0.0))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Time derivative term
    F = u * v * ufl.dx - u_n * v * ufl.dx
    
    # Diffusion term with variable kappa
    F += dt * ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Source term derived from manufactured solution
    t = 0.0
    u_exact_t = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    u_t = -ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    laplacian_u = -8 * ufl.pi**2 * ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    grad_kappa = ufl.as_vector([
        -0.6 * ufl.pi * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
        -0.6 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    ])
    grad_u = ufl.as_vector([
        2 * ufl.pi * ufl.exp(-t) * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        2 * ufl.pi * ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    ])
    f_expr = u_t - ufl.div(kappa_expr * ufl.grad(u_exact_t)) 
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    
    # Add source term
    F -= dt * f_func * v * ufl.dx
    
    # Prepare forms
    a = ufl.lhs(F)
    L = ufl.r rhs(F)
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Time-stepping
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Update boundary condition
        u_D.interpolate(lambda x: exact_solution(x, t))
        
        # Update source term
        f_expr_t = (-ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) 
                   - ufl.div(kappa_expr * ufl.grad(ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]))))
        f = fem.Expression(f_expr_t, V.element.interpolation_points)
        f_func.interpolate(f)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree
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
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:] = u_eval.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12
        }
    }