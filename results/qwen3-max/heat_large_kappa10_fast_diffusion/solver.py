import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    comm = MPI.COMM_WORLD
    t_end = 0.05
    dt = 0.005
    kappa = 10.0
    nx = ny = 32
    degree = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define time and solution functions
    u_n = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Exact solution for initial condition
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-ufl.Constant(domain, ScalarType(0.0))) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    
    # Time-dependent exact solution for boundary conditions
    t = 0.0
    def u_exact_func(x, t_val):
        return np.exp(-t_val) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Boundary condition
    def boundary_marker(x):
        return np.full_like(x[0], True)
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational form
    f_expr = ufl.exp(-ufl.Constant(domain, ScalarType(t))) * (2 * kappa * ufl.pi**2 - 1) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = fem.Constant(domain, ScalarType(0.0))
    
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx
    
    # Prepare forms
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
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.rtol = 1e-8
    
    # Time stepping
    num_steps = int(t_end / dt)
    for i in range(num_steps):
        t += dt
        # Update source term
        f_value = (2 * kappa * np.pi**2 - 1) * np.exp(-t)
        f.value = ScalarType(f_value)
        # Update boundary condition
        u_bc.interpolate(lambda x: u_exact_func(x, t))
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        # Solve
        uh = fem.Function(V)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
    
    # Create evaluation grid
    nx_grid = ny_grid = 50
    x = np.linspace(0, 1, nx_grid)
    y = np.linspace(0, 1, ny_grid)
    X, Y = np.meshgrid(x, y, indexing='ij')
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_grid, ny_grid))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8
        }
    }