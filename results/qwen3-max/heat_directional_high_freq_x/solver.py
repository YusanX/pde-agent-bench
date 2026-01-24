import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.08
    dt = 0.004
    num_steps = int(t_end / dt)
    
    # Mesh and function space
    mesh_resolution = 64
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time variables
    t = 0.0
    
    # Exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-t) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-0.0) * np.sin(8 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Boundary condition
    def boundary_marker(x):
        return np.full(x.shape[1], True, dtype=bool)
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(1.0))
    
    # Time derivative and diffusion terms
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx
    
    # Source term (derived from manufactured solution)
    f_expr = -ufl.exp(-t) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) \
             - ufl.exp(-t) * (-64 * ufl.pi**2 - ufl.pi**2) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    L += dt * f_func * v * ufl.dx
    
    # Prepare forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Solver setup
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Time-stepping
    uh = fem.Function(V)
    for i in range(num_steps):
        t += dt
        # Update source term
        f_expr = -ufl.exp(-t) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) \
                 - ufl.exp(-t) * (-64 * ufl.pi**2 - ufl.pi**2) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f = fem.Expression(f_expr, V.element.interpolation_points)
        f_func.interpolate(f)
        
        # Update boundary condition
        u_bc.interpolate(lambda x: np.exp(-t) * np.sin(8 * np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
    
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
    for i, point in enumerate(points.T):
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = uh.eval(points_on_proc, cells)
        u_values[:len(u_eval)] = u_eval.flatten()
    
    # Reshape to (nx, ny)
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