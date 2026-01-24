import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.1
    dt = 0.02
    num_steps = int(t_end / dt)
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 1
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    t = 0.0
    
    # Coefficients
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    kappa = fem.Expression(kappa_expr, V.element.interpolation_points)
    kappa_func = fem.Function(V)
    kappa_func.interpolate(kappa)
    
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    # Boundary conditions
    def boundary_marker(x):
        return np.full_like(x[0], True)  # All boundaries
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(g, dofs, V)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler time discretization
    a = u * v * ufl.dx + dt * ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx
    
    # Prepare linear solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    ksp_type = "cg"
    pc_type = "hypre"
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=1e-6)
    rtol = 1e-6
    
    u_h = fem.Function(V)
    
    # Time stepping
    for _ in range(num_steps):
        t += dt
        
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
    
    # Create evaluation grid
    nx, ny = 50, 50
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
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