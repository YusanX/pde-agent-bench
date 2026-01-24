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
    dt = 0.005
    kappa = 1.0
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define boundary condition
    def boundary_marker(x):
        return np.full_like(x[0], True)  # All boundaries
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Exact solution for boundary condition
    def u_exact_func(x, t):
        return np.exp(-t) * np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: u_exact_func(x, 0.0))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term derived from manufactured solution
    t = 0.0
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    f_expr = -ufl.diff(u_exact, t) + kappa * ufl.div(kappa * ufl.grad(u_exact))
    f = fem.Constant(domain, ScalarType(0.0))
    
    # Bilinear and linear forms
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx
    
    # Create boundary condition function
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Prepare forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    # Time-stepping
    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array
    
    for n in range(num_steps):
        t += dt
        
        # Update source term
        f_expr_t = -(-ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])) + \
                   kappa * ufl.div(kappa * ufl.grad(ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])))
        f_expr_compiled = fem.Expression(f_expr_t, V.element.interpolation_points)
        f.value = f_expr_compiled.eval(domain, np.zeros((3, 1)))[0]
        
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
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
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
        u_eval = uh.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
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