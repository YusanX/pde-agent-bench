import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.12
    dt = 0.006
    kappa = 0.5
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 1
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
        return np.exp(-10*t) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: u_exact_func(x, 0.0))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term derived from manufactured solution
    x = ufl.SpatialCoordinate(domain)
    t_ufl = ufl.Constant(domain, ScalarType(0.0))
    u_exact = ufl.exp(-10*t_ufl) * ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    f_expr = ufl.diff(u_exact, t_ufl) - kappa * ufl.div(ufl.grad(u_exact))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    
    # Variational form (Backward Euler)
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v + dt * f * v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Solver setup
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    # Time-stepping loop
    t = 0.0
    u_h = fem.Function(V)
    for n in range(num_steps):
        t += dt
        
        # Update source term
        t_ufl.value = t
        f_eval = fem.Function(V)
        f_eval.interpolate(f)
        
        # Update boundary condition
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: u_exact_func(x, t))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve linear system
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.zeros((3, nx*ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build evaluation mapping
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