import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    dt = 0.008
    t_end = 0.08
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    t = 0.0
    u_exact_expr = ufl.exp(-t) * ufl.exp(5 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-0.0) * np.exp(5 * x[1]) * np.sin(np.pi * x[0]))
    
    # Boundary condition (Dirichlet on entire boundary)
    def boundary_marker(x):
        return np.full(x.shape[1], True)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-dependent boundary condition function
    u_bc = fem.Function(V)
    
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficient
    kappa = fem.Constant(domain, ScalarType(1.0))
    
    # Variational form (Backward Euler)
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx
    
    # Compile forms
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
    solver.setType(ksp_type)
    if ksp_type == "preonly":
        solver.getPC().setType(pc_type)
    else:
        solver.rtol = rtol
    
    # Time stepping
    uh = fem.Function(V)
    for n in range(num_steps):
        t += dt
        
        # Update boundary condition
        u_bc.interpolate(lambda x: np.exp(-t) * np.exp(5 * x[1]) * np.sin(np.pi * x[0]))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for non-zero Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
    
    # Prepare evaluation grid
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
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