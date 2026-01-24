import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.2
    dt = 0.02
    kappa = 0.5
    
    # Agent-selectable parameters
    mesh_resolution = 32
    element_degree = 2
    
    # Setup
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    t = 0.0
    u_exact_expr = ufl.exp(-2*t) * ufl.cos(ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-2*0.0) * np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
    
    # Boundary condition (Dirichlet on entire boundary)
    def boundary_marker(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
        )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-dependent boundary condition function
    u_bc = fem.Function(V)
    
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term derived from manufactured solution
    # u = exp(-2t)*cos(pi*x)*cos(pi*y)
    # ut = -2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # laplacian_u = -2*pi^2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # f = ut - kappa*laplacian_u = (-2 + 2*kappa*pi^2)*exp(-2t)*cos(pi*x)*cos(pi*y)
    f_expr = (-2 + 2*kappa*np.pi**2) * ufl.exp(-2*t) * ufl.cos(ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1])
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Variational form (Backward Euler)
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Time stepping
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    
    for n in range(num_steps):
        t += dt
        
        # Update boundary condition
        u_bc.interpolate(lambda x: np.exp(-2*t) * np.cos(np.pi*x[0]) * np.cos(np.pi*x[1]))
        
        # Update source term
        f_value = (-2 + 2*kappa*np.pi**2) * np.exp(-2*t)
        f.value = f_value
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for non-zero Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array
    
    # Create evaluation grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx*ny))
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
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8
        }
    }