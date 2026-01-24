import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.1
    dt = 0.01
    num_steps = int(t_end / dt)
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    t = 0.0
    
    # Exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-t) * ufl.sin(3*ufl.pi*(x[0]+x[1])) * ufl.sin(ufl.pi*(x[0]-x[1]))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-0.0) * np.sin(3*np.pi*(x[0]+x[1])) * np.sin(np.pi*(x[0]-x[1])))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficient
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Time derivative and diffusion terms
    F = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * 0.0) * v * ufl.dx
    
    # Source term derived from manufactured solution
    # Compute f = ∂u/∂t - ∇·(κ∇u)
    u_man = ufl.exp(-t) * ufl.sin(3*ufl.pi*(x[0]+x[1])) * ufl.sin(ufl.pi*(x[0]-x[1]))
    u_t = ufl.diff(u_man, t)
    laplacian_u = ufl.div(kappa * ufl.grad(u_man))
    f_expr = u_t - laplacian_u
    
    # Update F to include source term
    F = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f_expr) * v * ufl.dx
    
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Boundary conditions (Dirichlet on entire boundary)
    def boundary_marker(x):
        return np.full(x.shape[1], True)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create boundary condition function
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Prepare linear problem
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    solver.setTolerances(rtol=rtol)
    
    # Time-stepping
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    
    for n in range(num_steps):
        t += dt
        
        # Update boundary condition
        u_bc.interpolate(lambda x: np.exp(-t) * np.sin(3*np.pi*(x[0]+x[1])) * np.sin(np.pi*(x[0]-x[1])))
        
        # Update source term
        f_expr = ufl.diff(u_man, t) - ufl.div(kappa * ufl.grad(u_man))
        L_new = (u_n + dt * f_expr) * v * ufl.dx
        L_form = fem.form(L_new)
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
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
    for i, point in enumerate(points.T):
        cell_candidates = geometry.compute_collisions_points(bb_tree, point)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point)
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = u_h.eval(points_on_proc, cells)
        u_values[:len(u_eval)] = u_eval.flatten()
    
    # Reshape to (nx, ny)
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