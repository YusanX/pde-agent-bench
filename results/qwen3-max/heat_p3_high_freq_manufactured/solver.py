import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.08
    dt = 0.008
    num_steps = int(t_end / dt)
    
    # Mesh and function space
    mesh_resolution = 64
    element_degree = 3
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time parameters
    t = 0.0
    
    # Exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-t) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-0.0) * np.sin(3 * np.pi * x[0]) * np.sin(3 * np.pi * x[1]))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term f derived from exact solution
    # u = exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # ∂u/∂t = -exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # ∇u = [3*pi*exp(-t)*cos(3*pi*x)*sin(3*pi*y), 3*pi*exp(-t)*sin(3*pi*x)*cos(3*pi*y)]
    # ∇²u = -9*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y) -9*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y) = -18*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # f = ∂u/∂t - ∇·(κ∇u) = ∂u/∂t - κ∇²u = -exp(-t)*sin(3*pi*x)*sin(3*pi*y) + 18*pi²*exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # f = (18*pi² - 1) * exp(-t) * sin(3*pi*x)*sin(3*pi*y)
    f_expr = (18 * ufl.pi**2 - 1) * ufl.exp(-t) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Bilinear and linear forms
    kappa = 1.0
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Boundary conditions (Dirichlet on entire boundary)
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create boundary condition function
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
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
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    
    for n in range(num_steps):
        t += dt
        
        # Update source term
        f_value = (18 * np.pi**2 - 1) * np.exp(-t)
        f.value = f_value
        
        # Update boundary condition
        u_bc.interpolate(lambda x: np.exp(-t) * np.sin(3 * np.pi * x[0]) * np.sin(3 * np.pi * x[1]))
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on 50x50 grid
    nx, ny = 50, 50
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build mapping
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
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }