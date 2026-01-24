import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    # Parameters from case_spec or defaults
    stabilization = case_spec.get("stabilization", "supg")
    upwind_parameter = case_spec.get("upwind_parameter", 0.5)
    dt = case_spec.get("dt", 0.02)
    mesh_resolution = case_spec.get("mesh_resolution", 64)
    ksp_type = case_spec.get("ksp_type", "gmres")
    pc_type = case_spec.get("pc_type", "ilu")
    rtol = case_spec.get("rtol", 1e-8)
    
    # Problem parameters
    epsilon = 0.02
    beta = [6.0, 3.0]
    t_end = 0.1
    
    # Create mesh and function space
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define boundary condition (u = g on ∂Ω)
    # Since no specific g is given, we assume homogeneous Dirichlet BCs
    def boundary_marker(x):
        return np.full_like(x[0], True)  # All boundaries
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term f = exp(-150*((x-0.4)**2 + (y-0.6)**2))*exp(-t)
    t = 0.0  # Will be updated in time loop
    f_expr = ufl.exp(-150 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2)) * ufl.exp(-t)
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Convection vector
    beta_vec = ufl.as_vector(beta)
    
    # Standard Galerkin terms
    a_galerkin = (u * v / dt + epsilon * ufl.dot(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
    L_galerkin = (u_n / dt + f) * v * ufl.dx
    
    # SUPG stabilization
    if stabilization == "supg":
        h = ufl.CellDiameter(domain)
        Pe = ufl.sqrt(ufl.dot(beta_vec, beta_vec)) * h / (2 * epsilon)
        tau = ufl.conditional(
            ufl.gt(Pe, 1),
            h / (2 * ufl.sqrt(ufl.dot(beta_vec, beta_vec))) * (1 / ufl.tanh(Pe) - 1 / Pe),
            h**2 / (12 * epsilon)
        )
        # Residual of the PDE
        R = u / dt - epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - f
        a_supg = tau * ufl.dot(beta_vec, ufl.grad(v)) * R * ufl.dx
        L_supg = tau * ufl.dot(beta_vec, ufl.grad(v)) * (u_n / dt - f) * ufl.dx
        
        a = a_galerkin + a_supg
        L = L_galerkin + L_supg
    else:
        a = a_galerkin
        L = L_galerkin
    
    # Prepare forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Set up solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.rtol = rtol
    
    # Time-stepping
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    
    for n in range(num_steps):
        t += dt
        # Update source term
        f.value = 0.0  # We'll handle it through the expression in L
        
        # Update f_expr in L by redefining L with current t
        f_expr_current = ufl.exp(-150 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2)) * ufl.exp(-t)
        if stabilization == "supg":
            R_current = u / dt - epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u)) - f_expr_current
            L_current = (u_n / dt + f_expr_current) * v * ufl.dx + tau * ufl.dot(beta_vec, ufl.grad(v)) * (u_n / dt - f_expr_current) * ufl.dx
        else:
            L_current = (u_n / dt + f_expr_current) * v * ufl.dx
        
        L_form = fem.form(L_current)
        
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
        u_eval = u_h.eval(points_on_proc, cells)
        u_values[:len(u_eval)] = u_eval.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": 1,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol
    }
    
    return {"u": u_grid, "solver_info": solver_info}