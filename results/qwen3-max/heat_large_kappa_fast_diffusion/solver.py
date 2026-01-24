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
    t_end = 0.08
    dt = 0.004
    kappa = 5.0
    
    # Mesh resolution and element degree (chosen for accuracy vs speed)
    nx = ny = 64
    element_degree = 2
    
    # Solver parameters
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time variables
    t = 0.0
    
    # Define exact solution for boundary and initial conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.exp(-0.0) * np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Boundary condition (Dirichlet on entire boundary)
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
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Time derivative and diffusion terms
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx
    
    # Create forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Set up solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.rtol = rtol
    
    # Time stepping
    num_steps = int(t_end / dt)
    for i in range(num_steps):
        t += dt
        
        # Update exact solution for boundary condition at current time
        u_exact_t = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.exp(-t) * np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Update source term f
        # f = ∂u/∂t - κ∇²u
        # ∂u/∂t = -exp(-t)*sin(2πx)*sin(πy)
        # ∇²u = -exp(-t)*(4π² + π²)*sin(2πx)*sin(πy) = -5π²*exp(-t)*sin(2πx)*sin(πy)
        # f = -exp(-t)*sin(2πx)*sin(πy) - κ*(-5π²*exp(-t)*sin(2πx)*sin(πy))
        #   = exp(-t)*sin(2πx)*sin(πy)*(-1 + 5*κ*π²)
        f_expr = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1 + 5 * kappa * ufl.pi**2)
        f = fem.Constant(domain, ScalarType(0.0))  # Placeholder, will be replaced in L
        
        # Update L with source term
        L_updated = u_n * v * ufl.dx + dt * f_expr * v * ufl.dx
        L_form_updated = fem.form(L_updated)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form_updated)
        
        # Apply lifting for non-zero Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        u_h = fem.Function(V)
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Update u_n for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # Prepare output on 50x50 grid
    nx_out = ny_out = 50
    x_points = np.linspace(0, 1, nx_out)
    y_points = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution on the grid
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
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }