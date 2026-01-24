import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    t_end = 0.06
    dt = 0.01
    num_steps = int(t_end / dt)
    
    # Mesh and function space
    mesh_resolution = 32
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time variables
    t = 0.0
    
    # Coefficient kappa
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1 + 0.4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    kappa = fem.Expression(kappa_expr, V.element.interpolation_points)
    kappa_func = fem.Function(V)
    kappa_func.interpolate(kappa)
    
    # Exact solution for boundary and initial conditions
    u_exact_expr = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_n_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u_n_expr)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term f derived from manufactured solution
    u_m = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_expr = -ufl.diff(u_m, t) + ufl.div(kappa_expr * ufl.grad(u_m))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    
    # Boundary condition (Dirichlet on entire boundary)
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-stepping forms
    a = u * v * ufl.dx + dt * ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f_func) * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    # Time-stepping loop
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    
    for i in range(num_steps):
        t += dt
        
        # Update source term
        f_expr_t = -ufl.diff(u_m, t) + ufl.div(kappa_expr * ufl.grad(u_m))
        f = fem.Expression(f_expr_t, V.element.interpolation_points)
        f_func.interpolate(f)
        
        # Update boundary condition
        u_exact_expr_t = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        u_bc_expr = fem.Expression(u_exact_expr_t, V.element.interpolation_points)
        u_bc = fem.Function(V)
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, dofs)
        
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
        u_n.x.array[:] = u_h.x.array
    
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
        cell = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        if len(cell) > 0:
            colliding_cells = geometry.compute_colliding_cells(domain, cell, point.reshape(1, 3))
            if len(colliding_cells.links(0)) > 0:
                cells.append(colliding_cells.links(0)[0])
                points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = u_h.eval(points_on_proc, cells)
        u_values[:len(u_eval)] = u_eval.flatten()
    
    # Reshape to 2D grid
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