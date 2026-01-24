import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 64
    element_degree = 1
    dt = 0.02
    t_end = 0.1
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    # Setup
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary condition (u = 0 on boundary)
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    # Time parameters
    num_steps = int(t_end / dt)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-220*((x[0]-0.25)**2 + (x[1]-0.25)**2)) + ufl.exp(-220*((x[0]-0.75)**2 + (x[1]-0.7)**2))
    f = fem.Expression(f_expr, V.element.interpolation_points)
    
    # Coefficient
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Bilinear and linear forms for backward Euler
    a = u * v * ufl.dx + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx
    
    # Create forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    # Time-stepping
    t = 0.0
    u_sol = fem.Function(V)
    for _ in range(num_steps):
        t += dt
        # Update source term
        f_eval = fem.Expression(f_expr, V.element.interpolation_points)
        f_func = fem.Function(V)
        f_func.interpolate(f_eval)
        
        # Update RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
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
        points_on_proc = np.array(points_on_proc)
        cells = np.array(cells, dtype=np.int32)
        u_eval = u_sol.eval(points_on_proc, cells)
        u_values[:] = u_eval.flatten()
    
    # Handle any NaN values (shouldn't happen on unit square with proper mesh)
    u_values = np.nan_to_num(u_values, nan=0.0)
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