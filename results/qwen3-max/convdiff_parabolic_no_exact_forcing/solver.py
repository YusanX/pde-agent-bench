import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters from case_spec or defaults
    mesh_resolution = 64
    element_degree = 1
    dt = 0.02
    t_end = 0.1
    epsilon = 0.05
    beta = [2.0, 1.0]
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary condition (u = 0 on boundary)
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
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
    f = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Convection vector
    beta_ufl = ufl.as_vector(beta)
    
    # SUPG stabilization parameters
    h = ufl.CellDiameter(domain)
    Pe = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl)) * h / (2.0 * epsilon)
    tau = ufl.conditional(
        ufl.gt(Pe, 1.0),
        h / (2.0 * ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe),
        h**2 / (12.0 * epsilon)
    )
    
    # Bilinear form with SUPG
    a = (
        u * v * ufl.dx 
        + dt * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - dt * ufl.inner(beta_ufl, ufl.grad(u)) * v * ufl.dx
        + dt * tau * ufl.inner(beta_ufl, ufl.grad(v)) * ufl.inner(beta_ufl, ufl.grad(u)) * ufl.dx
    )
    
    # Linear form with SUPG
    L = (
        u_n * v * ufl.dx 
        + dt * f * v * ufl.dx
        + dt * tau * ufl.inner(beta_ufl, ufl.grad(v)) * f * ufl.dx
    )
    
    # Prepare linear solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.rtol = 1e-8
    
    # Time-stepping
    u_h = fem.Function(V)
    for _ in range(num_steps):
        # Update time-dependent terms (none in this case)
        
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
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Find cells for evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    for i, point in enumerate(points.T):
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        eval_points = np.array(points_on_proc)
        eval_cells = np.array(cells, dtype=np.int32)
        u_eval = u_h.eval(eval_points, eval_cells)
        u_values[:] = u_eval.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8
        }
    }