import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    mesh_resolution = case_spec.get("mesh_resolution", 32)
    cell_type_str = case_spec.get("cell_type", "quadrilateral")
    reaction_rho = case_spec.get("reaction_rho", 10.0)
    
    # Map cell type string to dolfinx cell type
    if cell_type_str == "quadrilateral":
        cell_type = mesh.CellType.quadrilateral
    else:
        cell_type = mesh.CellType.triangle
    
    # Set up mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=cell_type)
    
    # Function space
    element_degree = 1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define boundary condition (u = 0 on all boundaries)
    def boundary_marker(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fem.Function(V)
    
    # Parameters
    eps = 0.01
    rho = reaction_rho
    f = fem.Constant(domain, ScalarType(1.0))
    
    # Logistic reaction term: R(u) = rho * u * (1 - u)
    # For steady state: -eps * ∇²u + rho * u * (1 - u) = f
    # Weak form: eps * ∇u · ∇v + rho * u * (1 - u) * v = f * v
    F = (eps * ufl.inner(ufl.grad(uh), ufl.grad(v)) 
         + rho * uh * (1 - uh) * v 
         - f * v) * ufl.dx
    
    # Nonlinear problem setup
    problem = petsc.NonlinearProblem(F, uh, bcs=[bc])
    solver = fem.nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.max_it = 50
    
    # Configure linear solver within Newton
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    # Solve
    n, converged = solver.solve(uh)
    assert converged, "Newton solver did not converge"
    uh.x.scatter_forward()
    
    # Create evaluation grid (65x65)
    nx, ny = 65, 65
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate solution on the grid
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
        u_eval = uh.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:] = u_eval.flatten()
    
    # Reshape to 2D grid
    u_grid = u_values.reshape((nx, ny))
    
    # Return results
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