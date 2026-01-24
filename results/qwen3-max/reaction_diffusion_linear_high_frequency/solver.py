import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    mesh_resolution = case_spec.get("mesh_resolution", 60)
    element_degree = case_spec.get("element_degree", 2)
    epsilon = case_spec.get("epsilon", 1.0)
    rtol = case_spec.get("rtol", 1e-8)
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Exact solution for manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    # Reaction term R(u) - for this case, we assume linear reaction R(u) = u
    # So the PDE is: -ε ∇²u + u = f
    # Compute f from exact solution
    laplacian_u = ufl.div(ufl.grad(u_exact))
    f_expr = -epsilon * laplacian_u + u_exact
    
    # Boundary condition (from exact solution)
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Create boundary condition
    boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, boundary_marker)
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(4 * np.pi * x[0]) * np.sin(3 * np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": rtol
        },
        petsc_options_prefix="reaction_diffusion_"
    )
    u_sol = problem.solve()
    
    # Create evaluation grid
    nx, ny = 70, 70
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol
        }
    }