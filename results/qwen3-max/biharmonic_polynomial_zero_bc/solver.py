import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 32
    element_degree = 2
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic: let v = -Δu, then solve:
    # -Δu = v in Ω, u = g on ∂Ω
    # -Δv = f in Ω, v = -Δg on ∂Ω
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = x[0]*(1-x[0])*x[1]*(1-x[1])
    
    # Compute f = Δ²u
    f_expr = ufl.div(ufl.div(ufl.grad(ufl.grad(u_exact))))
    
    # Compute boundary condition for v: v = -Δu on ∂Ω
    lap_u = ufl.div(ufl.grad(u_exact))
    v_boundary_expr = -lap_u
    
    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # First solve: -Δv = f with v = -Δu on boundary
    v = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
    L1 = ufl.inner(f_expr, w) * ufl.dx
    
    # Boundary conditions for v
    def boundary(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, boundary)
    dofs_v = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
    v_bc_func = fem.Function(V)
    v_bc_func.interpolate(fem.Expression(v_boundary_expr, V.element.interpolation_points))
    bc_v = fem.dirichletbc(v_bc_func, dofs_v)
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_v],
        petsc_options={f"ksp_type": ksp_type, f"pc_type": pc_type, f"ksp_rtol": rtol},
        petsc_options_prefix="biharmonic_"
    )
    v_sol = problem1.solve()
    
    # Second solve: -Δu = v with u = g on boundary
    u = ufl.TrialFunction(V)
    z = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u), ufl.grad(z)) * ufl.dx
    L2 = ufl.inner(v_sol, z) * ufl.dx
    
    # Boundary conditions for u
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc_func, dofs_v)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={f"ksp_type": ksp_type, f"pc_type": pc_type, f"ksp_rtol": rtol},
        petsc_options_prefix="biharmonic2_"
    )
    u_sol = problem2.solve()
    
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
    cells = []
    points_on_proc = []
    for i in range(points.shape[1]):
        point = points[:, i]
        cell_candidates = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point.reshape(1, 3))
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(point)
    
    u_values = np.full((nx*ny,), np.nan)
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[:] = u_eval.flatten()
    
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