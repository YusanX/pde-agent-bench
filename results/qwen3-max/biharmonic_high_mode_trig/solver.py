import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 32
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-6
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic: let v = -Δu, then solve:
    # -Δu = v in Ω, u = g on ∂Ω
    # -Δv = f in Ω, v = -Δg on ∂Ω
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Compute source term f = Δ²u
    f_expr = ufl.div(ufl.div(ufl.grad(ufl.grad(u_exact))))
    
    # Compute boundary data for v: v = -Δu on ∂Ω
    lap_u = ufl.div(ufl.grad(u_exact))
    v_boundary_expr = -lap_u
    
    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # First solve: -Δv = f with v = v_boundary on ∂Ω
    v = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
    L1 = ufl.inner(f_expr, w) * ufl.dx
    
    # Boundary condition for v
    def boundary(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs_v = fem.locate_dofs_topological(V, fdim, boundary_facets)
    v_bc_func = fem.Function(V)
    v_bc_func.interpolate(fem.Expression(v_boundary_expr, V.element.interpolation_points))
    bc_v = fem.dirichletbc(v_bc_func, dofs_v)
    
    problem_v = petsc.LinearProblem(
        a1, L1, bcs=[bc_v],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="biharmonic_v_"
    )
    v_sol = problem_v.solve()
    
    # Second solve: -Δu = v_sol with u = u_exact on ∂Ω
    u = ufl.TrialFunction(V)
    z = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u), ufl.grad(z)) * ufl.dx
    L2 = ufl.inner(v_sol, z) * ufl.dx
    
    # Boundary condition for u
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    problem_u = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="biharmonic_u_"
    )
    u_sol = problem_u.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Create bounding box tree and evaluate
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