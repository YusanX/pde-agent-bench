import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Parameters
    mesh_resolution = 64
    degree_u = 2
    degree_p = 1
    ksp_type = "minres"
    pc_type = "jacobi"
    rtol = 1e-8
    pressure_fixing = True
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function spaces (Taylor-Hood)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Exact solution for boundary conditions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.as_vector([ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
                                  -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])])
    p_exact_expr = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Interpolate exact solution for boundary conditions
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.vstack((np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
                                        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))))
    
    # Boundary condition
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, dofs_u, W.sub(0))
    
    # Viscosity
    nu = 0.1
    
    # Source term derived from exact solution
    f_expr = -nu * ufl.div(ufl.grad(u_exact_expr)) + ufl.grad(p_exact_expr)
    f = fem.Constant(domain, ScalarType((0.0, 0.0)))  # Will be replaced by expression in form
    
    # Bilinear and linear forms
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.inner(p, ufl.div(v)) + ufl.inner(ufl.div(u), q)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Handle pressure nullspace
    if pressure_fixing:
        # Pin pressure at a point to fix nullspace
        p_point = np.array([[0.0, 0.0, 0.0]])
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cells = []
        points_on_proc = []
        cell_candidates = geometry.compute_collisions_points(bb_tree, p_point)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, p_point)
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])
            points_on_proc.append(p_point[0])
        
        if len(points_on_proc) > 0:
            # Find pressure DOF at the point
            p_dof = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
            if len(p_dof) > 0:
                p_bc_val = fem.Constant(domain, ScalarType(1.0))  # cos(0)*cos(0) = 1
                p_bc = fem.dirichletbc(p_bc_val, p_dof[:1], W.sub(1))
                bcs = [bc, p_bc]
            else:
                bcs = [bc]
        else:
            bcs = [bc]
    else:
        bcs = [bc]
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_monitor": False
        },
        petsc_options_prefix="stokes_"
    )
    w_sol = problem.solve()
    (u_sol, p_sol) = w_sol.split()
    
    # Evaluate on 100x100 grid
    nx, ny = 100, 100
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate velocity
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute velocity magnitude
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }