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
    degree_u = 2
    degree_p = 1
    ksp_type = "minres"
    pc_type = "jacobi"
    rtol = 1e-8
    pressure_fixing = True
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.quadrilateral)
    
    # Function spaces (Taylor-Hood)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    W = V * Q
    
    # Trial and test functions
    w = ufl.TrialFunction(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.as_vector([ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
                            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])])
    p_exact = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term
    nu = 1.0
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Bilinear and linear forms
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.inner(p, ufl.div(v)) + ufl.inner(ufl.div(u), q)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    u_D = fem.Function(V)
    u_D.interpolate(lambda x: np.vstack((np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
                                        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))))
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, dofs_u, W.sub(0))
    bcs = [bc]
    
    # Handle pressure nullspace
    if pressure_fixing:
        # Pin pressure at a point to fix the constant
        p_D = fem.Function(Q)
        p_D.x.array[:] = 0.0
        
        # Find a single point to pin (e.g., corner)
        def pin_point(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
        
        pin_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), pin_point)
        if len(pin_dofs) > 0:
            bc_p = fem.dirichletbc(p_D, pin_dofs[:1], W.sub(1))
            bcs.append(bc_p)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-10,
            "ksp_max_it": 1000
        },
        petsc_options_prefix="stokes_"
    )
    w_sol = problem.solve()
    u_sol, p_sol = w_sol.split()
    
    # Create evaluation grid
    nx, ny = 100, 100
    x_eval = np.linspace(0, 1, nx)
    y_eval = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0] = X.flatten()
    points[1] = Y.flatten()
    
    # Evaluate velocity magnitude
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
    
    u_magnitude = np.linalg.norm(u_values, axis=1).reshape((nx, ny))
    
    return {
        "u": u_magnitude,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": {"velocity": degree_u, "pressure": degree_p},
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol
        }
    }