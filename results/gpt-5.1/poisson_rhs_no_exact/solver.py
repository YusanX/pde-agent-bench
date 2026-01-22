import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa * grad u) = f on [0,1]^2 with u=0 on the boundary.
    f = sin(3*pi*x)*sin(2*pi*y), kappa = 0.5
    Returns:
        {
            "u": u_grid (nx, ny), sampled on 48x48 uniform grid,
            "solver_info": {
                mesh_resolution, element_degree, ksp_type, pc_type, rtol
            }
        }
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parameters (chosen for accuracy vs speed)
    mesh_resolution = 40
    element_degree = 2

    # ----------------------------------------------------------------
    # Mesh and function space
    # ----------------------------------------------------------------
    domain = mesh.create_unit_square(
        comm,
        nx=mesh_resolution,
        ny=mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # ----------------------------------------------------------------
    # Boundary conditions: u = 0 on entire boundary
    # ----------------------------------------------------------------
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_indicator(x):
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_indicator)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # ----------------------------------------------------------------
    # Variational problem
    # ----------------------------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = fem.Constant(domain, PETSc.ScalarType(0.5))

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Solve linear problem
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-8,
        },
        petsc_options_prefix="pdebench_",
    )
    uh = problem.solve()

    # ----------------------------------------------------------------
    # Sample solution on 48x48 uniform grid on [0,1]^2
    # ----------------------------------------------------------------
    nx = ny = 48
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)

    # Create grid points as (3, N) array (z=0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Build bounding box tree and find containing cells
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    # Map global point indices to local evaluation data
    eval_points = []
    eval_cells = []
    map_back = []
    for i in range(points.shape[1]):
        cell_indices = colliding_cells.links(i)
        if len(cell_indices) > 0:
            eval_points.append(points[:, i])
            eval_cells.append(cell_indices[0])
            map_back.append(i)

    u_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    if eval_points:
        eval_points_arr = np.array(eval_points, dtype=np.float64)
        eval_cells_arr = np.array(eval_cells, dtype=np.int32)
        values = uh.eval(eval_points_arr, eval_cells_arr).reshape(-1)
        u_vals[np.array(map_back, dtype=np.int64)] = values

    # Gather sampled values to rank 0
    # Each rank has same set of points; only some points have valid (non-nan) data.
    local_valid = np.where(~np.isnan(u_vals))[0]
    local_data = np.vstack((local_valid, u_vals[local_valid]))  # shape (2, n_local)

    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        assembled = np.full(points.shape[1], np.nan, dtype=np.float64)
        for data in all_data:
            if data.size == 0:
                continue
            idx = data[0].astype(np.int64)
            vals = data[1]
            assembled[idx] = vals
        u_grid = assembled.reshape((nx, ny))
    else:
        u_grid = None

    # Broadcast u_grid to all ranks so that every rank returns same dict
    u_grid = comm.bcast(u_grid, root=0)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-8,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    # Simple manual test
    result = solve({})
    if MPI.COMM_WORLD.rank == 0:
        print("u_grid shape:", result["u"].shape)
        print("solver_info:", result["solver_info"])