import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -∇·(κ ∇u) = f in Ω=(0,1)^2 with u = 0 on ∂Ω,
    where κ = 1 and f = exp(-200*((x-0.25)^2 + (y-0.75)^2)).

    Returns:
        dict with:
            "u": numpy array of shape (nx, ny) with solution on 50x50 grid
            "solver_info": dict with keys:
                mesh_resolution, element_degree, ksp_type, pc_type, rtol
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # -------------------------------------------------------------------------
    # Discretization parameters (chosen for balance between accuracy and speed)
    # -------------------------------------------------------------------------
    mesh_resolution = 40  # number of cells in each direction
    element_degree = 1

    # -------------------------------------------------------------------------
    # Mesh and function space
    # -------------------------------------------------------------------------
    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # -------------------------------------------------------------------------
    # Boundary conditions: homogeneous Dirichlet on all boundaries
    # -------------------------------------------------------------------------
    def boundary_all(x):
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        )

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # -------------------------------------------------------------------------
    # Variational problem
    # -------------------------------------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    f_expr_ufl = ufl.exp(-200.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.75) ** 2))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr_ufl * v * ufl.dx

    # -------------------------------------------------------------------------
    # Solve linear system using high-level LinearProblem interface
    # -------------------------------------------------------------------------
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8

    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="pdebench_",
    )
    uh = problem.solve()

    # -------------------------------------------------------------------------
    # Interpolate solution onto a 50x50 uniform grid over [0,1]x[0,1]
    # -------------------------------------------------------------------------
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xv, yv = np.meshgrid(xs, ys, indexing="ij")  # shape (nx, ny)

    # Prepare points for evaluation: shape (gdim, N)
    pts = np.zeros((domain.geometry.dim, nx * ny), dtype=np.float64)
    pts[0, :] = xv.ravel()
    pts[1, :] = yv.ravel()

    # Build bounding box tree and find cells containing points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    # Map from global point index to those actually on this process
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full(pts.shape[1], np.nan, dtype=np.float64)
    if points_on_proc:
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        cells_on_proc = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(points_on_proc, cells_on_proc)
        values[eval_map] = vals.reshape(-1)

    # In parallel, gather values to root and construct full grid only on rank 0.
    # However, evaluator likely runs in serial; to be safe, we perform an allreduce
    # taking the maximum over processes (NaNs will be overwritten).
    values_global = values.copy()
    comm.Allreduce(PETSc.COMM_SELF.tompi4py().bcast, None)  # dummy to ensure PETSc init

    # Use MPI allreduce with custom handling via numpy: we can use np.nanmax but MPI
    # does not know NaN; instead, replace NaNs with very negative, then max.
    local_buf = values_global.copy()
    nan_mask = np.isnan(local_buf)
    local_buf[nan_mask] = -1e300
    global_buf = np.zeros_like(local_buf)
    comm.Allreduce(local_buf, global_buf, op=MPI.MAX)
    # Convert back: points that never had a valid value remain very negative; set to 0
    global_buf[global_buf < -1e299] = 0.0

    u_grid = global_buf.reshape((nx, ny))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    # Simple manual test
    out = solve({})
    if MPI.COMM_WORLD.rank == 0:
        print("u_grid shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])