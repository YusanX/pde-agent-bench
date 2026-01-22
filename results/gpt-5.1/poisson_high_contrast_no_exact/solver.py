import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = 1 in [0,1]^2 with u=0 on boundary.
    kappa(x,y) = 1 + 1000*exp(-100*(x-0.5)^2)
    Returns:
        {
          "u": u_grid (nx, ny),
          "solver_info": {
              "mesh_resolution": int,
              "element_degree": int,
              "ksp_type": str,
              "pc_type": str,
              "rtol": float
          }
        }
    """
    comm = MPI.COMM_WORLD

    # Parameters (chosen to meet accuracy/time targets)
    mesh_resolution = 48  # uniform in both directions
    element_degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8

    # -------------------------------------------------------------------------
    # Mesh and function space
    # -------------------------------------------------------------------------
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # -------------------------------------------------------------------------
    # Boundary conditions: u = 0 on entire boundary
    # -------------------------------------------------------------------------
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    # -------------------------------------------------------------------------
    # Coefficient kappa(x, y)
    #   kappa = 1 + 1000*exp(-100*(x-0.5)^2)
    # -------------------------------------------------------------------------
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5) ** 2)

    # -------------------------------------------------------------------------
    # Variational problem
    # -------------------------------------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_src = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_src, v) * ufl.dx

    # -------------------------------------------------------------------------
    # Linear solve using high-level wrapper
    # -------------------------------------------------------------------------
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="pdebench_",
    )
    uh = problem.solve()

    # -------------------------------------------------------------------------
    # Interpolate solution onto 45x45 uniform grid in [0,1]^2
    # -------------------------------------------------------------------------
    nx = ny = 45
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.vstack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))

    from dolfinx import geometry

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    map_back = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(cells_i[0])
            map_back.append(i)

    u_grid = np.full(points.shape[1], np.nan, dtype=float)
    if points_on_proc:
        pts = np.array(points_on_proc, dtype=float)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        values = uh.eval(pts.T, cells_arr)
        u_grid[map_back] = values.reshape(-1)

    u_grid = u_grid.reshape((nx, ny))

    # For parallel runs: gather to root and broadcast full array
    if comm.size > 1:
        # gather to root
        local = u_grid
        gathered = comm.gather(local, root=0)
        if comm.rank == 0:
            # all ranks should have identical NaN locations; just take the first
            global_grid = gathered[0]
        else:
            global_grid = None
        global_grid = comm.bcast(global_grid, root=0)
        u_grid = global_grid

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    out = solve({})
    # Simple sanity print on rank 0
    if MPI.COMM_WORLD.rank == 0:
        print("u shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])