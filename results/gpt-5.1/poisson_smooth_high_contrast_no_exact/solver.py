import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = f in [0,1]^2 with u = 0 on boundary.
    Returns:
        {
          "u": numpy array of shape (nx, ny) with nodal values on a 50x50 grid,
          "solver_info": {
              "mesh_resolution": int,
              "element_degree": int,
              "ksp_type": str,
              "pc_type": str,
              "rtol": float,
          }
        }
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parameters (chosen to balance accuracy and speed)
    mesh_resolution = 80
    element_degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8

    # -------------------------------------------------------------------------
    # Mesh and function space
    # -------------------------------------------------------------------------
    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # -------------------------------------------------------------------------
    # Boundary conditions: homogeneous Dirichlet on entire boundary
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

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    dofs_boundary = fem.locate_dofs_geometrical(V, boundary_all)
    bc = fem.dirichletbc(u_bc, dofs_boundary)

    # -------------------------------------------------------------------------
    # Coefficient kappa(x) and source term f(x)
    # -------------------------------------------------------------------------
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1 + 50 * ufl.exp(-200 * (x[0] - 0.5) ** 2)

    f_expr = 1 + ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])

    # Trial/test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # -------------------------------------------------------------------------
    # Assemble and solve linear system
    # -------------------------------------------------------------------------
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as blocal:
        blocal.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()

    uh = fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # -------------------------------------------------------------------------
    # Sample solution on a 50x50 uniform grid on [0,1]^2
    # -------------------------------------------------------------------------
    nx_sample = 50
    ny_sample = 50
    xs = np.linspace(0.0, 1.0, nx_sample)
    ys = np.linspace(0.0, 1.0, ny_sample)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx_sample * ny_sample), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Use geometry tools to find cells and evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    pts_on_proc = []
    cells_on_proc = []
    map_local_to_global = []

    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            pts_on_proc.append(points[:, i])
            cells_on_proc.append(cells_i[0])
            map_local_to_global.append(i)

    values = np.full(points.shape[1], np.nan, dtype=np.float64)
    if pts_on_proc:
        pts_np = np.array(pts_on_proc, dtype=np.float64)
        cells_np = np.array(cells_on_proc, dtype=np.int32)
        vals_loc = uh.eval(pts_np, cells_np)
        values[np.array(map_local_to_global, dtype=np.int64)] = vals_loc.flatten()

    # Gather to rank 0 and assemble final grid
    if comm.size > 1:
        # Each rank sends its local 'values'
        counts = comm.gather(len(values), root=0)
        if rank == 0:
            recvbuf = np.empty((comm.size, points.shape[1]), dtype=np.float64)
        else:
            recvbuf = None
        comm.Gather(values, recvbuf, root=0)

        if rank == 0:
            # Combine: for each point, take first non-NaN from ranks
            combined = np.full(points.shape[1], np.nan, dtype=np.float64)
            for r in range(comm.size):
                mask = ~np.isnan(recvbuf[r])
                combined[mask] = recvbuf[r, mask]
            u_grid = combined.reshape((nx_sample, ny_sample))
        else:
            u_grid = None
    else:
        u_grid = values.reshape((nx_sample, ny_sample))

    # Broadcast u_grid from root so all ranks return a consistent result
    u_grid = comm.bcast(u_grid, root=0)

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
    result = solve({})
    if MPI.COMM_WORLD.rank == 0:
        print("u shape:", result["u"].shape)
        print("solver_info:", result["solver_info"])