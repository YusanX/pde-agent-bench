import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _interpolate_to_grid(u, nx: int, ny: int):
    """
    Interpolate dolfinx Function u onto a uniform (nx, ny) grid on [0,1]x[0,1].
    Returns local numpy array of shape (nx, ny) on rank 0, None on others.
    """
    comm = u.function_space.mesh.comm
    rank = comm.rank

    # Create grid points (cell centers)
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    domain = u.function_space.mesh
    tdim = domain.topology.dim

    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, tdim)

    # Find candidate cells for all points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    # Collect points that lie on this process
    pts_proc = []
    cells_proc = []
    map_proc = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            pts_proc.append(points[:, i])
            cells_proc.append(cells_i[0])
            map_proc.append(i)

    # Evaluate u on local points
    local_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    if len(pts_proc) > 0:
        pts_proc_arr = np.array(pts_proc, dtype=np.float64)
        cells_proc_arr = np.array(cells_proc, dtype=np.int32)
        vals = u.eval(pts_proc_arr, cells_proc_arr)
        local_vals[map_proc] = vals.ravel()

    # Gather to rank 0
    if rank == 0:
        recvbuf = np.empty((comm.size, local_vals.size), dtype=np.float64)
    else:
        recvbuf = None
    comm.Gather(local_vals, recvbuf, root=0)

    if rank == 0:
        # Combine with precedence to non-NaN entries
        combined = np.full(local_vals.size, np.nan, dtype=np.float64)
        for r in range(comm.size):
            vals_r = recvbuf[r]
            mask = ~np.isnan(vals_r)
            combined[mask] = vals_r[mask]
        u_grid = combined.reshape((nx, ny))
        return u_grid
    else:
        return None


def solve(case_spec: dict) -> dict:
    """
    Solve -Δu = f in [0,1]^2 with u = g on boundary,
    where exact solution is u = sin(pi x) sin(pi y).
    """
    comm = MPI.COMM_WORLD

    # Parameters (chosen to satisfy accuracy/time constraints)
    mesh_resolution = 40            # number of elements per direction
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8

    # Mesh
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Exact solution and derived quantities
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = 2.0 * ufl.pi**2 * u_exact_expr  # -Δu = 2π^2 sin(πx) sin(πy)

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Dirichlet BC: u = u_exact on boundary
    u_D = fem.Function(V)
    u_D.interpolate(
        fem.Expression(u_exact_expr, V.element.interpolation_points)
    )

    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(u_D, boundary_dofs)

    # Assemble and solve linear system
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()

    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Interpolate solution to 50x50 grid for evaluator
    nx = case_spec.get("nx", 50)
    ny = case_spec.get("ny", 50)
    u_grid = _interpolate_to_grid(uh, nx, ny)

    # Broadcast u_grid to all ranks so solve() return is consistent
    if comm.rank == 0:
        shape = np.array(u_grid.shape, dtype=np.int32)
    else:
        shape = np.zeros(2, dtype=np.int32)
    comm.Bcast(shape, root=0)

    if comm.rank != 0:
        u_grid = np.empty((shape[0], shape[1]), dtype=np.float64)
    comm.Bcast(u_grid, root=0)

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
        print("u_grid shape:", result["u"].shape)
        print("solver_info:", result["solver_info"])