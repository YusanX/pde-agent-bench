import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = f on [0,1]^2 with u = g on boundary,
    manufactured solution u = sin(2*pi*x)*sin(pi*y), kappa = 1.
    Returns:
        {
            "u": u_grid (nx, ny),
            "solver_info": {
                mesh_resolution, element_degree, ksp_type, pc_type, rtol
            }
        }
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parameters (chosen for good accuracy vs cost)
    mesh_resolution = 64  # uniform in x,y
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    # Create unit square mesh
    domain = mesh.create_unit_square(
        comm,
        mesh_resolution,
        mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Manufactured exact solution and source term in UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute f = -Δu_exact
    # u_xx = -4π^2 sin(2πx) sin(πy)
    # u_yy = -π^2 sin(2πx) sin(πy)
    # Δu = u_xx + u_yy = -5π^2 sin(2πx) sin(πy)
    # f = -Δu = 5π^2 sin(2πx) sin(πy)
    f_expr = 5 * (ufl.pi ** 2) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Dirichlet boundary conditions: u = u_exact on entire boundary
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
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Interpolate exact solution to boundary function
    u_bc_func = fem.Function(V)
    # Build an Expression from the UFL field
    u_exact_interpolant = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_interpolant)
    bc = fem.dirichletbc(u_bc_func, bc_dofs)

    # Assemble and solve linear system with PETSc KSP
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

    # Interpolate solution onto 50x50 uniform grid on root
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    map_local_to_global = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(cells_i[0])
            map_local_to_global.append(i)

    values_all = None
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals_local = uh.eval(pts, cells_arr).real.reshape(-1)

        # Gather to root
        counts = comm.gather(len(vals_local), root=0)
        if rank == 0:
            displs = np.cumsum([0] + counts[:-1])
            values_all = np.empty(points.shape[1], dtype=np.float64)
            recvbuf = np.empty(sum(counts), dtype=np.float64)
        else:
            displs = None
            recvbuf = None

        comm.Gatherv(sendbuf=vals_local, recvbuf=(recvbuf, counts, displs, MPI.DOUBLE), root=0)

        if rank == 0:
            # We also need mapping indices from all ranks
            local_indices = np.array(map_local_to_global, dtype=np.int32)
            counts_idx = comm.gather(len(local_indices), root=0)
            if counts_idx is None:
                counts_idx = [len(local_indices)]
            displs_idx = np.cumsum([0] + counts_idx[:-1])
            recv_idx = np.empty(sum(counts_idx), dtype=np.int32)
            comm.Gatherv(sendbuf=local_indices, recvbuf=(recv_idx, counts_idx, displs_idx, MPI.INT), root=0)

            # Place values in correct positions
            for j, gidx in enumerate(recv_idx):
                values_all[gidx] = recvbuf[j]
    else:
        # Still need to participate in collectives
        comm.gather(0, root=0)
        comm.gather(0, root=0)
        if rank == 0:
            values_all = np.empty(points.shape[1], dtype=np.float64)

    # Second gather for indices for ranks that had points
    if len(points_on_proc) > 0:
        local_indices = np.array(map_local_to_global, dtype=np.int32)
        comm.Gatherv(sendbuf=local_indices, recvbuf=None, root=0)
    else:
        comm.Gatherv(sendbuf=np.array([], dtype=np.int32), recvbuf=None, root=0)

    if rank == 0:
        u_grid = values_all.reshape((nx, ny))
    else:
        u_grid = None

    # Broadcast u_grid to all ranks so function is collective-safe
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
    out = solve({})
    if MPI.COMM_WORLD.rank == 0:
        print("u shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])