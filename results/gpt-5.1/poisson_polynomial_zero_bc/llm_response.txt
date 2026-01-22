import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa * grad(u)) = f in [0,1]x[0,1] with u=0 on boundary.
    Manufactured exact solution: u = x*(1-x)*y*(1-y), kappa = 1.
    Returns solution sampled on a 50x50 grid (local part on each rank).
    """

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # -----------------------
    # Discretization parameters
    # -----------------------
    mesh_resolution = 32          # number of cells per direction
    element_degree = 2            # polynomial degree
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-8

    # -----------------------
    # Mesh and function space
    # -----------------------
    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # -----------------------
    # Dirichlet boundary condition (u=0 on entire boundary)
    # -----------------------
    def boundary_marker(x):
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        )

    dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    # -----------------------
    # Variational problem
    # -----------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Manufactured exact solution and source term
    u_exact_expr = x[0] * (1 - x[0]) * x[1] * (1 - x[1])

    # Compute Laplacian of u_exact: Δu_exact
    grad_u = ufl.grad(u_exact_expr)
    laplace_u = ufl.div(grad_u)
    # PDE: -Δu = f  ⇒ f = -Δu
    f_expr = -laplace_u

    # Define f as a UFL expression wrapped in fem.Expression for interpolation
    # but for the weak form, we can use f_expr directly:
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # -----------------------
    # Solve linear system
    # -----------------------
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="poisson_",
    )
    uh = problem.solve()

    # -----------------------
    # Sample solution on a 50x50 grid
    # -----------------------
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    # Create grid of points (3, N) with z=0
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Use geometry utilities to evaluate uh at these points
    from dolfinx import geometry

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    # Build local lists of points and cells where we can evaluate
    local_points = []
    local_cells = []
    local_indices = []

    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            local_points.append(points[:, i])
            local_cells.append(cells_i[0])
            local_indices.append(i)

    local_points = np.array(local_points, dtype=np.float64) if local_points else np.zeros((0, 3))
    local_cells = np.array(local_cells, dtype=np.int32) if local_cells else np.zeros((0,), dtype=np.int32)

    # Evaluate on local points
    local_values = np.zeros((0,), dtype=np.float64)
    if local_points.shape[0] > 0:
        vals = uh.eval(local_points, local_cells)
        local_values = vals.flatten()

    # Gather all values on rank 0
    # Each rank sends its (indices, values)
    send_counts = np.array([len(local_indices)], dtype=np.int32)
    recv_counts = np.empty(comm.size, dtype=np.int32)
    comm.Allgather(send_counts, recv_counts)

    send_indices = np.array(local_indices, dtype=np.int32)
    send_values = np.array(local_values, dtype=np.float64)

    # Displacements for gather
    displs_idx = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
    total_count = np.sum(recv_counts)

    all_indices = np.empty(total_count, dtype=np.int32)
    all_values = np.empty(total_count, dtype=np.float64)

    comm.Allgatherv(send_indices, [all_indices, recv_counts, displs_idx, MPI.INT])
    comm.Allgatherv(send_values, [all_values, recv_counts, displs_idx, MPI.DOUBLE])

    # Reconstruct global u_grid on all ranks (identical array)
    u_grid_flat = np.zeros(nx * ny, dtype=np.float64)
    u_grid_flat[all_indices] = all_values
    u_grid = u_grid_flat.reshape((nx, ny))

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
        u = result["u"]
        print("u_grid shape:", u.shape)
        print("solver_info:", result["solver_info"])