import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve -∇·(κ ∇u) = f in Ω with u = g on ∂Ω,
    Ω = [0,1]x[0,1], κ = 2.0,
    manufactured solution u = exp(x) * cos(2*pi*y).

    Returns:
        dict with:
            "u": numpy array of shape (nx, ny) with solution on uniform grid
            "solver_info": meta data
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parameters (chosen to meet accuracy/time targets)
    mesh_resolution = 40  # uniform in both directions
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    # ------------------------------------------------------------------
    # Mesh (quadrilateral unit square)
    # ------------------------------------------------------------------
    domain = mesh.create_unit_square(
        comm,
        nx=mesh_resolution,
        ny=mesh_resolution,
        cell_type=mesh.CellType.quadrilateral,
    )

    # ------------------------------------------------------------------
    # Function space
    # ------------------------------------------------------------------
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # ------------------------------------------------------------------
    # Manufactured solution and data
    # ------------------------------------------------------------------
    x = ufl.SpatialCoordinate(domain)
    kappa = ScalarType(2.0)

    u_exact_expr = ufl.exp(x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact_expr)
    f_expr = -ufl.div(kappa * grad_u_exact)

    # Boundary condition: u = g = u_exact on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_indicator(xx):
        return (
            np.isclose(xx[0], 0.0)
            | np.isclose(xx[0], 1.0)
            | np.isclose(xx[1], 0.0)
            | np.isclose(xx[1], 1.0)
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_indicator)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc, bc_dofs)

    # ------------------------------------------------------------------
    # Variational problem
    # ------------------------------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Function(V)
    f_expr_interp = fem.Expression(f_expr, V.element.interpolation_points)
    f.interpolate(f_expr_interp)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # ------------------------------------------------------------------
    # Linear solve using high-level LinearProblem
    # ------------------------------------------------------------------
    petsc_options = {
        "ksp_type": ksp_type,
        "ksp_rtol": rtol,
        "pc_type": pc_type,
    }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="poisson_",
    )
    uh = problem.solve()

    # ------------------------------------------------------------------
    # Sample solution on a 50x50 uniform grid over [0,1]x[0,1]
    # ------------------------------------------------------------------
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    from dolfinx import geometry

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_vals = np.full(nx * ny, np.nan, dtype=np.float64)

    # Build per-point mapping to cells owned by this rank
    local_points = []
    local_cells = []
    local_indices = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            local_points.append(points[:, i])
            local_cells.append(cells_i[0])
            local_indices.append(i)

    if local_points:
        local_points_arr = np.array(local_points, dtype=np.float64)
        local_cells_arr = np.array(local_cells, dtype=np.int32)
        values = uh.eval(local_points_arr, local_cells_arr)
        u_vals[local_indices] = values.real.flatten()

    # Gather all values to rank 0 and construct full grid
    if comm.size > 1:
        u_vals_global = None
        if rank == 0:
            u_vals_global = np.empty_like(u_vals)
        comm.Reduce(u_vals, u_vals_global, op=MPI.SUM, root=0)
    else:
        u_vals_global = u_vals

    if rank == 0:
        u_grid = u_vals_global.reshape((nx, ny))
    else:
        u_grid = None

    # Broadcast u_grid so all ranks return same data
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
    out = solve({})
    # Simple check (only on rank 0)
    if MPI.COMM_WORLD.rank == 0:
        print("u grid shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])