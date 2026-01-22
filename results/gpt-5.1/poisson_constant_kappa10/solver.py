import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa * grad(u)) = f in (0,1)x(0,1)
    with u = g on boundary, where the exact solution is
    u = sin(pi*x) * sin(2*pi*y), kappa = 10.
    """

    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType

    # -------------------------------
    # Discretization / solver choices
    # -------------------------------
    mesh_resolution = 64  # uniform in both directions
    element_degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-8

    # ----------------
    # Mesh and spaces
    # ----------------
    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # -----------------------------
    # Exact solution and RHS/source
    # -----------------------------
    x = ufl.SpatialCoordinate(domain)

    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = ScalarType(10.0)

    # Compute Laplacian of u_exact
    grad_uex = ufl.grad(u_exact_expr)
    laplace_uex = ufl.div(grad_uex)
    # PDE: -div(kappa grad u) = f  => f = -kappa * laplace(u_exact)
    f_expr = -kappa * laplace_uex

    # UFL trial/test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Coefficient (constant kappa)
    kappa_const = fem.Constant(domain, ScalarType(kappa))

    # Bilinear and linear forms
    a = kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # ----------------
    # Boundary conditions
    # ----------------
    # All boundaries: enforce Dirichlet with u_exact
    u_D = fem.Function(V)
    u_exact_interpolant = fem.Expression(
        u_exact_expr, V.element.interpolation_points
    )
    u_D.interpolate(u_exact_interpolant)

    # Use geometrical locator: all boundary points (x[0]==0/1 or x[1]==0/1)
    def boundary_all(x_):
        return np.logical_or.reduce(
            (
                np.isclose(x_[0], 0.0),
                np.isclose(x_[0], 1.0),
                np.isclose(x_[1], 0.0),
                np.isclose(x_[1], 1.0),
            )
        )

    bdofs = fem.locate_dofs_geometrical(V, boundary_all)
    bc = fem.dirichletbc(u_D, bdofs)

    # ----------------
    # Linear solve
    # ----------------
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

    # ----------------
    # Interpolate to uniform grid (50x50)
    # ----------------
    nx = ny = 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    # Create 3 x (nx*ny) array of points (z=0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Use geometry tools to evaluate uh at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(cells_i[0])
            eval_map.append(i)

    u_values = np.full(points.shape[1], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts, cells_arr)
        u_values[eval_map] = vals.flatten()

    # Gather values on root and broadcast assembled grid
    if comm.size > 1:
        # Root gathers
        all_u = None
        if comm.rank == 0:
            all_u = np.empty_like(u_values)
        comm.Reduce(
            u_values,
            all_u,
            op=MPI.SUM,
            root=0,
        )
        if comm.rank == 0:
            u_grid_root = all_u.reshape(nx, ny)
        else:
            u_grid_root = None
        u_grid_root = comm.bcast(u_grid_root, root=0)
        u_grid = u_grid_root
    else:
        u_grid = u_values.reshape(nx, ny)

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
        print("u.shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])