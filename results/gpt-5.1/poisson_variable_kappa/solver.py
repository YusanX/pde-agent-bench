import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _eval_on_grid(u, nx: int, ny: int) -> np.ndarray:
    """
    Evaluate a scalar dolfinx.fem.Function u on a uniform [0,1]x[0,1] grid
    of shape (nx, ny). Returns numpy array with shape (nx, ny).
    """
    domain = u.function_space.mesh
    # Grid points: evaluator samples on a 55x55 uniform grid including endpoints
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    points_T = points.T

    # Build bounding box tree and find cells for each point
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_T.shape[0]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            points_on_proc.append(points_T[i])
            cells_on_proc.append(cells_i[0])
            eval_map.append(i)

    values = np.full((nx * ny,), np.nan, dtype=np.float64)
    if points_on_proc:
        pts = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts, cells_arr).reshape(-1)
        values[np.array(eval_map, dtype=int)] = np.real(vals)

    return values.reshape((nx, ny))


def solve(case_spec: dict) -> dict:
    """
    Solve -∇·(κ ∇u) = f in Ω = (0,1)^2 with u = g on ∂Ω,
    where u_exact = sin(2πx) sin(2πy),
          κ(x,y) = 1 + 0.5 sin(2πx) sin(2πy).

    Returns:
        {
            "u": u_grid (numpy array, shape (nx, ny)),
            "solver_info": {
                mesh_resolution, element_degree, ksp_type, pc_type, rtol
            }
        }
    """
    comm = MPI.COMM_WORLD

    # Parameters (chosen for accuracy vs. time trade-off)
    mesh_resolution = 40          # number of cells per direction
    element_degree = 2            # P2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Create unit square mesh
    domain = mesh.create_unit_square(
        comm,
        nx=mesh_resolution,
        ny=mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Exact solution and coefficient kappa as UFL expressions
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa_expr = 1.0 + 0.5 * u_exact_expr  # since same sin(2πx)sin(2πy)

    # Compute source term f = -div(kappa * grad u_exact)
    u_exact = u_exact_expr
    grad_u = ufl.grad(u_exact)
    kappa = kappa_expr
    f_expr = -ufl.div(kappa * grad_u)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear and linear forms
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Dirichlet BC: u = u_exact on entire boundary
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(xg):
        tol = 1e-14
        return (
            np.isclose(xg[0], 0.0, atol=tol)
            | np.isclose(xg[0], 1.0, atol=tol)
            | np.isclose(xg[1], 0.0, atol=tol)
            | np.isclose(xg[1], 1.0, atol=tol)
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Interpolate exact solution for BC
    u_bc_func = fem.Function(V)

    def u_exact_callable(xg):
        return np.sin(2.0 * np.pi * xg[0]) * np.sin(2.0 * np.pi * xg[1])

    u_bc_func.interpolate(u_exact_callable)
    bc = fem.dirichletbc(u_bc_func, dofs)

    # Solve linear system
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

    # Evaluate on 55x55 grid
    nx = 55
    ny = 55
    u_grid = _eval_on_grid(uh, nx, ny)

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
        print("u grid shape:", result["u"].shape)
        print("solver_info:", result["solver_info"])