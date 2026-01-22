import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _exact_solution_expr(x):
    return np.exp(6.0 * x[0]) * np.sin(np.pi * x[1])


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters (chosen to meet accuracy/time requirements)
    mesh_resolution = 80  # uniform in both directions
    element_degree = 1

    # Create unit square mesh
    domain = mesh.create_unit_square(
        comm,
        nx=mesh_resolution,
        ny=mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Coordinates for manufactured solution in UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.exp(6.0 * x[0]) * ufl.sin(ufl.pi * x[1])

    # Diffusion coefficient
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

    # Source term f = -div(kappa * grad u_exact)
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Weak form
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    # Dirichlet boundary condition: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_D = fem.Function(V)
    u_D.interpolate(_exact_solution_expr)
    bc = fem.dirichletbc(u_D, bdofs)

    # Linear solve using high-level LinearProblem
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

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

    # Interpolate solution on a 50x50 uniform grid over [0,1]x[0,1]
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Use geometry utilities to evaluate uh at given points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(cells_i[0])
            eval_map.append(i)

    u_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    if points_on_proc:
        pts_arr = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_vals[np.array(eval_map, dtype=np.int32)] = vals.real.flatten()

    u_grid = u_vals.reshape((nx, ny))

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