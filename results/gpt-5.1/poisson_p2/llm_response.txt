import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa*grad(u)) = f on [0,1]^2 with u = g on boundary.
    Manufactured solution: u = sin(pi*x)*sin(pi*y), kappa = 1.
    Returns:
        {
            "u": u_grid (nx, ny) on uniform [0,1]^2,
            "solver_info": {
                mesh_resolution, element_degree, ksp_type, pc_type, rtol
            }
        }
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parameters (chosen to satisfy accuracy/time requirements)
    mesh_resolution = 64
    element_degree = 2

    # Create mesh
    domain = mesh.create_unit_square(
        comm,
        nx=mesh_resolution,
        ny=mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Coordinates
    x = ufl.SpatialCoordinate(domain)

    # Manufactured exact solution and data
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # For kappa=1: -Δu = f => f = 2*pi^2*sin(pi x)*sin(pi y)
    f_expr = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Source term as Expression -> interpolate to Function for use in form
    f_func = fem.Function(V)
    f_expr_fe = fem.Expression(f_expr, V.element.interpolation_points)
    f_func.interpolate(f_expr_fe)

    # Variational problem: ∫ grad u · grad v dx = ∫ f v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx

    # Dirichlet boundary: u = u_exact on entire boundary
    def boundary_all(x_b):
        return (
            np.isclose(x_b[0], 0.0)
            | np.isclose(x_b[0], 1.0)
            | np.isclose(x_b[1], 0.0)
            | np.isclose(x_b[1], 1.0)
        )

    u_D = fem.Function(V)
    u_D_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_D.interpolate(u_D_expr)

    dofs_bc = fem.locate_dofs_geometrical(V, boundary_all)
    bc = fem.dirichletbc(u_D, dofs_bc)

    # Assemble and solve linear system manually for efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()

    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Interpolate solution onto 45x45 uniform grid on [0,1]^2
    nx = ny = 45
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Use bb_tree and geometry utilities for evaluation
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

    u_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    if points_on_proc:
        pts = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts, cells_arr)
        u_vals[np.array(eval_map, dtype=np.int32)] = vals.reshape(-1)

    # Gather to rank 0 if running in parallel
    if comm.size > 1:
        # Each rank sends its local array; reduce by taking max over non-NaNs
        sendbuf = u_vals.copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
        u_vals = recvbuf

    u_grid = u_vals.reshape((nx, ny))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp.getType(),
        "pc_type": pc.getType(),
        "rtol": ksp.getTolerances()[0],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    out = solve({})
    # Simple sanity check: print min/max on rank 0
    if MPI.COMM_WORLD.rank == 0:
        print("u_grid min/max:", out["u"].min(), out["u"].max())
        print("solver_info:", out["solver_info"])