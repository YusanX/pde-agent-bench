import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _build_problem(mesh_resolution: int = 40, element_degree: int = 1):
    comm = MPI.COMM_WORLD

    # Mesh
    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Trial/test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Coefficient kappa = 1.0
    kappa = fem.Constant(domain, ScalarType(1.0))

    # Source term f = sin(8*pi*x)*sin(8*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_ufl = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])

    # Variational forms
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx

    # Dirichlet BC: u = 0 on entire boundary (g=0)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(_x):
        # All facets on boundary of unit square
        return np.logical_or.reduce(
            (
                np.isclose(_x[0], 0.0),
                np.isclose(_x[0], 1.0),
                np.isclose(_x[1], 0.0),
                np.isclose(_x[1], 1.0),
            )
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    return domain, V, a, L, [bc]


def _solve_linear(domain, V, a, L, bcs, ksp_type="cg", pc_type="hypre", rtol=1e-8):
    # Assemble matrix and vector
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    # Solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()

    uh = fem.Function(V)

    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    # Solve
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    return uh, ksp


def _sample_on_grid(domain, uh, nx: int = 50, ny: int = 50):
    """
    Sample solution uh on a uniform nx x ny grid over [0,1]x[0,1].
    Returns numpy array of shape (nx, ny) on rank 0 (others return None).
    """
    comm = domain.comm
    rank = comm.rank

    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    # Compute collisions
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    # Map local evaluation points
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(cells_i[0])
            eval_map.append(i)

    local_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc, dtype=np.float64)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts, cls)
        local_vals[np.array(eval_map, dtype=np.int32)] = vals.reshape(-1)

    # Gather values to rank 0
    all_vals = None
    if rank == 0:
        all_vals = np.empty_like(local_vals)
    comm.Reduce(local_vals, all_vals, op=MPI.SUM, root=0)

    if rank == 0:
        u_grid = all_vals.reshape((nx, ny))
        return u_grid
    else:
        return None


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = f on unit square with u=0 on boundary,
    f = sin(8*pi*x)*sin(8*pi*y), kappa=1.
    Returns:
        {
            "u": u_grid (nx, ny) on rank 0, or None on other ranks,
            "solver_info": {
                mesh_resolution,
                element_degree,
                ksp_type,
                pc_type,
                rtol
            }
        }
    """
    # Select parameters (tuned for accuracy vs. time)
    mesh_resolution = int(case_spec.get("mesh_resolution", 40))
    element_degree = int(case_spec.get("element_degree", 1))
    ksp_type = case_spec.get("ksp_type", "cg")
    pc_type = case_spec.get("pc_type", "hypre")
    rtol = float(case_spec.get("rtol", 1e-8))

    domain, V, a, L, bcs = _build_problem(mesh_resolution, element_degree)
    uh, ksp = _solve_linear(domain, V, a, L, bcs, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)

    # Sample solution on 50x50 grid
    u_grid = _sample_on_grid(domain, uh, nx=50, ny=50)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp.getType(),  # actual type used
        "pc_type": ksp.getPC().getType(),
        "rtol": rtol,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    # Simple manual test
    out = solve({})
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        u = out["u"]
        print("u_grid shape:", u.shape)
        print("solver_info:", out["solver_info"])