import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _create_mesh(mesh_resolution: int):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm,
        nx=mesh_resolution,
        ny=mesh_resolution,
        cell_type=mesh.CellType.quadrilateral,
    )
    return domain


def _solve_poisson(domain, element_degree: int, ksp_type: str, pc_type: str, rtol: float):
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType(1.0))
    kappa = fem.Constant(domain, ScalarType(1.0))

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Homogeneous Dirichlet on whole boundary: u = 0
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

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
    return uh


def _sample_on_grid(u, nx: int, ny: int):
    mesh_obj = u.function_space.mesh
    comm = mesh_obj.comm

    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Build bounding box tree and find cells with these points
    bb_tree = geometry.bb_tree(mesh_obj, mesh_obj.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(mesh_obj, cell_candidates, points.T)

    values = np.full((nx * ny,), np.nan, dtype=np.float64)

    pts_local = []
    cells_local = []
    map_local = []

    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points.T[i])
            cells_local.append(links[0])
            map_local.append(i)

    if len(pts_local) > 0:
        pts_local = np.array(pts_local, dtype=np.float64)
        cells_local = np.array(cells_local, dtype=np.int32)
        vals_local = u.eval(pts_local, cells_local)
        values[map_local] = vals_local.reshape(-1)

    # Gather values on rank 0; we want a complete grid there
    if comm.size > 1:
        all_values = comm.allreduce(values, op=MPI.SUM)
    else:
        all_values = values

    if comm.rank == 0:
        return all_values.reshape((nx, ny))
    else:
        return None


def solve(case_spec: dict) -> dict:
    # Select parameters aiming to meet accuracy/time requirements
    mesh_resolution = 40
    element_degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8

    domain = _create_mesh(mesh_resolution)
    uh = _solve_poisson(domain, element_degree, ksp_type, pc_type, rtol)

    nx = ny = 50
    u_grid = _sample_on_grid(uh, nx, ny)

    # Broadcast u_grid from rank 0 so all ranks return same data
    comm = domain.comm
    if comm.rank == 0:
        shape = np.array(u_grid.shape, dtype=np.int32)
    else:
        shape = np.empty(2, dtype=np.int32)
    comm.Bcast(shape, root=0)

    if comm.rank != 0:
        u_grid = np.empty(shape, dtype=np.float64)
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
    out = solve({})
    if MPI.COMM_WORLD.rank == 0:
        print("u_grid shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])