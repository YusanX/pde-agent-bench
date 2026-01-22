import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _interpolate_to_grid(u_h, nx: int, ny: int) -> np.ndarray:
    """
    Interpolate finite element function u_h to a regular nx x ny grid on [0,1]x[0,1].
    Works in parallel; returns the full grid on rank 0, zeros elsewhere.
    """
    domain = u_h.function_space.mesh
    comm = domain.comm
    rank = comm.rank

    # Generate grid points on [0,1]x[0,1]
    xs = np.linspace(0.0, 1.0, nx, endpoint=True)
    ys = np.linspace(0.0, 1.0, ny, endpoint=True)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()

    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    # Compute candidate cells and actual collisions
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    # For each point, pick first colliding cell (if any) local to this rank
    points_on_proc = []
    cells_on_proc = []
    local_indices = []
    for i in range(points.shape[1]):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(cells[0])
            local_indices.append(i)

    # Evaluate u_h at local points
    local_values = np.zeros((points.shape[1],), dtype=np.float64)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc, dtype=np.float64)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts, cls).flatten()
        local_values[local_indices] = vals

    # Sum over ranks (only one rank contributes to a given point)
    global_values = np.zeros_like(local_values)
    comm.Allreduce(local_values, global_values, op=MPI.SUM)

    # Reshape to (nx, ny)
    u_grid = global_values.reshape((nx, ny))

    # Ensure all ranks return the same array
    return u_grid


def solve(case_spec: dict) -> dict:
    """
    Solve -Δu = f in [0,1]^2 with u = sin(4πx)sin(4πy) on ∂Ω.

    Returns:
        dict with:
        - "u": numpy array of shape (nx, ny) on [0,1]x[0,1]
        - "solver_info": metadata dict
    """
    comm = MPI.COMM_WORLD

    # Parameters (tuned for accuracy vs. cost)
    mesh_resolution = case_spec.get("mesh_resolution", 80)
    element_degree = case_spec.get("element_degree", 1)
    ksp_type = case_spec.get("ksp_type", "cg")
    pc_type = case_spec.get("pc_type", "hypre")
    rtol = case_spec.get("rtol", 1.0e-10)

    # Create mesh
    domain = mesh.create_unit_square(
        comm,
        mesh_resolution,
        mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Exact solution and source term via UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    # For kappa = 1: -Δu = f, and Δu_exact = -32π^2 u_exact
    f_expr = 32.0 * ufl.pi**2 * u_exact_expr

    # Trial/test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Variational forms
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Dirichlet boundary conditions (u = u_exact on all boundaries)
    def boundary_all(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    boundary_dofs = fem.locate_dofs_geometrical(V, boundary_all)
    u_D = fem.Function(V)
    u_D_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_D.interpolate(u_D_expr)
    bc = fem.dirichletbc(u_D, boundary_dofs)

    # Assemble and solve linear system using high-level LinearProblem
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="poisson_hf_",
    )
    u_h = problem.solve()

    # Interpolate to uniform grid expected by evaluator
    nx, ny = 50, 50
    u_grid = _interpolate_to_grid(u_h, nx, ny)

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