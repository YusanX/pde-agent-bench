# FEniCSx (dolfinx v0.10.0) Quick Reference Guide

This guide provides the correct syntax and best practices for `dolfinx v0.10.0`. It is designed to help LLMs and developers avoid common API errors and deprecated patterns.

## 1. Imports & Setup

```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry, nls, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

# Define the scalar type (usually float64 or complex128)
ScalarType = PETSc.ScalarType
```

## 2. Mesh Generation

Use `mesh.create_*` functions. Note that `comm` is always the first argument.

```python
comm = MPI.COMM_WORLD

# 2D Unit Square (Triangles)
domain = mesh.create_unit_square(comm, nx=32, ny=32, cell_type=mesh.CellType.triangle)

# 2D Rectangle (Quadrilaterals)
p0 = np.array([0.0, 0.0])
p1 = np.array([2.0, 1.0])
domain_rect = mesh.create_rectangle(comm, [p0, p1], [32, 16], cell_type=mesh.CellType.quadrilateral)

# 3D Unit Cube (Tetrahedrons)
domain_3d = mesh.create_unit_cube(comm, nx=10, ny=10, nz=10, cell_type=mesh.CellType.tetrahedron)
```

## 3. Function Space Definitions

**Crucial:** Prefer `fem.functionspace` (lowercase) over `fem.FunctionSpace`.

```python
# Scalar Function Space (e.g., Pressure, Temperature) - P1 elements
V = fem.functionspace(domain, ("Lagrange", 1))

# Vector Function Space (e.g., Velocity)
# Note the shape tuple: (domain.geometry.dim,)
V_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# Tensor Function Space (e.g., Stress)
V_tensor = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, domain.geometry.dim)))

# Discontinuous Galerkin (DG)
V_dg = fem.functionspace(domain, ("DG", 0))
```

## 4. Boundary Conditions (Dirichlet)

### Method A: Topological (Recommended for labeled boundaries)

Requires finding facets based on dimensionality (`fdim = tdim - 1`).

```python
tdim = domain.topology.dim
fdim = tdim - 1

def boundary_marker(x):
    return np.isclose(x[0], 0.0) # Left boundary

# 1. Locate Facets
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)

# 2. Locate DOFs
# specific to the function space V
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# 3. Create BC
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.full_like(x[0], 0.0)) # Zero BC
bc = fem.dirichletbc(u_bc, dofs)
```

### Method B: Geometrical (Simpler for simple coordinates)

```python
def boundary_marker_geo(x):
    return np.isclose(x[0], 1.0) # Right boundary

dofs_geo = fem.locate_dofs_geometrical(V, boundary_marker_geo)
# For constant values, you can pass the value and the V directly if creating a Constant isn't needed
bc_geo = fem.dirichletbc(PETSc.ScalarType(1.0), dofs_geo, V) 
```

## 5. Variational Problem (Weak Form)

Use `ufl` for symbolic math.

```python
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Source term
f = fem.Constant(domain, ScalarType(1.0))

# Variational form: -div(grad(u)) = f
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
```

## 6. Solvers

### A. Linear Problem (High-Level)

The easiest way to solve $a(u, v) = L(v)$.

```python
problem = petsc.LinearProblem(
    a, L, bcs=[bc], 
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
u_sol = problem.solve()
```

### B. Linear Problem (Manual Assembly - Advanced)

Use this for time-dependent loops to avoid re-initializing KSP solvers.

```python
# 1. Create forms
a_form = fem.form(a)
L_form = fem.form(L)

# 2. Assemble Matrix (once if grid/coefficients don't change)
A = petsc.assemble_matrix(a_form, bcs=[bc])
A.assemble()

# 3. Create Vector
b = fem.Function(V)

# 4. Solver Setup
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Inside time loop:
#   - Update time-dependent constants/functions
#   - Assemble RHS
with b.vector.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b.vector, L_form)

#   - Apply Lifting (for non-zero Dirichlet BCs on RHS)
petsc.apply_lifting(b.vector, [a_form], bcs=[[bc]])
b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

#   - Apply BCs to RHS
petsc.set_bc(b.vector, [bc])

#   - Solve
solver.solve(b.vector, u_sol.vector)
u_sol.x.scatter_forward()
```

### C. Nonlinear Problem

For problems like $F(u, v) = 0$.

```python
# Define nonlinear residual F
F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx 
# + nonlinear terms like u**2 ...

problem = petsc.NonlinearProblem(F, u_sol, bcs=[bc])
solver = nls.petsc.NewtonSolver(domain.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6

n, converged = solver.solve(u_sol)
```

## 7. File I/O

**VTXWriter** (ADIOS2) is recommended for modern visualization (Paraview).

```python
# Writing
with io.VTXWriter(domain.comm, "output.bp", [u_sol], engine="BP4") as vtx:
    vtx.write(0.0) # Write at time t=0.0

# Legacy XDMF
with io.XDMFFile(domain.comm, "output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_sol)
```

## 8. Point Evaluation (Interpolation/Probe)

**Do NOT use `BoundingBoxTree` directly.** Use `geometry` utilities.

```python
from dolfinx import geometry

def probe_points(u_func, points_array):
    """
    u_func: dolfinx.fem.Function
    points_array: shape (3, N) numpy array
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)
    
    # Evaluate
    # u_values shape: (num_points, value_size)
    u_values = u_func.eval(points_array.T, colliding_cells.array)
    return u_values
```

## 9. Common Pitfalls & Checklist

1.  **Scalar Types**: Ensure constants match the PETSc scalar type.
    *   Bad: `fem.Constant(domain, 1.0)`
    *   Good: `fem.Constant(domain, PETSc.ScalarType(1.0))`
2.  **Ghost Updates**: When manually modifying vectors in parallel, always call `ghostUpdate()` or `scatter_forward()`.
3.  **Form Compilation**: Always wrap UFL expressions with `fem.form(...)` if you are not using `LinearProblem` / `NonlinearProblem` which do it internally.
4.  **Dimension Matching**: In `locate_dofs_topological`, ensure the entity dimension (`fdim`) matches the facets you found.
5.  **Interpolation**: `u.interpolate(f)` requires `f` to handle input shape `(3, N)` and return `(value_size, N)` or `(N,)` for scalars.

