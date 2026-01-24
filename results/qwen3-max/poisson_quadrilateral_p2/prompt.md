# Task: Solve Poisson Equation

## Problem Description

-∇·(κ ∇u) = f   in Ω
  u = g           on ∂Ω

Elliptic boundary value problem.

**Case ID:** poisson_quadrilateral_p2

**Math Type:** elliptic

**Manufactured Solution:** u = sin(pi*x)*sin(pi*y)
(Source term f and boundary data are derived from this exact solution)

**Coefficients:**
- κ = 1.0

**Domain:** [0,1] × [0,1] (unit square)

**Output Requirements (handled by evaluator):**
- Evaluator will sample solution on a 50 × 50 uniform grid
- Output field: scalar

---

## Implementation Requirements

Write a Python module using **dolfinx** (FEniCSx) that exposes:

```python
def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": u_grid, numpy array with shape (nx, ny)
    - "solver_info": required fields:
        mesh_resolution, element_degree, ksp_type, pc_type, rtol
    """
```

Notes:
1. Do NOT write files (no solution.npz / meta.json).
2. Evaluator will time your solve() call and write outputs.
3. You decide mesh resolution, element degree, solver, etc., but must report them in solver_info.

**Agent-Selectable Parameters:**
- mesh_resolution: Control spatial discretization fidelity
- element_degree: Polynomial degree of the finite element space
- ksp_type: Krylov solver choice
- pc_type: Preconditioner choice

---

**Pass/Fail Criteria (single tier):**
- Accuracy: error ≤ 1.00e-06
- Time: wall_time_sec ≤ 1.962s

---

**Output only the complete, runnable Python code.** No explanations needed.


---

## DOLFINX 0.10.0 Guide

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

## 6. Solvers (dolfinx 0.10.0)

### A. Linear Problem (High-Level)

The easiest way to solve $a(u, v) = L(v)$.

```python
problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="pdebench_"
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

# 3. Create Vector (PETSc Vec) for RHS
# Note: dolfinx 0.10.0 expects an iterable of function spaces in create_vector
# You can pass [V] or L_form.function_spaces
b = petsc.create_vector(L_form.function_spaces)

# 4. Solver Setup
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Inside time loop:
#   - Update time-dependent constants/functions
#   - Assemble RHS
with b.localForm() as loc:
    loc.set(0)
petsc.assemble_vector(b, L_form)

#   - Apply Lifting (for non-zero Dirichlet BCs on RHS)
petsc.apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

#   - Apply BCs to RHS
petsc.set_bc(b, [bc])

#   - Solve
solver.solve(b, u_sol.x.petsc_vec)
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

    # Build per-point mapping (avoid boolean mask mismatch)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points_array.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

**Common error:** 
`IndexError: boolean index did not match indexed array`  
Cause: using `colliding_cells.array` with a boolean mask of different length.  
Fix: use `colliding_cells.links(i)` and build `points_on_proc/cells_on_proc` mapping as above.
```

## 9. Common Pitfalls & Checklist

1.  **Scalar Types**: Ensure constants match the PETSc scalar type.
    *   Bad: `fem.Constant(domain, 1.0)`
    *   Good: `fem.Constant(domain, PETSc.ScalarType(1.0))`
2.  **Ghost Updates**: When manually modifying vectors in parallel, always call `ghostUpdate()` or `scatter_forward()`.
3.  **Form Compilation**: Always wrap UFL expressions with `fem.form(...)` if you are not using `LinearProblem` / `NonlinearProblem` which do it internally.
4.  **Dimension Matching**: In `locate_dofs_topological`, ensure the entity dimension (`fdim`) matches the facets you found.
5.  **Interpolation**: `u.interpolate(f)` requires `f` to handle input shape `(3, N)` and return `(value_size, N)` or `(N,)` for scalars.
6.  **interpolation_points API (0.10.0)**: `V.element.interpolation_points` is a property, not a callable.
    *   Bad: `V.element.interpolation_points()`
    *   Good: `V.element.interpolation_points`
7.  **fem.Expression expects UFL, not Python callables**:
    *   Bad: `fem.Expression(lambda x: ..., V.element.interpolation_points)`
    *   Good: build a UFL expression and pass it, e.g.:
        `f_expr = 2*ufl.pi**2*ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])`
        then `fem.Expression(f_expr, V.element.interpolation_points)`
8.  **LinearProblem requires petsc_options_prefix (0.10.0)**:
    *   Bad: missing `petsc_options_prefix` argument
    *   Good: pass `petsc_options_prefix="your_prefix_"` to avoid runtime error



## 10. Nonlinear PDE Quick Guide (e.g., Steady Incompressible Navier–Stokes)

This section complements the linear examples above with a practical template for **nonlinear PDEs** in `dolfinx 0.10.0`, especially steady **Navier–Stokes** (NS), where you typically solve a nonlinear system using **Newton** or **Picard (fixed-point)** iterations.

### 10.1 Mixed Function Spaces (Taylor–Hood) and Unknown Splitting

For 2D steady incompressible NS on Ω:
\[
-\nu \Delta u + (u\cdot\nabla)u + \nabla p = f,\quad \nabla\cdot u = 0
\]

Use a stable mixed pair (e.g., Taylor–Hood \(P2/P1\)):

```python
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

msh = mesh.create_unit_square(MPI.COMM_WORLD, 48, 48, cell_type=mesh.CellType.triangle)

V = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))  # velocity
Q = fem.functionspace(msh, ("Lagrange", 1))                      # pressure
W = V * Q

w = fem.Function(W)          # current iterate/unknown
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
```

### 10.2 Weak Form (Residual) for Newton

Define the nonlinear residual \(F(w; v,q)=0\). A common steady NS residual is:

```python
nu = 0.1
x = ufl.SpatialCoordinate(msh)
f = ufl.as_vector((0.0, 0.0))  # body force

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, p):
    return 2.0 * nu * eps(u) - p * ufl.Identity(msh.geometry.dim)

F = (
    ufl.inner(sigma(u, p), eps(v)) * ufl.dx
    + ufl.inner(ufl.grad(u) * u, v) * ufl.dx      # (u·∇)u, written as grad(u)*u
    - ufl.inner(f, v) * ufl.dx
    + ufl.inner(ufl.div(u), q) * ufl.dx
)
```

Notes:
- For the convection term, `ufl.grad(u) * u` is the standard compact form (matrix-vector product).
- You can swap in alternative forms (skew-symmetric, rotational) for stability; keep it consistent with your discretization goals.

### 10.3 Dirichlet BCs and Initial Guess (Critical)

Nonlinear solves are sensitive to BCs and initial guess.

```python
# Example: no-slip everywhere (replace with your problem’s BCs)
import numpy as np
u0 = fem.Function(V)
u0.interpolate(lambda X: np.zeros((msh.geometry.dim, X.shape[1])))

fdim = msh.topology.dim - 1
facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))
bcs = [bc_u]

# Initial guess: Stokes solve or u=0 often works for easy cases; harder flows need better guesses
w.x.array[:] = 0.0
```

Best practice:
- **Start from Stokes** (drop convection) to initialize `w`, then switch on convection and run Newton.
- Or do continuation in viscosity / Reynolds number.

### 10.4 Newton Solve with dolfinx.nls.petsc.NewtonSolver

In dolfinx 0.10.0, use `petsc.NonlinearProblem` + `nls.petsc.NewtonSolver`:

```python
import numpy as np
from dolfinx import nls
from dolfinx.fem import petsc
from petsc4py import PETSc

problem = petsc.NonlinearProblem(F, w, bcs=bcs)
solver = nls.petsc.NewtonSolver(msh.comm, problem)

# Convergence/robustness knobs
solver.convergence_criterion = "incremental"  # or "residual"
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 30

# Configure the linear solver used inside each Newton step (Jacobian solve)
ksp = solver.krylov_solver
ksp.setType(PETSc.KSP.Type.GMRES)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.ILU)          # for small/medium problems
# pc.setType(PETSc.PC.Type.LU)         # robust direct solve (often slower/memory-heavy)

n, converged = solver.solve(w)
assert converged
w.x.scatter_forward()
```

Common “it diverges” fixes:
- Use a better initial guess (Stokes → NS).
- Reduce nonlinearity with continuation (increase Reynolds gradually / decrease `nu` gradually).
- Use damping: try `solver.relaxation_parameter = 0.5` (or smaller) to stabilize Newton steps.
- Switch preconditioner (ILU ↔ Hypre ↔ LU) and adjust tolerances.

### 10.5 Picard (Fixed-Point) Iteration (Often More Robust Than Newton)

Picard linearizes convection by freezing the advecting velocity \(u_k\):
\[
(u_k\cdot\nabla)u_{k+1}
\]

Implementation pattern:
- Keep a `fem.Function` for `u_k`.
- Build a *linear* form with `u_k` inserted into convection term.
- Solve with `petsc.LinearProblem` inside an outer loop until \(\|u_{k+1}-u_k\|\) is small.

This is slower than Newton but frequently stabilizes hard cases; you can use Picard to “get close”, then switch to Newton.

### 10.6 Pressure Nullspace / Uniqueness (Important for Incompressible Flow)

For pure Dirichlet velocity BCs, pressure is determined only up to a constant. Symptoms:
- KSP stagnation or warnings about singular matrices.

Typical fixes:
- Impose a pressure gauge (e.g., set mean pressure to 0) or pin one DOF of `p`.
- Or provide a PETSc nullspace for the mixed operator (advanced).

Minimal practical approach (gauge):
- Add constraint “mean(p)=0” (requires extra handling), or
- Pin pressure at a point / on a boundary if it’s physically meaningful.

### 10.7 Debug Checklist for Nonlinear PDEs

- **BC sanity**: Are your Dirichlet BCs over/under-constraining the system?
- **Scaling**: Very small/large `nu`, forcing amplitude, or domain scaling can destroy conditioning.
- **Mesh/degree**: For NS, prefer stable mixed pairs; avoid equal-order \(P1/P1\) without stabilization.
- **Residual form**: Make sure the convection term is written consistently (and is not accidentally zero).
- **Solver logs**: Turn on PETSc logging to see what actually fails (KSP vs Newton vs assembly).
