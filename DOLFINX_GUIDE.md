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

# ✅ In dolfinx 0.10.0, NonlinearProblem embeds SNES — call .solve() directly.
#    Do NOT pass NonlinearProblem to nls.petsc.NewtonSolver (incompatible interfaces).
problem = petsc.NonlinearProblem(F, u_sol, bcs=[bc],
                                  petsc_options_prefix="nonlin_",
                                  petsc_options={"snes_rtol": 1e-6,
                                                 "snes_type": "newtonls"})
u_sol = problem.solve()
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

> **⚠️ dolfinx 0.10.0 Breaking Change — Mixed Spaces**
>
> `W = V * Q` (multiplying two `FunctionSpace` objects) is **legacy FEniCS syntax and does NOT work in dolfinx 0.10.0**.
> It raises `TypeError: unsupported operand type(s) for *: 'FunctionSpace' and 'FunctionSpace'`.
>
> You **must** build mixed elements via `basix.ufl.mixed_element` as shown below.

Use a stable mixed pair (e.g., Taylor–Hood \(P2/P1\)):

```python
from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl

msh = mesh.create_unit_square(MPI.COMM_WORLD, 48, 48, cell_type=mesh.CellType.triangle)

# Correct dolfinx 0.10.0 API for mixed spaces:
vel_el  = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(msh.geometry.dim,))
pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

V, _ = W.sub(0).collapse()  # velocity subspace — needed for BCs and probing
Q, _ = W.sub(1).collapse()  # pressure subspace — needed for pressure pinning

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

### 10.4 Nonlinear Solve with dolfinx.fem.petsc.NonlinearProblem

In dolfinx 0.10.0, `fem.petsc.NonlinearProblem` **embeds its own SNES solver** and exposes a `.solve()` method directly. Do **not** pass a `NonlinearProblem` to `nls.petsc.NewtonSolver` — they have incompatible interfaces and will crash at runtime with `AttributeError: 'NonlinearProblem' object has no attribute 'a'`.

```python
from dolfinx.fem import petsc
from petsc4py import PETSc

# Compute Jacobian symbolically (optional but recommended for convergence)
J = ufl.derivative(F, w)

# Pass SNES/KSP options directly via petsc_options dict
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-10,
    "snes_max_it": 30,
    "ksp_type": "gmres",
    "pc_type": "lu",
}

problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                  petsc_options_prefix="ns_",
                                  petsc_options=petsc_options)

# ✅ CORRECT: call solve() on the problem directly (SNES is embedded)
w_h = problem.solve()
w.x.scatter_forward()
```

Common "it diverges" fixes:
- Use a better initial guess (Stokes → NS).

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

For **pure Dirichlet velocity BCs** (velocity prescribed on the entire boundary), pressure is
only determined up to an additive constant. Without fixing this, the system is singular.

**Symptoms:** KSP divergence, PETSc warning `"Detected zero pivot"`, or pressure solution
drifting to huge values.

**Fix: Pin pressure to zero at one corner point (recommended)**

This is the most robust and portable approach. Pin the single pressure DOF nearest to (0, 0)
to enforce a gauge condition \(p(0,0) = 0\):

```python
# Assumes W is a mixed space (see Section 10.1), bcs already contains velocity BCs.

Q, _ = W.sub(1).collapse()   # collapse to get the standalone pressure subspace

# Locate the pressure DOF at the origin corner
p_dofs = fem.locate_dofs_geometrical(
    (W.sub(1), Q),
    lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
)

if len(p_dofs) > 0:
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    bcs.append(bc_p)   # add to your existing velocity BCs list
```

> **Why `locate_dofs_geometrical` is acceptable here (not a bug):**
> We deliberately want *one specific interior DOF* (the corner pressure node), so the
> geometrical selector `isclose(x,0) & isclose(y,0)` correctly targets a single point.
> This is different from velocity BCs where using `np.ones(...)` would wrongly select
> all DOFs including interior ones (see Section 11.2).

**Complete minimal Stokes example with pressure pinning:**

```python
from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx.fem.petsc import LinearProblem
import ufl, numpy as np

msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
gdim = msh.geometry.dim

vel_el  = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

nu = 0.1
f  = fem.Constant(msh, np.zeros(gdim))
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

a = (2*nu*ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))*ufl.dx
     - p*ufl.div(v)*ufl.dx
     + ufl.div(u)*q*ufl.dx)
L = ufl.inner(f, v)*ufl.dx

# Velocity BC: no-slip on all walls (topological — correct approach)
fdim = msh.topology.dim - 1
wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
u0 = fem.Function(V); u0.x.array[:] = 0.0
bc_u = fem.dirichletbc(u0,
                       fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets),
                       W.sub(0))

# Pressure pin: fix p(0,0) = 0 (geometrical — correct for single-point constraint)
p_dofs = fem.locate_dofs_geometrical(
    (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
)
p0 = fem.Function(Q); p0.x.array[:] = 0.0
bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

w_h = LinearProblem(a, L, bcs=[bc_u, bc_p],
                    petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
                    petsc_options_prefix="stokes_").solve()
u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
```

### 10.7 Debug Checklist for Nonlinear PDEs

- **BC sanity**: Are your Dirichlet BCs over/under-constraining the system?
- **Scaling**: Very small/large `nu`, forcing amplitude, or domain scaling can destroy conditioning.
- **Mesh/degree**: For NS, prefer stable mixed pairs; avoid equal-order \(P1/P1\) without stabilization.
- **Residual form**: Make sure the convection term is written consistently (and is not accidentally zero).
- **Solver logs**: Turn on PETSc logging to see what actually fails (KSP vs Newton vs assembly).

### 10.8 Critical dolfinx 0.10.0 API Changes for NS/Stokes (Read Before Writing Code)

> **⚠️ These are the key API breaking changes in dolfinx 0.10.0 for NS/Stokes solvers.**

**Error 1 — `NonlinearProblem` requires `petsc_options_prefix`**

```python
# ❌ WRONG: Missing petsc_options_prefix → TypeError at runtime
problem = petsc.NonlinearProblem(F, w, bcs=bcs)

# ✅ CORRECT
problem = petsc.NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns_")
```

**Error 2 — `Function.vector` does not exist; use `Function.x`**

```python
w = fem.Function(W)

# ❌ WRONG: AttributeError: 'Function' object has no attribute 'vector'
w.vector.set(0.0)
petsc.set_bc(w.vector, bcs)

# ✅ CORRECT
w.x.array[:] = 0.0
petsc.set_bc(w.x.petsc_vec, bcs)
w.x.scatter_forward()
```

**Error 3 — `fem.petsc.NonlinearProblem` + `nls.petsc.NewtonSolver` are incompatible interfaces**

`dolfinx.fem.petsc.NonlinearProblem` (SNES-based, has `.solve()`) and `dolfinx.nls.petsc.NewtonSolver` (DOLFINx's own Newton, expects `problem.a` UFL form) are **two different solver stacks**. Mixing them causes `AttributeError: 'NonlinearProblem' object has no attribute 'a'` inside dolfinx's own `NewtonSolver.__init__`.

```python
# ❌ WRONG — crashes inside dolfinx's NewtonSolver.__init__
#            "AttributeError: 'NonlinearProblem' object has no attribute 'a'"
from dolfinx.nls.petsc import NewtonSolver as PETScNewtonSolver
problem = petsc.NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="ns_")
solver = PETScNewtonSolver(msh.comm, problem)   # ← crashes here

# ✅ CORRECT — use NonlinearProblem.solve() directly (SNES is embedded inside)
problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                  petsc_options_prefix="ns_",
                                  petsc_options={"snes_type": "newtonls", ...})
w_h = problem.solve()
```

**Error 4 — `ufl.grad()` on a pure Python/UFL constant fails with "Cannot determine geometric dimension"**

This happens when the pressure field is zero and you write `ufl.grad(ufl.as_ufl(0.0))`:

```python
# ❌ WRONG: ufl.as_ufl(0.0) has no mesh context → ValueError in ufl.grad
pex = ufl.as_ufl(0.0)
grad_p = ufl.grad(pex)   # raises ValueError

# ✅ CORRECT: attach the constant to the mesh via SpatialCoordinate
x = ufl.SpatialCoordinate(msh)
pex = 0.0 * x[0]                            # zero with mesh context
grad_p = ufl.grad(pex)                       # works fine

# ✅ ALSO CORRECT: use fem.Constant
pex_const = fem.Constant(msh, PETSc.ScalarType(0.0))
# (cannot call ufl.grad on a scalar Constant directly, but can embed in vector:)
grad_p = ufl.as_vector([0.0 * x[0], 0.0 * x[0]])   # explicit zero gradient
```

---

## 11. Vector-Valued PDEs: Linear Elasticity Quick Guide

This section covers **linear elasticity** and other **vector-valued** PDEs where the unknown `u` is a vector field (displacement, velocity without pressure, etc.).

### 11.1 Function Space for Vector-Valued Unknowns

```python
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import numpy as np

msh = mesh.create_unit_square(MPI.COMM_WORLD, 40, 40, cell_type=mesh.CellType.triangle)

# Vector Lagrange space: shape=(gdim,) makes each node carry gdim components
gdim = msh.geometry.dim
V = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))   # P2 vector — recommended for elasticity
# V = fem.functionspace(msh, ("Lagrange", 1, (gdim,))) # P1 vector — may suffer volumetric locking
#                                                        #   near nu → 0.5 (near-incompressible)
```

> **⚠️ Near-Incompressible Materials (nu > 0.4)**
>
> P1 (linear) elements exhibit **volumetric locking** when `nu` is close to `0.5`, producing
> artificially large errors regardless of mesh size. Always use **P2 or higher** for these cases,
> or use a mixed displacement-pressure formulation.

### 11.2 Dirichlet BCs for Vector Spaces (Correct Approach)

> **Critical**: `fem.locate_dofs_geometrical(V, lambda x: np.ones(...))` selects **all DOFs**
> (including interior nodes), effectively prescribing the solution everywhere and bypassing
> the FEM solve. Always use the **topological** method shown below.

**Case A — Full Dirichlet BC on all boundaries (e.g., prescribed nonzero displacement)**

```python
fdim = msh.topology.dim - 1

# Step 1: locate boundary FACETS (topology-based — only touches the boundary)
boundary_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
)

# Step 2: locate DOFs on those facets
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# Step 3: create a Function to hold the BC value
u_bc = fem.Function(V)

# Option A: constant zero displacement
u_bc.x.array[:] = 0.0

# Option B: non-uniform BC from a given boundary expression
import ufl
x_coord = ufl.SpatialCoordinate(msh)
# Example: use whatever expression appears in case_spec["bc"]["dirichlet"]["value"]
bc_expr = ufl.as_vector([
    ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1]),
    ufl.sin(ufl.pi * x_coord[0]) * ufl.cos(ufl.pi * x_coord[1]),
])
u_bc.interpolate(
    fem.Expression(bc_expr, V.element.interpolation_points())
)

bc = fem.dirichletbc(u_bc, boundary_dofs)
```

**Case B — Partial Dirichlet BC on specific boundaries**

```python
# Bottom boundary (y = 0): fix vertical displacement to zero
bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
dofs_bottom = fem.locate_dofs_topological(V, fdim, bottom_facets)
u_bottom = fem.Function(V)
u_bottom.x.array[:] = 0.0
bc_bottom = fem.dirichletbc(u_bottom, dofs_bottom)

# Right boundary (x = 1): prescribed horizontal displacement = 0.01
right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
dofs_right = fem.locate_dofs_topological(V, fdim, right_facets)
u_right = fem.Function(V)
u_right.interpolate(lambda x: np.vstack([np.full(x.shape[1], 0.01),
                                          np.zeros(x.shape[1])]))
bc_right = fem.dirichletbc(u_right, dofs_right)

bcs = [bc_bottom, bc_right]
```

### 11.3 Weak Form for Linear Elasticity

```python
# Material parameters
E, nu = 1.0, 0.3
mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(msh)
f = ufl.as_vector([...])   # body force vector, shape = (gdim,)

a = ufl.inner(sigma(u), eps(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
```

### 11.4 Output: Displacement Magnitude

The benchmark evaluates the **displacement magnitude** `||u|| = sqrt(u·u)`, NOT individual components.
Make sure your output script computes and saves the magnitude on the required grid:

```python
import numpy as np

# After solving: u_sol is a fem.Function on V
# Sample on a regular grid and compute magnitude
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

def sample_magnitude_on_grid(u_sol, nx=50, ny=50, bbox=(0,1,0,1)):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = bb_tree(msh, msh.topology.dim)
    cells = []
    points_on_proc = []
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, cell_candidates, pts)
    for i, pt in enumerate(pts):
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pt)
            cells.append(colliding.links(i)[0])

    u_vals = u_sol.eval(np.array(points_on_proc), cells)  # shape (N, gdim)
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)
    return magnitude

magnitude_grid = sample_magnitude_on_grid(u_sol)
np.savez("output.npz", data=magnitude_grid)
```
---

## 12. Complex-domain meshes (L-shape, annulus, gear, circle, etc.)

For non-rectangular domains use **`pygmsh`** + **`meshio`** + **`dolfinx.io.XDMFFile`** 

```python
import os
import pygmsh
import meshio
from mpi4py import MPI
from dolfinx.io import XDMFFile

def build_mesh(char_length: float):
    with pygmsh.occ.Geometry() as geom:
        import gmsh
        gmsh.option.setNumber("General.Verbosity", 0)
        geom.characteristic_length_max = char_length

        # --- describe your domain here, e.g.: ---
        # polygon:  geom.add_polygon([[x,y,0], ...])
        # disk:     geom.add_disk([cx,cy,0], r)
        # annulus:  outer = geom.add_disk(...); inner = geom.add_disk(...)
        #           geom.boolean_difference(outer, inner)
        # union:    geom.boolean_union([shape1, shape2])

        mesh_data = geom.generate_mesh()

    # strip Z coordinate, keep only triangles
    pts = mesh_data.points[:, :2]
    out = meshio.Mesh(points=pts, cells={"triangle": mesh_data.cells_dict["triangle"]})
    fname = f"_tmp_mesh_{os.getpid()}"
    meshio.write(f"{fname}.xdmf", out)

    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xf:
        msh = xf.read_mesh(name="Grid")

    for ext in [".xdmf", ".h5"]:
        if os.path.exists(fname + ext):
            os.remove(fname + ext)
    return msh
```

After meshing, BCs work identically to rectangular meshes via `mesh.locate_entities_boundary` + `fem.locate_dofs_topological`.

| Domain type | gmsh `char_length` ≈ | submesh fallback `res` |
|---|---|---|
| L-shape, T-junction | 1/resolution | 80–120 |
| Circle, sector | 1/resolution | 100–150 |
| Annulus, eccentric annulus | 1/resolution | — (use gmsh only) |
| Gear, star, multi-hole | 1/resolution | — (use gmsh only) |

