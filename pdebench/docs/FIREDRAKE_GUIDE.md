# Firedrake API Reference Guide

This guide covers the key Firedrake APIs you need to solve PDEs for this benchmark.
It is intended for agents working inside **this repository**, so project-specific
interface requirements and pitfalls take precedence over generic Firedrake habits.
Firedrake version targeted: **0.13+** (2024+).

---

## 1. Imports

```python
from firedrake import (
    # Mesh
    UnitSquareMesh,
    # Spaces
    FunctionSpace, VectorFunctionSpace,
    # Functions & forms
    Function, TrialFunction, TestFunction,
    TrialFunctions, TestFunctions,
    SpatialCoordinate, Constant,
    # Calculus
    inner, grad, div, dot, nabla_grad, sym, tr, Identity,
    sqrt, sin, cos, exp, ln,
    as_vector, as_ufl,
    dx,
    # BCs & solve
    DirichletBC, solve,
    # Mixed problems
    split, MixedVectorSpaceBasis, VectorSpaceBasis,
    # Nonlinear problems
    NonlinearVariationalProblem, NonlinearVariationalSolver,
    derivative,
    # Interpolation
    interpolate,
    # NOTE: do NOT use VertexOnlyMesh for grid sampling — use Function.at(coords) instead
)
import numpy as np
```

---

## 2. Mesh

```python
# Unit square with nx×ny triangular cells
mesh = UnitSquareMesh(nx, ny)

# With quadrilateral cells
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)
```

---

## 3. Function Spaces

```python
# Scalar CG (Lagrange) space, degree p
V = FunctionSpace(mesh, "CG", p)

# Vector space (for elasticity, velocity)
V = VectorFunctionSpace(mesh, "CG", p)

# Taylor-Hood mixed space for Stokes/NS: P2-P1
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q                        # MixedFunctionSpace
```

---

## 4. Functions and Forms

```python
x = SpatialCoordinate(mesh)      # spatial coordinates as UFL vector; x[0]=x, x[1]=y

u = TrialFunction(V)
v = TestFunction(V)

# Build bilinear and linear forms
a = inner(grad(u), grad(v)) * dx
L = inner(Constant(1.0), v) * dx   # or simply 1.0 * v * dx for scalar problems

# Solve (modifies uh in-place)
uh = Function(V)
solve(a == L, uh, bcs=[...], solver_parameters={...})
```

---

## 5. Boundary Conditions

```python
# Homogeneous Dirichlet on entire boundary
bc = DirichletBC(V, Constant(0.0), "on_boundary")

# From an exact Function
u_exact = Function(V)
u_exact.interpolate(sin(pi * x[0]) * sin(pi * x[1]))
bc = DirichletBC(V, u_exact, "on_boundary")

# Specific boundary: UnitSquareMesh labels are 1(bottom), 2(right), 3(top), 4(left)
bc_left = DirichletBC(V, Constant(0.0), 4)     # left boundary (x=0)
bc_right = DirichletBC(V, Constant(1.0), 2)   # right boundary (x=1)
```

**Project-specific rule:** for vector-valued Dirichlet data in mixed problems,
pass either:

- a Firedrake `Constant([...])`, or
- a mesh-bound `Function`

Do **not** pass a plain UFL vector like `as_vector([Constant(...), Constant(...)])`
directly to `DirichletBC`. In this repository that pattern can trigger
`This integral is missing an integration domain.`

The same caution applies to **vector-valued source terms / RHS data**:
for expressions like `["0.0", "0.0"]` or `["sin(pi*x)", "0.0"]`, prefer
interpolating into a mesh-bound `Function` in the vector space instead of
assembling a domain-free UFL vector by hand.

---

## 6. Solver Parameters

Firedrake uses PETSc solver parameters directly:

```python
# CG with algebraic multigrid (good for elliptic)
solver_parameters = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-10,
}

# GMRES+ILU (sometimes useful, but not universally robust in this repository)
solver_parameters = {
    "ksp_type": "gmres",
    "pc_type": "ilu",
    "ksp_rtol": 1e-10,
}

# Direct solver (only when the Firedrake environment actually has the backend)
solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
```

**Project-specific note:** benchmark case configs may come from DOLFINx-oriented
setups. Do not assume `mumps` is available in the current Firedrake environment.
When portability matters, prefer iterative methods first and add safe fallbacks.

**Additional repository-specific notes:**

- For Helmholtz, `gmres + ilu` can fail on high-frequency or strongly localized
  source cases. If it diverges, try `gmres + hypre`, and then `preonly + lu`.
- For linear Stokes reference solves, `minres + hypre` is a good first attempt
  but not always sufficient on large open-boundary problems; be ready to retry
  with `gmres + hypre` or a `fieldsplit` Schur-complement setup.

---

## 7. Mixed Problems (Stokes / Navier-Stokes)

```python
from firedrake import (
    Function, TrialFunctions, TestFunctions, split,
    MixedVectorSpaceBasis, VectorSpaceBasis, solve, DirichletBC,
    as_vector, Constant, inner, grad, div, dx,
)

# Setup mixed space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Forms
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
a = (nu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * dx
L = inner(f, v) * dx

# BC on velocity component
# Safe options:
#   1) vector Constant
bc = DirichletBC(W.sub(0), Constant([0.0, 0.0]), "on_boundary")
#
#   2) or interpolate into a mesh-bound Function first
# bc_fn = Function(W.sub(0).collapse())
# bc_fn.interpolate(as_vector([x[1] * (1 - x[1]), 0.0]))
# bc = DirichletBC(W.sub(0), bc_fn, 4)

# Pressure handling:
# - If pressure is only determined up to a constant, provide a nullspace.
# - If you impose a pressure point constraint / pressure BC, do NOT also use the
#   constant-pressure nullspace in the same way.
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

w_h = Function(W)
solve(a == L, w_h, bcs=[bc], nullspace=nullspace, solver_parameters={
    "ksp_type": "minres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "ksp_rtol": 1e-10,
})

u_h, p_h = w_h.subfunctions   # extract components
```

**Project-specific note:** in this repository, large open-boundary Stokes
reference solves may diverge with `minres + hypre`. If a reference solve fails in
the linear stage, retry with a safer fallback such as `gmres + hypre`, and if
needed a `fieldsplit` Schur-complement setup.

---

## 8. Nonlinear Problems (Navier-Stokes, Allen-Cahn, etc.)

```python
from firedrake import (
    Function, TestFunctions, split, derivative,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
    inner, grad, div, dot, dx,
)

w = Function(W)
u, p = split(w)
v, q = TestFunctions(W)

# Residual form F(w; v) = 0
F = (
    nu * inner(grad(u), grad(v)) * dx
    + inner(dot(grad(u), u), v) * dx   # (u·∇)u convective term
    - p * div(v) * dx
    - q * div(u) * dx
    - inner(f, v) * dx
)

bcs = [DirichletBC(W.sub(0), ..., "on_boundary")]
problem = NonlinearVariationalProblem(F, w, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters={
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "snes_max_it": 50,
    "ksp_type": "gmres",
    "pc_type": "hypre",
})
solver.solve()
u_h, p_h = w.subfunctions
```

For incompressible mixed problems, remember that pressure still needs explicit
handling:

- use a pressure nullspace if pressure is defined only up to a constant, or
- apply a pressure point constraint / pressure BC if that is the intended model

Do not omit this just because the residual form looks complete.

**Critical repository-specific note for Navier-Stokes:**

Do **not** assume Newton from a zero initial guess will be robust enough.
In this repository, many Navier-Stokes cases only become reliable when the
initial guess is chosen deliberately:

- `init = "exact"`: for manufactured cases, initialize with exact velocity, and
  if available also exact pressure.
- `init = "stokes"`: solve a linear Stokes problem first and use it as a warm start.
- `init = "continuation"`: do not treat this as equivalent to zero init; it
  should map to a robust warm-start strategy, not a placeholder.

If Newton fails after entering nonlinear iterations, retrying **only** the
linear solver is often not enough. Also try nonlinear fallbacks such as:

- changing `snes_linesearch_type` from `bt` to `basic` or `l2`
- increasing `snes_max_it`
- improving the warm start rather than tightening tolerances blindly

---

## 9. Time Stepping (Backward Euler)

```python
dt = Constant(0.01)

u_prev = Function(V)
u_prev.interpolate(Constant(0.0))   # initial condition

u = TrialFunction(V)
v = TestFunction(V)
kappa = Constant(1.0)

a = (u * v + dt * inner(kappa * grad(u), grad(v))) * dx
uh = Function(V)

for step in range(n_steps):
    t = (step + 1) * float(dt)
    f_t = ...   # source at time t (UFL expression)
    L = (u_prev * v + dt * f_t * v) * dx
    bc = DirichletBC(V, ..., "on_boundary")
    solve(a == L, uh, bcs=[bc], solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})
    u_prev.assign(uh)
```

---

## 10. Sampling the Solution on a Grid

The evaluator will call your `solve()` and internally resample your output array.
Use **`Function.at(coords)`** to sample on a uniform grid — this is the correct API
because it guarantees values are returned in the **same order** as the input coordinates.

> ⚠️ **Do NOT use `VertexOnlyMesh` + `dat.data_ro`** for grid sampling.
> `VertexOnlyMesh` reorders points internally (even in serial mode), so
> `dat.data_ro` is in an arbitrary internal ordering that does **not** match
> the input coordinates — your grid will be completely scrambled.

```python
import numpy as np

def sample_scalar(u_h, nx, ny, bbox=(0.0, 1.0, 0.0, 1.0)):
    """Sample scalar field on (ny × nx) grid, indexing='xy'.
    
    Returns array where result[i, j] = u at (x=x_lin[j], y=y_lin[i]).
    """
    xmin, xmax, ymin, ymax = bbox
    x_lin = np.linspace(xmin, xmax, nx)
    y_lin = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")   # shape (ny, nx)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape (ny*nx, 2)

    # Function.at returns values in the same order as coords — guaranteed
    values = np.array(u_h.at(coords))   # shape (ny*nx,)
    return values.reshape(ny, nx)


def sample_vector_magnitude(u_h, nx, ny, bbox=(0.0, 1.0, 0.0, 1.0)):
    """Sample ||u|| of vector field on (ny × nx) grid."""
    xmin, xmax, ymin, ymax = bbox
    x_lin = np.linspace(xmin, xmax, nx)
    y_lin = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    vals = np.array(u_h.at(coords))   # shape (ny*nx, gdim)
    mag = np.linalg.norm(vals, axis=1)
    return mag.reshape(ny, nx)
```

**Important:** The grid shape must be `(ny, nx)` with `indexing="xy"` to match the evaluator's convention.

---

## 11. Return Format

In **this repository**, `solve(case_spec)` must return an `OracleResult`
dataclass, not a plain dict.

```python
from pdebench.oracle._types import OracleResult

return OracleResult(
    baseline_error=float(baseline_error),
    baseline_time=float(baseline_time),
    reference=u_grid,        # np.ndarray, shape (ny, nx)
    solver_info={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "degree": 2,
        # optional transient / nonlinear metadata is fine too
    },
    num_dofs=V.dof_count,
)
```

For this benchmark, `reference` is the sampled output field used by evaluation.
It is typically:

- scalar field values on the output grid, or
- vector magnitude on the output grid

Do not invent a different return schema unless the surrounding oracle interface
is also updated.

---

## 12. Key API Differences vs DOLFINx

| Concept | DOLFINx (FEniCSx) | Firedrake |
|---------|-------------------|-----------|
| Mesh (unit square) | `mesh.create_unit_square(...)` | `UnitSquareMesh(nx, ny)` |
| Function space | `fem.FunctionSpace(mesh, ("CG", p))` | `FunctionSpace(mesh, "CG", p)` |
| Dirichlet BC | `fem.dirichletbc(value, dofs)` | `DirichletBC(V, value, "on_boundary")` |
| Linear solve | `LinearProblem(a, L, bcs).solve()` | `solve(a == L, uh, bcs=bcs, ...)` |
| Nonlinear solve | `NonlinearProblem(F, w, J)` + `NewtonSolver` | `NonlinearVariationalProblem(F, w, bcs)` + `NonlinearVariationalSolver` |
| Spatial coords | `ufl.SpatialCoordinate(msh)` | `SpatialCoordinate(mesh)` |
| Point evaluation | `bb_tree` + `compute_collisions_points` | `Function.at(coords)` (use this, NOT VertexOnlyMesh) |
| Mixed subfunctions | `w.sub(0).collapse()` | `w.subfunctions` |
| DOF count | `V.dofmap.index_map.size_global` | `V.dof_count` |
