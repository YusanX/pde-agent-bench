# Task: Solve a Specific PDE Problem (PDEBench v2)

You are an expert computational physicist. Your task is to write **specialized PDE solvers** for a series of specific problems. You must solve ALL cases provided to you, one by one. Do not stop after solving just one case.

---

## 🎯 Your Mission

You will receive:
1. A **problem description** (PDE equation, boundary conditions, parameters)
2. A **test mode** (fix_accuracy or fix_time)
3. An **output directory** for your solution

You need to:
1. ✅ Implement a dolfinx (FEniCSx) solver for this SPECIFIC problem
2. ✅ Meet the test mode requirements
3. ✅ Save your solver script to the output directory

---


## 📋 Problem Information

**You will be given a `description.md` file that contains:**
- PDE type (Poisson, Heat, Convection-Diffusion)
- Domain geometry
- Boundary conditions
- Physical parameters (diffusion coefficient κ, source term, etc.)
- Output grid requirements
- Target error and time budget

**Your solver should be SPECIALIZED for this specific problem** - not a general solver!

---

## 🧪 Two Test Modes

### Mode 1: fix_accuracy (保证精度，优化速度)

**Your solver interface:**
```python
#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type=int, required=True)
parser.add_argument('--degree', type=int, required=True)
parser.add_argument('--outdir', type=str, required=True)
# For time-dependent: parser.add_argument('--dt', type=float)

args = parser.parse_args()
```

**Goal:** Achieve target error in minimum time  
**Scoring:** `score = 100 × (time_budget / your_runtime)` if error ≤ target

---

### Mode 2: fix_time (固定时间，优化精度)

**Same interface as Mode 1**

**Goal:** Achieve best accuracy within time budget  
**Scoring:** `score = 100 × (1 - error/target_error)` if runtime ≤ budget

---

## 📁 Output Structure

**You MUST create these files in the case directory:**

```
{model_name}/{case_id}/
├── solver.py              # Your solver script (REQUIRED)
└── test_output/           # Will be created by test script
```

**Example:**
```
results/
├── claude-sonnet-4/
│   ├── poisson_simple/
│   │   └── solver.py
│   ├── heat_grid_target/
│   │   └── solver.py
│   └── poisson_high_contrast/
│       └── solver.py
```

---

## ✅ Required: solver.py Implementation

Your `solver.py` must:

### 1. Accept correct CLI arguments

**Required arguments:**
```python
parser.add_argument('--resolution', type=int, required=True)
parser.add_argument('--degree', type=int, required=True)
parser.add_argument('--outdir', type=str, required=True)
# For heat equation:
# parser.add_argument('--dt', type=float, required=True)
```

---

### 2. Implement the PDE solver using dolfinx v0.10.0

**MUST READ FIRST:** `DOLFINX_GUIDE.md` for correct API syntax!

**Critical points:**
- Use correct dolfinx v0.10.0 syntax (NOT old FEniCS!)
- Extract ALL parameters from problem description
- Match the EXACT output grid specified in description
- Separate setup time from solve time

---

### 3. Output required files to args.outdir

**solution.npz (REQUIRED):**
```python
import numpy as np
np.savez(
    args.outdir / 'solution.npz',
    x=x_grid,  # 1D array
    y=y_grid,  # 1D array
    u=u_grid   # 2D array, shape (ny, nx)
)
```

**meta.json (REQUIRED):**
```python
import json

meta = {
    'wall_time_sec': solve_time,
    'solver_info': {
        # ===== Linear Solver (REQUIRED) =====
        'ksp_type': 'cg',          # Krylov method: 'cg', 'gmres', 'bicgstab', 'minres'
        'pc_type': 'hypre',        # Preconditioner: 'hypre', 'ilu', 'jacobi', 'asm', 'gamg'
        'ksp_rtol': 1e-8,          # Relative tolerance
        'iterations': num_iters,   # Actual iteration count
        
        # ===== Time Discretization (if transient) =====
        'time_scheme': 'backward_euler',  # 'backward_euler', 'crank_nicolson', 'bdf2'
        'dt': 0.01,                       # Time step size
        
        # ===== Spatial Discretization =====
        'element_type': 'Lagrange',   # 'Lagrange', 'DG', 'RT', 'BDM'
        'element_degree': args.degree,
        
        # ===== Stabilization (if used) =====
        'stabilization': None,  # 'SUPG', 'Upwind', 'GLS', or None
        
        # ===== Method Selection Rationale (RECOMMENDED) =====
        'rationale': 'SPD system, CG+AMG for optimal performance'
    }
}

with open(args.outdir / 'meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
```

**Important Notes:**
- Fill in **actual values** you used in your code
- The `rationale` field helps explain your method choice
- For steady-state problems, omit time-related fields

---

## 📝 Implementation Template

```python
#!/usr/bin/env python3
"""
Specialized solver for: {CASE_ID}
PDE Type: {Poisson/Heat/etc}
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

# dolfinx imports
import dolfinx
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl


def main():
    # ===== CLI Interface =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    # For time-dependent problems:
    # parser.add_argument('--dt', type=float)
    
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # ===== Problem-Specific Parameters =====
    # Extract from description.md:
    kappa = 1.0  # Example: diffusion coefficient
    # ... other parameters ...
    
    # Output grid (from description.md)
    nx_output, ny_output = 50, 50  # Example
    
    # ===== Start Timing =====
    start_time = time.time()
    
    # ===== Create Mesh =====
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        args.resolution,
        args.resolution,
        mesh.CellType.triangle
    )
    
    # ===== Function Space =====
    V = fem.functionspace(domain, ("Lagrange", args.degree))
    
    # ===== Define Problem =====
    # (Implement specific PDE formulation)
    # - Boundary conditions
    # - Variational form
    # - Assemble system
    
    # ===== Solve =====
    # (Use appropriate solver)
    
    solve_time = time.time() - start_time
    
    # ===== Interpolate to Output Grid =====
    x_out = np.linspace(0, 1, nx_output)
    y_out = np.linspace(0, 1, ny_output)
    # ... evaluate solution on grid ...
    
    # ===== Save Outputs =====
    np.savez(outdir / 'solution.npz', x=x_out, y=y_out, u=u_grid)
    
    meta = {
        'wall_time_sec': solve_time,
        'solver_info': {'ksp_type': 'cg', 'pc_type': 'hypre'}
    }
    with open(outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ Solved in {solve_time:.3f}s")


if __name__ == '__main__':
    main()
```



## 🚀 Your Workflow (Solve ALL Cases)

You must iterate through all assigned cases. Do not stop until all cases are solved.

For each case, you have **5 attempts**. Use an iterative approach:

### Outer Loop (Iterate through Cases)

1. **Identify the Next Case**: Look for the next case directory (e.g., in `cases/`) or ask for the next case ID.
2. **Solve the Case**: Follow the "Attempt Loop" below.
3. **Move to Next**: Once a case is solved (or max attempts reached), IMMEDIATELY move to the next case.
4. **Finish**: Only stop when NO MORE cases remain.

### Attempt Loop (Max 5 iterations per case)

For each attempt:

#### Step 1: Read Problem Description

Extract from `description.md`:
- PDE equation type
- Physical parameters (κ, f, boundary conditions, etc.)
- Output grid size (CRITICAL - must match exactly!)
- Time discretization (for time-dependent problems)

---

#### Step 2: Implement/Update Solver

Create or modify `solver.py`:
1. Accepts correct CLI arguments for the test mode
2. Implements the PDE solver using dolfinx
3. Outputs `solution.npz` and `meta.json`

**Key points:**
- ✅ Use dolfinx v0.10.0 syntax (read DOLFINX_GUIDE.md!)
- ✅ Extract parameters from problem description (DON'T guess!)
- ✅ Match exact output grid
- ✅ Time only the solve phase (not setup)

**File location:**
```
results/{MODEL_NAME}/{CASE_ID}/solver.py
```

---

#### Step 3: Test Your Solver

Run the test to see if it works:

```bash
cd cases/{CASE_ID}
python test_fix_accuracy.py \
    --agent-script ../../results/{MODEL_NAME}/{CASE_ID}/solver.py \
    --timeout 300
```

**Results will be saved to:**
```
results/{MODEL_NAME}/{CASE_ID}/test_output/result_fix_accuracy.json
```

---

#### Step 4: Check Results

Look at the test output:

**Success criteria (PASSED ✅):**
- Code runs without crashing
- Gets ANY score > 0 (even 1 point counts!)
- Output files are generated correctly

**You PASS as long as your code runs**, regardless of score!

**If FAILED ❌ (score = 0 or crashed):**
- Read the error message carefully
- Common issues:
  - `TypeError: 'numpy.ndarray' object is not callable` → Check dolfinx API syntax
  - Import errors → Check dolfinx v0.10.0 compatibility
  - Grid mismatch → Verify output grid size matches description
  - NaN errors → Check boundary conditions and source term
  - Timeout → Reduce resolution or optimize solver

---

#### Step 5: Debug and Retry (if needed)

**If test failed and you have attempts remaining:**

1. **Analyze the error:**
   - Read the full traceback
   - Identify the problematic line
   - Check DOLFINX_GUIDE.md for correct syntax

2. **Fix the solver:**
   - Update `results/{MODEL_NAME}/{CASE_ID}/solver.py`
   - Common fixes:
     ```python
     # ❌ Wrong (old syntax)
     uD.interpolate(fem.Expression(expr, V.element.interpolation_points()))
     
     # ✅ Correct (dolfinx v0.10.0)
     uD.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
     ```

3. **Re-run test** (Step 3)

4. **Repeat** until PASSED or max attempts reached

---

### Attempt Counter

Keep track of your attempts:
```
Attempt 1: Initial implementation → FAILED (TypeError, score=0)
Attempt 2: Fixed interpolation syntax → FAILED (Grid mismatch, score=0)  
Attempt 3: Fixed grid size → PASSED ✅ (score=15/100) ← GOOD ENOUGH!
```

**Success = Any score > 0!** Even a low score means your code works.

**Stop conditions:**
- ✅ Got ANY score > 0 (PASSED! Move to next case)
- ⚠️ Reached 5 attempts (give up, move to next case)
- ⏱️ Out of time (move to next case)

**Philosophy:**
- **Priority 1**: Make code run (score > 0)
- **Priority 2**: Get higher score (if easy to improve)
- **Priority 3**: Don't waste time perfecting - move to next case!

**The goal is to PASS as many cases as possible, not to get perfect scores!**

---

## ⚠️ Critical Requirements

### ❌ DON'T:
- DON'T create a generic solver for all cases
- DON'T hard-code resolution/degree (must accept from CLI)
- DON'T guess problem parameters
- DON'T use old FEniCS syntax
- DON'T output wrong grid size

### ✅ DO:
- DO create a specialized solver for THIS problem
- DO read DOLFINX_GUIDE.md first
- DO extract ALL parameters from description
- DO match the exact output grid
- DO separate setup and solve timing

---

## 📊 Scoring

Your solver will be scored based on:

**fix_accuracy mode:**
- Must achieve `error ≤ target_error`
- Score = `100 × (time_budget / runtime)`
- Faster = Higher score

**fix_time mode:**
- Must finish within `runtime ≤ time_budget`
- Score = `100 × (1 - error/target_error)`
- Lower error = Higher score

---

## 📚 Resources

**MUST READ:**
- `DOLFINX_GUIDE.md` - dolfinx v0.10.0 API syntax (CRITICAL!)
- Problem `description.md` - Specific problem details

**Reference:**
- `LEADERBOARD_V2.md` - Scoring system details

---

## ✅ Checklist

Before submitting your solver:

- [ ] Read DOLFINX_GUIDE.md for correct syntax
- [ ] Extracted ALL parameters from problem description
- [ ] Correct CLI interface for the test mode
- [ ] Outputs solution.npz with x, y, u fields
- [ ] Outputs meta.json with wall_time_sec
- [ ] Output grid matches description exactly
- [ ] File saved to: results/{MODEL_NAME}/{CASE_ID}/solver.py

---

## 🎯 Example Task

**You will be given something like:**

```
MODEL_NAME: claude-sonnet-4
CASE_ID: poisson_simple
TEST_MODE: fix_accuracy
DESCRIPTION: (content of description.md)
```

**You should output:**

```
results/claude-sonnet-4/poisson_simple/solver.py
```

That contains a specialized solver for poisson_simple case.

---

## 🏆 Success Criteria (Simple!)

Your solver is successful if it gets **ANY score > 0**:

1. ✅ Runs without crashing
2. ✅ Outputs correct format (solution.npz + meta.json)
3. ✅ Gets score > 0 (even 1 point = PASSED!)

**You DON'T need high scores!** Just make it work.

**Strategy:**
- Use 5 attempts to debug and fix errors
- Once you get score > 0, move to next case
- Don't waste time optimizing for perfect scores

**Now implement your specialized solver!** 🚀
