"""Baseline Krylov linear solver."""
import time
from petsc4py import PETSc

def solve_linear(A, b, ksp_params=None):
    """
    Solve Ax=b using Krylov method (baseline solver).
    
    Args:
        A: PETSc Mat
        b: PETSc Vec
        ksp_params: dict with keys 'type', 'pc_type', 'rtol', 'atol', 'max_it'
    
    Returns:
        (x, info): solution vector and solver info dict
    """
    if ksp_params is None:
        ksp_params = {}
    
    ksp_type = ksp_params.get('type', 'cg')
    pc_type = ksp_params.get('pc_type', 'jacobi')
    rtol = ksp_params.get('rtol', 1e-10)
    atol = ksp_params.get('atol', 1e-12)
    max_it = ksp_params.get('max_it', 10000)
    
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    
    pc = ksp.getPC()
    pc.setType(pc_type)
    
    ksp.setFromOptions()
    
    x = b.duplicate()
    
    t0 = time.time()
    ksp.solve(b, x)
    wall_time = time.time() - t0
    
    reason = ksp.getConvergedReason()
    iters = ksp.getIterationNumber()
    rnorm = ksp.getResidualNorm()
    
    info = {
        'ksp_type': ksp_type,
        'pc_type': pc_type,
        'rtol': rtol,
        'atol': atol,
        'iters': iters,
        'converged': reason > 0,
        'converged_reason': reason,
        'residual_norm': float(rnorm),
        'wall_time_sec': wall_time,
    }
    
    return x, info


def solve_linear_direct(A, b):
    """
    Solve Ax=b using direct LU (for reference solution).
    
    Args:
        A: PETSc Mat
        b: PETSc Vec
    
    Returns:
        (x, info): solution vector and solver info dict
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType('preonly')
    
    pc = ksp.getPC()
    pc.setType('lu')
    
    ksp.setFromOptions()
    
    x = b.duplicate()
    
    t0 = time.time()
    ksp.solve(b, x)
    wall_time = time.time() - t0
    
    reason = ksp.getConvergedReason()
    
    info = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'rtol': 0.0,
        'atol': 0.0,
        'iters': 0,
        'converged': reason > 0,
        'converged_reason': reason,
        'wall_time_sec': wall_time,
    }
    
    return x, info

