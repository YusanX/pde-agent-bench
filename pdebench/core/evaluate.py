"""Evaluate phase: compute all metrics."""
import json
import numpy as np
from pathlib import Path
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc

from ..solvers.base import create_mesh, create_function_space


def evaluate(case_spec, outdir):
    """
    Evaluate phase:
    1. Load solution from solve phase
    2. Load reference and exact solutions
    3. Compute all metrics
    4. Check validity against targets
    
    Args:
        case_spec: dict
        outdir: Path object
    """
    outdir = Path(outdir)
    
    # Load problem info
    with open(outdir / 'problem_info.json', 'r') as f:
        problem_info = json.load(f)
    
    # Load solution
    sol_data = np.load(outdir / 'solution.npz')
    u_sol_grid = sol_data['u']
    
    # Load reference
    ref_data = np.load(outdir / 'reference.npz')
    u_star_grid = ref_data['u_star']
    
    # Load exact if available
    has_exact = problem_info.get('has_exact', False)
    u_exact_grid = None
    if has_exact and (outdir / 'exact.npz').exists():
        exact_data = np.load(outdir / 'exact.npz')
        u_exact_grid = exact_data['u_exact']
    
    # Compute grid-based metrics
    metrics = {}
    
    # rel_L2_grid: compare with reference
    rel_L2_grid_ref = compute_rel_L2_grid(u_sol_grid, u_star_grid)
    metrics['rel_L2_grid'] = rel_L2_grid_ref
    
    # If exact solution exists, compute more metrics
    if has_exact and u_exact_grid is not None:
        rel_L2_grid_exact = compute_rel_L2_grid(u_sol_grid, u_exact_grid)
        metrics['rel_L2_grid_vs_exact'] = rel_L2_grid_exact
        
        # Compute FE-based error norms
        fe_metrics = compute_fe_error_norms(case_spec, outdir)
        metrics.update(fe_metrics)
    
    # Compute discrete/linear metrics
    linear_metrics = compute_linear_metrics(case_spec, outdir)
    metrics.update(linear_metrics)
    
    # Load cost metrics from meta.json
    with open(outdir / 'meta.json', 'r') as f:
        meta = json.load(f)
    
    metrics['cost'] = {
        'wall_time_sec': meta['wall_time_sec'],
        'iters': meta['solver_info']['iters'],
    }
    
    # Check validity against targets
    targets = case_spec['targets']
    target_metric = targets['metric']
    target_error = targets['target_error']
    
    # Map metric name to computed value
    if target_metric == 'rel_L2_grid':
        achieved_error = metrics['rel_L2_grid']
    elif target_metric == 'rel_L2_fe':
        achieved_error = metrics.get('rel_L2_fe', None)
    elif target_metric == 'rel_H1_semi_fe':
        achieved_error = metrics.get('rel_H1_semi_fe', None)
    else:
        achieved_error = None
    
    if achieved_error is None:
        validity = {
            'pass': False,
            'reason': f'Target metric {target_metric} not available',
        }
    elif achieved_error <= target_error:
        validity = {
            'pass': True,
            'reason': f'{target_metric}={achieved_error:.3e} <= target={target_error:.3e}',
        }
    else:
        validity = {
            'pass': False,
            'reason': f'{target_metric}={achieved_error:.3e} > target={target_error:.3e}',
        }
    
    metrics['validity'] = validity
    
    # Save metrics
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    status = "PASS" if validity['pass'] else "FAIL"
    print(f"[evaluate] {status}: {validity['reason']}")
    
    return metrics


def compute_rel_L2_grid(u1, u2):
    """Compute relative L2 error on grid (discrete approximation)."""
    diff = u1 - u2
    
    # Handle NaN values
    mask = ~(np.isnan(u1) | np.isnan(u2))
    diff_masked = diff[mask]
    u2_masked = u2[mask]
    
    if len(diff_masked) == 0:
        return np.nan
    
    l2_diff = np.sqrt(np.sum(diff_masked**2))
    l2_u2 = np.sqrt(np.sum(u2_masked**2))
    
    if l2_u2 < 1e-15:
        return l2_diff
    
    return l2_diff / l2_u2


def compute_fe_error_norms(case_spec, outdir):
    """Compute FE-based error norms (rel_L2_fe, rel_H1_semi_fe)."""
    # Reconstruct FE space
    mesh_spec = case_spec['mesh']
    domain_spec = case_spec['domain']
    fem_spec = case_spec['fem']
    
    msh = create_mesh(
        domain_spec['type'],
        mesh_spec['resolution'],
        mesh_spec.get('cell_type', 'triangle')
    )
    V = create_function_space(msh, fem_spec['family'], fem_spec['degree'])
    
    # Load solution vector
    viewer = PETSc.Viewer().createBinary(str(outdir / 'solution_u.dat'), 'r')
    u_vec = PETSc.Vec().load(viewer)
    viewer.destroy()
    
    u_h = fem.Function(V)
    u_h.x.array[:] = u_vec.array_r
    
    # Get exact solution
    manufactured = case_spec['pde'].get('manufactured_solution', {})
    if 'u' not in manufactured:
        return {}
    
    x = ufl.SpatialCoordinate(msh)
    from ..solvers.base import parse_expression
    
    pde_type = case_spec['pde']['type']
    
    if pde_type == 'heat':
        # Use final time
        with open(outdir / 'problem_info.json', 'r') as f:
            problem_info = json.load(f)
        t_final = problem_info['t_final']
        u_exact_expr = parse_expression(manufactured['u'], x, x, t=t_final)
    else:
        u_exact_expr = parse_expression(manufactured['u'], x, x)
    
    from ..solvers.base import interpolate_ufl_expression
    u_exact = fem.Function(V)
    interpolate_ufl_expression(u_exact, u_exact_expr)
    
    # Compute error
    e = u_h - u_exact
    
    # L2 norm
    L2_e_squared = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    L2_exact_squared = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    
    rel_L2_fe = np.sqrt(L2_e_squared) / np.sqrt(L2_exact_squared) if L2_exact_squared > 1e-15 else np.sqrt(L2_e_squared)
    
    # H1 seminorm
    H1_e_squared = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    H1_exact_squared = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact)) * ufl.dx))
    
    rel_H1_semi_fe = np.sqrt(H1_e_squared) / np.sqrt(H1_exact_squared) if H1_exact_squared > 1e-15 else np.sqrt(H1_e_squared)
    
    u_vec.destroy()
    
    return {
        'rel_L2_fe': float(rel_L2_fe),
        'rel_H1_semi_fe': float(rel_H1_semi_fe),
    }


def compute_linear_metrics(case_spec, outdir):
    """Compute discrete linear system metrics."""
    pde_type = case_spec['pde']['type']
    
    if pde_type == 'poisson':
        return compute_linear_metrics_poisson(outdir)
    elif pde_type == 'heat':
        # For heat, linear metrics are more complex (multiple timesteps)
        # Simplification: use final timestep only
        return {}
    else:
        return {}


def compute_linear_metrics_poisson(outdir):
    """Compute rel_res and rel_lin_err_M for Poisson."""
    # Load system
    viewer = PETSc.Viewer().createBinary(str(outdir / 'system_A.dat'), 'r')
    A = PETSc.Mat().load(viewer)
    viewer.destroy()
    
    viewer = PETSc.Viewer().createBinary(str(outdir / 'system_b.dat'), 'r')
    b = PETSc.Vec().load(viewer)
    viewer.destroy()
    
    # Load solution
    viewer = PETSc.Viewer().createBinary(str(outdir / 'solution_u.dat'), 'r')
    u = PETSc.Vec().load(viewer)
    viewer.destroy()
    
    # Load reference
    viewer = PETSc.Viewer().createBinary(str(outdir / 'reference_u_star.dat'), 'r')
    u_star = PETSc.Vec().load(viewer)
    viewer.destroy()
    
    # Compute residual: r = b - A*u
    r = b.duplicate()
    A.mult(u, r)
    r.aypx(-1.0, b)  # r = b - r
    
    rel_res = r.norm() / b.norm() if b.norm() > 1e-15 else r.norm()
    
    # Compute rel_lin_err_M: ||u - u_star||_2 / ||u_star||_2
    e = u.duplicate()
    e.axpy(1.0, u)
    e.axpy(-1.0, u_star)
    
    rel_lin_err_M = e.norm() / u_star.norm() if u_star.norm() > 1e-15 else e.norm()
    
    # Clean up
    r.destroy()
    e.destroy()
    u.destroy()
    u_star.destroy()
    A.destroy()
    b.destroy()
    
    return {
        'rel_res': float(rel_res),
        'rel_lin_err_M': float(rel_lin_err_M),
    }

