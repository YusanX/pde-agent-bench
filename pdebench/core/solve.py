"""Solve phase: use baseline Krylov solver."""
import json
import time
import numpy as np
from pathlib import Path
import psutil
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc
import ufl

from ..solvers.base import create_mesh, create_function_space, sample_on_grid
from ..solvers.heat import setup_heat_problem
from ..linsolve.baseline import solve_linear


def solve_case(case_spec, outdir, ksp_params=None):
    """
    Solve phase:
    1. Load discrete system from generate phase
    2. Solve using baseline Krylov solver
    3. Save solution and metadata
    
    Args:
        case_spec: dict (loaded case JSON)
        outdir: Path object
        ksp_params: optional dict to override KSP parameters
    """
    outdir = Path(outdir)
    
    # Load problem info
    with open(outdir / 'problem_info.json', 'r') as f:
        problem_info = json.load(f)
    
    pde_type = problem_info['pde_type']
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    t_start = time.time()
    
    if pde_type == 'poisson':
        solve_poisson(case_spec, outdir, ksp_params)
    elif pde_type == 'heat':
        solve_heat(case_spec, outdir, ksp_params)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    t_end = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024
    
    print(f"[solve] Completed in {t_end - t_start:.2f}s")


def solve_poisson(case_spec, outdir, ksp_params=None):
    """Solve Poisson using baseline Krylov."""
    # Load system
    viewer = PETSc.Viewer().createBinary(str(outdir / 'system_A.dat'), 'r')
    A = PETSc.Mat().load(viewer)
    viewer.destroy()
    
    viewer = PETSc.Viewer().createBinary(str(outdir / 'system_b.dat'), 'r')
    b = PETSc.Vec().load(viewer)
    viewer.destroy()
    
    # Setup KSP parameters
    if ksp_params is None:
        ksp_params = {}
    
    # Default parameters
    if 'type' not in ksp_params:
        # Check if matrix is symmetric (for CG vs GMRES)
        ksp_params['type'] = 'cg'  # Poisson is symmetric
    if 'pc_type' not in ksp_params:
        ksp_params['pc_type'] = 'jacobi'
    if 'rtol' not in ksp_params:
        ksp_params['rtol'] = 1e-10
    
    # Solve
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    t_start = time.time()
    u_vec, solver_info = solve_linear(A, b, ksp_params)
    t_end = time.time()
    
    mem_after = process.memory_info().rss / 1024 / 1024
    
    # Reconstruct FE function for sampling
    mesh_spec = case_spec['mesh']
    domain_spec = case_spec['domain']
    fem_spec = case_spec['fem']
    
    msh = create_mesh(
        domain_spec['type'],
        mesh_spec['resolution'],
        mesh_spec.get('cell_type', 'triangle')
    )
    V = create_function_space(msh, fem_spec['family'], fem_spec['degree'])
    
    u = fem.Function(V)
    u.x.array[:] = u_vec.array_r
    
    # Sample on grid
    output_spec = case_spec['output']
    grid_spec = output_spec['grid']
    x_grid, y_grid, u_grid = sample_on_grid(
        u, grid_spec['bbox'], grid_spec['nx'], grid_spec['ny']
    )
    
    # Save solution
    np.savez(
        outdir / 'solution.npz',
        x=x_grid,
        y=y_grid,
        u=u_grid,
    )
    
    # Save u vector for evaluation
    viewer = PETSc.Viewer().createBinary(str(outdir / 'solution_u.dat'), 'w')
    u_vec.view(viewer)
    viewer.destroy()
    
    # Extract exposed parameters
    expose_params = case_spec.get('expose_parameters', [])
    exposed = {}
    for param in expose_params:
        if param == 'mesh.resolution':
            exposed[param] = mesh_spec['resolution']
        elif param == 'fem.degree':
            exposed[param] = fem_spec['degree']
        elif param == 'ksp.type':
            exposed[param] = solver_info['ksp_type']
        elif param == 'ksp.rtol':
            exposed[param] = solver_info['rtol']
        elif param == 'pc.type':
            exposed[param] = solver_info['pc_type']
    
    # Save metadata
    meta = {
        'wall_time_sec': solver_info['wall_time_sec'],
        'peak_rss_mb': mem_after,
        'solver_info': {
            'ksp_type': solver_info['ksp_type'],
            'pc_type': solver_info['pc_type'],
            'rtol': solver_info['rtol'],
            'iters': solver_info['iters'],
            'converged': solver_info['converged'],
            'residual_norm': solver_info['residual_norm'],
        },
        'exposed_parameters': exposed,
    }
    
    with open(outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def solve_heat(case_spec, outdir, ksp_params=None):
    """Solve Heat equation using baseline Krylov."""
    # Reconstruct problem
    mesh_spec = case_spec['mesh']
    domain_spec = case_spec['domain']
    fem_spec = case_spec['fem']
    
    msh = create_mesh(
        domain_spec['type'],
        mesh_spec['resolution'],
        mesh_spec.get('cell_type', 'triangle')
    )
    V = create_function_space(msh, fem_spec['family'], fem_spec['degree'])
    
    time_spec = case_spec['pde']['time']
    t0 = time_spec.get('t0', 0.0)
    t_end = time_spec['t_end']
    dt = time_spec['dt']
    
    num_steps = int(np.ceil((t_end - t0) / dt))
    
    # Initial condition
    manufactured = case_spec['pde'].get('manufactured_solution', {})
    
    u_prev = fem.Function(V)
    
    if 'u' in manufactured:
        x = ufl.SpatialCoordinate(msh)
        from ..solvers.base import parse_expression, interpolate_ufl_expression
        u0_expr = parse_expression(manufactured['u'], x, x, t=t0)
        interpolate_ufl_expression(u_prev, u0_expr)
    elif 'u0' in manufactured:
        x = ufl.SpatialCoordinate(msh)
        from ..solvers.base import parse_expression, interpolate_ufl_expression
        u0_expr = parse_expression(manufactured['u0'], x, x)
        interpolate_ufl_expression(u_prev, u0_expr)
    else:
        u_prev.x.array[:] = 0.0
    
    # Setup KSP parameters
    if ksp_params is None:
        ksp_params = {}
    
    if 'type' not in ksp_params:
        ksp_params['type'] = 'cg'
    if 'pc_type' not in ksp_params:
        ksp_params['pc_type'] = 'jacobi'
    if 'rtol' not in ksp_params:
        ksp_params['rtol'] = 1e-10
    
    t_current = t0
    total_iters = 0
    total_solve_time = 0.0
    
    process = psutil.Process()
    
    # Time stepping
    for step in range(num_steps):
        t_current += dt
        
        A, b, bcs, _ = setup_heat_problem(msh, V, case_spec, u_prev, dt, t_current)
        
        u_vec, solver_info = solve_linear(A, b, ksp_params)
        
        u_prev.x.array[:] = u_vec.array_r
        
        total_iters += solver_info['iters']
        total_solve_time += solver_info['wall_time_sec']
        
        A.destroy()
        b.destroy()
        u_vec.destroy()
    
    mem_after = process.memory_info().rss / 1024 / 1024
    
    # Sample final solution
    output_spec = case_spec['output']
    grid_spec = output_spec['grid']
    x_grid, y_grid, u_grid = sample_on_grid(
        u_prev, grid_spec['bbox'], grid_spec['nx'], grid_spec['ny']
    )
    
    np.savez(
        outdir / 'solution.npz',
        x=x_grid,
        y=y_grid,
        u=u_grid,
        t_final=t_current,
    )
    
    # Save final u vector
    u_final_vec = PETSc.Vec().createSeq(len(u_prev.x.array))
    u_final_vec.array[:] = u_prev.x.array
    viewer = PETSc.Viewer().createBinary(str(outdir / 'solution_u.dat'), 'w')
    u_final_vec.view(viewer)
    viewer.destroy()
    u_final_vec.destroy()
    
    # Extract exposed parameters
    expose_params = case_spec.get('expose_parameters', [])
    exposed = {}
    for param in expose_params:
        if param == 'mesh.resolution':
            exposed[param] = mesh_spec['resolution']
        elif param == 'fem.degree':
            exposed[param] = fem_spec['degree']
        elif param == 'time.dt':
            exposed[param] = dt
        elif param == 'ksp.type':
            exposed[param] = ksp_params['type']
        elif param == 'ksp.rtol':
            exposed[param] = ksp_params['rtol']
        elif param == 'pc.type':
            exposed[param] = ksp_params['pc_type']
    
    # Save metadata
    meta = {
        'wall_time_sec': total_solve_time,
        'peak_rss_mb': mem_after,
        'solver_info': {
            'ksp_type': ksp_params['type'],
            'pc_type': ksp_params['pc_type'],
            'rtol': ksp_params['rtol'],
            'iters': total_iters,
            'num_timesteps': num_steps,
        },
        'exposed_parameters': exposed,
    }
    
    with open(outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

