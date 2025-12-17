"""Generate phase: build discrete system and reference solution."""
import json
import time
import numpy as np
from pathlib import Path
import psutil
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc
import ufl

from ..solvers.base import create_mesh, create_function_space, parse_expression, sample_on_grid
from ..solvers.poisson import setup_poisson_problem
from ..solvers.heat import setup_heat_problem
from ..linsolve.baseline import solve_linear_direct


def generate(case_spec, outdir):
    """
    Generate phase:
    1. Build mesh and FE space
    2. Assemble discrete system
    3. Generate reference solution u_star using direct LU
    4. Save system matrices and reference solution
    
    Args:
        case_spec: dict (loaded case JSON)
        outdir: Path object for output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create mesh
    mesh_spec = case_spec['mesh']
    domain_spec = case_spec['domain']
    msh = create_mesh(
        domain_spec['type'],
        mesh_spec['resolution'],
        mesh_spec.get('cell_type', 'triangle')
    )
    
    # Create function space
    fem_spec = case_spec['fem']
    V = create_function_space(msh, fem_spec['family'], fem_spec['degree'])
    
    pde_type = case_spec['pde']['type']
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    t_start = time.time()
    
    if pde_type == 'poisson':
        generate_poisson(case_spec, msh, V, outdir)
    elif pde_type == 'heat':
        generate_heat(case_spec, msh, V, outdir)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    t_end = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024
    
    # Save metadata
    meta = {
        'phase': 'generate',
        'wall_time_sec': t_end - t_start,
        'peak_rss_mb': mem_after,
        'mesh_resolution': mesh_spec['resolution'],
        'fem_degree': fem_spec['degree'],
        'num_dofs': V.dofmap.index_map.size_global,
    }
    
    with open(outdir / 'generate_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[generate] Completed in {t_end - t_start:.2f}s, DOFs={V.dofmap.index_map.size_global}")


def generate_poisson(case_spec, msh, V, outdir):
    """Generate reference solution for Poisson."""
    # Setup problem
    A, b, bcs, u_exact_func = setup_poisson_problem(msh, V, case_spec)
    
    # Solve with direct LU to get reference
    u_star_vec, ref_info = solve_linear_direct(A, b)
    
    # Save reference solution
    u_star = fem.Function(V)
    u_star.x.array[:] = u_star_vec.array_r
    
    # Sample on grid
    output_spec = case_spec['output']
    grid_spec = output_spec['grid']
    x_grid, y_grid, u_grid = sample_on_grid(
        u_star, grid_spec['bbox'], grid_spec['nx'], grid_spec['ny']
    )
    
    # Save reference
    np.savez(
        outdir / 'reference.npz',
        x=x_grid,
        y=y_grid,
        u_star=u_grid,
    )
    
    # Save exact solution if available
    if u_exact_func is not None:
        x_grid_ex, y_grid_ex, u_exact_grid = sample_on_grid(
            u_exact_func, grid_spec['bbox'], grid_spec['nx'], grid_spec['ny']
        )
        np.savez(
            outdir / 'exact.npz',
            x=x_grid_ex,
            y=y_grid_ex,
            u_exact=u_exact_grid,
        )
    
    # Save system (for solve phase)
    viewer = PETSc.Viewer().createBinary(str(outdir / 'system_A.dat'), 'w')
    A.view(viewer)
    viewer.destroy()
    
    viewer = PETSc.Viewer().createBinary(str(outdir / 'system_b.dat'), 'w')
    b.view(viewer)
    viewer.destroy()
    
    # Save u_star vector
    viewer = PETSc.Viewer().createBinary(str(outdir / 'reference_u_star.dat'), 'w')
    u_star_vec.view(viewer)
    viewer.destroy()
    
    # Save mesh/space info
    with open(outdir / 'problem_info.json', 'w') as f:
        json.dump({
            'pde_type': 'poisson',
            'num_dofs': V.dofmap.index_map.size_global,
            'has_exact': u_exact_func is not None,
        }, f, indent=2)


def generate_heat(case_spec, msh, V, outdir):
    """Generate reference solution for Heat equation."""
    time_spec = case_spec['pde']['time']
    t0 = time_spec.get('t0', 0.0)
    t_end = time_spec['t_end']
    dt = time_spec['dt']
    
    num_steps = int(np.ceil((t_end - t0) / dt))
    
    # Initial condition
    manufactured = case_spec['pde'].get('manufactured_solution', {})
    
    u_prev = fem.Function(V)
    
    if 'u' in manufactured:
        # Use manufactured solution at t=t0
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
        # Zero initial condition
        u_prev.x.array[:] = 0.0
    
    t_current = t0
    
    u_star_history = []
    
    # Time stepping with direct solver
    for step in range(num_steps):
        t_current += dt
        
        A, b, bcs, u_exact_func = setup_heat_problem(msh, V, case_spec, u_prev, dt, t_current)
        
        u_vec, _ = solve_linear_direct(A, b)
        
        u_prev.x.array[:] = u_vec.array_r
        
        u_star_history.append(u_prev.x.array.copy())
    
    # Save final reference solution
    output_spec = case_spec['output']
    grid_spec = output_spec['grid']
    x_grid, y_grid, u_grid = sample_on_grid(
        u_prev, grid_spec['bbox'], grid_spec['nx'], grid_spec['ny']
    )
    
    np.savez(
        outdir / 'reference.npz',
        x=x_grid,
        y=y_grid,
        u_star=u_grid,
        t_final=t_current,
    )
    
    # Save exact solution at t_end if available
    if 'u' in manufactured:
        u_exact_final = fem.Function(V)
        x = ufl.SpatialCoordinate(msh)
        from ..solvers.base import parse_expression, interpolate_ufl_expression
        u_exact_expr = parse_expression(manufactured['u'], x, x, t=t_current)
        interpolate_ufl_expression(u_exact_final, u_exact_expr)
        
        x_grid_ex, y_grid_ex, u_exact_grid = sample_on_grid(
            u_exact_final, grid_spec['bbox'], grid_spec['nx'], grid_spec['ny']
        )
        np.savez(
            outdir / 'exact.npz',
            x=x_grid_ex,
            y=y_grid_ex,
            u_exact=u_exact_grid,
            t_final=t_current,
        )
    
    # Save final u_star vector
    u_star_vec = PETSc.Vec().createSeq(len(u_star_history[-1]))
    u_star_vec.array[:] = u_star_history[-1]
    viewer = PETSc.Viewer().createBinary(str(outdir / 'reference_u_star.dat'), 'w')
    u_star_vec.view(viewer)
    viewer.destroy()
    u_star_vec.destroy()
    
    # Save problem info
    with open(outdir / 'problem_info.json', 'w') as f:
        json.dump({
            'pde_type': 'heat',
            'num_dofs': V.dofmap.index_map.size_global,
            'num_timesteps': num_steps,
            'dt': dt,
            't_final': t_current,
            'has_exact': 'u' in manufactured,
        }, f, indent=2)

