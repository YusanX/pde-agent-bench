"""Helper functions for reading agent's meta.json with PDE-specific parameters.

This module provides utilities to extract agent's parameter choices
from meta.json in a backward-compatible way.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def read_agent_meta(agent_output_dir: Path) -> Dict[str, Any]:
    """
    Read agent's meta.json and extract all parameter choices.
    
    Returns a flattened dictionary with:
    - Common parameters (mesh_resolution, element_degree, ksp_type, pc_type)
    - Time stepping (dt, n_steps, time_scheme) from solver_info
    - Iterations (iterations, nonlinear_iterations) from solver_info
    - PDE-specific parameters (nested under 'pde_specific')
    
    Args:
        agent_output_dir: Directory containing agent's meta.json
    
    Returns:
        Dictionary with agent's parameter choices, or empty dict if file not found
    """
    meta_data = {
        'mesh_resolution': None,
        'element_degree': None,
        'ksp_type': 'unknown',
        'pc_type': 'unknown',
        'dt': None,
        'n_steps': None,
        'time_scheme': 'unknown',
        'iterations': None,
        'nonlinear_iterations': None,
        'pde_specific': {}
    }
    
    try:
        meta_file = agent_output_dir / 'meta.json'
        if not meta_file.exists():
            return meta_data
        
        with open(meta_file) as f:
            meta = json.load(f)
        
        solver_info = meta.get('solver_info', {})
        
        # Extract common parameters
        meta_data['mesh_resolution'] = solver_info.get('mesh_resolution')
        meta_data['element_degree'] = solver_info.get('element_degree')
        meta_data['element_family'] = solver_info.get('element_family', 'Lagrange')
        meta_data['ksp_type'] = solver_info.get('ksp_type', 'unknown')
        meta_data['pc_type'] = solver_info.get('pc_type', 'unknown')
        
        # Extract time stepping parameters (new unified location)
        meta_data['dt'] = solver_info.get('dt')
        meta_data['n_steps'] = solver_info.get('n_steps')
        meta_data['time_scheme'] = solver_info.get('time_scheme', 'unknown')
        
        # Extract iteration counts (new unified location)
        meta_data['iterations'] = solver_info.get('iterations')
        meta_data['nonlinear_iterations'] = solver_info.get('nonlinear_iterations')
        meta_data['rationale'] = solver_info.get('rationale', '')
        
        # Extract PDE-specific parameters (legacy location, still useful)
        if 'pde_specific' in solver_info:
            meta_data['pde_specific'] = solver_info['pde_specific']
        
    except Exception as e:
        # Graceful degradation - return partially filled dict
        pass
    
    return meta_data


def get_time_stepping_params(
    agent_meta: Dict[str, Any],
    result: Dict[str, Any],
    oracle_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract time-stepping parameters with fallback logic.
    
    Priority:
    1. agent_meta (from solver_info: dt, n_steps, time_scheme)
    2. agent_meta['pde_specific']['time_stepping'] (legacy autonomous mode)
    3. result['test_params'] (guided mode, backward compatible)
    4. oracle_config['pde']['time'] (fallback)
    
    Returns:
        Dictionary with: dt, n_steps, scheme, t_end
    """
    time_params = {}
    
    # PRIMARY: Try agent's solver_info first (new unified location)
    dt = agent_meta.get('dt')
    n_steps = agent_meta.get('n_steps')
    scheme = agent_meta.get('time_scheme', 'unknown')
    
    # FALLBACK 1: Try agent's pde_specific (legacy location)
    if dt is None or n_steps is None:
        time_config = agent_meta.get('pde_specific', {}).get('time_stepping', {})
        if dt is None:
            dt = time_config.get('dt')
        if n_steps is None:
            n_steps = time_config.get('n_steps')
        if scheme == 'unknown':
            scheme = time_config.get('scheme', 'unknown')
    
    # FALLBACK 2: Try test_params (guided mode)
    if dt is None:
        dt = result.get('test_params', {}).get('dt')
    
    # FALLBACK 3: Oracle config
    oracle_time = oracle_config.get('pde', {}).get('time', {})
    t_end = oracle_time.get('t_end', 1.0)
    
    if dt is None:
        dt = oracle_time.get('dt', 0.01)
    
    if n_steps is None:
        import numpy as np
        n_steps = int(np.ceil(t_end / dt))
    
    time_params['dt'] = float(dt)
    time_params['n_steps'] = int(n_steps)
    time_params['t_end'] = float(t_end)
    time_params['scheme'] = scheme
    
    return time_params


def get_mesh_params(
    agent_meta: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract mesh parameters with fallback logic.
    
    Priority:
    1. agent_meta (autonomous mode)
    2. result['test_params'] (guided mode)
    
    Returns:
        Dictionary with: resolution, degree, family
    """
    resolution = agent_meta.get('mesh_resolution')
    degree = agent_meta.get('element_degree')
    family = agent_meta.get('element_family', 'Lagrange')
    
    # Fallback to test_params
    if resolution is None:
        resolution = result.get('test_params', {}).get('resolution', 0)
    if degree is None:
        degree = result.get('test_params', {}).get('degree', 1)
    
    return {
        'resolution': int(resolution) if resolution else 0,
        'degree': int(degree) if degree and degree != 'unknown' else 1,
        'family': family
    }


def compute_dof(resolution: int, degree: int, dim: int = 2) -> int:
    """
    Estimate degrees of freedom.
    
    Args:
        resolution: Mesh resolution (N in N×N mesh)
        degree: Polynomial degree
        dim: Spatial dimension (default 2D)
    
    Returns:
        Estimated number of DOFs
    """
    if dim == 2:
        if degree == 1:
            return resolution ** 2
        elif degree == 2:
            return (2 * resolution + 1) ** 2
        elif degree == 3:
            return (3 * resolution + 1) ** 2
        else:
            return resolution ** 2 * degree ** 2
    elif dim == 3:
        if degree == 1:
            return resolution ** 3
        elif degree == 2:
            return (2 * resolution + 1) ** 3
        else:
            return resolution ** 3 * degree ** 3
    else:
        return 0


def get_convection_diffusion_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract convection-diffusion specific parameters."""
    cd_params = agent_meta.get('pde_specific', {}).get('convection_diffusion', {})
    
    return {
        'stabilization': cd_params.get('stabilization', 'none'),
        'tau_parameter': cd_params.get('tau_parameter'),
        'upwind_parameter': cd_params.get('upwind_parameter')
    }


def get_incompressible_flow_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract incompressible flow specific parameters."""
    flow_params = agent_meta.get('pde_specific', {}).get('stokes_ns', {})
    
    return {
        'velocity_degree': flow_params.get('velocity_degree'),
        'pressure_degree': flow_params.get('pressure_degree'),
        'velocity_space': flow_params.get('velocity_space', 'Lagrange'),
        'pressure_space': flow_params.get('pressure_space', 'Lagrange')
    }


def get_nonlinear_solver_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract nonlinear solver parameters (for reaction-diffusion, etc.)."""
    nl_params = agent_meta.get('pde_specific', {}).get('nonlinear_solver', {})
    
    return {
        'method': nl_params.get('method', 'newton'),
        'max_iterations': nl_params.get('max_iterations'),
        'tolerance': nl_params.get('tolerance'),
        'line_search': nl_params.get('line_search', False)
    }


def get_stabilization_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stabilization parameters (for hyperbolic, compressible flow, etc.)."""
    stab_params = agent_meta.get('pde_specific', {}).get('stabilization', {})
    
    return {
        'method': stab_params.get('method', 'none'),
        'limiter': stab_params.get('limiter'),
        'order': stab_params.get('order'),
        'flux_scheme': agent_meta.get('pde_specific', {}).get('flux_scheme')
    }


def get_phase_space_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract phase-space parameters (for kinetic equations)."""
    ps_params = agent_meta.get('pde_specific', {}).get('phase_space', {})
    
    return {
        'x_resolution': ps_params.get('x_resolution'),
        'v_resolution': ps_params.get('v_resolution'),
        'v_max': ps_params.get('v_max')
    }


def get_fractional_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fractional PDE parameters."""
    frac_params = agent_meta.get('pde_specific', {}).get('fractional', {})
    
    return {
        'alpha': frac_params.get('alpha', 0.5),
        'approximation': frac_params.get('approximation', 'finite_difference')
    }


def get_stochastic_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stochastic PDE parameters."""
    stoch_params = agent_meta.get('pde_specific', {}).get('stochastic', {})
    
    return {
        'n_samples': stoch_params.get('n_samples'),
        'random_seed': stoch_params.get('random_seed'),
        'noise_type': stoch_params.get('noise_type', 'white')
    }


def get_multiphysics_params(agent_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract multiphysics coupling parameters."""
    mp_params = agent_meta.get('pde_specific', {}).get('multiphysics', {})
    
    return {
        'coupling_method': mp_params.get('coupling_method', 'monolithic'),
        'physics_fields': mp_params.get('physics_fields', []),
        'convergence_tolerance': mp_params.get('convergence_tolerance'),
        'max_coupling_iterations': mp_params.get('max_coupling_iterations')
    }


def extract_all_pde_specific_params(
    agent_meta: Dict[str, Any],
    pde_type: str
) -> Dict[str, Any]:
    """
    Extract all PDE-specific parameters based on PDE type.
    
    This is a convenience function that calls the appropriate getter
    based on the PDE type.
    
    Args:
        agent_meta: Agent's meta dictionary from read_agent_meta()
        pde_type: PDE type string (e.g., 'parabolic', 'hyperbolic', etc.)
    
    Returns:
        Dictionary with all relevant PDE-specific parameters
    """
    params = {}
    
    # Time-stepping (common to many types)
    if pde_type in ['parabolic', 'hyperbolic', 'dispersive', 'reaction_diffusion', 
                    'compressible_flow', 'stochastic']:
        time_step = agent_meta.get('pde_specific', {}).get('time_stepping', {})
        if time_step:
            params['time_stepping'] = time_step
    
    # Type-specific parameters
    if pde_type == 'mixed_type':
        params['convection_diffusion'] = get_convection_diffusion_params(agent_meta)
    
    elif pde_type == 'incompressible_flow':
        params['stokes_ns'] = get_incompressible_flow_params(agent_meta)
    
    elif pde_type == 'reaction_diffusion':
        params['nonlinear_solver'] = get_nonlinear_solver_params(agent_meta)
    
    elif pde_type in ['hyperbolic', 'compressible_flow']:
        params['stabilization'] = get_stabilization_params(agent_meta)
    
    elif pde_type == 'kinetic':
        params['phase_space'] = get_phase_space_params(agent_meta)
    
    elif pde_type == 'fractional':
        params['fractional'] = get_fractional_params(agent_meta)
    
    elif pde_type == 'stochastic':
        params['stochastic'] = get_stochastic_params(agent_meta)
    
    elif pde_type == 'multiphysics':
        params['multiphysics'] = get_multiphysics_params(agent_meta)
    
    return params

