"""Parabolic PDE specialized metrics computation.

Metrics for parabolic equations (Heat equation, diffusion, etc.):
- WorkRate: (DOF × N_steps) / T_total
- CFL number: Stability indicator
- Problem parameters (DOF, time steps, dt, t_end)
- Solver information (linear solver, iterations)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer
from .meta_reader import (
    read_agent_meta,
    get_time_stepping_params,
    get_mesh_params,
    compute_dof
)


class ParabolicMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for parabolic PDEs.
    
    Key metrics:
    - dof, n_steps, dt, t_end: Problem parameters
    - efficiency_workrate: Work per unit time (DOF × steps / time)
    - time_per_step: Average time per step
    - cfl_number: CFL stability indicator
    - linear_solver_type, preconditioner_type: Solver information
    - linear_iterations_mean, linear_iterations_max: Iteration counts
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute parabolic-specific metrics.
        
        Args:
            result: Test result containing runtime_sec, error, test_params
        
        Returns:
            Dictionary of specialized metrics
        """
        metrics = {}
        
        try:
            # 1. Read agent's parameter choices from meta.json (primary source)
            agent_meta = read_agent_meta(self.agent_output_dir)
            
            # Extract mesh parameters
            mesh_params = get_mesh_params(agent_meta, result)
            resolution = mesh_params['resolution']
            degree = mesh_params['degree']
            
            # Extract time stepping parameters
            time_params = get_time_stepping_params(agent_meta, result, self.config['oracle_config'])
            dt = time_params['dt']
            n_steps = time_params['n_steps']
            t_end = time_params['t_end']
            scheme = time_params['scheme']
            
            # Compute DOF
            dof = compute_dof(resolution, degree, dim=2)
            
            metrics['dof'] = int(dof)
            metrics['n_steps'] = n_steps
            metrics['dt'] = float(dt)
            metrics['t_end'] = float(t_end)
            metrics['agent_resolution'] = resolution
            metrics['agent_degree'] = degree
            if scheme and scheme != 'unknown':
                metrics['time_scheme'] = scheme
            
            # 2. Compute WorkRate
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                workrate = (dof * n_steps) / runtime
                metrics['efficiency_workrate'] = float(workrate)
                
                # Average time per step
                time_per_step = runtime / n_steps
                metrics['time_per_step'] = float(time_per_step)
            
            # 3. CFL number
            h = 1.0 / resolution if resolution > 0 else 1.0
            oracle_cfg = self.config.get('oracle_config', {})
            coeffs = oracle_cfg.get('pde', {}).get('coefficients', {})
            kappa_cfg = coeffs.get('kappa', {'value': 1.0})
            if isinstance(kappa_cfg, dict):
                kappa = kappa_cfg.get('value', 1.0)
            else:
                kappa = kappa_cfg
            if not isinstance(kappa, (int, float)):
                kappa = 1.0
            # For heat equation: CFL = κ * dt / h^2
            cfl = kappa * dt / (h ** 2)
            metrics['cfl_number'] = float(cfl)
            if cfl > 0.5:  # Explicit stability limit
                metrics['cfl_warning'] = f"CFL={cfl:.2f} > 0.5 (explicit unstable)"
            
            
            # 5. Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute parabolic metrics: {str(e)}"
        
        return metrics
    
    def _read_solver_info(self) -> Dict[str, Any]:
        """Read solver information from meta.json."""
        solver_info = {}
        
        try:
            meta_file = self.agent_output_dir / 'meta.json'
            if not meta_file.exists():
                return solver_info
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            # PRIMARY: Read from solver_info field (new unified location)
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    # Linear solver info
                    if 'ksp_type' in si:
                        solver_info['linear_solver_type'] = si['ksp_type']
                    if 'pc_type' in si:
                        solver_info['preconditioner_type'] = si['pc_type']
                    # Time scheme (should already be extracted above, but keep as fallback)
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
                    # Iterations
                    if 'iterations' in si:
                        iterations = si['iterations']
                        if isinstance(iterations, (int, float)):
                            solver_info['linear_iterations'] = int(iterations)
                        elif isinstance(iterations, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iterations))
                            solver_info['linear_iterations_max'] = int(np.max(iterations))
                            solver_info['linear_iterations_per_step'] = [int(x) for x in iterations]
            
            # FALLBACK: Read from legacy linear_solver field (backward compatibility)
            if 'linear_solver' in meta:
                ls = meta['linear_solver']
                if isinstance(ls, dict):
                    if 'type' in ls and 'linear_solver_type' not in solver_info:
                        solver_info['linear_solver_type'] = ls.get('type', 'unknown')
                    if 'preconditioner' in ls and 'preconditioner_type' not in solver_info:
                        solver_info['preconditioner_type'] = ls.get('preconditioner', 'none')
                    
                    if 'iterations' in ls and 'linear_iterations_mean' not in solver_info:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                            solver_info['linear_iterations_max'] = int(np.max(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info


