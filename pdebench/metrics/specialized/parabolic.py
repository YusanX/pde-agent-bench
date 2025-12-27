"""Parabolic PDE specialized metrics computation.

Metrics for parabolic equations (Heat equation, diffusion, etc.):
- WorkRate: (DOF × N_steps) / T_total
- Energy decay: L2 norm evolution
- CFL number: stability indicator
- Maximum principle verification
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class ParabolicMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for parabolic PDEs.
    
    Key metrics:
    - efficiency_workrate: Work per unit time (DOF × steps / time)
    - energy_decay_ratio: Energy dissipation
    - cfl_number: CFL stability indicator
    - max_principle_violated: Maximum principle check
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
            # 1. Compute DOF and time steps
            resolution = result.get('test_params', {}).get('resolution', 0)
            degree = result.get('test_params', {}).get('degree', 1)
            
            # DOF estimation (same as elliptic)
            if degree == 1:
                dof = resolution ** 2
            elif degree == 2:
                dof = (2 * resolution + 1) ** 2
            else:
                dof = resolution ** 2 * degree ** 2
            
            # Time stepping parameters
            oracle_time_config = self.config['oracle_config']['pde']['time']
            t_end = oracle_time_config['t_end']
            dt = result.get('test_params', {}).get('dt', oracle_time_config['dt'])
            n_steps = int(np.ceil(t_end / dt))
            
            metrics['dof'] = int(dof)
            metrics['n_steps'] = n_steps
            metrics['dt'] = float(dt)
            metrics['t_end'] = float(t_end)
            
            # 2. Compute WorkRate
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                workrate = (dof * n_steps) / runtime
                metrics['efficiency_workrate'] = float(workrate)
                
                # Average time per step
                time_per_step = runtime / n_steps
                metrics['time_per_step'] = float(time_per_step)
            
            # 3. CFL number
            h = 1.0 / resolution
            kappa = oracle_time_config.get('kappa', 1.0)
            # For heat equation: CFL = κ * dt / h^2
            cfl = kappa * dt / (h ** 2)
            metrics['cfl_number'] = float(cfl)
            if cfl > 0.5:  # Explicit stability limit
                metrics['cfl_warning'] = f"CFL={cfl:.2f} > 0.5 (explicit unstable)"
            
            # 4. Energy decay and maximum principle (if u_history available)
            u_history_file = self.agent_output_dir / 'u_history.npy'
            if u_history_file.exists():
                u_history = np.load(u_history_file)
                
                # Energy (L2 norm) history
                energy_history = np.array([
                    np.linalg.norm(u_history[i].flatten()) 
                    for i in range(len(u_history))
                ])
                
                # Check energy monotonicity
                energy_diffs = np.diff(energy_history)
                n_violations = np.sum(energy_diffs > 1e-10)
                
                metrics['energy_monotone'] = bool(n_violations == 0)
                metrics['energy_violations'] = int(n_violations)
                
                # Energy decay ratio
                if energy_history[0] > 1e-14:
                    decay_ratio = (energy_history[0] - energy_history[-1]) / energy_history[0]
                    metrics['energy_decay_ratio'] = float(decay_ratio)
                    
                    # Estimate decay rate λ (E(t) ~ E0 * exp(-λt))
                    if energy_history[-1] > 1e-14:
                        lambda_estimate = -np.log(energy_history[-1] / energy_history[0]) / t_end
                        metrics['decay_rate_lambda'] = float(lambda_estimate)
                
                # Maximum principle check
                initial_max = np.max(np.abs(u_history[0]))
                all_max = np.array([np.max(np.abs(u_history[i])) for i in range(len(u_history))])
                global_max = np.max(all_max)
                
                metrics['initial_max'] = float(initial_max)
                metrics['global_max'] = float(global_max)
                
                # Check violation (allow 1% numerical error)
                tolerance = initial_max * 0.01
                if global_max > initial_max + tolerance:
                    metrics['max_principle_violated'] = True
                    metrics['max_principle_overshoot'] = float(global_max - initial_max)
                else:
                    metrics['max_principle_violated'] = False
            
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
            
            # Read linear solver information
            if 'linear_solver' in meta:
                ls = meta['linear_solver']
                if isinstance(ls, dict):
                    solver_info['linear_solver_type'] = ls.get('type', 'unknown')
                    solver_info['preconditioner_type'] = ls.get('preconditioner', 'none')
                    
                    if 'iterations' in ls:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                            solver_info['linear_iterations_max'] = int(np.max(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
            # Alternative structure
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    if 'ksp_type' in si:
                        solver_info['linear_solver_type'] = si['ksp_type']
                    if 'pc_type' in si:
                        solver_info['preconditioner_type'] = si['pc_type']
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

