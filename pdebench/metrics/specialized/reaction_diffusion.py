"""Reaction-Diffusion PDE specialized metrics computation.

Metrics for reaction-diffusion equations (Allen-Cahn, Fisher-KPP, Gray-Scott):
- Front propagation speed: Traveling wave speed
- Nonlinear solver efficiency (Newton iterations)
- Solver information (time integrator, nonlinear method)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class ReactionDiffusionMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for reaction-diffusion PDEs.
    
    Key metrics:
    - front_propagation_speed: Traveling wave speed
    - newton_iterations_mean, newton_iterations_max: Nonlinear solver iterations
    - time_integrator, nonlinear_method: Solver information
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reaction-diffusion-specific metrics."""
        metrics = {}
        
        try:
            # 1. Read meta.json
            meta_file = self.agent_output_dir / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                
                # PRIMARY: Read from solver_info field (new unified location)
                solver_info = meta.get('solver_info', {})
                
                # Basic solver info (always available for steady problems)
                if 'mesh_resolution' in solver_info:
                    metrics['mesh_resolution'] = int(solver_info['mesh_resolution'])
                if 'element_degree' in solver_info:
                    metrics['element_degree'] = int(solver_info['element_degree'])
                
                # Linear solver iterations (for steady linear problems)
                if 'iterations' in solver_info:
                    iterations = solver_info['iterations']
                    if isinstance(iterations, (int, float)):
                        metrics['linear_iterations'] = int(iterations)
                    elif isinstance(iterations, list):
                        metrics['linear_iterations_mean'] = float(np.mean(iterations))
                        metrics['linear_iterations_max'] = int(np.max(iterations))
                
                # Nonlinear iterations (for reaction-diffusion with Newton)
                if 'nonlinear_iterations' in solver_info:
                    nl_iters = solver_info['nonlinear_iterations']
                    if isinstance(nl_iters, list) and len(nl_iters) > 0:
                        metrics['newton_iterations_mean'] = float(np.mean(nl_iters))
                        metrics['newton_iterations_max'] = int(np.max(nl_iters))
                
                # FALLBACK: Read from legacy nonlinear_solver field (backward compatibility)
                if 'nonlinear_solver' in meta and 'newton_iterations_mean' not in metrics:
                    ns = meta['nonlinear_solver']
                    if isinstance(ns, dict) and 'iterations' in ns:
                        iters = ns['iterations']
                        if isinstance(iters, list) and len(iters) > 0:
                            metrics['newton_iterations_mean'] = float(np.mean(iters))
                            metrics['newton_iterations_max'] = int(np.max(iters))
                
                # Time stepping info (for transient reaction-diffusion)
                if 'dt' in solver_info:
                    metrics['dt'] = float(solver_info['dt'])
                if 'n_steps' in solver_info:
                    metrics['n_steps'] = int(solver_info['n_steps'])
                if 'time_scheme' in solver_info:
                    metrics['time_integrator'] = solver_info['time_scheme']
            
            # 2. Front propagation speed (optional, requires u_initial.npy and u.npy)
            u0_file = self.agent_output_dir / 'u_initial.npy'
            u_final_file = self.agent_output_dir / 'u.npy'
            
            if u0_file.exists() and u_final_file.exists():
                u0 = np.load(u0_file)
                u_final = np.load(u_final_file)
                
                # Front propagation speed
                front_speed = self._estimate_front_speed(u0, u_final, result)
                if front_speed is not None:
                    metrics['front_propagation_speed'] = float(front_speed)
            
            # Read additional solver information
            solver_info_extra = self._read_solver_info()
            if solver_info_extra:
                # Merge without overwriting existing keys
                for k, v in solver_info_extra.items():
                    if k not in metrics:
                        metrics[k] = v
            
        except Exception as e:
            metrics['error'] = f"Failed to compute reaction-diffusion metrics: {str(e)}"
        
        return metrics
    
    def _estimate_front_speed(self, u0: np.ndarray, u_final: np.ndarray, result: Dict) -> Optional[float]:
        """Estimate traveling wave propagation speed."""
        try:
            if u0.ndim != 1:
                return None
            
            # Find half-max point location
            threshold = 0.5 * (np.max(u0) + np.min(u0))
            
            front_idx_0 = np.argmax(u0 > threshold)
            front_idx_final = np.argmax(u_final > threshold)
            
            dx = 1.0 / len(u0)
            distance = (front_idx_final - front_idx_0) * dx
            
            # Total time
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            if 'time' in pde_config:
                t_end = pde_config['time'].get('t_end', 1.0)
                speed = distance / t_end
                return speed
            else:
                return None
        except:
            return None
    
    def _read_solver_info(self) -> Dict[str, Any]:
        """Read solver information from meta.json."""
        solver_info = {}
        
        try:
            meta_file = self.agent_output_dir / 'meta.json'
            if not meta_file.exists():
                return solver_info
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            # PRIMARY: Read from solver_info field
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
                    # nonlinear_method could be a new optional field
                    if 'nonlinear_method' in si:
                        solver_info['nonlinear_method'] = si['nonlinear_method']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

