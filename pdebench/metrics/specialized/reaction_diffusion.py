"""Reaction-Diffusion PDE specialized metrics computation.

Metrics for reaction-diffusion equations (Allen-Cahn, Fisher-KPP, Gray-Scott):
- Energy decay (free energy monotonicity)
- Mass conservation
- Front propagation speed
- Nonlinear solver efficiency
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
    - energy_decay_ratio: Free energy dissipation
    - mass_conservation_error: Mass conservation
    - front_propagation_speed: Traveling wave speed
    - newton_iterations_mean: Nonlinear solver efficiency
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reaction-diffusion-specific metrics."""
        metrics = {}
        
        try:
            # 1. Read meta.json for nonlinear solver info
            meta_file = self.agent_output_dir / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                
                # Nonlinear iterations
                if 'nonlinear_solver' in meta:
                    ns = meta['nonlinear_solver']
                    if isinstance(ns, dict) and 'iterations' in ns:
                        iters = ns['iterations']
                        if isinstance(iters, list):
                            metrics['newton_iterations_mean'] = float(np.mean(iters))
                            metrics['newton_iterations_max'] = int(np.max(iters))
                
                # Energy evolution
                if 'energy_history' in meta:
                    energy = np.array(meta['energy_history'])
                    
                    energy_diffs = np.diff(energy)
                    n_violations = np.sum(energy_diffs > 1e-10)
                    
                    metrics['energy_monotone'] = bool(n_violations == 0)
                    metrics['energy_violations'] = int(n_violations)
                    
                    if len(energy) > 1 and np.abs(energy[0]) > 1e-14:
                        decay_ratio = (energy[0] - energy[-1]) / np.abs(energy[0])
                        metrics['energy_decay_ratio'] = float(decay_ratio)
            
            # 2. Mass conservation
            u0_file = self.agent_output_dir / 'u_initial.npy'
            u_final_file = self.agent_output_dir / 'u.npy'
            
            if u0_file.exists() and u_final_file.exists():
                u0 = np.load(u0_file)
                u_final = np.load(u_final_file)
                
                mass0 = np.sum(u0)
                mass_final = np.sum(u_final)
                
                if np.abs(mass0) > 1e-14:
                    mass_error = np.abs(mass_final - mass0) / np.abs(mass0)
                    metrics['mass_conservation_error'] = float(mass_error)
                
                # 3. Front propagation speed
                front_speed = self._estimate_front_speed(u0, u_final, result)
                if front_speed is not None:
                    metrics['front_propagation_speed'] = float(front_speed)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
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
            
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
                    if 'nonlinear_method' in si:
                        solver_info['nonlinear_method'] = si['nonlinear_method']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

