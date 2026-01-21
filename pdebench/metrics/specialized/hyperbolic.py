"""Hyperbolic PDE specialized metrics computation.

Metrics for hyperbolic equations (wave, advection, Burgers', etc.):
- CFL number: Stability indicator
- Total variation: TV norm for shock detection
- Solver information (time integrator, shock limiter)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import SpecializedMetricsComputer


class HyperbolicMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for hyperbolic PDEs.
    
    Key metrics:
    - cfl_number: CFL stability indicator
    - total_variation: TV norm for shock detection
    - time_integrator, shock_limiter: Solver information
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute hyperbolic-specific metrics."""
        metrics = {}
        
        try:
            # Basic parameters
            resolution = result.get('test_params', {}).get('resolution', 0)
            dt = result.get('test_params', {}).get('dt', 0.01)
            
            h = 1.0 / resolution
            
            # CFL number (for advection: CFL = |a| * dt / h)
            if 'velocity' in self.config['oracle_config']['pde']:
                vel = self.config['oracle_config']['pde']['velocity']
                if isinstance(vel, dict):
                    vx = vel.get('vx', 0.0)
                    vy = vel.get('vy', 0.0)
                    velocity_mag = np.sqrt(vx**2 + vy**2)
                else:
                    velocity_mag = float(vel)
                
                cfl = velocity_mag * dt / h
                metrics['cfl_number'] = float(cfl)
                if cfl > 1.0:
                    metrics['cfl_warning'] = f"CFL={cfl:.2f} > 1.0 (explicit unstable)"
            
            # Total variation and shock detection
            u_history_file = self.agent_output_dir / 'u_history.npy'
            if u_history_file.exists():
                u_history = np.load(u_history_file)
                
                # Total Variation at final time
                u_final = u_history[-1]
                tv = self._compute_total_variation(u_final)
                metrics['total_variation'] = float(tv)
                
                
            
            # Solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
        
        except Exception as e:
            metrics['error'] = f"Failed to compute hyperbolic metrics: {str(e)}"
        
        return metrics
    
    def _compute_total_variation(self, u: np.ndarray) -> float:
        """Compute discrete total variation."""
        # TV = sum |u[i+1] - u[i]| + |u[:,j+1] - u[:,j]|
        tv_x = np.sum(np.abs(np.diff(u, axis=1)))
        tv_y = np.sum(np.abs(np.diff(u, axis=0)))
        return tv_x + tv_y
    
    def _read_solver_info(self) -> Dict[str, Any]:
        """Read solver information."""
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
                    if 'limiter' in si:
                        solver_info['shock_limiter'] = si['limiter']
        
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info




