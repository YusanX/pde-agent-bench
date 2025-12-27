"""Hyperbolic PDE specialized metrics computation.

Metrics for hyperbolic equations (wave, advection, Burgers', etc.):
- Wave propagation speed
- CFL number
- Shock resolution (TVD, oscillation detection)
- Energy conservation
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
    - energy_conservation_error: Energy drift
    - oscillation_detected: Spurious oscillations near shocks
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
                
                # Oscillation detection near shocks
                oscillation_detected, max_overshoot = self._detect_oscillations(u_final)
                metrics['oscillation_detected'] = oscillation_detected
                if oscillation_detected:
                    metrics['max_overshoot'] = float(max_overshoot)
                
                # Energy conservation
                energy_history = np.array([
                    np.sum(u_history[i]**2) for i in range(len(u_history))
                ])
                energy_drift = np.abs(energy_history[-1] - energy_history[0]) / energy_history[0]
                metrics['energy_conservation_error'] = float(energy_drift)
            
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
    
    def _detect_oscillations(self, u: np.ndarray, threshold: float = 0.1) -> tuple:
        """Detect spurious oscillations (Gibbs phenomenon)."""
        # Simple detection: check for rapid sign changes in gradients
        grad_x = np.diff(u, axis=1)
        grad_y = np.diff(u, axis=0)
        
        # Find sign changes
        sign_changes_x = np.sum(np.diff(np.sign(grad_x), axis=1) != 0)
        sign_changes_y = np.sum(np.diff(np.sign(grad_y), axis=0) != 0)
        
        total_sign_changes = sign_changes_x + sign_changes_y
        
        # Overshoot detection
        u_max = np.max(u)
        u_min = np.min(u)
        range_u = u_max - u_min
        
        # Check if max/min are far from expected bounds
        max_overshoot = 0.0
        if range_u > 0:
            # Assuming solution should be bounded in [0, 1]
            if u_max > 1.0:
                max_overshoot = max(max_overshoot, u_max - 1.0)
            if u_min < 0.0:
                max_overshoot = max(max_overshoot, -u_min)
        
        oscillation_detected = total_sign_changes > threshold * u.size or max_overshoot > 0.01
        
        return oscillation_detected, max_overshoot
    
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

