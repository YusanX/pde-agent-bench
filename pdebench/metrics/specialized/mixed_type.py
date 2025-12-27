"""Mixed-type PDE specialized metrics computation.

Metrics for mixed-type equations (convection-diffusion):
- Péclet number characterization
- Overshoot/undershoot detection
- Boundary layer resolution
- Stabilization quality
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class MixedTypeMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for mixed-type PDEs.
    
    Key metrics:
    - peclet_number: Pe = ||b||L/ε
    - overshoot/undershoot: Non-physical oscillations
    - boundary_layer_error: BL resolution quality
    - total_variation: Oscillation detection
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute mixed-type-specific metrics."""
        metrics = {}
        
        try:
            # Read Péclet number
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            peclet = pde_config.get('peclet', None)
            if peclet is not None:
                metrics['peclet_number'] = float(peclet)
            
            # Read solution fields
            agent_u_file = self.agent_output_dir / 'u.npy'
            oracle_u_file = self.oracle_output_dir / 'u.npy'
            
            if agent_u_file.exists() and oracle_u_file.exists():
                u_agent = np.load(agent_u_file)
                u_oracle = np.load(oracle_u_file)
                
                # 1. Overshoot/undershoot indicators
                u_max_ref = np.max(u_oracle)
                u_min_ref = np.min(u_oracle)
                
                overshoot = max(0.0, np.max(u_agent) - u_max_ref)
                undershoot = max(0.0, u_min_ref - np.min(u_agent))
                
                metrics['overshoot'] = float(overshoot)
                metrics['undershoot'] = float(undershoot)
                
                solution_range = u_max_ref - u_min_ref
                if solution_range > 1e-14:
                    metrics['overshoot_relative'] = float(overshoot / solution_range)
                    metrics['undershoot_relative'] = float(undershoot / solution_range)
                
                # 2. Total variation (detect Gibbs oscillations)
                tv_agent = self._compute_total_variation(u_agent)
                tv_oracle = self._compute_total_variation(u_oracle)
                
                metrics['total_variation'] = float(tv_agent)
                if tv_oracle > 1e-14:
                    metrics['tv_ratio'] = float(tv_agent / tv_oracle)
                
                # 3. Boundary layer error
                if peclet is not None and peclet > 1:
                    bl_error = self._compute_boundary_layer_error(u_agent, u_oracle, peclet)
                    if bl_error is not None:
                        metrics['boundary_layer_error'] = float(bl_error)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute mixed-type metrics: {str(e)}"
        
        return metrics
    
    def _compute_total_variation(self, u: np.ndarray) -> float:
        """Compute total variation TV(u)."""
        if u.ndim == 1:
            return float(np.sum(np.abs(np.diff(u))))
        elif u.ndim == 2:
            tv_x = np.sum(np.abs(np.diff(u, axis=0)))
            tv_y = np.sum(np.abs(np.diff(u, axis=1)))
            return float(tv_x + tv_y)
        else:
            return 0.0
    
    def _compute_boundary_layer_error(self, u_agent: np.ndarray, u_oracle: np.ndarray, peclet: float) -> Optional[float]:
        """Compute error in boundary layer region."""
        try:
            if u_agent.ndim == 1:
                nx = len(u_agent)
                epsilon = 1.0 / (peclet + 1e-10)
                bl_thickness = 3 * epsilon
                bl_points = int(bl_thickness * nx)
                bl_points = max(bl_points, 5)
                bl_points = min(bl_points, nx // 4)
                
                err_left = np.linalg.norm(u_agent[:bl_points] - u_oracle[:bl_points])
                err_right = np.linalg.norm(u_agent[-bl_points:] - u_oracle[-bl_points:])
                
                return max(err_left, err_right)
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
                    if 'stabilization' in si:
                        solver_info['stabilization_method'] = si['stabilization']
                    if 'upwind_parameter' in si:
                        solver_info['upwind_parameter'] = float(si['upwind_parameter'])
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

