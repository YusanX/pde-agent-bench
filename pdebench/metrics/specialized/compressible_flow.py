"""Compressible Flow PDE specialized metrics computation.

Metrics for compressible flow equations (Euler, compressible Navier-Stokes):
- Shock resolution
- Density positivity preservation
- Conservation laws (mass, momentum, energy)
- Mach number characterization
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class CompressibleFlowMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for compressible flow PDEs.
    
    Key metrics:
    - density_positive: Density positivity check
    - shock_width: Shock resolution quality
    - mass_conservation_error: Mass conservation
    - mach_number: Flow regime indicator
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute compressible flow-specific metrics."""
        metrics = {}
        
        try:
            # Read density field
            agent_rho_file = self.agent_output_dir / 'rho.npy'
            
            if agent_rho_file.exists():
                rho_agent = np.load(agent_rho_file)
                
                # 1. Density positivity check
                rho_min = np.min(rho_agent)
                metrics['density_min'] = float(rho_min)
                metrics['density_positive'] = bool(rho_min > -1e-10)
                
                if rho_min < 0:
                    metrics['density_positivity_violation'] = float(np.abs(rho_min))
                
                # 2. Shock resolution
                if rho_agent.ndim == 1:
                    shock_width = self._compute_shock_width(rho_agent)
                    metrics['shock_width'] = float(shock_width)
            
            # 3. Mass conservation
            mass_error = self._check_mass_conservation()
            if mass_error is not None:
                metrics['mass_conservation_error'] = float(mass_error)
            
            # 4. Mach number
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            mach = pde_config.get('mach', None)
            if mach is not None:
                metrics['mach_number'] = float(mach)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute compressible flow metrics: {str(e)}"
        
        return metrics
    
    def _compute_shock_width(self, rho: np.ndarray) -> float:
        """Estimate shock width in grid points."""
        try:
            grad = np.abs(np.gradient(rho))
            max_grad = np.max(grad)
            
            if max_grad < 1e-6:
                return 0.0
            
            width_points = np.sum(grad > max_grad * 0.1)
            return float(width_points)
        except:
            return 0.0
    
    def _check_mass_conservation(self) -> Optional[float]:
        """Check mass conservation."""
        try:
            rho0_file = self.agent_output_dir / 'rho_initial.npy'
            rho_final_file = self.agent_output_dir / 'rho.npy'
            
            if rho0_file.exists() and rho_final_file.exists():
                rho0 = np.load(rho0_file)
                rho_final = np.load(rho_final_file)
                
                mass0 = np.sum(rho0)
                mass_final = np.sum(rho_final)
                
                if np.abs(mass0) > 1e-14:
                    return np.abs(mass_final - mass0) / np.abs(mass0)
            
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
                    if 'riemann_solver' in si:
                        solver_info['riemann_solver'] = si['riemann_solver']
                    if 'limiter' in si:
                        solver_info['shock_limiter'] = si['limiter']
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

