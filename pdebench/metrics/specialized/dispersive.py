"""Dispersive PDE specialized metrics computation.

Metrics for dispersive equations (Schrödinger, KdV, etc.):
- Mass: L2 norm of solution
- Solver information (time integrator, splitting method)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class DispersiveMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for dispersive PDEs.
    
    Key metrics:
    - mass_agent: Mass (L2 norm of solution)
    - time_integrator, splitting_method: Solver information
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute dispersive-specific metrics."""
        metrics = {}
        
        try:
            # Read solution fields
            agent_u_file = self.agent_output_dir / 'u.npy'
            
            if agent_u_file.exists():
                u_agent = np.load(agent_u_file)
                
                # Mass (L2 norm) - physical quantity, not an error metric
                mass_agent = np.linalg.norm(u_agent)
                metrics['mass_agent'] = float(mass_agent)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute dispersive metrics: {str(e)}"
        
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
            
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
                    if 'splitting_method' in si:
                        solver_info['splitting_method'] = si['splitting_method']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

