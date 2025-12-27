"""Multiphysics coupling specialized metrics computation.

Metrics for multiphysics coupled problems (FSI, thermo-mechanical, etc.):
- Coupling iterations
- Field conservation laws
- Interface continuity
- Load balancing across fields
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from . import SpecializedMetricsComputer


class MultiphysicsMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for multiphysics PDEs.
    
    Key metrics:
    - n_fields: Number of coupled fields
    - coupling_iterations_mean: Coupling solver iterations
    - interface_jump_L2: Interface continuity
    - available_fields: List of field variables
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute multiphysics-specific metrics."""
        metrics = {}
        
        try:
            # Read multiple fields
            fields = ['u', 'p', 'T', 'rho']  # Possible field variables
            available_fields = []
            
            for field_name in fields:
                field_file = self.agent_output_dir / f'{field_name}.npy'
                if field_file.exists():
                    available_fields.append(field_name)
            
            metrics['available_fields'] = available_fields
            metrics['n_fields'] = len(available_fields)
            
            # Read coupling iteration information
            meta_file = self.agent_output_dir / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                
                if 'coupling_iterations' in meta:
                    iters = meta['coupling_iterations']
                    if isinstance(iters, list):
                        metrics['coupling_iterations_mean'] = float(np.mean(iters))
                        metrics['coupling_iterations_max'] = int(np.max(iters))
                    else:
                        metrics['coupling_iterations'] = iters
            
            # Check interface continuity (if interface data available)
            interface_file = self.agent_output_dir / 'interface_jump.npy'
            if interface_file.exists():
                jump = np.load(interface_file)
                metrics['interface_jump_L2'] = float(np.linalg.norm(jump))
                metrics['interface_jump_max'] = float(np.max(np.abs(jump)))
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute multiphysics metrics: {str(e)}"
        
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
                    if 'coupling_scheme' in si:
                        solver_info['coupling_scheme'] = si['coupling_scheme']
                    if 'partitioned_method' in si:
                        solver_info['partitioned_method'] = si['partitioned_method']
                    if 'block_preconditioner' in si:
                        solver_info['block_preconditioner'] = si['block_preconditioner']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

