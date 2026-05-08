"""Fractional PDE specialized metrics computation.

Metrics for fractional PDEs (fractional Laplacian, Caputo derivatives):
- Fractional order α characterization
- Convergence rate related to α
- Computational cost (nonlocal operators → dense matrices)
- Matrix sparsity
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import SpecializedMetricsComputer


class FractionalMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for fractional PDEs.
    
    Key metrics:
    - fractional_order_alpha: Fractional order parameter
    - efficiency_dof_per_sec: Computational efficiency
    - matrix_sparsity: Sparsity of system matrix
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute fractional-specific metrics."""
        metrics = {}
        
        try:
            # Read fractional order parameter
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            alpha = pde_config.get('fractional_alpha', None)
            if alpha is not None:
                metrics['fractional_order_alpha'] = float(alpha)
            
            # Compute DOF (fractional methods usually need global DOF)
            resolution = result.get('test_params', {}).get('resolution', 0)
            degree = result.get('test_params', {}).get('degree', 1)
            dof = (resolution * degree) ** 2
            metrics['dof'] = int(dof)
            
            # Compute efficiency (fractional methods are typically slow)
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                efficiency = dof / runtime
                metrics['efficiency_dof_per_sec'] = float(efficiency)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
                
                # Check matrix fill-in
                if 'matrix_nnz' in solver_info and 'dof' in metrics:
                    nnz = solver_info['matrix_nnz']
                    dof = metrics['dof']
                    if dof > 0:
                        sparsity = 1.0 - nnz / (dof ** 2)
                        metrics['matrix_sparsity'] = float(sparsity)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute fractional metrics: {str(e)}"
        
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
                    if 'fractional_method' in si:
                        solver_info['fractional_method'] = si['fractional_method']
                    if 'matrix_nnz' in si:
                        solver_info['matrix_nnz'] = int(si['matrix_nnz'])
            
            # Linear solver info
            if 'linear_solver' in meta:
                ls = meta['linear_solver']
                if isinstance(ls, dict):
                    solver_info['linear_solver_type'] = ls.get('type', 'unknown')
                    if 'iterations' in ls:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

