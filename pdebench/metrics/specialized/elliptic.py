"""Elliptic PDE specialized metrics computation.

Metrics for elliptic equations (Poisson, Helmholtz, etc.):
- Efficiency: DOF/s (degrees of freedom per second)
- Solver performance: iteration counts, convergence
- Condition number estimation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import SpecializedMetricsComputer


class EllipticMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for elliptic PDEs.
    
    Key metrics:
    - dof: Total degrees of freedom
    - efficiency_dof_per_sec: Computational efficiency
    - solver_iterations: Linear solver iteration count
    - condition_number_estimate: Estimated condition number from CG iterations
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute elliptic-specific metrics.
        
        Args:
            result: Test result containing runtime_sec, error, test_params
        
        Returns:
            Dictionary of specialized metrics
        """
        metrics = {}
        
        try:
            # 1. Compute DOF (more accurate estimation)
            resolution = result.get('test_params', {}).get('resolution', 0)
            degree = result.get('test_params', {}).get('degree', 1)
            
            # Try to read from meta.json solver_info (preferred for autonomous mode)
            meta_file = self.agent_output_dir / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                solver_info = meta.get('solver_info', {})
                resolution = solver_info.get('mesh_resolution', resolution)
                degree = solver_info.get('element_degree', degree)
            
            # 2D triangular mesh DOF estimation
            # P1: DOF ≈ N^2
            # P2: DOF ≈ (2N+1)^2 (includes edge midpoints)
            # P3+: DOF ≈ N^2 * degree^2 (rough approximation)
            if degree == 1:
                dof = resolution ** 2
            elif degree == 2:
                dof = (2 * resolution + 1) ** 2
            else:
                dof = resolution ** 2 * degree ** 2
            
            metrics['dof'] = int(dof)
            metrics['resolution'] = int(resolution) if resolution else 0
            metrics['degree'] = int(degree) if degree else 1
            
            # 2. Compute efficiency DOF/s
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                efficiency = dof / runtime
                metrics['efficiency_dof_per_sec'] = float(efficiency)
            
            # 3. Read solver information from meta.json
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
                
                # Condition number estimate (from CG iterations)
                # For SPD systems: CG iterations ~ sqrt(κ)
                if 'linear_iterations_mean' in solver_info:
                    iters = solver_info['linear_iterations_mean']
                    if iters > 0:
                        kappa_estimate = iters ** 2
                        metrics['condition_number_estimate'] = float(kappa_estimate)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute elliptic metrics: {str(e)}"
        
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
                    
                    # Iteration count
                    if 'iterations' in ls:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                            solver_info['linear_iterations_max'] = int(np.max(iters))
                            solver_info['linear_iterations_total'] = int(np.sum(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
            # Read solver_info field (alternative structure)
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    if 'ksp_type' in si:
                        solver_info['linear_solver_type'] = si['ksp_type']
                    if 'pc_type' in si:
                        solver_info['preconditioner_type'] = si['pc_type']
                    if 'iterations' in si:
                        solver_info['linear_iterations'] = si['iterations']
            
            # Read discretization method
            if 'discretization_method' in meta:
                solver_info['discretization_method'] = meta['discretization_method']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info




