"""Stochastic PDE specialized metrics computation.

Metrics for stochastic PDEs (SPDEs with random forcing/coefficients):
- Sample statistics (mean, variance, confidence intervals)
- Monte Carlo convergence rate
- Moment accuracy
- Uncertainty quantification
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import SpecializedMetricsComputer


class StochasticMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for stochastic PDEs.
    
    Key metrics:
    - n_samples: Number of MC samples
    - mean_L2_norm: Solution mean norm
    - mean_variance: Average variance
    - mean_ci_width: Confidence interval width
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute stochastic-specific metrics."""
        metrics = {}
        
        try:
            # Read multiple samples (u_sample_0.npy, u_sample_1.npy, ...)
            samples = []
            i = 0
            while True:
                sample_file = self.agent_output_dir / f'u_sample_{i}.npy'
                if not sample_file.exists():
                    break
                u_sample = np.load(sample_file)
                samples.append(u_sample)
                i += 1
            
            if len(samples) > 0:
                samples = np.array(samples)  # shape: (n_samples, ...)
                
                # 1. Mean and variance
                u_mean = np.mean(samples, axis=0)
                u_var = np.var(samples, axis=0)
                
                metrics['n_samples'] = len(samples)
                metrics['mean_L2_norm'] = float(np.linalg.norm(u_mean))
                metrics['mean_variance'] = float(np.mean(u_var))
                metrics['max_variance'] = float(np.max(u_var))
                
                # 2. Confidence interval width (95% CI)
                u_std = np.std(samples, axis=0)
                ci_width = 1.96 * u_std / np.sqrt(len(samples))
                metrics['mean_ci_width'] = float(np.mean(ci_width))
                
                # 3. Compare with oracle mean (if exists)
                oracle_mean_file = self.oracle_output_dir / 'u_mean.npy'
                if oracle_mean_file.exists():
                    u_mean_oracle = np.load(oracle_mean_file)
                    mean_error = np.linalg.norm(u_mean - u_mean_oracle) / np.linalg.norm(u_mean_oracle)
                    metrics['mean_solution_error'] = float(mean_error)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute stochastic metrics: {str(e)}"
        
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
                    if 'stochastic_method' in si:
                        solver_info['stochastic_method'] = si['stochastic_method']
                    if 'mc_samples' in si:
                        solver_info['mc_samples'] = int(si['mc_samples'])
                    if 'polynomial_chaos_order' in si:
                        solver_info['pc_order'] = int(si['polynomial_chaos_order'])
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

