"""Incompressible Flow (Stokes/Navier-Stokes) specialized metrics computation.

Metrics for incompressible flow equations (Stokes, Navier-Stokes):
- Divergence-free constraint: ||∇·u||
- Mass conservation: ∫ ∇·u dx
- Pressure mean value (nullspace handling)
- Inf-sup stability indicators
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class IncompressibleFlowMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for incompressible flow PDEs.
    
    Key metrics:
    - divergence_L2: Divergence-free constraint violation
    - mass_conservation_error: Global mass conservation
    - pressure_mean_enforced: Pressure nullspace handling
    - velocity_gradient_L2: Shear rate
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute incompressible flow-specific metrics."""
        metrics = {}
        
        try:
            # 1. Read velocity field
            agent_u_file = self.agent_output_dir / 'u.npy'
            if agent_u_file.exists():
                u = np.load(agent_u_file)
                
                # Velocity L2 norm
                metrics['velocity_L2'] = float(np.linalg.norm(u))
                
                # Compute divergence (finite difference)
                if u.ndim >= 3:  # (nx, ny, dim) or (nx, ny, nz, dim)
                    div_u = self._compute_divergence_fd(u)
                    
                    # Divergence L2 norm
                    div_L2 = np.linalg.norm(div_u)
                    metrics['divergence_L2'] = float(div_L2)
                    
                    u_L2 = np.linalg.norm(u)
                    if u_L2 > 1e-14:
                        metrics['divergence_relative'] = float(div_L2 / u_L2)
                    
                    # Global mass flux error: ∫ ∇·u dx
                    mass_flux = np.sum(div_u)
                    metrics['mass_flux_integral'] = float(mass_flux)
                    
                    # Relative mass conservation error
                    total_velocity_mag = np.sum(np.abs(u))
                    if total_velocity_mag > 1e-14:
                        mass_error = np.abs(mass_flux) / total_velocity_mag
                        metrics['mass_conservation_error'] = float(mass_error)
                    
                    # Velocity gradient norm (shear rate)
                    grad_u_norm = self._compute_velocity_gradient_norm(u)
                    metrics['velocity_gradient_L2'] = float(grad_u_norm)
            
            # 2. Read pressure field
            agent_p_file = self.agent_output_dir / 'p.npy'
            if agent_p_file.exists():
                p = np.load(agent_p_file)
                
                p_L2 = np.linalg.norm(p)
                metrics['pressure_L2'] = float(p_L2)
                
                # Pressure mean (nullspace check)
                p_mean = np.mean(p)
                p_std = np.std(p)
                
                metrics['pressure_mean'] = float(p_mean)
                metrics['pressure_std'] = float(p_std)
                
                # Check if pressure constant is handled
                if np.abs(p_mean) > 0.01 * p_std:
                    metrics['pressure_mean_enforced'] = False
                else:
                    metrics['pressure_mean_enforced'] = True
            
            # 3. Inf-sup stability indicator
            if agent_u_file.exists() and agent_p_file.exists():
                u = np.load(agent_u_file)
                p = np.load(agent_p_file)
                
                # Pressure gradient norm
                grad_p = self._compute_pressure_gradient_norm(p)
                
                # Heuristic inf-sup indicator: ||∇p|| / ||u||
                u_norm = np.linalg.norm(u)
                if u_norm > 1e-14:
                    inf_sup_indicator = grad_p / u_norm
                    metrics['inf_sup_indicator'] = float(inf_sup_indicator)
            
            # 4. Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute incompressible flow metrics: {str(e)}"
        
        return metrics
    
    def _compute_divergence_fd(self, u: np.ndarray) -> np.ndarray:
        """Compute divergence using finite differences."""
        if u.ndim == 3:  # 2D: (nx, ny, 2)
            nx, ny, _ = u.shape
            h = 1.0 / nx
            
            # ∂u_x/∂x + ∂u_y/∂y
            du_x = np.gradient(u[:, :, 0], h, axis=0)
            du_y = np.gradient(u[:, :, 1], h, axis=1)
            div_u = du_x + du_y
            
            return div_u
        else:
            return np.zeros_like(u[:, :, 0])
    
    def _compute_velocity_gradient_norm(self, u: np.ndarray) -> float:
        """Compute velocity gradient norm ||∇u||_L2."""
        try:
            if u.ndim == 3:  # 2D: (nx, ny, 2)
                nx, ny, _ = u.shape
                h = 1.0 / nx
                
                # ∂u_x/∂x, ∂u_x/∂y, ∂u_y/∂x, ∂u_y/∂y
                du_x_dx = np.gradient(u[:, :, 0], h, axis=0)
                du_x_dy = np.gradient(u[:, :, 0], h, axis=1)
                du_y_dx = np.gradient(u[:, :, 1], h, axis=0)
                du_y_dy = np.gradient(u[:, :, 1], h, axis=1)
                
                # Frobenius norm
                grad_norm_sq = du_x_dx**2 + du_x_dy**2 + du_y_dx**2 + du_y_dy**2
                return np.sqrt(np.sum(grad_norm_sq))
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_pressure_gradient_norm(self, p: np.ndarray) -> float:
        """Compute pressure gradient norm ||∇p||_L2."""
        try:
            if p.ndim == 2:  # 2D: (nx, ny)
                nx, ny = p.shape
                h = 1.0 / nx
                
                # ∂p/∂x, ∂p/∂y
                dp_dx = np.gradient(p, h, axis=0)
                dp_dy = np.gradient(p, h, axis=1)
                
                grad_norm_sq = dp_dx**2 + dp_dy**2
                return np.sqrt(np.sum(grad_norm_sq))
            else:
                return 0.0
        except:
            return 0.0
    
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
                    
                    if 'iterations' in ls:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                            solver_info['linear_iterations_max'] = int(np.max(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
            # Block preconditioner info
            if 'block_preconditioner' in meta:
                solver_info['block_preconditioner'] = meta['block_preconditioner']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

