"""Kinetic PDE specialized metrics computation.

Metrics for kinetic equations (Vlasov, Boltzmann, Fokker-Planck):
- Mass/momentum/energy conservation in phase space
- Entropy production (H-theorem)
- Computational cost (high-dimensional DOF)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import SpecializedMetricsComputer


class KineticMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for kinetic PDEs.
    
    Key metrics:
    - total_mass: ∫f dxdv
    - total_momentum: ∫v·f dxdv
    - total_energy: ∫v²·f dxdv
    - mass_conservation_error: Conservation in phase space
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute kinetic-specific metrics."""
        metrics = {}
        
        try:
            # Read distribution function f(x, v)
            agent_f_file = self.agent_output_dir / 'f.npy'
            
            if agent_f_file.exists():
                f_agent = np.load(agent_f_file)
                
                # Assume f shape is (nx, nv)
                if f_agent.ndim == 2:
                    nx, nv = f_agent.shape
                    
                    # Compute macroscopic quantities (simplified: v ∈ [-v_max, v_max])
                    v_max = 5.0  # Read from config if available
                    dv = 2 * v_max / nv
                    v_grid = np.linspace(-v_max, v_max, nv)
                    
                    # Density: ρ(x) = ∫f dv
                    rho = np.sum(f_agent, axis=1) * dv
                    metrics['total_mass'] = float(np.sum(rho))
                    
                    # Momentum: m(x) = ∫v·f dv
                    momentum = np.sum(f_agent * v_grid[None, :], axis=1) * dv
                    metrics['total_momentum'] = float(np.sum(momentum))
                    
                    # Energy: E = ∫v²·f dv
                    energy = np.sum(f_agent * (v_grid[None, :]**2), axis=1) * dv
                    metrics['total_energy'] = float(np.sum(energy))
            
            # Check conservation (need initial state)
            f0_file = self.agent_output_dir / 'f_initial.npy'
            if f0_file.exists() and agent_f_file.exists():
                f0 = np.load(f0_file)
                f_final = np.load(agent_f_file)
                
                if f0.ndim == 2:
                    nx, nv = f0.shape
                    v_max = 5.0
                    dv = 2 * v_max / nv
                    
                    mass0 = np.sum(f0) * dv
                    mass_final = np.sum(f_final) * dv
                    
                    if np.abs(mass0) > 1e-14:
                        mass_error = np.abs(mass_final - mass0) / np.abs(mass0)
                        metrics['mass_conservation_error'] = float(mass_error)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute kinetic metrics: {str(e)}"
        
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
                    if 'phase_space_method' in si:
                        solver_info['phase_space_method'] = si['phase_space_method']
                    if 'collision_operator' in si:
                        solver_info['collision_operator'] = si['collision_operator']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

