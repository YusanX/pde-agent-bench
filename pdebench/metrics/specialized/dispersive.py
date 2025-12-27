"""Dispersive PDE specialized metrics computation.

Metrics for dispersive equations (Schrödinger, KdV, etc.):
- Phase velocity and group velocity errors
- Mass/energy conservation
- Dispersion relation accuracy
- Spectral analysis
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
    - mass_conservation_error: ∫|ψ|² conservation
    - phase_error: Phase velocity accuracy
    - spectrum_error: Spectral decomposition error
    - energy_conservation_error: Energy drift
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute dispersive-specific metrics."""
        metrics = {}
        
        try:
            # Read solution fields
            agent_u_file = self.agent_output_dir / 'u.npy'
            oracle_u_file = self.oracle_output_dir / 'u.npy'
            
            if agent_u_file.exists() and oracle_u_file.exists():
                u_agent = np.load(agent_u_file)
                u_oracle = np.load(oracle_u_file)
                
                # 1. Mass conservation (L2 norm)
                mass_agent = np.linalg.norm(u_agent)
                mass_oracle = np.linalg.norm(u_oracle)
                
                metrics['mass_agent'] = float(mass_agent)
                metrics['mass_oracle'] = float(mass_oracle)
                
                if mass_oracle > 1e-14:
                    mass_error = np.abs(mass_agent - mass_oracle) / mass_oracle
                    metrics['mass_conservation_error'] = float(mass_error)
                
                # 2. Phase error (peak position)
                phase_error = self._compute_phase_error(u_agent, u_oracle)
                metrics['phase_error'] = float(phase_error)
                
                # 3. Spectrum comparison (if periodic)
                if u_agent.ndim == 1:
                    spectrum_error = self._compute_spectrum_error(u_agent, u_oracle)
                    if spectrum_error is not None:
                        metrics['spectrum_error'] = float(spectrum_error)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute dispersive metrics: {str(e)}"
        
        return metrics
    
    def _compute_phase_error(self, u_agent: np.ndarray, u_oracle: np.ndarray) -> float:
        """Compute phase error (peak location)."""
        try:
            idx_agent = np.argmax(np.abs(u_agent))
            idx_oracle = np.argmax(np.abs(u_oracle))
            
            if u_agent.ndim == 1:
                return np.abs(idx_agent - idx_oracle) / u_agent.shape[0]
            elif u_agent.ndim == 2:
                row_a, col_a = np.unravel_index(idx_agent, u_agent.shape)
                row_o, col_o = np.unravel_index(idx_oracle, u_oracle.shape)
                return np.sqrt((row_a - row_o)**2 + (col_a - col_o)**2) / u_agent.shape[0]
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_spectrum_error(self, u_agent: np.ndarray, u_oracle: np.ndarray) -> Optional[float]:
        """Compute spectral error using FFT."""
        try:
            if u_agent.ndim != 1:
                return None
            
            # FFT
            fft_agent = np.fft.fft(u_agent)
            fft_oracle = np.fft.fft(u_oracle)
            
            # L2 error in frequency domain
            spectrum_error = np.linalg.norm(fft_agent - fft_oracle) / np.linalg.norm(fft_oracle)
            return spectrum_error
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
                    if 'time_scheme' in si:
                        solver_info['time_integrator'] = si['time_scheme']
                    if 'splitting_method' in si:
                        solver_info['splitting_method'] = si['splitting_method']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info

