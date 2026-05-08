"""Specialized metrics computation for different PDE types.

Each PDE type has its own metrics computer that computes
domain-specific performance indicators.
"""

from pathlib import Path
from typing import Dict, Any, Optional


class SpecializedMetricsComputer:
    """Base class for PDE-specific metrics computation."""
    
    def __init__(
        self,
        agent_output_dir: Path,
        oracle_output_dir: Path,
        config: Dict[str, Any]
    ):
        """
        Initialize metrics computer.
        
        Args:
            agent_output_dir: Directory with agent's output files
            oracle_output_dir: Directory with oracle reference files
            config: Case configuration dictionary
        """
        self.agent_output_dir = Path(agent_output_dir)
        self.oracle_output_dir = Path(oracle_output_dir)
        self.config = config
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute specialized metrics.
        
        Args:
            result: Test result containing runtime_sec, error, test_params
        
        Returns:
            Dictionary of specialized metrics
        """
        raise NotImplementedError("Subclasses must implement compute()")


def get_specialized_metrics_computer(
    pde_type: str,
    agent_output_dir: Path,
    oracle_output_dir: Path,
    config: Dict[str, Any]
) -> Optional[SpecializedMetricsComputer]:
    """
    Factory function to get appropriate metrics computer for PDE type.
    
    Args:
        pde_type: PDE type tag (e.g., 'elliptic', 'parabolic')
        agent_output_dir: Agent output directory
        oracle_output_dir: Oracle output directory
        config: Case configuration
    
    Returns:
        Appropriate metrics computer instance or None if not implemented
    """
    # Import here to avoid circular dependencies
    from .elliptic import EllipticMetricsComputer
    from .parabolic import ParabolicMetricsComputer
    from .hyperbolic import HyperbolicMetricsComputer
    from .incompressible_flow import IncompressibleFlowMetricsComputer
    from .mixed_type import MixedTypeMetricsComputer
    from .dispersive import DispersiveMetricsComputer
    from .reaction_diffusion import ReactionDiffusionMetricsComputer
    from .compressible_flow import CompressibleFlowMetricsComputer
    from .kinetic import KineticMetricsComputer
    from .fractional import FractionalMetricsComputer
    from .stochastic import StochasticMetricsComputer
    from .multiphysics import MultiphysicsMetricsComputer
    
    computers = {
        'elliptic': EllipticMetricsComputer,
        'parabolic': ParabolicMetricsComputer,
        'hyperbolic': HyperbolicMetricsComputer,
        'incompressible_flow': IncompressibleFlowMetricsComputer,
        'mixed_type': MixedTypeMetricsComputer,
        'dispersive': DispersiveMetricsComputer,
        'reaction_diffusion': ReactionDiffusionMetricsComputer,
        'compressible_flow': CompressibleFlowMetricsComputer,
        'kinetic': KineticMetricsComputer,
        'fractional': FractionalMetricsComputer,
        'stochastic': StochasticMetricsComputer,
        'multiphysics': MultiphysicsMetricsComputer,
    }
    
    computer_class = computers.get(pde_type.lower())
    if computer_class is None:
        return None
    
    return computer_class(agent_output_dir, oracle_output_dir, config)
