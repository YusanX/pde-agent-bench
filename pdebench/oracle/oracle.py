"""Unified oracle entry point."""
from __future__ import annotations

from typing import Any, Dict

from .common import OracleResult
from .linear_elasticity import LinearElasticitySolver
from .convection_diffusion import ConvectionDiffusionSolver
from .biharmonic import BiharmonicSolver
from .heat import HeatSolver
from .helmholtz import HelmholtzSolver
from .navier_stokes import NavierStokesSolver
from .poisson import PoissonSolver
from .stokes import StokesSolver
from .darcy import DarcySolver
from .reaction_diffusion import ReactionDiffusionSolver


class OracleSolver:
    """Dispatch to PDE-specific ground-truth solvers."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        pde_type = case_spec["pde"]["type"]

        if pde_type == "poisson":
            return PoissonSolver().solve(case_spec)
        if pde_type == "heat":
            return HeatSolver().solve(case_spec)
        if pde_type == "convection_diffusion":
            return ConvectionDiffusionSolver().solve(case_spec)
        if pde_type == "stokes":
            return StokesSolver().solve(case_spec)
        if pde_type == "navier_stokes":
            return NavierStokesSolver().solve(case_spec)
        if pde_type == "helmholtz":
            return HelmholtzSolver().solve(case_spec)
        if pde_type == "biharmonic":
            return BiharmonicSolver().solve(case_spec)
        if pde_type == "linear_elasticity":
            return LinearElasticitySolver().solve(case_spec)
        if pde_type == "darcy":
            return DarcySolver().solve(case_spec)
        if pde_type == "reaction_diffusion":
            return ReactionDiffusionSolver().solve(case_spec)

        raise ValueError(f"Unsupported PDE type: {pde_type}")
