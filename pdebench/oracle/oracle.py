"""Unified oracle entry point."""
from __future__ import annotations

from typing import Any, Dict

from .common import OracleResult
from .convection_diffusion import ConvectionDiffusionSolver
from .heat import HeatSolver
from .navier_stokes_incompressible import NavierStokesIncompressibleSolver
from .poisson import PoissonSolver
from .stokes import StokesSolver


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
        if pde_type == "navier_stokes_incompressible":
            return NavierStokesIncompressibleSolver().solve(case_spec)

        raise ValueError(f"Unsupported PDE type: {pde_type}")
