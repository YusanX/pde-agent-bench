"""Firedrake oracle dispatcher — mirrors oracle/oracle.py for the Firedrake backend."""
from __future__ import annotations

from typing import Any, Dict

from .._types import OracleResult
from .poisson import FiredrakePoissonSolver
from .heat import FiredrakeHeatSolver
from .helmholtz import FiredrakeHelmholtzSolver
from .biharmonic import FiredrakeBiharmonicSolver
from .convection_diffusion import FiredrakeConvectionDiffusionSolver
from .linear_elasticity import FiredrakeLinearElasticitySolver
from .reaction_diffusion import FiredrakeReactionDiffusionSolver
from .stokes import FiredrakeStokesOracle
from .navier_stokes import FiredrakeNavierStokesOracle


class FiredrakeOracleSolver:
    """Dispatch to Firedrake PDE-specific ground-truth solvers."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        pde_type = case_spec["pde"]["type"]

        if pde_type == "poisson":
            return FiredrakePoissonSolver().solve(case_spec)
        if pde_type == "heat":
            return FiredrakeHeatSolver().solve(case_spec)
        if pde_type == "convection_diffusion":
            return FiredrakeConvectionDiffusionSolver().solve(case_spec)
        if pde_type == "helmholtz":
            return FiredrakeHelmholtzSolver().solve(case_spec)
        if pde_type == "biharmonic":
            return FiredrakeBiharmonicSolver().solve(case_spec)
        if pde_type == "linear_elasticity":
            return FiredrakeLinearElasticitySolver().solve(case_spec)
        if pde_type == "reaction_diffusion":
            return FiredrakeReactionDiffusionSolver().solve(case_spec)
        if pde_type == "stokes":
            return FiredrakeStokesOracle().solve(case_spec)
        if pde_type == "navier_stokes":
            return FiredrakeNavierStokesOracle().solve(case_spec)

        raise ValueError(
            f"Unsupported PDE type for Firedrake oracle: {pde_type}. "
            f"Supported: poisson, heat, convection_diffusion, helmholtz, biharmonic, "
            f"linear_elasticity, reaction_diffusion, stokes, navier_stokes"
        )
