"""Unified oracle entry point.

Supports multiple solver backends via the ``solver_library`` argument:
- ``'dolfinx'``  (default): FEniCSx / DOLFINx
- ``'firedrake'``: Firedrake (requires Firedrake installation)
- ``'dealii'``   : deal.II C++ FEM (requires deal.II ≥ 9.3 and cmake)
"""
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
from .burger import BurgersSolver
from .wave import WaveSolver


class OracleSolver:
    """Dispatch to PDE-specific ground-truth solvers."""

    def solve(
        self,
        case_spec: Dict[str, Any],
        solver_library: str = "dolfinx",
        use_docker: bool = False,
        docker_image: str = None,
    ) -> OracleResult:
        """
        求解一个 PDE case 并返回 OracleResult。

        Args:
            case_spec:      oracle_config 字典。
            solver_library: 'dolfinx' (默认) | 'firedrake' | 'dealii'。
            use_docker:     为 True 时在 Docker 容器内运行 firedrake/dealii oracle，
                            宿主机无需安装对应库。
            docker_image:   覆盖默认 Docker 镜像名（None 时使用内置默认值）。
        """
        if use_docker and solver_library in ("firedrake", "dealii"):
            from .docker_bridge import solve_via_docker
            return solve_via_docker(
                case_spec,
                library=solver_library,
                docker_image=docker_image,
            )

        if solver_library == "firedrake":
            try:
                from .firedrake_oracle import FiredrakeOracleSolver
            except ImportError as e:
                raise ImportError(
                    "Firedrake oracle requires Firedrake to be installed. "
                    "See https://www.firedrakeproject.org/download.html\n"
                    "Alternatively, use --docker to run in a container."
                ) from e
            return FiredrakeOracleSolver().solve(case_spec)

        if solver_library == "dealii":
            from .dealii_oracle import DealIIOracleSolver
            return DealIIOracleSolver().solve(case_spec)

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
            # ...existing code...
        if pde_type == "burgers":
            return BurgersSolver().solve(case_spec)
        if pde_type == "wave":
            return WaveSolver().solve(case_spec)

        raise ValueError(f"Unsupported PDE type: {pde_type}")
