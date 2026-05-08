"""
deal.II-based oracle solver package.

Imports are lazy so that loading this sub-package in an environment
without deal.II installed does NOT cause an ImportError at import time.
The C++ binaries are compiled on first use by DealIIOracleSolver.solve().
"""

__all__ = ["DealIIOracleSolver"]


def __getattr__(name: str):
    if name == "DealIIOracleSolver":
        from .oracle import DealIIOracleSolver
        return DealIIOracleSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
