"""Firedrake-based oracle solver package.

Imports are lazy so that loading the sub-package in a DOLFINx environment
(where Firedrake is absent) does NOT cause an ImportError at import time.
"""

__all__ = ["FiredrakeOracleSolver"]


def __getattr__(name: str):
    if name == "FiredrakeOracleSolver":
        from .oracle import FiredrakeOracleSolver  # triggers firedrake import only here
        return FiredrakeOracleSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
