"""Oracle solver package.

Top-level imports are intentionally lazy so that importing a sub-package
(e.g. ``pdebench.oracle.firedrake_oracle``) does NOT pull in DOLFINx.
Use ``from pdebench.oracle import OracleSolver`` to get the DOLFINx backend,
or ``from pdebench.oracle.firedrake_oracle import FiredrakeOracleSolver``
for the Firedrake backend.
"""

__all__ = ["OracleSolver"]


def __getattr__(name: str):
    if name == "OracleSolver":
        from .oracle import OracleSolver  # triggers dolfinx import only here
        return OracleSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
