"""Pure-Python types shared by all oracle backends (DOLFINx, Firedrake, …).

This module intentionally has NO dependency on dolfinx, firedrake, mpi4py or
petsc4py so that either backend can import it without pulling in the other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class OracleResult:
    baseline_error: float
    baseline_time: float
    reference: np.ndarray
    solver_info: Dict[str, Any]
    num_dofs: int


def compute_rel_L2_grid(u1: np.ndarray, u2: np.ndarray) -> float:
    """Relative L2 error between two grid arrays (NaN-safe)."""
    mask = ~(np.isnan(u1) | np.isnan(u2))
    diff = (u1 - u2)[mask]
    ref = u2[mask]
    if diff.size == 0:
        return float("nan")
    l2_diff = math.sqrt(float(np.sum(diff ** 2)))
    l2_ref = math.sqrt(float(np.sum(ref ** 2)))
    if l2_ref < 1e-15:
        return l2_diff
    return l2_diff / l2_ref
