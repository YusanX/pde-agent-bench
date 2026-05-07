"""Universal metrics for all PDE types (inspired by RealPDEBench).

Data-oriented:
  - RMSE  (Root Mean Square Error)
  - MAE   (Mean Absolute Error)
  - R²    (Coefficient of Determination)

Physics-oriented:
  - fRMSE (Fourier-space RMSE, spatial; low / mid / high bands)

All metrics operate on a pair of arrays (u_agent, u_ref) loaded from
``agent_output/solution.npz`` and the oracle reference cache.  They are
NaN-safe: domain-exterior points marked with NaN on either side are
excluded automatically.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_solution_pair(
    agent_output: Path,
    oracle_info: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load agent solution and oracle reference, returning masked valid arrays.

    Returns ``(u_agent_valid, u_ref_valid, mask)`` where *valid* means both
    sides are finite, or *None* when the data cannot be loaded / matched.
    """
    try:
        u_agent = np.load(agent_output / "solution.npz")["u"]
    except Exception:
        return None

    ref_flat = oracle_info.get("reference")
    if ref_flat is None:
        return None

    u_ref = np.array(
        [np.nan if v is None else float(v) for v in ref_flat], dtype=float
    )
    ref_shape = oracle_info.get("reference_shape")
    if ref_shape:
        u_ref = u_ref.reshape(ref_shape)
    elif u_ref.size == u_agent.size:
        u_ref = u_ref.reshape(u_agent.shape)

    if u_agent.shape != u_ref.shape:
        return None

    mask = ~(np.isnan(u_agent) | np.isnan(u_ref))
    if mask.sum() == 0:
        return None

    return u_agent[mask], u_ref[mask], mask


# ── data-oriented metrics ────────────────────────────────────────────────────

def _rmse(diff: np.ndarray) -> float:
    return float(np.sqrt(np.mean(diff ** 2)))


def _mae(diff: np.ndarray) -> float:
    return float(np.mean(np.abs(diff)))


def _r_squared(u_ref: np.ndarray, diff: np.ndarray) -> float:
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((u_ref - np.mean(u_ref)) ** 2))
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else float("-inf")
    return 1.0 - ss_res / ss_tot


# ── physics-oriented: spatial fRMSE ─────────────────────────────────────────

def _spatial_frmse(
    u_agent_2d: np.ndarray,
    u_ref_2d: np.ndarray,
    n_bands: int = 3,
) -> Dict[str, float]:
    """Compute Fourier-space RMSE on a 2-D spatial field.

    Uses ``norm="ortho"`` so that Parseval's theorem holds:
    ``sum |F_diff|^2 == sum |diff|^2``, keeping fRMSE on the same scale as
    spatial RMSE and avoiding amplification by sqrt(nx*ny).

    The frequency magnitudes are partitioned into *n_bands* equal-width bins
    (labelled ``low``, ``mid``, ``high`` for 3 bands).  Returns per-band and
    total fRMSE.
    """
    diff_2d = u_agent_2d - u_ref_2d

    # replace any remaining NaN with 0 before FFT
    diff_2d = np.where(np.isfinite(diff_2d), diff_2d, 0.0)

    # norm="ortho" normalises by 1/sqrt(nx*ny), preserving Parseval's theorem
    F_diff = np.fft.fft2(diff_2d, norm="ortho")
    ny, nx = diff_2d.shape

    freq_y = np.fft.fftfreq(ny)
    freq_x = np.fft.fftfreq(nx)
    FY, FX = np.meshgrid(freq_y, freq_x, indexing="ij")
    freq_mag = np.sqrt(FX ** 2 + FY ** 2)

    max_freq = freq_mag.max()
    if max_freq < 1e-15:
        return {}

    band_labels = ["low", "mid", "high"] if n_bands == 3 else [
        f"band_{i}" for i in range(n_bands)
    ]
    edges = np.linspace(0, max_freq, n_bands + 1)

    result: Dict[str, float] = {}
    total_sq = 0.0
    total_count = 0

    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bands - 1:
            band_mask = (freq_mag >= lo) & (freq_mag < hi)
        else:
            band_mask = (freq_mag >= lo) & (freq_mag <= hi)

        coeffs = F_diff[band_mask]
        if coeffs.size == 0:
            continue

        band_mse = float(np.mean(np.abs(coeffs) ** 2))
        result[f"frmse_{band_labels[i]}"] = float(math.sqrt(band_mse))
        total_sq += float(np.sum(np.abs(coeffs) ** 2))
        total_count += coeffs.size

    if total_count > 0:
        result["frmse_total"] = float(math.sqrt(total_sq / total_count))

    return result


# ── public API ───────────────────────────────────────────────────────────────

def compute_universal_metrics(
    agent_output: Path,
    oracle_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute universal (PDE-type-agnostic) quality metrics.

    Returns a dict with keys ``rmse``, ``mae``, ``r2``, and ``frmse_*``.
    On failure the dict is empty (never raises).
    """
    pair = _load_solution_pair(agent_output, oracle_info)
    if pair is None:
        return {}

    u_agent_flat, u_ref_flat, mask = pair
    diff = u_agent_flat - u_ref_flat

    metrics: Dict[str, Any] = {
        "rmse": _rmse(diff),
        "mae": _mae(diff),
        "r2": _r_squared(u_ref_flat, diff),
    }

    # spatial fRMSE needs the original 2-D arrays
    try:
        u_agent_2d = np.load(agent_output / "solution.npz")["u"]
        ref_flat = oracle_info["reference"]
        u_ref_2d = np.array(
            [np.nan if v is None else float(v) for v in ref_flat], dtype=float
        )
        ref_shape = oracle_info.get("reference_shape")
        if ref_shape:
            u_ref_2d = u_ref_2d.reshape(ref_shape)
        else:
            u_ref_2d = u_ref_2d.reshape(u_agent_2d.shape)

        if u_agent_2d.ndim == 2:
            frmse = _spatial_frmse(u_agent_2d, u_ref_2d)
            metrics.update(frmse)
    except Exception:
        pass

    return metrics


UNIVERSAL_METRIC_KEYS: List[str] = [
    "rmse", "mae", "r2",
    "frmse_low", "frmse_mid", "frmse_high", "frmse_total",
]
"""Canonical key names produced by :func:`compute_universal_metrics`."""
