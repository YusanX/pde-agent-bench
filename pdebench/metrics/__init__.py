"""Metrics module: scoring, tier levels, and PDE-specialized metrics.

This module is responsible for all metric computations:
- Generic scoring logic (fix_accuracy, fix_time modes)
- Tier level determination (L1, L2, L3)
- PDE-specific specialized metrics (elliptic, parabolic, etc.)
"""

from .scoring import compute_score
from .tier_levels import check_tier_levels, generate_tiers_from_baseline
from .specialized import get_specialized_metrics_computer

__all__ = [
    'compute_score',
    'check_tier_levels',
    'generate_tiers_from_baseline',
    'get_specialized_metrics_computer',
]




