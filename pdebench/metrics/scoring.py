"""Generic scoring logic for PDEBench.

This module implements the core scoring algorithms for two evaluation modes:
- fix_accuracy: Fixed accuracy target, optimize speed
- fix_time: Fixed time budget, optimize accuracy
"""

from typing import Literal


def compute_score(
    mode: Literal['fix_accuracy', 'fix_time'],
    runtime: float,
    error: float,
    target_error: float,
    time_budget: float
) -> float:
    """
    Compute score (0-100) based on evaluation mode.
    
    fix_accuracy mode: Fixed accuracy, optimize speed
    - Must achieve error ≤ target_error
    - Score = 100 * (time_budget / runtime), faster is better
    - Capped at 100 points
    
    fix_time mode: Fixed time budget, optimize accuracy
    - Must finish within time_budget
    - Score = 100 * (1 - error/target_error), lower error is better
    - Minimum 10 points if error exceeds target
    
    Args:
        mode: Evaluation mode ('fix_accuracy' or 'fix_time')
        runtime: Actual runtime in seconds
        error: Achieved error (relative L2 norm)
        target_error: Target error threshold
        time_budget: Time budget in seconds
    
    Returns:
        Score in range [0, 100]
    
    Examples:
        >>> # fix_accuracy: achieve error=0.005, target=0.01, runtime=1s, budget=10s
        >>> compute_score('fix_accuracy', 1.0, 0.005, 0.01, 10.0)
        100.0  # 10x faster than budget
        
        >>> # fix_time: achieve error=0.005, target=0.01, runtime=8s, budget=10s
        >>> compute_score('fix_time', 8.0, 0.005, 0.01, 10.0)
        50.0  # 50% better than target
    """
    if mode == 'fix_accuracy':
        return _compute_score_fix_accuracy(runtime, error, target_error, time_budget)
    elif mode == 'fix_time':
        return _compute_score_fix_time(runtime, error, target_error, time_budget)
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}. Must be 'fix_accuracy' or 'fix_time'")


def _compute_score_fix_accuracy(
    runtime: float,
    error: float,
    target_error: float,
    time_budget: float
) -> float:
    """
    Scoring for fix_accuracy mode: fixed precision, optimize speed.
    
    Requirements:
    1. Must achieve error ≤ target_error (otherwise score = 0)
    2. Score based on speedup relative to time budget
    
    Formula:
    - If error > target: score = 0
    - If error ≤ target and runtime ≤ budget: score = 100 * (budget / runtime), capped at 100
    - If error ≤ target and runtime > budget: score = 50 * (budget / runtime), partial credit
    """
    # Check accuracy requirement
    if error > target_error:
        return 0.0
    
    # Check runtime
    if runtime <= 0:
        return 0.0  # Invalid runtime
    
    # Compute speedup ratio
    speedup = time_budget / runtime
    
    if runtime <= time_budget:
        # Within budget: full credit based on speedup
        score = 100.0 * speedup
        return min(score, 100.0)  # Cap at 100
    else:
        # Over budget: partial credit (50% weight)
        score = 50.0 * speedup
        return max(score, 0.0)


def _compute_score_fix_time(
    runtime: float,
    error: float,
    target_error: float,
    time_budget: float
) -> float:
    """
    Scoring for fix_time mode: fixed time budget, optimize accuracy.
    
    Requirements:
    1. Must finish within time_budget (otherwise score = 0)
    2. Score based on accuracy improvement relative to target
    
    Formula:
    - If runtime > budget: score = 0
    - If error ≥ target: score = 10 (baseline)
    - If error < target: score = 100 * (1 - error/target), better accuracy = higher score
    """
    # Check time requirement
    if runtime > time_budget:
        return 0.0
    
    # Check error validity
    if error < 0 or target_error <= 0:
        return 0.0
    
    # Compute accuracy ratio
    error_ratio = error / target_error
    
    if error_ratio >= 1.0:
        # Did not meet target accuracy, but within time budget
        return 10.0  # Baseline score
    else:
        # Met or exceeded target accuracy
        # Score increases as error decreases
        score = 100.0 * (1.0 - error_ratio)
        return max(score, 0.0)


def compute_weighted_score(
    score_accuracy: float,
    score_time: float,
    weight_accuracy: float = 0.6,
    weight_time: float = 0.4
) -> float:
    """
    Compute weighted average score for agents evaluated in both modes.
    
    This is useful for overall rankings that consider both speed and accuracy.
    
    Args:
        score_accuracy: Score from fix_accuracy mode
        score_time: Score from fix_time mode
        weight_accuracy: Weight for accuracy score (default 0.6)
        weight_time: Weight for time score (default 0.4)
    
    Returns:
        Weighted average score in range [0, 100]
    
    Example:
        >>> compute_weighted_score(80.0, 60.0)
        72.0  # 0.6*80 + 0.4*60
    """
    if weight_accuracy + weight_time != 1.0:
        raise ValueError(f"Weights must sum to 1.0, got {weight_accuracy + weight_time}")
    
    return weight_accuracy * score_accuracy + weight_time * score_time


def compute_aggregate_score(case_scores: list[float]) -> dict[str, float]:
    """
    Compute aggregate statistics over multiple cases.
    
    Args:
        case_scores: List of scores from different cases
    
    Returns:
        Dictionary with aggregate metrics:
        - mean: Average score
        - median: Median score
        - min: Minimum score
        - max: Maximum score
        - pass_rate: Fraction of cases with score > 0
    
    Example:
        >>> compute_aggregate_score([100, 80, 0, 60, 0])
        {'mean': 48.0, 'median': 60.0, 'min': 0.0, 'max': 100.0, 'pass_rate': 0.6}
    """
    import numpy as np
    
    if not case_scores:
        return {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'pass_rate': 0.0
        }
    
    scores_array = np.array(case_scores)
    
    return {
        'mean': float(np.mean(scores_array)),
        'median': float(np.median(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'pass_rate': float(np.sum(scores_array > 0) / len(scores_array))
    }


