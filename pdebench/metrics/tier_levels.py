"""Tier level determination logic for PDEBench.

This module implements the 3-tier evaluation system (L1/L2/L3):
- L1: Low precision / Fast (engineering grade)
- L2: Medium precision / Medium (standard grade)
- L3: High precision / Slow (scientific grade)

Tiers are derived from Oracle baseline performance to ensure fairness.
"""

from typing import Dict, Any, Optional, List


def check_tier_levels(
    mode: str,
    runtime: float,
    error: float,
    tiers: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check which tier levels an agent solution has passed.
    
    Two evaluation modes:
    - fix_accuracy: Within time budget (medium), achieve different accuracy levels
      * Level 1: Achieve level_1 accuracy (10x relaxed)
      * Level 2: Achieve level_2 accuracy (Oracle baseline)
      * Level 3: Achieve level_3 accuracy (100x stricter)
    
    - fix_time: Meet accuracy requirement (level_2), achieve different speeds
      * Level 1: Within slow time budget (10x Oracle)
      * Level 2: Within medium time budget (Oracle baseline)
      * Level 3: Within fast time budget (0.1x Oracle)
    
    Args:
        mode: Evaluation mode ('fix_accuracy' or 'fix_time')
        runtime: Agent runtime in seconds
        error: Agent error (relative L2)
        tiers: Tier configuration from oracle_baseline.json
    
    Returns:
        Dictionary with:
        - passed: List of passed level numbers [1, 2, 3]
        - total: Total number of levels (always 3)
        - level_details: Dict mapping 'level_1', 'level_2', 'level_3' to bool
    
    Example (fix_accuracy mode - check accuracy tiers):
        >>> tiers = {
        ...     'accuracy': {
        ...         'level_1': {'target_error': 1e-2},
        ...         'level_2': {'target_error': 1e-4},
        ...         'level_3': {'target_error': 1e-6}
        ...     },
        ...     'speed': {
        ...         'medium': {'time_budget': 1.0}
        ...     }
        ... }
        >>> # Runtime 0.5s < 1.0s budget, error 1e-5 passes L1,L2 but not L3
        >>> check_tier_levels('fix_accuracy', 0.5, 1e-5, tiers)
        {'passed': [1, 2], 'total': 3, 'level_details': {...}}
    """
    if not tiers:
        return {
            'passed': [],
            'total': 3,
            'level_details': {f'level_{i}': False for i in [1, 2, 3]}
        }
    
    passed = []
    
    if mode == 'fix_accuracy':
        # Accuracy tiers: check error against accuracy targets
        # Must first meet time requirement (medium is the baseline)
        time_budget = tiers['speed']['medium']['time_budget']
        
        if runtime <= time_budget * 1.2:
            # Time requirement met, check accuracy levels
            if error <= tiers['accuracy']['level_3']['target_error']:
                passed = [1, 2, 3]  # High accuracy: pass all levels
            elif error <= tiers['accuracy']['level_2']['target_error']:
                passed = [1, 2]  # Medium accuracy: pass L1, L2
            elif error <= tiers['accuracy']['level_1']['target_error']:
                passed = [1]  # Low accuracy: pass only L1
    
    elif mode == 'fix_time':
        # Speed tiers: check runtime against time budgets
        # Must first meet accuracy requirement (level_2 is the baseline)
        target_error = tiers['accuracy']['level_2']['target_error']
        
        if error <= target_error * 1.2:
            # Accuracy requirement met, check speed levels
            if runtime <= tiers['speed']['fast']['time_budget']:
                passed = [1, 2, 3]  # Fast: pass all levels
            elif runtime <= tiers['speed']['medium']['time_budget']:
                passed = [1, 2]  # Medium: pass L1, L2
            elif runtime <= tiers['speed']['slow']['time_budget']:
                passed = [1]  # Slow: pass only L1
    
    return {
        'passed': passed,
        'total': 3,
        'level_details': {
            f'level_{i}': (i in passed) for i in [1, 2, 3]
        }
    }


def generate_tiers_from_baseline(
    error_ref: float,
    time_ref: float,
    accuracy_multipliers: Optional[tuple[float, float, float]] = None,
    speed_multipliers: Optional[tuple[float, float, float]] = None
) -> Dict[str, Any]:
    """
    Generate tier configuration from Oracle baseline performance.
    
    Default multipliers (relative to Oracle baseline):
    - Accuracy: [100x, 1x, 0.01x] for [L1, L2, L3]
    - Speed: [0.1x, 1x, 10x] for [Fast, Medium, Slow]
    
    Args:
        error_ref: Oracle baseline error (from direct LU solve)
        time_ref: Oracle baseline runtime
        accuracy_multipliers: Custom multipliers for [L1, L2, L3] accuracy
        speed_multipliers: Custom multipliers for [Fast, Medium, Slow] speed
    
    Returns:
        Tier configuration dictionary compatible with check_tier_levels()
    
    Example:
        >>> generate_tiers_from_baseline(1e-10, 1.0)
        {
            'accuracy': {
                'level_1': {'target_error': 1e-8, 'name': 'Low/Engineering'},
                'level_2': {'target_error': 1e-10, 'name': 'Medium/Standard'},
                'level_3': {'target_error': 1e-12, 'name': 'High/Scientific'}
            },
            'speed': {
                'fast': {'time_budget': 0.1, 'name': 'Real-time'},
                'medium': {'time_budget': 1.0, 'name': 'Interactive'},
                'slow': {'time_budget': 10.0, 'name': 'Batch'}
            }
        }
    """
    if accuracy_multipliers is None:
        accuracy_multipliers = (100.0, 1.0, 0.01)  # L1: 100x, L2: 1x, L3: 0.01x
    
    if speed_multipliers is None:
        speed_multipliers = (0.1, 1.0, 10.0)  # Fast: 0.1x, Medium: 1x, Slow: 10x
    
    return {
        'accuracy': {
            'level_1': {
                'target_error': error_ref * accuracy_multipliers[0],
                'name': 'Low/Engineering'
            },
            'level_2': {
                'target_error': error_ref * accuracy_multipliers[1],
                'name': 'Medium/Standard'
            },
            'level_3': {
                'target_error': error_ref * accuracy_multipliers[2],
                'name': 'High/Scientific'
            }
        },
        'speed': {
            'fast': {
                'time_budget': time_ref * speed_multipliers[0],
                'name': 'Real-time'
            },
            'medium': {
                'time_budget': time_ref * speed_multipliers[1],
                'name': 'Interactive'
            },
            'slow': {
                'time_budget': time_ref * speed_multipliers[2],
                'name': 'Batch'
            }
        }
    }


def compute_tier_pass_rates(
    case_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute tier pass rates across multiple cases.
    
    Args:
        case_results: List of case result dictionaries, each containing:
            - tier_levels: {'passed': [1, 2], 'total': 3, ...}
    
    Returns:
        Dictionary with pass rates for each level:
        - level_1_pass_rate: Fraction passing L1
        - level_2_pass_rate: Fraction passing L2
        - level_3_pass_rate: Fraction passing L3
        - avg_levels_passed: Average number of levels passed
    
    Example:
        >>> results = [
        ...     {'tier_levels': {'passed': [1, 2, 3], 'total': 3}},
        ...     {'tier_levels': {'passed': [1], 'total': 3}},
        ...     {'tier_levels': {'passed': [], 'total': 3}}
        ... ]
        >>> compute_tier_pass_rates(results)
        {
            'level_1_pass_rate': 0.667,
            'level_2_pass_rate': 0.333,
            'level_3_pass_rate': 0.333,
            'avg_levels_passed': 1.333
        }
    """
    if not case_results:
        return {
            'level_1_pass_rate': 0.0,
            'level_2_pass_rate': 0.0,
            'level_3_pass_rate': 0.0,
            'avg_levels_passed': 0.0
        }
    
    total_cases = len(case_results)
    level_counts = {1: 0, 2: 0, 3: 0}
    total_levels_passed = 0
    
    for result in case_results:
        if 'tier_levels' not in result:
            continue
        
        passed_levels = result['tier_levels'].get('passed', [])
        total_levels_passed += len(passed_levels)
        
        for level in passed_levels:
            if level in level_counts:
                level_counts[level] += 1
    
    return {
        'level_1_pass_rate': level_counts[1] / total_cases,
        'level_2_pass_rate': level_counts[2] / total_cases,
        'level_3_pass_rate': level_counts[3] / total_cases,
        'avg_levels_passed': total_levels_passed / total_cases
    }


def compute_weighted_tier_score(
    tier_levels: Dict[str, Any],
    weights: Optional[tuple[float, float, float]] = None
) -> float:
    """
    Compute weighted score based on tier levels passed.
    
    Default weights: L1=15, L2=35, L3=50 (total 100 points)
    
    Args:
        tier_levels: Tier levels dict with 'passed' list
        weights: Custom weights for [L1, L2, L3]
    
    Returns:
        Weighted score in range [0, 100]
    
    Example:
        >>> compute_weighted_tier_score({'passed': [1, 2], 'total': 3})
        50.0  # 15 + 35 = 50 points
    """
    if weights is None:
        weights = (15.0, 35.0, 50.0)  # L1, L2, L3 weights
    
    if sum(weights) != 100.0:
        raise ValueError(f"Weights must sum to 100, got {sum(weights)}")
    
    passed = tier_levels.get('passed', [])
    
    score = 0.0
    for level in passed:
        if 1 <= level <= 3:
            score += weights[level - 1]
    
    return score

