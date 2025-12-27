"""Harness module: orchestrates test execution and evaluation.

This module provides the core evaluation infrastructure:
- CaseRunner: Runs a single test case
- BatchEvaluator: Evaluates an agent across multiple cases
"""

from .case_runner import CaseRunner
from .batch_evaluator import BatchEvaluator

__all__ = ['CaseRunner', 'BatchEvaluator']

