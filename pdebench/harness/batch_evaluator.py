"""Batch evaluator: evaluates an agent across multiple test cases.

This module coordinates evaluation of an entire agent submission:
- Discovers all cases
- Runs each case
- Aggregates results
- Generates summary statistics
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional
from datetime import datetime

from .case_runner import CaseRunner


class BatchEvaluator:
    """
    Evaluates an agent across multiple test cases.
    
    This orchestrates the entire evaluation workflow for a complete
    agent submission, producing structured results for leaderboard generation.
    """
    
    def __init__(self, cases_dir: Path):
        """
        Initialize batch evaluator.
        
        Args:
            cases_dir: Directory containing case definitions (cases/)
        """
        self.cases_dir = Path(cases_dir)
        if not self.cases_dir.exists():
            raise FileNotFoundError(f"Cases directory not found: {self.cases_dir}")
    
    def evaluate_agent(
        self,
        agent_name: str,
        agent_dir: Path,
        modes: Optional[List[Literal['fix_accuracy', 'fix_time']]] = None,
        case_filter: Optional[List[str]] = None,
        timeout_sec: int = 300
    ) -> Dict[str, Any]:
        """
        Evaluate an agent across all cases.
        
        Args:
            agent_name: Name of the agent (for identification)
            agent_dir: Directory containing agent's solver.py files
            modes: Evaluation modes to run (default: both)
            case_filter: Optional list of case IDs to evaluate (default: all)
            timeout_sec: Timeout per case
        
        Returns:
            Complete evaluation results dictionary
        
        Example:
            >>> evaluator = BatchEvaluator(Path('cases'))
            >>> results = evaluator.evaluate_agent(
            ...     agent_name='gpt-4',
            ...     agent_dir=Path('results/gpt-4'),
            ...     modes=['fix_accuracy', 'fix_time']
            ... )
            >>> print(results['summary']['pass_rate'])
        """
        if modes is None:
            modes = ['fix_accuracy', 'fix_time']
        
        print(f"\n{'='*80}")
        print(f"🚀 Evaluating Agent: {agent_name}")
        print(f"📁 Cases Directory: {self.cases_dir}")
        print(f"📋 Modes: {', '.join(modes)}")
        print(f"{'='*80}\n")
        
        # Discover cases
        case_dirs = self._discover_cases(case_filter)
        print(f"Found {len(case_dirs)} cases to evaluate\n")
        
        # Evaluate each case
        all_results = {}
        for mode in modes:
            all_results[mode] = []
            
            print(f"\n{'─'*80}")
            print(f"📊 Running {mode} mode")
            print(f"{'─'*80}\n")
            
            for i, case_dir in enumerate(case_dirs, 1):
                case_id = case_dir.name
                print(f"[{i}/{len(case_dirs)}] Evaluating: {case_id}")
                
                # Check if agent submitted solver for this case
                agent_solver = agent_dir / case_id / 'solver.py'
                
                if not agent_solver.exists():
                    print(f"   ⚠️  Solver not found, skipping\n")
                    all_results[mode].append({
                        'case_id': case_id,
                        'test_mode': mode,
                        'status': 'NOT_SUBMITTED',
                        'score': 0.0,
                        'error': 'Solver script not found'
                    })
                    continue
                
                # Run case
                try:
                    runner = CaseRunner(case_dir, agent_dir)
                    result = runner.run(agent_solver, mode, timeout_sec)
                    all_results[mode].append(result)
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}\n")
                    all_results[mode].append({
                        'case_id': case_id,
                        'test_mode': mode,
                        'status': 'ERROR',
                        'score': 0.0,
                        'error': str(e)
                    })
        
        # Compute summary statistics
        summary = self._compute_summary(agent_name, all_results)
        
        # Build complete evaluation result
        evaluation = {
            'agent_name': agent_name,
            'evaluation_date': datetime.now().isoformat(),
            'cases_dir': str(self.cases_dir),
            'agent_dir': str(agent_dir),
            'summary': summary,
            'results': all_results
        }
        
        # Print final summary
        self._print_summary(summary)
        
        return evaluation
    
    def _discover_cases(self, case_filter: Optional[List[str]] = None) -> List[Path]:
        """
        Discover all case directories.
        
        Args:
            case_filter: Optional list of case IDs to include
        
        Returns:
            List of case directory paths
        """
        case_dirs = []
        
        for item in sorted(self.cases_dir.iterdir()):
            if not item.is_dir():
                continue
            
            # Check if config.json exists
            if not (item / 'config.json').exists():
                continue
            
            # Apply filter if provided
            if case_filter and item.name not in case_filter:
                continue
            
            case_dirs.append(item)
        
        return case_dirs
    
    def _compute_summary(
        self,
        agent_name: str,
        all_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate summary statistics.
        
        Args:
            agent_name: Agent name
            all_results: Results for all modes
        
        Returns:
            Summary statistics dictionary
        """
        import numpy as np
        
        summary = {'agent_name': agent_name}
        
        for mode, results in all_results.items():
            # Filter valid results
            valid_results = [r for r in results if r.get('status') not in ['NOT_SUBMITTED', 'ERROR']]
            passed_results = [r for r in valid_results if r.get('status') == 'PASSED']
            
            total_cases = len(results)
            submitted = len([r for r in results if r.get('status') != 'NOT_SUBMITTED'])
            passed = len(passed_results)
            
            # Compute score statistics
            scores = [r.get('score', 0.0) for r in valid_results]
            
            mode_summary = {
                'total_cases': total_cases,
                'submitted': submitted,
                'passed': passed,
                'failed': submitted - passed,
                'pass_rate': passed / submitted if submitted > 0 else 0.0,
                'avg_score': float(np.mean(scores)) if scores else 0.0,
                'median_score': float(np.median(scores)) if scores else 0.0,
                'max_score': float(np.max(scores)) if scores else 0.0
            }
            
            # Tier statistics
            if valid_results:
                tier_counts = {1: 0, 2: 0, 3: 0}
                for r in valid_results:
                    if 'tier_levels' in r:
                        for level in r['tier_levels'].get('passed', []):
                            if level in tier_counts:
                                tier_counts[level] += 1
                
                mode_summary['tier_pass_rates'] = {
                    f'level_{i}': tier_counts[i] / len(valid_results)
                    for i in [1, 2, 3]
                }
            
            summary[mode] = mode_summary
        
        # Overall statistics (across both modes)
        all_scores = []
        for results in all_results.values():
            all_scores.extend([r.get('score', 0.0) for r in results if r.get('status') not in ['NOT_SUBMITTED', 'ERROR']])
        
        if all_scores:
            summary['overall'] = {
                'avg_score': float(np.mean(all_scores)),
                'median_score': float(np.median(all_scores)),
                'total_evaluations': len(all_scores)
            }
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary."""
        print(f"\n{'='*80}")
        print(f"📊 Evaluation Summary: {summary['agent_name']}")
        print(f"{'='*80}\n")
        
        for mode in ['fix_accuracy', 'fix_time']:
            if mode not in summary:
                continue
            
            mode_data = summary[mode]
            print(f"{'─'*80}")
            print(f"Mode: {mode}")
            print(f"{'─'*80}")
            print(f"  Total Cases: {mode_data['total_cases']}")
            print(f"  Submitted: {mode_data['submitted']}")
            print(f"  Passed: {mode_data['passed']}")
            print(f"  Failed: {mode_data['failed']}")
            print(f"  Pass Rate: {mode_data['pass_rate']:.1%}")
            print(f"  Avg Score: {mode_data['avg_score']:.2f}/100")
            print(f"  Median Score: {mode_data['median_score']:.2f}/100")
            print(f"  Max Score: {mode_data['max_score']:.2f}/100")
            
            if 'tier_pass_rates' in mode_data:
                print(f"  Tier Pass Rates:")
                for level in [1, 2, 3]:
                    rate = mode_data['tier_pass_rates'][f'level_{level}']
                    print(f"    L{level}: {rate:.1%}")
            print()
        
        if 'overall' in summary:
            print(f"{'─'*80}")
            print(f"Overall Performance")
            print(f"{'─'*80}")
            print(f"  Avg Score (all modes): {summary['overall']['avg_score']:.2f}/100")
            print(f"  Median Score: {summary['overall']['median_score']:.2f}/100")
            print(f"  Total Evaluations: {summary['overall']['total_evaluations']}")
        
        print(f"\n{'='*80}\n")

