"""Case runner: orchestrates execution of a single test case.

This module coordinates all steps of testing a single case:
1. Check/generate Oracle cache
2. Execute agent script
3. Compute error
4. Compute score
5. Compute tier levels
6. Compute specialized metrics
7. Save results
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Literal, Optional
import numpy as np

from ..sandbox.executor import execute_agent_script
from ..oracle import generate
from ..evaluation.validator import validate_solution
from ..metrics import compute_score, check_tier_levels, get_specialized_metrics_computer


class CaseRunner:
    """
    Orchestrates execution and evaluation of a single test case.
    
    This class encapsulates the entire test workflow for one case,
    separating concerns from the PDE-specific test classes.
    """
    
    def __init__(self, case_dir: Path, agent_dir: Optional[Path] = None):
        """
        Initialize case runner.
        
        Args:
            case_dir: Directory containing case definition (config.json, description.md)
            agent_dir: Optional agent results directory (for output organization)
        """
        self.case_dir = Path(case_dir)
        self.config = self._load_config()
        self.case_id = self.config['id']
        
        # Determine output directory
        if agent_dir:
            self.output_dir = agent_dir / self.case_id / 'test_output'
        else:
            self.output_dir = self.case_dir / 'test_output'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_output_dir = self.output_dir / 'agent_output'
        self.oracle_output_dir = self.output_dir / 'oracle_output'
        self.agent_output_dir.mkdir(parents=True, exist_ok=True)
        self.oracle_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load case configuration."""
        config_file = self.case_dir / 'config.json'
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file) as f:
            return json.load(f)
    
    def run(
        self,
        agent_script: Path,
        test_mode: Literal['fix_accuracy', 'fix_time'],
        timeout_sec: int = 300
    ) -> Dict[str, Any]:
        """
        Run complete evaluation for this case.
        
        Args:
            agent_script: Path to agent's solver.py
            test_mode: Evaluation mode ('fix_accuracy' or 'fix_time')
            timeout_sec: Timeout in seconds
        
        Returns:
            Complete test result dictionary
        """
        print(f"\n{'='*80}")
        print(f"🧪 Case: {self.case_id}")
        print(f"📋 Test Mode: {test_mode}")
        print(f"{'='*80}\n")
        
        # Get test parameters based on mode
        test_params = self._get_test_params(test_mode)
        
        # Generate Oracle reference (check cache first)
        print("🔮 Preparing oracle reference...")
        self._ensure_oracle(self.oracle_output_dir)
        
        # Execute agent script
        print(f"\n🤖 Executing agent script: {agent_script.name}")
        
        # Check if autonomous mode is enabled (default: True)
        eval_config = self.config.get('evaluation_config', {})
        exec_mode = eval_config.get('execution_mode', 'autonomous')
        
        if exec_mode == 'autonomous':
            print(f"   Mode: Autonomous (agent decides parameters)")
            agent_result = execute_agent_script(
                script_path=agent_script,
                outdir=self.agent_output_dir,
                timeout_sec=timeout_sec,
                mode='autonomous'
            )
        else:  # guided mode (backward compatible)
            print(f"   Mode: Guided (system provides parameters)")
            print(f"   Parameters: {test_params}")
            agent_result = execute_agent_script(
                script_path=agent_script,
                outdir=self.agent_output_dir,
                timeout_sec=timeout_sec,
                mode='guided',
                **test_params
            )
        
        if not agent_result.success:
            result = {
                'case_id': self.case_id,
                'test_mode': test_mode,
                'status': 'FAILED',
                'error': agent_result.error_message,
                'score': 0.0,
                'tier_levels': {
                    'passed': [],
                    'total': 3,
                    'level_details': {f'level_{i}': False for i in [1, 2, 3]}
                }
            }
            self._save_result(result, test_mode)
            self._print_result(result)
            return result
        
        print(f"   ✅ Agent execution completed in {agent_result.t_agent_run:.3f}s")
        
        # Compute error
        validation_result = validate_solution(
            agent_outdir=self.agent_output_dir,
            oracle_outdir=self.oracle_output_dir,
            evaluation_config=self.config['evaluation_config'],
            oracle_config=self.config['oracle_config']
        )
        
        error = validation_result.rel_L2_error
        
        if np.isnan(error):
            result = {
                'case_id': self.case_id,
                'test_mode': test_mode,
                'status': 'FAILED',
                'error': 'Error computation returned NaN',
                'score': 0.0,
                'tier_levels': {
                    'passed': [],
                    'total': 3,
                    'level_details': {f'level_{i}': False for i in [1, 2, 3]}
                }
            }
            self._save_result(result, test_mode)
            self._print_result(result)
            return result
        
        print(f"   📊 Relative L2 Error: {error:.6e}")
        
        # Compute score
        score = compute_score(
            mode=test_mode,
            runtime=agent_result.t_agent_run,
            error=error,
            target_error=self.config['evaluation_config']['target_error'],
            time_budget=self.config['evaluation_config'].get('time_budget', 60.0)
        )
        
        # Compute tier levels
        tier_levels = self._check_tier_levels(test_mode, agent_result.t_agent_run, error)
        
        # Build result
        result = {
            'case_id': self.case_id,
            'test_mode': test_mode,
            'status': 'PASSED' if score > 0 else 'FAILED',
            'runtime_sec': agent_result.t_agent_run,
            'error': float(error),
            'target_error': self.config['evaluation_config']['target_error'],
            'score': score,
            'test_params': test_params,
            'tier_levels': tier_levels
        }
        
        # Compute specialized metrics
        specialized_metrics = self._compute_specialized_metrics(result)
        if specialized_metrics:
            result['specialized_metrics'] = specialized_metrics
        
        # Save and print
        self._save_result(result, test_mode)
        self._print_result(result)
        
        return result
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        Get test parameters based on evaluation mode.
        
        This reads from config.json's evaluation_config section.
        """
        eval_config = self.config.get('evaluation_config', {})
        
        if mode not in eval_config:
            # Fallback to default parameters
            if mode == 'fix_accuracy':
                params = {'resolution': 128, 'degree': 2}
            elif mode == 'fix_time':
                params = {'resolution': 32, 'degree': 1}
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            params = eval_config[mode].copy()
        
        # Handle time-dependent PDEs
        if 'time' in self.config['oracle_config'].get('pde', {}):
            time_config = self.config['oracle_config']['pde']['time']
            if 'dt' not in params:
                # Use oracle dt as default
                params['dt'] = time_config.get('dt', 0.01)
        
        return params
    
    def _ensure_oracle(self, oracle_outdir: Path):
        """Ensure Oracle reference exists (generate if needed)."""
        # Check cache first
        oracle_cache = self.case_dir / 'oracle_cache'
        
        if oracle_cache.exists() and (oracle_cache / 'reference.npz').exists():
            # Use cached Oracle
            import shutil
            print(f"   ✅ Using cached Oracle from {oracle_cache.name}")
            # Copy cache to output directory
            for item in oracle_cache.iterdir():
                dest = oracle_outdir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
        else:
            # Generate fresh Oracle
            print(f"   🔄 Generating fresh Oracle...")
            oracle_config = self.config['oracle_config']
            generate(oracle_config, oracle_outdir)
            print(f"   ✅ Oracle saved to {oracle_outdir}")
    
    def _check_tier_levels(self, mode: str, runtime: float, error: float) -> Dict[str, Any]:
        """Check tier levels using centralized logic."""
        # Try to load tiers from oracle_baseline.json
        baseline_file = self.case_dir / 'oracle_baseline.json'
        
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline = json.load(f)
                tiers = baseline.get('tiers')
        elif 'tiers' in self.config:
            # Fallback to config.json (legacy name)
            tiers = self.config['tiers']
        elif 'difficulty_tiers' in self.config:
            # Support difficulty_tiers field name
            tiers = self.config['difficulty_tiers']
        else:
            # No tiers available
            return {
                'passed': [],
                'total': 3,
                'level_details': {f'level_{i}': False for i in [1, 2, 3]}
            }
        
        return check_tier_levels(mode, runtime, error, tiers)
    
    def _compute_specialized_metrics(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Compute PDE-specific metrics."""
        pde_type = self.config.get('tags', {}).get('pde_type', ['unknown'])[0]
        
        computer = get_specialized_metrics_computer(
            pde_type=pde_type,
            agent_output_dir=self.agent_output_dir,
            oracle_output_dir=self.oracle_output_dir,
            config=self.config
        )
        
        if computer is None:
            return None
        
        try:
            return computer.compute(result)
        except Exception as e:
            return {'error': f"Failed to compute specialized metrics: {str(e)}"}
    
    def _save_result(self, result: Dict[str, Any], test_mode: str):
        """Save result to JSON file."""
        result_file = self.output_dir / f'result_{test_mode}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _print_result(self, result: Dict[str, Any]):
        """Print formatted result summary."""
        print(f"\n{'─'*80}")
        print(f"📊 Test Result: {result['case_id']}")
        print(f"{'─'*80}")
        print(f"Status: {result['status']}")
        
        if 'runtime_sec' in result:
            print(f"Runtime: {result['runtime_sec']:.3f}s")
        if 'error' in result and isinstance(result['error'], (int, float)):
            print(f"Error: {result['error']:.6e}")
        elif 'error' in result:
            print(f"Error: {result['error']}")
        if 'target_error' in result:
            print(f"Target Error: {result['target_error']:.6e}")
        print(f"Score: {result.get('score', 0.0):.1f}/100")
        
        if 'tier_levels' in result:
            levels = result['tier_levels']
            passed = levels['passed']
            total = levels['total']
            print(f"Tier Levels: {len(passed)}/{total} passed {passed}")
        
        if 'specialized_metrics' in result:
            print(f"Specialized Metrics: {len(result['specialized_metrics'])} computed")
        
        print(f"{'─'*80}\n")

