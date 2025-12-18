#!/usr/bin/env python3
"""Evaluate agent performance on the benchmark dataset.

Complete evaluation pipeline:
1. Load dataset entries
2. For each entry:
   a. Execute agent script in sandbox
   b. Generate oracle ground truth
   c. Validate solution against oracle
   d. Compute metrics
3. Generate summary report

Usage:
    python scripts/evaluate_agent.py --dataset datasets/level_2_1_basic.jsonl \
                                      --agent-script path/to/agent_solver.py \
                                      --outdir results/run_001
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.datasets.schema import load_dataset
from pdebench.sandbox.executor import execute_agent_script_with_oracle
from pdebench.evaluation.validator import validate_solution


def evaluate_single_case(
    case_id: str,
    agent_script: Path,
    oracle_config: Dict[str, Any],
    evaluation_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Evaluate agent on a single benchmark case.
    
    Args:
        case_id: Case identifier
        agent_script: Path to agent script
        oracle_config: Oracle configuration (case spec)
        evaluation_config: Evaluation parameters
        output_dir: Output directory for this case
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'─'*80}")
    print(f"🔬 正在评测: {case_id}")
    print(f"{'─'*80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()
    
    # Execute agent script and generate oracle
    try:
        exec_result, agent_outdir, oracle_outdir = execute_agent_script_with_oracle(
            script_path=agent_script,
            oracle_config=oracle_config,
            base_outdir=output_dir,
            evaluation_config=evaluation_config
        )
        
        exec_icon = "✅" if exec_result.success else "❌"
        print(f"  {exec_icon} 执行状态: {'成功' if exec_result.success else '失败'}")
        print(f"  ⏱️  运行耗时: {exec_result.wall_time_sec:.2f}s")
        
        if not exec_result.success:
            error_msg = exec_result.stderr or "未知错误"
            # Truncate long error messages
            print(f"  ⚠️  错误信息: {error_msg}")
            
            return {
                'case_id': case_id,
                'success': False,
                'execution': exec_result.to_dict(),
                'validation': None,
                'total_time_sec': time.time() - t_start,
            }
        
        # Validate solution (计时)
        t_val_start = time.time()
        validation_result = validate_solution(
            agent_outdir=agent_outdir,
            oracle_outdir=oracle_outdir,
            evaluation_config=evaluation_config
        )
        exec_result.t_validation = time.time() - t_val_start
        
        val_icon = "🎯" if validation_result.is_valid else "❌"
        print(f"  {val_icon} 验证结果: {'通过' if validation_result.is_valid else '未通过'}")
        print(f"  📊 {validation_result.reason}")
        
        # 提取 DoF 信息（从 oracle problem_info）
        dof = None
        problem_info_path = oracle_outdir / 'problem_info.json'
        if problem_info_path.exists():
            try:
                with open(problem_info_path) as f:
                    info = json.load(f)
                    dof = info.get('num_dofs')
            except:
                pass
        
        result = {
            'case_id': case_id,
            'success': validation_result.is_valid,
            'execution': exec_result.to_dict(),
            'validation': validation_result.to_dict(),
            'timing': {
                't_agent': exec_result.t_agent_run,
                't_oracle': exec_result.t_oracle_run,
                't_validation': exec_result.t_validation,
                't_total': time.time() - t_start,
            },
            'cost': {  # 新增：代价指标
                'dof': dof,
                'time_per_dof': exec_result.t_agent_run / dof if dof else None,
            },
        }
        
        # Save case result
        with open(output_dir / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        
        return {
            'case_id': case_id,
            'success': False,
            'execution': None,
            'validation': None,
            'error': str(e),
            'total_time_sec': time.time() - t_start,
        }


def generate_mock_agent_script(
    entry_id: str,
    oracle_config: Dict[str, Any],
    output_path: Path
):
    """
    Generate a mock agent script that uses the oracle solver.
    
    This is useful for testing the evaluation pipeline without a real agent.
    
    Args:
        entry_id: Dataset entry ID
        oracle_config: Oracle configuration
        output_path: Path to save mock script
    """
    # Convert config to Python repr format (not JSON)
    import pprint
    config_str = pprint.pformat(oracle_config, indent=2, width=100)
    
    script_content = f'''#!/usr/bin/env python3
"""Mock agent script for case: {entry_id}

This script uses the Oracle solver to verify the evaluation pipeline.
In a real scenario, this would be generated by an AI agent.
"""

import argparse
import sys
from pathlib import Path

# Import oracle solver
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pdebench.oracle import generate, solve_case

# Oracle configuration (hidden from real agent)
ORACLE_CONFIG = {config_str}


def main():
    parser = argparse.ArgumentParser(description='PDE solver')
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    
    # Update config with provided parameters
    config = ORACLE_CONFIG.copy()
    config['mesh']['resolution'] = args.resolution
    config['fem']['degree'] = args.degree
    
    # Use oracle solver (this simulates perfect agent performance)
    generate(config, outdir)
    solve_case(config, outdir)


if __name__ == '__main__':
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    output_path.chmod(0o755)  # Make executable


def generate_summary_report(
    results: List[Dict[str, Any]],
    output_file: Path
):
    """
    Generate summary report from evaluation results.
    
    Args:
        results: List of case evaluation results
        output_file: Path to save report
    """
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.get('success', False))
    failed_cases = total_cases - successful_cases
    
    success_rate = successful_cases / total_cases if total_cases > 0 else 0.0
    
    # 计算 Pass@ε 统计（多档精度通过率）
    pass_at_eps = {}
    precision_levels = ['1e-2', '1e-3', '1e-4', 'default']
    
    for level in precision_levels:
        passed_count = 0
        for r in results:
            val = r.get('validation')
            if val and 'target' in val:
                passed_levels = val['target'].get('passed_levels', [])
                if level in passed_levels:
                    passed_count += 1
        if total_cases > 0:
            pass_at_eps[f'Pass@{level}'] = passed_count / total_cases
        else:
            pass_at_eps[f'Pass@{level}'] = 0.0
    
    # Compute statistics
    valid_results = [r for r in results if r.get('validation') is not None]
    
    if valid_results:
        rel_L2_errors = [
            r['validation']['accuracy']['rel_L2_error']
            for r in valid_results
            if not (r['validation']['accuracy']['rel_L2_error'] != r['validation']['accuracy']['rel_L2_error'])  # not NaN
        ]
        
        avg_L2_error = sum(rel_L2_errors) / len(rel_L2_errors) if rel_L2_errors else float('nan')
        max_L2_error = max(rel_L2_errors) if rel_L2_errors else float('nan')
        min_L2_error = min(rel_L2_errors) if rel_L2_errors else float('nan')
    else:
        avg_L2_error = float('nan')
        max_L2_error = float('nan')
        min_L2_error = float('nan')
    
    # Group by level（从 dataset entry 获取）
    level_stats = {}
    for r in results:
        level = r.get('level', 'unknown')
        
        if level not in level_stats:
            level_stats[level] = {'total': 0, 'passed': 0, 'pass_rate': 0.0}
        
        level_stats[level]['total'] += 1
        if r.get('success', False):
            level_stats[level]['passed'] += 1
    
    # 计算每个 level 的通过率
    for level in level_stats:
        total = level_stats[level]['total']
        passed = level_stats[level]['passed']
        level_stats[level]['pass_rate'] = passed / total if total > 0 else 0.0
    
    # 计算时间统计（仅 agent 时间）
    agent_times = [r.get('timing', {}).get('t_agent', 0.0) for r in results if r.get('success')]
    avg_agent_time = sum(agent_times) / len(agent_times) if agent_times else 0.0
    total_agent_time = sum(agent_times)
    
    # 计算 Leaderboard 评分（按 PassRate → Time → Error 排序）
    leaderboard_score = {
        'pass_rate': success_rate,  # 主排序
        'total_agent_time': total_agent_time,  # 次排序（越低越好）
        'avg_error': avg_L2_error,  # 第三排序（越低越好）
        'ranking_formula': 'PassRate(↑) → TotalTime(↓) → AvgError(↓)',
    }
    
    report = {
        'summary': {
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'success_rate': success_rate,
        },
        'leaderboard_score': leaderboard_score,  # 新增：排行榜评分
        'pass_at_epsilon': pass_at_eps,  # 多档精度通过率
        'accuracy_statistics': {
            'avg_rel_L2_error': avg_L2_error,
            'min_rel_L2_error': min_L2_error,
            'max_rel_L2_error': max_L2_error,
        },
        'timing_statistics': {
            'total_agent_time': total_agent_time,
            'avg_agent_time': avg_agent_time,
        },
        'level_breakdown': level_stats,
        'cases': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print beautiful summary to console
    print(f"\n{'='*80}")
    print("📊 评测结果详情")
    print(f"{'='*80}")
    print(f"{'Case ID':<28} | {'状态':^6} | {'耗时(s)':>8} | {'DoF':>8} | {'备注':<20}")
    print("-" * 80)
    
    for r in results:
        case_id = r.get('case_id', 'unknown')
        success = r.get('success', False)
        status_icon = "✅" if success else "❌"
        
        # Extract metrics
        timing = r.get('timing', {})
        t_agent = timing.get('t_agent', 0.0)
        
        # Extract DoF
        cost = r.get('cost', {})
        dof = cost.get('dof', None)
        dof_str = f"{dof:8d}" if dof else "    N/A"
        
        val_info = r.get('validation', {})
        if val_info:
            accuracy = val_info.get('accuracy', {})
            rel_error = accuracy.get('rel_L2_error', float('nan'))
            note = f"L2={rel_error:.2e}"
        else:
            error_msg = r.get('error', 'execution failed')
            # Truncate long error messages
            note = error_msg[:20] if len(error_msg) <= 20 else error_msg[:17] + "..."
        
        print(f"{status_icon} {case_id:<26} | {'PASS' if success else 'FAIL':^6} | {t_agent:>8.3f} | {dof_str} | {note:<20}")
    
    print("-" * 80)
    
    # Calculate total time (分离 agent 和总时间)
    agent_times_all = [r.get('timing', {}).get('t_agent', 0.0) for r in results]
    total_agent_time = sum(agent_times_all)
    
    total_time_all = sum(r.get('timing', {}).get('t_total', 0.0) for r in results)
    
    print(f"\n{'='*80}")
    print("🏆 最终得分摘要")
    print(f"{'='*80}")
    print(f"⏱️  Agent 总耗时: {total_agent_time:.4f} 秒 (不含 Oracle)")
    print(f"⏱️  评测总耗时: {total_time_all:.4f} 秒 (含 Oracle + Validation)")
    print(f"✓  基础通过率: {successful_cases}/{total_cases} ({success_rate*100:.1f}%)")
    
    # 显示 Pass@ε 统计
    print(f"\n📊 多档精度通过率 (Pass@ε):")
    for eps_name, pass_rate in sorted(pass_at_eps.items()):
        print(f"   {eps_name}: {pass_rate*100:.1f}%")
    
    # 显示 Level breakdown
    if level_stats and len(level_stats) > 1:  # 只在有多个 level 时显示
        print(f"\n📋 按难度分级统计:")
        for level, stats in sorted(level_stats.items()):
            print(f"   Level {level}: {stats['passed']}/{stats['total']} ({stats['pass_rate']*100:.1f}%)")
    
    if not all(x != x for x in [avg_L2_error]):  # Check if not NaN
        print(f"\n📈 误差统计 (通过案例):")
        print(f"   平均相对误差: {avg_L2_error:.3e}")
        print(f"   最小误差: {min_L2_error:.3e}")
        print(f"   最大误差: {max_L2_error:.3e}")
    
    print(f"{'='*80}")
    print(f"\n💾 详细报告已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate agent on PDE benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with mock agent (for testing)
  python scripts/evaluate_agent.py --dataset datasets/level_2_1_basic.jsonl \\
                                    --mock-agent --outdir results/mock_test
  
  # Evaluate with real agent script
  python scripts/evaluate_agent.py --dataset datasets/level_2_1_basic.jsonl \\
                                    --agent-script my_agent_solver.py \\
                                    --outdir results/agent_run_001
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset JSONL file'
    )
    
    parser.add_argument(
        '--agent-script',
        type=Path,
        help='Path to agent script (if not using mock agent)'
    )
    
    parser.add_argument(
        '--mock-agent',
        action='store_true',
        help='Use mock agent (Oracle solver) for testing'
    )
    
    parser.add_argument(
        '--outdir',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of cases to evaluate (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.mock_agent and not args.agent_script:
        parser.error("Must provide either --agent-script or --mock-agent")
    
    if not args.dataset.exists():
        parser.error(f"Dataset file not found: {args.dataset}")
    
    # Load dataset
    print(f"\n{'='*80}")
    print(f"🚀 PDEBench 评测系统")
    print(f"{'='*80}")
    print(f"📁 数据集: {args.dataset}")
    
    entries = load_dataset(str(args.dataset))
    
    if args.limit:
        entries = entries[:args.limit]
        print(f"📦 加载案例: {len(entries)} 个 (限制前 {args.limit} 个)")
    else:
        print(f"📦 加载案例: {len(entries)} 个")
    
    print(f"🤖 Agent 模式: {'Mock Agent (Oracle)' if args.mock_agent else f'自定义脚本 ({args.agent_script})'}")
    print(f"📤 输出目录: {args.outdir}")
    print(f"{'='*80}")
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each case
    results = []
    
    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] 📋 案例: {entry.id}")
        
        case_outdir = args.outdir / entry.id
        
        # Generate mock agent script if needed
        if args.mock_agent:
            agent_script = case_outdir / 'mock_agent.py'
            agent_script.parent.mkdir(parents=True, exist_ok=True)
            generate_mock_agent_script(entry.id, entry.oracle_config, agent_script)
        else:
            agent_script = args.agent_script
        
        # Evaluate
        result = evaluate_single_case(
            case_id=entry.id,
            agent_script=agent_script,
            oracle_config=entry.oracle_config,
            evaluation_config=entry.evaluation_config,
            output_dir=case_outdir
        )
        
        # 添加 level 信息
        result['level'] = entry.level
        
        results.append(result)
    
    # Generate summary report
    summary_file = args.outdir / 'summary.json'
    generate_summary_report(results, summary_file)


if __name__ == '__main__':
    main()

