#!/usr/bin/env python3
"""Compare multiple evaluation results and generate leaderboard.

Usage:
    python scripts/compare_results.py \\
        results/model_a/summary.json \\
        results/model_b/summary.json \\
        results/model_c/summary.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_summary(path: Path) -> Dict[str, Any]:
    """Load summary JSON."""
    with open(path) as f:
        return json.load(f)


def compare_models(summaries: List[tuple[str, Dict[str, Any]]]):
    """
    Generate leaderboard from multiple model results.
    
    Sorting order: PassRate(↑) → TotalTime(↓) → AvgError(↓)
    """
    # 准备排行数据
    leaderboard = []
    
    for model_name, summary in summaries:
        # 兼容新旧格式
        score = summary.get('leaderboard_score', {})
        summary_info = summary.get('summary', {})
        timing_info = summary.get('timing_statistics', {})
        accuracy_info = summary.get('accuracy_statistics', {})
        pass_at_eps = summary.get('pass_at_epsilon', {})
        
        # 优先从 leaderboard_score 读取，回退到其他字段
        pass_rate = score.get('pass_rate') or summary_info.get('success_rate', 0.0)
        total_time = score.get('total_agent_time') or timing_info.get('total_agent_time', float('inf'))
        avg_error = score.get('avg_error') or accuracy_info.get('avg_rel_L2_error', float('inf'))
        
        entry = {
            'model': model_name,
            'pass_rate': pass_rate,
            'total_time': total_time,
            'avg_error': avg_error,
            'pass_1e2': pass_at_eps.get('Pass@1e-2', 0.0),
            'pass_1e3': pass_at_eps.get('Pass@1e-3', 0.0),
            'pass_1e4': pass_at_eps.get('Pass@1e-4', 0.0),
        }
        leaderboard.append(entry)
    
    # 排序：PassRate 降序 → TotalTime 升序 → AvgError 升序
    leaderboard.sort(
        key=lambda x: (-x['pass_rate'], x['total_time'], x['avg_error'])
    )
    
    return leaderboard


def print_leaderboard(leaderboard: List[Dict[str, Any]]):
    """Print leaderboard table."""
    print(f"\n{'='*100}")
    print("🏆 PDEBench Leaderboard")
    print(f"{'='*100}")
    print(f"{'排名':<4} | {'模型':<20} | {'通过率':>8} | {'耗时(s)':>10} | {'平均误差':>12} | Pass@1e-2 | Pass@1e-3 | Pass@1e-4")
    print("-" * 100)
    
    for rank, entry in enumerate(leaderboard, 1):
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, '  ')
        
        pass_rate = entry['pass_rate']
        total_time = entry['total_time']
        avg_error = entry['avg_error']
        
        # 格式化
        time_str = f"{total_time:>10.3f}" if total_time < float('inf') else "      N/A"
        error_str = f"{avg_error:>12.3e}" if avg_error < float('inf') else "         N/A"
        
        print(f"{medal} #{rank:<2} | {entry['model']:<20} | {pass_rate*100:>7.1f}% | "
              f"{time_str} | {error_str} | "
              f"  {entry['pass_1e2']*100:>5.1f}% | "
              f"  {entry['pass_1e3']*100:>5.1f}% | "
              f"  {entry['pass_1e4']*100:>5.1f}%")
    
    print("=" * 100)
    print("\n📊 排序规则: PassRate(↑) → TotalTime(↓) → AvgError(↓)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare evaluation results and generate leaderboard'
    )
    parser.add_argument(
        'summaries',
        nargs='+',
        type=Path,
        help='Paths to summary.json files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save leaderboard to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load all summaries
    summaries = []
    for path in args.summaries:
        if not path.exists():
            print(f"⚠️  Warning: {path} not found, skipping")
            continue
        
        # 使用父目录名作为模型名
        model_name = path.parent.name
        summary = load_summary(path)
        summaries.append((model_name, summary))
    
    if not summaries:
        print("❌ No valid summary files found")
        return
    
    # Generate leaderboard
    leaderboard = compare_models(summaries)
    
    # Print
    print_leaderboard(leaderboard)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'leaderboard': leaderboard,
                'num_models': len(leaderboard),
            }, f, indent=2)
        print(f"\n💾 Leaderboard saved to: {args.output}")


if __name__ == '__main__':
    main()

