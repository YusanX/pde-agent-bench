#!/usr/bin/env python3
"""
测试所有 case 的 Oracle 求解器

验证每个 case 的 ground truth 是否都能正确生成。
支持多后端：dolfinx（默认）、firedrake、dealii。

用法示例：
  # 测试 DOLFINx（默认）
  python scripts/test_oracle_cases.py --equation-types poisson

  # 测试 Firedrake（需先 source venv-firedrake/bin/activate）
  python scripts/test_oracle_cases.py --solver-library firedrake --equation-types poisson

  # 测试 deal.II（需先设置 DEAL_II_DIR）
  python scripts/test_oracle_cases.py --solver-library dealii --equation-types poisson
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import traceback

# 添加 pdebench 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def _normalize_num_dofs(value):
    """Convert scalar/tuple numpy dof counts to a plain total integer."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(sum(_normalize_num_dofs(v) for v in value))
    return int(value)


def _import_oracle(solver_library: str):
    """根据 solver_library 导入对应的 oracle，避免在不同环境下强制交叉依赖。"""
    if solver_library == "firedrake":
        try:
            from pdebench.oracle.firedrake_oracle import FiredrakeOracleSolver
            return FiredrakeOracleSolver()
        except ImportError as e:
            print(f"❌ 无法导入 Firedrake oracle: {e}")
            print("   请先激活 Firedrake 虚拟环境：source venv-firedrake/bin/activate")
            sys.exit(1)
    elif solver_library == "dealii":
        try:
            from pdebench.oracle.dealii_oracle import DealIIOracleSolver
            return DealIIOracleSolver()
        except ImportError as e:
            print(f"❌ 无法导入 deal.II oracle: {e}")
            print("   请先激活 PDEBench Python 环境，并设置 DEAL_II_DIR")
            sys.exit(1)
    else:
        try:
            from pdebench.oracle import OracleSolver
            return OracleSolver()
        except ImportError as e:
            print(f"❌ 无法导入 DOLFINx oracle: {e}")
            print("   请先激活含 DOLFINx 的 conda 环境：conda activate pdebench")
            sys.exit(1)


def load_all_cases(data_file: Path, equation_types: List[str] | None = None) -> List[Dict[str, Any]]:
    """加载所有 cases，可按方程类型过滤"""
    cases = []
    eq_types = [t.lower() for t in equation_types] if equation_types else None
    with open(data_file) as f:
        for line in f:
            if line.strip():
                case = json.loads(line)
                if eq_types is not None:
                    pde_type = case.get('oracle_config', {}).get('pde', {}).get('type', '').lower()
                    if pde_type not in eq_types:
                        continue
                cases.append(case)
    return cases


def test_oracle_case(
    case: Dict[str, Any],
    oracle_solver,
    solver_library: str = "dolfinx",
) -> Dict[str, Any]:
    """
    测试单个 case 的 Oracle 求解器
    
    Returns:
        result: 包含成功/失败信息的字典
    """
    case_id = case['id']
    
    result = {
        'case_id': case_id,
        'success': False,
        'error': None,
        'error_value': None,
        'time': None,
        'num_dofs': None,
        'reference_shape': None
    }
    
    try:
        start_time = time.time()
        
        # 运行 Oracle 求解器
        # Firedrake/deal.II oracle 直接 .solve(spec)；
        # DOLFINx unified oracle 需要 solver_library 参数。
        if solver_library in {"firedrake", "dealii"}:
            oracle_result = oracle_solver.solve(case['oracle_config'])
        else:
            oracle_result = oracle_solver.solve(case['oracle_config'], solver_library=solver_library)
        
        elapsed_time = time.time() - start_time
        
        import numpy as np
        reference = oracle_result.reference
        
        if reference is None or reference.size == 0:
            result['error'] = "Reference solution is None or empty"
            return result

        # --- 健壮的 NaN 检查逻辑 ---
        # 计算有效点数量（非 NaN 且非 Inf）
        valid_mask = np.isfinite(reference)
        valid_count = np.sum(valid_mask)
        total_count = reference.size
        valid_ratio = valid_count / total_count

        # 定义不同几何形状的最低有效比例门槛
        # L-shape 理论有效面积 0.75, Annulus 可能更小
        min_threshold = 0.3  # 最低要求 30% 的点在域内
        
        if valid_count == 0:
            result['error'] = "Reference is all NaN/Inf"
            return result
            
        if valid_ratio < min_threshold:
            result['error'] = f"Too many NaNs: only {valid_ratio:.1%} of points are inside the domain"
            return result

        # 为了后续 baseline_error 计算不崩，将域外的 NaN/Inf 填充为 0
        # 只有当 valid_ratio 达标时才允许填充
        if valid_count < total_count:
            reference[~valid_mask] = 0.0
            
        # 成功
        result['success'] = True
        result['error_value'] = float(oracle_result.baseline_error)
        result['time'] = elapsed_time
        result['num_dofs'] = _normalize_num_dofs(oracle_result.num_dofs)
        result['reference_shape'] = oracle_result.reference.shape
        
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return result


def print_progress_bar(current: int, total: int, width: int = 50):
    """打印进度条"""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percent = progress * 100
    print(f'\r[{bar}] {current}/{total} ({percent:.1f}%)', end='', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Test Oracle solver for benchmark cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--equation-types',
        nargs='+',
        default=None,
        help='Equation type(s) to test, e.g., poisson heat',
    )
    parser.add_argument(
        '--solver-library',
        default='dolfinx',
        choices=['dolfinx', 'firedrake', 'dealii'],
        help='Solver backend to use (default: dolfinx)',
    )
    parser.add_argument(
        '--data-file',
        default=None,
        help='Path to benchmark.jsonl (default: data/benchmark.jsonl)',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file path (default: oracle_test_results_{library}.json)',
    )
    args = parser.parse_args()

    solver_library = args.solver_library

    print("="*70)
    print("🧪 PDEBench Oracle Cases Test")
    print("="*70)
    print(f"📚 Solver Library: {solver_library}")
    print()

    # 加载所有 cases
    data_file = (
        Path(args.data_file)
        if args.data_file
        else Path(__file__).parent.parent / 'data' / 'benchmark_v2.jsonl'
    )

    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)

    print(f"📁 Loading cases from: {data_file}")
    cases = load_all_cases(data_file, equation_types=args.equation_types)
    if args.equation_types:
        print(f"🎯 Filtered equation types: {', '.join(args.equation_types)}")
    print(f"✅ Loaded {len(cases)} cases")
    print()

    # 创建 Oracle 求解器
    oracle_solver = _import_oracle(solver_library)

    # 测试所有 cases
    print(f"🔬 Testing Oracle solver ({solver_library}) for each case...")
    print()

    results = []
    success_count = 0
    failed_cases = []

    start_total_time = time.time()

    for i, case in enumerate(cases, 1):
        case_id = case['id']
        print(f"[{i}/{len(cases)}] Testing: {case_id:<40}", end='', flush=True)

        result = test_oracle_case(case, oracle_solver, solver_library=solver_library)
        results.append(result)
        
        # 无论成功失败，都尝试获取时间
        t_str = f"{result['time']:.2f}s" if result['time'] else "N/A"
        
        if result['success']:
            success_count += 1
            print(f" ✅ (err={result['error_value']:.2e}, time={t_str})")
        else:
            failed_cases.append(result)
            # 打印失败原因和已消耗的时间
            print(f" ❌ {result['error'][:40]}... (time={t_str})")
    
    total_time = time.time() - start_total_time

    # 打印汇总
    print()
    print("="*70)
    print("📊 Test Summary")
    print("="*70)
    print(f"Solver Library: {solver_library}")
    print(f"Total cases:    {len(cases)}")
    print(f"✅ Passed:      {success_count} ({success_count/len(cases)*100:.1f}%)")
    print(f"❌ Failed:      {len(failed_cases)} ({len(failed_cases)/len(cases)*100:.1f}%)")
    print(f"⏱️  Total time:  {total_time:.1f}s ({total_time/len(cases):.2f}s per case)")
    print()
    
    # 如果有失败的 cases，显示详细信息
    if failed_cases:
        print("="*70)
        print("❌ Failed Cases Details")
        print("="*70)
        for result in failed_cases:
            print(f"\nCase: {result['case_id']}")
            print(f"Error: {result['error']}")
            if 'traceback' in result:
                print("Traceback:")
                print(result['traceback'][:500])  # 限制长度
                if len(result['traceback']) > 500:
                    print("... (truncated)")
        print()
    
    # 成功 cases 的统计
    if success_count > 0:
        print("="*70)
        print("📈 Success Cases Statistics")
        print("="*70)
        
        success_results = [r for r in results if r['success']]
        
        # 误差统计
        errors = [r['error_value'] for r in success_results if r['error_value'] is not None]
        times = [r['time'] for r in success_results if r['time'] is not None]
        dofs = [r['num_dofs'] for r in success_results if r['num_dofs'] is not None]
        
        import numpy as np
        
        if errors:
            print(f"\nOracle Errors (relative L2):")
            print(f"  Min:    {np.min(errors):.2e}")
            print(f"  Max:    {np.max(errors):.2e}")
            print(f"  Mean:   {np.mean(errors):.2e}")
            print(f"  Median: {np.median(errors):.2e}")
            
            # 误差分布
            zero_error = sum(1 for e in errors if e == 0)
            if zero_error > 0:
                print(f"  Zero errors: {zero_error}/{len(errors)} cases")
        
        if times:
            print(f"\nOracle Times:")
            print(f"  Min:    {np.min(times):.3f}s")
            print(f"  Max:    {np.max(times):.3f}s")
            print(f"  Mean:   {np.mean(times):.3f}s")
            print(f"  Median: {np.median(times):.3f}s")
        
        if dofs:
            print(f"\nDegrees of Freedom:")
            print(f"  Min:    {np.min(dofs):,}")
            print(f"  Max:    {np.max(dofs):,}")
            print(f"  Mean:   {np.mean(dofs):,.0f}")
            print(f"  Median: {np.median(dofs):,.0f}")
        
        print()
    
    # 保存结果
    if args.output:
        output_file = Path(args.output)
    else:
        suffix = f"_{solver_library}" if solver_library != "dolfinx" else ""
        output_file = Path(__file__).parent.parent / f'oracle_test_results{suffix}.json'

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'solver_library': solver_library,
                'total_cases': len(cases),
                'passed': success_count,
                'failed': len(failed_cases),
                'pass_rate': success_count / len(cases),
                'total_time': total_time
            },
            'results': results
        }, f, indent=2)
    
    print(f"💾 Detailed results saved to: {output_file}")
    print()
    
    # 按 PDE 类型分组统计
    print("="*70)
    print("📊 Results by PDE Type")
    print("="*70)
    
    pde_types = {}
    for case, result in zip(cases, results):
        pde_type = case.get('pde_classification', {}).get('equation_type', 'unknown')
        if pde_type not in pde_types:
            pde_types[pde_type] = {'total': 0, 'passed': 0}
        pde_types[pde_type]['total'] += 1
        if result['success']:
            pde_types[pde_type]['passed'] += 1
    
    for pde_type, stats in sorted(pde_types.items()):
        pass_rate = stats['passed'] / stats['total'] * 100
        status = '✅' if stats['passed'] == stats['total'] else '⚠️'
        print(f"{status} {pde_type:30s}: {stats['passed']}/{stats['total']} ({pass_rate:.0f}%)")
    
    print()
    print("="*70)
    
    # 退出码
    if len(failed_cases) == 0:
        print("🎉 All tests passed!")
        print("="*70)
        sys.exit(0)
    else:
        print(f"⚠️  {len(failed_cases)} test(s) failed!")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
