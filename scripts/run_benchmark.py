#!/usr/bin/env python3
"""
PDEBench 统一评测入口

用法:
    # 评测单个LLM
    python run_benchmark.py --agent gpt-4o
    
    # 评测多个LLM
    python run_benchmark.py --agent gpt-4o sonnet-3.5 gemini
    
    # 只测试特定cases
    python run_benchmark.py --agent gpt-4o --cases poisson_basic heat_basic
    
    # 只测试特定方程类型
    python run_benchmark.py --agent gpt-4o --equation-types poisson heat
    
    # 跳过LLM调用，只评测已有solver
    python run_benchmark.py --agent gpt-4o --skip-generation
    
    # 使用已有solver.py
    python run_benchmark.py --agent gpt-4o --solver-path /Users/yusan/agent/pdebench/results/gpt-5.1/poisson_basic/solver.py --cases poisson_basic

流程:
    1. 从 data/benchmark.jsonl 加载cases
    2. 对每个case:
       a. 运行oracle获取参考解（带缓存）
       b. 生成prompt
       c. 调用LLM生成solver代码
       d. 执行solver，计算误差
       e. 单档通过率评测（精度→时间）
    3. 汇总结果，保存报告
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# 添加pdebench到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.core.prompt_builder import generate_prompt
from pdebench.core.llm_client import call_llm, LLMClient
from pdebench.metrics.specialized import get_specialized_metrics_computer


# =============================================================================
# 数据加载
# =============================================================================

def load_benchmark_cases(
    data_file: Path,
    case_filter: Optional[List[str]] = None,
    equation_types: Optional[List[str]] = None
) -> List[Dict]:
    """从benchmark.jsonl加载cases"""
    cases = []
    eq_types = [t.lower() for t in equation_types] if equation_types else None
    with open(data_file) as f:
        for line in f:
            if line.strip():
                case = json.loads(line)
                if case_filter is not None and case['id'] not in case_filter:
                    continue
                if eq_types is not None:
                    pde_type = case.get('oracle_config', {}).get('pde', {}).get('type', '').lower()
                    if pde_type not in eq_types:
                        continue
                cases.append(case)
    return cases


# =============================================================================
# Oracle求解器 (v2 - 统一入口)
# =============================================================================

def run_oracle(case: Dict, cache_dir: Path) -> Dict:
    """
    运行 Oracle 求解器获取 baseline（带缓存）
    
    使用统一 OracleSolver，输出 L2 reference 和参考时间。
    """
    case_id = case['id']
    cache_file = cache_dir / f"{case_id}.json"
    
    # 检查缓存
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        print(f"   ✅ Using cached oracle")
        return cached
    
    print(f"   🔮 Running oracle...")
    
    try:
        from pdebench.oracle import OracleSolver
        
        oracle = OracleSolver()
        oracle_config = case['oracle_config']
        
        # 调用统一 Oracle 求解器
        result = oracle.solve(oracle_config)
        
        # 构建缓存数据
        cached = {
            'error': result.baseline_error,
            'time': result.baseline_time,
            'case_id': case_id,
            'num_dofs': result.num_dofs,
            'solver_info': result.solver_info,
            # 存储参考解（用于误差计算）
            'reference': result.reference.tolist(),
        }
        
        # 保存缓存
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cached, f, indent=2)
        
        print(f"   ✅ Oracle: error={result.baseline_error:.2e}, time={result.baseline_time:.3f}s")
        return cached
        
    except Exception as e:
        import traceback
        print(f"   ⚠️  Oracle failed: {e}")
        traceback.print_exc()
        return {'error': 1e-2, 'time': 10.0, 'case_id': case_id, 'reference': None}


# =============================================================================
# 执行与评测
# =============================================================================

def execute_solver(solver_code: str, case: Dict, output_dir: Path, timeout: int = 300) -> Dict:
    """执行solver并返回结果"""
    from pdebench.sandbox.executor import execute_agent_function
    
    # 保存solver代码
    solver_path = output_dir / "solver.py"
    solver_path.write_text(solver_code)
    
    agent_output = output_dir / "agent_output"
    agent_output.mkdir(parents=True, exist_ok=True)
    
    # 执行
    result = execute_agent_function(
        script_path=solver_path,
        outdir=agent_output,
        case_spec=case,
        timeout_sec=timeout
    )
    
    if not result.success:
        return {
            'success': False,
            'error': None,
            'time': result.t_agent_run,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'error_message': result.error_message
        }
    
    return {
        'success': True,
        'error': None,  # 稍后计算
        'time': result.t_agent_run,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'agent_output': agent_output
    }


def compute_error(agent_output: Path, oracle_info: Dict) -> float:
    """
    计算相对L2误差
    
    Args:
        agent_output: Agent 输出目录（包含 solution.npz）
        oracle_info: Oracle 结果（包含 reference 列表）
    
    Returns:
        相对 L2 误差
    """
    try:
        # 加载 agent 解
        agent_sol = np.load(agent_output / "solution.npz")
        u_agent = agent_sol['u']
        
        # 从 oracle_info 获取参考解
        if oracle_info.get('reference') is None:
            print(f"   ⚠️  No reference solution in oracle cache")
            return float('nan')
        
        u_ref = np.array(oracle_info['reference'])
        
        # 处理形状不匹配
        if u_agent.shape != u_ref.shape:
            from scipy.ndimage import zoom
            factors = np.array(u_ref.shape) / np.array(u_agent.shape)
            u_agent = zoom(u_agent, factors, order=1)
        
        # 计算相对L2误差
        diff = u_agent - u_ref
        ref_norm = np.sqrt(np.sum(u_ref**2))
        
        if ref_norm < 1e-15:
            return np.sqrt(np.sum(diff**2))
        
        rel_L2 = np.sqrt(np.sum(diff**2)) / ref_norm
        
        return float(rel_L2)
        
    except Exception as e:
        print(f"   ⚠️  Error computation failed: {e}")
        return float('nan')


# =============================================================================
# 单Case流程
# =============================================================================

def run_single_case(
    case: Dict,
    agent_name: str,
    output_dir: Path,
    oracle_cache_dir: Path,
    solver_path_override: Optional[Path] = None,
    skip_generation: bool = False,
    timeout: int = 300
) -> Dict:
    """运行单个case的完整流程"""
    
    case_id = case['id']
    case_output = output_dir / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    
    oracle_output = case_output / "oracle_output"
    oracle_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📋 Case: {case_id}")
    print(f"{'='*60}")
    
    # Step 1: 获取oracle参考解
    oracle_info = run_oracle(case, oracle_cache_dir)
    _write_oracle_reference(case, oracle_info, oracle_output)
    
    # Step 2: 生成prompt
    prompt = generate_prompt(case, oracle_info)
    (case_output / "prompt.md").write_text(prompt)
    
    # Step 3: 调用LLM或加载已有solver
    solver_path = case_output / "solver.py"
    
    if solver_path_override is not None:
        if not solver_path_override.exists():
            return _make_error_result(case_id, 'SOLVER_NOT_FOUND', f"Solver path not found: {solver_path_override}")
        solver_code = solver_path_override.read_text()
    elif skip_generation and solver_path.exists():
        print(f"   ⏭️  Using existing solver")
        solver_code = solver_path.read_text()
    else:
        print(f"   🤖 Calling {agent_name}...")
        try:
            response = call_llm(agent_name, prompt)
            
            if not response.success:
                print(f"   ❌ LLM call failed: {response.error}")
                return _make_error_result(case_id, 'LLM_ERROR', response.error)
            
            solver_code = response.code
            (case_output / "llm_response.txt").write_text(response.raw_response)
            
            if response.usage:
                print(f"   📊 Tokens: in={response.usage['input_tokens']}, out={response.usage['output_tokens']}")
                
        except Exception as e:
            print(f"   ❌ LLM call failed: {e}")
            return _make_error_result(case_id, 'LLM_ERROR', str(e))
    
    # Step 4: 执行solver
    print(f"   🔧 Executing solver...")
    exec_result = execute_solver(solver_code, case, case_output, timeout)
    
    if not exec_result['success']:
        print(f"   ❌ Execution failed: {exec_result.get('error_message', 'Unknown')[:100]}")
        return _make_error_result(case_id, 'EXECUTION_ERROR', exec_result.get('error_message'), exec_result.get('stderr'))
    
    # Step 5: 计算误差
    error = compute_error(exec_result['agent_output'], oracle_info)
    
    if np.isnan(error):
        print(f"   ❌ Error computation failed")
        return _make_error_result(case_id, 'EVALUATION_ERROR', 'Error computation returned NaN')
    
    print(f"   📊 Error: {error:.2e}, Time: {exec_result['time']:.3f}s")
    
    # Step 6: 单档评测 (精度 -> 时间)
    eval_cfg = case.get('evaluation_config', {})
    legacy_tolerance = eval_cfg.get('tolerance', 1.2)
    accuracy_tolerance = eval_cfg.get('accuracy_tolerance', legacy_tolerance)
    time_tolerance = eval_cfg.get('time_tolerance', legacy_tolerance)
    target_error = oracle_info['error'] * accuracy_tolerance
    target_time = oracle_info['time'] * time_tolerance
    
    if error > target_error:
        status = 'FAIL'
        fail_reason = f"ACCURACY_FAIL: error={error:.2e} > target={target_error:.2e}"
    elif exec_result['time'] > target_time:
        status = 'FAIL'
        fail_reason = f"TIME_FAIL: time={exec_result['time']:.3f}s > target={target_time:.3f}s"
    else:
        status = 'PASS'
        fail_reason = None
    
    print(f"   ✅ Status: {status}")
    
    # 保存结果
    result = {
        'case_id': case_id,
        'status': status,
        'error': error,
        'time': exec_result['time'],
        'oracle_error': oracle_info['error'],
        'oracle_time': oracle_info['time'],
        'tolerance': legacy_tolerance,
        'accuracy_tolerance': accuracy_tolerance,
        'time_tolerance': time_tolerance,
        'target_error': target_error,
        'target_time': target_time,
        'fail_reason': fail_reason,
    }
    
    # 计算各math_type子榜指标
    math_types = case.get('pde_classification', {}).get('math_type', [])
    math_type_metrics = {}
    for mt in math_types:
        computer = get_specialized_metrics_computer(
            mt, exec_result['agent_output'], oracle_output, case
        )
        if computer is None:
            continue
        metrics = computer.compute({
            'runtime_sec': exec_result['time'],
            'error': error,
            'test_params': {}
        })
        math_type_metrics[mt] = metrics
    
    if math_type_metrics:
        result['math_types'] = math_types
        result['math_type_metrics'] = math_type_metrics
    
    with open(case_output / "result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def _make_error_result(case_id: str, status: str, error_msg: str, stderr: str = None) -> Dict:
    """创建错误结果"""
    result = {
        'case_id': case_id,
        'status': status,
        'error_message': error_msg
    }
    if stderr:
        result['stderr'] = stderr
    return result


def _write_oracle_reference(case: Dict, oracle_info: Dict, oracle_output: Path):
    """保存oracle参考解到oracle_output"""
    if oracle_info.get('reference') is None:
        return
    try:
        grid_cfg = case['oracle_config']['output']['grid']
        x = np.linspace(grid_cfg['bbox'][0], grid_cfg['bbox'][1], grid_cfg['nx'])
        y = np.linspace(grid_cfg['bbox'][2], grid_cfg['bbox'][3], grid_cfg['ny'])
        u_star = np.array(oracle_info['reference'])
        np.savez(oracle_output / "reference.npz", x=x, y=y, u_star=u_star)
    except Exception as e:
        print(f"   ⚠️  Failed to write oracle reference: {e}")


# =============================================================================
# 主流程
# =============================================================================

def run_benchmark(
    agents: List[str],
    output_dir: Path,
    data_file: Path,
    case_filter: Optional[List[str]] = None,
    equation_types: Optional[List[str]] = None,
    solver_path: Optional[Path] = None,
    skip_generation: bool = False,
    timeout: int = 300
):
    """运行完整benchmark"""
    
    print("\n" + "="*80)
    print("🚀 PDEBench - LLM/Code Agent Evaluation")
    print("="*80)
    print(f"📁 Data: {data_file}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Agents: {', '.join(agents)}")
    print(f"⏱️  Timeout: {timeout}s")
    print("="*80)
    
    # 验证agents
    for agent in agents:
        if agent not in LLMClient.SUPPORTED_AGENTS:
            print(f"❌ Unknown agent: {agent}")
            print(f"   Supported: {list(LLMClient.SUPPORTED_AGENTS.keys())}")
            sys.exit(1)
    
    # 加载cases
    cases = load_benchmark_cases(data_file, case_filter, equation_types)
    print(f"\n📊 Loaded {len(cases)} cases")
    
    if not cases:
        print("❌ No cases found!")
        sys.exit(1)
    
    oracle_cache_dir = output_dir / ".oracle_cache"
    all_results = {}
    
    for agent_name in agents:
        print(f"\n\n{'#'*80}")
        print(f"# Agent: {agent_name}")
        print(f"{'#'*80}")
        
        agent_output = output_dir / agent_name
        agent_results = []
        
        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}]", end="")
            result = run_single_case(
                case=case,
                agent_name=agent_name,
                output_dir=agent_output,
                oracle_cache_dir=oracle_cache_dir,
                solver_path_override=solver_path,
                skip_generation=skip_generation,
                timeout=timeout
            )
            agent_results.append(result)
        
        # 汇总统计
        summary = compute_summary(agent_name, agent_results)
        
        # 保存汇总
        with open(agent_output / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        all_results[agent_name] = summary
        print_summary(summary)
    
    # 保存总汇总
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)


def compute_summary(agent_name: str, results: List[Dict]) -> Dict:
    """计算汇总统计"""
    total = len(results)
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    errors = [r['error'] for r in results if r.get('status') in ['PASS', 'FAIL'] and r.get('error') is not None]
    times = [r['time'] for r in results if r.get('status') in ['PASS', 'FAIL'] and r.get('time') is not None]
    
    # math_type 子榜
    math_type_summary: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for mt in r.get('math_types', []):
            if mt not in math_type_summary:
                math_type_summary[mt] = {
                    'cases': 0,
                    'passed': 0,
                    'metric_sums': {},
                    'metric_counts': {}
                }
            math_type_summary[mt]['cases'] += 1
            if r.get('status') == 'PASS':
                math_type_summary[mt]['passed'] += 1
            metrics = r.get('math_type_metrics', {}).get(mt, {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    math_type_summary[mt]['metric_sums'][k] = (
                        math_type_summary[mt]['metric_sums'].get(k, 0.0) + float(v)
                    )
                    math_type_summary[mt]['metric_counts'][k] = (
                        math_type_summary[mt]['metric_counts'].get(k, 0) + 1
                    )
    
    for mt, info in math_type_summary.items():
        info['pass_rate'] = info['passed'] / info['cases'] if info['cases'] > 0 else 0.0
        avg_metrics = {}
        for k, total_val in info['metric_sums'].items():
            count = info['metric_counts'].get(k, 0)
            if count > 0:
                avg_metrics[k] = total_val / count
        info['avg_metrics'] = avg_metrics
        info.pop('metric_sums', None)
        info.pop('metric_counts', None)
    
    return {
        'agent_name': agent_name,
        'timestamp': datetime.now().isoformat(),
        'total_cases': total,
        'passed_cases': passed,
        'pass_rate': passed / total if total > 0 else 0,
        'avg_error': float(np.mean(errors)) if errors else None,
        'avg_time': float(np.mean(times)) if times else None,
        'math_type_summary': math_type_summary,
        'results': results
    }


def print_summary(summary: Dict):
    """打印汇总信息"""
    print(f"\n{'─'*60}")
    print(f"📊 Summary: {summary['agent_name']}")
    print(f"{'─'*60}")
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']} ({summary['pass_rate']:.1%})")
    if summary['avg_error'] is not None:
        print(f"Avg Error: {summary['avg_error']:.2e}")
    if summary['avg_time'] is not None:
        print(f"Avg Time: {summary['avg_time']:.3f}s")
    print(f"{'─'*60}")


# =============================================================================
# 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PDEBench LLM/Code Agent Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--agent', '-a',
        nargs='+',
        required=True,
        help=f"Agent name(s): {list(LLMClient.SUPPORTED_AGENTS.keys())}"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results'),
        help='Output directory (default: results/)'
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/benchmark.jsonl'),
        help='Benchmark data file (default: data/benchmark.jsonl)'
    )
    
    parser.add_argument(
        '--cases',
        nargs='+',
        default=None,
        help='Specific case IDs to run (default: all)'
    )
    
    parser.add_argument(
        '--equation-types',
        nargs='+',
        default=None,
        help='Equation type(s) to run, e.g., poisson heat (default: all)'
    )
    
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip LLM generation, use existing solvers'
    )
    
    parser.add_argument(
        '--solver-path',
        type=Path,
        default=None,
        help='Use an existing solver.py instead of LLM generation'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per case in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    root_dir = Path(__file__).parent.parent
    data_file = root_dir / args.data
    output_dir = root_dir / args.output
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    run_benchmark(
        agents=args.agent,
        output_dir=output_dir,
        data_file=data_file,
        case_filter=args.cases,
        equation_types=args.equation_types,
        solver_path=args.solver_path,
        skip_generation=args.skip_generation,
        timeout=args.timeout
    )


if __name__ == '__main__':
    main()
