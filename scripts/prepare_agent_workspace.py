#!/usr/bin/env python3
"""
为 Code Agent 准备工作空间（仅生成 oracle_output 和 prompt.md）

用法:
    # 为单个 agent 准备
    python prepare_agent_workspace.py --agent openhands
    
    # 为多个 agent 准备
    python prepare_agent_workspace.py --agent openhands mini-swe-agent
    
    # 只准备特定 cases
    python prepare_agent_workspace.py --agent openhands --cases poisson_basic heat_basic
    
    # 只准备特定方程类型
    python prepare_agent_workspace.py --agent openhands --equation-types poisson heat

功能:
    1. 从 data/benchmark.jsonl 加载 cases
    2. 对每个 case:
       a. 运行 oracle 获取参考解（带缓存）
       b. 生成 prompt（与纯 LLM 完全一致）
       c. 保存到 results/{agent_name}/{case_id}/oracle_output/ 和 prompt.md
    3. 不执行代码生成、执行、评测等环节
    
输出目录结构:
    results/{agent_name}/{case_id}/
        ├── oracle_output/
        │   └── reference.npz
        └── prompt.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# 添加 pdebench 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.core.prompt_builder import generate_prompt


# =============================================================================
# 数据加载（复用 run_benchmark.py 的逻辑）
# =============================================================================

def load_benchmark_cases(
    data_file: Path,
    case_filter: Optional[List[str]] = None,
    equation_types: Optional[List[str]] = None
) -> List[Dict]:
    """从 benchmark.jsonl 加载 cases"""
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
# Oracle 求解器（复用 run_benchmark.py 的逻辑）
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


def write_oracle_reference(case: Dict, oracle_info: Dict, oracle_output: Path):
    """保存 oracle 参考解到 oracle_output 目录"""
    if oracle_info.get('reference') is None:
        print(f"   ⚠️  No reference solution available")
        return
    
    try:
        grid_cfg = case['oracle_config']['output']['grid']
        x = np.linspace(grid_cfg['bbox'][0], grid_cfg['bbox'][1], grid_cfg['nx'])
        y = np.linspace(grid_cfg['bbox'][2], grid_cfg['bbox'][3], grid_cfg['ny'])
        u_star = np.array(oracle_info['reference'])
        
        oracle_output.mkdir(parents=True, exist_ok=True)
        np.savez(oracle_output / "reference.npz", x=x, y=y, u_star=u_star)
        print(f"   ✅ Saved oracle reference to: {oracle_output / 'reference.npz'}")
        
    except Exception as e:
        print(f"   ⚠️  Failed to write oracle reference: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 单 Case 准备流程
# =============================================================================

def prepare_single_case(
    case: Dict,
    agent_name: str,
    output_dir: Path,
    oracle_cache_dir: Path
) -> bool:
    """
    为单个 case 准备工作空间
    
    Returns:
        True if successful, False otherwise
    """
    case_id = case['id']
    case_output = output_dir / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    
    oracle_output = case_output / "oracle_output"
    
    print(f"\n{'='*60}")
    print(f"📋 Case: {case_id}")
    print(f"{'='*60}")
    
    # Step 1: 获取 oracle 参考解
    oracle_info = run_oracle(case, oracle_cache_dir)
    write_oracle_reference(case, oracle_info, oracle_output)
    
    # Step 2: 生成 prompt（与纯 LLM 完全一致）
    prompt = generate_prompt(case, oracle_info)

    prompt += "## ⚠️ Stricly follow the following constraints:\n\n"
    prompt += "- Do not run any file finding or command execution commands in the terminal, ONLY create `solver.py` and other necessary files in the SAME directory as this `prompt.md`.(/results/openhands)\n"
    prompt += "- You CANNOT access `oracle_output/` or any reference solutions.\n"
    prompt += "- This run is NON-INTERACTIVE (no user will reply). DO NOT ask questions or request clarification.\n"
    prompt += "- you **have no ACCESS** to the directory 'results/openhands_workspaces'\n"
    prompt += "- Proceed with reasonable assumptions if anything is unspecified; do not pause.\n"
    prompt += "- Your ONLY task: create/overwrite a file named `solver.py` and other necessary files mentioned above in the SAME directory as this `prompt.md`.(/results/openhands)\n"
    prompt += "- `solver.py` must be complete, runnable Python and must define `solve(case_spec: dict) -> dict` exactly as required above.\n"
    prompt += "- After writing `solver.py`, STOP. Do not output explanations, summaries, or extra files.\n"
    
    prompt_file = case_output / "prompt.md"
    prompt_file.write_text(prompt)
    print(f"   ✅ Saved prompt to: {prompt_file}")
    
    print(f"   ✅ Workspace prepared for {agent_name}/{case_id}")
    return True


# =============================================================================
# 主流程
# =============================================================================

def prepare_agent_workspaces(
    agents: List[str],
    output_dir: Path,
    data_file: Path,
    case_filter: Optional[List[str]] = None,
    equation_types: Optional[List[str]] = None
):
    """为多个 agent 准备工作空间"""
    
    print("\n" + "="*80)
    print("🚀 PDEBench - Code Agent Workspace Preparation")
    print("="*80)
    print(f"📁 Data: {data_file}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Agents: {', '.join(agents)}")
    print("="*80)
    
    # 加载 cases
    cases = load_benchmark_cases(data_file, case_filter, equation_types)
    print(f"\n📊 Loaded {len(cases)} cases from benchmark")
    
    if not cases:
        print("❌ No cases to prepare!")
        sys.exit(1)
    
    # 共享的 oracle 缓存目录
    oracle_cache_dir = output_dir / ".oracle_cache"
    
    # 为每个 agent 准备工作空间
    for agent_name in agents:
        print(f"\n\n{'#'*80}")
        print(f"# Agent: {agent_name}")
        print(f"{'#'*80}")
        
        agent_output = output_dir / agent_name
        success_count = 0
        fail_count = 0
        
        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}]", end="")
            success = prepare_single_case(
                case=case,
                agent_name=agent_name,
                output_dir=agent_output,
                oracle_cache_dir=oracle_cache_dir
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        # 统计信息
        print(f"\n{'─'*80}")
        print(f"📊 Summary for {agent_name}:")
        print(f"   ✅ Successfully prepared: {success_count} cases")
        if fail_count > 0:
            print(f"   ❌ Failed: {fail_count} cases")
        print(f"   📁 Output directory: {agent_output}")
        print(f"{'─'*80}")
    
    print("\n" + "="*80)
    print("✅ Workspace Preparation Complete!")
    print(f"📁 All workspaces saved to: {output_dir}")
    print("\n💡 Next steps:")
    for agent_name in agents:
        print(f"   - Use {agent_name} CLI to solve cases in results/{agent_name}/")
        print(f"     Example: openhands -t results/{agent_name}/{{case_id}}/prompt.md")
    print("="*80)


# =============================================================================
# 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare Code Agent Workspaces (oracle + prompt only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--agent', '-a',
        nargs='+',
        required=True,
        help="Agent name(s), e.g., openhands mini-swe-agent"
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
        help='Specific case IDs to prepare (default: all)'
    )
    
    parser.add_argument(
        '--equation-types',
        nargs='+',
        default=None,
        help='Equation type(s) to prepare, e.g., poisson heat (default: all)'
    )
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    root_dir = Path(__file__).parent.parent
    data_file = root_dir / args.data
    output_dir = root_dir / args.output
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    prepare_agent_workspaces(
        agents=args.agent,
        output_dir=output_dir,
        data_file=data_file,
        case_filter=args.cases,
        equation_types=args.equation_types
    )


if __name__ == '__main__':
    main()
