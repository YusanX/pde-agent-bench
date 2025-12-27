"""
测试脚本模板生成器
"""

from typing import Dict, Any



MATH_TYPE_TO_TEST_CLASS = {
    'elliptic': 'EllipticCaseTest',
    'parabolic': 'ParabolicCaseTest',
    'hyperbolic': 'HyperbolicCaseTest',
    'mixed_type': 'MixedTypeCaseTest',
    'incompressible_flow': 'IncompressibleFlowCaseTest',
}


def generate_test_script(config: Dict[str, Any], mode: str) -> str:
    """
    生成测试脚本
    
    Args:
        config: Case 配置
        mode: 'fix_accuracy' 或 'fix_time'
    """
    
    case_id = config['id']
    math_types = config['pde_classification']['math_type']
    primary_type = math_types[0] if math_types else 'elliptic'
    
    test_class = MATH_TYPE_TO_TEST_CLASS.get(primary_type, 'EllipticCaseTest')
    
    mode_title = 'Fix Accuracy, Optimize Speed' if mode == 'fix_accuracy' else 'Fix Time Budget, Optimize Accuracy'
    
    script = f'''#!/usr/bin/env python3
"""
测试脚本 - Case: {case_id} ({mode_title})

目标：{"保证精度要求，优化速度" if mode == "fix_accuracy" else "在时间预算内，优化精度"}

用法：
    python test_{mode}.py --agent-script /path/to/solver.py
"""

import argparse
import sys
from pathlib import Path

# 添加pdebench到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdebench.testing.test_base import {test_class}


def main():
    parser = argparse.ArgumentParser(
        description='Test case: {case_id} ({mode_title})',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--agent-script',
        type=Path,
        required=True,
        help='Path to agent solver script'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    # 验证agent脚本存在
    if not args.agent_script.exists():
        print(f"❌ Error: Agent script not found: {{args.agent_script}}")
        sys.exit(1)
    
    # 提取agent目录
    agent_dir = args.agent_script.parent.parent
    
    # 创建测试实例
    case_dir = Path(__file__).parent
    tester = {test_class}(case_dir, agent_dir=agent_dir)
    
    # 运行测试
    result = tester.run_test(
        agent_script=args.agent_script,
        test_mode='{mode}',
        timeout_sec=args.timeout
    )
    
    # 返回状态码
    if result['status'] == 'PASSED':
        print("\\n✅ TEST PASSED")
        sys.exit(0)
    else:
        print("\\n❌ TEST FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
'''
    
    return script

