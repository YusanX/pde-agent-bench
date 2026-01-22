"""
反馈 Prompt 生成器 - 用于多轮迭代

用于实验 2.1: Multi-Attempt Test
"""

from typing import Dict, Any, Optional


def create_feedback_prompt(
    original_prompt: str,
    previous_attempt: Dict[str, Any],
    target_error: float,
    target_time: float,
    oracle_info: Dict[str, Any],
    attempt_num: int
) -> str:
    """
    创建包含错误反馈的 prompt，用于多轮迭代
    
    Args:
        original_prompt: 原始任务 prompt
        previous_attempt: 上一次尝试的结果
        target_error: 目标误差阈值
        target_time: 目标时间阈值
        oracle_info: Oracle 基准信息
        attempt_num: 当前尝试次数
    
    Returns:
        包含反馈的新 prompt
    """
    
    feedback_header = f"""
{'='*70}
📝 ATTEMPT {attempt_num} - FEEDBACK FROM PREVIOUS ATTEMPT
{'='*70}

Your previous code had issues. Here's what went wrong and how to improve:

"""
    
    # 1. 显示之前的代码
    code_section = f"""
## Previous Code (Attempt {attempt_num - 1}):

```python
{previous_attempt['code'][:2000]}  # 限制长度
{'...(code truncated)' if len(previous_attempt['code']) > 2000 else ''}
```

"""
    
    # 2. 根据失败类型提供反馈
    feedback_details = ""
    
    # 执行失败
    if not previous_attempt['success']:
        error_msg = previous_attempt.get('error_message', 'Unknown error')
        stderr = previous_attempt.get('stderr', '')
        
        feedback_details = f"""
## ❌ Execution Failed

### Error Message:
```
{error_msg[:800]}
```

### Debug Output (stderr):
```
{stderr[:800]}
```

### What to Fix:
- **Carefully read the error message above**
- Common issues:
  1. **Syntax errors**: Check indentation, brackets, quotes
  2. **Import errors**: Ensure all DOLFINx modules are correctly imported
  3. **API errors**: Verify function signatures and parameter types
  4. **Undefined variables**: Make sure all variables are defined before use

### Suggested Actions:
1. Fix the specific error shown above
2. Test your logic step-by-step
3. Ensure the `solve(case_spec)` function signature is correct
4. Verify all imports are at the top of the file

"""
    
    # 精度不达标
    elif previous_attempt['error'] is not None and previous_attempt['error'] > target_error:
        feedback_details = f"""
## ⚠️ Accuracy Not Met

### Results:
- ✅ Code executed successfully
- ❌ **Accuracy insufficient**

### Performance:
- **Your error**: {previous_attempt['error']:.2e}
- **Target error**: {target_error:.2e} (must be ≤ this)
- **Oracle error**: {oracle_info['error']:.2e}

Your solution is **{previous_attempt['error'] / target_error:.1f}x** worse than acceptable.

### What to Fix:
The numerical accuracy is too low. Consider:

1. **Refine the mesh**:
   - Increase mesh resolution (e.g., from 32 to 64)
   - Use finer grid near boundaries or regions of interest

2. **Increase element degree**:
   - Use higher-order elements (P2 or P3 instead of P1)
   - This improves accuracy without too much cost

3. **Improve solver convergence**:
   - Tighten solver tolerances (rtol, atol)
   - Increase maximum iterations if solver didn't fully converge

4. **Check discretization**:
   - Verify the weak formulation is correct
   - Ensure boundary conditions are properly enforced

### Suggested Actions:
- **Priority 1**: Try doubling the mesh resolution
- **Priority 2**: Check if solver converged (look at residuals)
- **Priority 3**: Consider using higher-order elements

"""
    
    # 时间超限
    elif previous_attempt['time'] > target_time:
        feedback_details = f"""
## ⏱️ Timeout - Code Too Slow

### Results:
- ✅ Code executed successfully
- ✅ Accuracy met
- ❌ **Execution too slow**

### Performance:
- **Your time**: {previous_attempt['time']:.2f}s
- **Target time**: {target_time:.2f}s (must be ≤ this)
- **Oracle time**: {oracle_info['time']:.2f}s

Your solution is **{previous_attempt['time'] / target_time:.1f}x** slower than acceptable.

### What to Fix:
The solver is too slow. Consider:

1. **Optimize linear solver**:
   - Use iterative solvers (CG, GMRES) instead of direct solvers
   - Add appropriate preconditioners
   - Set reasonable convergence tolerances

2. **Reduce unnecessary computations**:
   - Avoid redundant operations in loops
   - Reuse computed quantities when possible
   - Minimize expensive function calls

3. **Algorithm efficiency**:
   - Check if you're using efficient data structures
   - Avoid creating large temporary arrays unnecessarily
   - Use NumPy vectorized operations where possible

4. **Solver parameters**:
   - Don't make solver tolerances unnecessarily tight
   - Set reasonable maximum iterations

### Suggested Actions:
- **Priority 1**: Switch to iterative solver with preconditioner
- **Priority 2**: Review your code for inefficient loops
- **Priority 3**: Check solver convergence parameters

"""
    
    # 如果通过了（这种情况不应该出现，但以防万一）
    else:
        feedback_details = """
## ✅ Previous attempt passed

(This shouldn't happen - if code passed, we wouldn't retry)

"""
    
    # 3. 重申任务
    task_reminder = f"""
{'='*70}
🎯 YOUR TASK (Attempt {attempt_num})
{'='*70}

{original_prompt}

## Requirements Reminder:
- Generate **complete, runnable Python code**
- Function signature: `def solve(case_spec: dict) -> str`
- Save results to 'solution.npz' with field 'u'
- Target accuracy: ≤ {target_error:.2e}
- Target time: ≤ {target_time:.2f}s

## Output Format:
```python
import numpy as np
from dolfinx import mesh, fem
import ufl
# ... your imports ...

def solve(case_spec: dict) -> str:
    '''Main solver function'''
    # Your improved implementation here
    output_path = "solution.npz"
    # Save results
    return output_path

if __name__ == '__main__':
    import sys, json
    case_spec = json.loads(sys.argv[1])
    output_file = solve(case_spec)
    print(f"Solution saved to: {{output_file}}")
```

**Please provide the COMPLETE improved code now.**
"""
    
    # 组合所有部分
    full_prompt = feedback_header + code_section + feedback_details + task_reminder
    
    return full_prompt
