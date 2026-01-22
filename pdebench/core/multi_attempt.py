"""
多轮迭代模块 - 用于实验 2.1

允许 Agent 看到错误后进行多次尝试。
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from .feedback_prompt import create_feedback_prompt


def run_with_attempts(
    original_prompt: str,
    context: Dict[str, Any],
    target_error: float,
    target_time: float,
    oracle_info: Dict[str, Any],
    agent_call_fn: Callable,
    execute_fn: Callable,
    compute_error_fn: Callable,
    max_attempts: int = 3,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    运行多轮迭代
    
    Args:
        original_prompt: 原始任务 prompt
        context: 上下文信息
        target_error: 目标误差
        target_time: 目标时间
        oracle_info: Oracle 基准信息
        agent_call_fn: 调用 Agent 的函数 (prompt, context) -> response
        execute_fn: 执行代码的函数 (code, context) -> exec_result
        compute_error_fn: 计算误差的函数 (output) -> error
        max_attempts: 最大尝试次数
        output_dir: 输出目录（保存中间结果）
    
    Returns:
        包含所有尝试历史的结果字典
    """
    
    attempts_history = []
    final_result = None
    
    for attempt_num in range(1, max_attempts + 1):
        print(f"\n   🔄 Attempt {attempt_num}/{max_attempts}...")
        
        # 准备 prompt
        if attempt_num == 1:
            prompt_to_use = original_prompt
        else:
            # 构建反馈 prompt
            previous_attempt = attempts_history[-1]
            prompt_to_use = create_feedback_prompt(
                original_prompt=original_prompt,
                previous_attempt=previous_attempt,
                target_error=target_error,
                target_time=target_time,
                oracle_info=oracle_info,
                attempt_num=attempt_num
            )
            
            # 保存反馈 prompt
            if output_dir:
                feedback_prompt_path = output_dir / f"feedback_prompt_attempt_{attempt_num}.md"
                feedback_prompt_path.write_text(prompt_to_use)
        
        # 调用 Agent
        try:
            response = agent_call_fn(prompt_to_use, context)
            
            if not response.success:
                # Agent 调用失败
                attempt_record = {
                    'attempt_num': attempt_num,
                    'code': '',
                    'success': False,
                    'error': None,
                    'time': None,
                    'error_message': response.error,
                    'stderr': '',
                    'status': 'AGENT_ERROR',
                    'llm_usage': response.usage if hasattr(response, 'usage') else {}
                }
                attempts_history.append(attempt_record)
                print(f"   ❌ Agent call failed: {response.error}")
                
                # Agent 调用失败，不继续尝试
                break
            
            code = response.code
            
            # 保存代码
            if output_dir:
                code_path = output_dir / f"solver_attempt_{attempt_num}.py"
                code_path.write_text(code)
            
            # 执行代码
            exec_result = execute_fn(code, context)
            
            # 计算误差（如果执行成功）
            error = None
            if exec_result['success']:
                try:
                    error = compute_error_fn(exec_result['agent_output'])
                except Exception as e:
                    exec_result['success'] = False
                    exec_result['error_message'] = f"Error computation failed: {e}"
            
            # 判定状态
            if not exec_result['success']:
                status = 'EXECUTION_ERROR'
                fail_reason = exec_result.get('error_message', 'Unknown')
            elif error is None or error != error:  # NaN check
                status = 'EVALUATION_ERROR'
                fail_reason = 'Error computation failed'
            elif error > target_error:
                status = 'ACCURACY_FAIL'
                fail_reason = f"error={error:.2e} > target={target_error:.2e}"
            elif exec_result['time'] > target_time:
                status = 'TIME_FAIL'
                fail_reason = f"time={exec_result['time']:.3f}s > target={target_time:.3f}s"
            else:
                status = 'PASS'
                fail_reason = None
            
            # 记录本次尝试
            attempt_record = {
                'attempt_num': attempt_num,
                'code': code,
                'success': exec_result['success'],
                'error': error,
                'time': exec_result['time'],
                'error_message': exec_result.get('error_message'),
                'stderr': exec_result.get('stderr'),
                'status': status,
                'fail_reason': fail_reason,
                'llm_usage': response.usage if hasattr(response, 'usage') else {}
            }
            attempts_history.append(attempt_record)
            
            # 打印结果
            if status == 'PASS':
                print(f"   ✅ Passed on attempt {attempt_num}!")
                break
            else:
                print(f"   ❌ {status}: {fail_reason[:80]}")
                
                # 如果是最后一次尝试
                if attempt_num == max_attempts:
                    print(f"   ⚠️  Failed after {max_attempts} attempts")
        
        except Exception as e:
            # 捕获意外错误
            attempt_record = {
                'attempt_num': attempt_num,
                'code': '',
                'success': False,
                'error': None,
                'time': None,
                'error_message': f"Unexpected error: {str(e)}",
                'stderr': '',
                'status': 'UNEXPECTED_ERROR',
                'llm_usage': {}
            }
            attempts_history.append(attempt_record)
            print(f"   ❌ Unexpected error: {e}")
            break
    
    # 分析改进轨迹
    improvement_analysis = analyze_improvement(attempts_history)
    
    # 保存尝试历史
    if output_dir:
        history_path = output_dir / "attempts_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'attempts': attempts_history,
                'improvement_analysis': improvement_analysis
            }, f, indent=2)
    
    # 构建最终结果
    final_attempt = attempts_history[-1]
    final_result = {
        'final_status': final_attempt['status'],
        'final_error': final_attempt.get('error'),
        'final_time': final_attempt.get('time'),
        'num_attempts': len(attempts_history),
        'attempts_history': attempts_history,
        'improvement_analysis': improvement_analysis,
        'code': final_attempt.get('code', ''),
        'error_message': final_attempt.get('error_message'),
        'stderr': final_attempt.get('stderr'),
        'success': final_attempt.get('success', False),
    }
    
    return final_result


def analyze_improvement(attempts_history: list) -> Dict[str, Any]:
    """
    分析改进轨迹
    
    Args:
        attempts_history: 所有尝试的列表
    
    Returns:
        改进分析字典
    """
    analysis = {
        'total_attempts': len(attempts_history),
        'final_success': attempts_history[-1]['status'] == 'PASS',
        'improved': False,
        'error_trajectory': [],
        'time_trajectory': [],
        'status_trajectory': []
    }
    
    # 收集轨迹
    for attempt in attempts_history:
        analysis['status_trajectory'].append(attempt['status'])
        
        if attempt['success'] and attempt['error'] is not None:
            analysis['error_trajectory'].append(attempt['error'])
            analysis['time_trajectory'].append(attempt['time'])
    
    # 判断是否有改进
    if len(analysis['error_trajectory']) > 1:
        first_error = analysis['error_trajectory'][0]
        last_error = analysis['error_trajectory'][-1]
        
        # 误差下降 = 改进
        if last_error < first_error:
            analysis['improved'] = True
            analysis['error_reduction_pct'] = (first_error - last_error) / first_error * 100
        
        # 记录最佳误差
        analysis['best_error'] = min(analysis['error_trajectory'])
        analysis['best_error_attempt'] = analysis['error_trajectory'].index(analysis['best_error']) + 1
    
    # 统计各类失败
    failure_counts = {}
    for attempt in attempts_history:
        if attempt['status'] != 'PASS':
            failure_counts[attempt['status']] = failure_counts.get(attempt['status'], 0) + 1
    
    analysis['failure_distribution'] = failure_counts
    
    return analysis
