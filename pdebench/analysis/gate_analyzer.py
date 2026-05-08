"""
Gate-level Analysis for PDEBench
实验 4.1: Case-level 通过率分析（Accuracy-first → Time Gate）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass
class GateBreakdown:
    """单个 case 的门槛通过情况"""
    case_id: str
    exec_valid: bool = False      # Gate 1: 执行有效性
    accuracy_pass: bool = False   # Gate 2: 精度达标
    time_pass: bool = False       # Gate 3: 时间达标
    final_pass: bool = False      # 最终通过
    
    # 失败原因分类
    failure_stage: str = None     # 'exec', 'accuracy', 'time', None (if pass)
    failure_reason: str = None


class GateAnalyzer:
    """门槛分析器"""
    
    def analyze_single_case(
        self, 
        case_id: str,
        exec_result: Dict[str, Any],
        eval_result: Dict[str, Any],
        oracle_info: Dict[str, Any]
    ) -> GateBreakdown:
        """
        分析单个 case 通过了哪些门槛
        
        Args:
            case_id: case ID
            exec_result: 执行结果（包含 success, error, time, stderr 等）
            eval_result: 评测结果（包含 status, fail_reason 等）
            oracle_info: Oracle 信息（包含 error, time）
        
        Returns:
            GateBreakdown: 门槛分析结果
        """
        breakdown = GateBreakdown(case_id=case_id)
        
        # Gate 1: 执行有效性检查
        if exec_result.get('success', False):
            breakdown.exec_valid = True
            
            # Gate 2: 精度检查
            agent_error = exec_result.get('error')
            target_error = eval_result.get('target_error')
            
            if agent_error is not None and target_error is not None:
                if not np.isnan(agent_error) and agent_error <= target_error:
                    breakdown.accuracy_pass = True
                    
                    # Gate 3: 时间检查
                    agent_time = exec_result.get('time')
                    target_time = eval_result.get('target_time')
                    
                    if agent_time is not None and target_time is not None:
                        if agent_time <= target_time:
                            breakdown.time_pass = True
                            breakdown.final_pass = True
                        else:
                            breakdown.failure_stage = 'time'
                            breakdown.failure_reason = eval_result.get('fail_reason', 'TIME_FAIL')
                    else:
                        # 时间信息缺失
                        breakdown.failure_stage = 'time'
                        breakdown.failure_reason = 'MISSING_TIME_INFO'
                else:
                    # 精度未达标
                    breakdown.failure_stage = 'accuracy'
                    breakdown.failure_reason = eval_result.get('fail_reason', 'ACCURACY_FAIL')
            else:
                # 误差信息缺失或无效
                breakdown.failure_stage = 'accuracy'
                breakdown.failure_reason = 'MISSING_ERROR_INFO'
        else:
            # 执行失败
            breakdown.failure_stage = 'exec'
            breakdown.failure_reason = exec_result.get('error_message', 'EXECUTION_FAILED')
        
        return breakdown
    
    def compute_aggregate_statistics(
        self, 
        breakdowns: List[GateBreakdown]
    ) -> Dict[str, Any]:
        """
        计算聚合统计信息
        
        Args:
            breakdowns: 所有 case 的门槛分析结果
        
        Returns:
            统计信息字典
        """
        total = len(breakdowns)
        if total == 0:
            return {
                'total_cases': 0,
                'exec_valid_count': 0,
                'accuracy_pass_count': 0,
                'time_pass_count': 0,
                'final_pass_count': 0,
                'exec_valid_rate': 0.0,
                'accuracy_pass_rate': 0.0,
                'time_pass_rate': 0.0,
                'final_pass_rate': 0.0,
                'failure_breakdown': {}
            }
        
        # 计数
        exec_valid_count = sum(1 for b in breakdowns if b.exec_valid)
        accuracy_pass_count = sum(1 for b in breakdowns if b.accuracy_pass)
        time_pass_count = sum(1 for b in breakdowns if b.time_pass)
        final_pass_count = sum(1 for b in breakdowns if b.final_pass)
        
        # 失败阶段统计
        failure_stages = {}
        for b in breakdowns:
            if b.failure_stage:
                failure_stages[b.failure_stage] = failure_stages.get(b.failure_stage, 0) + 1
        
        stats = {
            'total_cases': total,
            'exec_valid_count': exec_valid_count,
            'accuracy_pass_count': accuracy_pass_count,
            'time_pass_count': time_pass_count,
            'final_pass_count': final_pass_count,
            
            # 通过率（相对总数）
            'exec_valid_rate': exec_valid_count / total,
            'accuracy_pass_rate': accuracy_pass_count / total,
            'time_pass_rate': time_pass_count / total,
            'final_pass_rate': final_pass_count / total,
            
            # 条件通过率（相对前一门槛）
            'accuracy_pass_rate_given_exec': (
                accuracy_pass_count / exec_valid_count if exec_valid_count > 0 else 0.0
            ),
            'time_pass_rate_given_accuracy': (
                time_pass_count / accuracy_pass_count if accuracy_pass_count > 0 else 0.0
            ),
            
            # 失败分布
            'failure_breakdown': failure_stages,
            'failure_breakdown_pct': {
                stage: count / total for stage, count in failure_stages.items()
            }
        }
        
        return stats
    
    def analyze_by_pde_type(
        self, 
        breakdowns: List[GateBreakdown],
        cases: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        按 PDE 类型分组分析
        
        Args:
            breakdowns: 所有 case 的门槛分析结果
            cases: 原始 case 配置（用于获取 PDE 类型）
        
        Returns:
            按 PDE 类型分组的统计
        """
        # 创建 case_id -> pde_type 映射
        case_to_pde = {}
        for case in cases:
            pde_type = case.get('oracle_config', {}).get('pde', {}).get('type', 'unknown')
            case_to_pde[case['id']] = pde_type
        
        # 按 PDE 类型分组
        pde_groups = {}
        for breakdown in breakdowns:
            pde_type = case_to_pde.get(breakdown.case_id, 'unknown')
            if pde_type not in pde_groups:
                pde_groups[pde_type] = []
            pde_groups[pde_type].append(breakdown)
        
        # 计算每个 PDE 类型的统计
        pde_stats = {}
        for pde_type, group_breakdowns in pde_groups.items():
            pde_stats[pde_type] = self.compute_aggregate_statistics(group_breakdowns)
        
        return pde_stats
