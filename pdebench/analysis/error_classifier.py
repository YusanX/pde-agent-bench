"""
错误分类器 - 自动分析失败原因

用于实验 4.5: Error Analysis
"""

import re
from typing import Dict, Any, Optional


class ErrorClassifier:
    """失败原因分类器"""
    
    # 错误类别定义
    SYNTAX_ERROR = 'syntax_error'              # Python/dolfinx语法错误
    API_ERROR = 'api_error'                    # dolfinx API使用错误
    MATH_ERROR = 'math_error'                  # PDE离散化错误
    PARAMETER_ERROR = 'parameter_error'        # 参数选择错误（网格太粗/细）
    STABILIZATION_MISSING = 'stabilization_missing'  # 缺少稳定化（SUPG等）
    TIMEOUT = 'timeout'                        # 算法效率低，超时
    IMPORT_ERROR = 'import_error'              # 导入错误
    CONVERGENCE_ERROR = 'convergence_error'    # 求解器不收敛
    OTHER = 'other'                            # 其他错误
    
    def classify(
        self,
        result: Dict[str, Any],
        case: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        分类错误
        
        Args:
            result: run_single_case 返回的结果字典
            case: 可选的 case 配置（用于高级分析）
        
        Returns:
            错误类别字符串
        """
        status = result.get('status')
        
        # 如果成功，没有错误
        if status == 'PASS':
            return None
        
        # LLM 错误
        if status == 'LLM_ERROR' or status == 'AGENT_ERROR':
            return 'llm_call_error'
        
        # 获取错误信息
        error_msg = result.get('error_message', '')
        stderr = result.get('stderr', '')
        full_error = f"{error_msg}\n{stderr}".lower()
        
        # 1. 语法错误
        if self._is_syntax_error(full_error):
            return self.SYNTAX_ERROR
        
        # 2. 导入错误
        if self._is_import_error(full_error):
            return self.IMPORT_ERROR
        
        # 3. API 错误
        if self._is_api_error(full_error):
            return self.API_ERROR
        
        # 4. 超时
        if self._is_timeout(full_error, status):
            return self.TIMEOUT
        
        # 5. 收敛错误
        if self._is_convergence_error(full_error):
            return self.CONVERGENCE_ERROR
        
        # 6. 数学/离散化错误
        if self._is_math_error(full_error):
            return self.MATH_ERROR
        
        # 7. 参数选择错误（需要代码分析）
        code = result.get('code', '')
        if code and self._is_parameter_error(full_error, code):
            return self.PARAMETER_ERROR
        
        # 8. 稳定化缺失（需要 case 信息）
        if case and code and self._is_stabilization_missing(full_error, code, case):
            return self.STABILIZATION_MISSING
        
        # 默认：其他
        return self.OTHER
    
    def _is_syntax_error(self, error_text: str) -> bool:
        """检测语法错误"""
        syntax_patterns = [
            'syntaxerror',
            'indentationerror',
            'invalid syntax',
            'unexpected indent',
            'unindent does not match',
        ]
        return any(pattern in error_text for pattern in syntax_patterns)
    
    def _is_import_error(self, error_text: str) -> bool:
        """检测导入错误"""
        import_patterns = [
            'importerror',
            'modulenotfounderror',
            'no module named',
            'cannot import name',
        ]
        return any(pattern in error_text for pattern in import_patterns)
    
    def _is_api_error(self, error_text: str) -> bool:
        """检测 API 使用错误"""
        api_patterns = [
            'attributeerror',
            'has no attribute',
            'typeerror',
            'takes.*positional argument',
            'missing.*required.*argument',
            'got an unexpected keyword argument',
        ]
        return any(re.search(pattern, error_text) for pattern in api_patterns)
    
    def _is_timeout(self, error_text: str, status: str) -> bool:
        """检测超时"""
        return 'timeout' in error_text or 'timed out' in error_text or status == 'TIMEOUT'
    
    def _is_convergence_error(self, error_text: str) -> bool:
        """检测求解器不收敛"""
        convergence_patterns = [
            'diverged',
            'did not converge',
            'convergence failed',
            'ksp_diverged',
            'snes_diverged',
            'linear solver failed',
        ]
        return any(pattern in error_text for pattern in convergence_patterns)
    
    def _is_math_error(self, error_text: str) -> bool:
        """检测数学/数值错误"""
        math_patterns = [
            'singular',
            'nan',
            'inf',
            'divide by zero',
            'division by zero',
            'invalid value encountered',
            'linalgerror',
        ]
        return any(pattern in error_text for pattern in math_patterns)
    
    def _is_parameter_error(self, error_text: str, code: str) -> bool:
        """检测参数选择错误（启发式）"""
        # 检查是否因为网格太粗导致精度不够
        if 'accuracy' in error_text or 'error too large' in error_text:
            # 检查代码中的网格分辨率
            mesh_match = re.search(r'create.*mesh.*\(.*?,\s*(\d+)', code, re.IGNORECASE)
            if mesh_match:
                resolution = int(mesh_match.group(1))
                if resolution < 20:  # 网格太粗
                    return True
        
        return False
    
    def _is_stabilization_missing(
        self,
        error_text: str,
        code: str,
        case: Dict[str, Any]
    ) -> bool:
        """检测稳定化缺失（高级分析）"""
        # 检查是否是对流占优问题
        pde_type = case.get('pde_classification', {}).get('equation_type', '')
        
        if 'advection' in pde_type or 'convection' in pde_type:
            # 检查代码中是否有稳定化项
            stabilization_keywords = ['supg', 'streamline', 'stabiliz']
            has_stabilization = any(kw in code.lower() for kw in stabilization_keywords)
            
            # 如果没有稳定化且出现震荡
            if not has_stabilization and ('oscillat' in error_text or 'unstable' in error_text):
                return True
        
        return False
    
    def get_error_description(self, error_category: str) -> str:
        """获取错误类别的描述"""
        descriptions = {
            self.SYNTAX_ERROR: "Python/DOLFINx syntax error",
            self.API_ERROR: "DOLFINx API usage error",
            self.MATH_ERROR: "PDE discretization or numerical error",
            self.PARAMETER_ERROR: "Parameter selection error (mesh too coarse/fine)",
            self.STABILIZATION_MISSING: "Missing stabilization (e.g., SUPG)",
            self.TIMEOUT: "Algorithm too slow (timeout)",
            self.IMPORT_ERROR: "Import or dependency error",
            self.CONVERGENCE_ERROR: "Solver did not converge",
            self.OTHER: "Other unclassified error",
        }
        return descriptions.get(error_category, "Unknown error category")
    
    def analyze_errors_batch(self, results: list) -> Dict[str, Any]:
        """批量分析错误"""
        error_distribution = {}
        total_failures = 0
        
        for result in results:
            if result.get('status') != 'PASS':
                total_failures += 1
                category = self.classify(result)
                error_distribution[category] = error_distribution.get(category, 0) + 1
        
        # 计算百分比
        error_percentages = {
            cat: (count / total_failures * 100) if total_failures > 0 else 0
            for cat, count in error_distribution.items()
        }
        
        return {
            'total_failures': total_failures,
            'error_distribution': error_distribution,
            'error_percentages': error_percentages,
            'most_common': max(error_distribution.items(), key=lambda x: x[1])[0] if error_distribution else None
        }
