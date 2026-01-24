"""
CodePDE Wrapper

CodePDE 是专门为 PDE 求解器生成设计的框架。

论文: CodePDE: An Inference Framework for LLM-driven PDE Solver Generation
      https://arxiv.org/abs/2505.08783
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from .base_agent import BaseAgent, AgentResponse


class CodePDEWrapper(BaseAgent):
    """
    CodePDE Wrapper
    
    使用 CodePDE 的 repeated_sample 模式生成 PDE 求解器。
    """
    
    def _setup(self):
        """初始化 CodePDE"""
        # CodePDE 路径
        self.codepde_path = Path(self.config.get(
            'codepde_path',
            '/Users/yusan/agent/CodePDE'
        ))
        
        if not self.codepde_path.exists():
            raise FileNotFoundError(
                f"CodePDE not found at {self.codepde_path}. "
                f"Please set 'codepde_path' in config."
            )
        
        # 添加 CodePDE 到 Python 路径
        sys.path.insert(0, str(self.codepde_path))
        
        # 导入 CodePDE 模块
        try:
            from code_generation import generate_initial_prompt_without_seed
            from llm_api import generate_response
            
            self.generate_prompt_fn = generate_initial_prompt_without_seed
            self.generate_response_fn = generate_response
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import CodePDE modules: {e}. "
                f"Make sure CodePDE is properly installed."
            )
        
        # LLM 配置
        self.model_name = self.config.get('model', 'gpt-4o')
        self.model_family = self._get_model_family(self.model_name)
        self.api_key = self.config.get('api_key', None)
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 4096)
        
        # CodePDE repeated_sample 配置（适配 pdebench）
        self.num_repeated_samples = max(1, int(self.config.get('num_repeated_samples', 3)))
        self.num_debugging_trials = max(1, int(self.config.get('num_debugging_trials_per_sample', 2)))
        self.sample_delay_sec = float(self.config.get('sample_delay_sec', 1.0))
        self.evaluate_candidates = bool(self.config.get('evaluate_candidates', False))
        self.eval_timeout = int(self.config.get('eval_timeout', self.config.get('timeout', 300)))
    
    def _get_model_family(self, model_name: str) -> str:
        """从模型名推断模型家族"""
        if 'gpt' in model_name or model_name.startswith('o'):
            return 'gpt'
        elif 'claude' in model_name:
            return 'claude'
        elif 'gemini' in model_name:
            return 'gemini'
        elif 'deepseek' in model_name:
            return 'deepseek'
        elif 'qwen' in model_name:
            return 'qwen'
        else:
            return 'gpt'  # 默认
    
    def generate_solution(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> AgentResponse:
        """
        使用 CodePDE 生成代码
        
        注意：CodePDE 有自己的 prompt 生成机制，但我们会尝试
              在用户 prompt 的基础上引导它生成合适的代码。
        """
        start_time = time.time()
        
        try:
            # 创建简化的配置对象（模拟 hydra config）
            cfg = self._create_config(prompt, context)
            
            base_messages = self._prepare_messages(prompt, context)
            case_spec = context.get('case_spec')
            oracle_info = context.get('oracle_info')
            
            best_candidate = None
            best_error = float('inf')
            best_score = float('-inf')
            last_code = ''
            last_response = None
            total_usage = {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0}
            
            for sample_idx in range(self.num_repeated_samples):
                messages = [dict(m) for m in base_messages]
                if not (self.evaluate_candidates and case_spec and oracle_info):
                    response = self.generate_response_fn(messages, cfg)
                    last_response = response
                    code = self._extract_code(response)
                    last_code = code
                    self._accumulate_usage(total_usage, response)
                    
                    candidate = {
                        'code': code,
                        'response': response,
                        'success': True,
                        'error': None,
                        'time': None
                    }
                    candidate_score = len(code.strip())
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_candidate = candidate
                else:
                    for trial_idx in range(self.num_debugging_trials):
                        response = self.generate_response_fn(messages, cfg)
                        last_response = response
                        code = self._extract_code(response)
                        last_code = code
                        self._accumulate_usage(total_usage, response)
                        
                        candidate = {
                            'code': code,
                            'response': response,
                            'success': True,
                            'error': None,
                            'time': None
                        }
                        
                        eval_result = self._evaluate_candidate(
                            code=code,
                            case_spec=case_spec,
                            oracle_info=oracle_info,
                            timeout=self.eval_timeout
                        )
                        candidate.update(eval_result)
                        
                        if candidate['success']:
                            if candidate['error'] is not None and candidate['error'] < best_error:
                                best_error = candidate['error']
                                best_candidate = candidate
                            break
                        
                        if trial_idx < self.num_debugging_trials - 1:
                            debug_msg = self._build_debug_message(
                                code=code,
                                eval_result=eval_result
                            )
                            messages = self._append_debug_messages(messages, code, debug_msg)
                
                if self.sample_delay_sec > 0 and sample_idx < self.num_repeated_samples - 1:
                    time.sleep(self.sample_delay_sec)
            
            if best_candidate is None and last_response is not None:
                best_candidate = {
                    'code': last_code,
                    'response': last_response,
                    'success': True,
                    'error': None,
                    'time': None
                }
            
            if best_candidate is None:
                raise RuntimeError("CodePDE generation failed: no valid candidate produced")
            
            latency = time.time() - start_time
            
            # 构建 usage 信息
            usage = {
                'latency_sec': latency,
                'total_tokens': total_usage['total_tokens'],
                'input_tokens': total_usage['input_tokens'],
                'output_tokens': total_usage['output_tokens'],
                'cost_usd': 0.0,  # TODO: 根据 model 计算成本
            }
            
            return AgentResponse(
                success=True,
                code=best_candidate['code'],
                raw_response=str(best_candidate['response']),
                agent_name=self.agent_name,
                usage=usage
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return AgentResponse(
                success=False,
                code='',
                raw_response='',
                agent_name=self.agent_name,
                error=str(e),
                usage={'latency_sec': latency, 'total_tokens': 0, 'cost_usd': 0.0}
            )
    
    def _create_config(self, prompt: str, context: Dict[str, Any]):
        """创建简化的配置对象"""
        class SimpleConfig:
            class Model:
                def __init__(self, name, family_name, api_key, base_url=None):
                    self.name = name
                    self.family_name = family_name
                    self.api_key = api_key
                    self.base_url = base_url
            
            def __init__(self, model_name, family_name, api_key, base_url=None):
                self.model = self.Model(model_name, family_name, api_key, base_url)
        
        # 获取 API key（从环境变量或配置）
        import os
        api_key = self.api_key
        if not api_key:
            if self.model_family == 'gpt':
                api_key = os.environ.get('OPENAI_API_KEY')
            elif self.model_family == 'claude':
                api_key = os.environ.get('ANTHROPIC_API_KEY')
            elif self.model_family == 'gemini':
                api_key = os.environ.get('GOOGLE_API_KEY')
            elif self.model_family == 'qwen':
                api_key = os.environ.get('DASHSCOPE_API_KEY')
        
        base_url = self.config.get('base_url', None)
        
        return SimpleConfig(
            self.model_name,
            self.model_family,
            api_key,
            base_url
        )
    
    def _prepare_messages(self, prompt: str, context: Dict[str, Any]) -> list:
        """
        准备发送给 LLM 的消息
        
        我们直接使用 pdebench 的 prompt，并添加必要的格式说明
        """
        system_prompt = """You are an expert in numerical PDEs and scientific computing, particularly with FEniCSx/DOLFINx.

Generate COMPLETE, RUNNABLE Python code that:
1. Imports all necessary libraries
2. Defines a function: def solve(case_spec: dict) -> str
3. The solve() function should:
   - Parse case_spec to get problem parameters
   - Build mesh and function spaces using DOLFINx
   - Set up and solve the PDE
   - Save solution to 'solution.npz' with field 'u'
   - Return the output filename
4. Include error handling
5. Add helpful comments

Output ONLY the Python code, no explanations before or after."""
        
        user_message = f"""{prompt}

Please generate the complete Python code following this structure:

```python
import numpy as np
from dolfinx import mesh, fem
import ufl
# ... other imports ...

def solve(case_spec: dict) -> str:
    '''Main solver function'''
    # Your implementation here
    output_path = "solution.npz"
    # ... save results ...
    return output_path

if __name__ == '__main__':
    import sys
    import json
    case_spec = json.loads(sys.argv[1])
    output_file = solve(case_spec)
    print(f"Solution saved to: {{output_file}}")
```"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def _extract_code(self, response) -> str:
        """从 LLM 响应中提取代码"""
        import re
        
        # 获取响应内容
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        else:
            content = str(response)
        
        # 提取代码块
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
        
        if code_blocks:
            # 返回最长的代码块
            return max(code_blocks, key=len).strip()
        
        # 如果没有代码块标记，假设整个响应就是代码
        return content.strip()
    
    def _append_debug_messages(
        self,
        messages: List[Dict[str, str]],
        code: str,
        debug_message: str
    ) -> List[Dict[str, str]]:
        updated = list(messages)
        updated.append({"role": "assistant", "content": code})
        updated.append({"role": "user", "content": debug_message})
        return updated
    
    def _build_debug_message(self, code: str, eval_result: Dict[str, Any]) -> str:
        error_message = eval_result.get('error_message') or eval_result.get('stderr') or ''
        stdout = eval_result.get('stdout') or ''
        return (
            "The previous code failed to run or produced invalid results.\n"
            "Please fix the issues and return a FULL, runnable Python script only.\n"
            "Make sure solve(case_spec) returns a dict with keys: u (or u_grid) and solver_info.\n\n"
            f"Error message:\n{error_message}\n\n"
            f"Stdout:\n{stdout}\n\n"
            f"Previous code:\n{code}\n"
        )
    
    def _accumulate_usage(self, total_usage: Dict[str, int], response) -> None:
        usage = getattr(response, 'usage', None)
        if usage is None:
            return
        total_usage['total_tokens'] += self._get_usage_value(usage, 'total_tokens')
        total_usage['input_tokens'] += self._get_usage_value(usage, 'prompt_tokens')
        total_usage['output_tokens'] += self._get_usage_value(usage, 'completion_tokens')
    
    def _get_usage_value(self, usage, key: str, default: int = 0) -> int:
        if isinstance(usage, dict):
            return int(usage.get(key, default) or 0)
        return int(getattr(usage, key, default) or 0)
    
    def _evaluate_candidate(
        self,
        code: str,
        case_spec: Dict[str, Any],
        oracle_info: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        from pdebench.sandbox.executor import execute_agent_function
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            solver_path = tmp_path / "solver.py"
            solver_path.write_text(code)
            agent_output = tmp_path / "agent_output"
            agent_output.mkdir(parents=True, exist_ok=True)
            
            result = execute_agent_function(
                script_path=solver_path,
                outdir=agent_output,
                case_spec=case_spec,
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
            
            error = self._compute_error(agent_output, oracle_info)
            if np.isnan(error):
                return {
                    'success': False,
                    'error': error,
                    'time': result.t_agent_run,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'error_message': 'Error computation returned NaN'
                }
            
            return {
                'success': True,
                'error': error,
                'time': result.t_agent_run,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'error_message': None
            }
    
    def _compute_error(self, agent_output: Path, oracle_info: Dict[str, Any]) -> float:
        try:
            agent_sol = np.load(agent_output / "solution.npz")
            u_agent = agent_sol['u']
            
            if oracle_info.get('reference') is None:
                return float('nan')
            
            u_ref = np.array(oracle_info['reference'])
            
            if u_agent.shape != u_ref.shape:
                try:
                    from scipy.ndimage import zoom
                except Exception:
                    return float('nan')
                factors = np.array(u_ref.shape) / np.array(u_agent.shape)
                u_agent = zoom(u_agent, factors, order=1)
            
            diff = u_agent - u_ref
            ref_norm = np.sqrt(np.sum(u_ref**2))
            
            if ref_norm < 1e-15:
                return float(np.sqrt(np.sum(diff**2)))
            
            rel_L2 = np.sqrt(np.sum(diff**2)) / ref_norm
            return float(rel_L2)
            
        except Exception:
            return float('nan')
    
    def cleanup(self):
        """清理资源"""
        # 从 sys.path 中移除 CodePDE
        if str(self.codepde_path) in sys.path:
            sys.path.remove(str(self.codepde_path))
