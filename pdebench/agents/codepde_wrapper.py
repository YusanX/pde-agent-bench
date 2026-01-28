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
        # 兼容不同 OpenAI 模型：gpt-5.* 需要 max_completion_tokens
        self.max_tokens = self.config.get('max_tokens', 4096)
        self.max_completion_tokens = self.config.get('max_completion_tokens', None)
        
        # CodePDE repeated_sample 配置（适配 pdebench）
        self.num_repeated_samples = max(1, int(self.config.get('num_repeated_samples', 3)))
        self.num_debugging_trials = max(1, int(self.config.get('num_debugging_trials_per_sample', 1)))
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
                def __init__(
                    self,
                    name,
                    family_name,
                    api_key,
                    base_url=None,
                    temperature=None,
                    max_tokens=None,
                    max_completion_tokens=None,
                ):
                    self.name = name
                    self.family_name = family_name
                    self.api_key = api_key
                    self.base_url = base_url
                    self.temperature = temperature
                    self.max_tokens = max_tokens
                    self.max_completion_tokens = max_completion_tokens
            
            def __init__(
                self,
                model_name,
                family_name,
                api_key,
                base_url=None,
                temperature=None,
                max_tokens=None,
                max_completion_tokens=None,
            ):
                self.model = self.Model(
                    model_name,
                    family_name,
                    api_key,
                    base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                )
        
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
            base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens or self.max_tokens,
        )
    
    def _prepare_messages(self, prompt: str, context: Dict[str, Any]) -> list:
        """
        准备发送给 LLM 的消息
        
        策略1: 使用 DOLFINx templates 作为 few-shot examples
        """
        # 1. 提取 PDE 类型
        case_spec = context.get('case_spec', {})
        pde_type = case_spec.get('oracle_config', {}).get('pde', {}).get('type', 'poisson')
        
        # 检查是否有 time 字段 (用于区分稳态/瞬态)
        has_time = 'time' in case_spec.get('oracle_config', {}).get('pde', {})
        
        # 2. 选择合适的模板
        template_name, template_code = self._load_dolfinx_template(pde_type, has_time)
        
        # 3. 构造 system prompt (强调使用 DOLFINx)
        system_prompt = """You are an expert in numerical PDEs and DOLFINx (FEniCSx).

Generate COMPLETE, RUNNABLE Python code using DOLFINx that:
1. Imports: numpy, dolfinx (mesh, fem, default_scalar_type), ufl, mpi4py
2. Defines: def solve(case_spec: dict) -> dict
3. Returns: {"u": ndarray, "solver_info": dict, ...}
4. Uses ONLY DOLFINx API (NO PyTorch, NO JAX)
5. Follows the reference template structure shown below

Key points:
- Parse case_spec to extract all PDE parameters
- Create mesh using dolfinx.mesh.create_unit_square()
- Define variational forms with ufl
- Use PETSc linear solvers via fem.petsc.LinearProblem
- Extract solution on uniform grid for evaluation

Output ONLY Python code, no markdown or explanations."""
        
        # 4. 构造 user message (PDEBench prompt + template)
        if template_code:
            user_message = f"""{prompt}

---

## Reference DOLFINx Implementation

Here is a complete, working DOLFINx implementation for {template_name}:

```python
{template_code}
```

**Instructions:**
1. Study the reference implementation above
2. Adapt it to solve the specific problem described in the task
3. Keep the same structure: parse case_spec → create mesh → define variational form → solve → extract grid
4. Return dict with "u" (solution array) and "solver_info" (metadata)
5. Handle boundary conditions and parameters based on case_spec

Generate the complete Python code now."""
        else:
            # 后备：无模板可用
            user_message = f"""{prompt}

Generate complete DOLFINx code following this structure:

```python
import numpy as np
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl

def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec["oracle_config"]["pde"]
    
    # 2. Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # 4. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # ... define a and L ...
    
    # 5. Boundary conditions
    # ...
    
    # 6. Solve
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={{...}})
    uh = problem.solve()
    
    # 7. Extract on grid
    # ...
    
    return {{"u": u_grid, "solver_info": {{...}}}}
```"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def _load_dolfinx_template(self, pde_type: str, has_time: bool = False) -> tuple:
        """
        加载对应的 DOLFINx 模板
        
        Returns:
            (template_name, template_code) 或 (None, None)
        """
        # PDE 类型映射
        template_map = {
            'poisson': 'poisson_template.py',
            'heat': 'heat_template.py',
            'convection_diffusion': 'convection_diffusion_template.py',
            'convection_diffusion_transient': 'heat_template.py',  # 使用 heat 模板作为时间步进参考
            'darcy': 'poisson_template.py',  # Darcy 类似 Poisson
            'reaction_diffusion': 'heat_template.py' if has_time else 'poisson_template.py',
            'helmholtz': 'poisson_template.py',
        }
        
        # 选择模板
        if has_time and pde_type == 'convection_diffusion':
            template_file = 'heat_template.py'  # 时间步进参考
        elif pde_type in template_map:
            template_file = template_map[pde_type]
        else:
            template_file = 'generic_template.py'
        
        # 加载模板代码
        template_path = self.codepde_path / 'solvers' / 'dolfinx_templates' / template_file
        
        if template_path.exists():
            try:
                template_code = template_path.read_text()
                template_name = template_file.replace('_template.py', '').replace('_', ' ').title()
                return (template_name, template_code)
            except Exception as e:
                print(f"   ⚠️  Failed to load template {template_file}: {e}")
                return (None, None)
        else:
            print(f"   ⚠️  Template not found: {template_path}")
            return (None, None)
    
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
