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
from typing import Dict, Any

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
            
            # 使用 CodePDE 的 LLM API 直接生成代码
            # 注意：这里我们简化了 CodePDE 的完整流程，只做单次生成
            messages = self._prepare_messages(prompt, context)
            
            # 调用 LLM
            response = self.generate_response_fn(messages, cfg)
            
            # 提取代码
            code = self._extract_code(response)
            
            latency = time.time() - start_time
            
            # 构建 usage 信息
            usage = {
                'latency_sec': latency,
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
                'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'output_tokens': getattr(response.usage, 'completion_tokens', 0),
                'cost_usd': 0.0,  # TODO: 根据 model 计算成本
            }
            
            return AgentResponse(
                success=True,
                code=code,
                raw_response=str(response),
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
    
    def cleanup(self):
        """清理资源"""
        # 从 sys.path 中移除 CodePDE
        if str(self.codepde_path) in sys.path:
            sys.path.remove(str(self.codepde_path))
