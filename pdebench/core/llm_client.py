"""
LLM API 客户端 - 调用各种LLM生成solver代码
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应结果"""
    success: bool
    code: str
    raw_response: str
    model: str
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


def extract_code(response: str) -> str:
    """从LLM响应中提取Python代码"""
    # 匹配 ```python ... ``` 代码块
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # 匹配 ``` ... ``` 代码块
    code_blocks = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # 尝试整个响应作为代码（如果看起来像Python）
    lines = response.strip().split('\n')
    if any(line.strip().startswith(('import ', 'from ', 'def ', 'class ')) for line in lines[:10]):
        return response.strip()
    
    return response.strip()


class LLMClient:
    """统一的LLM客户端"""
    
    SUPPORTED_AGENTS = {
        'gpt-4o': {'provider': 'openai', 'model': 'gpt-4o'},
        'gpt-4o-mini': {'provider': 'openai', 'model': 'gpt-4o-mini'},
        'gpt-5.1': {'provider': 'openai', 'model': 'gpt-5.1'},
        'o3-mini': {'provider': 'openai', 'model': 'o3-mini'},
        'sonnet-3.5': {'provider': 'anthropic', 'model': 'anthropic.claude-3-5-sonnet-20241022-v2:0'},
        'sonnet-3.6': {'provider': 'anthropic', 'model': 'anthropic.claude-sonnet-4-20250514-v1:0'},
        'haiku': {'provider': 'anthropic', 'model': 'anthropic.claude-3-haiku-20240307-v1:0'},
        'gemini': {'provider': 'google', 'model': 'gemini-3.0-pro'},
    }
    
    def __init__(self, agent_name: str, temperature: float = 0.0):
        """
        初始化LLM客户端
        
        Args:
            agent_name: 代理名称，如 'gpt-4o', 'sonnet-3.5', 'gemini'
            temperature: 采样温度
        """
        if agent_name not in self.SUPPORTED_AGENTS:
            raise ValueError(f"Unknown agent: {agent_name}. Supported: {list(self.SUPPORTED_AGENTS.keys())}")
        
        self.agent_name = agent_name
        self.config = self.SUPPORTED_AGENTS[agent_name]
        self.provider = self.config['provider']
        self.model = self.config['model']
        self.temperature = temperature
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self):
        """初始化API客户端"""
        if self.provider == 'openai':
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY environment variable")
            self.client = OpenAI(api_key=api_key)
            
        elif self.provider == 'anthropic':
            import boto3
            self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
            
        elif self.provider == 'google':
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing GOOGLE_API_KEY environment variable")
            self.client = genai.Client(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        调用LLM生成代码
        
        Args:
            prompt: 用户prompt
            system_prompt: 系统prompt（可选）
        
        Returns:
            LLMResponse对象
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert in numerical PDEs and FEniCSx/dolfinx. "
                "Generate complete, runnable Python code for the given PDE problem. "
                "Your code should be well-structured and efficient."
            )
        
        try:
            if self.provider == 'openai':
                return self._call_openai(prompt, system_prompt)
            elif self.provider == 'anthropic':
                return self._call_anthropic(prompt, system_prompt)
            elif self.provider == 'google':
                return self._call_google(prompt, system_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return LLMResponse(
                success=False,
                code="",
                raw_response="",
                model=self.model,
                error=str(e)
            )
    
    def _call_openai(self, prompt: str, system_prompt: str) -> LLMResponse:
        """调用OpenAI API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # o3-mini不支持temperature
        kwargs = {"model": self.model, "messages": messages}
        if self.agent_name != 'o3-mini':
            kwargs["temperature"] = self.temperature
        
        response = self.client.chat.completions.create(**kwargs)
        
        raw_response = response.choices[0].message.content.strip()
        code = extract_code(raw_response)
        
        usage = None
        if response.usage:
            usage = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
        
        return LLMResponse(
            success=True,
            code=code,
            raw_response=raw_response,
            model=self.model,
            usage=usage
        )
    
    def _call_anthropic(self, prompt: str, system_prompt: str) -> LLMResponse:
        """调用Anthropic (via AWS Bedrock)"""
        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + prompt}
        ]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "temperature": self.temperature,
            "messages": messages
        }
        
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read().decode("utf-8"))
        raw_response = response_body["content"][0]["text"]
        code = extract_code(raw_response)
        
        usage = None
        if 'usage' in response_body:
            usage = {
                'input_tokens': response_body['usage'].get('input_tokens', 0),
                'output_tokens': response_body['usage'].get('output_tokens', 0)
            }
        
        return LLMResponse(
            success=True,
            code=code,
            raw_response=raw_response,
            model=self.model,
            usage=usage
        )
    
    def _call_google(self, prompt: str, system_prompt: str) -> LLMResponse:
        """调用Google Gemini"""
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=system_prompt
            )
        )
        
        raw_response = response.text
        code = extract_code(raw_response)
        
        usage = None
        if response.usage_metadata:
            usage = {
                'input_tokens': response.usage_metadata.prompt_token_count,
                'output_tokens': response.usage_metadata.candidates_token_count
            }
        
        return LLMResponse(
            success=True,
            code=code,
            raw_response=raw_response,
            model=self.model,
            usage=usage
        )


def call_llm(agent_name: str, prompt: str, temperature: float = 0.0) -> LLMResponse:
    """
    便捷函数：调用LLM生成代码
    
    Args:
        agent_name: 代理名称
        prompt: 完整prompt
        temperature: 采样温度
    
    Returns:
        LLMResponse对象
    """
    client = LLMClient(agent_name, temperature)
    return client.generate(prompt)
