"""
Code Agent 模块

提供统一的 Code Agent 接口和实现。

使用方式：
    from pdebench.agents import get_agent, AgentRegistry
    
    # 创建 agent
    agent = get_agent('swe-agent', config={'model': 'gpt-4'})
    
    # 生成解决方案
    response = agent.generate_solution(prompt, context)
    
    # 清理
    agent.cleanup()
"""

from .base_agent import BaseAgent, AgentResponse
from .agent_registry import AgentRegistry, get_agent

# 导入所有 agent wrappers
from .swe_agent_wrapper import SWEAgentWrapper
from .openhands_wrapper import OpenhandsWrapper
from .qwen_agent_wrapper import QwenAgentWrapper
from .codepde_wrapper import CodePDEWrapper

# 注册所有 Agent
AgentRegistry.register('swe-agent', SWEAgentWrapper)
AgentRegistry.register('openhands', OpenhandsWrapper)
AgentRegistry.register('qwen-agent', QwenAgentWrapper)
AgentRegistry.register('codepde', CodePDEWrapper)

# 导出
__all__ = [
    'BaseAgent',
    'AgentResponse',
    'AgentRegistry',
    'get_agent',
    'SWEAgentWrapper',
    'OpenhandsWrapper',
    'QwenAgentWrapper',
    'CodePDEWrapper',
]
