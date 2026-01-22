"""
Agent 注册中心

统一管理所有 Code Agent 的注册和实例化。
"""

from typing import Dict, Type, Optional
from .base_agent import BaseAgent


class AgentRegistry:
    """Agent 注册中心"""
    
    _registry: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]):
        """
        注册 Agent
        
        Args:
            name: Agent 名称（如 'swe-agent'）
            agent_class: Agent 类
        """
        cls._registry[name] = agent_class
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict] = None) -> BaseAgent:
        """
        创建 Agent 实例
        
        Args:
            name: Agent 名称
            config: 可选配置（如果不提供，使用默认配置）
        
        Returns:
            Agent 实例
        
        Raises:
            ValueError: 如果 agent 未注册
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown agent: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        agent_class = cls._registry[name]
        return agent_class(name, config)
    
    @classmethod
    def list_agents(cls) -> list:
        """列出所有已注册的 Agent"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查 agent 是否已注册"""
        return name in cls._registry


def get_agent(name: str, config: Optional[Dict] = None) -> BaseAgent:
    """
    便捷函数：获取 Agent 实例
    
    Args:
        name: Agent 名称
        config: 可选配置
    
    Returns:
        Agent 实例
    """
    return AgentRegistry.create(name, config)
