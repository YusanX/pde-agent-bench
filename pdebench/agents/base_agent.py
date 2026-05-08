"""
Code Agent 统一接口定义

所有 Code Agent wrapper 必须继承 BaseAgent 并实现相应方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AgentResponse:
    """Agent 响应结果（统一格式，与 LLMResponse 保持一致）"""
    success: bool
    code: str                          # 生成的代码
    raw_response: str                  # 原始响应（用于调试）
    agent_name: str
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # token、成本、时间等统计


class BaseAgent(ABC):
    """
    Code Agent 统一接口
    
    设计原则：
    1. 与 LLMClient 保持相同的接口风格
    2. 使用相同的 prompt（不重新生成）
    3. 返回统一的 AgentResponse
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any] = None):
        """
        初始化 Agent
        
        Args:
            agent_name: Agent 名称（如 'swe-agent', 'openhands'）
            config: Agent 配置（可选，从 configs/{agent_name}.json 加载）
        """
        self.agent_name = agent_name
        self.config = config or {}
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """
        初始化 Agent（子类实现）
        
        用于：
        - 检查依赖
        - 初始化客户端
        - 创建工作目录等
        """
        pass
    
    @abstractmethod
    def generate_solution(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> AgentResponse:
        """
        生成 PDE 求解代码
        
        ⚠️ 重要：prompt 参数是已经由 generate_prompt() 生成的完整 prompt，
                  不要在这里重新生成或修改 prompt！
        
        Args:
            prompt: 完整的任务描述 prompt（与 LLM 使用的完全相同）
            context: 上下文信息，包含：
                - case_id: str
                - case_spec: Dict (完整的 case 配置)
                - oracle_info: Dict (oracle 结果)
        
        Returns:
            AgentResponse 对象
        """
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        获取使用统计（可选，用于成本分析）
        
        Returns:
            统计信息字典，可能包含：
            - total_tokens: int
            - input_tokens: int
            - output_tokens: int
            - latency_sec: float
            - cost_usd: float
        """
        return {}
    
    @abstractmethod
    def cleanup(self):
        """
        清理资源
        
        用于：
        - 删除临时文件
        - 关闭进程
        - 释放资源等
        """
        pass
