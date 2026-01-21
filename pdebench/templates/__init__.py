"""
PDEBench Templates Module

提供动态生成 Prompt 和测试脚本的模板系统
"""

from .prompts import generate_prompt
from .scripts import generate_test_script

__all__ = ['generate_prompt', 'generate_test_script']




