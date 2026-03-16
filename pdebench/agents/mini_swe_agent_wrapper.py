"""
Mini SWE-Agent Wrapper

Mini SWE-Agent 是一个轻量级的代码编辑 agent，基于 litellm 支持所有主流 LLM。

GitHub: https://github.com/SWE-agent/mini-swe-agent
安装: pip install mini-swe-agent
"""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, AgentResponse


def _build_task(prompt: str) -> str:
    """将 benchmark prompt 包装为 agent 任务描述。

    prompt 已包含完整的方程描述、接口定义和通过标准，无需额外改动内容，
    只需在末尾告知 agent 将代码写入当前目录的 solver.py。
    """
    return (
        prompt
        + "\n\n---\n\n"
        "Write the complete implementation to `solver.py` in the current directory.\n"
        "The file must define `def solve(case_spec: dict) -> dict` at module level.\n"
        "Do not print anything to stdout during the solve() call.\n"
    )


class MiniSWEAgentWrapper(BaseAgent):
    """
    Mini SWE-Agent Wrapper

    工作流程：
    1. 在临时目录创建空白 workspace
    2. 使用与 pure-LLM 完全相同的 prompt 调用 mini-swe-agent Python API
    3. Agent 将代码写入 solver.py 后读取文件内容作为代码返回
    4. 清理临时目录
    """

    def _setup(self):
        """初始化并检查 mini-swe-agent 依赖"""
        try:
            import yaml  # noqa: F401
            from minisweagent.agents.interactive import InteractiveAgent  # noqa: F401
            from minisweagent.config import builtin_config_dir  # noqa: F401
            from minisweagent.models import get_model  # noqa: F401
            from minisweagent.environments.local import LocalEnvironment  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"mini-swe-agent not installed: {e}. "
                "Run: pip install mini-swe-agent"
            ) from e

        # LLM 配置（尽量与 `mini` CLI 保持一致）
        self.model_name: str = self.config.get("model", "openai/gpt-4o")
        self.api_key: Optional[str] = self.config.get("api_key", None)
        self.base_url: Optional[str] = self.config.get("base_url", None)
        self.cost_limit: float = float(self.config.get("cost_limit", 3.0))
        self.step_limit: int = int(self.config.get("step_limit", 50))
        self.timeout: int = int(self.config.get("timeout", 600))

        # 临时工作目录（每次 generate_solution 调用创建，cleanup 时销毁）
        self._tmp_dir: Optional[tempfile.TemporaryDirectory] = None

    def generate_solution(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """
        调用 mini-swe-agent 生成 PDE solver 代码

        步骤：
        1. 创建临时 workspace
        2. 以与 pure-LLM 完全相同的 prompt 调用 agent
        3. 读取 agent 写入的 solver.py 内容
        4. 验证代码包含 def solve 后返回 AgentResponse
        """
        import yaml
        from minisweagent.agents.interactive import InteractiveAgent
        from minisweagent.config import builtin_config_dir
        from minisweagent.models import get_model
        from minisweagent.environments.local import LocalEnvironment

        start_time = time.time()

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="pdebench_miniswe_")
        workspace = Path(self._tmp_dir.name)

        try:
            task_description = _build_task(prompt)

            # 对齐 `mini` CLI: 加载内置 mini.yaml，并在此基础上覆盖少量配置
            config_path = Path(builtin_config_dir) / "mini.yaml"
            mini_config = yaml.safe_load(config_path.read_text())
            mini_config.setdefault("model", {})
            mini_config.setdefault("agent", {})
            mini_config.setdefault("environment", {})

            if self.api_key:
                mini_config["model"].setdefault("model_kwargs", {})["api_key"] = self.api_key
            if self.base_url:
                mini_config["model"].setdefault("model_kwargs", {})["base_url"] = self.base_url

            mini_config["agent"]["mode"] = "yolo"
            mini_config["agent"]["confirm_exit"] = False
            mini_config["agent"]["cost_limit"] = self.cost_limit
            mini_config["agent"]["step_limit"] = self.step_limit

            model = get_model(self.model_name, mini_config.get("model", {}))
            env = LocalEnvironment(
                cwd=str(workspace),
                **mini_config.get("environment", mini_config.get("env", {})),
            )
            agent = InteractiveAgent(
                model,
                env,
                **mini_config.get("agent", {}),
            )

            exit_status, result = agent.run(task_description)

            latency = time.time() - start_time
            solver_path = workspace / "solver.py"
            return _validate_solver(
                solver_path,
                f"{exit_status}: {result}",
                latency,
                self.agent_name,
            )

        except Exception as e:
            latency = time.time() - start_time
            return AgentResponse(
                success=False,
                code="",
                raw_response="",
                agent_name=self.agent_name,
                error=str(e),
                usage={"latency_sec": latency},
            )

    def cleanup(self):
        """清理临时 workspace"""
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except Exception:
                try:
                    shutil.rmtree(self._tmp_dir.name, ignore_errors=True)
                except Exception:
                    pass
            self._tmp_dir = None


def _validate_solver(
    solver_path: Path,
    raw_output: str,
    latency: float,
    agent_name: str,
) -> AgentResponse:
    """读取 solver.py，做基本验证后构造 AgentResponse。"""
    if not solver_path.exists():
        return AgentResponse(
            success=False,
            code="",
            raw_response=raw_output,
            agent_name=agent_name,
            error="solver.py not found after agent run",
            usage={"latency_sec": latency},
        )

    code = solver_path.read_text()

    if "def solve" not in code:
        return AgentResponse(
            success=False,
            code=code,
            raw_response=raw_output,
            agent_name=agent_name,
            error="solver.py does not contain 'def solve' after agent run",
            usage={"latency_sec": latency},
        )

    return AgentResponse(
        success=True,
        code=code,
        raw_response=raw_output,
        agent_name=agent_name,
        usage={"latency_sec": latency},
    )
