"""
OpenHands Wrapper

OpenHands（原 OpenDevin）是一个功能全面的代码 agent 框架。

GitHub: https://github.com/All-Hands-AI/OpenHands
安装:
  CLI 模式（默认）: pip install openhands-cli
  SDK 模式:         pip install openhands-sdk openhands-tools

支持两种调用模式（通过配置 "mode" 字段切换）：
  - "cli"（默认）：通过 subprocess 调用 `openhands --headless` 命令行
  - "sdk"：通过 Python SDK 调用（需要 openhands-sdk）
"""

import os
import json
import shutil
import subprocess
import sys
import tempfile
import time
import threading
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
        "You must solve the task independently from the problem statement only.\n"
        "Do not inspect, search, open, import, copy, or adapt any files outside the current workspace directory.\n"
        "In particular, do not read any project files, benchmark code, evaluator code, oracle solvers, cached results, prompts from other cases, or previously generated solvers.\n"
        "Do not attempt to recover hidden reference implementations from the repository, git history, shell commands, Python imports, or filesystem traversal.\n"
        "Any attempt to use repository files or reference solutions as external help is forbidden and counts as cheating.\n"
    )


def _infer_openhands_env(model_name: str) -> Dict[str, str]:
    """根据模型名推断 OpenHands 所需的 LLM 环境变量覆盖。"""
    model_lower = model_name.lower()

    if model_lower.startswith(("gpt-", "o1", "o3")) or model_lower.startswith("openai/"):
        return {
            "LLM_MODEL": model_name.removeprefix("openai/"),
            "LLM_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "LLM_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }

    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return {
            "LLM_MODEL": model_name.removeprefix("anthropic/"),
            "LLM_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "LLM_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
        }

    if "gemini" in model_lower or model_lower.startswith("google/"):
        return {
            "LLM_MODEL": model_name.removeprefix("google/"),
            "LLM_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
            "LLM_BASE_URL": os.environ.get("GOOGLE_BASE_URL", ""),
        }

    return {"LLM_MODEL": model_name}


def _infer_max_output_tokens(model_name: str) -> int:
    """为不同 provider 选择一个保守的 max_output_tokens，避免沿用用户本机旧配置。"""
    model_lower = model_name.lower()
    if model_lower.startswith(("gpt-", "o1", "o3")) or model_lower.startswith("openai/"):
        return 16000
    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return 32000
    if "gemini" in model_lower or model_lower.startswith("google/"):
        return 16000
    return 8192


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


class OpenHandsWrapper(BaseAgent):
    """
    OpenHands Wrapper

    支持两种运行模式（通过配置 "mode" 控制）：

    1. CLI 模式（mode="cli"，默认）：
       - 调用 `python -m openhands --headless -t <task> --workspace <dir>`
       - 兼容性最佳，适合大多数安装场景

    2. SDK 模式（mode="sdk"）：
       - 使用 openhands-sdk Python API
       - 需要额外安装 openhands-sdk 和 openhands-tools

    工作流程：
    1. 在临时目录创建空白 workspace
    2. 以与 pure-LLM 完全相同的 prompt 调用 agent，agent 负责创建 solver.py
    3. 读取 solver.py 内容返回 AgentResponse
    """

    def _setup(self):
        """初始化并检查依赖"""
        self.mode: str = self.config.get("mode", "cli")
        self.model_name: str = self.config.get(
            "model", "anthropic/claude-sonnet-4-5-20250929"
        )
        self.api_key: Optional[str] = self.config.get("api_key", None)
        self.base_url: Optional[str] = self.config.get("base_url", None)
        self.timeout: int = int(self.config.get("timeout", 600))
        self.max_output_tokens: int = int(
            self.config.get("max_output_tokens", _infer_max_output_tokens(self.model_name))
        )

        if self.mode == "sdk":
            self._check_sdk_deps()
        else:
            self._check_cli_deps()

        self._tmp_dir: Optional[tempfile.TemporaryDirectory] = None

    def _check_cli_deps(self):
        """检查 openhands CLI 是否可用（支持 `openhands` 命令或 `python -m openhands`）"""
        if shutil.which("openhands") is None:
            # 尝试 python -m openhands
            result = subprocess.run(
                [sys.executable, "-m", "openhands", "--help"],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "openhands CLI not found. Run: pip install openhands-cli"
                )

    def _check_sdk_deps(self):
        """检查 openhands SDK 是否可用"""
        try:
            import openhands.sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"openhands SDK not installed: {e}. "
                "Run: pip install openhands-sdk openhands-tools"
            ) from e

    def generate_solution(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """调用 OpenHands 生成 PDE solver 代码"""
        if self.mode == "sdk":
            return self._generate_via_sdk(prompt, context)
        return self._generate_via_cli(prompt, context)

    # ------------------------------------------------------------------
    # CLI 模式
    # ------------------------------------------------------------------

    def _generate_via_cli(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """通过 subprocess 调用 openhands --headless，并实时透传日志"""
        start_time = time.time()

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="pdebench_openhands_")
        workspace = Path(self._tmp_dir.name)

        try:
            task_description = _build_task(prompt)
            cli_home = self._prepare_cli_home(workspace)

            cmd = [
                "openhands",
                "--headless",
                "--override-with-envs",
                "--exit-without-confirmation",
                "-t", task_description,
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self._build_env(cli_home),
                cwd=str(workspace),
                bufsize=1,
            )

            output_chunks: list[str] = []

            def _stream_output() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    output_chunks.append(line)
                    # 实时透传到 benchmark 所在终端，便于观察 agent 进度
                    print(line, end="", flush=True)

            reader = threading.Thread(target=_stream_output, daemon=True)
            reader.start()

            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                reader.join(timeout=2)
                latency = time.time() - start_time
                return AgentResponse(
                    success=False,
                    code="",
                    raw_response="".join(output_chunks),
                    agent_name=self.agent_name,
                    error=f"OpenHands CLI timed out after {self.timeout}s",
                    usage={"latency_sec": latency},
                )

            reader.join(timeout=2)

            latency = time.time() - start_time
            raw_output = "".join(output_chunks)

            # OpenHands CLI 返回非 0 时，优先直接暴露真实错误
            if proc.returncode != 0:
                error_preview = raw_output.strip() or "no output"
                if len(error_preview) > 500:
                    error_preview = error_preview[:500] + "..."
                return AgentResponse(
                    success=False,
                    code="",
                    raw_response=raw_output,
                    agent_name=self.agent_name,
                    error=f"OpenHands CLI exited with code {proc.returncode}: {error_preview}",
                    usage={"latency_sec": latency},
                )

            return _validate_solver(
                workspace / "solver.py", raw_output, latency, self.agent_name
            )

        except subprocess.TimeoutExpired:
            latency = time.time() - start_time
            return AgentResponse(
                success=False,
                code="",
                raw_response="",
                agent_name=self.agent_name,
                error=f"OpenHands CLI timed out after {self.timeout}s",
                usage={"latency_sec": latency},
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

    # ------------------------------------------------------------------
    # SDK 模式
    # ------------------------------------------------------------------

    def _generate_via_sdk(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """通过 openhands-sdk Python API 调用"""
        from openhands.sdk import LLM, Agent, Conversation, Tool  # type: ignore

        start_time = time.time()

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="pdebench_openhands_")
        workspace = Path(self._tmp_dir.name)

        try:
            task_description = _build_task(prompt)

            api_key = (
                self.api_key
                or os.environ.get("LLM_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

            llm = LLM(
                model=self.model_name,
                api_key=api_key,
                base_url=self.base_url or os.environ.get("LLM_BASE_URL"),
            )

            try:
                from openhands.tools.file_editor import FileEditorTool  # type: ignore
                from openhands.tools.terminal import TerminalTool  # type: ignore
                tools = [Tool(name=FileEditorTool.name), Tool(name=TerminalTool.name)]
            except ImportError:
                tools = []

            agent = Agent(llm=llm, tools=tools)
            conversation = Conversation(agent=agent, workspace=str(workspace))
            conversation.send_message(task_description)
            conversation.run()

            latency = time.time() - start_time
            return _validate_solver(
                workspace / "solver.py", "", latency, self.agent_name
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

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _prepare_cli_home(self, workspace: Path) -> Path:
        """创建隔离的 HOME/.openhands，并写入适配当前模型的 agent_settings.json。"""
        source_settings = Path.home() / ".openhands" / "agent_settings.json"
        cli_home = workspace / ".openhands_home"
        persistence_dir = cli_home / ".openhands"
        persistence_dir.mkdir(parents=True, exist_ok=True)
        (persistence_dir / "conversations").mkdir(parents=True, exist_ok=True)

        if source_settings.exists():
            settings = json.loads(source_settings.read_text())
        else:
            settings = {
                "llm": {},
                "tools": [
                    {"name": "terminal", "params": {}},
                    {"name": "file_editor", "params": {}},
                    {"name": "task_tracker", "params": {}},
                ],
                "mcp_config": {},
                "filter_tools_regex": None,
                "include_default_tools": ["FinishTool", "ThinkTool"],
                "agent_context": None,
                "system_prompt_filename": "system_prompt.j2",
                "security_policy_filename": "security_policy.j2",
                "system_prompt_kwargs": {"llm_security_analyzer": True},
                "critic": None,
                "kind": "Agent",
            }

        inferred = _infer_openhands_env(self.model_name) if self.model_name else {}
        model_name = inferred.get("LLM_MODEL") or self.model_name
        api_key = self.api_key or inferred.get("LLM_API_KEY") or settings.get("llm", {}).get("api_key")
        base_url = self.base_url or inferred.get("LLM_BASE_URL") or settings.get("llm", {}).get("base_url")

        settings.setdefault("llm", {})
        settings["llm"]["model"] = model_name
        settings["llm"]["api_key"] = api_key
        settings["llm"]["base_url"] = base_url
        settings["llm"]["max_output_tokens"] = self.max_output_tokens
        settings["llm"]["drop_params"] = True
        settings["llm"]["modify_params"] = True

        if settings.get("condenser", {}).get("llm"):
            settings["condenser"]["llm"]["model"] = model_name
            settings["condenser"]["llm"]["api_key"] = api_key
            settings["condenser"]["llm"]["base_url"] = base_url
            settings["condenser"]["llm"]["max_output_tokens"] = min(
                self.max_output_tokens, 8192
            )
            settings["condenser"]["llm"]["drop_params"] = True
            settings["condenser"]["llm"]["modify_params"] = True

        (persistence_dir / "agent_settings.json").write_text(
            json.dumps(settings, ensure_ascii=False)
        )
        return cli_home

    def _build_env(self, cli_home: Optional[Path] = None) -> Dict[str, str]:
        """构建包含 LLM 配置的环境变量字典（用于 CLI 子进程）"""
        env = dict(os.environ)

        # 优先使用配置文件显式指定；否则根据模型名从宿主环境推断 provider 配置，
        # 避免只覆盖模型名却沿用 ~/.openhands/agent_settings.json 中的默认 proxy。
        inferred = _infer_openhands_env(self.model_name) if self.model_name else {}
        for key, value in inferred.items():
            if value:
                env[key] = value

        if self.model_name:
            env["LLM_MODEL"] = self.model_name.removeprefix("openai/").removeprefix("anthropic/").removeprefix("google/")
        if self.api_key:
            env["LLM_API_KEY"] = self.api_key
        if self.base_url:
            env["LLM_BASE_URL"] = self.base_url
        if cli_home is not None:
            env["HOME"] = str(cli_home)
        return env

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
