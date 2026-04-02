"""
pdebench/oracle/docker_bridge.py
=================================

宿主机侧的 Docker 桥接层：在容器内运行 oracle，通过挂载卷交换输入/输出。

调用方式（由 OracleSolver.solve 在 use_docker=True 时调用）：

  from pdebench.oracle.docker_bridge import solve_via_docker
  result = solve_via_docker(case_spec, library="dealii")

数据流：
  1. 将 case_spec 序列化为临时目录下的 case_spec.json
  2. docker run --rm -v <tmpdir>:<tmpdir> <image>
         python -m pdebench.oracle.runner <case_spec.json> <out/> <library>
  3. 容器在 <tmpdir>/out/ 写入 reference.npy + meta.json
  4. 读回文件，重建 OracleResult 返回给主流程
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ._types import OracleResult

# 默认镜像名（可通过 docker_image 参数覆盖）
# 格式：<Docker Hub 用户名>/<镜像名>:latest
# 请与 docker/build_images.sh 里的 REPO 变量保持一致
DEFAULT_IMAGES: Dict[str, str] = {
    "dealii":    "pdebench/dealii:latest",
    "firedrake": "pdebench/firedrake:latest",
}

# 项目根目录（用于将源码挂载进容器，使 runner.py 能 import pdebench）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def solve_via_docker(
    case_spec:    Dict[str, Any],
    library:      str,
    docker_image: Optional[str] = None,
    timeout_sec:  int = 600,
) -> OracleResult:
    """
    在 Docker 容器内运行 oracle 求解器。

    Args:
        case_spec:    oracle_config 字典（与本地 OracleSolver.solve() 入参相同）。
        library:      'dealii' 或 'firedrake'。
        docker_image: 覆盖默认镜像名（None 时使用 DEFAULT_IMAGES[library]）。
        timeout_sec:  docker run 的超时时间（秒）。

    Returns:
        OracleResult，与本地执行结果格式完全一致。

    Raises:
        RuntimeError: Docker 调用失败时抛出，附带 stdout/stderr。
        FileNotFoundError: Docker 未安装或镜像不存在时抛出。
    """
    if library not in DEFAULT_IMAGES:
        raise ValueError(
            f"docker_bridge: unsupported library {library!r}. "
            f"Expected one of {list(DEFAULT_IMAGES)}."
        )

    image = docker_image or DEFAULT_IMAGES[library]

    _check_docker_available()

    with tempfile.TemporaryDirectory(prefix="pdebench_oracle_docker_") as tmpdir:
        work = Path(tmpdir)
        case_json = work / "case_spec.json"
        out_dir   = work / "out"
        out_dir.mkdir()

        case_json.write_text(json.dumps(case_spec))

        cmd = _build_docker_cmd(image, case_json, out_dir, library)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Docker oracle timed out after {timeout_sec}s "
                f"(image={image}, library={library}).\n"
                f"stdout: {exc.stdout or ''}\nstderr: {exc.stderr or ''}"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Docker oracle failed (exit {proc.returncode}, "
                f"image={image}, library={library}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        return _read_oracle_result(out_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_docker_available() -> None:
    """确认 docker 命令可用，否则提前报错。"""
    if shutil.which("docker") is None:
        raise FileNotFoundError(
            "Docker is not installed or not found in PATH. "
            "Please install Docker: https://docs.docker.com/get-docker/"
        )


def _build_docker_cmd(
    image:     str,
    case_json: Path,
    out_dir:   Path,
    library:   str,
) -> list[str]:
    """
    构造 docker run 命令。

    挂载策略：
      - tmpdir (case_json 与 out_dir 所在目录): 容器内路径与宿主机相同
      - 项目源码根目录: 同样以相同路径挂载，并设置 PYTHONPATH，
        使容器内 runner.py 能直接 import pdebench（无需将代码烧入镜像）
    """
    tmpdir = case_json.parent

    cmd = [
        "docker", "run", "--rm",
        # 挂载临时工作目录（输入 JSON + 输出结果）
        "-v", f"{tmpdir}:{tmpdir}",
        # 挂载项目源码，使容器内可 import pdebench
        "-v", f"{_PROJECT_ROOT}:{_PROJECT_ROOT}",
        "-e", f"PYTHONPATH={_PROJECT_ROOT}",
        image,
        "python", "-m", "pdebench.oracle.runner",
        str(case_json),
        str(out_dir),
        library,
    ]
    return cmd


def _read_oracle_result(out_dir: Path) -> OracleResult:
    """从容器输出目录中读取并重建 OracleResult。"""
    ref_npy  = out_dir / "reference.npy"
    meta_json = out_dir / "meta.json"

    if not ref_npy.exists():
        raise FileNotFoundError(
            f"Docker oracle: reference.npy not found in {out_dir}. "
            "The container may have crashed silently."
        )
    if not meta_json.exists():
        raise FileNotFoundError(
            f"Docker oracle: meta.json not found in {out_dir}. "
            "The container may have crashed silently."
        )

    reference = np.load(str(ref_npy))
    meta      = json.loads(meta_json.read_text())

    return OracleResult(
        baseline_error = float(meta["baseline_error"]),
        baseline_time  = float(meta["baseline_time"]),
        reference      = reference,
        solver_info    = meta["solver_info"],
        num_dofs       = int(meta["num_dofs"]),
    )
