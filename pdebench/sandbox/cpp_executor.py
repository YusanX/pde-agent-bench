"""
cpp_executor.py
===============

Execution sandbox for agent-generated **C++ deal.II** solver code.

Workflow for each execute() call:
  1. Write agent's solver.cpp + CMakeLists.txt template to a build dir
  2. cmake configure  (deal.II + nlohmann/json)
  3. make -j N        (compile solver)
  4. Run binary with  <case_spec.json> <output_dir>
  5. Read solution_grid.bin + meta.json → convert to solution.npz

The agent's C++ code must satisfy the following interface:

  int main(int argc, char* argv[]) {
      // argv[1]: path to case_spec.json
      // argv[2]: output directory (already exists)
      // ...solve...
      // Write argv[2]/solution_grid.bin  (float64, row-major [ny, nx])
      // Write argv[2]/meta.json          (see below)
  }

  meta.json required fields:
    { "nx": <int>, "ny": <int>,
      "wall_time_sec": <float>,
      "solver_info": {
          "mesh_resolution": <int>,
          "element_degree":  <int>,
          "ksp_type":        <str>,
          "pc_type":         <str>,
          "rtol":            <float>
      }
    }

Alternatively, agent may write solution.npz directly (numpy format)
with field "u" of shape (ny, nx) – in that case the binary step is skipped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .executor import ExecutionResult


# Prefer local headers that may already be present from the oracle build,
# so agent execution does not depend on a fresh network download.
_LOCAL_NLOHMANN_INCLUDE_DIRS = [
    Path(__file__).resolve().parents[1] / "oracle" / "dealii_oracle" / "build" / "_deps" / "nlohmann_json-src" / "include",
    Path(__file__).resolve().parents[1] / "oracle" / "dealii_oracle" / "build" / "_deps" / "nlohmann_json-src" / "single_include",
]
_LOCAL_NLOHMANN_INCLUDE_HINTS = "\n".join(
    f'      "{path}"' for path in _LOCAL_NLOHMANN_INCLUDE_DIRS
)


# ---------------------------------------------------------------------------
# CMakeLists.txt template injected next to agent's solver.cpp
# ---------------------------------------------------------------------------
_CMAKE_TEMPLATE = """\
cmake_minimum_required(VERSION 3.13.4)

# --- deal.II ----------------------------------------------------------------
find_package(deal.II 9.3
  REQUIRED
  HINTS
    $ENV{DEAL_II_DIR}
    /opt/homebrew/opt/deal.ii
    /usr/local/opt/deal.ii
    /usr/local
    $ENV{HOME}/dealii-candi/deal.II-v9.7.0
)
deal_ii_initialize_cached_variables()

project(agent_solver CXX)

# --- nlohmann/json ----------------------------------------------------------
find_package(nlohmann_json 3.9 QUIET)
if(NOT nlohmann_json_FOUND)
  foreach(_prefix
      /opt/homebrew/include
      /usr/local/include
__LOCAL_NLOHMANN_INCLUDE_HINTS__
  )
    if(EXISTS "${_prefix}/nlohmann/json.hpp")
      add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
      target_include_directories(nlohmann_json::nlohmann_json
        INTERFACE "${_prefix}")
      set(nlohmann_json_FOUND TRUE)
      break()
    endif()
  endforeach()
endif()
if(NOT nlohmann_json_FOUND)
  # Last resort: FetchContent (requires internet)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- Agent solver -----------------------------------------------------------
add_executable(agent_solver solver.cpp)
deal_ii_setup_target(agent_solver)
# deal_ii_setup_target() uses the plain target_link_libraries signature in
# newer deal.II releases, so keep the signature consistent here.
target_link_libraries(agent_solver nlohmann_json::nlohmann_json)
target_compile_features(agent_solver PRIVATE cxx_std_17)
set_target_properties(agent_solver
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
"""
_CMAKE_TEMPLATE = _CMAKE_TEMPLATE.replace(
    "__LOCAL_NLOHMANN_INCLUDE_HINTS__",
    _LOCAL_NLOHMANN_INCLUDE_HINTS,
)

_PLACEHOLDER_SOLVER_CPP = """\
#include <iostream>

int main(int argc, char* argv[]) {
  std::cerr << "Placeholder solver should be rebuilt before execution." << std::endl;
  return 1;
}
"""


# ---------------------------------------------------------------------------
# Error parsing helpers (extract structured error from C++ output)
# ---------------------------------------------------------------------------

def _extract_compile_errors(stderr: str, max_lines: int = 30) -> str:
    """
    Return the most relevant lines from a C++ compilation error output.
    Filters for lines containing 'error:' or 'warning:' (first occurrence).
    """
    lines = stderr.splitlines()
    error_lines: List[str] = []
    for line in lines:
        if re.search(r'\berror\b', line, re.IGNORECASE):
            error_lines.append(line)
        if len(error_lines) >= max_lines:
            break
    return "\n".join(error_lines) if error_lines else stderr[:2000]


def _sanitize_cpp_code(code: str) -> str:
    """
    Strip common Markdown fencing from model output before writing solver.cpp.

    Handles cases like:
      ```cpp
      ...
      ```
    as well as a stray first-line language tag such as "cpp".
    """
    text = code.strip()

    fence_match = re.match(r"^```[A-Za-z0-9_+\-]*\s*\n([\s\S]*?)\n```$", text)
    if fence_match:
        text = fence_match.group(1).strip()

    lines = text.splitlines()
    if lines and lines[0].strip().lower() in {"cpp", "c++", "cc", "cxx"}:
        text = "\n".join(lines[1:]).lstrip()

    return text


# ---------------------------------------------------------------------------
# Main executor class
# ---------------------------------------------------------------------------

class CppExecutor:
    """
    Compiles and runs an agent-generated deal.II C++ solver.

    Usage:
        executor = CppExecutor()
        result = executor.execute(
            solver_cpp   = "...",   # C++ source code string
            case_spec    = {...},   # oracle_config from benchmark.jsonl
            outdir       = Path("output/case_id/agent_output"),
            timeout_sec  = 300,
        )

    Docker mode (deal.II not required on host):
        executor = CppExecutor(docker_image="pdebench/dealii:latest")
        result = executor.execute(...)
    """

    _configure_lock = threading.Lock()
    _compile_lock = threading.Lock()

    def __init__(self, docker_image: Optional[str] = None):
        """
        Args:
            docker_image: Docker 镜像名。非 None 时所有 cmake/make/run 步骤都在
                          容器内执行，宿主机无需安装 deal.II。
        """
        self._docker_image = docker_image
        if docker_image is not None:
            if shutil.which("docker") is None:
                raise FileNotFoundError(
                    "Docker is not installed or not found in PATH. "
                    "Please install Docker: https://docs.docker.com/get-docker/"
                )

    def _wrap_docker(
        self,
        cmd: List[str],
        *mount_dirs: Path,
        workdir: Optional[Path] = None,
    ) -> List[str]:
        """
        将一条本地命令包装为 docker run 命令。

        所有 mount_dirs 都以相同路径挂载进容器（host_path:host_path），
        使路径字符串在容器内外保持一致。
        workdir 对应容器内的 -w（工作目录）。
        """
        if self._docker_image is None:
            return cmd

        docker_cmd = ["docker", "run", "--rm"]
        seen: set = set()
        for d in mount_dirs:
            abs_d = str(d.resolve())
            if abs_d not in seen:
                docker_cmd += ["-v", f"{abs_d}:{abs_d}"]
                seen.add(abs_d)
        if workdir is not None:
            docker_cmd += ["-w", str(workdir.resolve())]
        docker_cmd.append(self._docker_image)
        docker_cmd.extend(cmd)
        return docker_cmd

    @staticmethod
    def _shared_build_dir() -> Path:
        return Path(__file__).resolve().parent / ".dealii_agent_build"

    @staticmethod
    def _shared_build_meta_file(build_dir: Path) -> Path:
        return build_dir / "build_meta.json"

    @staticmethod
    def _solver_meta_file(build_dir: Path) -> Path:
        return build_dir / "solver_meta.json"

    @staticmethod
    def _build_signature(deal_ii_dir: str) -> Dict[str, Any]:
        return {
            "deal_ii_dir": deal_ii_dir,
            "cmake_template_sha256": hashlib.sha256(_CMAKE_TEMPLATE.encode("utf-8")).hexdigest(),
        }

    @classmethod
    def _build_signature_matches(cls, build_dir: Path, signature: Dict[str, Any]) -> bool:
        meta_file = cls._shared_build_meta_file(build_dir)
        if not meta_file.exists():
            return False
        try:
            return json.loads(meta_file.read_text()) == signature
        except Exception:
            return False

    @classmethod
    def _solver_signature_matches(cls, build_dir: Path, solver_cpp: str) -> bool:
        meta_file = cls._solver_meta_file(build_dir)
        binary = build_dir / "agent_solver"
        if not meta_file.exists() or not binary.exists():
            return False
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            return False
        solver_hash = hashlib.sha256(_sanitize_cpp_code(solver_cpp).encode("utf-8")).hexdigest()
        return meta.get("solver_sha256") == solver_hash

    @classmethod
    def _write_solver_signature(cls, build_dir: Path, solver_cpp: str) -> None:
        cls._solver_meta_file(build_dir).write_text(json.dumps({
            "solver_sha256": hashlib.sha256(_sanitize_cpp_code(solver_cpp).encode("utf-8")).hexdigest(),
        }, indent=2))

    def _ensure_configured(
        self,
        build_dir: Path,
        deal_ii_dir: str,
        timeout_sec: int,
        t_wall_start: float,
    ) -> Optional[ExecutionResult]:
        signature = self._build_signature(deal_ii_dir)
        with self._configure_lock:
            build_dir.mkdir(parents=True, exist_ok=True)

            if self._build_signature_matches(build_dir, signature):
                return None

            # Reconfigure from scratch when the toolchain/template changes.
            shutil.rmtree(build_dir, ignore_errors=True)
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "CMakeLists.txt").write_text(_CMAKE_TEMPLATE)
            (build_dir / "solver.cpp").write_text(_PLACEHOLDER_SOLVER_CPP)

            cmake_cmd_base = ["cmake", ".", "-DCMAKE_BUILD_TYPE=Release"]
            if deal_ii_dir:
                cmake_cmd_base.append(f"-DDEAL_II_DIR={deal_ii_dir}")

            cmake_cmd = self._wrap_docker(cmake_cmd_base, build_dir, workdir=build_dir)

            configure_timeout = max(300, timeout_sec)
            try:
                cmake_res = subprocess.run(
                    cmake_cmd,
                    cwd=str(build_dir),
                    capture_output=True,
                    text=True,
                    timeout=configure_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.time() - t_wall_start
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    t_agent_run=elapsed,
                    wall_time_sec=elapsed,
                    timeout_occurred=True,
                    error_message=(
                        f"CMake configure timed out after {configure_timeout}s. "
                        "This usually means deal.II discovery or dependency resolution was too slow."
                    ),
                )

            if cmake_res.returncode != 0:
                elapsed = time.time() - t_wall_start
                return ExecutionResult(
                    success=False,
                    exit_code=cmake_res.returncode,
                    stdout=cmake_res.stdout,
                    stderr=cmake_res.stderr,
                    t_agent_run=elapsed,
                    wall_time_sec=elapsed,
                    error_message=(
                        "CMake configure failed.\n"
                        + _extract_compile_errors(cmake_res.stderr)
                    ),
                )

            self._shared_build_meta_file(build_dir).write_text(
                json.dumps(signature, indent=2)
            )

        return None

    def execute(
        self,
        solver_cpp:  str,
        case_spec:   Dict[str, Any],
        outdir:      Path,
        timeout_sec: int = 300,
    ) -> ExecutionResult:
        """
        Compile agent's C++ code and run it against the given case_spec.

        Args:
            solver_cpp:  Full C++ source code as a string.
            case_spec:   Full case dict from benchmark.jsonl (not just oracle_config).
            outdir:      Directory where solution.npz and meta.json will be written.
            timeout_sec: Wall-clock limit for the *run* step (compilation excluded).

        Returns:
            ExecutionResult compatible with the Python executor interface.
        """
        outdir.mkdir(parents=True, exist_ok=True)
        build_dir = self._shared_build_dir()
        run_outdir = outdir / "_cpp_output"
        shutil.rmtree(run_outdir, ignore_errors=True)
        run_outdir.mkdir(parents=True, exist_ok=True)

        t_wall_start = time.time()

        case_json = outdir / "case_spec_agent.json"
        oracle_config = case_spec.get("oracle_config", case_spec)
        case_json.write_text(json.dumps(oracle_config))

        deal_ii_dir = os.environ.get("DEAL_II_DIR", "")
        configure_error = self._ensure_configured(build_dir, deal_ii_dir, timeout_sec, t_wall_start)
        if configure_error is not None:
            return configure_error

        # ── Step 2: compile agent solver in shared build dir ───────────────
        with self._compile_lock:
            solver_file = build_dir / "solver.cpp"
            sanitized_solver_cpp = _sanitize_cpp_code(solver_cpp)
            needs_compile = not self._solver_signature_matches(build_dir, sanitized_solver_cpp)
            if needs_compile:
                solver_file.write_text(sanitized_solver_cpp)

                n_jobs = max(1, os.cpu_count() or 4)
                compile_timeout = max(600, timeout_sec * 2)
                make_cmd_base = ["make", f"-j{n_jobs}", "agent_solver"]
                make_cmd = self._wrap_docker(make_cmd_base, build_dir, workdir=build_dir)
                try:
                    make_res = subprocess.run(
                        make_cmd,
                        cwd=str(build_dir),
                        capture_output=True,
                        text=True,
                        timeout=compile_timeout,
                    )
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.time() - t_wall_start
                    return ExecutionResult(
                        success=False,
                        exit_code=-1,
                        stdout=exc.stdout or "",
                        stderr=exc.stderr or "",
                        t_agent_run=elapsed,
                        wall_time_sec=elapsed,
                        timeout_occurred=True,
                        error_message=(
                            f"Compilation timed out after {compile_timeout}s. "
                            "The generated deal.II solver likely takes too long to build."
                        ),
                    )
                if make_res.returncode != 0:
                    elapsed = time.time() - t_wall_start
                    return ExecutionResult(
                        success=False,
                        exit_code=make_res.returncode,
                        stdout=make_res.stdout,
                        stderr=make_res.stderr,
                        t_agent_run=elapsed,
                        wall_time_sec=elapsed,
                        error_message=(
                            "Compilation failed (make).\n"
                            + _extract_compile_errors(make_res.stderr)
                        ),
                    )
                self._write_solver_signature(build_dir, sanitized_solver_cpp)

        # ── Step 3: run binary ────────────────────────────────────────────
        binary = build_dir / "agent_solver"
        if not binary.exists() and self._docker_image is None:
            elapsed = time.time() - t_wall_start
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="", stderr="",
                t_agent_run=elapsed,
                wall_time_sec=elapsed,
                error_message="Binary 'agent_solver' not found after make.",
            )

        run_cmd_base = [str(binary), str(case_json), str(run_outdir)]
        run_cmd = self._wrap_docker(
            run_cmd_base,
            build_dir, outdir, run_outdir,
        )

        t_run_start = time.time()
        timeout_occurred = False
        try:
            run_res = subprocess.run(
                run_cmd,
                capture_output=True, text=True,
                timeout=timeout_sec,
            )
            exit_code = run_res.returncode
            stdout    = run_res.stdout
            stderr    = run_res.stderr
        except subprocess.TimeoutExpired:
            timeout_occurred = True
            exit_code = -1
            stdout = ""
            stderr = f"Execution timeout after {timeout_sec}s"

        t_run = time.time() - t_run_start
        t_wall = time.time() - t_wall_start

        if exit_code != 0 or timeout_occurred:
            return ExecutionResult(
                success=False,
                exit_code=exit_code,
                stdout=stdout, stderr=stderr,
                t_agent_run=t_run,
                wall_time_sec=t_wall,
                timeout_occurred=timeout_occurred,
                error_message=stderr[:1000] if stderr else "Runtime error",
            )

        # ── Step 5: convert output → solution.npz ────────────────────────
        solution_file = outdir / "solution.npz"
        meta_out_file = outdir / "meta.json"

        try:
            solution_file, meta_out_file = _convert_output(
                run_outdir, outdir, oracle_config, t_run
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                exit_code=0,
                stdout=stdout, stderr=stderr,
                t_agent_run=t_run,
                wall_time_sec=t_wall,
                error_message=f"Output conversion failed: {exc}",
            )

        return ExecutionResult(
            success=True,
            exit_code=0,
            stdout=stdout, stderr=stderr,
            t_agent_run=t_run,
            wall_time_sec=t_wall,
            solution_file=solution_file,
            meta_file=meta_out_file,
        )


# ---------------------------------------------------------------------------
# Output conversion helper
# ---------------------------------------------------------------------------

def _convert_output(
    run_outdir:    Path,
    outdir:        Path,
    oracle_config: Dict[str, Any],
    t_run:         float,
) -> tuple[Path, Path]:
    """
    Convert C++ binary output (solution_grid.bin + meta.json  OR
    solution.npz) to the canonical agent output files:
      outdir/solution.npz  (numpy, field "u" shape (ny, nx))
      outdir/meta.json     (includes wall_time_sec + solver_info)
    """
    grid_cfg = oracle_config.get("output", {}).get("grid", {})
    nx       = int(grid_cfg.get("nx", 50))
    ny       = int(grid_cfg.get("ny", 50))

    solution_npz  = outdir / "solution.npz"
    meta_out_file = outdir / "meta.json"

    # --- Case A: agent wrote solution.npz directly -------------------------
    if (run_outdir / "solution.npz").exists():
        shutil.copy(run_outdir / "solution.npz", solution_npz)
        if (run_outdir / "meta.json").exists():
            shutil.copy(run_outdir / "meta.json", meta_out_file)
        else:
            _write_default_meta(meta_out_file, t_run)
        return solution_npz, meta_out_file

    # --- Case B: agent wrote solution_grid.bin + meta.json -----------------
    bin_file  = run_outdir / "solution_grid.bin"
    meta_file = run_outdir / "meta.json"

    if not bin_file.exists():
        raise FileNotFoundError(
            "C++ solver wrote neither solution.npz nor solution_grid.bin"
        )

    raw  = np.fromfile(str(bin_file), dtype=np.float64)
    grid = raw.reshape(ny, nx)
    x    = np.linspace(*_bbox_x(oracle_config), nx)
    y    = np.linspace(*_bbox_y(oracle_config), ny)
    np.savez(str(solution_npz), x=x, y=y, u=grid)

    # Build meta.json from C++ meta + timing
    meta: Dict[str, Any] = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())

    solver_info = meta.pop("solver_info", {})
    # Ensure required keys exist
    if not solver_info:
        solver_info = {
            "mesh_resolution": oracle_config.get("mesh", {}).get("resolution", 0),
            "element_degree":  oracle_config.get("fem",  {}).get("degree",     1),
            "ksp_type":        "unknown",
            "pc_type":         "unknown",
            "rtol":            0.0,
        }
    out_meta = {
        "wall_time_sec": meta.get("baseline_time", meta.get("wall_time_sec", t_run)),
        "solver_info":   solver_info,
    }
    meta_out_file.write_text(json.dumps(out_meta, indent=2))

    return solution_npz, meta_out_file


def _write_default_meta(path: Path, t_run: float) -> None:
    path.write_text(json.dumps({
        "wall_time_sec": t_run,
        "solver_info": {
            "mesh_resolution": 0,
            "element_degree":  1,
            "ksp_type": "unknown",
            "pc_type":  "unknown",
            "rtol":     0.0,
        }
    }, indent=2))


def _bbox_x(oracle_config: Dict[str, Any]):
    bbox = oracle_config.get("output", {}).get("grid", {}).get("bbox", [0, 1, 0, 1])
    return bbox[0], bbox[1]


def _bbox_y(oracle_config: Dict[str, Any]):
    bbox = oracle_config.get("output", {}).get("grid", {}).get("bbox", [0, 1, 0, 1])
    return bbox[2], bbox[3]
