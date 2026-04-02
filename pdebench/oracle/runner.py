"""
pdebench/oracle/runner.py
=========================

容器内 oracle 入口脚本，由 docker_bridge.py 在 Docker 容器中调用：

  python -m pdebench.oracle.runner <case_spec.json> <output_dir> <library>

输出：
  <output_dir>/reference.npy  – numpy 数组（参考解网格）
  <output_dir>/meta.json      – baseline_error / baseline_time / solver_info / num_dofs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python -m pdebench.oracle.runner "
            "<case_spec.json> <output_dir> <library>",
            file=sys.stderr,
        )
        sys.exit(1)

    case_spec_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    library = sys.argv[3]

    case_spec = json.loads(case_spec_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    if library == "dealii":
        from pdebench.oracle.dealii_oracle import DealIIOracleSolver
        result = DealIIOracleSolver().solve(case_spec)
    elif library == "firedrake":
        from pdebench.oracle.firedrake_oracle import FiredrakeOracleSolver
        result = FiredrakeOracleSolver().solve(case_spec)
    else:
        print(f"Unknown library: {library!r}. Expected 'dealii' or 'firedrake'.", file=sys.stderr)
        sys.exit(1)

    np.save(output_dir / "reference.npy", result.reference)
    meta = {
        "baseline_error": result.baseline_error,
        "baseline_time":  result.baseline_time,
        "solver_info":    result.solver_info,
        "num_dofs":       result.num_dofs,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
