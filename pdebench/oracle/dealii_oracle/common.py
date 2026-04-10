"""
dealii_oracle/common.py
=======================

Python utilities shared by the deal.II oracle backend:

1. Expression preprocessing (sympy → muParser strings)
   - `preprocess_case_spec(oracle_config)` injects _computed_* fields
     that the C++ programs consume via nlohmann/json.

2. Build management
   - `ensure_built(programs_dir, build_dir)` runs cmake+make on first call
     and caches the build dir for subsequent calls.

3. Subprocess runner
   - `run_program(binary_path, case_spec_json, outdir, timeout)` calls
     the compiled C++ oracle binary and returns the output directory.

4. Result parsing
   - `parse_output(outdir)` reads solution_grid.bin + meta.json and
     returns a numpy array + metadata dict.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ============================================================================
# 1.  muParser expression helpers
# ============================================================================

def _sympy_to_muparser(sym_expr) -> str:
    """
    Convert a sympy expression to a muParser-compatible string.

    Key differences vs Python/sympy:
      **  →  ^          (power operator)
      pi  →  pi         (muParser built-in constant)
    """
    import sympy as sp
    from sympy.printing.str import StrPrinter

    class _MuPrinter(StrPrinter):
        def _print_Pow(self, expr):
            base = self._print(expr.base)
            exp_ = self._print(expr.exp)
            if expr.exp == sp.S.NegativeOne:
                return f"(1.0/({base}))"
            if expr.exp == sp.S.Half:
                return f"sqrt({base})"
            return f"({base})^({exp_})"

        def _print_Pi(self, _):
            return "pi"

        def _print_Exp1(self, _):
            # Euler's number e
            return "exp(1)"

        def _print_Float(self, expr):
            return f"{float(expr):.17g}"

        def _print_Rational(self, expr):
            return f"({int(expr.p)}.0/{int(expr.q)}.0)"

        def _print_NegativeOne(self, _):
            return "(-1)"

        def _print_Half(self, _):
            return "0.5"

    return _MuPrinter().doprint(sym_expr)


def _parse_sym(expr_str: str, extra_locals: Optional[dict] = None):
    """Parse a sympy expression string, recognising x, y, z, t, pi."""
    import sympy as sp
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    local_dict = {"x": sx, "y": sy, "z": sz, "t": st, "pi": sp.pi}
    if extra_locals:
        local_dict.update(extra_locals)
    return sp.sympify(str(expr_str), locals=local_dict)


def _expr_to_mu(expr_str: str) -> str:
    """Parse a Python/sympy string and return muParser-compatible form."""
    return _sympy_to_muparser(_parse_sym(str(expr_str)))


# ============================================================================
# 2.  PDE-specific preprocessing functions
# ============================================================================

def _preprocess_poisson(pde: dict, bc: dict, dim: int = 2) -> None:
    import sympy as sp
    sx, sy, sz = sp.symbols("x y z", real=True)
    coords = (sx, sy) if dim == 2 else (sx, sy, sz)

    kappa_spec = pde.get("coefficients", {}).get(
        "kappa", {"type": "constant", "value": 1.0}
    )
    if kappa_spec["type"] == "constant":
        kappa_sym = sp.Float(kappa_spec["value"])
        pde["_computed_kappa"] = str(float(kappa_spec["value"]))
    else:
        kappa_sym = _parse_sym(kappa_spec["expr"])
        pde["_computed_kappa"] = _sympy_to_muparser(kappa_sym)

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured:
        u_sym = _parse_sym(manufactured["u"])
        f_sym = -sum(sp.diff(kappa_sym * sp.diff(u_sym, c), c) for c in coords)
        pde["_computed_source"] = _sympy_to_muparser(sp.expand(f_sym))
        pde["_computed_bc"]     = _sympy_to_muparser(u_sym)
        pde["_has_exact"]       = True
    else:
        src = pde.get("source_term", "0.0")
        pde["_computed_source"] = _expr_to_mu(src)
        bc_val = bc.get("value", "0.0")
        pde["_computed_bc"]     = "0.0" if bc_val == "u" else _expr_to_mu(bc_val)
        pde["_has_exact"]       = False


def _preprocess_heat(pde: dict, bc: dict, dim: int = 2) -> None:
    """
    Heat equation:  ∂u/∂t - κ Δu = f
    For the oracle we only need the *final* time snapshot (u at t_end).
    The C++ solver reads _computed_kappa, _computed_source (may contain t),
    _computed_bc (may contain t), _computed_ic (initial condition at t=0).
    """
    import sympy as sp
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    coords = (sx, sy) if dim == 2 else (sx, sy, sz)

    kappa_spec = pde.get("coefficients", {}).get(
        "kappa", {"type": "constant", "value": 1.0}
    )
    if kappa_spec["type"] == "constant":
        kappa_sym = sp.Float(kappa_spec["value"])
        pde["_computed_kappa"] = str(float(kappa_spec["value"]))
    else:
        kappa_sym = _parse_sym(kappa_spec["expr"])
        pde["_computed_kappa"] = _sympy_to_muparser(kappa_sym)

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured:
        u_sym = _parse_sym(manufactured["u"])

        # Source term from PDE: f = ∂u/∂t - κ Δu
        du_dt = sp.diff(u_sym, st)
        laplacian = sum(sp.diff(kappa_sym * sp.diff(u_sym, c), c) for c in coords)
        f_sym = sp.expand(du_dt - laplacian)

        pde["_computed_source"] = _sympy_to_muparser(f_sym)
        pde["_computed_bc"]     = _sympy_to_muparser(u_sym)

        # Initial condition: u(x,y,0)
        ic_sym = u_sym.subs(st, sp.Integer(0))
        pde["_computed_ic"] = _sympy_to_muparser(sp.simplify(ic_sym))
        pde["_has_exact"]   = True
    else:
        src = pde.get("source_term", "0.0")
        pde["_computed_source"] = _expr_to_mu(src)

        ic_str = pde.get("initial_condition", "0.0")
        pde["_computed_ic"] = _expr_to_mu(ic_str)

        bc_val = bc.get("value", "0.0")
        pde["_computed_bc"] = "0.0" if bc_val == "u" else _expr_to_mu(bc_val)
        pde["_has_exact"]   = False


def _preprocess_convection_diffusion(pde: dict, bc: dict, dim: int = 2) -> None:
    """
    ε Δu + β·∇u = f   (steady)  or  ∂u/∂t + β·∇u - ε Δu = f  (transient).
    Source term injected as _computed_source; β components as _computed_beta_x/y.
    """
    import sympy as sp
    sx, sy, sz, st = sp.symbols("x y z t", real=True)
    coords = (sx, sy) if dim == 2 else (sx, sy, sz)

    params = pde.get("pde_params", {})
    epsilon = float(params.get("epsilon", 0.01))
    beta    = params.get("beta", [1.0, 1.0])
    if isinstance(beta, list):
        beta_vals = [float(v) for v in beta]
    else:
        beta_vals = [float(beta)] * dim
    while len(beta_vals) < dim:
        beta_vals.append(0.0)
    bx = beta_vals[0]
    by = beta_vals[1] if dim >= 2 else 0.0
    bz = beta_vals[2] if dim >= 3 else 0.0

    pde["_computed_epsilon"] = str(epsilon)
    pde["_computed_beta_x"]  = str(bx)
    pde["_computed_beta_y"]  = str(by)
    if dim >= 3:
        pde["_computed_beta_z"]  = str(bz)

    is_transient = "time" in pde

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured:
        u_sym = _parse_sym(manufactured["u"])
        laplacian = sum(sp.diff(u_sym, c, 2) for c in coords)
        grad_u = sum(beta_vals[i] * sp.diff(u_sym, coords[i]) for i in range(dim))

        if is_transient:
            f_sym = sp.diff(u_sym, st) + grad_u - epsilon * laplacian
        else:
            f_sym = grad_u - epsilon * laplacian

        pde["_computed_source"] = _sympy_to_muparser(sp.expand(f_sym))
        pde["_computed_bc"]     = _sympy_to_muparser(u_sym)

        if is_transient:
            ic_sym = u_sym.subs(st, sp.Integer(0))
            pde["_computed_ic"] = _sympy_to_muparser(sp.simplify(ic_sym))

        pde["_has_exact"] = True
    else:
        src = pde.get("source_term", "0.0")
        pde["_computed_source"] = _expr_to_mu(src)
        bc_val = bc.get("value", "0.0")
        pde["_computed_bc"]     = "0.0" if bc_val == "u" else _expr_to_mu(bc_val)
        if is_transient:
            ic_str = pde.get("initial_condition", "0.0")
            pde["_computed_ic"] = _expr_to_mu(ic_str)
        pde["_has_exact"] = False


def _preprocess_helmholtz(pde: dict, bc: dict, dim: int = 2) -> None:
    """
    Δu + k² u = f   with Dirichlet BC.
    """
    import sympy as sp
    sx, sy, sz = sp.symbols("x y z", real=True)
    coords = (sx, sy) if dim == 2 else (sx, sy, sz)

    params = pde.get("pde_params", {})
    if "k2" in params:
        k2 = float(params["k2"])
    else:
        k = float(params.get("k", params.get("wave_number", 1.0)))
        k2 = k * k
    pde["_computed_k2"] = str(k2)

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured:
        u_sym = _parse_sym(manufactured["u"])
        f_sym = -sum(sp.diff(u_sym, c, 2) for c in coords) - k2 * u_sym
        pde["_computed_source"] = _sympy_to_muparser(sp.expand(f_sym))
        pde["_computed_bc"]     = _sympy_to_muparser(u_sym)
        pde["_has_exact"]       = True
    else:
        src = pde.get("source_term", "0.0")
        pde["_computed_source"] = _expr_to_mu(src)
        bc_val = bc.get("value", "0.0")
        pde["_computed_bc"]     = "0.0" if bc_val == "u" else _expr_to_mu(bc_val)
        pde["_has_exact"]       = False


def _preprocess_biharmonic(pde: dict, bc: dict) -> None:
    """Δ²u = f  with simply-supported or clamped BC (u=0, Δu=0 on ∂Ω)."""
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured:
        u_sym = _parse_sym(manufactured["u"])
        lap = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
        f_sym = sp.diff(lap, sx, 2) + sp.diff(lap, sy, 2)
        pde["_computed_source"]  = _sympy_to_muparser(sp.expand(f_sym))
        pde["_computed_bc"]      = _sympy_to_muparser(u_sym)
        # Laplacian of exact solution as second BC (Δu=0 for biharmonic)
        pde["_computed_bc_laplacian"] = _sympy_to_muparser(sp.expand(lap))
        pde["_has_exact"]        = True
    else:
        src = pde.get("source_term", "0.0")
        pde["_computed_source"]  = _expr_to_mu(src)
        pde["_computed_bc"]      = "0.0"
        pde["_computed_bc_laplacian"] = "0.0"
        pde["_has_exact"]        = False


def _preprocess_linear_elasticity(pde: dict, bc: dict, dim: int = 2) -> None:
    """
    -∇·σ(u) = f, σ = λ(∇·u)I + μ(∇u + ∇uᵀ)
    Manufactured solution: u = [ux, uy] or [ux, uy, uz].
    """
    import sympy as sp
    sx, sy, sz = sp.symbols("x y z", real=True)
    coords = (sx, sy) if dim == 2 else (sx, sy, sz)

    params = pde.get("pde_params", {})
    if "lambda" in params and "mu" in params:
        lam = float(params["lambda"])
        mu  = float(params["mu"])
    else:
        E = float(params.get("E", 1.0))
        nu = float(params.get("nu", 0.3))
        # Plane strain Lamé parameters, matching the Python oracle.
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    pde["_computed_lambda"] = str(lam)
    pde["_computed_mu"]     = str(mu)

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured and isinstance(manufactured["u"], list):
        u_syms = [_parse_sym(comp) for comp in manufactured["u"]]
        if len(u_syms) != dim:
            raise ValueError(
                f"linear_elasticity manufactured solution expects {dim} components, "
                f"got {len(u_syms)}"
            )

        div_u = sum(sp.diff(u_syms[i], coords[i]) for i in range(dim))
        f_syms = []
        for i in range(dim):
            lap_u_i = sum(sp.diff(u_syms[i], c, 2) for c in coords)
            grad_div_i = sp.diff(div_u, coords[i])
            f_syms.append(sp.expand(-(mu * lap_u_i + (lam + mu) * grad_div_i)))

        pde["_computed_source_x"] = _sympy_to_muparser(f_syms[0])
        pde["_computed_source_y"] = _sympy_to_muparser(f_syms[1])
        pde["_computed_bc_x"]     = _sympy_to_muparser(u_syms[0])
        pde["_computed_bc_y"]     = _sympy_to_muparser(u_syms[1])
        if dim >= 3:
            pde["_computed_source_z"] = _sympy_to_muparser(f_syms[2])
            pde["_computed_bc_z"]     = _sympy_to_muparser(u_syms[2])
        pde["_has_exact"]         = True
    else:
        src = pde.get("source_term", ["0.0"] * dim)
        if isinstance(src, (list, tuple)) and len(src) >= dim:
            pde["_computed_source_x"] = _expr_to_mu(src[0])
            pde["_computed_source_y"] = _expr_to_mu(src[1])
            if dim >= 3:
                pde["_computed_source_z"] = _expr_to_mu(src[2])
        else:
            pde["_computed_source_x"] = "0.0"
            pde["_computed_source_y"] = "0.0"
            if dim >= 3:
                pde["_computed_source_z"] = "0.0"

        # For no-exact cases, C++ reads the raw bc.dirichlet JSON directly to
        # support boundary subsets such as x0/x1/y0/y1 and lists of constraints.
        pde["_computed_bc_x"]     = "0.0"
        pde["_computed_bc_y"]     = "0.0"
        if dim >= 3:
            pde["_computed_bc_z"] = "0.0"
        pde["_has_exact"]         = False


def _preprocess_reaction_diffusion(pde: dict, bc: dict) -> None:
    """
    Steady:    -Δu + σ u = f
    Transient: ∂u/∂t - Δu + σ u = f
    """
    import sympy as sp
    sx, sy, st = sp.symbols("x y t", real=True)

    params = pde.get("pde_params", {})
    sigma = float(params.get("sigma", 1.0))
    pde["_computed_sigma"] = str(sigma)

    is_transient = "time" in pde

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured:
        u_sym = _parse_sym(manufactured["u"])
        laplacian = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)

        if is_transient:
            du_dt = sp.diff(u_sym, st)
            f_sym = du_dt - laplacian + sigma * u_sym
            pde["_computed_source"] = _sympy_to_muparser(sp.expand(f_sym))
            pde["_computed_bc"]     = _sympy_to_muparser(u_sym)
            ic_sym = u_sym.subs(st, sp.Integer(0))
            pde["_computed_ic"]     = _sympy_to_muparser(sp.simplify(ic_sym))
        else:
            # Steady: if manufactured solution accidentally contains t, evaluate at t=0
            if u_sym.has(st):
                u_sym    = u_sym.subs(st, sp.Integer(0))
                laplacian = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)
            f_sym = -laplacian + sigma * u_sym
            pde["_computed_source"] = _sympy_to_muparser(sp.expand(f_sym))
            pde["_computed_bc"]     = _sympy_to_muparser(u_sym)

        pde["_has_exact"] = True
    else:
        src = pde.get("source_term", "0.0")
        pde["_computed_source"] = _expr_to_mu(src)
        bc_val = bc.get("value", "0.0")
        pde["_computed_bc"]     = "0.0" if bc_val == "u" else _expr_to_mu(bc_val)
        if is_transient:
            ic_str = pde.get("initial_condition", "0.0")
            pde["_computed_ic"] = _expr_to_mu(ic_str)
        pde["_has_exact"] = False


def _preprocess_stokes(pde: dict, bc: dict, dim: int = 2) -> None:
    """
    -ν Δu + ∇p = f,  ∇·u = 0
    Manufactured: u = [ux, uy] or [ux, uy, uz],  p = p_exact.
    """
    import sympy as sp
    sx, sy, sz = sp.symbols("x y z", real=True)
    coords = (sx, sy) if dim == 2 else (sx, sy, sz)

    params = pde.get("pde_params", {})
    nu = float(params.get("nu", 1.0))
    pde["_computed_nu"] = str(nu)

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured and "p" in manufactured and isinstance(manufactured["u"], list):
        u_syms = [_parse_sym(comp) for comp in manufactured["u"]]
        if len(u_syms) != dim:
            raise ValueError(
                f"stokes manufactured solution expects {dim} velocity components, "
                f"got {len(u_syms)}"
            )
        p_sym  = _parse_sym(manufactured["p"])

        f_syms = []
        for i in range(dim):
            lap_ui = sum(sp.diff(u_syms[i], c, 2) for c in coords)
            dp_i = sp.diff(p_sym, coords[i])
            f_syms.append(sp.expand(-nu * lap_ui + dp_i))

        pde["_computed_source_x"] = _sympy_to_muparser(f_syms[0])
        pde["_computed_source_y"] = _sympy_to_muparser(f_syms[1])
        pde["_computed_bc_x"]     = _sympy_to_muparser(u_syms[0])
        pde["_computed_bc_y"]     = _sympy_to_muparser(u_syms[1])
        if dim >= 3:
            pde["_computed_source_z"] = _sympy_to_muparser(f_syms[2])
            pde["_computed_bc_z"]     = _sympy_to_muparser(u_syms[2])
        pde["_computed_p_exact"]  = _sympy_to_muparser(p_sym)
        pde["_has_exact"]         = True
    else:
        src = pde.get("source_term", ["0.0"] * dim)
        if isinstance(src, (list, tuple)) and len(src) >= dim:
            pde["_computed_source_x"] = _expr_to_mu(src[0])
            pde["_computed_source_y"] = _expr_to_mu(src[1])
            if dim >= 3:
                pde["_computed_source_z"] = _expr_to_mu(src[2])
        else:
            pde["_computed_source_x"] = "0.0"
            pde["_computed_source_y"] = "0.0"
            if dim >= 3:
                pde["_computed_source_z"] = "0.0"

        # For no-exact cases, C++ reads the raw bc.dirichlet JSON directly to
        # support boundary subsets such as x0/x1/y0/y1 and mixed inflow/outflow.
        pde["_computed_bc_x"]     = "0.0"
        pde["_computed_bc_y"]     = "0.0"
        if dim >= 3:
            pde["_computed_bc_z"] = "0.0"
        pde["_has_exact"]         = False


def _preprocess_navier_stokes(pde: dict, bc: dict) -> None:
    """
    (u·∇)u - ν Δu + ∇p = f,  ∇·u = 0  (steady incompressible NS).

    Generates:
      _computed_source_x/y  – body force consistent with ν∇u:∇v weak form
      _computed_bc_x/y      – velocity BC (manufactured or "0.0")
      _bc_segments          – JSON list [{id, ex, ey}] for per-wall BCs
                              Boundary ids: 0=x0,1=x1,2=y0,3=y1
    """
    import json as _json
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)

    params = pde.get("pde_params", {})
    nu = float(params.get("nu", params.get("viscosity", 0.01)))
    pde["_computed_nu"] = str(nu)

    # Map boundary name → boundary id (matches make_mesh() tagging)
    WALL_IDS = {
        "x0": 0, "xmin": 0,
        "x1": 1, "xmax": 1,
        "y0": 2, "ymin": 2,
        "y1": 3, "ymax": 3,
    }

    manufactured = pde.get("manufactured_solution", {})
    if "u" in manufactured and "p" in manufactured and isinstance(manufactured["u"], list):
        ux_sym = _parse_sym(manufactured["u"][0])
        uy_sym = _parse_sym(manufactured["u"][1])
        p_sym  = _parse_sym(manufactured["p"])

        lap_ux = sp.diff(ux_sym, sx, 2) + sp.diff(ux_sym, sy, 2)
        lap_uy = sp.diff(uy_sym, sx, 2) + sp.diff(uy_sym, sy, 2)
        dp_dx  = sp.diff(p_sym, sx)
        dp_dy  = sp.diff(p_sym, sy)

        # Convection (u·∇)u; source consistent with -νΔu (= ν∇u:∇v weak form)
        conv_x = ux_sym * sp.diff(ux_sym, sx) + uy_sym * sp.diff(ux_sym, sy)
        conv_y = ux_sym * sp.diff(uy_sym, sx) + uy_sym * sp.diff(uy_sym, sy)

        fx_sym = conv_x - nu * lap_ux + dp_dx
        fy_sym = conv_y - nu * lap_uy + dp_dy

        pde["_computed_source_x"] = _sympy_to_muparser(sp.expand(fx_sym))
        pde["_computed_source_y"] = _sympy_to_muparser(sp.expand(fy_sym))
        pde["_computed_bc_x"]     = _sympy_to_muparser(ux_sym)
        pde["_computed_bc_y"]     = _sympy_to_muparser(uy_sym)
        pde["_computed_p_exact"]  = _sympy_to_muparser(p_sym)
        pde["_has_exact"]         = True

        # Manufactured: apply exact solution on all four walls
        ex_str = pde["_computed_bc_x"]
        ey_str = pde["_computed_bc_y"]
        segments = [{"id": bid, "ex": ex_str, "ey": ey_str} for bid in [0, 1, 2, 3]]

    else:
        # No-exact: forward source_term from case spec
        src = pde.get("source_term")
        if isinstance(src, list) and len(src) >= 2:
            pde["_computed_source_x"] = _expr_to_mu(str(src[0]))
            pde["_computed_source_y"] = _expr_to_mu(str(src[1]))
        else:
            pde["_computed_source_x"] = "0.0"
            pde["_computed_source_y"] = "0.0"
        pde["_computed_bc_x"] = "0.0"
        pde["_computed_bc_y"] = "0.0"
        pde["_has_exact"]     = False

        # Build per-wall BC segments from bc.dirichlet list
        dirichlet = bc if isinstance(bc, list) else ([bc] if isinstance(bc, dict) else [])
        segments: list = []
        for seg in dirichlet:
            on  = seg.get("on", "all").lower()
            val = seg.get("value", ["0.0", "0.0"])
            if isinstance(val, list) and len(val) >= 2:
                vx, vy = _expr_to_mu(str(val[0])), _expr_to_mu(str(val[1]))
            else:
                vx = vy = _expr_to_mu(str(val))
            if on in ("all", "*"):
                for bid in [0, 1, 2, 3]:
                    segments.append({"id": bid, "ex": vx, "ey": vy})
            elif on in WALL_IDS:
                segments.append({"id": WALL_IDS[on], "ex": vx, "ey": vy})
        # If no BCs specified, default to zero on all walls
        if not segments:
            for bid in [0, 1, 2, 3]:
                segments.append({"id": bid, "ex": "0.0", "ey": "0.0"})

    pde["_bc_segments"] = _json.dumps(segments)


# ============================================================================
# 3.  Top-level preprocessor
# ============================================================================

_PREPROCESSORS = {
    "poisson":              _preprocess_poisson,
    "heat":                 _preprocess_heat,
    "convection_diffusion": _preprocess_convection_diffusion,
    "helmholtz":            _preprocess_helmholtz,
    "biharmonic":           _preprocess_biharmonic,
    "linear_elasticity":    _preprocess_linear_elasticity,
    "reaction_diffusion":   _preprocess_reaction_diffusion,
    "stokes":               _preprocess_stokes,
    "navier_stokes":        _preprocess_navier_stokes,
}


def preprocess_case_spec(oracle_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-copy the oracle_config dict and inject _computed_* fields into
    pde section so C++ programs can parse expressions via muParser / FunctionParser.

    Returns the enriched copy (original is never mutated).
    """
    spec = copy.deepcopy(oracle_config)
    pde  = spec["pde"]
    bc   = spec.get("bc", {}).get("dirichlet", {})
    grid = spec.get("output", {}).get("grid", {})
    dim = 3 if len(grid.get("bbox", [])) == 6 or spec.get("domain", {}).get("type") == "unit_cube" else 2

    pde_type = pde["type"]
    processor = _PREPROCESSORS.get(pde_type)
    if processor is None:
        raise ValueError(
            f"DealII oracle: no preprocessor for PDE type '{pde_type}'. "
            f"Supported: {list(_PREPROCESSORS)}"
        )
    if pde_type in ("poisson", "heat", "convection_diffusion", "helmholtz", "linear_elasticity", "stokes"):
        processor(pde, bc, dim)
    else:
        processor(pde, bc)
    return spec


# ============================================================================
# 4.  Build management
# ============================================================================

_BUILD_LOCK: Dict[str, bool] = {}  # programs_dir:pde_type → built flag


def ensure_built(programs_dir: Path, build_dir: Path, pde_type: str) -> None:
    """
    Configure and compile the deal.II oracle program needed for one PDE type.
    Skips recompilation if the requested binary already exists.

    Raises RuntimeError if cmake or make fails.
    """
    binary_name = _PDE_TO_BINARY.get(pde_type)
    if binary_name is None:
        raise ValueError(f"No deal.II binary configured for PDE type: {pde_type}")

    key = f"{programs_dir}:{pde_type}"
    if _BUILD_LOCK.get(key):
        return

    # Only require the binary needed for the current PDE.
    if (build_dir / binary_name).exists():
        _BUILD_LOCK[key] = True
        return

    build_dir.mkdir(parents=True, exist_ok=True)

    # ── cmake configure ────────────────────────────────────────────────────
    deal_ii_dir = os.environ.get("DEAL_II_DIR", "")
    cmake_cmd = [
        "cmake",
        str(programs_dir),
        f"-DCMAKE_BUILD_TYPE=Release",
    ]
    if deal_ii_dir:
        cmake_cmd.append(f"-DDEAL_II_DIR={deal_ii_dir}")

    print(f"[dealii_oracle] cmake configure: {' '.join(cmake_cmd)}", flush=True)
    result = subprocess.run(
        cmake_cmd, cwd=str(build_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"deal.II oracle cmake configure failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # ── make target ───────────────────────────────────────────────────────
    n_jobs = max(1, os.cpu_count() or 4)
    make_cmd = ["make", f"-j{n_jobs}", binary_name]
    print(f"[dealii_oracle] make -j{n_jobs} {binary_name} …", flush=True)
    result = subprocess.run(
        make_cmd, cwd=str(build_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"deal.II oracle make failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    _BUILD_LOCK[key] = True
    print("[dealii_oracle] Build complete.", flush=True)


# ============================================================================
# 5.  Subprocess runner
# ============================================================================

_PDE_TO_BINARY: Dict[str, str] = {
    "poisson":              "poisson_solver",
    "heat":                 "heat_solver",
    "convection_diffusion": "convection_diffusion_solver",
    "helmholtz":            "helmholtz_solver",
    "biharmonic":           "biharmonic_solver",
    "linear_elasticity":    "linear_elasticity_solver",
    "reaction_diffusion":   "reaction_diffusion_solver",
    "stokes":               "stokes_solver",
    "navier_stokes":        "navier_stokes_solver",
}


def run_oracle_program(
    pde_type:    str,
    case_spec:   Dict[str, Any],
    build_dir:   Path,
    timeout_sec: int = 900,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Write case_spec to a temp JSON, invoke the compiled C++ binary,
    read solution_grid.bin + meta.json, and return (grid_array, meta).
    """
    binary_name = _PDE_TO_BINARY.get(pde_type)
    if binary_name is None:
        raise ValueError(f"No deal.II binary for PDE type: {pde_type}")

    binary_path = build_dir / binary_name
    if not binary_path.exists():
        raise FileNotFoundError(
            f"deal.II oracle binary not found: {binary_path}\n"
            "Run ensure_built() first."
        )

    with tempfile.TemporaryDirectory(prefix="dealii_oracle_") as tmpdir:
        spec_file   = Path(tmpdir) / "case_spec.json"
        output_dir  = Path(tmpdir) / "output"
        output_dir.mkdir()

        spec_file.write_text(json.dumps(case_spec))

        result = subprocess.run(
            [str(binary_path), str(spec_file), str(output_dir)],
            capture_output=True, text=True,
            timeout=timeout_sec,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"deal.II oracle binary '{binary_name}' failed "
                f"(exit {result.returncode}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        return parse_output(output_dir)


# ============================================================================
# 6.  Output parsing
# ============================================================================

def parse_output(outdir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read solution_grid.bin and meta.json from the C++ output directory.

    Returns:
        grid  – np.ndarray of shape (ny, nx) or (nz, ny, nx), float64
        meta  – dict with nx, ny, [nz], num_dofs, baseline_time, ksp_type, …
    """
    bin_file  = outdir / "solution_grid.bin"
    meta_file = outdir / "meta.json"

    if not bin_file.exists():
        raise FileNotFoundError(f"solution_grid.bin not found in {outdir}")
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found in {outdir}")

    meta = json.loads(meta_file.read_text())
    nx   = int(meta["nx"])
    ny   = int(meta["ny"])
    nz   = int(meta["nz"]) if "nz" in meta else None

    raw  = np.fromfile(str(bin_file), dtype=np.float64)
    expected_size = nx * ny if nz is None else nx * ny * nz
    if raw.size != expected_size:
        raise ValueError(
            f"Binary size mismatch: expected {expected_size} values, got {raw.size}"
        )

    grid = raw.reshape(ny, nx) if nz is None else raw.reshape(nz, ny, nx)
    return grid, meta
