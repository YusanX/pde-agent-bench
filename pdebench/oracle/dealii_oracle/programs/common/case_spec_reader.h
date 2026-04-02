#pragma once
/**
 * case_spec_reader.h
 *
 * Parses the oracle_config JSON (written by DealIIOracleSolver.preprocess())
 * into a strongly-typed CaseSpec struct used by every deal.II solver program.
 *
 * Expected JSON layout (subset of benchmark.jsonl oracle_config):
 *
 *   {
 *     "pde": {
 *       "type": "poisson",
 *       "coefficients": { "kappa": {"type": "constant", "value": 1.0} },
 *       "manufactured_solution": { "u": "sin(pi*x)*sin(pi*y)" },
 *       // injected by Python preprocessing:
 *       "_computed_kappa":  "1.0",
 *       "_computed_source": "2*(pi)^(2)*sin(pi*x)*sin(pi*y)",
 *       "_computed_bc":     "sin(pi*x)*sin(pi*y)",
 *       "_has_exact":       true
 *     },
 *     "domain":  { "type": "unit_square" },
 *     "mesh":    { "resolution": 120, "cell_type": "triangle" },
 *     "fem":     { "family": "Lagrange", "degree": 1 },
 *     "output":  { "grid": { "bbox": [0,1,0,1], "nx": 50, "ny": 50 } },
 *     "oracle_solver": {
 *       "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10
 *     }
 *   }
 *
 * Note on cell_type: deal.II always uses subdivided quad meshes (FE_Q);
 * the cell_type field is parsed but not used to change element type.
 */

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Sub-structs
// ---------------------------------------------------------------------------

struct MeshSpec {
  int         resolution = 64;
  std::string cell_type  = "triangle";  // ignored – always quad mesh
};

struct DomainSpec {
  std::string type = "unit_square";
};

struct FemSpec {
  std::string family   = "Lagrange";
  int         degree   = 1;
  int         degree_u = 2;  // for mixed spaces (Stokes / NS)
  int         degree_p = 1;
};

struct OutputGridSpec {
  std::array<double, 4> bbox = {0.0, 1.0, 0.0, 1.0};
  int nx = 50;
  int ny = 50;
};

struct SolverSpec {
  std::string ksp_type  = "cg";
  std::string pc_type   = "hypre";
  double      rtol      = 1e-10;
  double      atol      = 1e-12;
};

// ---------------------------------------------------------------------------
// CaseSpec  – top-level container
// ---------------------------------------------------------------------------
struct CaseSpec {
  std::string    pde_type;      // "poisson", "heat", …
  MeshSpec       mesh;
  DomainSpec     domain;
  FemSpec        fem;
  OutputGridSpec output_grid;
  SolverSpec     oracle_solver;

  // Raw PDE node – PDE-specific fields read directly in each solver
  nlohmann::json pde;
  // Raw time node (heat / convection-diffusion / reaction-diffusion)
  nlohmann::json time_cfg;
  // Raw pde_params node (Stokes ν, NS ν, CD ε/β, Helmholtz k²)
  nlohmann::json pde_params;

  // Convenience helpers for injected expressions (empty string = not set)
  std::string computed_kappa()  const { return _get_str("_computed_kappa");  }
  std::string computed_source() const { return _get_str("_computed_source"); }
  std::string computed_bc()     const { return _get_str("_computed_bc");     }
  bool        has_exact()       const { return pde.value("_has_exact", false); }

 private:
  std::string _get_str(const std::string& key) const {
    if (pde.contains(key) && pde[key].is_string())
      return pde[key].get<std::string>();
    return "";
  }
};

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------
inline CaseSpec read_case_spec(const std::string& filepath) {
  std::ifstream f(filepath);
  if (!f.is_open())
    throw std::runtime_error("Cannot open case_spec file: " + filepath);

  nlohmann::json j;
  f >> j;

  CaseSpec spec;

  // ---- PDE ----------------------------------------------------------------
  spec.pde      = j.at("pde");
  spec.pde_type = spec.pde.at("type").get<std::string>();

  if (spec.pde.contains("time"))
    spec.time_cfg = spec.pde["time"];

  if (spec.pde.contains("pde_params"))
    spec.pde_params = spec.pde["pde_params"];

  // ---- Domain -------------------------------------------------------------
  if (j.contains("domain"))
    spec.domain.type = j["domain"].value("type", "unit_square");

  // ---- Mesh ---------------------------------------------------------------
  if (j.contains("mesh")) {
    spec.mesh.resolution = j["mesh"].value("resolution", 64);
    spec.mesh.cell_type  = j["mesh"].value("cell_type", "triangle");
  }

  // ---- FEM ----------------------------------------------------------------
  if (j.contains("fem")) {
    auto& fem = j["fem"];
    spec.fem.family   = fem.value("family",   "Lagrange");
    spec.fem.degree   = fem.value("degree",   1);
    spec.fem.degree_u = fem.value("degree_u", 2);
    spec.fem.degree_p = fem.value("degree_p", 1);
  }

  // ---- Output grid --------------------------------------------------------
  if (j.contains("output") && j["output"].contains("grid")) {
    auto& g = j["output"]["grid"];
    if (g.contains("bbox")) {
      auto b = g["bbox"].get<std::vector<double>>();
      spec.output_grid.bbox = {b[0], b[1], b[2], b[3]};
    }
    spec.output_grid.nx = g.value("nx", 50);
    spec.output_grid.ny = g.value("ny", 50);
  }

  // ---- Oracle solver -------------------------------------------------------
  if (j.contains("oracle_solver")) {
    auto& s = j["oracle_solver"];
    spec.oracle_solver.ksp_type = s.value("ksp_type", "cg");
    spec.oracle_solver.pc_type  = s.value("pc_type",  "hypre");
    spec.oracle_solver.rtol     = s.value("rtol",     1e-10);
    spec.oracle_solver.atol     = s.value("atol",     1e-12);
  }

  return spec;
}
