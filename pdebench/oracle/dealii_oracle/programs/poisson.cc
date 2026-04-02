/**
 * poisson.cc  –  deal.II oracle for the Poisson equation
 *
 *   -∇·(κ ∇u) = f   in Ω = [0,1]²
 *           u = g   on ∂Ω
 *
 * Usage:
 *   ./poisson_solver <case_spec.json> <output_dir>
 *
 * The case_spec.json must contain _computed_kappa, _computed_source and
 * _computed_bc fields (injected by the Python DealIIOracleSolver.preprocess()).
 * All expression strings use muParser syntax (^ for power, pi as constant).
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

// deal.II headers
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

// Project headers
#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

static const std::map<std::string, double> MU_CONSTANTS = {{"pi", M_PI}};

// Build a FunctionParser from a pre-computed muParser expression string.
// Returns a heap-allocated pointer – caller manages lifetime via unique_ptr.
std::unique_ptr<FunctionParser<2>>
make_func(const std::string& expr, bool time_dep = false) {
  auto fp = std::make_unique<FunctionParser<2>>(1 /*n_components*/);
  fp->initialize("x,y", expr, MU_CONSTANTS, time_dep);
  return fp;
}

}  // namespace

// ---------------------------------------------------------------------------
// PoissonOracle class
// ---------------------------------------------------------------------------
class PoissonOracle {
 public:
  explicit PoissonOracle(const CaseSpec& spec)
      : spec_(spec), fe_(spec.fem.degree), dof_handler_(tria_) {}

  void run(const std::string& outdir) {
    // Create output directory if needed
    std::filesystem::create_directories(outdir);

    Timer timer;
    timer.start();

    make_mesh();
    setup_system();
    assemble_system();
    solve_system();

    timer.stop();
    const double elapsed = timer.wall_time();

    oracle_util::write_scalar_grid(
        dof_handler_, solution_,
        spec_.output_grid.bbox,
        spec_.output_grid.nx, spec_.output_grid.ny,
        outdir, elapsed,
        spec_.oracle_solver.ksp_type,
        spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<2>           tria_;
  FE_Q<2>                    fe_;
  DoFHandler<2>              dof_handler_;
  AffineConstraints<double>  constraints_;
  SparsityPattern            sparsity_pattern_;
  SparseMatrix<double>       system_matrix_;
  Vector<double>             solution_;
  Vector<double>             system_rhs_;

  // ------------------------------------------------------------------
  // Mesh:  subdivided_hyper_cube gives exactly resolution² quad cells.
  // Boundary ID 0 covers all four sides (default from GridGenerator).
  // ------------------------------------------------------------------
  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(
        tria_, spec_.mesh.resolution, 0.0, 1.0);
  }

  // ------------------------------------------------------------------
  // Setup:  distribute DOFs, enforce Dirichlet BC via constraints.
  // ------------------------------------------------------------------
  void setup_system() {
    dof_handler_.distribute_dofs(fe_);
    constraints_.clear();

    // Dirichlet BC on all boundaries (boundary_id = 0)
    const std::string bc_expr = spec_.computed_bc();
    if (bc_expr.empty())
      throw std::runtime_error("Poisson oracle: _computed_bc missing in case_spec");

    auto bc_func = make_func(bc_expr);
    VectorTools::interpolate_boundary_values(
        dof_handler_, 0, *bc_func, constraints_);
    constraints_.close();

    DynamicSparsityPattern dsp(dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_, dsp, constraints_);
    sparsity_pattern_.copy_from(dsp);

    system_matrix_.reinit(sparsity_pattern_);
    solution_.reinit(dof_handler_.n_dofs());
    system_rhs_.reinit(dof_handler_.n_dofs());
  }

  // ------------------------------------------------------------------
  // Assembly:  cell-by-cell integration of  κ ∇u·∇v = f v
  // ------------------------------------------------------------------
  void assemble_system() {
    const std::string kappa_expr  = spec_.computed_kappa();
    const std::string source_expr = spec_.computed_source();
    if (kappa_expr.empty() || source_expr.empty())
      throw std::runtime_error("Poisson oracle: _computed_kappa or _computed_source missing");

    auto kappa_func  = make_func(kappa_expr);
    auto source_func = make_func(source_expr);

    const QGauss<2>  quadrature(fe_.degree + 1);
    FEValues<2> fe_values(fe_, quadrature,
                          update_values | update_gradients |
                          update_JxW_values | update_quadrature_points);

    const unsigned int n_dpc  = fe_.n_dofs_per_cell();
    const unsigned int n_q    = quadrature.size();

    FullMatrix<double> cell_matrix(n_dpc, n_dpc);
    Vector<double>     cell_rhs(n_dpc);
    std::vector<types::global_dof_index> local_dof_indices(n_dpc);

    for (auto& cell : dof_handler_.active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < n_q; ++q) {
        const Point<2>& qp    = fe_values.quadrature_point(q);
        const double    kappa = kappa_func->value(qp);
        const double    f_val = source_func->value(qp);
        const double    JxW   = fe_values.JxW(q);

        for (unsigned int i = 0; i < n_dpc; ++i) {
          for (unsigned int j = 0; j < n_dpc; ++j)
            cell_matrix(i, j) +=
                kappa * fe_values.shape_grad(i, q) *
                fe_values.shape_grad(j, q) * JxW;
          cell_rhs(i) += f_val * fe_values.shape_value(i, q) * JxW;
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints_.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices,
          system_matrix_, system_rhs_);
    }
  }

  // ------------------------------------------------------------------
  // Linear solve:  choose solver based on ksp_type / pc_type.
  //
  // PETSc ksp/pc names → deal.II native equivalents:
  //   cg   + hypre/ilu/jacobi → SolverCG  + SSOR (robust, no PETSc needed)
  //   gmres + *              → SolverGMRES + SSOR
  //   preonly + lu/mumps     → SparseDirectUMFPACK (direct)
  // ------------------------------------------------------------------
  void solve_system() {
    const std::string ksp = spec_.oracle_solver.ksp_type;
    const std::string pc  = spec_.oracle_solver.pc_type;
    const double      rtol = spec_.oracle_solver.rtol;
    const double      atol = spec_.oracle_solver.atol;

    // Direct solver path
    if (ksp == "preonly" ||
        (ksp == "cg" && pc == "lu") ||
        pc == "lu" || pc == "mumps") {
      SparseDirectUMFPACK direct;
      direct.factorize(system_matrix_);
      direct.vmult(solution_, system_rhs_);
      constraints_.distribute(solution_);
      return;
    }

    // Iterative solver
    ReductionControl solver_control(/*max_iter=*/50000, atol, rtol);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    // omega=1.2 is a common near-optimal value for structured meshes
    preconditioner.initialize(system_matrix_, 1.2);

    if (ksp == "cg") {
      SolverCG<Vector<double>> solver(solver_control);
      solver.solve(system_matrix_, solution_, system_rhs_, preconditioner);
    } else {
      // gmres / minres / bcgs all fall back to GMRES
      SolverGMRES<Vector<double>> solver(solver_control);
      solver.solve(system_matrix_, solution_, system_rhs_, preconditioner);
    }

    constraints_.distribute(solution_);
  }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: poisson_solver <case_spec.json> <output_dir>\n";
    return 1;
  }

  try {
    const CaseSpec spec = read_case_spec(argv[1]);
    PoissonOracle  oracle(spec);
    oracle.run(argv[2]);
  } catch (const std::exception& exc) {
    std::cerr << "ERROR: " << exc.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "ERROR: unknown exception\n";
    return 1;
  }

  return 0;
}
