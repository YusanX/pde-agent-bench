/**
 * poisson.cc  –  deal.II oracle for the Poisson equation
 *
 *   -∇·(κ ∇u) = f   in Ω
 *           u = g   on ∂Ω
 *
 * Supports both unit-square/cube domains (FE_Q, structured quad mesh) and
 * complex 2-D geometries loaded from a Gmsh .msh file (FE_SimplexP, triangular
 * mesh).  The mesh type is detected at run-time via mesh_factory.h.
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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

#include "case_spec_reader.h"
#include "grid_writer.h"
#include "mesh_factory.h"

using namespace dealii;

namespace {
static const std::map<std::string, double> MU_CONSTANTS = {{"pi", M_PI}};

template <int dim>
std::unique_ptr<FunctionParser<dim>>
make_func(const std::string& expr, bool time_dep = false) {
  auto fp = std::make_unique<FunctionParser<dim>>(1);
  fp->initialize(dim == 2 ? "x,y" : "x,y,z", expr, MU_CONSTANTS, time_dep);
  return fp;
}
}  // namespace

template <int dim>
class PoissonOracle {
 public:
  explicit PoissonOracle(const CaseSpec& spec)
      : spec_(spec), dof_handler_(tria_) {}

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);

    Timer timer;
    timer.start();

    make_mesh();
    setup_system();
    assemble_system();
    solve_system();

    timer.stop();
    const double elapsed = timer.wall_time();

    oracle_util::write_scalar_grid<dim>(
        dof_handler_, solution_, spec_.output_grid.bbox, spec_.output_grid.nx,
        spec_.output_grid.ny, spec_.output_grid.nz, outdir, elapsed,
        spec_.oracle_solver.ksp_type, spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<dim>         tria_;
  bool                       is_simplex_ = false;
  std::unique_ptr<FiniteElement<dim>> fe_;
  DoFHandler<dim>            dof_handler_;
  AffineConstraints<double>  constraints_;
  SparsityPattern            sparsity_pattern_;
  SparseMatrix<double>       system_matrix_;
  Vector<double>             solution_;
  Vector<double>             system_rhs_;

  void make_mesh() {
    is_simplex_ = oracle_util::make_mesh<dim>(spec_, tria_);
    fe_ = oracle_util::make_scalar_fe<dim>(spec_.fem.degree, is_simplex_);
  }

  void setup_system() {
    dof_handler_.distribute_dofs(*fe_);
    constraints_.clear();

    const std::string bc_expr = spec_.computed_bc();
    if (bc_expr.empty())
      throw std::runtime_error("Poisson oracle: _computed_bc missing in case_spec");

    auto bc_func = make_func<dim>(bc_expr);
    VectorTools::interpolate_boundary_values(dof_handler_, 0, *bc_func, constraints_);
    constraints_.close();

    DynamicSparsityPattern dsp(dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_, dsp, constraints_);
    sparsity_pattern_.copy_from(dsp);

    system_matrix_.reinit(sparsity_pattern_);
    solution_.reinit(dof_handler_.n_dofs());
    system_rhs_.reinit(dof_handler_.n_dofs());
  }

  void assemble_system() {
    const std::string kappa_expr  = spec_.computed_kappa();
    const std::string source_expr = spec_.computed_source();
    if (kappa_expr.empty() || source_expr.empty())
      throw std::runtime_error("Poisson oracle: _computed_kappa or _computed_source missing");

    auto kappa_func  = make_func<dim>(kappa_expr);
    auto source_func = make_func<dim>(source_expr);

    const auto quadrature = oracle_util::make_quadrature<dim>(*fe_);
    FEValues<dim> fe_values(*fe_, quadrature,
                            update_values | update_gradients |
                            update_JxW_values | update_quadrature_points);

    const unsigned int n_dpc  = fe_->n_dofs_per_cell();
    const unsigned int n_q    = quadrature.size();

    FullMatrix<double> cell_matrix(n_dpc, n_dpc);
    Vector<double>     cell_rhs(n_dpc);
    std::vector<types::global_dof_index> local_dof_indices(n_dpc);

    for (auto& cell : dof_handler_.active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < n_q; ++q) {
        const Point<dim>& qp    = fe_values.quadrature_point(q);
        const double      kappa = kappa_func->value(qp);
        const double      f_val = source_func->value(qp);
        const double      JxW   = fe_values.JxW(q);

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
          cell_matrix, cell_rhs, local_dof_indices, system_matrix_, system_rhs_);
    }
  }

  void solve_system() {
    const std::string ksp  = spec_.oracle_solver.ksp_type;
    const std::string pc   = spec_.oracle_solver.pc_type;
    const double      rtol = spec_.oracle_solver.rtol;
    const double      atol = spec_.oracle_solver.atol;

    if (ksp == "preonly" || (ksp == "cg" && pc == "lu") || pc == "lu" || pc == "mumps") {
      SparseDirectUMFPACK direct;
      direct.factorize(system_matrix_);
      direct.vmult(solution_, system_rhs_);
      constraints_.distribute(solution_);
      return;
    }

    ReductionControl solver_control(50000, atol, rtol);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix_, 1.2);

    if (ksp == "cg") {
      SolverCG<Vector<double>> solver(solver_control);
      solver.solve(system_matrix_, solution_, system_rhs_, preconditioner);
    } else {
      SolverGMRES<Vector<double>> solver(solver_control);
      solver.solve(system_matrix_, solution_, system_rhs_, preconditioner);
    }

    constraints_.distribute(solution_);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: poisson_solver <case_spec.json> <output_dir>\n";
    return 1;
  }

  try {
    const CaseSpec spec = read_case_spec(argv[1]);
    if (spec.output_grid.is_3d() || spec.domain.type == "unit_cube")
      PoissonOracle<3>(spec).run(argv[2]);
    else
      PoissonOracle<2>(spec).run(argv[2]);
  } catch (const std::exception& exc) {
    std::cerr << "ERROR: " << exc.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "ERROR: unknown exception\n";
    return 1;
  }

  return 0;
}
