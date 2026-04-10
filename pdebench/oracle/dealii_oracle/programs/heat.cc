/**
 * heat.cc  –  deal.II oracle for the heat equation
 *
 *   ∂u/∂t - κ Δu = f(x,t)   in Ω×(t₀,t_end]
 *   u = g(x,t)              on ∂Ω×(t₀,t_end]
 *   u = u₀(x)               in Ω at t = t₀
 *
 * Scheme: backward Euler (implicit)  →  (M + dt·K) uⁿ⁺¹ = M·uⁿ + dt·fⁿ⁺¹
 *
 * Supports both 2-D and 3-D unit domains.
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;

namespace {

static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}};

template <int dim>
void initialize_parser(FunctionParser<dim>& parser,
                       const std::string& expr,
                       const bool time_dependent = false) {
  if (time_dependent)
    parser.initialize(dim == 2 ? "x,y,t" : "x,y,z,t", expr, MU_CONST, true);
  else
    parser.initialize(dim == 2 ? "x,y" : "x,y,z", expr, MU_CONST, false);
}

}  // namespace

template <int dim>
class HeatOracle {
 public:
  explicit HeatOracle(const CaseSpec& s)
      : spec_(s), fe_(s.fem.degree), dh_(tria_) {}

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    Timer timer; timer.start();

    make_mesh();
    setup_system();
    assemble_matrices();
    time_march();

    timer.stop();
    oracle_util::write_scalar_grid<dim>(
        dh_, solution_, spec_.output_grid.bbox,
        spec_.output_grid.nx, spec_.output_grid.ny, spec_.output_grid.nz,
        outdir, timer.wall_time(),
        spec_.oracle_solver.ksp_type,
        spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<dim>         tria_;
  FE_Q<dim>                  fe_;
  DoFHandler<dim>            dh_;
  AffineConstraints<double>  constraints_;
  SparsityPattern            sp_;
  SparseMatrix<double>       mass_matrix_;
  SparseMatrix<double>       stiffness_matrix_;
  SparseMatrix<double>       system_matrix_;
  Vector<double>             solution_;
  Vector<double>             old_solution_;
  Vector<double>             system_rhs_;

  // time config
  double t0_    = 0.0, t_end_ = 1.0, dt_ = 0.01;
  std::string time_scheme_ = "backward_euler";

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
    if (!spec_.time_cfg.is_null()) {
      t0_          = spec_.time_cfg.value("t0",     0.0);
      t_end_       = spec_.time_cfg.value("t_end",  1.0);
      dt_          = spec_.time_cfg.value("dt",     0.01);
      time_scheme_ = spec_.time_cfg.value("scheme", "backward_euler");
    }
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    constraints_.clear();

    // We will update BCs each time step; just close empty constraints for sparsity.
    constraints_.close();

    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp);
    sp_.copy_from(dsp);

    mass_matrix_.reinit(sp_);
    stiffness_matrix_.reinit(sp_);
    system_matrix_.reinit(sp_);
    solution_.reinit(dh_.n_dofs());
    old_solution_.reinit(dh_.n_dofs());
    system_rhs_.reinit(dh_.n_dofs());
  }

  void assemble_matrices() {
    const std::string kappa_expr = spec_.computed_kappa();
    if (kappa_expr.empty())
      throw std::runtime_error("Heat oracle: _computed_kappa missing");

    FunctionParser<dim> kappa_func(1);
    initialize_parser<dim>(kappa_func, kappa_expr, false);

    QGauss<dim> quad(fe_.degree + 1);
    FEValues<dim> fev(fe_, quad,
                    update_values | update_gradients |
                    update_JxW_values | update_quadrature_points);

    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Me(n, n), Ke(n, n);
    std::vector<types::global_dof_index> ids(n);

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell);
      Me = 0; Ke = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const double kap = kappa_func.value(fev.quadrature_point(q));
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j) {
            Me(i, j) += fev.shape_value(i, q) * fev.shape_value(j, q) * JxW;
            Ke(i, j) += kap * fev.shape_grad(i, q) * fev.shape_grad(j, q) * JxW;
          }
      }
      cell->get_dof_indices(ids);
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
          mass_matrix_.add(ids[i], ids[j], Me(i, j));
          stiffness_matrix_.add(ids[i], ids[j], Ke(i, j));
        }
    }
  }

  void apply_bc(double t, AffineConstraints<double>& cons) {
    const std::string bc_expr = spec_.computed_bc();
    if (bc_expr.empty()) return;
    FunctionParser<dim> bc_func(1);
    initialize_parser<dim>(bc_func, bc_expr, true);
    bc_func.set_time(t);
    cons.clear();
    VectorTools::interpolate_boundary_values(dh_, 0, bc_func, cons);
    cons.close();
  }

  void set_initial_condition() {
    const std::string ic_expr = spec_.pde.value("_computed_ic", "0.0");
    FunctionParser<dim> ic_func(1);
    initialize_parser<dim>(ic_func, ic_expr, false);
    VectorTools::interpolate(dh_, ic_func, solution_);
  }

  void time_march() {
    set_initial_condition();
    old_solution_ = solution_;

    const std::string src_expr = spec_.computed_source();
    const bool has_src = !src_expr.empty();
    FunctionParser<dim> src_func(1);
    if (has_src)
      initialize_parser<dim>(src_func, src_expr, true);

    QGauss<dim> quad(fe_.degree + 1);

    double t = t0_;
    while (t < t_end_ - 1e-12 * dt_) {
      const double dt = std::min(dt_, t_end_ - t);
      t += dt;

      // Apply time-dependent BC
      AffineConstraints<double> cons;
      apply_bc(t, cons);

      // Build system:  (M + dt·K) uⁿ⁺¹ = M·uⁿ + dt·f(tⁿ⁺¹)
      system_matrix_.copy_from(mass_matrix_);
      system_matrix_.add(dt, stiffness_matrix_);

      // Apply constraints to system matrix (eliminate Dirichlet rows/cols)
      {
        AffineConstraints<double> empty;
        empty.close();
        cons.condense(system_matrix_);
      }

      // RHS = M·uⁿ
      mass_matrix_.vmult(system_rhs_, old_solution_);

      // Add source term contribution dt·∫f v dx
      if (has_src) {
        src_func.set_time(t);
        FEValues<dim> fev(fe_, quad,
                        update_values | update_JxW_values | update_quadrature_points);
        const unsigned int n = fe_.n_dofs_per_cell();
        Vector<double>     fe(n);
        std::vector<types::global_dof_index> ids(n);
        for (auto& cell : dh_.active_cell_iterators()) {
          fev.reinit(cell); fe = 0;
          for (unsigned int q = 0; q < quad.size(); ++q) {
            const double f = src_func.value(fev.quadrature_point(q));
            for (unsigned int i = 0; i < n; ++i)
              fe(i) += dt * f * fev.shape_value(i, q) * fev.JxW(q);
          }
          cell->get_dof_indices(ids);
          for (unsigned int i = 0; i < n; ++i)
            system_rhs_(ids[i]) += fe(i);
        }
      }

      cons.condense(system_rhs_);

      // Solve
      ReductionControl ctrl(50000, spec_.oracle_solver.atol, spec_.oracle_solver.rtol);
      PreconditionSSOR<SparseMatrix<double>> prec;
      prec.initialize(system_matrix_, 1.2);
      if (spec_.oracle_solver.ksp_type == "gmres") {
        SolverGMRES<Vector<double>> gmres(ctrl);
        gmres.solve(system_matrix_, solution_, system_rhs_, prec);
      } else {
        SolverCG<Vector<double>> cg(ctrl);
        cg.solve(system_matrix_, solution_, system_rhs_, prec);
      }
      cons.distribute(solution_);

      old_solution_ = solution_;
    }
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: heat_solver <case_spec.json> <output_dir>\n";
    return 1;
  }
  try {
    const CaseSpec spec = read_case_spec(argv[1]);
    if (spec.output_grid.is_3d() || spec.domain.type == "unit_cube")
      HeatOracle<3>(spec).run(argv[2]);
    else
      HeatOracle<2>(spec).run(argv[2]);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n"; return 1;
  }
  return 0;
}
