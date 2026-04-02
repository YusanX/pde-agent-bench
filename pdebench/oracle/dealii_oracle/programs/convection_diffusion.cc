/**
 * convection_diffusion.cc  –  deal.II oracle for convection-diffusion
 *
 * Steady:    β·∇u - ε Δu = f   in Ω = [0,1]²,  u = g on ∂Ω
 * Transient: ∂u/∂t + β·∇u - ε Δu = f   (backward Euler time stepping)
 *
 * Stabilisation: SUPG (Streamline Upwind Petrov-Galerkin) is applied when
 * the local Péclet number Pe_loc > 1.
 *
 * Solver: GMRES + SSOR (matrix is non-symmetric due to advection term).
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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;
namespace { static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}}; }

class ConvectionDiffusionOracle {
 public:
  explicit ConvectionDiffusionOracle(const CaseSpec& s)
      : spec_(s), fe_(s.fem.degree), dh_(tria_) {
    epsilon_ = std::stod(spec_.pde.value("_computed_epsilon", "0.01"));
    beta_x_  = std::stod(spec_.pde.value("_computed_beta_x",  "1.0"));
    beta_y_  = std::stod(spec_.pde.value("_computed_beta_y",  "0.0"));
    if (!spec_.time_cfg.is_null()) {
      t0_    = spec_.time_cfg.value("t0",    0.0);
      t_end_ = spec_.time_cfg.value("t_end", 1.0);
      dt_    = spec_.time_cfg.value("dt",    0.01);
      transient_ = true;
    }
  }

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    Timer timer; timer.start();
    make_mesh(); setup_system();
    if (transient_) time_march();
    else            solve_steady();
    timer.stop();
    oracle_util::write_scalar_grid(dh_, u_,
        spec_.output_grid.bbox, spec_.output_grid.nx, spec_.output_grid.ny,
        outdir, timer.wall_time(),
        spec_.oracle_solver.ksp_type, spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<2>           tria_;
  FE_Q<2>                    fe_;
  DoFHandler<2>              dh_;
  AffineConstraints<double>  cons_;
  SparsityPattern            sp_;
  SparseMatrix<double>       K_;    // advection-diffusion stiffness
  SparseMatrix<double>       M_;    // mass matrix (transient)
  SparseMatrix<double>       sys_;  // M + dt*K
  Vector<double>             u_, old_u_, rhs_;

  double epsilon_ = 0.01, beta_x_ = 1.0, beta_y_ = 0.0;
  double t0_ = 0.0, t_end_ = 1.0, dt_ = 0.01;
  bool   transient_ = false;

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    cons_.clear();
    FunctionParser<2> bc(1);
    bc.initialize("x,y", spec_.computed_bc(), MU_CONST, false);
    VectorTools::interpolate_boundary_values(dh_, 0, bc, cons_);
    cons_.close();

    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp, cons_);
    sp_.copy_from(dsp);
    K_.reinit(sp_);
    if (transient_) { M_.reinit(sp_); sys_.reinit(sp_); }
    u_.reinit(dh_.n_dofs());
    old_u_.reinit(dh_.n_dofs());
    rhs_.reinit(dh_.n_dofs());
  }

  // Cell-level SUPG stabilisation parameter
  double supg_tau(double h_cell) const {
    const double beta_norm = std::sqrt(beta_x_*beta_x_ + beta_y_*beta_y_);
    if (beta_norm < 1e-14) return 0.0;
    const double Pe_loc = beta_norm * h_cell / (2.0 * epsilon_);
    if (Pe_loc <= 1.0) return 0.0;
    return h_cell / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_loc);
  }

  void assemble_KM(double t = 0.0) {
    K_ = 0;
    if (transient_) M_ = 0;

    FunctionParser<2> src(1);
    const bool has_t = transient_;
    if (has_t)
      src.initialize("x,y,t", spec_.computed_source(), MU_CONST, true);
    else
      src.initialize("x,y",   spec_.computed_source(), MU_CONST, false);
    if (has_t) src.set_time(t);

    // h_cell approximation: diameter of reference cell
    const double dx = 1.0 / spec_.mesh.resolution;

    QGauss<2>   quad(fe_.degree + 1);
    FEValues<2> fev(fe_, quad,
                    update_values | update_gradients |
                    update_JxW_values | update_quadrature_points);

    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n), Me(n, n);
    Vector<double>     Fe(n);
    std::vector<types::global_dof_index> ids(n);

    const double tau = supg_tau(dx * std::sqrt(2.0));  // cell diameter approx.

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Ke = 0; Me = 0; Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const double f   = src.value(fev.quadrature_point(q));
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i) {
          // SUPG test function modifier: v + τ β·∇v
          const double supg_i = tau * (beta_x_ * fev.shape_grad(i,q)[0]
                                     + beta_y_ * fev.shape_grad(i,q)[1]);
          const double vi     = fev.shape_value(i,q) + supg_i;

          for (unsigned int j = 0; j < n; ++j) {
            const double conv_j = beta_x_ * fev.shape_grad(j,q)[0]
                                + beta_y_ * fev.shape_grad(j,q)[1];
            Ke(i,j) += (epsilon_ * fev.shape_grad(i,q) * fev.shape_grad(j,q)
                       + fev.shape_value(i,q) * conv_j
                       + supg_i * conv_j  // SUPG term
                       ) * JxW;
            if (transient_)
              Me(i,j) += vi * fev.shape_value(j,q) * JxW;
          }
          Fe(i) += f * vi * JxW;
        }
      }
      cell->get_dof_indices(ids);
      cons_.distribute_local_to_global(Ke, Fe, ids, K_, rhs_);
      if (transient_) {
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            M_.add(ids[i], ids[j], Me(i,j));
      }
    }
  }

  void solve_steady() {
    assemble_KM();
    ReductionControl ctrl(50000, spec_.oracle_solver.atol, spec_.oracle_solver.rtol);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(K_, 1.2);
    SolverGMRES<Vector<double>> gmres(ctrl);
    gmres.solve(K_, u_, rhs_, prec);
    cons_.distribute(u_);
  }

  void time_march() {
    // Set IC
    const std::string ic = spec_.pde.value("_computed_ic", "0.0");
    FunctionParser<2> ic_func(1);
    ic_func.initialize("x,y", ic, MU_CONST, false);
    VectorTools::interpolate(dh_, ic_func, u_);
    old_u_ = u_;

    double t = t0_;
    while (t < t_end_ - 1e-12 * dt_) {
      const double dt = std::min(dt_, t_end_ - t);
      t += dt;

      rhs_ = 0;
      assemble_KM(t);   // rebuilds K_ and M_, adds source to rhs_

      // sys = M + dt*K
      sys_.copy_from(M_);
      sys_.add(dt, K_);
      // RHS += M * u_old
      Vector<double> Mu(dh_.n_dofs());
      M_.vmult(Mu, old_u_);
      rhs_ += Mu;

      // Apply BC at time t
      AffineConstraints<double> cons_t;
      cons_t.clear();
      FunctionParser<2> bc_t(1);
      bc_t.initialize("x,y,t", spec_.computed_bc(), MU_CONST, true);
      bc_t.set_time(t);
      VectorTools::interpolate_boundary_values(dh_, 0, bc_t, cons_t);
      cons_t.close();
      cons_t.condense(sys_);
      cons_t.condense(rhs_);

      ReductionControl ctrl(50000, spec_.oracle_solver.atol, spec_.oracle_solver.rtol);
      PreconditionSSOR<SparseMatrix<double>> prec;
      prec.initialize(sys_, 1.2);
      SolverGMRES<Vector<double>> gmres(ctrl);
      gmres.solve(sys_, u_, rhs_, prec);
      cons_t.distribute(u_);
      old_u_ = u_;
    }
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: convection_diffusion_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  try { ConvectionDiffusionOracle(read_case_spec(argv[1])).run(argv[2]); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
