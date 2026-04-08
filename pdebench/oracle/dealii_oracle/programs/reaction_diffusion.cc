/**
 * reaction_diffusion.cc  –  deal.II oracle for the reaction-diffusion equation
 *
 * Steady:    -Δu + σ u = f   in Ω=[0,1]²,  u=g on ∂Ω
 * Transient: ∂u/∂t - Δu + σ u = f,  u(·,0)=u₀,  u=g(·,t) on ∂Ω
 *
 * Backward-Euler: (M + dt·K) u^{n+1} = dt·f^{n+1} + M·u^n
 *
 * Key optimisations for transient:
 *  - K and M assembled ONCE (linear PDE, constant coefficients).
 *  - System matrix (M + dt·K) condensed and factorised ONCE.
 *  - Time-varying Dirichlet BCs handled via the "lifting" technique so the
 *    factorisation can be reused at every time step.
 *  - CG (not GMRES) for the SPD reaction-diffusion system.
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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;
namespace { static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}}; }

class ReactionDiffusionOracle {
 public:
  explicit ReactionDiffusionOracle(const CaseSpec& s)
      : spec_(s), fe_(s.fem.degree), dh_(tria_) {
    if (!spec_.time_cfg.is_null()) {
      t0_        = spec_.time_cfg.value("t0",    0.0);
      t_end_     = spec_.time_cfg.value("t_end", 1.0);
      dt_        = spec_.time_cfg.value("dt",    0.01);
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
  const CaseSpec&   spec_;
  Triangulation<2>  tria_;
  FE_Q<2>           fe_;
  DoFHandler<2>     dh_;
  SparsityPattern   sp_;
  SparseMatrix<double> K_, M_;
  SparseMatrix<double> sys_;        // condensed (M + dt*K), reused each step
  SparseMatrix<double> sys_uncond_; // un-condensed copy, needed for lifting
  Vector<double>    u_, old_u_, rhs_;
  SparseDirectUMFPACK direct_;

  bool   transient_ = false;
  double t0_ = 0.0, t_end_ = 1.0, dt_ = 0.01;

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp);
    sp_.copy_from(dsp);
    K_.reinit(sp_); M_.reinit(sp_);
    sys_.reinit(sp_); sys_uncond_.reinit(sp_);
    u_.reinit(dh_.n_dofs()); rhs_.reinit(dh_.n_dofs());
    if (transient_) old_u_.reinit(dh_.n_dofs());
  }

  // --- helpers ----------------------------------------------------------

  bool use_direct() const {
    const std::string& ksp = spec_.oracle_solver.ksp_type;
    const std::string& pc  = spec_.oracle_solver.pc_type;
    // ilu/icc: SparseILU has no symmetric positive-definite guarantee, so
    // falling through to CG+SSOR is often worse; use UMFPACK instead.
    return pc=="lu" || pc=="direct" || pc=="mumps" || ksp=="preonly"
        || pc=="ilu" || pc=="ilu0"  || pc=="icc";
  }

  // Assemble K (stiffness+reaction) and M (mass) – called ONCE.
  void assemble_KM(double sigma) {
    QGauss<2>   quad(fe_.degree + 1);
    FEValues<2> fev(fe_, quad,
                    update_values | update_gradients |
                    update_JxW_values | update_quadrature_points);
    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n,n), Me(n,n);
    std::vector<types::global_dof_index> ids(n);
    K_ = 0; M_ = 0;
    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Ke = 0; Me = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i) {
          const double      vi = fev.shape_value(i,q);
          const Tensor<1,2> gi = fev.shape_grad(i,q);
          for (unsigned int j = 0; j < n; ++j) {
            const double      vj = fev.shape_value(j,q);
            const Tensor<1,2> gj = fev.shape_grad(j,q);
            Ke(i,j) += (gi*gj + sigma*vi*vj) * JxW;
            Me(i,j) += vi*vj * JxW;
          }
        }
      }
      cell->get_dof_indices(ids);
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
          K_.add(ids[i], ids[j], Ke(i,j));
          M_.add(ids[i], ids[j], Me(i,j));
        }
    }
  }

  // Accumulate source-term load into rhs_ (does NOT zero rhs_ first).
  void add_source_to_rhs(Function<2>& src) {
    QGauss<2>   quad(fe_.degree + 1);
    FEValues<2> fev(fe_, quad,
                    update_values | update_JxW_values | update_quadrature_points);
    const unsigned int n = fe_.n_dofs_per_cell();
    Vector<double> Fe(n);
    std::vector<types::global_dof_index> ids(n);
    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const double f   = src.value(fev.quadrature_point(q));
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i)
          Fe(i) += f * fev.shape_value(i,q) * JxW;
      }
      cell->get_dof_indices(ids);
      for (unsigned int i = 0; i < n; ++i) rhs_[ids[i]] += Fe(i);
    }
  }

  // CG+SSOR solve on the (already condensed) sys_ matrix.
  void solve_cg() {
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(sys_, 1.2);
    ReductionControl ctrl(spec_.oracle_solver.max_it,
                          spec_.oracle_solver.atol, spec_.oracle_solver.rtol);
    SolverCG<Vector<double>> cg(ctrl);
    cg.solve(sys_, u_, rhs_, prec);
  }

  // --- steady solver ----------------------------------------------------
  void solve_steady() {
    const double sigma = std::stod(spec_.pde.value("_computed_sigma", "1.0"));
    assemble_KM(sigma);

    FunctionParser<2> src(1), bc_func(1);
    src.initialize("x,y",     spec_.computed_source(), MU_CONST, false);
    bc_func.initialize("x,y", spec_.computed_bc(),     MU_CONST, false);

    AffineConstraints<double> cons;
    cons.clear();
    VectorTools::interpolate_boundary_values(dh_, 0, bc_func, cons);
    cons.close();

    rhs_ = 0;
    add_source_to_rhs(src);
    sys_.copy_from(K_);
    cons.condense(sys_, rhs_);

    if (use_direct()) {
      direct_.initialize(sys_);
      direct_.vmult(u_, rhs_);
    } else {
      u_ = 0.0;
      solve_cg();
    }
    cons.distribute(u_);
  }

  // --- transient solver (Backward Euler) --------------------------------
  void time_march() {
    const double sigma     = std::stod(spec_.pde.value("_computed_sigma", "1.0"));
    const std::string src_expr = spec_.computed_source();
    const std::string bc_expr  = spec_.computed_bc();

    // ---- IC ----
    FunctionParser<2> ic_func(1);
    ic_func.initialize("x,y", spec_.pde.value("_computed_ic", "0.0"), MU_CONST, false);
    VectorTools::interpolate(dh_, ic_func, u_);
    old_u_ = u_;

    // ---- Assemble K and M once ----
    assemble_KM(sigma);

    // ---- Build sys_uncond = M + dt*K (assume dt_ for typical step) ----
    const double dt0 = std::min(dt_, t_end_ - t0_);
    sys_uncond_.copy_from(M_);
    sys_uncond_.add(dt0, K_);

    // ---- Build condensed sys_ and factorise/setup ONCE ----
    // We only need the BC PATTERN (which DOFs are on the boundary) for
    // condensation; the values are irrelevant for the matrix.
    {
      FunctionParser<2> bc0(1);
      bc0.initialize("x,y,t", bc_expr, MU_CONST, true);
      bc0.set_time(t0_ + dt0);
      AffineConstraints<double> cons0;
      VectorTools::interpolate_boundary_values(dh_, 0, bc0, cons0);
      cons0.close();

      sys_.copy_from(sys_uncond_);
      Vector<double> dummy(dh_.n_dofs());
      cons0.condense(sys_, dummy);   // zeros rows/cols, diagonal=1
    }
    if (use_direct()) direct_.initialize(sys_);

    // ---- Reusable function parsers ----
    FunctionParser<2> src_t(1), bc_t(1);
    src_t.initialize("x,y,t", src_expr, MU_CONST, true);
    bc_t.initialize("x,y,t",  bc_expr,  MU_CONST, true);

    double t        = t0_;
    double prev_dt  = dt0;

    while (t < t_end_ - 1e-12 * dt_) {
      const double dt = std::min(dt_, t_end_ - t);
      t += dt;

      // If dt changed (usually only on the very last step), rebuild and
      // re-factorize the system matrix.
      if (std::abs(dt - prev_dt) > 1e-14 * dt_) {
        sys_uncond_.copy_from(M_); sys_uncond_.add(dt, K_);

        FunctionParser<2> bc_rebuild(1);
        bc_rebuild.initialize("x,y,t", bc_expr, MU_CONST, true);
        bc_rebuild.set_time(t);
        AffineConstraints<double> cons_rebuild;
        VectorTools::interpolate_boundary_values(dh_, 0, bc_rebuild, cons_rebuild);
        cons_rebuild.close();

        sys_.copy_from(sys_uncond_);
        Vector<double> dummy(dh_.n_dofs());
        cons_rebuild.condense(sys_, dummy);
        if (use_direct()) direct_.initialize(sys_);
        prev_dt = dt;
      }

      // ---- Build RHS = dt·f(t) + M·u_old ----
      src_t.set_time(t);
      rhs_ = 0;
      add_source_to_rhs(src_t);
      rhs_ *= dt;

      Vector<double> Mu(dh_.n_dofs());
      M_.vmult(Mu, old_u_);
      rhs_ += Mu;

      // ---- Lifting for time-varying Dirichlet BCs ----
      // With a cached factorisation the matrix cannot be modified each step.
      // Instead apply: rhs -= sys_uncond * g_vec, then rhs[boundary] = g_i(t).
      // This is equivalent to the standard condense(mat,rhs) approach when
      // the boundary pattern is fixed (same DOFs constrained at every step).
      bc_t.set_time(t);
      std::map<types::global_dof_index, double> bv;
      VectorTools::interpolate_boundary_values(dh_, 0, bc_t, bv);

      // g_vec: BC values at boundary DOFs, zero elsewhere
      Vector<double> g_vec(dh_.n_dofs());
      for (auto& [idx, val] : bv) g_vec[idx] = val;

      // Lifting: rhs -= sys_uncond * g_vec
      Vector<double> lifting(dh_.n_dofs());
      sys_uncond_.vmult(lifting, g_vec);
      rhs_ -= lifting;

      // Impose RHS at boundary DOFs
      for (auto& [idx, val] : bv) rhs_[idx] = val;

      // ---- Solve ----
      if (use_direct()) {
        direct_.vmult(u_, rhs_);
      } else {
        // Cold start: ensures ||r₀||=||b||≈O(1), so rtol·||r₀|| stays
        // well above machine epsilon and ReductionControl can be satisfied.
        u_ = 0.0;
        solve_cg();
      }

      // Enforce boundary values exactly (direct solve is usually exact,
      // but be defensive in case of any rounding)
      for (auto& [idx, val] : bv) u_[idx] = val;
      old_u_ = u_;
    }
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: reaction_diffusion_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  try { ReactionDiffusionOracle(read_case_spec(argv[1])).run(argv[2]); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
