/**
 * convection_diffusion.cc  –  deal.II oracle for convection-diffusion
 *
 * Steady:    β·∇u - ε Δu = f   in Ω = [0,1]²,  u = g on ∂Ω
 * Transient: ∂u/∂t + β·∇u - ε Δu = f   (backward Euler time stepping)
 *
 * Stabilisation: SUPG (Streamline Upwind Petrov-Galerkin) applied when
 * oracle_solver.stabilization == "supg" (or pde_params.stabilization == "supg").
 * The full residual-based SUPG bilinear form is used:
 *   a_SUPG(u,v) += τ (β·∇v)(β·∇u - ε Δu)
 *
 * Solver: GMRES + ILU (for pc_type="ilu") or SSOR (default) or AMG (hypre).
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
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
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
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
class ConvectionDiffusionOracle {
 public:
  explicit ConvectionDiffusionOracle(const CaseSpec& s)
      : spec_(s), fe_(s.fem.degree), dh_(tria_) {
    epsilon_ = std::stod(spec_.pde.value("_computed_epsilon", "0.01"));
    beta_[0] = std::stod(spec_.pde.value("_computed_beta_x",  "1.0"));
    if constexpr (dim >= 2)
      beta_[1] = std::stod(spec_.pde.value("_computed_beta_y",  "0.0"));
    if constexpr (dim >= 3)
      beta_[2] = std::stod(spec_.pde.value("_computed_beta_z",  "0.0"));

    // Stabilization: oracle_solver overrides pde_params
    stabilization_ = spec_.oracle_solver.stabilization;
    if (stabilization_ == "none" && spec_.pde_params.contains("stabilization"))
      stabilization_ = spec_.pde_params.value("stabilization", "none");

    upwind_parameter_ = spec_.oracle_solver.upwind_parameter;

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
    oracle_util::write_scalar_grid<dim>(dh_, u_,
        spec_.output_grid.bbox, spec_.output_grid.nx, spec_.output_grid.ny, spec_.output_grid.nz,
        outdir, timer.wall_time(),
        spec_.oracle_solver.ksp_type, spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<dim>         tria_;
  FE_Q<dim>                  fe_;
  DoFHandler<dim>            dh_;
  AffineConstraints<double>  cons_;
  SparsityPattern            sp_;
  SparseMatrix<double>       K_;    // advection-diffusion stiffness
  SparseMatrix<double>       M_;    // mass matrix (transient)
  SparseMatrix<double>       sys_;  // M + dt*K
  Vector<double>             u_, old_u_, rhs_;

  double              epsilon_ = 0.01;
  Tensor<1, dim>      beta_;
  double      t0_ = 0.0, t_end_ = 1.0, dt_ = 0.01;
  std::string stabilization_    = "none";
  double      upwind_parameter_ = 1.0;
  bool        transient_        = false;

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    cons_.clear();
    FunctionParser<dim> bc(1);
    if (transient_) {
      initialize_parser<dim>(bc, spec_.computed_bc(), true);
      bc.set_time(t0_);
    } else {
      initialize_parser<dim>(bc, spec_.computed_bc(), false);
    }
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

  // SUPG stabilisation parameter τ = α * h / (2|β|)
  // Matches dolfinx formula: tau = upwind_parameter * h / (2 * beta_norm)
  double supg_tau(double h_cell) const {
    if (stabilization_ != "supg") return 0.0;
    const double beta_norm = beta_.norm();
    if (beta_norm < 1e-14) return 0.0;
    return upwind_parameter_ * h_cell / (2.0 * beta_norm);
  }

  void assemble_KM(double t = 0.0) {
    K_ = 0;
    if (transient_) M_ = 0;

    FunctionParser<dim> src(1);
    const bool has_t = transient_;
    if (has_t)
      initialize_parser<dim>(src, spec_.computed_source(), true);
    else
      initialize_parser<dim>(src, spec_.computed_source(), false);
    if (has_t) src.set_time(t);

    // h_cell: diameter of the reference cell on a unit hypercube mesh
    const double dx = 1.0 / spec_.mesh.resolution;
    const double tau = supg_tau(dx * std::sqrt(static_cast<double>(dim)));

    // update_hessians needed for the full SUPG residual term (−ε τ (β·∇v) Δu)
    // For Q1 elements the Laplacian of shape functions is zero, so it is a
    // no-op for degree=1.  For Q2+ elements it provides the consistency term.
    QGauss<dim>   quad(fe_.degree + 1);
    FEValues<dim> fev(fe_, quad,
                    update_values | update_gradients | update_hessians |
                    update_JxW_values | update_quadrature_points);

    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n), Me(n, n);
    Vector<double>     Fe(n);
    std::vector<types::global_dof_index> ids(n);

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Ke = 0; Me = 0; Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const double f   = src.value(fev.quadrature_point(q));
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i) {
          // SUPG test function modifier: v + τ β·∇v
          double beta_dot_grad_i = 0.0;
          for (unsigned int d = 0; d < dim; ++d)
            beta_dot_grad_i += beta_[d] * fev.shape_grad(i,q)[d];
          const double supg_i = tau * beta_dot_grad_i;
          const double vi     = fev.shape_value(i,q) + supg_i;

          for (unsigned int j = 0; j < n; ++j) {
            double conv_j = 0.0;
            for (unsigned int d = 0; d < dim; ++d)
              conv_j += beta_[d] * fev.shape_grad(j,q)[d];
            // Laplacian of shape function j (zero for Q1, nonzero for Q2+)
            double lap_j = 0.0;
            for (unsigned int d = 0; d < dim; ++d)
              lap_j += fev.shape_hessian(j,q)[d][d];

            // Full residual-based SUPG bilinear form:
            //   ε(∇u,∇v) + (β·∇u)v + τ(β·∇v)(β·∇u − ε Δu)
            Ke(i,j) += (epsilon_ * fev.shape_grad(i,q) * fev.shape_grad(j,q)
                       + fev.shape_value(i,q) * conv_j
                       + supg_i * (conv_j - epsilon_ * lap_j)
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

  // ---------------------------------------------------------------------------
  // Helper: GMRES + BoomerAMG via PETSc wrappers.
  // Used when oracle_solver.pc_type == "hypre".
  // ---------------------------------------------------------------------------
  void solve_petsc_amg(SparseMatrix<double>& mat, Vector<double>& sol,
                       const Vector<double>& rhs_vec) {
    const unsigned int n = dh_.n_dofs();

    PETScWrappers::SparseMatrix K_petsc(sp_);
    for (unsigned int row = 0; row < n; ++row)
      for (auto it = mat.begin(row); it != mat.end(row); ++it)
        K_petsc.set(row, it->column(), it->value());
    K_petsc.compress(VectorOperation::insert);

    PETScWrappers::MPI::Vector rhs_petsc(MPI_COMM_SELF, rhs_vec, n);
    PETScWrappers::MPI::Vector sol_petsc(MPI_COMM_SELF, n, n);

    const double rhs_norm = rhs_petsc.l2_norm();
    const double tol = std::max(spec_.oracle_solver.atol,
                                spec_.oracle_solver.rtol * rhs_norm);
    SolverControl ctrl(spec_.oracle_solver.max_it, tol);

    PETScWrappers::SolverGMRES gmres(ctrl);
    PETScWrappers::PreconditionBoomerAMG amg;
    PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.symmetric_operator = false;
    amg.initialize(K_petsc, data);
    gmres.solve(K_petsc, sol_petsc, rhs_petsc, amg);

    for (unsigned int i = 0; i < n; ++i)
      sol(i) = sol_petsc(i);
  }

  // ---------------------------------------------------------------------------
  // Unified linear solver dispatcher.
  // pc_type == "hypre"/"boomeramg"/"amg" → PETSc BoomerAMG
  // pc_type == "ilu"/"ilu0"/"icc"        → SparseILU  (matches dolfinx/PETSc)
  // anything else                         → SSOR
  // ---------------------------------------------------------------------------
  void solve_linear(SparseMatrix<double>& mat, Vector<double>& sol,
                    const Vector<double>& rhs_vec) {
    const std::string ksp = spec_.oracle_solver.ksp_type;
    const std::string pc  = spec_.oracle_solver.pc_type;

    // Direct solver (UMFPACK): pc_type="lu"/"direct" or ksp_type="preonly"
    if (pc == "lu" || pc == "direct" || pc == "mumps" ||
        ksp == "preonly") {
      SparseDirectUMFPACK direct;
      direct.initialize(mat);
      direct.vmult(sol, rhs_vec);
      return;
    }

    if (pc == "hypre" || pc == "boomeramg" || pc == "amg") {
      solve_petsc_amg(mat, sol, rhs_vec);
      return;
    }

    ReductionControl ctrl(spec_.oracle_solver.max_it,
                          spec_.oracle_solver.atol,
                          spec_.oracle_solver.rtol);
    SolverGMRES<Vector<double>> gmres(ctrl);

    if (pc == "ilu" || pc == "ilu0" || pc == "icc") {
      SparseILU<double> prec;
      prec.initialize(mat);
      gmres.solve(mat, sol, rhs_vec, prec);
    } else {
      // Default: SSOR (for symmetric/nearly-symmetric problems)
      PreconditionSSOR<SparseMatrix<double>> prec;
      prec.initialize(mat, 1.2);
      gmres.solve(mat, sol, rhs_vec, prec);
    }
  }

  void solve_steady() {
    assemble_KM();
    solve_linear(K_, u_, rhs_);
    cons_.distribute(u_);
  }

  void time_march() {
    // Set IC
    const std::string ic = spec_.pde.value("_computed_ic", "0.0");
    FunctionParser<dim> ic_func(1);
    initialize_parser<dim>(ic_func, ic, false);
    VectorTools::interpolate(dh_, ic_func, u_);
    old_u_ = u_;

    double t = t0_;
    while (t < t_end_ - 1e-12 * dt_) {
      const double dt = std::min(dt_, t_end_ - t);
      t += dt;

      rhs_ = 0;
      assemble_KM(t);   // rebuilds K_ and M_, adds source F^n to rhs_

      // Backward Euler: (M + dt*K) u^n = dt*F^n + M*u^{n-1}
      // Scale source contribution by dt before adding M*u_old
      rhs_ *= dt;

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
      FunctionParser<dim> bc_t(1);
      initialize_parser<dim>(bc_t, spec_.computed_bc(), true);
      bc_t.set_time(t);
      VectorTools::interpolate_boundary_values(dh_, 0, bc_t, cons_t);
      cons_t.close();
      cons_t.condense(sys_);
      cons_t.condense(rhs_);

      solve_linear(sys_, u_, rhs_);
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
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  try {
    const CaseSpec spec = read_case_spec(argv[1]);
    if (spec.output_grid.is_3d() || spec.domain.type == "unit_cube")
      ConvectionDiffusionOracle<3>(spec).run(argv[2]);
    else
      ConvectionDiffusionOracle<2>(spec).run(argv[2]);
  }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
