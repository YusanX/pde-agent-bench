/**
 * linear_elasticity.cc  –  deal.II oracle for linear elasticity
 *
 *   -∇·σ(u) = f   in Ω = [0,1]²
 *         u = g   on ∂Ω
 *
 *   σ = λ(∇·u)I + μ(∇u + ∇uᵀ)   (Cauchy stress, plane stress/strain)
 *
 * FE space: Q2 Taylor-Hood or Q1 vector Lagrange (degree from case_spec).
 * Output:   velocity / displacement magnitude  ‖u‖ = √(u₁²+u₂²).
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;
namespace { static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}}; }

// Vector BC function: components (ux, uy) provided as two separate expressions
class VectorBCFunc : public Function<2> {
 public:
  VectorBCFunc(const std::string& expr_x, const std::string& expr_y)
      : Function<2>(2), fx_(1), fy_(1) {
    std::map<std::string, double> c = {{"pi", M_PI}};
    fx_.initialize("x,y", expr_x, c, false);
    fy_.initialize("x,y", expr_y, c, false);
  }
  double value(const Point<2>& p, unsigned int comp = 0) const override {
    return (comp == 0) ? fx_.value(p) : fy_.value(p);
  }
  void vector_value(const Point<2>& p, Vector<double>& v) const override {
    v(0) = fx_.value(p); v(1) = fy_.value(p);
  }
 private:
  mutable FunctionParser<2> fx_, fy_;
};

class LinearElasticityOracle {
 public:
  explicit LinearElasticityOracle(const CaseSpec& s)
      : spec_(s),
        fe_(FE_Q<2>(s.fem.degree), 2),   // vector Q_p space
        dh_(tria_) {
    lam_ = std::stod(spec_.pde.value("_computed_lambda", "1.0"));
    mu_  = std::stod(spec_.pde.value("_computed_mu",     "1.0"));
  }

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    Timer timer; timer.start();
    make_mesh(); setup_system(); assemble(); solve();
    timer.stop();

    // Write vector magnitude grid
    oracle_util::write_vector_magnitude_grid(dh_, u_,
        spec_.output_grid.bbox, spec_.output_grid.nx, spec_.output_grid.ny,
        outdir, timer.wall_time(),
        spec_.oracle_solver.ksp_type, spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<2>           tria_;
  FESystem<2>                fe_;
  DoFHandler<2>              dh_;
  AffineConstraints<double>  cons_;
  SparsityPattern            sp_;
  SparseMatrix<double>       K_;
  Vector<double>             u_, rhs_;
  double                     lam_, mu_;

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    cons_.clear();

    const std::string bc_x = spec_.pde.value("_computed_bc_x", "0.0");
    const std::string bc_y = spec_.pde.value("_computed_bc_y", "0.0");
    VectorBCFunc bc_func(bc_x, bc_y);
    VectorTools::interpolate_boundary_values(
        dh_, 0, bc_func, cons_,
        ComponentMask());   // apply to all components
    cons_.close();

    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp, cons_);
    sp_.copy_from(dsp);
    K_.reinit(sp_);
    u_.reinit(dh_.n_dofs());
    rhs_.reinit(dh_.n_dofs());
  }

  void assemble() {
    const std::string fx_expr = spec_.pde.value("_computed_source_x", "0.0");
    const std::string fy_expr = spec_.pde.value("_computed_source_y", "0.0");
    std::map<std::string, double> c = {{"pi", M_PI}};
    FunctionParser<2> fx(1), fy(1);
    fx.initialize("x,y", fx_expr, c, false);
    fy.initialize("x,y", fy_expr, c, false);

    QGauss<2>   quad(fe_.degree + 1);
    FEValues<2> fev(fe_, quad,
                    update_values | update_gradients |
                    update_JxW_values | update_quadrature_points);

    const FEValuesExtractors::Vector displ(0);
    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n); Vector<double> Fe(n);
    std::vector<types::global_dof_index> ids(n);

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Ke = 0; Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const Point<2>& qp  = fev.quadrature_point(q);
        const double    JxW = fev.JxW(q);

        // body force
        Tensor<1, 2> f_vec;
        f_vec[0] = fx.value(qp);
        f_vec[1] = fy.value(qp);

        for (unsigned int i = 0; i < n; ++i) {
          // symmetric strain tensor for test function i
          SymmetricTensor<2,2> eps_i = fev[displ].symmetric_gradient(i, q);
          Tensor<1,2>          vi    = fev[displ].value(i, q);

          for (unsigned int j = 0; j < n; ++j) {
            SymmetricTensor<2,2> eps_j = fev[displ].symmetric_gradient(j, q);
            // σ(u_j) : ε(v_i)
            double trace_j = eps_j[0][0] + eps_j[1][1];
            double sigma_eps = lam_ * trace_j * (eps_i[0][0] + eps_i[1][1])
                             + 2.0 * mu_ * double_contract<0,0,1,1>(eps_i, eps_j);
            Ke(i, j) += sigma_eps * JxW;
          }
          Fe(i) += vi * f_vec * JxW;
        }
      }
      cell->get_dof_indices(ids);
      cons_.distribute_local_to_global(Ke, Fe, ids, K_, rhs_);
    }
  }

  void solve() {
    ReductionControl ctrl(50000, spec_.oracle_solver.atol, spec_.oracle_solver.rtol);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(K_, 1.2);
    SolverCG<Vector<double>> cg(ctrl);
    cg.solve(K_, u_, rhs_, prec);
    cons_.distribute(u_);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: linear_elasticity_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  try { LinearElasticityOracle(read_case_spec(argv[1])).run(argv[2]); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
