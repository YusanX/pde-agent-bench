/**
 * stokes.cc  –  deal.II oracle for the steady Stokes equations
 *
 *   -ν Δu + ∇p = f   in Ω = [0,1]²
 *         ∇·u = 0    in Ω
 *           u = g    on ∂Ω
 *
 * FE space: Taylor-Hood  Q2 × Q1  (inf-sup stable).
 * Solver:   SparseDirectUMFPACK (direct, handles saddle-point structure).
 * Output:   velocity magnitude ‖u‖.
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
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;
namespace { static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}}; }

// Vector BC for velocity (2 components)
class VelocityBC : public Function<2> {
 public:
  VelocityBC(const std::string& ex, const std::string& ey)
      : Function<2>(2), fx_(1), fy_(1) {
    std::map<std::string, double> c = {{"pi", M_PI}};
    fx_.initialize("x,y", ex, c, false);
    fy_.initialize("x,y", ey, c, false);
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

class StokesOracle {
 public:
  explicit StokesOracle(const CaseSpec& s)
      : spec_(s),
        fe_(FE_Q<2>(s.fem.degree_u), 2,   // velocity Q_{du}
            FE_Q<2>(s.fem.degree_p), 1),   // pressure Q_{dp}
        dh_(tria_) {
    nu_ = std::stod(spec_.pde.value("_computed_nu", "1.0"));
  }

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    Timer timer; timer.start();
    make_mesh(); setup_system(); assemble(); solve();
    timer.stop();

    // Extract velocity sub-vector for grid sampling
    // Velocity DOFs are first in component ordering
    FEValuesExtractors::Vector vel(0);
    // We write the full solution vector; FEFieldFunction will handle extraction
    oracle_util::write_vector_magnitude_grid(dh_, solution_,
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
  Vector<double>             solution_, rhs_;
  double                     nu_;

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    // Renumber DOFs: velocity first, pressure second
    DoFRenumbering::component_wise(dh_);

    cons_.clear();
    const std::string bc_x = spec_.pde.value("_computed_bc_x", "0.0");
    const std::string bc_y = spec_.pde.value("_computed_bc_y", "0.0");
    VelocityBC bc_func(bc_x, bc_y);

    // Apply BC only to velocity components (0 and 1)
    ComponentMask vel_mask(3, false);
    vel_mask.set(0, true);
    vel_mask.set(1, true);
    VectorTools::interpolate_boundary_values(dh_, 0, bc_func, cons_, vel_mask);
    cons_.close();

    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp, cons_);
    sp_.copy_from(dsp);
    K_.reinit(sp_);
    solution_.reinit(dh_.n_dofs());
    rhs_.reinit(dh_.n_dofs());
  }

  void assemble() {
    const FEValuesExtractors::Vector vel(0);
    const FEValuesExtractors::Scalar pres(2);

    const std::string fx_e = spec_.pde.value("_computed_source_x", "0.0");
    const std::string fy_e = spec_.pde.value("_computed_source_y", "0.0");
    std::map<std::string, double> c = {{"pi", M_PI}};
    FunctionParser<2> fx(1), fy(1);
    fx.initialize("x,y", fx_e, c, false);
    fy.initialize("x,y", fy_e, c, false);

    QGauss<2>   quad(fe_.degree + 1);
    FEValues<2> fev(fe_, quad,
                    update_values | update_gradients |
                    update_JxW_values | update_quadrature_points);

    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n); Vector<double> Fe(n);
    std::vector<types::global_dof_index> ids(n);

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Ke = 0; Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const Point<2>& qp  = fev.quadrature_point(q);
        const double    JxW = fev.JxW(q);
        Tensor<1,2> f_vec; f_vec[0] = fx.value(qp); f_vec[1] = fy.value(qp);

        for (unsigned int i = 0; i < n; ++i) {
          auto eps_i   = fev[vel].symmetric_gradient(i, q);
          auto div_i   = fev[vel].divergence(i, q);
          auto q_i     = fev[pres].value(i, q);
          auto vi      = fev[vel].value(i, q);

          for (unsigned int j = 0; j < n; ++j) {
            auto eps_j = fev[vel].symmetric_gradient(j, q);
            auto div_j = fev[vel].divergence(j, q);
            auto p_j   = fev[pres].value(j, q);

            // ν (ε(u):ε(v)) - p ∇·v - q ∇·u
            Ke(i,j) += (2.0 * nu_ * double_contract<0,0,1,1>(eps_i, eps_j)
                       - q_i * div_j
                       - p_j * div_i) * JxW;
          }
          Fe(i) += vi * f_vec * JxW;
        }
      }
      cell->get_dof_indices(ids);
      cons_.distribute_local_to_global(Ke, Fe, ids, K_, rhs_);
    }
  }

  void solve() {
    SparseDirectUMFPACK direct;
    direct.factorize(K_);
    direct.vmult(solution_, rhs_);
    cons_.distribute(solution_);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: stokes_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  try { StokesOracle(read_case_spec(argv[1])).run(argv[2]); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
