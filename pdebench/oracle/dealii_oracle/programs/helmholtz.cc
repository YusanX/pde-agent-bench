/**
 * helmholtz.cc  –  deal.II oracle for the Helmholtz equation
 *
 *   -Δu - k² u = f   in Ω = [0,1]²
 *          u = g   on ∂Ω
 *
 * Note: For k²>0 the equation is sign-indefinite; for large k the matrix
 * is no longer positive definite and CG may fail.  We fall back to GMRES.
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
namespace { static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}}; }

class HelmholtzOracle {
 public:
  explicit HelmholtzOracle(const CaseSpec& s)
      : spec_(s), fe_(s.fem.degree), dh_(tria_) {}

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    Timer timer; timer.start();
    make_mesh(); setup_system(); assemble(); solve();
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
  SparseMatrix<double>       K_;
  Vector<double>             u_, rhs_;

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
    u_.reinit(dh_.n_dofs());
    rhs_.reinit(dh_.n_dofs());
  }

  void assemble() {
    const double k2 = std::stod(spec_.pde.value("_computed_k2", "1.0"));
    FunctionParser<2> src(1);
    src.initialize("x,y", spec_.computed_source(), MU_CONST, false);

    QGauss<2>   quad(fe_.degree + 1);
    FEValues<2> fev(fe_, quad,
                    update_values | update_gradients |
                    update_JxW_values | update_quadrature_points);

    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n);
    Vector<double>     Fe(n);
    std::vector<types::global_dof_index> ids(n);

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell); Ke = 0; Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const double f   = src.value(fev.quadrature_point(q));
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i) {
          for (unsigned int j = 0; j < n; ++j)
            Ke(i, j) += (fev.shape_grad(i,q) * fev.shape_grad(j,q)
                         - k2 * fev.shape_value(i,q) * fev.shape_value(j,q)) * JxW;
          Fe(i) += f * fev.shape_value(i,q) * JxW;
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
    // Use GMRES: Helmholtz matrix can be indefinite for large k
    SolverGMRES<Vector<double>> solver(ctrl);
    solver.solve(K_, u_, rhs_, prec);
    cons_.distribute(u_);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) { std::cerr << "Usage: helmholtz_solver <case_spec.json> <outdir>\n"; return 1; }
  try { HelmholtzOracle(read_case_spec(argv[1])).run(argv[2]); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
