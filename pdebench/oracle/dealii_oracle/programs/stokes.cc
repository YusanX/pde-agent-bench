/**
 * stokes.cc  –  deal.II oracle for the steady Stokes equations
 *
 *   -ν Δu + ∇p = f   in Ω = [0,1]^d
 *         ∇·u = 0    in Ω
 *           u = g    on ∂Ω
 *
 * FE space: Taylor-Hood  Qk × Q(k-1).
 * Output: velocity magnitude ‖u‖.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

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
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;

namespace {

static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}};

std::string json_to_expr(const nlohmann::json &value) {
  if (value.is_string())
    return value.get<std::string>();
  if (value.is_number())
    return std::to_string(value.get<double>());
  return "0.0";
}

template <int dim>
std::vector<std::string> json_to_vector_exprs(const nlohmann::json &value) {
  std::vector<std::string> out(dim, "0.0");
  if (value.is_array()) {
    for (unsigned int d = 0; d < dim && d < value.size(); ++d)
      out[d] = json_to_expr(value[d]);
    return out;
  }
  const std::string scalar = json_to_expr(value);
  std::fill(out.begin(), out.end(), scalar);
  return out;
}

std::vector<nlohmann::json> normalize_dirichlet_cfg(const nlohmann::json &bc) {
  if (bc.is_null() || !bc.contains("dirichlet"))
    return {};

  const auto &dirichlet = bc["dirichlet"];
  if (dirichlet.is_array()) {
    std::vector<nlohmann::json> out;
    for (const auto &cfg : dirichlet)
      out.push_back(cfg);
    return out;
  }
  if (dirichlet.is_object())
    return {dirichlet};
  return {};
}

template <int dim>
std::vector<types::boundary_id> all_boundary_ids();

template <>
std::vector<types::boundary_id> all_boundary_ids<2>() { return {0, 1, 2, 3}; }

template <>
std::vector<types::boundary_id> all_boundary_ids<3>() { return {0, 1, 2, 3, 4, 5}; }

template <int dim>
std::vector<types::boundary_id> boundary_ids_for_selector(const std::string &on) {
  std::string key = on;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (key == "all" || key == "*")
    return all_boundary_ids<dim>();
  if (key == "x0" || key == "xmin")
    return {0};
  if (key == "x1" || key == "xmax")
    return {1};
  if (key == "y0" || key == "ymin")
    return {2};
  if (key == "y1" || key == "ymax")
    return {3};
  if constexpr (dim == 3) {
    if (key == "z0" || key == "zmin")
      return {4};
    if (key == "z1" || key == "zmax")
      return {5};
  }
  throw std::runtime_error("Unknown stokes boundary selector: " + on);
}

template <int dim>
std::string coordinate_vars() {
  return dim == 2 ? "x,y" : "x,y,z";
}

template <int dim>
class VelocityBC : public Function<dim> {
 public:
  explicit VelocityBC(const std::vector<std::string>& exprs)
      : Function<dim>(dim) {
    if (exprs.size() != dim)
      throw std::runtime_error("VelocityBC expression count mismatch");
    for (unsigned int d = 0; d < dim; ++d) {
      funcs_[d] = std::make_unique<FunctionParser<dim>>(1);
      funcs_[d]->initialize(coordinate_vars<dim>(), exprs[d], MU_CONST, false);
    }
  }

  double value(const Point<dim>& p, unsigned int comp = 0) const override {
    return funcs_[comp]->value(p);
  }

  void vector_value(const Point<dim>& p, Vector<double>& v) const override {
    for (unsigned int d = 0; d < dim; ++d)
      v(d) = funcs_[d]->value(p);
  }

 private:
  std::array<std::unique_ptr<FunctionParser<dim>>, dim> funcs_;
};

}  // namespace

template <int dim>
class StokesOracle {
 public:
  explicit StokesOracle(const CaseSpec& s)
      : spec_(s),
        fe_(FE_Q<dim>(s.fem.degree_u), dim, FE_Q<dim>(s.fem.degree_p), 1),
        dh_(tria_) {
    nu_ = std::stod(spec_.pde.value("_computed_nu", "1.0"));
  }

  void run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    Timer timer;
    timer.start();
    make_mesh();
    setup_system();
    assemble();
    solve();
    timer.stop();

    oracle_util::write_vector_magnitude_grid<dim>(
        dh_, solution_, spec_.output_grid.bbox, spec_.output_grid.nx,
        spec_.output_grid.ny, spec_.output_grid.nz, outdir, timer.wall_time(),
        spec_.oracle_solver.ksp_type, spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
  }

 private:
  const CaseSpec&            spec_;
  Triangulation<dim>         tria_;
  FESystem<dim>              fe_;
  DoFHandler<dim>            dh_;
  AffineConstraints<double>  cons_;
  SparsityPattern            sp_;
  SparseMatrix<double>       K_;
  Vector<double>             solution_, rhs_;
  double                     nu_;

  void make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
    for (const auto &cell : tria_.active_cell_iterators()) {
      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        const auto face = cell->face(face_no);
        if (!face->at_boundary())
          continue;

        const auto c = face->center();
        if (std::abs(c[0] - 0.0) < 1e-12)
          face->set_boundary_id(0);
        else if (std::abs(c[0] - 1.0) < 1e-12)
          face->set_boundary_id(1);
        else if (std::abs(c[1] - 0.0) < 1e-12)
          face->set_boundary_id(2);
        else if (std::abs(c[1] - 1.0) < 1e-12)
          face->set_boundary_id(3);
        else if constexpr (dim == 3) {
          if (std::abs(c[2] - 0.0) < 1e-12)
            face->set_boundary_id(4);
          else if (std::abs(c[2] - 1.0) < 1e-12)
            face->set_boundary_id(5);
        }
      }
    }
  }

  void setup_system() {
    dh_.distribute_dofs(fe_);
    DoFRenumbering::component_wise(dh_);

    cons_.clear();
    ComponentMask vel_mask(dim + 1, false);
    for (unsigned int d = 0; d < dim; ++d)
      vel_mask.set(d, true);

    if (spec_.has_exact()) {
      std::vector<std::string> bc_exprs = {
          spec_.pde.value("_computed_bc_x", "0.0"),
          spec_.pde.value("_computed_bc_y", "0.0")};
      if constexpr (dim == 3)
        bc_exprs.push_back(spec_.pde.value("_computed_bc_z", "0.0"));
      VelocityBC<dim> bc_func(bc_exprs);
      for (const auto boundary_id : all_boundary_ids<dim>()) {
        VectorTools::interpolate_boundary_values(dh_, boundary_id, bc_func, cons_, vel_mask);
      }
    } else {
      for (const auto &cfg : normalize_dirichlet_cfg(spec_.bc)) {
        const std::string on = cfg.value("on", "all");
        const nlohmann::json value =
            cfg.contains("value") ? cfg["value"] : nlohmann::json(std::vector<std::string>(dim, "0.0"));
        VelocityBC<dim> bc_func(json_to_vector_exprs<dim>(value));
        for (const auto boundary_id : boundary_ids_for_selector<dim>(on)) {
          VectorTools::interpolate_boundary_values(dh_, boundary_id, bc_func, cons_, vel_mask);
        }
      }
    }

    const std::string pressure_fixing = spec_.oracle_solver.pressure_fixing;
    if (pressure_fixing == "point") {
      const ComponentMask pressure_mask = fe_.component_mask(FEValuesExtractors::Scalar(dim));
      const IndexSet pressure_dofs = DoFTools::extract_dofs(dh_, pressure_mask);
      for (auto it = pressure_dofs.begin(); it != pressure_dofs.end(); ++it) {
        cons_.add_line(*it);
        cons_.set_inhomogeneity(*it, 0.0);
        break;
      }
    } else if (pressure_fixing != "none") {
      throw std::runtime_error("Unsupported stokes pressure_fixing: " + pressure_fixing);
    }

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
    const FEValuesExtractors::Scalar pres(dim);

    std::vector<std::string> f_exprs = {
        spec_.pde.value("_computed_source_x", "0.0"),
        spec_.pde.value("_computed_source_y", "0.0")};
    if constexpr (dim == 3)
      f_exprs.push_back(spec_.pde.value("_computed_source_z", "0.0"));

    std::array<std::unique_ptr<FunctionParser<dim>>, dim> f_funcs;
    for (unsigned int d = 0; d < dim; ++d) {
      f_funcs[d] = std::make_unique<FunctionParser<dim>>(1);
      f_funcs[d]->initialize(coordinate_vars<dim>(), f_exprs[d], MU_CONST, false);
    }

    QGauss<dim>   quad(fe_.degree + 1);
    FEValues<dim> fev(fe_, quad,
                      update_values | update_gradients |
                      update_JxW_values | update_quadrature_points);

    const unsigned int n = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n);
    Vector<double>     Fe(n);
    std::vector<types::global_dof_index> ids(n);

    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell);
      Ke = 0;
      Fe = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        const Point<dim>& qp  = fev.quadrature_point(q);
        const double      JxW = fev.JxW(q);
        Tensor<1, dim> f_vec;
        for (unsigned int d = 0; d < dim; ++d)
          f_vec[d] = f_funcs[d]->value(qp);

        for (unsigned int i = 0; i < n; ++i) {
          const auto grad_i = fev[vel].gradient(i, q);
          const auto div_i = fev[vel].divergence(i, q);
          const auto q_i   = fev[pres].value(i, q);
          const auto vi    = fev[vel].value(i, q);

          for (unsigned int j = 0; j < n; ++j) {
            const auto grad_j = fev[vel].gradient(j, q);
            const auto div_j = fev[vel].divergence(j, q);
            const auto p_j   = fev[pres].value(j, q);

            // Match the other oracle backends and benchmark MMS definitions:
            // ν ∇u:∇v - p ∇·v - q ∇·u  ↔  -ν Δu + ∇p = f.
            Ke(i, j) += (nu_ * scalar_product(grad_i, grad_j)
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
    const std::string ksp = spec_.oracle_solver.ksp_type;
    const std::string pc  = spec_.oracle_solver.pc_type;

    if (ksp == "preonly" || pc == "lu" || pc == "mumps") {
      SparseDirectUMFPACK direct;
      direct.factorize(K_);
      direct.vmult(solution_, rhs_);
      cons_.distribute(solution_);
      return;
    }

    ReductionControl ctrl(spec_.oracle_solver.max_it,
                          spec_.oracle_solver.atol,
                          spec_.oracle_solver.rtol);

    if (ksp == "minres") {
      PreconditionIdentity prec;
      SolverMinRes<Vector<double>> minres(ctrl);
      minres.solve(K_, solution_, rhs_, prec);
    } else if (pc == "none" || pc == "hypre" || pc == "ilu") {
      PreconditionIdentity prec;
      SolverGMRES<Vector<double>> gmres(ctrl);
      gmres.solve(K_, solution_, rhs_, prec);
    } else if (pc == "jacobi") {
      PreconditionJacobi<SparseMatrix<double>> prec;
      prec.initialize(K_, 1.0);
      SolverGMRES<Vector<double>> gmres(ctrl);
      gmres.solve(K_, solution_, rhs_, prec);
    } else {
      PreconditionIdentity prec;
      SolverGMRES<Vector<double>> gmres(ctrl);
      gmres.solve(K_, solution_, rhs_, prec);
    }
    cons_.distribute(solution_);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: stokes_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  try {
    const CaseSpec spec = read_case_spec(argv[1]);
    if (spec.output_grid.is_3d() || spec.domain.type == "unit_cube")
      StokesOracle<3>(spec).run(argv[2]);
    else
      StokesOracle<2>(spec).run(argv[2]);
  }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
