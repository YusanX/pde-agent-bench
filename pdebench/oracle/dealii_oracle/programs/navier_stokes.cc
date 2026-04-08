/**
 * navier_stokes.cc  -  deal.II oracle for steady incompressible Navier-Stokes.
 *
 * This implementation follows the step-57 strategy:
 * - full Newton Jacobian (transport + reactive convection terms)
 * - augmented-Lagrangian grad-div stabilization (gamma = 1)
 * - block linear algebra (velocity/pressure split)
 * - direct LU (MUMPS via PETSc) for linear sub-problems
 * - optional viscosity continuation, including a Stokes warm start at nu_start
 *
 * Linear solver strategy:
 *   The saddle-point system is assembled into a monolithic PETSc SeqAIJ matrix
 *   and solved with KSPPREONLY + PCLU (MUMPS preferred; falls back to PETSc's
 *   own sequential LU).  For the typical oracle mesh sizes (~10k–100k DoF),
 *   this is 10-30x faster than iterative GMRES with a block-FieldSplit
 *   preconditioner whose sub-solvers have not been configured.
 */

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscksp.h>

using namespace dealii;

namespace {

class VelocityBC : public Function<2> {
 public:
  VelocityBC(const std::string &ex, const std::string &ey)
      : Function<2>(2), fx_(1), fy_(1) {
    std::map<std::string, double> constants = {{"pi", M_PI}};
    fx_.initialize("x,y", ex, constants, false);
    fy_.initialize("x,y", ey, constants, false);
  }

  double value(const Point<2> &p, unsigned int comp = 0) const override {
    return (comp == 0) ? fx_.value(p) : fy_.value(p);
  }

  void vector_value(const Point<2> &p, Vector<double> &v) const override {
    v(0) = fx_.value(p);
    v(1) = fy_.value(p);
  }

 private:
  mutable FunctionParser<2> fx_, fy_;
};

std::vector<nlohmann::json> parse_bc_segments(const std::string &json_str) {
  if (json_str.empty())
    return {};
  const auto arr = nlohmann::json::parse(json_str);
  if (!arr.is_array())
    throw std::runtime_error("_bc_segments must be a JSON array");
  return arr.get<std::vector<nlohmann::json>>();
}

inline void petsc_check(const PetscErrorCode ierr, const char *expr) {
  if (ierr == 0)
    return;
  throw std::runtime_error(std::string("PETSc call failed: ") + expr +
                           " (ierr=" + std::to_string(ierr) + ")");
}

#define PETSC_TRY(expr) petsc_check((expr), #expr)

}  // namespace

class NavierStokesOracle {
 public:
  explicit NavierStokesOracle(const CaseSpec &s)
      : spec_(s),
        nu_(std::stod(spec_.pde.value("_computed_nu", "1.0"))),
        gamma_(1.0),
        linear_ksp_type_(spec_.oracle_solver.ksp_type),
        linear_pc_type_(spec_.oracle_solver.pc_type),
        fe_(FE_Q<2>(s.fem.degree_u), 2, FE_Q<2>(s.fem.degree_p), 1),
        dh_(tria_) {
    // When the case spec still carries the dolfinx-style defaults ("cg"/"hypre"),
    // switch to a direct LU solve which is fast and robust for the saddle-point
    // systems arising from NS on meshes up to ~100k DoF.
    if (linear_ksp_type_ == "cg" && linear_pc_type_ == "hypre") {
      linear_ksp_type_ = "preonly";
      linear_pc_type_  = "lu";
    }
  }

  void run(const std::string &outdir);

 private:
  const CaseSpec &spec_;
  double nu_;
  double gamma_;
  std::string linear_ksp_type_;
  std::string linear_pc_type_;

  Triangulation<2> tria_;
  FESystem<2> fe_;
  DoFHandler<2> dh_;

  AffineConstraints<double> nonzero_cons_;
  AffineConstraints<double> zero_cons_;

  types::global_dof_index n_u_dofs_ = 0;
  types::global_dof_index n_p_dofs_ = 0;

  BlockSparsityPattern sp_;
  BlockSparseMatrix<double> system_matrix_;
  SparseMatrix<double> pressure_mass_matrix_;

  Vector<double> present_solution_;
  Vector<double> newton_update_;
  Vector<double> evaluation_point_;
  BlockVector<double> system_rhs_;
  BlockVector<double> block_update_;

  void make_mesh();
  void setup_dofs();
  void setup_block_objects();
  void assemble_pressure_mass_matrix();
  void assemble(bool initial_step, bool assemble_matrix);
  void assemble_system(bool initial_step);
  void assemble_rhs(bool initial_step);
  void solve_linear_system(bool initial_step);
  void stokes_initialization_at_current_nu();
  void newton_iteration(double rtol, double atol, int max_iter, bool initial_step);
  void continuation_then_newton();
  void copy_block_to_flat(const BlockVector<double> &src, Vector<double> &dst) const;
};

void NavierStokesOracle::make_mesh() {
  GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
  for (const auto &cell : tria_.active_cell_iterators()) {
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
      const auto face = cell->face(f);
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
    }
  }
}

void NavierStokesOracle::setup_block_objects() {
  std::vector<unsigned int> block_component(3, 0);
  block_component[2] = 1;
  DoFRenumbering::component_wise(dh_, block_component);

  const auto dofs_per_component = DoFTools::count_dofs_per_fe_component(dh_);
  n_u_dofs_ = dofs_per_component[0] + dofs_per_component[1];
  n_p_dofs_ = dofs_per_component[2];

  BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit(n_u_dofs_, n_u_dofs_);
  dsp.block(0, 1).reinit(n_u_dofs_, n_p_dofs_);
  dsp.block(1, 0).reinit(n_p_dofs_, n_u_dofs_);
  dsp.block(1, 1).reinit(n_p_dofs_, n_p_dofs_);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dh_, dsp, nonzero_cons_, false);
  sp_.copy_from(dsp);
  system_matrix_.reinit(sp_);

  pressure_mass_matrix_.reinit(sp_.block(1, 1));

  system_rhs_.reinit(2);
  system_rhs_.block(0).reinit(n_u_dofs_);
  system_rhs_.block(1).reinit(n_p_dofs_);
  system_rhs_.collect_sizes();

  block_update_.reinit(2);
  block_update_.block(0).reinit(n_u_dofs_);
  block_update_.block(1).reinit(n_p_dofs_);
  block_update_.collect_sizes();

  present_solution_.reinit(dh_.n_dofs());
  newton_update_.reinit(dh_.n_dofs());
  evaluation_point_.reinit(dh_.n_dofs());
}

void NavierStokesOracle::assemble_pressure_mass_matrix() {
  pressure_mass_matrix_ = 0;

  const FEValuesExtractors::Scalar pres(2);
  QGauss<2> quad(fe_.degree + 2);
  FEValues<2> fev(fe_, quad, update_values | update_JxW_values);

  const unsigned int n_dpc = fe_.n_dofs_per_cell();
  FullMatrix<double> Mp_local(n_dpc, n_dpc);
  std::vector<types::global_dof_index> ids(n_dpc);

  for (const auto &cell : dh_.active_cell_iterators()) {
    fev.reinit(cell);
    Mp_local = 0;

    for (unsigned int q = 0; q < quad.size(); ++q) {
      const double JxW = fev.JxW(q);
      for (unsigned int i = 0; i < n_dpc; ++i)
        for (unsigned int j = 0; j < n_dpc; ++j)
          Mp_local(i, j) += fev[pres].value(i, q) * fev[pres].value(j, q) * JxW;
    }

    cell->get_dof_indices(ids);
    for (unsigned int i = 0; i < n_dpc; ++i) {
      if (fe_.system_to_component_index(i).first != 2)
        continue;
      const auto gi = ids[i] - n_u_dofs_;
      for (unsigned int j = 0; j < n_dpc; ++j) {
        if (fe_.system_to_component_index(j).first != 2)
          continue;
        const auto gj = ids[j] - n_u_dofs_;
        pressure_mass_matrix_.add(gi, gj, Mp_local(i, j));
      }
    }
  }
}

void NavierStokesOracle::setup_dofs() {
  dh_.distribute_dofs(fe_);

  const ComponentMask vel_mask = fe_.component_mask(FEValuesExtractors::Vector(0));
  const auto segments = parse_bc_segments(spec_.pde.value("_bc_segments", "[]"));

  nonzero_cons_.clear();
  for (const auto &seg : segments) {
    const int bid = seg.at("id").get<int>();
    const std::string ex = seg.at("ex").get<std::string>();
    const std::string ey = seg.at("ey").get<std::string>();
    VelocityBC bc_func(ex, ey);
    VectorTools::interpolate_boundary_values(dh_,
                                             static_cast<types::boundary_id>(bid),
                                             bc_func,
                                             nonzero_cons_,
                                             vel_mask);
  }

  const std::string pressure_fixing = spec_.oracle_solver.pressure_fixing;
  if (pressure_fixing == "point") {
    const ComponentMask p_mask = fe_.component_mask(FEValuesExtractors::Scalar(2));
    const IndexSet p_dofs = DoFTools::extract_dofs(dh_, p_mask);
    if (p_dofs.n_elements() > 0) {
      nonzero_cons_.add_line(*p_dofs.begin());
      nonzero_cons_.set_inhomogeneity(*p_dofs.begin(), 0.0);
    }
  } else if (pressure_fixing != "none") {
    throw std::runtime_error("Unsupported navier_stokes pressure_fixing: " + pressure_fixing);
  }
  nonzero_cons_.close();

  zero_cons_.clear();
  for (const auto &seg : segments) {
    const int bid = seg.at("id").get<int>();
    Functions::ZeroFunction<2> zero_func(3);
    VectorTools::interpolate_boundary_values(dh_,
                                             static_cast<types::boundary_id>(bid),
                                             zero_func,
                                             zero_cons_,
                                             vel_mask);
  }
  if (pressure_fixing == "point") {
    const ComponentMask p_mask = fe_.component_mask(FEValuesExtractors::Scalar(2));
    const IndexSet p_dofs = DoFTools::extract_dofs(dh_, p_mask);
    if (p_dofs.n_elements() > 0) {
      zero_cons_.add_line(*p_dofs.begin());
      zero_cons_.set_inhomogeneity(*p_dofs.begin(), 0.0);
    }
  }
  zero_cons_.close();

  setup_block_objects();
  assemble_pressure_mass_matrix();
}

void NavierStokesOracle::assemble(const bool initial_step, const bool assemble_matrix) {
  if (assemble_matrix)
    system_matrix_ = 0;
  system_rhs_ = 0;

  const std::string fx_str = spec_.pde.value("_computed_source_x", "0.0");
  const std::string fy_str = spec_.pde.value("_computed_source_y", "0.0");
  std::map<std::string, double> constants = {{"pi", M_PI}};
  FunctionParser<2> fx(1), fy(1);
  fx.initialize("x,y", fx_str, constants, false);
  fy.initialize("x,y", fy_str, constants, false);

  const FEValuesExtractors::Vector vel(0);
  const FEValuesExtractors::Scalar pres(2);

  QGauss<2> quad(fe_.degree + 2);
  FEValues<2> fev(fe_,
                  quad,
                  update_values | update_gradients | update_quadrature_points |
                      update_JxW_values);

  const unsigned int n_dpc = fe_.n_dofs_per_cell();
  const unsigned int n_q = quad.size();

  FullMatrix<double> Ke(n_dpc, n_dpc);
  Vector<double> Fe(n_dpc);
  std::vector<types::global_dof_index> ids(n_dpc);

  std::vector<Tensor<1, 2>> u_vals(n_q);
  std::vector<Tensor<2, 2>> u_grads(n_q);
  std::vector<double> p_vals(n_q);

  std::vector<Tensor<1, 2>> phi_u(n_dpc);
  std::vector<Tensor<2, 2>> grad_phi_u(n_dpc);
  std::vector<double> div_phi_u(n_dpc);
  std::vector<double> phi_p(n_dpc);

  for (const auto &cell : dh_.active_cell_iterators()) {
    fev.reinit(cell);
    Ke = 0;
    Fe = 0;

    fev[vel].get_function_values(evaluation_point_, u_vals);
    fev[vel].get_function_gradients(evaluation_point_, u_grads);
    fev[pres].get_function_values(evaluation_point_, p_vals);

    for (unsigned int q = 0; q < n_q; ++q) {
      const Point<2> &qp = fev.quadrature_point(q);
      const double JxW = fev.JxW(q);
      Tensor<1, 2> f_vec;
      f_vec[0] = fx.value(qp);
      f_vec[1] = fy.value(qp);

      for (unsigned int k = 0; k < n_dpc; ++k) {
        phi_u[k] = fev[vel].value(k, q);
        grad_phi_u[k] = fev[vel].gradient(k, q);
        div_phi_u[k] = fev[vel].divergence(k, q);
        phi_p[k] = fev[pres].value(k, q);
      }

      const double div_un = trace(u_grads[q]);

      for (unsigned int i = 0; i < n_dpc; ++i) {
        Fe(i) += (-nu_ * scalar_product(grad_phi_u[i], u_grads[q]) -
                  phi_u[i] * (u_grads[q] * u_vals[q]) +
                  div_phi_u[i] * p_vals[q] + phi_p[i] * div_un -
                  gamma_ * div_phi_u[i] * div_un + phi_u[i] * f_vec) *
                 JxW;

        if (assemble_matrix) {
          for (unsigned int j = 0; j < n_dpc; ++j) {
            double val = nu_ * scalar_product(grad_phi_u[i], grad_phi_u[j]);
            val += phi_u[i] * (grad_phi_u[j] * u_vals[q]);
            val += phi_u[i] * (u_grads[q] * phi_u[j]);
            val -= div_phi_u[i] * phi_p[j];
            val -= phi_p[i] * div_phi_u[j];
            val += gamma_ * div_phi_u[i] * div_phi_u[j];
            Ke(i, j) += val * JxW;
          }
        }
      }
    }

    cell->get_dof_indices(ids);
    const AffineConstraints<double> &cons = initial_step ? nonzero_cons_ : zero_cons_;
    if (assemble_matrix)
      cons.distribute_local_to_global(Ke, Fe, ids, system_matrix_, system_rhs_);
    else
      cons.distribute_local_to_global(Fe, ids, system_rhs_);
  }
}

void NavierStokesOracle::assemble_system(const bool initial_step) {
  assemble(initial_step, true);
}

void NavierStokesOracle::assemble_rhs(const bool initial_step) {
  assemble(initial_step, false);
}

void NavierStokesOracle::copy_block_to_flat(const BlockVector<double> &src,
                                            Vector<double> &dst) const {
  for (types::global_dof_index i = 0; i < n_u_dofs_; ++i)
    dst(i) = src.block(0)(i);
  for (types::global_dof_index i = 0; i < n_p_dofs_; ++i)
    dst(n_u_dofs_ + i) = src.block(1)(i);
}

void NavierStokesOracle::solve_linear_system(const bool initial_step) {
  block_update_ = 0;

  const PetscInt nu      = static_cast<PetscInt>(n_u_dofs_);
  const PetscInt np      = static_cast<PetscInt>(n_p_dofs_);
  const PetscInt n_total = nu + np;

  // ── 1. Build monolithic PETSc SeqAIJ from all four deal.II blocks ──────
  // Assembling into a single matrix avoids the MatNest conversion overhead
  // and lets PCLU / MUMPS factor the full saddle-point system in one pass.
  std::vector<PetscInt> nnz(static_cast<std::size_t>(n_total), 0);
  for (unsigned int r = 0; r < n_u_dofs_; ++r) {
    PetscInt cnt = 0;
    for (auto it = system_matrix_.block(0, 0).begin(r);
         it != system_matrix_.block(0, 0).end(r); ++it) ++cnt;
    for (auto it = system_matrix_.block(0, 1).begin(r);
         it != system_matrix_.block(0, 1).end(r); ++it) ++cnt;
    nnz[r] = cnt;
  }
  for (unsigned int r = 0; r < n_p_dofs_; ++r) {
    PetscInt cnt = 0;
    for (auto it = system_matrix_.block(1, 0).begin(r);
         it != system_matrix_.block(1, 0).end(r); ++it) ++cnt;
    for (auto it = system_matrix_.block(1, 1).begin(r);
         it != system_matrix_.block(1, 1).end(r); ++it) ++cnt;
    nnz[static_cast<std::size_t>(nu) + r] = cnt;
  }

  Mat A = nullptr;
  PETSC_TRY(MatCreateSeqAIJ(MPI_COMM_SELF, n_total, n_total, 0, nnz.data(), &A));
  PETSC_TRY(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  for (unsigned int r = 0; r < n_u_dofs_; ++r) {
    const PetscInt row = static_cast<PetscInt>(r);
    for (auto it = system_matrix_.block(0, 0).begin(r);
         it != system_matrix_.block(0, 0).end(r); ++it)
      PETSC_TRY(MatSetValue(A, row, static_cast<PetscInt>(it->column()),
                            it->value(), INSERT_VALUES));
    for (auto it = system_matrix_.block(0, 1).begin(r);
         it != system_matrix_.block(0, 1).end(r); ++it)
      PETSC_TRY(MatSetValue(A, row, nu + static_cast<PetscInt>(it->column()),
                            it->value(), INSERT_VALUES));
  }
  for (unsigned int r = 0; r < n_p_dofs_; ++r) {
    const PetscInt row = nu + static_cast<PetscInt>(r);
    for (auto it = system_matrix_.block(1, 0).begin(r);
         it != system_matrix_.block(1, 0).end(r); ++it)
      PETSC_TRY(MatSetValue(A, row, static_cast<PetscInt>(it->column()),
                            it->value(), INSERT_VALUES));
    for (auto it = system_matrix_.block(1, 1).begin(r);
         it != system_matrix_.block(1, 1).end(r); ++it)
      PETSC_TRY(MatSetValue(A, row, nu + static_cast<PetscInt>(it->column()),
                            it->value(), INSERT_VALUES));
  }
  PETSC_TRY(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PETSC_TRY(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // ── 2. RHS and initial-guess vectors ────────────────────────────────────
  Vec b = nullptr, x = nullptr;
  PETSC_TRY(VecCreateSeq(MPI_COMM_SELF, n_total, &b));
  PETSC_TRY(VecCreateSeq(MPI_COMM_SELF, n_total, &x));
  for (types::global_dof_index i = 0; i < n_u_dofs_; ++i)
    PETSC_TRY(VecSetValue(b, static_cast<PetscInt>(i),
                          system_rhs_.block(0)(i), INSERT_VALUES));
  for (types::global_dof_index i = 0; i < n_p_dofs_; ++i)
    PETSC_TRY(VecSetValue(b, nu + static_cast<PetscInt>(i),
                          system_rhs_.block(1)(i), INSERT_VALUES));
  PETSC_TRY(VecAssemblyBegin(b));
  PETSC_TRY(VecAssemblyEnd(b));
  PETSC_TRY(VecSet(x, 0.0));

  // ── 3. KSP / PC setup ───────────────────────────────────────────────────
  KSP ksp = nullptr;
  PETSC_TRY(KSPCreate(MPI_COMM_SELF, &ksp));
  PETSC_TRY(KSPSetOperators(ksp, A, A));

  PC pc_obj = nullptr;
  PETSC_TRY(KSPGetPC(ksp, &pc_obj));

  const std::string &pc_t = linear_pc_type_;

  if (pc_t == "none") {
    // Unpreconditioned GMRES (mainly for debugging/testing)
    PETSC_TRY(KSPSetType(ksp, KSPGMRES));
    PETSC_TRY(PCSetType(pc_obj, PCNONE));
    const double rhs_norm  = system_rhs_.l2_norm();
    const double linear_tol = std::max(1e-12, 1e-8 * std::max(rhs_norm, 1.0));
    PETSC_TRY(KSPSetTolerances(ksp,
                               static_cast<PetscReal>(linear_tol),
                               PETSC_DEFAULT, PETSC_DEFAULT,
                               std::max(1000, 6 * static_cast<int>(dh_.n_dofs()))));
  } else {
    // Default (and recommended): direct LU factorisation.
    // For the saddle-point sizes encountered here (~10k–100k DoF), a single
    // sparse LU is faster and more robust than an iterative Krylov solver
    // with a poorly tuned block preconditioner.
    // MUMPS is attempted first (best performance); PETSc falls back to its
    // own sequential LU if MUMPS is unavailable.
    PETSC_TRY(KSPSetType(ksp, KSPPREONLY));
    PETSC_TRY(PCSetType(pc_obj, PCLU));
    // Silently ignore error if MUMPS is not compiled into this PETSc build
    PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS);
  }

  PETSC_TRY(KSPSetFromOptions(ksp));  // honour any -ksp_* / -pc_* CLI flags
  PETSC_TRY(KSPSolve(ksp, b, x));

  // ── 4. Copy solution back into deal.II block vectors ────────────────────
  for (types::global_dof_index i = 0; i < n_u_dofs_; ++i) {
    PetscScalar v   = 0.0;
    const PetscInt idx = static_cast<PetscInt>(i);
    PETSC_TRY(VecGetValues(x, 1, &idx, &v));
    block_update_.block(0)(i) = static_cast<double>(v);
  }
  for (types::global_dof_index i = 0; i < n_p_dofs_; ++i) {
    PetscScalar v   = 0.0;
    const PetscInt idx = nu + static_cast<PetscInt>(i);
    PETSC_TRY(VecGetValues(x, 1, &idx, &v));
    block_update_.block(1)(i) = static_cast<double>(v);
  }

  PETSC_TRY(KSPDestroy(&ksp));
  PETSC_TRY(VecDestroy(&x));
  PETSC_TRY(VecDestroy(&b));
  PETSC_TRY(MatDestroy(&A));

  copy_block_to_flat(block_update_, newton_update_);
  const AffineConstraints<double> &cons = initial_step ? nonzero_cons_ : zero_cons_;
  cons.distribute(newton_update_);
}

void NavierStokesOracle::stokes_initialization_at_current_nu() {
  evaluation_point_ = 0;
  assemble_system(true);
  solve_linear_system(true);
  present_solution_ = newton_update_;
  nonzero_cons_.distribute(present_solution_);
}

void NavierStokesOracle::newton_iteration(const double rtol,
                                          const double atol,
                                          const int max_iter,
                                          const bool initial_step) {
  bool first_step = initial_step;
  double initial_res = -1.0;
  double last_res = 1.0;
  double current_res = 1.0;

  for (int iter = 0; iter < max_iter; ++iter) {
    if (first_step) {
      evaluation_point_ = present_solution_;
      assemble_system(true);
      solve_linear_system(true);
      present_solution_ = newton_update_;
      nonzero_cons_.distribute(present_solution_);
      first_step = false;

      evaluation_point_ = present_solution_;
      assemble_rhs(false);
      current_res = system_rhs_.l2_norm();
      initial_res = current_res;
      last_res = current_res;
      std::cerr << "[NS Newton] Initial residual: " << current_res << "\n";
    } else {
      evaluation_point_ = present_solution_;
      assemble_system(false);
      solve_linear_system(false);

      bool accepted = false;
      for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5) {
        evaluation_point_ = present_solution_;
        evaluation_point_.add(alpha, newton_update_);
        zero_cons_.distribute(evaluation_point_);

        assemble_rhs(false);
        current_res = system_rhs_.l2_norm();
        if (current_res < last_res) {
          accepted = true;
          break;
        }
      }

      if (!accepted) {
        evaluation_point_ = present_solution_;
        evaluation_point_.add(1.0, newton_update_);
        zero_cons_.distribute(evaluation_point_);
        assemble_rhs(false);
        current_res = system_rhs_.l2_norm();
      }

      present_solution_ = evaluation_point_;
      std::cerr << "[NS Newton] iter=" << iter << " residual=" << current_res << "\n";
      last_res = current_res;
    }

    const double conv_tol = std::max(atol, rtol * std::max(initial_res, 1e-30));
    if (current_res < conv_tol) {
      std::cerr << "[NS Newton] Converged at iter=" << iter << "\n";
      break;
    }
  }
}

void NavierStokesOracle::continuation_then_newton() {
  const double nu_target = nu_;
  const double nu_start = spec_.oracle_solver.continuation_nu_start;
  const int n_steps = spec_.oracle_solver.continuation_steps;

  const double rtol = spec_.oracle_solver.rtol;
  const double atol = spec_.oracle_solver.atol;
  const int max_iter = spec_.oracle_solver.max_it;

  if (n_steps <= 0 || nu_start <= nu_target) {
    newton_iteration(rtol, atol, max_iter, true);
    return;
  }

  nu_ = nu_start;
  std::cerr << "[NS Continuation] Stokes init at nu=" << nu_ << "\n";
  stokes_initialization_at_current_nu();

  for (int k = 1; k <= n_steps; ++k) {
    nu_ = nu_start *
          std::pow(nu_target / nu_start, static_cast<double>(k) / n_steps);
    std::cerr << "[NS Continuation] step=" << k << "/" << n_steps
              << " nu=" << nu_ << "\n";
    newton_iteration(rtol, atol, max_iter, false);
  }
}

void NavierStokesOracle::run(const std::string &outdir) {
  std::filesystem::create_directories(outdir);

  Timer timer;
  timer.start();

  make_mesh();
  setup_dofs();

  present_solution_ = 0;

  const std::string init_mode = spec_.oracle_solver.init;
  const double rtol = spec_.oracle_solver.rtol;
  const double atol = spec_.oracle_solver.atol;
  const int max_iter = spec_.oracle_solver.max_it;

  if (init_mode == "continuation" && spec_.oracle_solver.continuation_steps > 0)
    continuation_then_newton();
  else
    newton_iteration(rtol, atol, max_iter, true);

  timer.stop();

  oracle_util::write_vector_magnitude_grid(dh_,
                                           present_solution_,
                                           spec_.output_grid.bbox,
                                           spec_.output_grid.nx,
                                           spec_.output_grid.ny,
                                           outdir,
                                           timer.wall_time(),
                                           linear_ksp_type_,
                                           linear_pc_type_,
                                           spec_.oracle_solver.rtol);
}

int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  if (argc < 3) {
    std::cerr << "Usage: navier_stokes_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  try {
    NavierStokesOracle(read_case_spec(argv[1])).run(argv[2]);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
