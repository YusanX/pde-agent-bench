/**
 * biharmonic.cc  –  deal.II oracle for the biharmonic equation
 *
 *   Δ²u = f   in Ω = [0,1]²
 *
 * Boundary conditions (simply-supported plate):
 *   u = 0   on ∂Ω
 *   Δu = 0  on ∂Ω   (equivalent to ∂²u/∂n² = 0 for axis-aligned boundaries)
 *
 * Method: Interior Penalty (C0-IP) formulation on Q2 elements.
 * References: deal.II Step-47.
 *
 * Weak form (C0-IP):
 *   a(u,v) = ∫ D²u:D²v dx
 *            + ∫_faces γ/h [∂u/∂n][∂v/∂n] ds     (penalty)
 *            - ∫_faces {{D²u·n}}·[∂v/∂n] ds       (consistency)
 *            - ∫_faces [∂u/∂n]·{{D²v·n}} ds       (symmetry)
 *
 * where γ = penalty coefficient, h = cell size.
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
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

using namespace dealii;
namespace { static const std::map<std::string, double> MU_CONST = {{"pi", M_PI}}; }

class BiharmonicOracle {
 public:
  explicit BiharmonicOracle(const CaseSpec& s)
      : spec_(s),
        fe_(std::max(s.fem.degree, 2)),  // need at least Q2 for C0-IP
        dh_(tria_) {}

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

    // u = 0 on boundary
    FunctionParser<2> bc0(1);
    bc0.initialize("x,y", spec_.computed_bc(), MU_CONST, false);
    VectorTools::interpolate_boundary_values(dh_, 0, bc0, cons_);
    cons_.close();

    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp, cons_,
                                    /*keep_constrained=*/false);
    // Also add face couplings for C0-IP
    DoFTools::make_flux_sparsity_pattern(dh_, dsp);
    sp_.copy_from(dsp);
    K_.reinit(sp_);
    u_.reinit(dh_.n_dofs());
    rhs_.reinit(dh_.n_dofs());
  }

  void assemble() {
    const double penalty = 64.0;  // γ, typical value for C0-IP on Q2

    FunctionParser<2> src(1);
    src.initialize("x,y", spec_.computed_source(), MU_CONST, false);

    QGauss<2>      cell_quad(fe_.degree + 1);
    QGauss<1>      face_quad(fe_.degree + 1);

    FEValues<2> fev(fe_, cell_quad,
                    update_values | update_gradients | update_hessians |
                    update_JxW_values | update_quadrature_points);

    FEInterfaceValues<2> fiv(fe_, face_quad,
                             update_values | update_gradients |
                             update_hessians | update_normal_vectors |
                             update_JxW_values);

    const unsigned int n   = fe_.n_dofs_per_cell();
    FullMatrix<double> Ke(n, n);
    Vector<double>     Fe(n);
    std::vector<types::global_dof_index> ids(n);

    // --- Cell contributions ---
    for (auto& cell : dh_.active_cell_iterators()) {
      fev.reinit(cell);
      Ke = 0; Fe = 0;
      for (unsigned int q = 0; q < cell_quad.size(); ++q) {
        const double f   = src.value(fev.quadrature_point(q));
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n; ++i) {
          // D²φ_i : D²φ_j  (full Hessian contraction)
          for (unsigned int j = 0; j < n; ++j) {
            double d2d2 = 0;
            for (unsigned int a = 0; a < 2; ++a)
              for (unsigned int b = 0; b < 2; ++b)
                d2d2 += fev.shape_hessian(i,q)[a][b] *
                        fev.shape_hessian(j,q)[a][b];
            Ke(i,j) += d2d2 * JxW;
          }
          Fe(i) += f * fev.shape_value(i,q) * JxW;
        }
      }
      cell->get_dof_indices(ids);
      cons_.distribute_local_to_global(Ke, Fe, ids, K_, rhs_);
    }

    // --- Interior face contributions (C0-IP penalty) ---
    for (auto& cell : dh_.active_cell_iterators()) {
      for (unsigned int face_no = 0;
           face_no < GeometryInfo<2>::faces_per_cell; ++face_no) {
        if (cell->face(face_no)->at_boundary()) continue;
        if (cell->face(face_no)->has_children()) continue;

        auto neighbor = cell->neighbor(face_no);
        if (neighbor->index() < cell->index()) continue;  // process once

        fiv.reinit(cell, face_no,
                   numbers::invalid_unsigned_int,
                   neighbor, cell->neighbor_of_neighbor(face_no),
                   numbers::invalid_unsigned_int);

        const double h   = cell->face(face_no)->measure();
        const double gam = penalty / h;

        const unsigned int n_dofs = fiv.n_current_interface_dofs();
        FullMatrix<double> Kface(n_dofs, n_dofs);
        Kface = 0;

        for (unsigned int q = 0; q < face_quad.size(); ++q) {
          const Tensor<1,2>& n_vec = fiv.normal(q);
          const double JxW = fiv.JxW(q);

          for (unsigned int i = 0; i < n_dofs; ++i) {
            for (unsigned int j = 0; j < n_dofs; ++j) {
              // ∂φ/∂n  (normal derivative jump)
              double jmp_dn_i = fiv.jump_in_shape_gradients(i, q) * n_vec;
              double jmp_dn_j = fiv.jump_in_shape_gradients(j, q) * n_vec;

              // Penalty + consistency + symmetry terms
              // scalar_product performs the full double contraction H:N for Tensor<2,2>
              // (deal.II 9.6+ removed the implicit double-contraction via operator*)
              const auto nn = outer_product(n_vec, n_vec);
              const double Hnn_i = scalar_product(fiv.average_of_shape_hessians(i,q), nn);
              const double Hnn_j = scalar_product(fiv.average_of_shape_hessians(j,q), nn);
              Kface(i,j) +=
                  (gam * jmp_dn_i * jmp_dn_j
                   - Hnn_i * jmp_dn_j
                   - jmp_dn_i * Hnn_j
                  ) * JxW;
            }
          }
        }

        std::vector<types::global_dof_index> face_ids =
            fiv.get_interface_dof_indices();
        cons_.distribute_local_to_global(Kface, face_ids, K_);
      }
    }
  }

  void solve() {
    const std::string ksp = spec_.oracle_solver.ksp_type;
    const std::string pc  = spec_.oracle_solver.pc_type;

    // For SPD biharmonic C0-IP matrices, use PETSc CG + BoomerAMG (Hypre) when
    // requested. BoomerAMG is far more efficient than UMFPACK for large systems
    // (condition number scales as h^{-4}, so AMG's O(N) complexity wins).
    if ((ksp == "cg" || ksp == "minres") &&
        (pc == "hypre" || pc == "boomeramg" || pc == "amg")) {
      solve_petsc_amg();
      return;
    }

    // Default: UMFPACK direct solver (reliable for small/medium systems)
    SparseDirectUMFPACK direct;
    direct.factorize(K_);
    direct.vmult(u_, rhs_);
    cons_.distribute(u_);
  }

  void solve_petsc_amg() {
    const unsigned int n = dh_.n_dofs();

    // Copy assembled native SparseMatrix into a serial PETSc matrix.
    // SparseMatrix(SparsityPatternType) is a templated constructor that accepts
    // deal.II's SparsityPattern and allocates a SEQAIJ PETSc matrix.
    PETScWrappers::SparseMatrix K_petsc(sp_);
    for (unsigned int row = 0; row < n; ++row)
      for (auto it = K_.begin(row); it != K_.end(row); ++it)
        K_petsc.set(row, it->column(), it->value());
    K_petsc.compress(VectorOperation::insert);

    // Serial PETSc vectors via MPI::Vector with MPI_COMM_SELF.
    // deal.II v9.7 only has PETScWrappers::MPI::Vector (no separate serial type).
    // Template constructor copies directly from a dealii::Vector<double>.
    PETScWrappers::MPI::Vector rhs_petsc(MPI_COMM_SELF, rhs_, n);
    PETScWrappers::MPI::Vector sol_petsc(MPI_COMM_SELF, n, n);

    // Solver control: honour atol and rtol from oracle_solver spec
    const double rhs_norm = rhs_petsc.l2_norm();
    const double tol = std::max(spec_.oracle_solver.atol,
                                spec_.oracle_solver.rtol * rhs_norm);
    SolverControl ctrl(spec_.oracle_solver.max_it, tol);

    // CG + BoomerAMG (symmetric_operator = true → symmetric smoothers)
    PETScWrappers::SolverCG cg(ctrl);
    PETScWrappers::PreconditionBoomerAMG amg;
    PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.symmetric_operator = true;
    amg.initialize(K_petsc, data);
    cg.solve(K_petsc, sol_petsc, rhs_petsc, amg);

    // Copy solution back to native vector for grid writing.
    // operator()(i) uses VecGetValues internally – safe in serial mode.
    for (unsigned int i = 0; i < n; ++i)
      u_(i) = sol_petsc(i);
    cons_.distribute(u_);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: biharmonic_solver <case_spec.json> <outdir>\n";
    return 1;
  }
  // PETSc (and thus BoomerAMG) requires MPI to be initialized even in serial.
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  try { BiharmonicOracle(read_case_spec(argv[1])).run(argv[2]); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
  return 0;
}
