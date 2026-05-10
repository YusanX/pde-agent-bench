#pragma once
/**
 * mesh_factory.h
 *
 * Provides two utilities used by every deal.II oracle solver:
 *
 *   oracle_util::make_mesh<dim>(spec, tria)
 *     – Populates `tria` from the CaseSpec.
 *     – If spec.domain.mesh_file is non-empty: reads a Gmsh .msh file
 *       written by the Python `generate_domain_mesh_file()` helper and
 *       resets ALL boundary face IDs to 0 so that existing BC code
 *       (which always targets boundary_id = 0) works unchanged.
 *     – Otherwise: uses GridGenerator::subdivided_hyper_cube (unit square
 *       or cube), producing a uniform quadrilateral mesh.
 *     – Returns true  if the mesh contains simplex cells (triangles/tets).
 *     – Returns false if the mesh contains hypercube cells (quads/hexes).
 *
 *   oracle_util::make_quadrature<dim>(fe, n_points)
 *     – Returns a Quadrature<dim> appropriate for the FE reference cell:
 *       QGaussSimplex for simplices, QGauss for hypercubes.
 *
 * Both FE_Q (quadrilateral meshes) and FE_SimplexP (triangular meshes) are
 * supported by deal.II ≥ 9.3.
 */

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include "case_spec_reader.h"

namespace oracle_util {

// ---------------------------------------------------------------------------
// make_mesh<dim>
// ---------------------------------------------------------------------------
template <int dim>
bool make_mesh(const CaseSpec& spec, dealii::Triangulation<dim>& tria) {
  if (!spec.domain.mesh_file.empty()) {
    // ---- Read external Gmsh mesh (.msh) -----------------------------------
    dealii::GridIn<dim> grid_in;
    grid_in.attach_triangulation(tria);
    std::ifstream f(spec.domain.mesh_file);
    if (!f.is_open())
      throw std::runtime_error(
          "mesh_factory: cannot open mesh file: " + spec.domain.mesh_file);
    grid_in.read_msh(f);

    // Reset all boundary face IDs to 0.
    // Solvers apply BCs on boundary_id = 0; this ensures that convention
    // holds regardless of what physical group tags Gmsh assigned.
    for (auto& face : tria.active_face_iterators())
      if (face->at_boundary())
        face->set_boundary_id(0);

    // Detect cell type from the first active cell.
    return tria.begin_active()->reference_cell().is_simplex();
  }

  // ---- Built-in generator: unit square / cube ----------------------------
  dealii::GridGenerator::subdivided_hyper_cube(
      tria, spec.mesh.resolution, 0.0, 1.0);
  return false;  // hypercube cells → use FE_Q
}

// ---------------------------------------------------------------------------
// make_quadrature<dim>
// ---------------------------------------------------------------------------
template <int dim>
dealii::Quadrature<dim> make_quadrature(
    const dealii::FiniteElement<dim>& fe,
    int n_points = -1) {
  if (n_points < 0)
    n_points = static_cast<int>(fe.degree) + 1;
  if (fe.reference_cell().is_simplex())
    return dealii::QGaussSimplex<dim>(n_points);
  else
    return dealii::QGauss<dim>(n_points);
}

// ---------------------------------------------------------------------------
// make_face_quadrature<dim>   (needed by biharmonic interface terms)
// ---------------------------------------------------------------------------
template <int dim>
dealii::Quadrature<dim - 1> make_face_quadrature(
    const dealii::FiniteElement<dim>& fe,
    int n_points = -1) {
  if (n_points < 0)
    n_points = static_cast<int>(fe.degree) + 1;
  if (fe.reference_cell().is_simplex())
    return dealii::QGaussSimplex<dim - 1>(n_points);
  else
    return dealii::QGauss<dim - 1>(n_points);
}

// ---------------------------------------------------------------------------
// make_scalar_fe<dim>   – FE_Q or FE_SimplexP, degree given by spec
// ---------------------------------------------------------------------------
template <int dim>
std::unique_ptr<dealii::FiniteElement<dim>>
make_scalar_fe(int degree, bool is_simplex) {
  if (is_simplex)
    return std::make_unique<dealii::FE_SimplexP<dim>>(degree);
  else
    return std::make_unique<dealii::FE_Q<dim>>(degree);
}

// ---------------------------------------------------------------------------
// make_vector_fe<dim>   – FESystem wrapping dim copies of a scalar FE
// ---------------------------------------------------------------------------
template <int dim>
std::unique_ptr<dealii::FiniteElement<dim>>
make_vector_fe(int degree, bool is_simplex) {
  if (is_simplex)
    return std::make_unique<dealii::FESystem<dim>>(
        dealii::FE_SimplexP<dim>(degree), dim);
  else
    return std::make_unique<dealii::FESystem<dim>>(
        dealii::FE_Q<dim>(degree), dim);
}

// ---------------------------------------------------------------------------
// make_mixed_fe<dim>    – Taylor-Hood FESystem: vel (deg_u) × dim + pres (deg_p)
// ---------------------------------------------------------------------------
template <int dim>
std::unique_ptr<dealii::FiniteElement<dim>>
make_mixed_fe(int deg_u, int deg_p, bool is_simplex) {
  if (is_simplex)
    return std::make_unique<dealii::FESystem<dim>>(
        dealii::FE_SimplexP<dim>(deg_u), dim,
        dealii::FE_SimplexP<dim>(deg_p), 1);
  else
    return std::make_unique<dealii::FESystem<dim>>(
        dealii::FE_Q<dim>(deg_u), dim,
        dealii::FE_Q<dim>(deg_p), 1);
}

}  // namespace oracle_util
