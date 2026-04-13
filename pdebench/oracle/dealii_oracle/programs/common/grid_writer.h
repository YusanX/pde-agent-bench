#pragma once
/**
 * grid_writer.h
 *
 * Samples a scalar or vector deal.II FE solution on a uniform 2-D or 3-D
 * output grid and writes:
 *
 *   {outdir}/solution_grid.bin   – raw float64, C-order [ny, nx] or [nz, ny, nx]
 *   {outdir}/meta.json           – nx, ny, [nz], num_dofs, baseline_time, …
 *
 * Grid ordering convention (matches Firedrake oracle and DOLFINx oracle):
 *   grid[j, i] = u( x_lin[i], y_lin[j] )
 *
 * Evaluation strategy (chosen at run-time based on mesh cell type):
 *
 *   Hypercube meshes (FE_Q, unit square/cube):
 *     Uses FEFieldFunction::value_list for fast batch evaluation.
 *     All output grid points lie inside the bbox = domain, so no exception.
 *
 *   Simplex meshes (FE_SimplexP, complex 2-D domains from Gmsh):
 *     Uses VectorTools::point_value with MappingFE (correct simplex mapping)
 *     in a per-point loop with try-catch.
 *     Points outside the domain (bbox corners outside the L-shape, T-shape,
 *     etc.) catch the exception and are set to 0.0, matching the DOLFINx
 *     oracle's nan_to_num convention.
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <nlohmann/json.hpp>

namespace oracle_util {

// ---------------------------------------------------------------------------
// Internal: build a uniform evaluation grid.
// ---------------------------------------------------------------------------
template <int dim>
std::vector<dealii::Point<dim>>
make_grid_points(const std::vector<double>& bbox, int nx, int ny, int nz = 0) {
  static_assert(dim == 2 || dim == 3, "Only 2-D and 3-D grids are supported");

  if constexpr (dim == 2) {
    if (bbox.size() != 4)
      throw std::runtime_error("2-D grid writer expects bbox size 4");
    const double xmin = bbox[0], xmax = bbox[1];
    const double ymin = bbox[2], ymax = bbox[3];
    std::vector<dealii::Point<dim>> pts(static_cast<std::size_t>(nx) * ny);
    std::size_t idx = 0;
    for (int j = 0; j < ny; ++j) {
      const double y = (ny > 1) ? ymin + j * (ymax - ymin) / (ny - 1)
                                : 0.5 * (ymin + ymax);
      for (int i = 0; i < nx; ++i) {
        const double x = (nx > 1) ? xmin + i * (xmax - xmin) / (nx - 1)
                                  : 0.5 * (xmin + xmax);
        pts[idx++] = dealii::Point<dim>(x, y);
      }
    }
    return pts;
  } else {
    if (bbox.size() != 6)
      throw std::runtime_error("3-D grid writer expects bbox size 6");
    if (nz <= 0)
      throw std::runtime_error("3-D grid writer expects nz > 0");
    const double xmin = bbox[0], xmax = bbox[1];
    const double ymin = bbox[2], ymax = bbox[3];
    const double zmin = bbox[4], zmax = bbox[5];
    std::vector<dealii::Point<dim>> pts(static_cast<std::size_t>(nx) * ny * nz);
    std::size_t idx = 0;
    for (int k = 0; k < nz; ++k) {
      const double z = (nz > 1) ? zmin + k * (zmax - zmin) / (nz - 1)
                                : 0.5 * (zmin + zmax);
      for (int j = 0; j < ny; ++j) {
        const double y = (ny > 1) ? ymin + j * (ymax - ymin) / (ny - 1)
                                  : 0.5 * (ymin + ymax);
        for (int i = 0; i < nx; ++i) {
          const double x = (nx > 1) ? xmin + i * (xmax - xmin) / (nx - 1)
                                    : 0.5 * (xmin + xmax);
          pts[idx++] = dealii::Point<dim>(x, y, z);
        }
      }
    }
    return pts;
  }
}

// ---------------------------------------------------------------------------
// Write meta.json
// ---------------------------------------------------------------------------
inline void write_meta(const std::string& outdir,
                       int                nx,
                       int                ny,
                       int                nz,
                       std::size_t        num_dofs,
                       double             baseline_time,
                       const std::string& ksp_type = "",
                       const std::string& pc_type  = "",
                       double             rtol      = 0.0) {
  nlohmann::json meta;
  meta["nx"]            = nx;
  meta["ny"]            = ny;
  if (nz > 0) meta["nz"] = nz;
  meta["num_dofs"]      = num_dofs;
  meta["baseline_time"] = baseline_time;
  meta["ksp_type"]      = ksp_type;
  meta["pc_type"]       = pc_type;
  meta["rtol"]          = rtol;
  std::ofstream f(outdir + "/meta.json");
  if (!f.is_open())
    throw std::runtime_error("Cannot write meta.json to: " + outdir);
  f << meta.dump(2) << "\n";
}

// ---------------------------------------------------------------------------
// Write raw binary float64 array
// ---------------------------------------------------------------------------
inline void write_binary(const std::string&         outdir,
                         const std::vector<double>& values) {
  std::ofstream f(outdir + "/solution_grid.bin", std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Cannot write solution_grid.bin to: " + outdir);
  f.write(reinterpret_cast<const char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(double)));
}

// ---------------------------------------------------------------------------
// Internal: detect if the triangulation uses simplex cells (triangles/tets).
// ---------------------------------------------------------------------------
template <int dim>
bool tria_is_simplex(const dealii::DoFHandler<dim>& dh) {
  if (dh.get_triangulation().n_active_cells() == 0) return false;
  return dh.get_triangulation().begin_active()->reference_cell().is_simplex();
}

// ---------------------------------------------------------------------------
// eval_scalar_at_points
//
// Hypercube mesh: FEFieldFunction batch evaluation (fast, no exception for
//   unit square/cube where all bbox points are inside the domain).
// Simplex mesh  : VectorTools::point_value with MappingFE (correct simplex
//   reference-to-physical mapping), per-point with try-catch so that bbox
//   corners outside the complex domain are silently set to 0.
// ---------------------------------------------------------------------------
template <int dim>
std::vector<double>
eval_scalar_at_points(const dealii::DoFHandler<dim>&         dof_handler,
                      const dealii::Vector<double>&           solution,
                      const std::vector<dealii::Point<dim>>& pts) {
  const std::size_t n_pts = pts.size();
  std::vector<double> values(n_pts, 0.0);

  if (!tria_is_simplex<dim>(dof_handler)) {
    // ---- Hypercube path (original approach, unchanged) --------------------
    dealii::Functions::FEFieldFunction<dim> field_func(dof_handler, solution);
    field_func.value_list(pts, values);
  } else {
    // ---- Simplex path: per-point with correct mapping + exception guard ----
    // MappingFE with P1 basis correctly handles the affine reference-to-
    // physical mapping for straight-sided triangles loaded from Gmsh.
    dealii::MappingFE<dim> mapping(dealii::FE_SimplexP<dim>(1));
    for (std::size_t i = 0; i < n_pts; ++i) {
      try {
        values[i] = dealii::VectorTools::point_value(
            mapping, dof_handler, solution, pts[i]);
      } catch (...) {
        values[i] = 0.0;  // point is outside the complex domain → 0
      }
    }
  }

  return values;
}

// ---------------------------------------------------------------------------
// eval_vector_at_points
//
// Same dual strategy for vector-valued FE solutions (Stokes, linear elast.).
// Returns a flat array of Vector<double> of length n_pts; each has n_comps
// entries.  Points outside the domain → zero vector.
// ---------------------------------------------------------------------------
template <int dim>
std::vector<dealii::Vector<double>>
eval_vector_at_points(const dealii::DoFHandler<dim>&         dof_handler,
                      const dealii::Vector<double>&           solution,
                      const std::vector<dealii::Point<dim>>& pts) {
  const std::size_t  n_pts   = pts.size();
  const unsigned int n_comps = dof_handler.get_fe().n_components();
  std::vector<dealii::Vector<double>> values(
      n_pts, dealii::Vector<double>(n_comps));

  if (!tria_is_simplex<dim>(dof_handler)) {
    // ---- Hypercube path ---------------------------------------------------
    dealii::Functions::FEFieldFunction<dim> field_func(dof_handler, solution);
    field_func.vector_value_list(pts, values);
  } else {
    // ---- Simplex path -----------------------------------------------------
    dealii::MappingFE<dim> mapping(dealii::FE_SimplexP<dim>(1));
    for (std::size_t i = 0; i < n_pts; ++i) {
      try {
        dealii::VectorTools::point_value(
            mapping, dof_handler, solution, pts[i], values[i]);
      } catch (...) {
        values[i] = 0.0;
      }
    }
  }

  return values;
}

// ---------------------------------------------------------------------------
// write_scalar_grid  (with nz for 3-D)
// ---------------------------------------------------------------------------
template <int dim = 2>
void write_scalar_grid(const dealii::DoFHandler<dim>&  dof_handler,
                       const dealii::Vector<double>&   solution,
                       const std::vector<double>&      bbox,
                       int                             nx,
                       int                             ny,
                       int                             nz,
                       const std::string&              outdir,
                       double                          baseline_time,
                       const std::string&              ksp_type = "",
                       const std::string&              pc_type  = "",
                       double                          rtol     = 0.0) {
  auto pts    = make_grid_points<dim>(bbox, nx, ny, nz);
  auto values = eval_scalar_at_points<dim>(dof_handler, solution, pts);
  write_binary(outdir, values);
  write_meta(outdir, nx, ny, dim == 3 ? nz : 0, dof_handler.n_dofs(),
             baseline_time, ksp_type, pc_type, rtol);
}

// 2-D convenience overload (no nz)
template <int dim = 2>
void write_scalar_grid(const dealii::DoFHandler<dim>&  dof_handler,
                       const dealii::Vector<double>&   solution,
                       const std::vector<double>&      bbox,
                       int                             nx,
                       int                             ny,
                       const std::string&              outdir,
                       double                          baseline_time,
                       const std::string&              ksp_type = "",
                       const std::string&              pc_type  = "",
                       double                          rtol     = 0.0) {
  write_scalar_grid<dim>(dof_handler, solution, bbox, nx, ny, 0,
                         outdir, baseline_time, ksp_type, pc_type, rtol);
}

// ---------------------------------------------------------------------------
// write_vector_magnitude_grid  (Stokes / Navier-Stokes / linear_elasticity)
// ---------------------------------------------------------------------------
template <int dim = 2>
void write_vector_magnitude_grid(const dealii::DoFHandler<dim>&  dof_handler,
                                 const dealii::Vector<double>&   solution,
                                 const std::vector<double>&      bbox,
                                 int                             nx,
                                 int                             ny,
                                 int                             nz,
                                 const std::string&              outdir,
                                 double                          baseline_time,
                                 const std::string&              ksp_type = "",
                                 const std::string&              pc_type  = "",
                                 double                          rtol     = 0.0) {
  auto pts      = make_grid_points<dim>(bbox, nx, ny, nz);
  auto vec_vals = eval_vector_at_points<dim>(dof_handler, solution, pts);

  const std::size_t n_pts = pts.size();
  std::vector<double> magnitudes(n_pts, 0.0);
  for (std::size_t k = 0; k < n_pts; ++k) {
    double mag2 = 0.0;
    for (int d = 0; d < dim; ++d)
      mag2 += vec_vals[k][d] * vec_vals[k][d];
    magnitudes[k] = std::sqrt(mag2);
  }

  write_binary(outdir, magnitudes);
  write_meta(outdir, nx, ny, dim == 3 ? nz : 0, dof_handler.n_dofs(),
             baseline_time, ksp_type, pc_type, rtol);
}

// 2-D convenience overload
template <int dim = 2>
void write_vector_magnitude_grid(const dealii::DoFHandler<dim>&  dof_handler,
                                 const dealii::Vector<double>&   solution,
                                 const std::vector<double>&      bbox,
                                 int                             nx,
                                 int                             ny,
                                 const std::string&              outdir,
                                 double                          baseline_time,
                                 const std::string&              ksp_type = "",
                                 const std::string&              pc_type  = "",
                                 double                          rtol     = 0.0) {
  write_vector_magnitude_grid<dim>(dof_handler, solution, bbox, nx, ny, 0,
                                   outdir, baseline_time, ksp_type, pc_type, rtol);
}

}  // namespace oracle_util
