#pragma once
/**
 * grid_writer.h
 *
 * Samples a scalar deal.II FE solution on a uniform 2-D grid and writes:
 *
 *   {outdir}/solution_grid.bin   – raw float64, C-order [ny, nx]
 *   {outdir}/meta.json           – nx, ny, num_dofs, baseline_time, …
 *
 * Python reads the binary as:
 *   grid = np.fromfile("solution_grid.bin", dtype=np.float64).reshape(ny, nx)
 *
 * Grid ordering convention (matches Firedrake oracle and DOLFINx oracle):
 *   grid[j, i] = u( x_lin[i], y_lin[j] )
 *   x_lin = linspace(xmin, xmax, nx)
 *   y_lin = linspace(ymin, ymax, ny)
 *   → outer loop over j (y), inner loop over i (x), i.e. row-major.
 *
 * For vector fields (Stokes velocity), a separate write_vector_grid() is
 * provided that writes |u| (magnitude) in the same format.
 */

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>

#include <nlohmann/json.hpp>

namespace oracle_util {

// ---------------------------------------------------------------------------
// Internal: build uniform (ny × nx) grid of Point<2>
// Order: row-major with j (y) outer, i (x) inner → grid[j*nx + i]
// ---------------------------------------------------------------------------
inline std::vector<dealii::Point<2>>
make_grid_points(const std::array<double, 4>& bbox, int nx, int ny) {
  const double xmin = bbox[0], xmax = bbox[1];
  const double ymin = bbox[2], ymax = bbox[3];

  std::vector<dealii::Point<2>> pts(static_cast<std::size_t>(nx) * ny);
  std::size_t idx = 0;
  for (int j = 0; j < ny; ++j) {
    double y = (ny > 1) ? ymin + j * (ymax - ymin) / (ny - 1) : 0.5 * (ymin + ymax);
    for (int i = 0; i < nx; ++i) {
      double x = (nx > 1) ? xmin + i * (xmax - xmin) / (nx - 1) : 0.5 * (xmin + xmax);
      pts[idx++] = dealii::Point<2>(x, y);
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Write meta.json
// ---------------------------------------------------------------------------
inline void write_meta(const std::string&     outdir,
                       int                    nx,
                       int                    ny,
                       std::size_t            num_dofs,
                       double                 baseline_time,
                       const std::string&     ksp_type = "",
                       const std::string&     pc_type  = "",
                       double                 rtol     = 0.0) {
  nlohmann::json meta;
  meta["nx"]            = nx;
  meta["ny"]            = ny;
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
// Sample scalar FE solution and write output files
// ---------------------------------------------------------------------------
template <int dim = 2>
void write_scalar_grid(const dealii::DoFHandler<dim>&  dof_handler,
                       const dealii::Vector<double>&   solution,
                       const std::array<double, 4>&    bbox,
                       int                             nx,
                       int                             ny,
                       const std::string&              outdir,
                       double                          baseline_time,
                       const std::string&              ksp_type = "",
                       const std::string&              pc_type  = "",
                       double                          rtol     = 0.0) {
  auto pts = make_grid_points(bbox, nx, ny);
  const std::size_t n_pts = pts.size();

  // FEFieldFunction wraps solution for arbitrary-point evaluation
  dealii::Functions::FEFieldFunction<dim> field_func(dof_handler, solution);

  std::vector<double> values(n_pts);
  field_func.value_list(pts, values);

  write_binary(outdir, values);
  write_meta(outdir, nx, ny, dof_handler.n_dofs(),
             baseline_time, ksp_type, pc_type, rtol);
}

// ---------------------------------------------------------------------------
// Sample vector FE solution: write |u| (magnitude)
// Used by Stokes and Navier-Stokes (velocity component 0..1)
// ---------------------------------------------------------------------------
template <int dim = 2>
void write_vector_magnitude_grid(const dealii::DoFHandler<dim>&  dof_handler,
                                 const dealii::Vector<double>&   solution,
                                 const std::array<double, 4>&    bbox,
                                 int                             nx,
                                 int                             ny,
                                 const std::string&              outdir,
                                 double                          baseline_time,
                                 const std::string&              ksp_type = "",
                                 const std::string&              pc_type  = "",
                                 double                          rtol     = 0.0) {
  auto pts = make_grid_points(bbox, nx, ny);
  const std::size_t n_pts = pts.size();

  dealii::Functions::FEFieldFunction<dim> field_func(dof_handler, solution);

  // FEFieldFunction::vector_value_list requires vectors sized to the full number
  // of FE components (e.g. 3 for NS/Stokes: ux, uy, p).  Using dim=2 instead
  // causes out-of-bounds writes in Release mode (assertions disabled), silently
  // corrupting the output and producing NaN magnitudes.
  const unsigned int n_comps = dof_handler.get_fe().n_components();
  std::vector<dealii::Vector<double>> vec_values(n_pts, dealii::Vector<double>(n_comps));
  field_func.vector_value_list(pts, vec_values);

  std::vector<double> magnitudes(n_pts);
  for (std::size_t k = 0; k < n_pts; ++k) {
    double mag2 = 0.0;
    for (int d = 0; d < dim; ++d)   // only the first dim components are velocity
      mag2 += vec_values[k][d] * vec_values[k][d];
    magnitudes[k] = std::sqrt(mag2);
  }

  write_binary(outdir, magnitudes);
  write_meta(outdir, nx, ny, dof_handler.n_dofs(),
             baseline_time, ksp_type, pc_type, rtol);
}

}  // namespace oracle_util
