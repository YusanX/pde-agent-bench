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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace dealii;

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;

    Timer timer;
    timer.start();

    // 1. Read case_spec.json
    std::ifstream spec_file(argv[1]);
    nlohmann::json spec;
    spec_file >> spec;

    const int nx = spec["output"]["grid"]["nx"].get<int>();
    const int ny = spec["output"]["grid"]["ny"].get<int>();
    const auto bbox = spec["output"]["grid"]["bbox"].get<std::vector<double>>();
    const std::string outdir(argv[2]);

    // Solver parameters
    const int mesh_resolution = 50;
    const int element_degree = 1;
    const double rtol = 1e-10;

    // 2. Build mesh, FE space, assemble, solve
    Triangulation<2> triangulation;
    GridGenerator::subdivided_hyper_cube(triangulation, mesh_resolution, 0.0, 1.0);

    FE_Q<2> fe(element_degree);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    AffineConstraints<double> constraints;
    constraints.clear();
    std::map<std::string, double> constants = {{"pi", M_PI}};
    FunctionParser<2> boundary_function(1);
    boundary_function.initialize("x,y", "sin(pi*x)*sin(pi*y)", constants, false);
    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_function, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);

    SparseMatrix<double> system_matrix;
    system_matrix.reinit(sparsity_pattern);
    Vector<double> solution(dof_handler.n_dofs());
    Vector<double> system_rhs(dof_handler.n_dofs());

    QGauss<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FunctionParser<2> source_function(1);
    source_function.initialize("x,y", "2*(pi)^2*sin(pi*x)*sin(pi*y)", constants, false);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q = 0; q < quadrature_formula.size(); ++q) {
            const double rhs_val = source_function.value(fe_values.quadrature_point(q));
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q) *
                                          fe_values.shape_grad(j, q) *
                                          fe_values.JxW(q));
                }
                cell_rhs(i) += (fe_values.shape_value(i, q) *
                                rhs_val *
                                fe_values.JxW(q));
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

    ReductionControl solver_control(5000, 1e-12, rtol);
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    // 3. Sample solution on uniform grid
    Functions::FEFieldFunction<2> field_function(dof_handler, solution);
    std::vector<Point<2>> points(nx * ny);
    std::vector<double> values(nx * ny);

    for (int j = 0; j < ny; ++j) {
        double y = bbox[2] + j * (bbox[3] - bbox[2]) / (ny - 1);
        for (int i = 0; i < nx; ++i) {
            double x = bbox[0] + i * (bbox[1] - bbox[0]) / (nx - 1);
            points[j * nx + i] = Point<2>(x, y);
        }
    }
    field_function.value_list(points, values);

    // 4. Write output files
    std::ofstream bin_file(outdir + "/solution_grid.bin", std::ios::binary);
    bin_file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
    bin_file.close();

    timer.stop();

    nlohmann::json meta;
    meta["nx"] = nx;
    meta["ny"] = ny;
    meta["wall_time_sec"] = timer.wall_time();
    meta["solver_info"] = {
        {"mesh_resolution", mesh_resolution},
        {"element_degree", element_degree},
        {"ksp_type", "cg"},
        {"pc_type", "ssor"},
        {"rtol", rtol}
    };

    std::ofstream meta_file(outdir + "/meta.json");
    meta_file << meta.dump(2) << std::endl;

    return 0;
}