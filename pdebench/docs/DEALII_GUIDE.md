# deal.II 9.x C++ API Reference Guide

This guide covers the deal.II patterns you need to solve 2-D scalar/vector PDEs
on the unit square [0,1]² and write output in the required format.

---

## 1. Required includes

```cpp
#include <deal.II/base/function_parser.h>   // FunctionParser<2>
#include <deal.II/base/quadrature_lib.h>    // QGauss<2>
#include <deal.II/base/timer.h>             // Timer
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>                // FE_Q<2>  – scalar Lagrange
#include <deal.II/fe/fe_system.h>           // FESystem<2> – vector/mixed
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>     // ReductionControl, SolverControl
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>      // SparseDirectUMFPACK
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>
#include <nlohmann/json.hpp>

using namespace dealii;
```

---

## 2. Mesh creation

```cpp
Triangulation<2> tria;

// subdivided_hyper_cube: nx×nx quad cells in [0,1]²  (recommended)
GridGenerator::subdivided_hyper_cube(tria, /*repetitions=*/nx, 0.0, 1.0);

// All four sides share boundary_id = 0 by default.
```

---

## 3. FE spaces

```cpp
// Scalar (Poisson, heat, Helmholtz, …)
FE_Q<2>       fe(degree);   // degree = 1, 2, 3

// Vector (linear elasticity)
FESystem<2>   fe(FE_Q<2>(degree), 2);   // 2-component displacement

// Mixed Taylor-Hood (Stokes)
FESystem<2>   fe(FE_Q<2>(2), 2,   // velocity  Q2
                 FE_Q<2>(1), 1);  // pressure  Q1
```

---

## 4. DOF distribution and sparsity

```cpp
DoFHandler<2> dof_handler(tria);
dof_handler.distribute_dofs(fe);

DynamicSparsityPattern dsp(dof_handler.n_dofs());
DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
SparsityPattern sparsity_pattern;
sparsity_pattern.copy_from(dsp);

SparseMatrix<double> system_matrix;
system_matrix.reinit(sparsity_pattern);

Vector<double> solution(dof_handler.n_dofs());
Vector<double> system_rhs(dof_handler.n_dofs());
```

---

## 5. Expression parsing with FunctionParser

FunctionParser uses **muParser** syntax: `^` for power, `pi` as constant.

```cpp
std::map<std::string, double> constants = {{"pi", M_PI}};

// Scalar function f(x,y)
FunctionParser<2> source(1);
source.initialize("x,y", "2*(pi)^(2)*sin(pi*x)*sin(pi*y)", constants, false);

// Time-dependent f(x,y,t)
FunctionParser<2> td_func(1);
td_func.initialize("x,y,t", "exp(-t)*sin(pi*x)*sin(pi*y)", constants, true);
td_func.set_time(0.5);   // set current time before evaluation

// Evaluate at a point
Point<2> p(0.5, 0.5);
double val = source.value(p);
```

---

## 6. Dirichlet boundary conditions

```cpp
AffineConstraints<double> constraints;
constraints.clear();

FunctionParser<2> bc_func(1);
bc_func.initialize("x,y", bc_expression_string, constants, false);

// Boundary ID 0 = all four sides of subdivided_hyper_cube
VectorTools::interpolate_boundary_values(
    dof_handler, 0, bc_func, constraints);
constraints.close();
```

---

## 7. Cell-by-cell assembly (scalar PDE)

```cpp
QGauss<2>  quadrature(fe.degree + 1);
FEValues<2> fev(fe, quadrature,
                update_values | update_gradients |
                update_JxW_values | update_quadrature_points);

const unsigned int n_dpc = fe.n_dofs_per_cell();
FullMatrix<double> cell_mat(n_dpc, n_dpc);
Vector<double>     cell_rhs(n_dpc);
std::vector<types::global_dof_index> dof_ids(n_dpc);

for (auto& cell : dof_handler.active_cell_iterators()) {
    fev.reinit(cell);
    cell_mat = 0;
    cell_rhs = 0;
    for (unsigned int q = 0; q < quadrature.size(); ++q) {
        const Point<2>& qp  = fev.quadrature_point(q);
        const double    kap = kappa_func.value(qp);
        const double    f   = source_func.value(qp);
        const double    JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n_dpc; ++i) {
            for (unsigned int j = 0; j < n_dpc; ++j)
                cell_mat(i,j) += kap * fev.shape_grad(i,q) *
                                        fev.shape_grad(j,q) * JxW;
            cell_rhs(i) += f * fev.shape_value(i,q) * JxW;
        }
    }
    cell->get_dof_indices(dof_ids);
    constraints.distribute_local_to_global(
        cell_mat, cell_rhs, dof_ids, system_matrix, system_rhs);
}
```

---

## 8. Linear solvers

```cpp
// CG with SSOR preconditioner (symmetric positive definite)
ReductionControl control(50000, 1e-12, 1e-10);
PreconditionSSOR<SparseMatrix<double>> prec;
prec.initialize(system_matrix, 1.2);
SolverCG<Vector<double>> cg(control);
cg.solve(system_matrix, solution, system_rhs, prec);
constraints.distribute(solution);

// GMRES (non-symmetric, e.g. convection-diffusion)
SolverGMRES<Vector<double>> gmres(control);
gmres.solve(system_matrix, solution, system_rhs, prec);
constraints.distribute(solution);

// Direct solver (Stokes, Navier-Stokes)
SparseDirectUMFPACK direct;
direct.factorize(system_matrix);
direct.vmult(solution, system_rhs);
constraints.distribute(solution);
```

---

## 9. Mass matrix assembly (time-stepping)

```cpp
SparseMatrix<double> mass_matrix;
mass_matrix.reinit(sparsity_pattern);

// MatrixCreator::create_mass_matrix includes boundary contributions;
// use cell loop for consistency with constraint application.
for (auto& cell : dof_handler.active_cell_iterators()) {
    fev.reinit(cell);
    cell_mat = 0;
    for (unsigned int q = 0; q < quadrature.size(); ++q) {
        const double JxW = fev.JxW(q);
        for (unsigned int i = 0; i < n_dpc; ++i)
            for (unsigned int j = 0; j < n_dpc; ++j)
                cell_mat(i,j) += fev.shape_value(i,q) *
                                  fev.shape_value(j,q) * JxW;
    }
    cell->get_dof_indices(dof_ids);
    constraints.distribute_local_to_global(cell_mat, dof_ids, mass_matrix);
}
```

---

## 10. Grid sampling (required output step)

```cpp
#include <deal.II/numerics/fe_field_function.h>

// IMPORTANT: deal.II 9.x API — template args are <dim, VectorType, spacedim>.
// Use the 1-argument form below. The OLD form <dim, DoFHandler<dim>, VectorType>
// no longer compiles and must NOT be used.
Functions::FEFieldFunction<2> field_func(dof_handler, solution);

int nx = ..., ny = ...;
double xmin = ..., xmax = ..., ymin = ..., ymax = ...;

std::vector<Point<2>> pts(nx * ny);
int idx = 0;
for (int j = 0; j < ny; ++j) {          // outer loop = y
    double y = ymin + j * (ymax - ymin) / (ny - 1);
    for (int i = 0; i < nx; ++i) {      // inner loop = x
        double x = xmin + i * (xmax - xmin) / (nx - 1);
        pts[idx++] = Point<2>(x, y);
    }
}
std::vector<double> vals(nx * ny);
field_func.value_list(pts, vals);        // fills vals in input order

// Write raw binary (float64, row-major [ny, nx])
std::ofstream bin(outdir + "/solution_grid.bin", std::ios::binary);
bin.write(reinterpret_cast<const char*>(vals.data()),
          vals.size() * sizeof(double));
```

---

## 11. Write meta.json

```cpp
#include <nlohmann/json.hpp>

nlohmann::json meta;
meta["nx"]           = nx;
meta["ny"]           = ny;
meta["wall_time_sec"]= elapsed_seconds;
meta["solver_info"]  = {
    {"mesh_resolution", resolution},
    {"element_degree",  degree},
    {"ksp_type",        "cg"},
    {"pc_type",         "ssor"},
    {"rtol",            1e-10}
};

std::ofstream mf(outdir + "/meta.json");
mf << meta.dump(2) << "\n";
```

---

## 12. Reading case_spec.json

```cpp
#include <nlohmann/json.hpp>
#include <fstream>

std::ifstream f(argv[1]);
nlohmann::json spec;
f >> spec;

int    resolution = spec["mesh"]["resolution"].get<int>();
int    degree     = spec["fem"]["degree"].get<int>();
double kappa      = spec["pde"]["coefficients"]["kappa"]["value"].get<double>();
int    nx         = spec["output"]["grid"]["nx"].get<int>();
int    ny         = spec["output"]["grid"]["ny"].get<int>();
auto   bbox       = spec["output"]["grid"]["bbox"].get<std::vector<double>>();
// bbox = [xmin, xmax, ymin, ymax]
```

---

## 13. Complete minimal Poisson example

```cpp
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
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
#include <filesystem>

using namespace dealii;

int main(int argc, char* argv[]) {
    std::ifstream f(argv[1]);
    nlohmann::json spec; f >> spec;

    const int    res  = spec["mesh"]["resolution"].get<int>();
    const int    deg  = spec["fem"]["degree"].get<int>();
    const int    nx   = spec["output"]["grid"]["nx"].get<int>();
    const int    ny   = spec["output"]["grid"]["ny"].get<int>();
    auto         bbox = spec["output"]["grid"]["bbox"].get<std::vector<double>>();
    const std::string outdir(argv[2]);

    Triangulation<2> tria;
    GridGenerator::subdivided_hyper_cube(tria, res, 0.0, 1.0);
    FE_Q<2>       fe(deg);
    DoFHandler<2> dh(tria);
    dh.distribute_dofs(fe);

    AffineConstraints<double> cons;
    cons.clear();
    std::map<std::string, double> consts = {{"pi", M_PI}};
    FunctionParser<2> bc(1);
    bc.initialize("x,y", "sin(pi*x)*sin(pi*y)", consts, false);
    VectorTools::interpolate_boundary_values(dh, 0, bc, cons);
    cons.close();

    DynamicSparsityPattern dsp(dh.n_dofs());
    DoFTools::make_sparsity_pattern(dh, dsp, cons);
    SparsityPattern sp; sp.copy_from(dsp);
    SparseMatrix<double> K; K.reinit(sp);
    Vector<double> u(dh.n_dofs()), rhs(dh.n_dofs());

    QGauss<2> quad(deg + 1);
    FEValues<2> fev(fe, quad, update_values|update_gradients|update_JxW_values|update_quadrature_points);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> Ke(n,n); Vector<double> Fe(n);
    std::vector<types::global_dof_index> ids(n);
    FunctionParser<2> src(1);
    src.initialize("x,y", "2*(pi)^(2)*sin(pi*x)*sin(pi*y)", consts, false);

    for (auto& cell : dh.active_cell_iterators()) {
        fev.reinit(cell); Ke=0; Fe=0;
        for (unsigned int q=0; q<quad.size(); ++q) {
            double f=src.value(fev.quadrature_point(q)), JxW=fev.JxW(q);
            for (unsigned int i=0;i<n;++i) {
                for (unsigned int j=0;j<n;++j)
                    Ke(i,j)+=fev.shape_grad(i,q)*fev.shape_grad(j,q)*JxW;
                Fe(i)+=f*fev.shape_value(i,q)*JxW;
            }
        }
        cell->get_dof_indices(ids);
        cons.distribute_local_to_global(Ke,Fe,ids,K,rhs);
    }

    ReductionControl ctrl(50000,1e-12,1e-10);
    PreconditionSSOR<SparseMatrix<double>> prec; prec.initialize(K,1.2);
    SolverCG<Vector<double>> cg(ctrl);
    cg.solve(K,u,rhs,prec); cons.distribute(u);

    Functions::FEFieldFunction<2> ff(dh, u);
    std::vector<Point<2>> pts(nx*ny);
    int idx=0;
    for(int j=0;j<ny;++j){
        double y=bbox[2]+j*(bbox[3]-bbox[2])/(ny-1);
        for(int i=0;i<nx;++i){
            double x=bbox[0]+i*(bbox[1]-bbox[0])/(nx-1);
            pts[idx++]=Point<2>(x,y);
        }
    }
    std::vector<double> vals(nx*ny);
    ff.value_list(pts,vals);

    std::filesystem::create_directories(outdir);
    std::ofstream bin(outdir+"/solution_grid.bin",std::ios::binary);
    bin.write(reinterpret_cast<const char*>(vals.data()),vals.size()*sizeof(double));

    nlohmann::json meta;
    meta["nx"]=nx; meta["ny"]=ny; meta["wall_time_sec"]=0.0;
    meta["solver_info"]={{"mesh_resolution",res},{"element_degree",deg},
                         {"ksp_type","cg"},{"pc_type","ssor"},{"rtol",1e-10}};
    std::ofstream mf(outdir+"/meta.json"); mf<<meta.dump(2)<<"\n";
    return 0;
}
```

---

## 14. Mixed FEM: DOF counting and block extraction (Stokes / linear elasticity)

For `FESystem` with multiple components, use the modern API to count DOFs.

```cpp
#include <deal.II/dofs/dof_renumbering.h>   // DoFRenumbering
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

// --- Step 1: renumber DOFs by component ---
DoFRenumbering::component_wise(dof_handler);

// --- Step 2: count DOFs per FE component (returns a vector, do NOT pass output by ref) ---
// Modern API (deal.II 9.x): returns std::vector<types::global_dof_index>
const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);

// --- Step 3: count DOFs per block (for block matrices) ---
// block_component maps each FE component to a block index.
// Example for Stokes (dim=2): velocity→block0, pressure→block1
const unsigned int dim = 2;
std::vector<unsigned int> block_component(dim + 1, 0); // velocity components → block 0
block_component[dim] = 1;                              // pressure component  → block 1
DoFRenumbering::component_wise(dof_handler, block_component);

const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

const types::global_dof_index n_u = dofs_per_block[0];
const types::global_dof_index n_p = dofs_per_block[1];
```

> **API change note:** The old functions `count_dofs_per_component`,
> `count_dofs_per_fe_block(dh, vec, vec)` (3-argument form), and
> `count_dofs_per_component` are **removed** in deal.II 9.x.
> Always use the 1- or 2-argument forms that **return** the vector.

---

## 15. Common pitfalls

| Problem | Fix |
|---|---|
| `FunctionParser` doesn't recognize `**` | Use `^` instead: `sin(pi*x)^2` |
| `FEFieldFunction::value_list` crashes | Check points are inside mesh bounds |
| Stiffness matrix is singular | Boundary conditions not applied before solve |
| Large residual after CG | Mesh too coarse; increase resolution |
| Mixed space: `count_dofs_per_component` not found | Use `DoFTools::count_dofs_per_fe_component(dof_handler)` (returns vector, see §14) |
| Mixed space: `count_dofs_per_fe_block` compile error | Use 2-arg form: `DoFTools::count_dofs_per_fe_block(dof_handler, block_component)` |
| `FEFieldFunction<dim, DoFHandler<dim>, Vec>` compile error | Use `Functions::FEFieldFunction<dim>` (1 template arg, see §10) |
| `reduction_control.h: No such file` | Include `<deal.II/lac/solver_control.h>` instead (see §1) |
| Time loop BC not updated | Call `bc_func.set_time(t)` before `interpolate_boundary_values` |
