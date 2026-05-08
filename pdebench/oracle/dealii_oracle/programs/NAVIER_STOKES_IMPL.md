# navier_stokes.cc 实现规范

> **目标**：在 `pdebench/pdebench/oracle/dealii_oracle/programs/navier_stokes.cc` 中实现
> 定常不可压 Navier-Stokes 方程的 deal.II C++ oracle 求解器。
>
> **参考教程**：deal.II step-57（稳态 NS，Newton + 增广 Lagrangian + 回溯线搜索）
>
> **集成要求**：文件必须与现有基础设施完全兼容：
> - `programs/common/case_spec_reader.h`：JSON → `CaseSpec` 解析
> - `programs/common/grid_writer.h`：`write_vector_magnitude_grid()` 输出
> - `CMakeLists.txt`：第 85 行已声明 `add_oracle_executable(navier_stokes_solver navier_stokes.cc)`
> - Python 端 `common.py` 的 `_preprocess_navier_stokes()` 已注入的字段（见下文）

---

## 一、方程与弱形式

**强形式**（稳态不可压 NS）：
```
(u·∇)u − ν Δu + ∇p = f   in Ω = [0,1]²
        ∇·u = 0            in Ω
          u = g            on ∂Ω
```

**NS 残差** F(u, p; v, q)（弱形式）：
```
F = ν ∫ ∇u:∇v dx  +  ∫ (u·∇u)·v dx  −  ∫ p ∇·v dx  −  ∫ q ∇·u dx  −  ∫ f·v dx
```

**完整 Newton Jacobian**（step-57 方案，非 Picard）：

给定当前迭代值 (u^n, p^n)，求 Newton 修正量 (δu, δp)：

**左端 A(δu, δp; v, q)**：
```
ν ∫ ∇δu:∇v dx
+ ∫ (u^n · ∇δu) · v dx          ← 对流项1（transport，Picard 也有）
+ ∫ (δu · ∇u^n) · v dx          ← 对流项2（reactive，Newton 特有）
− ∫ δp ∇·v dx
− ∫ q ∇·δu dx
+ γ ∫ ∇·δu ∇·v dx               ← 增广 Lagrangian（grad-div 稳定化，γ=1）
```

**右端 b(v, q)**（= 负残差 + 源项）：
```
−ν ∫ ∇u^n:∇v dx
− ∫ (u^n·∇u^n)·v dx
+ ∫ p^n ∇·v dx
+ ∫ q ∇·u^n dx
− γ ∫ ∇·u^n ∇·v dx
+ ∫ f·v dx                       ← 外力源项（step-57 中 f=0，我们需要）
```

---

## 二、Python 预处理注入的字段（来自 `common.py`）

C++ 程序通过 `spec.pde` (nlohmann::json) 读取以下字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `_computed_nu` | string | 粘性系数 ν（muParser 表达式） |
| `_computed_source_x` | string | 体力 f_x（muParser 表达式） |
| `_computed_source_y` | string | 体力 f_y（muParser 表达式） |
| `_bc_segments` | string (JSON) | 边界条件列表，格式见下 |
| `_has_exact` | bool | 是否有制造解 |
| `_computed_bc_x` | string | 制造解速度 BC，ux(x,y) |
| `_computed_bc_y` | string | 制造解速度 BC，uy(x,y) |

`_bc_segments` 解析后为 `vector<{id, ex, ey}>`：
```json
[
  {"id": 0, "ex": "0.0",      "ey": "0.0"},
  {"id": 1, "ex": "1.0",      "ey": "0.0"},
  {"id": 2, "ex": "0.0",      "ey": "0.0"},
  {"id": 3, "ex": "0.0",      "ey": "0.0"}
]
```
边界 id 对应：0=x0(左), 1=x1(右), 2=y0(下), 3=y1(上)。

`SolverSpec`（`case_spec_reader.h`）中已有的 NS 相关字段：
```cpp
spec.oracle_solver.init                  // "zero" | "continuation"
spec.oracle_solver.continuation_nu_start // 延续法起始 ν（默认 1.0 = Re=1）
spec.oracle_solver.continuation_steps    // 延续步数（=0 时跳过延续）
spec.oracle_solver.rtol                  // Newton 收敛容差（默认 1e-10）
spec.oracle_solver.atol                  // Newton 绝对容差（默认 1e-12）
spec.oracle_solver.max_it                // Newton 最大迭代次数（默认 200000）
```

---

## 三、文件结构与 include

```cpp
// navier_stokes.cc 顶部 include（参考 stokes.cc，新增必要部分）
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
#include <deal.II/lac/sparse_direct.h>  // UMFPACK
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "case_spec_reader.h"
#include "grid_writer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dealii;
```

---

## 四、辅助类

### 4.1 VelocityBC（与 stokes.cc 相同）

```cpp
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
        v(0) = fx_.value(p);
        v(1) = fy_.value(p);
    }
private:
    mutable FunctionParser<2> fx_, fy_;
};
```

### 4.2 辅助函数

```cpp
// 解析 _bc_segments JSON 字符串
std::vector<nlohmann::json> parse_bc_segments(const std::string& json_str) {
    auto arr = nlohmann::json::parse(json_str);
    return arr.get<std::vector<nlohmann::json>>();
}
```

---

## 五、主求解器类 NavierStokesOracle

### 5.1 成员变量

```cpp
class NavierStokesOracle {
public:
    explicit NavierStokesOracle(const CaseSpec& s);
    void run(const std::string& outdir);

private:
    // --- 数据成员 ---
    const CaseSpec&           spec_;
    double                    nu_;
    double                    gamma_;      // 增广 Lagrangian 系数，固定 = 1.0

    Triangulation<2>          tria_;
    FESystem<2>               fe_;         // Q_{du} × Q_{dp}（du=2, dp=1）
    DoFHandler<2>             dh_;

    // step-57 风格：两个约束集合
    AffineConstraints<double> nonzero_cons_;  // 初始步：强制非零 BC
    AffineConstraints<double> zero_cons_;     // Newton 更新步：零 BC

    SparsityPattern           sp_;
    SparseMatrix<double>      system_matrix_;

    // 当前解、Newton 更新量、用于组装的评估点（step-57 pattern）
    Vector<double>            present_solution_;
    Vector<double>            newton_update_;
    Vector<double>            system_rhs_;
    Vector<double>            evaluation_point_;

    // --- 私有方法 ---
    void make_mesh();
    void setup_dofs();
    void assemble(bool initial_step, bool assemble_matrix);
    void assemble_system(bool initial_step);
    void assemble_rhs(bool initial_step);
    void solve_linear_system(bool initial_step);
    void newton_iteration(double tol, int max_iter, bool initial_step);
    void continuation_then_newton();
};
```

### 5.2 构造函数

```cpp
NavierStokesOracle::NavierStokesOracle(const CaseSpec& s)
    : spec_(s),
      fe_(FE_Q<2>(s.fem.degree_u), 2,
          FE_Q<2>(s.fem.degree_p), 1),
      dh_(tria_),
      gamma_(1.0)
{
    nu_ = std::stod(spec_.pde.value("_computed_nu", "1.0"));
}
```

### 5.3 make_mesh（与 stokes.cc 相同）

```cpp
void NavierStokesOracle::make_mesh() {
    GridGenerator::subdivided_hyper_cube(tria_, spec_.mesh.resolution, 0.0, 1.0);
    // 标记边界 id: 0=x0, 1=x1, 2=y0, 3=y1
    for (const auto& cell : tria_.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
            const auto face = cell->face(f);
            if (!face->at_boundary()) continue;
            const auto c = face->center();
            if      (std::abs(c[0] - 0.0) < 1e-12) face->set_boundary_id(0);
            else if (std::abs(c[0] - 1.0) < 1e-12) face->set_boundary_id(1);
            else if (std::abs(c[1] - 0.0) < 1e-12) face->set_boundary_id(2);
            else if (std::abs(c[1] - 1.0) < 1e-12) face->set_boundary_id(3);
        }
    }
}
```

### 5.4 setup_dofs（step-57 风格：两个约束集合）

```cpp
void NavierStokesOracle::setup_dofs() {
    dh_.distribute_dofs(fe_);
    DoFRenumbering::component_wise(dh_);

    // 构建速度 ComponentMask（前 2 个分量）
    const ComponentMask vel_mask = fe_.component_mask(FEValuesExtractors::Vector(0));

    // 解析边界段
    const std::string bc_seg_str = spec_.pde.value("_bc_segments", "[]");
    auto segments = nlohmann::json::parse(bc_seg_str);

    // --- nonzero_cons_：非零 Dirichlet BC（用于初始步和维持解满足 BC）---
    nonzero_cons_.clear();
    for (const auto& seg : segments) {
        int bid = seg["id"].get<int>();
        std::string ex = seg["ex"].get<std::string>();
        std::string ey = seg["ey"].get<std::string>();
        VelocityBC bc_func(ex, ey);
        VectorTools::interpolate_boundary_values(dh_, (types::boundary_id)bid,
                                                  bc_func, nonzero_cons_, vel_mask);
    }

    // 压力点固定（固定第一个压力 DOF = 0，消除压力的不唯一性）
    // 仅当 pressure_fixing == "point" 时
    if (spec_.oracle_solver.pressure_fixing == "point") {
        const ComponentMask p_mask = fe_.component_mask(FEValuesExtractors::Scalar(2));
        const IndexSet p_dofs = DoFTools::extract_dofs(dh_, p_mask);
        if (p_dofs.n_elements() > 0) {
            nonzero_cons_.add_line(*p_dofs.begin());
            nonzero_cons_.set_inhomogeneity(*p_dofs.begin(), 0.0);
        }
    }
    nonzero_cons_.close();

    // --- zero_cons_：Newton 更新步用零 BC（保证修正量在边界上为零）---
    zero_cons_.clear();
    for (const auto& seg : segments) {
        int bid = seg["id"].get<int>();
        Functions::ZeroFunction<2> zero_func(3);  // dim+1=3 分量
        VectorTools::interpolate_boundary_values(dh_, (types::boundary_id)bid,
                                                  zero_func, zero_cons_, vel_mask);
    }
    if (spec_.oracle_solver.pressure_fixing == "point") {
        const ComponentMask p_mask = fe_.component_mask(FEValuesExtractors::Scalar(2));
        const IndexSet p_dofs = DoFTools::extract_dofs(dh_, p_mask);
        if (p_dofs.n_elements() > 0) {
            zero_cons_.add_line(*p_dofs.begin());
            zero_cons_.set_inhomogeneity(*p_dofs.begin(), 0.0);
        }
    }
    zero_cons_.close();

    // 构建稀疏模式（用 nonzero_cons_ 使结构包含所有必要条目）
    DynamicSparsityPattern dsp(dh_.n_dofs());
    DoFTools::make_sparsity_pattern(dh_, dsp, nonzero_cons_);
    sp_.copy_from(dsp);
    system_matrix_.reinit(sp_);

    present_solution_.reinit(dh_.n_dofs());
    newton_update_.reinit(dh_.n_dofs());
    system_rhs_.reinit(dh_.n_dofs());
    evaluation_point_.reinit(dh_.n_dofs());
}
```

### 5.5 assemble（核心，step-57 风格的完整 Newton Jacobian）

```cpp
void NavierStokesOracle::assemble(bool initial_step, bool assemble_matrix) {
    if (assemble_matrix) system_matrix_ = 0;
    system_rhs_ = 0;

    // 读取源项（muParser 表达式）
    const std::string fx_str = spec_.pde.value("_computed_source_x", "0.0");
    const std::string fy_str = spec_.pde.value("_computed_source_y", "0.0");
    std::map<std::string, double> mu_consts = {{"pi", M_PI}};
    FunctionParser<2> fx(1), fy(1);
    fx.initialize("x,y", fx_str, mu_consts, false);
    fy.initialize("x,y", fy_str, mu_consts, false);

    const FEValuesExtractors::Vector vel(0);
    const FEValuesExtractors::Scalar pres(2);

    QGauss<2> quad(fe_.degree + 2);
    FEValues<2> fev(fe_, quad,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values);

    const unsigned int n_dpc = fe_.n_dofs_per_cell();
    const unsigned int n_q   = quad.size();

    FullMatrix<double> Ke(n_dpc, n_dpc);
    Vector<double>     Fe(n_dpc);
    std::vector<types::global_dof_index> ids(n_dpc);

    // 存储当前评估点 (u^n, p^n) 在积分点处的值/梯度
    std::vector<Tensor<1,2>>       u_vals(n_q);    // u^n(q)
    std::vector<Tensor<2,2>>       u_grads(n_q);   // ∇u^n(q)
    std::vector<double>            p_vals(n_q);    // p^n(q)

    // 形函数缓存
    std::vector<Tensor<1,2>>       phi_u(n_dpc);
    std::vector<Tensor<2,2>>       grad_phi_u(n_dpc);
    std::vector<double>            div_phi_u(n_dpc);
    std::vector<double>            phi_p(n_dpc);

    for (auto& cell : dh_.active_cell_iterators()) {
        fev.reinit(cell);
        Ke = 0; Fe = 0;

        // 从 evaluation_point_ 取当前迭代值
        fev[vel].get_function_values(evaluation_point_, u_vals);
        fev[vel].get_function_gradients(evaluation_point_, u_grads);
        fev[pres].get_function_values(evaluation_point_, p_vals);

        for (unsigned int q = 0; q < n_q; ++q) {
            const Point<2>& qp  = fev.quadrature_point(q);
            const double    JxW = fev.JxW(q);
            Tensor<1,2> f_vec;
            f_vec[0] = fx.value(qp);
            f_vec[1] = fy.value(qp);

            // 预取形函数值
            for (unsigned int k = 0; k < n_dpc; ++k) {
                phi_u[k]     = fev[vel].value(k, q);
                grad_phi_u[k]= fev[vel].gradient(k, q);
                div_phi_u[k] = fev[vel].divergence(k, q);
                phi_p[k]     = fev[pres].value(k, q);
            }

            // u^n 的散度（用于 AL 残差项）
            double div_un = trace(u_grads[q]);

            for (unsigned int i = 0; i < n_dpc; ++i) {
                // --- 右端：负残差 b_i = -F(u^n, p^n) + f ---
                // -ν ∇u^n:∇v_i  →  -ν * scalar_product(grad_phi_u[i], u_grads[q])
                // -(u^n·∇u^n)·v_i  →  -phi_u[i] · (u_grads[q] * u_vals[q])
                // +p^n ∇·v_i  →  +div_phi_u[i] * p_vals[q]
                // +q_i ∇·u^n  →  +phi_p[i] * div_un
                // -γ ∇·v_i ∇·u^n  →  -gamma_ * div_phi_u[i] * div_un
                // +f·v_i  →  +phi_u[i] * f_vec
                Fe(i) += (
                    -nu_ * scalar_product(grad_phi_u[i], u_grads[q])
                    - phi_u[i] * (u_grads[q] * u_vals[q])
                    + div_phi_u[i] * p_vals[q]
                    + phi_p[i] * div_un
                    - gamma_ * div_phi_u[i] * div_un
                    + phi_u[i] * f_vec
                ) * JxW;

                if (assemble_matrix) {
                    for (unsigned int j = 0; j < n_dpc; ++j) {
                        // --- 左端：Newton Jacobian A(δu_j, δp_j; v_i, q_i) ---

                        // ν ∇δu_j : ∇v_i
                        double val = nu_ * scalar_product(grad_phi_u[i], grad_phi_u[j]);

                        // (u^n · ∇)δu_j · v_i  →  transport 项
                        // = phi_u[i] · (grad_phi_u[j] · u_vals[q])
                        val += phi_u[i] * (grad_phi_u[j] * u_vals[q]);

                        // (δu_j · ∇)u^n · v_i  →  reactive 项（Newton 特有）
                        // = phi_u[i] · (u_grads[q] * phi_u[j])
                        val += phi_u[i] * (u_grads[q] * phi_u[j]);

                        // −δp_j ∇·v_i
                        val -= div_phi_u[i] * phi_p[j];

                        // −q_i ∇·δu_j
                        val -= phi_p[i] * div_phi_u[j];

                        // γ ∇·δu_j ∇·v_i  （增广 Lagrangian）
                        val += gamma_ * div_phi_u[i] * div_phi_u[j];

                        Ke(i, j) += val * JxW;
                    }
                }
            }
        }

        cell->get_dof_indices(ids);
        const AffineConstraints<double>& cons =
            initial_step ? nonzero_cons_ : zero_cons_;
        if (assemble_matrix)
            cons.distribute_local_to_global(Ke, Fe, ids, system_matrix_, system_rhs_);
        else
            cons.distribute_local_to_global(Fe, ids, system_rhs_);
    }
}

void NavierStokesOracle::assemble_system(bool initial_step) { assemble(initial_step, true); }
void NavierStokesOracle::assemble_rhs(bool initial_step)    { assemble(initial_step, false); }
```

### 5.6 solve_linear_system（UMFPACK 直接求解）

```cpp
void NavierStokesOracle::solve_linear_system(bool initial_step) {
    SparseDirectUMFPACK direct;
    direct.factorize(system_matrix_);
    direct.vmult(newton_update_, system_rhs_);

    const AffineConstraints<double>& cons =
        initial_step ? nonzero_cons_ : zero_cons_;
    cons.distribute(newton_update_);
}
```

### 5.7 newton_iteration（带回溯线搜索，step-57 风格）

```cpp
void NavierStokesOracle::newton_iteration(double tol, int max_iter, bool initial_step) {
    bool first_step = initial_step;
    double last_res  = 1.0;
    double current_res = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (first_step) {
            // 初始步：用非零 BC 装配并求解，得初始猜测
            evaluation_point_ = present_solution_;
            assemble_system(true);
            solve_linear_system(true);
            present_solution_ = newton_update_;
            nonzero_cons_.distribute(present_solution_);
            first_step = false;

            // 计算初始残差
            evaluation_point_ = present_solution_;
            assemble_rhs(false);
            current_res = system_rhs_.l2_norm();
            last_res = current_res;
            std::cerr << "[NS Newton] Initial residual: " << current_res << "\n";
        } else {
            // 标准 Newton 步：装配 Jacobian + 负残差，求解更新量
            evaluation_point_ = present_solution_;
            assemble_system(false);
            solve_linear_system(false);

            // 回溯线搜索：α ∈ {1, 0.5, 0.25, ...}
            for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5) {
                evaluation_point_ = present_solution_;
                evaluation_point_.add(alpha, newton_update_);
                zero_cons_.distribute(evaluation_point_);

                assemble_rhs(false);
                current_res = system_rhs_.l2_norm();
                if (current_res < last_res) break;
            }

            present_solution_ = evaluation_point_;
            std::cerr << "[NS Newton] iter=" << iter
                      << " residual=" << current_res << "\n";
            last_res = current_res;
        }

        // 收敛检查
        if (current_res < tol) {
            std::cerr << "[NS Newton] Converged at iter=" << iter << "\n";
            break;
        }
    }
}
```

### 5.8 延续法（高 Re 时 ν 从大到小逐步求解）

```cpp
void NavierStokesOracle::continuation_then_newton() {
    const double nu_target = nu_;
    const double nu_start  = spec_.oracle_solver.continuation_nu_start;  // 默认 1.0
    const int    n_steps   = spec_.oracle_solver.continuation_steps;      // 默认 0

    const double tol      = spec_.oracle_solver.rtol;
    const int    max_iter = spec_.oracle_solver.max_it;

    if (n_steps <= 0 || nu_start <= nu_target) {
        // 不需要延续法，直接 Newton
        newton_iteration(tol, max_iter, /*initial_step=*/true);
        return;
    }

    // ν 从 nu_start 按等比数列减小到 nu_target
    bool is_initial = true;
    for (int k = 1; k <= n_steps; ++k) {
        nu_ = nu_start * std::pow(nu_target / nu_start,
                                   static_cast<double>(k) / n_steps);
        std::cerr << "[NS Continuation] step=" << k << "/" << n_steps
                  << " nu=" << nu_ << "\n";

        // 每个 ν 值重新 setup（nu_ 已更新，setup_dofs 不依赖 nu_）
        // 注意：稀疏模式不变，只需清零矩阵
        newton_iteration(tol, max_iter, is_initial);
        is_initial = false;
    }
}
```

### 5.9 run（主入口）

```cpp
void NavierStokesOracle::run(const std::string& outdir) {
    std::filesystem::create_directories(outdir);

    Timer timer; timer.start();

    make_mesh();
    setup_dofs();

    // 初始化解为零
    present_solution_ = 0;

    // 根据 init 模式选择求解策略
    const std::string init_mode = spec_.oracle_solver.init;  // "zero" 或 "continuation"
    const double tol      = spec_.oracle_solver.rtol;
    const int    max_iter = spec_.oracle_solver.max_it;

    if (init_mode == "continuation" && spec_.oracle_solver.continuation_steps > 0) {
        continuation_then_newton();
    } else {
        newton_iteration(tol, max_iter, /*initial_step=*/true);
    }

    timer.stop();

    // 输出速度模长 ‖u‖ 到 solution_grid.bin + meta.json
    oracle_util::write_vector_magnitude_grid(
        dh_, present_solution_,
        spec_.output_grid.bbox,
        spec_.output_grid.nx,
        spec_.output_grid.ny,
        outdir,
        timer.wall_time(),
        spec_.oracle_solver.ksp_type,
        spec_.oracle_solver.pc_type,
        spec_.oracle_solver.rtol);
}
```

### 5.10 main 函数

```cpp
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: navier_stokes_solver <case_spec.json> <outdir>\n";
        return 1;
    }
    try {
        NavierStokesOracle(read_case_spec(argv[1])).run(argv[2]);
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

---

## 六、关键实现细节与注意事项

### 6.1 张量运算对照（避免歧义）

| 数学表达式 | deal.II C++ 代码 | 说明 |
|-----------|----------------|------|
| `ν ∇δu:∇v` | `nu_ * scalar_product(grad_phi_u[i], grad_phi_u[j])` | Tensor<2>双缩并 |
| `(u^n·∇)δu · v` | `phi_u[i] * (grad_phi_u[j] * u_vals[q])` | `Tensor<2>*Tensor<1>→Tensor<1>`，再点积 |
| `(δu·∇)u^n · v` | `phi_u[i] * (u_grads[q] * phi_u[j])` | 同上，u_grads[q] 是 ∇u^n |
| `(u^n·∇u^n)·v` | `phi_u[i] * (u_grads[q] * u_vals[q])` | RHS 中的非线性项 |
| `∇·u^n` | `trace(u_grads[q])` | Tensor<2> 的迹 = 散度 |

### 6.2 压力唯一性固定

- 默认 `pressure_fixing = "point"`：固定第一个压力 DOF = 0。
- 在 `nonzero_cons_` 和 `zero_cons_` 中**均**加入此约束行（inhomogeneity=0）。
- 压力仅需在一处固定（常见于纯 Dirichlet 速度边界问题）。

### 6.3 增广 Lagrangian 系数 γ

- 固定 `gamma_ = 1.0`（与 step-57 相同）。
- 不暴露为参数（oracle 不需要调节）。
- AL 项 `γ ∫ ∇·δu ∇·v dx` 同时出现在矩阵（LHS）和右端（RHS 中的 `-γ ∫ ∇·u^n ∇·v dx`）。

### 6.4 两个约束集合的使用规则

| 场景 | 使用的约束 | 原因 |
|------|-----------|------|
| 第一步（`initial_step=true`）组装 | `nonzero_cons_` | 强制解满足非零边界条件 |
| 后续 Newton 更新步组装 | `zero_cons_` | 更新量 δu 在边界上必须为零 |
| `distribute()` 在第一步后 | `nonzero_cons_` | 设置 `present_solution_` 到正确 BC |
| `distribute()` 在线搜索中 | `zero_cons_` | 评估点也是更新量叠加 |

### 6.5 FE 空间次数

从 `spec.fem.degree_u` 和 `spec.fem.degree_p` 读取，默认 Q2×Q1（Taylor-Hood）：
```cpp
FESystem<2> fe_(FE_Q<2>(s.fem.degree_u), 2,   // 速度：Q_{du}，2 分量
                FE_Q<2>(s.fem.degree_p), 1);   // 压力：Q_{dp}，1 分量
```
FE 分量索引：0=ux, 1=uy, 2=p → `FEValuesExtractors::Vector vel(0)`, `FEValuesExtractors::Scalar pres(2)`。

### 6.6 quadrature 阶数

```cpp
QGauss<2> quad(fe_.degree + 2);
```
`fe_.degree` 是 FESystem 的最高次数（即 degree_u），+2 保证精确积分 Q2 元的 2次非线性项。

### 6.7 write_vector_magnitude_grid 的正确调用

`grid_writer.h` 中的 `write_vector_magnitude_grid` 期望：
- `dof_handler`：整个混合空间的 DoFHandler（不是 collapsed 的速度子空间）
- `solution`：整个解向量（velocity + pressure DOFs）
- 内部通过 `FEFieldFunction::vector_value_list` 自动处理 dim=3 (ux,uy,p) 的向量场
- **注意**：`write_vector_magnitude_grid` 计算所有 dim 个分量的模长，而此函数写的是 `dim=2` 的模长（不包含压力），因为模板参数 `dim=2` 且 `FEFieldFunction` 的 `dim` 是空间维度。

> **验证**：`grid_writer.h` 第 130-162 行，`FEFieldFunction<dim>` 中 `dim` 是空间维度（=2），`vector_value_list` 返回大小为 dim（=2）的 `Vector<double>`，即 (ux, uy)，压力不参与模长计算。因此调用时直接传入混合空间的 `dh_` 和 `present_solution_` 即可，无需 collapse。

---

## 七、与 stokes.cc 的完整差异对比

| 特性 | stokes.cc | navier_stokes.cc |
|------|-----------|-----------------|
| 问题类型 | 线性 | **非线性** |
| 约束集合 | 1 个（nonzero） | **2 个**（nonzero + zero） |
| 外层迭代 | 无（单次求解） | **Newton 迭代（含线搜索）** |
| 矩阵装配 | 一次 | **每次 Newton 迭代重新装配** |
| 对流项 | 无 | **2 项：transport + reactive** |
| 增广 Lagrangian | 无 | **有，γ=1** |
| RHS | 线性源项 | **负残差 + 源项** |
| 线搜索 | 无 | **回溯线搜索** |
| 延续法 | 无 | **有（高 Re）** |
| 线性求解 | UMFPACK | **UMFPACK（每 Newton 步）** |
| 评估点 | 无 | **evaluation_point_（step-57 pattern）** |

---

## 八、验证检查清单

实现完成后，按以下顺序验证：

1. **编译**：在 `build/` 目录执行 `cmake .. && make navier_stokes_solver`，无错误。
2. **零体力 + 全壁零 BC**：应收敛到零速度场（残差第一步即 < tol）。
3. **制造解测试**（`_has_exact=true`）：给定 `manufactured_solution.u/p`，体力由 `_computed_source_x/y` 给出，最终 `‖u_h - u_exact‖ / ‖u_exact‖` 应随网格加密二阶收敛（Q2 元）。
4. **Lid-driven cavity**（`bc.dirichlet` 顶壁 uy=1，其余为零，无源项）：Re=100 时应收敛（约 5-10 Newton 步），Re=1000 时需开启延续法。
5. **输出文件**：`outdir/solution_grid.bin`（float64，形状 ny×nx）和 `outdir/meta.json` 均存在且合法。

---

## 九、文件位置与 CMake 状态

```
pdebench/pdebench/oracle/dealii_oracle/programs/
├── CMakeLists.txt                      ← 第 85 行已有：add_oracle_executable(navier_stokes_solver navier_stokes.cc)
├── navier_stokes.cc                    ← ✅ 需要创建（本规范描述的文件）
├── stokes.cc                           ← 参考实现
├── common/
│   ├── case_spec_reader.h              ← CaseSpec 解析（含 continuation_* 字段）
│   └── grid_writer.h                   ← write_vector_magnitude_grid
└── ...
```

CMakeLists.txt 无需修改，第 85 行已声明：
```cmake
add_oracle_executable(navier_stokes_solver navier_stokes.cc)
```

---

## 十、Python 端接口一致性确认

`oracle.py` 的 `solve()` 方法中（第 513-518 行）：
```python
elif pde_type in ("stokes", "navier_stokes", "linear_elasticity"):
    ref_grid, ref_info = _vector_magnitude_pde_reference_grid(self, case_spec, pde_type)
```

`common.py` 中（第 656 行）：
```python
"navier_stokes": "navier_stokes_solver",
```

`_preprocess_navier_stokes()` 已完整实现注入（第 444-533 行），C++ 只需读取即可。

**无需修改任何 Python 文件**，仅需创建 `navier_stokes.cc`。
