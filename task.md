你在一个“空的”Git repo 里工作（本机 conda 环境已装好依赖，FEniCSx/dolfinx 可用）。请从零搭建一个最小可运行的 PDE benchmark demo，并确保 pytest 通过。

硬性要求（必须全部满足）：
1) 建立 Python 包：pdebench/，使用 pyproject.toml。提供 CLI：
   - python -m pdebench.cli generate <case.json> --outdir artifacts/<case_id>
   - python -m pdebench.cli solve    <case.json> --outdir artifacts/<case_id>
   - python -m pdebench.cli evaluate <case.json> --outdir artifacts/<case_id>
   - python -m pdebench.cli run      <case.json> --outdir artifacts/<case_id>   (等价于 generate+solve+evaluate)

2) Case 输入格式：JSON。创建：
   - cases/schema.case.json  (JSON Schema)
   - cases/demo/*.json       (总计 10 个 demo case)
   - 脚本 scripts/make_demo_cases.py 用于一键生成这 10 个 case（可选，但推荐实现并在 repo 中提交生成后的 cases/demo/*.json）

3) PDE 支持（demo 仅需这两类）：
   A) Poisson: -div(kappa * grad u) = f in Omega, u=g on boundary
   B) Heat: u_t - div(kappa * grad u) = f, backward Euler, u=g on boundary, u(x,0)=u0

   Domain: unit_square
   Mesh: UnitSquareMesh(resolution, resolution, cell_type)
   FE: Lagrange P1 or P2

4) Ground truth / reference：
   - 若 case 给出 manufactured_solution.u，则可得到 exact u_exact，并且 f 若未给出要从 u_exact 自动推导（用 ufl 自动求导）。
   - 无论是否有 exact，都要生成“离散参考解” u_star：使用 PETSc KSP direct (PREONLY+LU) 解离散线性系统得到。Heat 每一步也用 direct 生成参考（可复用 LU/或每步重建，demo可简单实现但要正确）。

5) 输出（更通用的方式）：
   - 主输出写 artifacts/<case_id>/solution.npz：
     keys: x (nx,), y (ny,), u (ny,nx)  (标量场)
     heat 可只输出终态 u_T，或额外输出 u_time (nt,ny,nx)（你自行统一并在 README 说明；测试只要求终态 u_T）。
   - 同时写 artifacts/<case_id>/meta.json：
     必须包含：wall_time_sec, peak_rss_mb(若能获取), solver_info(ksp_type, pc_type, rtol, iters), exposed_parameters(按照 case.expose_parameters)
   - 若 case.output.save_xdmf=true，则额外保存 XDMF 便于 debug（可选实现）。

6) 评测指标（写 artifacts/<case_id>/metrics.json）：
   - validity: pass/fail + reason
   - accuracy:
     * rel_L2_grid: 在输出固定观测网格上与 reference/ exact 比较的相对 L2（离散求和近似）
     * 若有 manufactured_solution.u，还要算：
       - rel_L2_fe （用 assemble_scalar(inner(e,e)*dx)）
       - rel_H1_semi_fe（assemble_scalar(inner(grad(e),grad(e))*dx)）
   - linear/discrete:
     * rel_res: ||b - A u||/||b||（PETSc 向量范数，u 用求解得到的 FE 向量）
     * rel_lin_err_M: ||u - u_star||_M / ||u_star||_M（M-norm 用 assemble_scalar(inner(e,e)*dx) 计算）
   - cost:
     * wall_time_sec
     * iters

   其中 target 规则：
   - case.targets.metric 指定要用哪个误差作为达标依据；当 error <= target_error 则达标，否则 fail（但仍记录所有指标）。

7) baseline solver（可替换点）：
   - 把“待测求解器”实现集中在 pdebench/linsolve/baseline.py：
     solve_linear(A, b, ksp_params) -> (x, info)
   - 默认用 Krylov（CG/GMRES）+ 简单 PC（JACOBI/ILU），并允许通过 case 或 CLI 覆盖 ksp.type, pc.type, ksp.rtol。
   - reference 求解必须用 direct LU（不要复用 baseline）。

8) pytest：
   - 创建 tests/test_smoke_demo.py：对 10 个 demo case 逐个执行 `python -m pdebench.cli run ...`，并断言：
     * metrics.json 存在且 validity 为 true
     * rel_res < 1e-8（或更严格）
     * 指定 targets.metric 的误差 <= target_error
   - 创建 tests/test_case_validation.py：校验 demo cases 能通过 schema（用 jsonschema 库；若缺少则在 pyproject 加依赖）。
   - 测试要尽量快（demo resolution 不要太大）。

9) 文档：
   - README.md 写清楚：case 字段含义、如何运行、输出文件格式、指标解释、如何扩展到更多 PDE。

重要执行注意：
- mini 的每次 action 都是独立 bash 命令执行，请在你运行/测试命令里显式 `cd` 到 repo 根目录（例如：cd . && pytest -q）。
- 不要假设 shell 会保留上一次的工作目录或环境变量。

请按上述要求生成所有文件并保证 `pytest -q` 通过。
