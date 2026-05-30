"""
案例3：投资组合优化 (MIQP)
难度：★★☆☆☆
求解器：Gurobi (后备 HiGHS LP松弛)
依赖：pip install gurobipy numpy
运行：python case03_portfolio.py

简述：从 500 只股票中选 20 只构建投资组合，用 MIQP 同时优化
      选股决策(y)和资金分配(w)，最大化「收益 - λ×风险」。
      对比三种方案（等权全选 / 等权随机 / MIQP 优化）的收益、
      波动率和 Sharpe 比率。

技术背景：
  经典马科维茨均值-方差模型是连续 QP（所有股票都可投任意比例）。
  加入「恰好选 k 只」的离散约束后变成 MIQP——目标函数含二次项
  w[i]*w[j]*σ[i,j]（协方差），同时有 y[i] ∈ {0,1} 和 linking
  constraints w[i] ≤ y[i]。求解器在每个 branch-and-bound 节点上
  需要求解一个 QP 子问题，计算量远大于普通 MIP。

公式逐符号解释:

  目标函数:
    max  Σ[i] r[i] * w[i]  -  λ * Σ[i] Σ[j] w[i] * w[j] * σ[i,j]

    r[i]    → 股票 i 的预期年化收益率（从历史数据估计）
    w[i]    → 投资于股票 i 的资金比例（连续变量 ∈ [0, 1]）
    λ       → 风险厌恶系数（越大越怕风险，越小越激进）
    σ[i,j]  → 股票 i 和 j 的收益率协方差
               σ[i,i] = i 的方差（波动率²）
               σ[i,j] (i≠j) = 同涨同跌程度

  约束:
    (1) Σ w[i] = 1                  ← 资金 100% 分配
    (2) Σ y[i] = k                  ← 恰好选 k 只股票
    (3) w[i] ≤ y[i]   ∀i            ← 选了才能投钱
    (4) 0 ≤ w[i] ≤ w_max   ∀i       ← 单只上限，防过度集中
    (5) y[i] ∈ {0, 1}   ∀i          ← 离散决策：选/不选

  决策变量:
    y[i] ∈ {0, 1}  → 是否将股票 i 纳入组合（0-1 整数变量）
    w[i] ∈ [0, 1]  → 股票 i 的投资权重（连续变量）
"""


# 教学注释：重点看变量、约束和目标函数如何把业务规则翻译成 MIP 模型。
# 求解日志或结果可用来理解分支定界、松弛和可行解质量。



import numpy as np
import time
import sys


# ============================================================
# 数据生成（模拟市场数据）
# ============================================================

def generate_market_data(n_stocks, seed=42, rf=0.025):
    """
    生成模拟股票市场数据。

    参数:
        n_stocks: 股票数量
        seed: 随机种子（保证可复现）
        rf: 无风险利率（用于计算 Sharpe 比率），默认 2.5%

    返回:
        returns: shape (n_stocks,) — 各股票预期年化收益率
        cov_matrix: shape (n_stocks, n_stocks) — 协方差矩阵
        rf: 无风险利率
    """
    rng = np.random.default_rng(seed)

    # --- 收益率：从正态分布抽取，均值 ~10%，标准差 ~6% ---
    # 实际市场中预期收益很难准确估计，这里用模拟数据演示模型逻辑
    returns = rng.normal(loc=0.10, scale=0.06, size=n_stocks)
    # 限制在合理范围 [1%, 25%]
    returns = np.clip(returns, 0.01, 0.25)

    # --- 协方差矩阵：用因子模型模拟 ---
    # 真实股票收益率可以用 Fama-French 三因子或 PCA 来生成协方差
    # 这里简化为：一个全局市场因子 + 每个股票独立的个股因子
    n_factors = 3                 # 模拟 3 个行业因子
    factor_loadings = rng.normal(0, 1, size=(n_stocks, n_factors))
    # 因子协方差
    factor_cov = np.eye(n_factors) * 0.015
    # 残差方差（个股特质波动）
    idiosyncratic_var = rng.uniform(0.005, 0.025, size=n_stocks)

    # Σ = B · Ω · Bᵀ + D
    cov_matrix = (factor_loadings @ factor_cov @ factor_loadings.T
                  + np.diag(idiosyncratic_var))

    # 归一化：使平均年化波动率在 15% 左右
    avg_vol = np.sqrt(np.mean(np.diag(cov_matrix)))
    scale_factor = (0.15 ** 2) / (avg_vol ** 2)
    cov_matrix = cov_matrix * scale_factor

    return returns, cov_matrix, rf


# ============================================================
# 辅助函数：计算组合统计量
# ============================================================

def compute_portfolio_stats(weights, returns, cov_matrix, rf=0.025):
    """
    给定权重向量，计算组合的预期收益、波动率和 Sharpe 比率。

    参数:
        weights: shape (n_stocks,) — 投资权重向量，必须 ∑w=1
        returns: shape (n_stocks,) — 预期收益率
        cov_matrix: shape (n_stocks, n_stocks) — 协方差矩阵
        rf: 无风险利率

    返回:
        exp_return: 组合预期年化收益率
        volatility: 组合年化波动率
        sharpe: Sharpe 比率 = (exp_return - rf) / volatility
    """
    exp_return = float(np.dot(weights, returns))
    variance = float(weights @ cov_matrix @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))  # 防止浮点负值
    sharpe = (exp_return - rf) / volatility if volatility > 1e-10 else 0.0
    return exp_return, volatility, sharpe


# ============================================================
# 主求解函数：MIQP 投资组合优化
# ============================================================

def solve_portfolio_miqp(n_stocks=500, k=20, lam=1.0, w_max=0.15,
                         w_min=0.01, seed=42, use_highs_fallback=False):
    """
    用 MIQP 求解投资组合优化问题：从 n_stocks 只股票中选 k 只，
    最大化「收益 - λ×风险」。

    参数:
        n_stocks: 候选股票数量
        k: 需要选入组合的股票数量
        lam: 风险厌恶系数 (λ)，越大越保守
        w_max: 单只股票最大权重 (防止过度集中)
        w_min: 单只股票最小权重 (防止持仓碎片化)
        seed: 随机种子
        use_highs_fallback: 是否强制使用 HiGHS LP 松弛 (调试用)

    返回:
        dict: 包含权重、指标、求解状态等信息
    """
    # --- 数据准备 ---
    returns, cov_matrix, rf = generate_market_data(n_stocks, seed=seed)

    # 用 numpy 的 float64 转 Python float，避免 gurobipy 类型兼容问题
    returns_list = [float(r) for r in returns]

    print(f"\n{'='*60}")
    print(f"  投资组合优化 (MIQP): {n_stocks} 只股票, 选 {k} 只")
    print(f"  风险厌恶 λ = {lam}, 单只上限 {w_max*100:.0f}%, 下限 {w_min*100:.0f}%")
    print(f"{'='*60}")

    # ============================================================
    # 求解器选择：优先 Gurobi，降级到 HiGHS LP 松弛
    # ============================================================
    use_gurobi = False
    model = None

    if not use_highs_fallback:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            use_gurobi = True
            print(f"  求解器: Gurobi (MIQP)")
        except ImportError:
            print("  [WARNING] gurobipy 未安装，降级到 HiGHS LP 松弛")
            print("    安装 Gurobi: pip install gurobipy")
            print("    免费学术许可: https://www.gurobi.com/downloads/")
            use_gurobi = False
    else:
        print(f"  求解器: HiGHS (LP 松弛 — 手动指定)")

    # ============================================================
    # 路径 A: Gurobi MIQP
    # ============================================================
    if use_gurobi:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            env = gp.Env(params={"OutputFlag": 0})
            model = gp.Model("portfolio", env=env)

            # --- 决策变量 ---
            # w[i]: 股票 i 的权重（连续）
            w = {
                i: model.addVar(
                    lb=0.0, ub=w_max, name=f"w_{i}",
                    vtype=GRB.CONTINUOUS
                )
                for i in range(n_stocks)
            }
            # y[i]: 是否选中股票 i（0-1）
            y = {
                i: model.addVar(
                    vtype=GRB.BINARY, name=f"y_{i}"
                )
                for i in range(n_stocks)
            }

            # --- 目标函数: max Σ r[i]*w[i] - λ * Σ Σ w[i]*w[j]*σ[i,j] ---
            # gurobipy 中 setObjective 接受二次表达式
            linear_expr = gp.quicksum(returns_list[i] * w[i]
                                      for i in range(n_stocks))

            quad_expr = gp.QuadExpr()
            for i in range(n_stocks):
                for j in range(n_stocks):
                    if abs(cov_matrix[i, j]) > 1e-12:
                        quad_expr += w[i] * cov_matrix[i, j] * w[j]

            objective = linear_expr - lam * quad_expr
            model.setObjective(objective, GRB.MAXIMIZE)

            # --- 约束 ---
            # (1) 预算约束: Σ w[i] = 1
            model.addConstr(
                gp.quicksum(w[i] for i in range(n_stocks)) == 1,
                name="budget"
            )

            # (2) 选股数量约束: Σ y[i] = k
            model.addConstr(
                gp.quicksum(y[i] for i in range(n_stocks)) == k,
                name="select_k"
            )

            # (3) Linking 约束: w[i] ≤ y[i] （选了才能投钱）
            for i in range(n_stocks):
                model.addConstr(w[i] <= y[i], name=f"link_{i}")

            # (4) 最小权重: 如果选了，至少投 w_min
            for i in range(n_stocks):
                model.addConstr(w[i] >= w_min * y[i], name=f"minwt_{i}")

            # --- 求解 ---
            print(f"  变量数: {model.NumVars} (连续+0-1)")
            print(f"  约束数: {model.NumConstrs}")
            print(f"  二次项数: {model.NumQNZs}")
            print(f"\n  求解中...")

            start_time = time.time()
            model.optimize()
            solve_time = time.time() - start_time

            status_map = {
                GRB.OPTIMAL: "最优解",
                GRB.INFEASIBLE: "不可行",
                GRB.UNBOUNDED: "无界",
                GRB.TIME_LIMIT: "超时(次优解)",
                GRB.SUBOPTIMAL: "次优解",
                GRB.INF_OR_UNBD: "不可行或无界",
            }
            status_str = status_map.get(model.Status, f"状态码: {model.Status}")

            print(f"\n  求解状态: {status_str}")
            print(f"  求解耗时: {solve_time:.1f}s")
            if model.Status == GRB.OPTIMAL:
                print(f"  目标值: {model.ObjVal:.6f}")

            # --- 提取结果 ---
            if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                w_vals = np.array([w[i].X for i in range(n_stocks)])
                y_vals = np.array([int(round(y[i].X)) for i in range(n_stocks)])
            else:
                w_vals = np.zeros(n_stocks)
                y_vals = np.zeros(n_stocks)

            # 清理环境
            model.dispose()
            env.dispose()

            is_lp_relaxation = False

        except Exception as e:
            print(f"  [ERROR] Gurobi 出错: {e}")
            print(f"  降级到 HiGHS LP 松弛...")
            use_gurobi = False
            # 继续到 HiGHS 路径
            pass

    # ============================================================
    # 路径 B: HiGHS LP 松弛（后备方案）
    # ============================================================
    if not use_gurobi or (use_gurobi and 'w_vals' not in dir()):
        try:
            import highspy
        except ImportError:
            print("  [FATAL] highspy 也未安装。")
            print("  pip install highspy")
            print("  无法继续——至少需要 highspy 或 gurobipy 之一。")
            return None

        print("  求解器: HiGHS (LP 松弛 — y[i] 从 {0,1} 放宽到 [0,1])")
        print("  注意: 结果为 LP 松弛，y[i] 可能为分数，")
        print("        仅用于验证模型结构，不能用于实际投资决策。")

        model_h = highspy.Highs()
        model_h.setOptionValue("output_flag", False)

        # --- 变量: w 连续, y 连续（松弛） ---
        # HiGHS 不支持直接 MIQP，所以我们构建 LP 松弛
        # 并将二次项通过 McCormick 包络或其他方式近似
        # 这里采用简化方案：忽略协方差交叉项，仅用方差做线性代理
        #
        # 因为我们无法在 HiGHS 中表示真正的二次目标，
        # 我们改为：max Σ r[i]*w[i] - λ * Σ variance[i] * w[i]²
        # 这仍然是一个二次目标。更深度的降级无法实现。
        #
        # 实际的降级策略：构造一个近似的线性模型
        # 目标: max Σ r[i]*w[i] - λ * Σ σ[i,i] * w[i]
        # (用方差一阶近似风险，虽然不精确但足以展示 MIP 建模思路)

        n = n_stocks

        # 构建列式模型
        # w_cols[i]: 权重变量的列索引
        # y_cols[i]: 选股变量的列索引
        w_cols = {}
        y_cols = {}

        for i in range(n):
            idx_w = model_h.addVar(0.0, w_max)
            model_h.changeColCost(idx_w, returns_list[i])  # 收益部分
            w_cols[i] = idx_w

        for i in range(n):
            idx_y = model_h.addVar(0.0, 1.0)
            model_h.changeColCost(idx_y, 0.0)
            y_cols[i] = idx_y

        # 风险惩罚：对每只股票加上 -λ * σ[i,i] 作为成本的线性近似
        # 这显然不完美（忽略了交叉项），但至少目标函数有意义
        for i in range(n):
            current_cost = model_h.getColCost(w_cols[i])
            risk_penalty = -lam * cov_matrix[i, i]
            model_h.changeColCost(w_cols[i], current_cost + risk_penalty)

        num_vars = model_h.getNumCol()
        print(f"  变量数: {num_vars} (LP松弛, 全部连续)")

        # --- 约束 ---
        # (1) 预算: Σ w[i] = 1
        w_idx_list = [w_cols[i] for i in range(n)]
        ones = [1.0] * n
        model_h.addRow(1.0, 1.0, n, w_idx_list, ones)

        # (2) 选股数量: Σ y[i] = k
        y_idx_list = [y_cols[i] for i in range(n)]
        model_h.addRow(float(k), float(k), n, y_idx_list, ones)

        # (3) Linking: w[i] - y[i] ≤ 0  =>  w[i] ≤ y[i]
        for i in range(n):
            model_h.addRow(
                -highspy.kHighsInf, 0.0,
                2,
                [w_cols[i], y_cols[i]],
                [1.0, -1.0]
            )

        # (4) 最小权重: w[i] - w_min * y[i] ≥ 0
        for i in range(n):
            model_h.addRow(
                0.0, highspy.kHighsInf,
                2,
                [w_cols[i], y_cols[i]],
                [1.0, -w_min]
            )

        print(f"  约束数: {model_h.getNumRows()}")
        print(f"\n  求解中...")

        start_time = time.time()
        model_h.run()
        solve_time = time.time() - start_time

        status = model_h.getModelStatus()
        status_map_h = {
            highspy.HighsModelStatus.kOptimal: "最优解 (LP松弛)",
            highspy.HighsModelStatus.kInfeasible: "不可行",
            highspy.HighsModelStatus.kTimeLimit: "超时(次优解)",
        }
        status_str = status_map_h.get(
            status, f"状态码: {status}"
        )

        print(f"\n  求解状态: {status_str}")
        print(f"  求解耗时: {solve_time:.1f}s")

        if status == highspy.HighsModelStatus.kOptimal:
            w_vals = np.array(
                [model_h.getSolution().col_value[w_cols[i]]
                 for i in range(n)]
            )
            y_vals = np.array(
                [model_h.getSolution().col_value[y_cols[i]]
                 for i in range(n)]
            )
        else:
            w_vals = np.zeros(n)
            y_vals = np.zeros(n)

        is_lp_relaxation = True

    # ============================================================
    # 结果分析与去噪
    # ============================================================

    # 去噪：过滤数值极小的权重（视为 0）
    w_clean = np.where(w_vals > 1e-4, w_vals, 0.0)
    # 重新归一化
    w_sum = np.sum(w_clean)
    if w_sum > 1e-10:
        w_clean = w_clean / w_sum
    else:
        # 如果所有值为 0，uniform 均分
        w_clean = np.ones(n_stocks) / n_stocks

    selected_indices = np.where(w_clean > 1e-6)[0]
    n_selected = len(selected_indices)

    # 计算统计量
    exp_return, volatility, sharpe = compute_portfolio_stats(
        w_clean, returns, cov_matrix, rf
    )

    exp_return_pct = exp_return * 100
    volatility_pct = volatility * 100

    print(f"\n  {'─'*50}")
    print(f"  组合结果")
    print(f"  {'─'*50}")
    print(f"  实际选中股票数: {n_selected}")
    print(f"  组合预期年化收益: {exp_return_pct:.2f}%")
    print(f"  组合年化波动率:   {volatility_pct:.2f}%")
    print(f"  Sharpe 比率 (rf={rf*100:.1f}%): {sharpe:.3f}")

    # 打印选中的股票权重（前 10 大持仓）
    sorted_idx = np.argsort(w_clean)[::-1]
    print(f"\n  前 10 大持仓:")
    for rank, idx in enumerate(sorted_idx[:10]):
        if w_clean[idx] > 1e-6:
            print(f"    {rank+1:2d}. 股票 #{idx:3d}  "
                  f"收益={returns[idx]*100:5.2f}%  "
                  f"权重={w_clean[idx]*100:5.2f}%  "
                  f"波动={np.sqrt(cov_matrix[idx,idx])*100:5.2f}%")

    return dict(
        weights=w_clean,
        y_vals=y_vals,
        n_selected=n_selected,
        exp_return=exp_return,
        volatility=volatility,
        sharpe=sharpe,
        solve_time=solve_time,
        status=status_str,
        is_lp_relaxation=is_lp_relaxation,
        returns=returns,
        cov_matrix=cov_matrix,
        rf=rf,
    )


# ============================================================
# 对比方案构建
# ============================================================

def build_comparison(returns, cov_matrix, rf, k, seed=99):
    """
    构建三组对比方案。

    返回:
        list of dict: 每个方案的 {'name', 'exp_return', 'volatility',
                       'sharpe', 'weights'}
    """
    n = len(returns)
    rng = np.random.default_rng(seed)

    results = []

    # --- 方案 1: 等权-全选 (Baseline) ---
    w_all = np.ones(n) / n
    er, vol, sh = compute_portfolio_stats(w_all, returns, cov_matrix, rf)
    results.append({
        'name': '等权-全选',
        'exp_return': er, 'volatility': vol, 'sharpe': sh,
        'weights': w_all, 'desc': f'{n}只全买，每只 {100/n:.1f}%'
    })

    # --- 方案 2: 等权-随机 ---
    rng_local = np.random.default_rng(seed + 1)
    chosen = rng_local.choice(n, size=k, replace=False)
    w_random = np.zeros(n)
    w_random[chosen] = 1.0 / k
    er, vol, sh = compute_portfolio_stats(w_random, returns, cov_matrix, rf)
    results.append({
        'name': '等权-随机',
        'exp_return': er, 'volatility': vol, 'sharpe': sh,
        'weights': w_random, 'desc': f'随机选{k}只，每只 {100/k:.1f}%'
    })

    # --- 方案 3: MIQP 优化 (通过 solve_portfolio_miqp 获取) ---
    # 这里对比时直接用 MIQP 的结果
    # solve_portfolio_miqp 返回的 dict 包含了所需字段
    # 不在此重复调用，由主程序传入

    return results


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  案例3：投资组合优化 (MIQP)                                 ║
║  求解器: Gurobi (MIQP) / HiGHS (LP松弛备选)                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    # --- 小规模测试: 100 只, 选 10 ---
    print("\n### 小规模测试 (100只, 选10) ###")
    result_small = solve_portfolio_miqp(
        n_stocks=100, k=10, lam=1.0, w_max=0.20, w_min=0.02,
        seed=42
    )

    print("\n")
    # --- 中等规模测试: 500 只, 选 20 ---
    print("### 中等规模测试 (500只, 选20) ###")
    result_main = solve_portfolio_miqp(
        n_stocks=500, k=20, lam=1.0, w_max=0.15, w_min=0.01,
        seed=42
    )

    # --- 构建对比方案 ---
    if result_main is not None:
        returns = result_main['returns']
        cov_matrix = result_main['cov_matrix']
        rf = result_main['rf']
        k = 20
        n_stocks = len(returns)

        comparison = build_comparison(returns, cov_matrix, rf, k, seed=99)

        # 添加 MIQP 优化方案
        comparison.append({
            'name': 'MIQP 优化',
            'exp_return': result_main['exp_return'],
            'volatility': result_main['volatility'],
            'sharpe': result_main['sharpe'],
            'weights': result_main['weights'],
            'desc': f'MIQP选{k}只，权重优化'
        })

        # --- 三方案对比表 ---
        print(f"\n{'═'*60}")
        print(f"  三方案对比 ({n_stocks}只, 选{k})")
        print(f"{'═'*60}")
        print(f"  {'方案':<16s} {'收益':>8s} {'波动率':>8s} {'Sharpe':>8s}  {'说明'}")
        print(f"  {'─'*60}")
        for c in comparison:
            print(f"  {c['name']:<16s} "
                  f"{c['exp_return']*100:>7.2f}% "
                  f"{c['volatility']*100:>7.2f}% "
                  f"{c['sharpe']:>7.3f}  "
                  f"{c['desc']}")
        print(f"  {'─'*60}")

        # 计算改进幅度
        baseline_sharpe = comparison[0]['sharpe']  # 等权-全选
        miqp_sharpe = comparison[2]['sharpe']      # MIQP 优化
        improvement = (miqp_sharpe / baseline_sharpe - 1) * 100

        print(f"\n  MIQP 优化 vs 等权-全选 (Baseline):")
        print(f"    Sharpe: {baseline_sharpe:.3f} → {miqp_sharpe:.3f} "
              f"(提升 {improvement:+.0f}%)")
        print(f"    等权-全选: 超额收益 {baseline_sharpe:.3f}×波动率 "
              f"= {(comparison[0]['exp_return']-rf)*100:.2f}%")
        print(f"    MIQP 优化: 超额收益 {miqp_sharpe:.3f}×波动率 "
              f"= {(comparison[2]['exp_return']-rf)*100:.2f}%")

        # 业务价值换算
        fund_size = 1e8  # 1亿
        extra_return = (comparison[2]['exp_return']
                        - comparison[0]['exp_return'])
        extra_profit = fund_size * extra_return

        print(f"\n  业务价值 (管理规模 1 亿):")
        print(f"    等权-全选年收益: {comparison[0]['exp_return']*100:.2f}% "
              f"= {fund_size*comparison[0]['exp_return']/1e4:.0f} 万")
        print(f"    MIQP 优化年收益: {comparison[2]['exp_return']*100:.2f}% "
              f"= {fund_size*comparison[2]['exp_return']/1e4:.0f} 万")
        print(f"    每年多赚: {extra_profit/1e4:.0f} 万")
        print(f"    同时波动率: {comparison[0]['volatility']*100:.1f}% → "
              f"{comparison[2]['volatility']*100:.1f}% "
              f"(降低 {(1-comparison[2]['volatility']/comparison[0]['volatility'])*100:.0f}%)")

        if result_main['is_lp_relaxation']:
            print(f"\n  ⚠️  以上 MIQP 优化结果为 LP 松弛，仅用于演示。")
            print(f"     实际 MIQP 结果需要 Gurobi 许可。")
            print(f"     参见 05-case-portfolio.md 第10节获取许可指南。")

    print(f"\n{'═'*60}")
    print(f"  完成。")
    print(f"  更多解释见: 05-case-portfolio.md")
    print(f"{'═'*60}\n")
