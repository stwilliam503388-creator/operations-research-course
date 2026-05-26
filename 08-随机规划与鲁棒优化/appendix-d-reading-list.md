<!-- 文件: stochastic-optimization-course/appendix-d-reading-list.md -->
# 附录D：推荐阅读

> 以下资源按「先读哪个」排了序。不要全读——**挑你需要的读**。

---

## 🟢 入门级（一个周末能读完）

| 资源 | 类型 | 为什么推荐 | 先读哪部分 |
|------|------|---------|----------|
| Birge & Louveaux — *Introduction to Stochastic Programming* | 教科书 | 随机规划领域标准教材，经典中的经典 | 前 5 章（两阶段、对等模型、Benders 分解） |
| Ben-Tal, El Ghaoui, Nemirovski — *Robust Optimization* | 教科书 | 鲁棒优化圣经，三人合著 | 第 1-3 章（基本概念、不确定性集合、对等模型） |
| Shapiro, Dentcheva, Ruszczyński — *Lectures on Stochastic Programming* | 教科书 | 偏理论但严谨，SAA 理论最强 | 第 1-3 章（建模、SAA 收敛性、两阶段） |

**阅读策略**：三本都太厚了。选一本作为主力教材：
- 想从实践出发 → Birge & Louveaux
- 主要用鲁棒优化 → Ben-Tal et al.
- 想理解理论保证 → Shapiro et al.

---

## 🟡 进阶级（读论文）

| 论文 | 为什么重要 | 一句话概括 |
|------|---------|----------|
| **Bertsimas & Sim (2004)** — *The Price of Robustness* | 预算不确定性的原始论文——工业界最常用的鲁棒建模方法 | 「不是所有参数同时最差，用预算 Γ 控制保守度」 |
| **Bertsimas & Sim (2003)** — *Robust Discrete Optimization* | 鲁棒优化在整数规划中的应用 | 「离散问题的鲁棒对等模型仍然是 MIP」 |
| **Mohajerin Esfahani & Kuhn (2018)** — *Data-driven DRO with Wasserstein Metric* | Wasserstein DRO 开山之作 | 「用 Wasserstein 球包围经验分布，提供 out-of-sample 保证」 |
| **Elmachtoub & Grigas (2021)** — *Smart Predict-then-Optimize* | DFL 方向奠基论文 | 「让 ML 模型直接优化下游决策损失（SPO loss）」 |

**阅读策略**：论文每条读 3 遍——
1. 读摘要 + 介绍（搞懂问题）
2. 读核心定理和例子（搞懂方法）
3. 读实验（搞懂效果）

---

## 🔵 实战级（代码/工具）

| 资源 | 类型 | 用途 |
|------|------|------|
| Gurobi 官方文档 — Stochastic Programming Examples | 代码示例 | 直接可运行的 Gurobi 随机规划案例 |
| Pyomo 官方文档 — Stochastic Programming | 框架 | 开源优化框架，支持随机规划建模 |
| SDDP.jl (Julia) | 库 | 随机对偶动态规划，适合多阶段水文/电力调度 |
| RSOME (Python) | 库 | 鲁棒优化建模库，支持多种不确定性集合 |

**几个有用的链接**：
- [OR StackExchange (or.stackexchange.com)](https://or.stackexchange.com) — 运筹优化问答社区，搜「stochastic programming」有海量实战讨论
- [GitHub - Stochastic Programming Examples](https://github.com/Gurobi/stochastic-programming-examples) — Gurobi 官方案例
- [RSOME 文档](https://xiongpengnus.github.io/rsome/) — 支持鲁棒优化和分布鲁棒优化的 Python 包

---

## 🟣 中文资源

| 资源 | 类型 | 推荐理由 |
|------|------|---------|
| 运筹OR帷幄 (知乎专栏 / 公众号) | 社区 | 中文运筹圈最活跃的社区，有大量入门文章 |
| 刘兴禄 — 《运筹优化常用模型、算法与案例实战》 | 教材 | 中文实践导向的运筹学教材，包含随机规划章节 |
| 知乎/CSDN 上的随机规划入门文章 | 文章 | 搜索「随机规划 入门」「鲁棒优化 实例」——质量参差不齐但数量多 |

---

## 🏆 本课程自学路线对照

```
读完本课程 → 入门教科书（选一本） → 对应案例的论文 → 实操代码
     ↓                                    ↓
周一读完                      周末读完
```

| 阶段 | 本课程对应文件 | 下一步阅读 |
|------|-------------|----------|
| 基础知识 | 01, 02 | Birge & Louveaux 第 1-5 章 |
| 新闻摊贩 | 03 | — |
| 两阶段生产 | 04 | Birge & Louveaux 第 3 章（两阶段） |
| 鲁棒投资组合 | 05 | Bertsimas & Sim (2004) |
| 供应链网络 | 06 | Benders 分解相关文献 |
| 分布鲁棒 | 07 | Mohajerin Esfahani & Kuhn (2018) |
| ML 交叉 | 附录 C | Elmachtoub & Grigas (2021) |

---

## 一句忠告

> **不要试图读完所有推荐资源。** 从你最需要解决的那个实际问题出发——读文档里能让你解决问题的那一章、论文里能让你写出代码的那几页就够了。理论的深潜，留到你想发论文的时候再做。

> 🆘 逃生通道：如果你不知道该从哪本开始读——问自己一个问题：「我是想先搞懂基本原理（读 Birge & Louveaux 前 3 章），还是想先找到一个能直接用的工具（读 Gurobi 案例）？」答案会告诉你去哪。

> [文件完]


