# 📚 运筹学与优化方法课程系列

> 9 门课，从概率论到多目标优化——一套完整的自学教程体系。
> 共 **200+ 个文件**，**~50 个 Python 教学案例**，**~100 篇文档**。

---

## 课程总览

```
                            ┌──────────────────────────────────┐
                            │      📈 概率论与数理统计          │
                            │  (所有课程的数学基础)               │
                            └──────────────┬───────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
     ┌────────────────┐         ┌──────────────────┐      ┌──────────────────────┐
     │ 🧮 算法设计    │         │ 📐 PDE 物理方程   │      │ 数学基础             │
     │ ★★☆☆☆~★★★★☆ │         │ ★★☆☆☆~★★★★☆   │      │                      │
     └───────┬────────┘         └────────┬─────────┘      └──────────────────────┘
             │                           │
             └───────────┬───────────────┘
                         ▼
              ┌──────────────────────┐
              │  📊 运筹学基础与实战  │
              │  核心：LP/MIP/网络流  │
              │  ★★☆☆☆~★★★★☆      │
              └──┬───────┬───────┬───┘
                 │       │       │
     ┌───────────┼───────┼───────┼───────────┐
     ▼           ▼       ▼       ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│ 🔧 MIP │ │ ♟️ 博弈 │ │ 📦 供  │ │ 🎲 随  │ │ 🎯 多目标    │
│ 求解器 │ │ 论     │ │ 应链   │ │ 机规划 │ │ 优化         │
│ ★★~★★★★│ │ ★★~★★★★│ │ ★★~★★★★│ │ ★★★~★★★★│ │ ★★~★★★★     │
└────────┘ └────────┘ └────────┘ └────────┘ └──────────────┘
```

---

## 快速导航

| 目录 | 课程 | 难度 | 核心内容 | 文件数 |
|------|------|------|---------|--------|
| `probability/` | 📈 概率论与数理统计 | ★★~★★★★ | 6大分布、CLT、贝叶斯、假设检验、蒙特卡洛 | 20 |
| `low-complexity-algorithm/` | 🧮 低复杂度算法设计 | ★★~★★★★ | DP、贪心、分治、二分、滑动窗口、单调栈 | 20 |
| `pde-simulation/` | 📐 物理方程求解 | ★★~★★★★ | FDM、热传导、波动、Burgers、Laplace、梁 | 20 |
| `mip-solver/` | 🔧 MIP求解器技术 | ★★~★★★★ | 分支定界、割平面、VRP、排程、投资组合 | 27 |
| `operations-research/` | 📊 运筹学基础与实战 | ★★~★★★★ | LP、MIP、排队论、仿真、元启发式、Pyomo | 28 |
| `game-theory/` | ♟️ 博弈论基础与实战 | ★★~★★★★ | 纳什均衡、拍卖、谈判、合作博弈、信号传递 | 20 |
| `supply-chain/` | 📦 库存与供应链管理 | ★★~★★★★ | EOQ、报童、牛鞭效应、契约设计、网络设计 | 20 |
| `stochastic-optimization/` | 🎲 随机规划与鲁棒优化 | ★★★~★★★★ | 两阶段、场景生成、不确定性集合、分布鲁棒 | 28 |
| `multiobjective-optimization/` | 🎯 多目标优化 | ★★~★★★★ | ε-约束、NSGA-II、Pareto前沿、目标规划 | 24 |

---

## 每门课一句话

| 课程 | 一句话 |
|------|--------|
| 📈 概率论 | 随机世界的地图——所有不确定决策的数学基础 |
| 🧮 算法设计 | 不只是背 Big-O，而是学会把暴力解系统地优化到低复杂度 |
| 📐 PDE 物理方程 | 从热传导到流体——用数值方法让物理方程可视化 |
| 🔧 MIP 求解器 | 从零到能用 Python 写工业级整数规划模型 |
| 📊 运筹学 | 业务问题 → 数学模型 → 算法求解的完整方法论 |
| ♟️ 博弈论 | 当你的最佳选择取决于别人的选择——策略互动的数学 |
| 📦 供应链 | 库存、运输、牛鞭效应——供应链优化的核心模型 |
| 🎲 随机规划 | 当未来不确定但你必须现在决策 |
| 🎯 多目标优化 | 不是找一个最优解，而是画一条「不能再更好」的前沿 |

---

## 学习路径

### 从零开始（推荐）
```
probability → low-complexity-algorithm → operations-research → game-theory → supply-chain → multiobjective-optimization
```

### 面试/竞赛
```
low-complexity-algorithm → operations-research（LP/网络流/DP）→ game-theory
```

### 工业应用
```
probability → operations-research → supply-chain → stochastic-optimization → multiobjective-optimization
```

### 研究方向
```
probability → pde-simulation → operations-research → stochastic-optimization → multiobjective-optimization
```

### 求解器深度
```
probability → operations-research → mip-solver
```

---

## 每门课快速开始

```bash
# 概率论与数理统计
cd probability/ && cat README.md

# 低复杂度算法设计
cd low-complexity-algorithm/ && cat README.md

# 物理方程求解
cd pde-simulation/ && cat README.md

# MIP 求解器技术
cd mip-solver/ && cat README.md

# 运筹学基础与实战
cd operations-research/ && cat README.md

# 博弈论
cd game-theory/ && cat README.md

# 供应链管理
cd supply-chain/ && cat README.md

# 随机规划与鲁棒优化
cd stochastic-optimization/ && cat README.md

# 多目标优化
cd multiobjective-optimization/ && cat README.md
```

---

## 预备知识

| 课程 | 需要的预备知识 |
|------|---------------|
| 📈 概率论 | 高中数学（排列组合、函数） |
| 🧮 算法设计 | Python/C++ 基础 + 基本数据结构（数组、链表、树） |
| 📐 PDE 物理方程 | 高等数学（微积分、线性代数）+ Python |
| 🔧 MIP 求解器 | 高中数学 + Python |
| 📊 运筹学 | 高中数学 + Python |
| ♟️ 博弈论 | 高中数学 + Python |
| 📦 供应链 | 高中数学 + Python |
| 🎲 随机规划 | 概率论 + 运筹学基础 |
| 🎯 多目标优化 | 运筹学基础 + Python |

---

## 常见问题

**Q: 我需要按顺序学吗？**  
不必须。但概率论是随机规划和随机优化的基础，运筹学是多目标优化和 MIP 求解器的前提。如果你想跳着学，先确认前置知识够了。

**Q: 每门课要学多久？**  
设计目标都是「一个周末」（约 12-16 小时）。每门课 8 章 + 案例 + 附录，按你自己节奏来。

**Q: 代码能跑吗？**  
能。每门课都配有 Python 代码示例（主要在 `code/` 子目录），依赖在各自 README 里说明。

**Q: 这个 monorepo 和原来的独立 repo 有什么关系？**  
这是所有课程的统一入口。原来的独立 repo（`probability-course`、`game-theory-course` 等）已合并到此 monorepo，后续更新都在这里进行。
