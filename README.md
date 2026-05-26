# 📚 运筹学与优化方法课程系列
# Operations Research & Optimization Course Series

> 9 门课，从概率论到多目标优化——一套完整的自学教程体系。
> 9 courses, from probability to multi‑objective optimization — a complete self‑study curriculum.
> 共 **200+ 个文件**，**~50 个 Python 教学案例**，**~100 篇文档**。| **200+ files**, **~50 Python tutorials**, **~100 docs**.

---

## 🌐 双语目录 / Bilingual Navigation

> 💡 **中文名和英文名指向同一门课**。英文名是符号链接（symlink），在 GitHub / macOS / Linux 下都正常工作。`cd probability/` = `cd 01-概率论与数理统计/`。
> **Chinese & English names point to the same course.** English names are symlinks. Works on GitHub / macOS / Linux.

| 中文目录 / CN Dir | English Symlink | 课程 / Course | 难度 | 核心内容 | 文件 |
|---|---|---|---|---|---|
| `01-概率论与数理统计/` | `probability/` | 📈 概率论与数理统计<br>Probability & Statistics | ★★~★★★★ | 6大分布、CLT、贝叶斯、蒙特卡洛 | 20 |
| `02-低复杂度算法设计/` | `low-complexity-algorithm/` | 🧮 低复杂度算法设计<br>Low‑Complexity Algorithm Design | ★★~★★★★ | DP、贪心、分治、二分、滑动窗口 | 20 |
| `03-PDE物理方程求解/` | `pde-simulation/` | 📐 物理方程求解<br>PDE Simulation | ★★~★★★★ | FDM、热传导、波动、Burgers、Laplace | 20 |
| `04-MIP求解器技术/` | `mip-solver/` | 🔧 MIP求解器技术<br>MIP Solver Technology | ★★~★★★★ | 分支定界、割平面、VRP、排程、投资组合 | 27 |
| `05-运筹学基础与实战/` | `operations-research/` | 📊 运筹学基础与实战<br>Operations Research | ★★~★★★★ | LP、MIP、排队论、仿真、元启发式、Pyomo | 28 |
| `06-博弈论基础与实战/` | `game-theory/` | ♟️ 博弈论基础与实战<br>Game Theory | ★★~★★★★ | 纳什均衡、拍卖、谈判、合作博弈、信号传递 | 20 |
| `07-库存与供应链管理/` | `supply-chain/` | 📦 库存与供应链管理<br>Supply Chain Management | ★★~★★★★ | EOQ、报童、牛鞭效应、契约设计、网络设计 | 20 |
| `08-随机规划与鲁棒优化/` | `stochastic-optimization/` | 🎲 随机规划与鲁棒优化<br>Stochastic & Robust Optimization | ★★★~★★★★ | 两阶段、场景生成、不确定性集合、分布鲁棒 | 28 |
| `09-多目标优化/` | `multiobjective-optimization/` | 🎯 多目标优化<br>Multi‑Objective Optimization | ★★~★★★★ | ε-约束、NSGA-II、Pareto前沿、目标规划 | 24 |

---

## 课程总览 / Course Overview

```
                            ┌──────────────────────────────────┐
                            │      📈 概率论 / Probability      │
                            │    (所有课程的数学基础 / Math Foundation)  │
                            └──────────────┬───────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
     ┌────────────────┐         ┌──────────────────┐      ┌──────────────────────┐
     │ 🧮 算法设计    │         │ 📐 PDE 物理方程   │      │                      │
     │ Algo Design    │         │ PDE Simulation    │      │                      │
     │ ★★☆☆☆~★★★★☆ │         │ ★★☆☆☆~★★★★☆   │      │                      │
     └───────┬────────┘         └────────┬─────────┘      └──────────────────────┘
             │                           │
             └───────────┬───────────────┘
                         ▼
              ┌──────────────────────┐
              │  📊 运筹学 / OR      │
              │  核心 / Core:        │
              │  LP/MIP/Network Flow │
              │  ★★☆☆☆~★★★★☆      │
              └──┬───────┬───────┬───┘
                 │       │       │
     ┌───────────┼───────┼───────┼───────────┐
     ▼           ▼       ▼       ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│ 🔧 MIP │ │ ♟️ 博弈 │ │ 📦 供  │ │ 🎲 随  │ │ 🎯 多目标    │
│ Solver │ │ Game   │ │ 应链   │ │ 机规划 │ │ Multi‑Obj    │
│ ★★~★★★★│ │ ★★~★★★★│ │ ★★~★★★★│ │ ★★★~★★★★│ │ ★★~★★★★     │
└────────┘ └────────┘ └────────┘ └────────┘ └──────────────┘
```

---

## 每门课一句话 / One Sentence Per Course

| 课程 / Course | 一句话 / TL;DR |
|---|---|
| 📈 概率论 / Probability | 随机世界的地图——所有不确定决策的数学基础<br>*The map of randomness — math foundation for all decisions under uncertainty* |
| 🧮 算法设计 / Algo Design | 不只是背 Big-O，而是学会把暴力解系统地优化到低复杂度<br>*Beyond Big-O: systematically optimize brute‑force to low complexity* |
| 📐 PDE / PDE Simulation | 从热传导到流体——用数值方法让物理方程可视化<br>*From heat to fluids — visualize physics with numerical methods* |
| 🔧 MIP / MIP Solver | 从零到能用 Python 写工业级整数规划模型<br>*From zero to writing industrial‑grade MIP models in Python* |
| 📊 运筹学 / OR | 业务问题 → 数学模型 → 算法求解的完整方法论<br>*Business problem → math model → algorithmic solution* |
| ♟️ 博弈论 / Game Theory | 当你的最佳选择取决于别人的选择——策略互动的数学<br>*When your best move depends on theirs — the math of strategic interaction* |
| 📦 供应链 / Supply Chain | 库存、运输、牛鞭效应——供应链优化的核心模型<br>*Inventory, logistics, bullwhip — core models of supply chain optimization* |
| 🎲 随机规划 / Stochastic | 当未来不确定但你必须现在决策<br>*When the future is uncertain but you must decide now* |
| 🎯 多目标 / Multi‑Obj | 不是找一个最优解，而是画一条「不能再更好」的前沿<br>*Not one best solution, but a frontier where nothing can be improved without tradeoffs* |

---

## 学习路径 / Learning Paths

### 从零开始 / From Scratch（推荐 / Recommended）
```
01-概率论 → 02-算法设计 → 05-运筹学 → 06-博弈论 → 07-供应链 → 09-多目标
probability → low-complexity-algorithm → operations-research → game-theory → supply-chain → multiobjective-optimization
```

### 面试/竞赛 / Interviews & Contests
```
02-算法设计 → 05-运筹学（LP/网络流/DP）→ 06-博弈论
low-complexity-algorithm → operations-research → game-theory
```

### 工业应用 / Industrial Applications
```
01-概率论 → 05-运筹学 → 07-供应链 → 08-随机规划 → 09-多目标
probability → operations-research → supply-chain → stochastic-optimization → multiobjective-optimization
```

### 研究方向 / Research Track
```
01-概率论 → 03-PDE → 05-运筹学 → 08-随机规划 → 09-多目标
probability → pde-simulation → operations-research → stochastic-optimization → multiobjective-optimization
```

### 求解器深度 / Solver Deep Dive
```
01-概率论 → 05-运筹学 → 04-MIP求解器
probability → operations-research → mip-solver
```

---

## 快速开始 / Quick Start

```bash
# 中文 / Chinese（真实目录 / real directories）
cd 01-概率论与数理统计/ && cat README.md
cd 02-低复杂度算法设计/   && cat README.md
cd 03-PDE物理方程求解/    && cat README.md
cd 04-MIP求解器技术/      && cat README.md
cd 05-运筹学基础与实战/   && cat README.md
cd 06-博弈论基础与实战/   && cat README.md
cd 07-库存与供应链管理/   && cat README.md
cd 08-随机规划与鲁棒优化/ && cat README.md
cd 09-多目标优化/         && cat README.md

# English（符号链接 / symlinks — 同一内容 / same content）
cd probability/                  && cat README.md
cd low-complexity-algorithm/     && cat README.md
cd pde-simulation/               && cat README.md
cd mip-solver/                   && cat README.md
cd operations-research/          && cat README.md
cd game-theory/                  && cat README.md
cd supply-chain/                 && cat README.md
cd stochastic-optimization/      && cat README.md
cd multiobjective-optimization/  && cat README.md
```

---

## 预备知识 / Prerequisites

| 课程 / Course | 需要的预备知识 / Prerequisites |
|---|---|
| 📈 概率论 / Probability | 高中数学（排列组合、函数）/ *High school math* |
| 🧮 算法设计 / Algo Design | Python/C++ 基础 + 基本数据结构 / *Basic programming + data structures* |
| 📐 PDE / PDE Simulation | 高等数学（微积分、线性代数）+ Python / *Calculus + linear algebra + Python* |
| 🔧 MIP / MIP Solver | 高中数学 + Python / *High school math + Python* |
| 📊 运筹学 / OR | 高中数学 + Python / *High school math + Python* |
| ♟️ 博弈论 / Game Theory | 高中数学 + Python / *High school math + Python* |
| 📦 供应链 / Supply Chain | 高中数学 + Python / *High school math + Python* |
| 🎲 随机规划 / Stochastic | 概率论 + 运筹学基础 / *Probability + OR basics* |
| 🎯 多目标 / Multi‑Obj | 运筹学基础 + Python / *OR basics + Python* |

---

## 一键运行 / One‑Click Run

```bash
make          # 列出所有代码 / List all code
make python   # 运行所有 Python 验证 / Run all Python
make clean    # 清理缓存 / Clean cache
```

---

## FAQ

**Q: 中文和英文目录有什么区别？ / What's the difference?**  
没区别——英文名是符号链接，指向同一个中文目录。`cd probability/` 和 `cd 01-概率论与数理统计/` 完全等价。
*No difference — English names are symlinks pointing to Chinese dirs. `cd probability/` = `cd 01-概率论与数理统计/`.*

**Q: 符号链接在 Windows 上能用吗？ / Do symlinks work on Windows?**  
Git Bash / WSL 可以。原生 cmd 需开启开发者模式。中文名作为后备。
*Yes in Git Bash / WSL. Native cmd needs Developer Mode. Use Chinese names as fallback.*

**Q: 我需要按顺序学吗？ / Must I follow the order?**  
不必须。但概率论是随机规划的数学基础，运筹学是多目标和 MIP 的前提。
*No. But probability is prerequisite for stochastic optimization; OR is prerequisite for multi‑obj and MIP.*

**Q: 每门课要学多久？ / How long per course?**  
设计目标「一个周末」（约 12-16 小时）。
*Designed for "one weekend" (~12-16 hrs).*

**Q: 原来那些独立 repo 呢？ / What about the old standalone repos?**  
已全部合并到此。`course-series`、`probability-course` 等将归档。
*All merged here. Standalone repos to be archived.*
