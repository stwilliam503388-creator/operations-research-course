# MIP求解器技术基础课程 — 详细提示词

> 此提示词用于生成一份完整的「MIP求解器技术基础课程」学习教程。直接复制给任意 LLM 使用即可。

---

## 角色设定

你是一位运筹优化领域的资深算法专家，拥有15年以上的混合整数规划（MIP）求解器研发与工程落地经验。你精通 Gurobi、CPLEX、SCIP、HiGHS 等主流求解器的内部原理，同时也深谙如何将数学优化模型转化为企业可落地的业务解决方案。你擅长用深入浅出的方式讲解复杂概念，并能用真实业务案例让学习者理解抽象算法的实际价值。

---

## 任务目标

请为我编写一份结构完整、内容详实的 **「MIP求解器技术基础课程」学习教程**，面向有一定数学基础（线性代数、微积分）和编程基础（Python）但未系统学过运筹优化的技术从业者。要求「教程性」强于「论文性」——重在循序渐进、由浅入深、概念清晰、案例驱动。

---

## 输出结构要求

教程必须严格按以下三大块组织，每块再细分章节：

---

### 第一部分：背景知识 — 理解MIP的数学与历史根基

#### 1.1 什么是数学优化
- 优化的本质定义：目标、约束、决策变量
- 连续优化 vs 离散优化的直观区别
- 线性规划 (LP)、整数规划 (IP)、混合整数规划 (MIP) 的定义与包含关系
- 一句话解释为什么 MIP 是「最难但最有用的优化子领域」

#### 1.2 历史演进
- 1947: Dantzig 单纯形法 — 线性规划的基石
- 1954: Dantzig-Fulkerson-Johnson 旅行商问题 — 离散优化的起点
- 1960: Land-Doig 分支定界算法
- 1984: Karmarkar 内点法
- 1990s-至今: 求解器工业化的黄金时代（CPLEX 商业化、Gurobi 崛起、SCIP 开源）
- 关键时间线图示（用文字表格呈现）

#### 1.3 数学基础速览
- 多面体理论与极点的几何直观（不做严格数学证明，重在直觉）
- 线性规划的对偶理论：对偶变量就是影子价格
- NP-hardness 的含义以及「为什么 MIP 可以实际求解」的矛盾性解释（worst-case vs. average-case）
- 松弛 (Relaxation) 概念：LP 松弛、拉格朗日松弛、代理松弛

#### 1.4 常见 MIP 模型范式
- 分配问题 (Assignment)
- 背包问题 (Knapsack)
- 集合覆盖/划分 (Set Covering/Partitioning)
- 设施选址 (Facility Location)
- 生产调度 (Scheduling)
- 网络流 (Network Flow)
- 每个范式给出：数学公式 + 一句话业务场景 + 变量规模量级参考

---

### 第二部分：MIP求解器技术入门 — 引擎盖下发生了什么

#### 2.1 MIP 求解器的整体架构
- 求解流程图（用 ASCII 或 mermaid 绘制）
- 六大核心组件概览：
  (1) Presolve 预处理
  (2) LP Relaxation Solver（LP 松弛求解器）
  (3) Branching Strategy（分支策略）
  (4) Cutting Plane Generation（割平面生成）
  (5) Primal Heuristics（原始启发式）
  (6) Node Selection & Search Management（节点选择与搜索管理）
- 各组件之间的协作关系与数据流

#### 2.2 Presolve 预处理：让模型「瘦身」
- 变量边界收紧 (Bound Tightening)
- 冗余约束删除 (Redundant Constraint Deletion)
- 系数化简 (Coefficient Reduction)
- 变量合并与替换 (Variable Substitution)
- 预处理对求解时间的典型影响（有数据支撑的对比表格）
- 一个具体例子：presolve 如何将一个看似复杂的模型缩减为等价小模型

#### 2.3 线性规划求解器：MIP 的发动机
- 为什么 LP 是 MIP 的核心：每一步都要解 LP 松弛
- 单纯形法核心思想（基变量、出入基操作、退化处理）
- 内点法核心思想（中心路径、牛顿步）
- 单纯形 vs 内点法适用场景对比表
- LP 求解器选择对 MIP 整体性能的影响——为什么 Gurobi 用双单纯形而 CPLEX 可选

#### 2.4 分支定界 (Branch & Bound)：搜索之树
- 原理：分裂、定界、剪枝三步循环
- 分支变量选择策略详解：
  - Strong Branching（强分支）
  - Pseudo-cost Branching（伪成本分支）
  - Reliability Branching（可靠性分支）— Gurobi 的实际默认策略
  - 混合策略（Hybrid Branching）
- 节点选择策略 (Best-First vs. Depth-First vs. Best-Projection/Estimate)
- 上界与下界的收敛过程图示
- Gap 的定义与含义 (MIP Gap = |BP-BD|/|BP|)

#### 2.5 割平面 (Cutting Planes)：切割不切问题
- 核心思想：不切掉任何整数可行解的前提下，切掉 LP 松弛最优解所在的分数区域
- 为什么要割：收紧 LP 松弛，缩小 feasible region
- Chvátal-Gomory Cut 原理
- Mixed-Integer Rounding (MIR) Cut
- Cover Cut (背包覆盖割)
- Clique Cut
- Implied Bound Cut
- Flow Cover Cut
- 割的生成与分离 (Separation) 问题
- 割平面的管控策略：什么时候停？太多了反而慢

#### 2.6 原始启发式 (Primal Heuristics)：快速找到可行解
- 为什么需要启发式：B&B 可能需要很久才能找到第一个可行解
- 主要启发式算法分类：
  - Rounding Heuristics（简单舍入、Feasibility Pump）
  - Diving Heuristics（Frac Diving, Coefficient Diving 等）
  - RINS (Relaxation Induced Neighborhood Search)
  - Local Branching
  - RENS (Relaxation Enforced Neighborhood Search)
  - Zero-Objective Heuristics
  - Sub-MIP Heuristics
- 每种启发式的核心思路（一句话 + 伪代码级描述）
- 启发式在求解器日志中的体现（如何读懂日志中的 H 标记）

#### 2.7 搜索管理 (Search Management)
- 搜索树剪枝的核心条件
- 重启策略 (Restart)
- 并行分支策略 (Concurrent Root LP, Distributed MIP)
- 对称性检测与处理 (Orbital Fixing, Symmetry Breaking Constraints)
- 终止条件：absolute gap, relative gap, time limit, node limit, solution limit

#### 2.8 现代求解器调优实战（以 Gurobi/CPLEX 为例）
- 最重要的 10 个参数及其含义（MIPFocus, Heuristics, Cuts, Presolve, Threads, MIPGap, TimeLimit, NodeLimit, Method, ScaleFlag）
- 三阶段调参方法论：
  - Phase 1: 让模型跑通（检查数值稳定性、scaling）
  - Phase 2: 找可行解优先（MIPFocus=1, 加强启发式）
  - Phase 3: 证明最优（MIPFocus=2/3, 加强割平面和分支）
- 常见性能陷阱：数值不稳定、对称性爆炸、big-M 太大、弱松弛界
- 求解器日志解读完整指南（逐行解读一条真实的 Gurobi log）

#### 2.9 开源求解器与商业求解器对比
- 商业：Gurobi, CPLEX, FICO Xpress, COPT
- 开源：SCIP, HiGHS, CBC, GLPK, OR-Tools CP-SAT
- 性能对比（公认的 Benchmark 结果汇总表）
- 开源求解器的定位与适用场景
- 学术 vs 商业许可指南

---

### 第三部分：业务案例 — 从问题到解决方案的完整链路

#### 3.1 案例一：物流配送路径优化 (VRP)
- 业务背景：电商最后一公里配送
- 数学模型：Capacitated VRP 的 MIP 形式化（带时间窗的扩展讨论）
- 模型亮点：子环消除约束 (Subtour Elimination) 的多种实现方式（MTZ, DFJ cut, lazy constraint callback）
- 实现：Python + Gurobi 完整代码（含注释）
- 基准运行结果：不同规模实例的求解时间与 Gap
- 关键洞察：为什么 30 个点可以精确求解，100 个点需要启发式
- 业务价值：从单车装载率 65% → 82% 的 ROI 量化分析

#### 3.2 案例二：生产排程优化
- 业务背景：半导体工厂的光刻机排程
- 数学模型：带序列依赖设置时间 (Sequence-Dependent Setup) 的 Job Shop Scheduling
- 核心建模技巧：
  - 时间索引 (Time-indexed) vs 析取 (Disjunctive) 建模的取舍
  - Big-M 的 Tight Value 选择方法
- 实现：Python + CPLEX 完整代码
- 基准结果数据
- 关键洞察：为什么添加 Symmetry Breaking 约束使求解提速 10x
- 业务价值：设备利用率提升 8% = 年化 $1.2M 成本节省

#### 3.3 案例三：投资组合优化
- 业务背景：量化对冲基金的组合构建
- 数学模型：带基数约束 (Cardinality Constraint) 的均值方差模型
- 模型亮点：处理非凸约束的方法（SOS1, indicator constraint, big-M）
- 实现：Python + Gurobi 完整代码
- 基准结果数据
- 关键洞察：为什么 Gurobi 的 MIQP 性能直接碾压手工启发式
- 业务价值：Sharpe Ratio 从 1.6 → 2.1 的绩效归因

#### 3.4 案例四：能源调度优化
- 业务背景：虚拟电厂的日前发电计划
- 数学模型：Unit Commitment (UC) 的 MIP 形式化
- 模型亮点：Ramping constraints, min up/down time constraints
- 实现：Python + HiGHS 开源方案（面向预算敏感场景）
- 基准结果数据
- 关键洞察：为什么 VPP 用 MIP 求解而不是启发式 —— 5% 的优化差距 = 百万级年损失
- 业务价值：日运营成本降低 12% 的量化分析

#### 3.5 案例五：网络设计优化
- 业务背景：电信运营商的骨干网扩容规划
- 数学模型：Multi-Commodity Flow Network Design
- 模型亮点：Benders 分解 vs 一体化 MIP 的性能对比（附运行时间数据）
- 大规模实例的处理技巧：Column Generation 简介
- 关键洞察：分解算法在特定结构问题上碾压直接 MIP 的数学原因
- 业务价值：CAPEX 节省 20% 的决策支持

---

## 输出质量要求

1. **语言**：中文，术语保留英文原名并在首次出现时加括号标注中文
2. **深度**：每个概念解释到「我知道为什么这样设计」的层次，而非仅仅「这是什么」
3. **代码**：每个案例的 Python 代码必须是完整可运行、有详细注释的（至少 50 行）
4. **数据**：所有性能对比和数据必须有具体的数字和时间（可以是合理估值的模拟数据，但需标注「模拟」）
5. **可视化**：关键流程用 ASCII 图示或 mermaid 绘制；对比数据用 Markdown 表格
6. **参考文献**：每部分末尾列出 2-3 个推荐阅读的经典论文或书籍
7. **预估总字数**：15,000-25,000 字

---

## 输出格式

Markdown 格式，层级结构严格遵循上述章节编号。

---

## 用户补充

[此处可自行添加额外要求，如特定行业案例、特定求解器偏好、受众的数学基础调整等]
