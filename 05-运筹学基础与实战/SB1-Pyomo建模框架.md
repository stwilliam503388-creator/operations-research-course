# SB1：Pyomo 建模框架 — 从手写到框架

> 难度 ★★☆☆☆ · 建模工具 · 求解器无关 · 代码：`code/case01b_pyomo_transport.py` · 预计阅读：20min

---

## 1 场景

你已经在 `03-case-logistics.md` 里用手写 MODI 算法解了运输问题。现在换一个问法：

> **如果业务方说"再加一个约束——每个仓库最多服务 30 个客户"，你要改多少代码？**

手写 MODI 算法——你得改算法逻辑，加约束传播，重新推导对偶变量。至少半天。
Pyomo 建模框架——加一行 `m.res.add(...)`，搞定。两分钟。

这就是建模框架的价值：**把「改模型」的成本从「改算法」降低到「改声明」**。

---

## 2 分析：业务→工具映射

| 层面 | 手写 MODI 算法 | Pyomo 建模框架 |
|------|---------------|---------------|
| 修改模型 | 改算法逻辑（高风险） | 加一行约束（低风险）|
| 换求解器 | 换算法实现 | 改一行 `SolverFactory('...')` |
| 代码量 | 400+ 行 | ~80 行 |
| 可读性 | 只有作者能维护 | 任何人能读 |
| 适用问题 | 仅运输问题 | 任意 LP/MIP/NLP |

**核心直觉**：Pyomo 让你用数学公式的写法写优化模型。你不需要关心求解器怎么解——你只关心模型对不对。

---

## 3 核心思路

### 3.1 Pyomo 建模五要素

```
from pyomo.environ import *

model = ConcreteModel()

# 1. 集合 (Set)     → 索引维度（工厂、仓库、客户、时间…）
# 2. 参数 (Param)   → 已知数据（运费、产能、需求…）
# 3. 变量 (Var)     → 决策变量（连续/整数/0-1）
# 4. 目标 (Objective) → 最大化/最小化
# 5. 约束 (Constraint) → 限制条件
```

### 3.2 从数学公式到 Pyomo 代码

**数学公式**：
```
min  Σᵢ Σⱼ c_ij · x_ij
s.t. Σⱼ x_ij ≤ supply_i    ∀i
     Σᵢ x_ij = demand_j     ∀j
     x_ij ≥ 0
```

**Pyomo 代码**（逐行对应）：
```python
# 集合
model.I = pyo.RangeSet(0, n_factories-1)   # 工厂 i
model.J = pyo.RangeSet(0, n_warehouses-1)  # 仓库 j

# 参数
model.supply = pyo.Param(model.I, initialize={i: supply[i] for i in ...})
model.cost   = pyo.Param(model.I, model.J, initialize={(i,j): c[i][j] for ...})

# 变量
model.x = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals)

# 目标
model.obj = pyo.Objective(
    rule=lambda m: sum(m.cost[i,j] * m.x[i,j] for i in m.I for j in m.J),
    sense=pyo.minimize)

# 约束
def supply_rule(m, i):
    return sum(m.x[i,j] for j in m.J) <= m.supply[i]
model.supply_con = pyo.Constraint(model.I, rule=supply_rule)
```

### 3.3 三行换求解器

```python
# HiGHS（免费，开源）
SolverFactory('appsi_highs')

# 改成 Gurobi（商业，最快）
SolverFactory('gurobi')

# 改成 COPT（国产，第一梯队）
SolverFactory('copt')

# 改成 CBC（免费，开源）
SolverFactory('cbc')
```

---

## 4 代码实现概览

| 函数 | 作用 |
|------|------|
| `build_and_solve()` | 通用建模函数：建 LP → 求解 → 返回结果 |
| `model = ConcreteModel()` | 声明模型容器 |
| `SolverFactory('appsi_highs').solve(model)` | 调用 HiGHS 求解 |

完整代码见 `code/case01b_pyomo_transport.py`。

---

## 5 运行结果

```
求解状态: ok, optimal
最优总成本: 44,280.45

工厂1 → 仓库5: 100.0 单位  运费=32.5
工厂1 → 仓库6: 100.0 单位  运费=32.5
...
仓库1: 服务 26 个客户, 出货量 100
仓库2: 服务 1 个客户, 出货量 100
...
未服务客户数: 0 ✅
```

---

## ✅ 验证标准

1. 求解状态为 `optimal`
2. 所有客户至少被一个仓库覆盖（无遗漏）
3. 总出货量 ≤ 总供给量
4. 每个仓库进货 ≤ 仓库容量

---

## 6 关键洞察

### 洞察 1：建模框架的核心价值不是「快」，是「可改」

手写算法快？不一定。但**改模型时 Pyomo 比手写快 100 倍**。真实业务中模型平均改 5-10 次才定稿。框架胜在迭代成本。

### 洞察 2：模型 = 代码。版本控制它。

Pyomo 模型是纯 Python 代码，可以 git 管理、code review、单元测试。手写算法做不到——因为它是一个隐藏在编译器里的黑盒。

### 洞察 3：先手写一个，再用框架

学习顺序很重要：先手写 MODI 理解单纯形法原理，再用 Pyomo 提高效率。
**不要一上来就用框架——你无法 debug 你不理解的求解器。**

---

## 7 延伸

| 方向 | 怎么改 Pyomo 模型 |
|------|-----------------|
| 设施选址（加仓库固定成本） | 加 0/1 变量 `y_j` + Big-M 约束 |
| 多周期运输 | 加时间索引 `x[i,j,t]` |
| 容量可扩展 | 加整数变量 `z_j` 表示仓库扩建级数 |
| 随机需求 | 加场景索引 `x[i,j,s]` |

---

> 下一个: [SB2: SimPy 仿真](SB2-SimPy仿真.md)
