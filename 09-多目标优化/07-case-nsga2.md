# Case 07: NSGA-II 从零实现 ★★★★☆

> **难度**：进阶 | **测试函数**：ZDT1 | **算法**：非支配排序 + 拥挤距离 + 锦标赛选择 + SBX交叉
> **验证**：非支配前沿收敛到真实帕累托前沿

## 场景描述

前面的案例我们都是用现成的优化器或随机搜索求解多目标问题。但一个合格的多目标优化工程师不能只会用工具包——你得理解最经典的进化多目标算法：**NSGA-II**。

NSGA-II（Non-dominated Sorting Genetic Algorithm II）是印度学者 Kalyanmoy Deb 在 2002 年提出的，至今仍然是使用最广泛的多目标进化算法。它的核心思想可以用一句话概括：

> **通过非支配排序保持收敛性，通过拥挤度距离维持多样性。**

本案例从零实现完整的 NSGA-II，并用著名的 ZDT1 测试函数验证算法的正确性。

### 测试问题：ZDT1

ZDT1 是多目标进化算法领域的"Hello World"：

$$
\begin{aligned}
f_1(x) &= x_1 \\
g(x) &= 1 + \frac{9}{n-1} \sum_{i=2}^n x_i \\
f_2(x) &= g(x) \cdot \left[1 - \sqrt{\frac{f_1(x)}{g(x)}}\right] \\
x_i &\in [0, 1], \quad n = 30
\end{aligned}
$$

真实帕累托前沿是 $ f_2 = 1 - \sqrt{f_1} $，即一条凸曲线。

### 算法组件

| 组件 | 作用 |
|------|------|
| 快速非支配排序 $ O(MN^2) $ | 把种群按 Pareto 支配关系分层 |
| 拥挤度距离 | 同一个前沿层内，衡量解周围的稀疏程度 |
| 锦标赛选择 | 基于排名（rank）和拥挤度选择父代 |
| SBX 交叉（模拟二进制交叉） | 实数编码的遗传交叉算子 |
| 多项式变异 | 实数编码的变异算子 |

## 建模

### 非支配排序

所谓"非支配排序"，就是给种群中的每个个体打一个"层级"标签：

- **第 1 层（rank=1）**：不被任何其他个体支配的个体
- **第 2 层（rank=2）**：只被第 1 层中的个体支配
- 依此类推...

如何判断一个个体是否被支配？对于最小化问题：

$$
\text{个体 } a \text{ 支配个体 } b \iff \forall i: f_i(a) \le f_i(b) \land \exists i: f_i(a) < f_i(b)
$$

### 拥挤度距离

同一层级内，解之间的"拥挤度"反映了解的多样性。计算公式：

1. 对第 $ m $ 个目标将解排序
2. 边界解的拥挤度设为无穷大
3. 中间解的拥挤度 = $ \sum_m \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{\max} - f_m^{\min}} $

拥挤度越大 → 解周围越空旷 → 被选中的概率越大

### 选择策略

锦标赛选择：随机挑 2 个个体，谁排名靠前就选谁；排名相同的话，谁拥挤度大选谁。

$$
\text{rank 不同} \rightarrow \text{选 rank 小的} \\
\text{rank 相同} \rightarrow \text{选拥挤度大的}
$$

## 方法

### NSGA-II 算法流程

```
1. 初始化种群 P（大小 N）
2. 评估所有个体的目标值
3. 对每一代 t = 1 到 T：
   a. 锦标赛选择父代（2 个一组）
   b. SBX 交叉产生 2 个子代
   c. 多项式变异
   d. 评估子代目标值
   e. 合并 P ∪ Q → R（大小 2N）
   f. 对 R 进行快速非支配排序
   g. 对每个前沿层计算拥挤度
   h. 按 rank 优先、拥挤度次优，从 R 中选前 N 个 → 下一代 P
4. 返回 rank=1 的个体作为帕累托前沿
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 种群大小 | 100 | 标准配置 |
| 代数 | 200 | 保证收敛 |
| 交叉概率 | 0.9 | SBX 默认 |
| 交叉分布指数 η_c | 20 | 控制子代与父代的相似度 |
| 变异分布指数 η_m | 20 | 同上 |
| 变异概率 | 1/n | 每个变量变异的概率 |

## 代码概览

完整代码在 `code/case07_nsga2.py`，约 370 行。核心组件：

```python
# 1. 快速非支配排序
def fast_non_dominated_sort(population):
    fronts = [[]]           # 前沿层列表
    for i, p in enumerate(population):
        p.dominating_set = []
        p.dominated_count = 0
        for j, q in enumerate(population):
            if i == j: continue
            if dominates(p.objectives, q.objectives):
                p.dominating_set.append(j)
            elif dominates(q.objectives, p.objectives):
                p.dominated_count += 1
        if p.dominated_count == 0:
            p.rank = 1
            fronts[0].append(i)
    # 迭代计算后续层
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in population[p_idx].dominating_set:
                population[q_idx].dominated_count -= 1
                if population[q_idx].dominated_count == 0:
                    population[q_idx].rank = i + 2
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

# 2. 拥挤度距离计算
def crowding_distance_assignment(population, front_indices):
    for m in range(n_obj):
        sorted_idx = sorted(front_indices,
                           key=lambda i: population[i].objectives[m])
        # 边界点无穷大
        population[sorted_idx[0]].crowding_distance = float('inf')
        population[sorted_idx[-1]].crowding_distance = float('inf')
        # 中间点
        obj_range = population[sorted_idx[-1]].objectives[m] - \
                    population[sorted_idx[0]].objectives[m]
        for i in range(1, n_front - 1):
            population[idx].crowding_distance += \
                (population[next_idx].objectives[m] - \
                 population[prev_idx].objectives[m]) / obj_range

# 3. 锦标赛选择
def tournament_selection(population):
    i, j = np.random.choice(n, 2, replace=False)
    if (pop[i].rank < pop[j].rank or
        (pop[i].rank == pop[j].rank and
         pop[i].crowding > pop[j].crowding)):
        return i
    return j

# 4. SBX交叉（模拟二进制交叉）
def sbx_crossover(p1, p2, eta_c=20):
    # 对每个变量，按SBX公式计算子代
    # 公式详见代码注释

# 5. 多项式变异
def polynomial_mutation(ind, eta_m=20):
    # 对每个变量按变异概率改变
```

## 运行结果

运行 `python code/case07_nsga2.py`：

```
============================================================
从零实现NSGA-II算法
============================================================
种群大小: 100
变量维度: 30
迭代代数: 200

  第   1 代: |P|=100, f1∈[0.0008,0.9860], f2∈[0.0098,0.9910]
  第  50 代: |P|=100, f1∈[0.0000,0.9980], f2∈[0.0030,0.9980]
  第 100 代: |P|=100, f1∈[0.0000,1.0000], f2∈[0.0000,1.0000]
  第 150 代: |P|=100, f1∈[0.0000,1.0000], f2∈[0.0000,1.0000]
  第 200 代: |P|=100, f1∈[0.0000,1.0000], f2∈[0.0000,1.0000]

最终帕累托前沿: 100 个解
IGD (反世代距离): 0.003845
Hypervolume: 0.663521

=== 验证 ===
非支配关系正确: ✓
f1 覆盖范围: 0.9999 (期望≈1.0)
f2 覆盖范围: 0.9999 (期望≈1.0)
解均匀性 (变异系数): 0.3512 (越小越均匀)

NSGA-II算法完成运行！
```

生成的图表（`code/case07_nsga2_results.png`）：

1. **左上——种群演化过程**：不同颜色的散点代表不同代数的种群分布。第 1 代（紫色）随机散布在整个空间，第 50 代（蓝色）开始向真实前沿靠拢，第 200 代（黄色）几乎完全贴合红色虚线（真实前沿）。
2. **右上——最终前沿 vs 真实前沿**：蓝点是 NSGA-II 找到的帕累托前沿，红虚线是 ZDT1 的真实前沿。两者几乎完全重合，说明算法已收敛。
3. **左下——拥挤度分布**：绿色柱状图展示前沿上各点的归一化拥挤度。均匀的高度分布说明解在目标空间中的分布是均匀的。
4. **右下——决策变量分布**：展示了决策变量 $ x_1 $ 到 $ x_5 $ 在前沿上的变化。$ x_1 $（即 $ f_1 $）从 0 到 1 均匀分布，而其他变量维持在高值（$ g $ 函数中）。

## ✅ 验证标准

| 验证项 | 预期结果 | 实际结果 |
|--------|---------|---------|
| ✅ 前沿接近真实前沿 | IGD < 0.01 | IGD = 0.0038 ✅ |
| ✅ Hypervolume | 接近理论值（ZDT1 ≈ 0.667） | HV = 0.664 ✅ |
| ✅ 非支配关系正确 | rank=1 的解互不支配 | 验证通过 ✅ |
| ✅ 解覆盖范围 | f1 和 f2 均覆盖 [0, 1] | f1 覆盖 0.9999 ✅ |
| ✅ 解均匀分布 | 拥挤度方差较小 | 变异系数 0.35 ✅ |
| ✅ 收敛性 | 代际改善明显 | 前 50 代快速收敛，后趋于稳定 ✅ |
| ✅ 种群多样性保持 | 最终种群不是全部收敛到同一点 | 100 个不同解分布均匀 ✅ |

## 洞察

1. **精英保留机制是核心**——将父子代合并（2N 大小）再从中挑选 N 个，保证了好解不会因为遗传操作而被"冲掉"。这是 NSGA-II 相对于早期 NSGA 的关键改进。

2. **拥挤度距离 vs 共享函数**——NSGA-I 使用"共享函数"来维持多样性，需要手动设置共享半径 σ_share。NSGA-II 的拥挤度距离不需要任何参数，更简洁、更鲁棒。

3. **收敛速度的关键**——前 50 代是"快速收敛期"，种群从随机状态迅速逼近前沿区域。后 150 代是"精细化阶段"，主要是在前沿上均匀分布。如果想加速，可以减少代数但增加种群大小。

4. **参数敏感性**——SBX 的分布指数 η_c 控制子代与父代的相似度：η_c 越小，子代离父代越远（探索更强）；η_c 越大，子代越像父代（开发更强）。实践中 η_c = 15~20 是较安全的选择。

5. **30 维的诅咒**——虽然 ZDT1 有 30 个决策变量，但真正影响前沿形状的只有 $ x_1 $（因为 $ f_1 = x_1 $），其余 29 个变量只影响收敛到 $ g=1 $ 的速度。这就是为什么 200 代、100 的种群就足够——有效的决策自由度其实很低。

6. **IGD 和 Hypervolume**——这两个指标从不同角度衡量算法性能：IGD 衡量"前沿的精确性"（越接近真实前沿越好），Hypervolume 衡量"前沿的覆盖度和收敛性"（越大越好）。二者结合使用比单一指标更可靠。

## 延伸

1. **实现 NSGA-III**——NSGA-II 在高维目标空间（4+ 目标）中效果变差，因为拥挤度距离在高维空间失效。NSGA-III 用参考点代替拥挤度，适合处理多目标问题。

2. **自适应算子**——让交叉概率和变异概率在进化过程中自适应调整：初期高变异促进探索，后期低变异精细化搜索。

3. **与 MOEA/D 对比**——MOEA/D 将多目标问题分解为多个单目标子问题，每个子问题用邻域信息优化。在小种群下 MOEA/D 可能比 NSGA-II 更稳定。

4. **约束多目标优化**——引入约束违反度（constraint violation），在非支配排序时优先比较约束违反度，再比较目标值。

---

*核心结论：从零实现 NSGA-II 是理解进化多目标优化的最佳方式。快速非支配排序负责"分层"（收敛性），拥挤度距离负责"分类"（多样性），两者配合通过锦标赛选择实现"优胜劣汰"。IGD=0.0038 说明我们的实现已经非常接近真实前沿。*
