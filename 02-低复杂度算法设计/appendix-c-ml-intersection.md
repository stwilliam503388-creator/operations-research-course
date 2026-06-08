<!-- 文件: low-complexity-algo-course/appendix-c-ml-intersection.md -->

# 附录C：算法优化与 ML 的交叉点

> 算法优化和机器学习不是两个独立的世界。它们之间有四个最直接的交叉口。

---

## C.1 DP + 强化学习（DP + RL）

**连接点**：DP 的状态转移方程和 RL 的 Bellman 方程本质上是同一个东西。

| 概念 | DP 版本 | RL 版本 |
|------|---------|---------|
| **状态** | dp[i][j] | s (environment state) |
| **动作** | 转移选择 | a (action) |
| **转移** | dp[i] = f(dp[i-1]) | V(s) = max_a [R(s,a) + γV(s')] |
| **最优性** | 最优子结构 | Bellman Optimality |

**经典交叉案例**：
- **AlphaGo 的 MCTS + 价值网络**：树搜索（回溯+剪枝的变体）用神经网络来指导搜索方向，替代纯算法的启发式
- **路径规划**：传统用 DP（如 Floyd-Warshall）算全局最短路；RL 方法可以学到「动态避障」的局部策略

**实用建议**：
```
如果状态空间不大（≤ 10^6）→ 用 DP，确定性好，有最优保证
如果状态空间巨大且转移随机 → 考虑用 RL 学一个近似策略
两者的中间地带：用 DP 算确定性部分 + 用 ML 模型估计随机部分
```

## C.2 贪心 + 多臂老虎机（Greedy + Multi-Armed Bandit）

**连接点**：贪心算法的核心困境是「局部最优 ≠ 全局最优」。多臂老虎机算法是贪心算法的「聪明版本」——它在一定比例下做随机探索。

**ε-贪心算法**：
```python
# 传统贪心：永远选当前最优
def greedy():
    return argmax(scores)

# ε-贪心：以概率 ε 随机探索
def epsilon_greedy(scores, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(range(len(scores)))
    return argmax(scores)
```

**应用场景**：
- **广告推荐**：贪心给你看最热门的广告 → 新的好广告永远没机会曝光 → 用 MAB 算法（如 UCB、Thompson Sampling）平衡探索与利用
- **A/B 测试**：传统 A/B 测试固定分流量 → 用 MAB 动态分配流量到表现更好的版本

**实用建议**：
```
如果所有信息都在开始时已知 → 用传统贪心（可以证明最优）
如果信息在不断更新（在线学习）→ 用 MAB 算法
MAB 比纯贪心多付出的代价是「探索成本」，但避免了「错过最优」的风险
```

## C.3 二分答案 + 阈值搜索（Binary Search + Threshold Search）

**连接点**：二分答案的 `can_do(X)` 判定函数，在 ML 领域对应的是**阈值分类器**的评估。

**交叉案例**：
- **异常检测**：传统用「均值 + 3σ」作为阈值，ML 模型输出异常分数后，用二分搜索找到「使得 F1 最大」的阈值
- **模型校准**：输出概率的模型需要一个决策阈值，「概率 > 0.5 判为正类」不一定最优 → 二分搜索最优阈值

```python
def find_optimal_threshold(scores, labels):
    """找到使 F1 最大的概率阈值"""
    lo, hi = 0.0, 1.0
    best_f1, best_th = 0, 0.5
    for _ in range(50):  # 二分 50 次 ≈ 1e-15 精度
        mid = (lo + hi) / 2
        f1_mid = f1_score(labels, scores > mid)
        f1_mid_plus = f1_score(labels, scores > mid + 1e-4)
        if f1_mid >= f1_mid_plus:
            best_f1, best_th = f1_mid, mid
            hi = mid  # 阈值向右移动 → 更严格 → 试试更小的
        else:
            lo = mid
    return best_th
```

**注意**：这不一定是「标准二分」（因为 F1 可能不是单调的），但在实践中通常能找到局部最优。

## C.4 滑动窗口 + 特征工程（Sliding Window + Feature Engineering）

**连接点**：滑动窗口是时序特征工程的**核心操作**，而特征工程又是 ML 中最能提升模型效果的手段。

**常见滑动窗口特征**：

| 窗口统计量 | 用途 | 一次性计算复杂度 |
|-----------|------|----------------|
| 移动平均 | 平滑噪声，捕捉趋势 | O(n) 用前缀和 |
| 移动标准差 | 衡量波动性 | O(n) 用 Welford 算法 |
| 移动最大值/最小值 | 捕捉极值信号 | O(n) 用单调队列 |
| 移动相关系数 | 两个时间序列的相关性 | O(n) 用滑动协方差 |
| 移动熵 | 信息量变化 | O(n) 用滑动直方图 |

**案例**：量化交易中的技术指标
```python
# 用滑动窗口计算 20 日均线（移动平均）
def moving_average(prices, window=20):
    cumsum = [0]
    for p in prices:
        cumsum.append(cumsum[-1] + p)
    return [(cumsum[i] - cumsum[i-window]) / window
            for i in range(window, len(prices)+1)]
```

**实用建议**：
```
ML 特征工程中 80% 的时序特征可以用滑动窗口在 O(n) 内计算完成。
核心思想：一次扫描，维护多个窗口统计量。
这正是算法复杂度优化能力在 ML 特征工程中的直接应用。
```

---

## C.5 总结

| 算法范式 | ML 交叉点 | 适合场景 |
|---------|----------|---------|
| DP | 强化学习（Bellman 方程） | 序列决策、路径规划 |
| 贪心 | 多臂老虎机（ε-贪心） | 在线推荐、动态分配 |
| 二分答案 | 阈值搜索 | 模型调参、异常检测 |
| 滑动窗口 | 时序特征工程 | 量化交易、传感器数据处理 |

> 理解这些交叉点最大的价值在于：当你拿到一个「看起来像 ML 问题」的任务时，能意识到它的子问题可能可以用「纯算法」优雅地解决——而不是一上来就叠一个神经网络。

---

> [完，下一个: appendix-d-reading-list.md]
