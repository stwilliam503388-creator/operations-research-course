/*
 * test_cpp_utils.cpp — C++ 单元测试
 *
 * 本文件为各课程 C++ 示例中可独立验证的工具函数编写单元测试。
 * 不依赖第三方测试框架，仅使用 <cassert> 和标准库。
 * 编译：g++ -std=c++17 -O2 test_cpp_utils.cpp -o test_cpp_utils && ./test_cpp_utils
 *
 * 被测函数与课程对应关系
 * ====================
 * - merge_sort        → 课程 02-低复杂度算法设计 (case03_merge_sort)
 * - fractional_bound  → 课程 04-MIP求解器技术 (case04_branch_bound_knapsack)
 * - dominates         → 课程 09-多目标优化 (case07_pareto_sort)
 * - loss (MSE)        → 课程 10-凸优化与非线性优化 (case03_gradient_descent)
 * - profit            → 课程 07-库存与供应链管理 (case04_newsvendor)
 * - average_cost      → 课程 08-随机规划与鲁棒优化 (case03_saa_newsvendor)
 * - MinCostFlow       → 课程 05-运筹学基础与实战 (case03_min_cost_flow)
 * - Nash equilibrium  → 课程 06-博弈论基础与实战 (case03_prisoner_equilibrium)
 * - heat_step (1D)    → 课程 03-PDE物理方程求解 (case03_heat_equation)
 * - monte_carlo_pi    → 课程 01-概率论与数理统计 (case07_monte_carlo)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <vector>

static int tests_passed = 0;

// ──────────────────────────────────────────────────────────
// 课程 02-低复杂度算法设计: merge_sort
// 来源: 02-低复杂度算法设计/code/cpp/case03_merge_sort.cpp
// ──────────────────────────────────────────────────────────

void merge_sort(std::vector<int>& a, std::vector<int>& buffer, int left, int right) {
    if (right - left <= 1) return;
    const int mid = left + (right - left) / 2;
    merge_sort(a, buffer, left, mid);
    merge_sort(a, buffer, mid, right);
    int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        buffer[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    }
    while (i < mid) buffer[k++] = a[i++];
    while (j < right) buffer[k++] = a[j++];
    for (int t = left; t < right; ++t) a[t] = buffer[t];
}

void test_merge_sort_basic() {
    std::vector<int> v = {5, 3, 8, 1, 2, 7, 4, 6};
    std::vector<int> buf(v.size());
    merge_sort(v, buf, 0, static_cast<int>(v.size()));
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(v == expected);
    ++tests_passed;
}

void test_merge_sort_empty_and_single() {
    std::vector<int> empty;
    std::vector<int> buf;
    merge_sort(empty, buf, 0, 0);
    assert(empty.empty());

    std::vector<int> single = {42};
    std::vector<int> buf1(1);
    merge_sort(single, buf1, 0, 1);
    assert(single[0] == 42);
    ++tests_passed;
}

void test_merge_sort_already_sorted() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> buf(v.size());
    merge_sort(v, buf, 0, static_cast<int>(v.size()));
    std::vector<int> expected = {1, 2, 3, 4, 5};
    assert(v == expected);
    ++tests_passed;
}

void test_merge_sort_reverse_sorted() {
    std::vector<int> v = {5, 4, 3, 2, 1};
    std::vector<int> buf(v.size());
    merge_sort(v, buf, 0, static_cast<int>(v.size()));
    std::vector<int> expected = {1, 2, 3, 4, 5};
    assert(v == expected);
    ++tests_passed;
}

void test_merge_sort_duplicates() {
    std::vector<int> v = {3, 1, 3, 1, 2, 2};
    std::vector<int> buf(v.size());
    merge_sort(v, buf, 0, static_cast<int>(v.size()));
    std::vector<int> expected = {1, 1, 2, 2, 3, 3};
    assert(v == expected);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 04-MIP求解器技术: fractional_bound (分支定界上界)
// 来源: 04-MIP求解器技术/code/cpp/case04_branch_bound_knapsack.cpp
// ──────────────────────────────────────────────────────────

struct Item {
    int value;
    int weight;
    double density;
};

struct Node {
    int level;
    int value;
    int weight;
    double bound;
};

double fractional_bound(const Node& node, const std::vector<Item>& items, int capacity) {
    if (node.weight >= capacity) return 0.0;
    double bound = node.value;
    int total_weight = node.weight;
    for (int i = node.level; i < static_cast<int>(items.size()); ++i) {
        if (total_weight + items[i].weight <= capacity) {
            total_weight += items[i].weight;
            bound += items[i].value;
        } else {
            const int remain = capacity - total_weight;
            bound += remain * items[i].density;
            break;
        }
    }
    return bound;
}

void test_fractional_bound_full_fit() {
    // 所有物品都能装下，上界等于所有物品价值之和
    std::vector<Item> items = {{10, 2, 5.0}, {20, 3, 6.67}};
    Node root{0, 0, 0, 0.0};
    double bound = fractional_bound(root, items, 100);
    assert(std::abs(bound - 30.0) < 1e-9);
    ++tests_passed;
}

void test_fractional_bound_partial() {
    // 按密度降序排列: (20,3,6.67), (10,2,5.0)
    std::vector<Item> items = {{20, 3, 20.0 / 3}, {10, 2, 5.0}};
    Node root{0, 0, 0, 0.0};
    // capacity=4: 拿第一个 (w=3,v=20)，剩余 1 单位分数取第二个 (1*5.0=5)
    double bound = fractional_bound(root, items, 4);
    assert(std::abs(bound - 25.0) < 1e-9);
    ++tests_passed;
}

void test_fractional_bound_overweight_returns_zero() {
    std::vector<Item> items = {{10, 2, 5.0}};
    Node over{0, 0, 10, 0.0};  // 已超重
    assert(fractional_bound(over, items, 5) == 0.0);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 09-多目标优化: dominates (Pareto 支配)
// 来源: 09-多目标优化/code/cpp/case07_pareto_sort.cpp
// ──────────────────────────────────────────────────────────

struct Point {
    double cost;
    double emission;
};

bool dominates(const Point& a, const Point& b) {
    const bool no_worse = a.cost <= b.cost && a.emission <= b.emission;
    const bool strictly_better = a.cost < b.cost || a.emission < b.emission;
    return no_worse && strictly_better;
}

void test_dominates_strict() {
    // a 在两个目标上都更优
    assert(dominates({1.0, 1.0}, {2.0, 2.0}));
    ++tests_passed;
}

void test_dominates_one_better_one_equal() {
    // a 在一个目标上更优，另一个相等 → 支配
    assert(dominates({1.0, 2.0}, {2.0, 2.0}));
    assert(dominates({2.0, 1.0}, {2.0, 2.0}));
    ++tests_passed;
}

void test_dominates_not_dominated_equal() {
    // 完全相等 → 不构成支配
    assert(!dominates({1.0, 1.0}, {1.0, 1.0}));
    ++tests_passed;
}

void test_dominates_incomparable() {
    // 各有所长 → 互不支配
    assert(!dominates({1.0, 3.0}, {2.0, 1.0}));
    assert(!dominates({2.0, 1.0}, {1.0, 3.0}));
    ++tests_passed;
}

void test_pareto_front_extraction() {
    // 与课程示例相同的 10 个方案，验证 Pareto 前沿提取正确性
    std::vector<Point> pop = {
        {100, 80}, {95, 90}, {110, 60}, {90, 110}, {105, 70},
        {120, 50}, {85, 130}, {115, 65}, {98, 88}, {130, 45}
    };
    std::vector<int> front;
    for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
        bool dominated = false;
        for (int j = 0; j < static_cast<int>(pop.size()); ++j) {
            if (i != j && dominates(pop[j], pop[i])) { dominated = true; break; }
        }
        if (!dominated) front.push_back(i);
    }
    // Pareto 前沿不应为空，且不应包含所有点
    assert(!front.empty());
    assert(front.size() < pop.size());
    // 验证前沿上的点互不支配
    for (int i = 0; i < static_cast<int>(front.size()); ++i) {
        for (int j = 0; j < static_cast<int>(front.size()); ++j) {
            if (i != j) assert(!dominates(pop[front[i]], pop[front[j]]));
        }
    }
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 10-凸优化与非线性优化: loss (均方误差)
// 来源: 10-凸优化与非线性优化/code/cpp/case03_gradient_descent.cpp
// ──────────────────────────────────────────────────────────

double loss(const std::vector<double>& x, const std::vector<double>& y, double w, double b) {
    double total = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double err = w * x[i] + b - y[i];
        total += err * err;
    }
    return total / static_cast<double>(x.size());
}

void test_loss_perfect_fit() {
    // y = 2x + 1，用 w=2, b=1 应得 loss=0
    std::vector<double> x = {0, 1, 2, 3};
    std::vector<double> y = {1, 3, 5, 7};
    assert(std::abs(loss(x, y, 2.0, 1.0)) < 1e-12);
    ++tests_passed;
}

void test_loss_known_value() {
    // 单点：x=0, y=0, w=0, b=1 → err=1, loss=1
    std::vector<double> x = {0.0};
    std::vector<double> y = {0.0};
    assert(std::abs(loss(x, y, 0.0, 1.0) - 1.0) < 1e-12);
    ++tests_passed;
}

void test_loss_nonnegative() {
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {5, -1, 7};
    assert(loss(x, y, 3.14, -2.7) >= 0.0);
    ++tests_passed;
}

void test_gradient_descent_converges() {
    // 简单梯度下降应能拟合 y = 3x + 5
    std::vector<double> x, y;
    for (int i = 0; i <= 20; ++i) {
        double xi = static_cast<double>(i) / 2.0;
        x.push_back(xi);
        y.push_back(3.0 * xi + 5.0);
    }
    double w = 0.0, b = 0.0;
    const double lr = 0.01;
    for (int step = 0; step < 5000; ++step) {
        double gw = 0.0, gb = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            double err = w * x[i] + b - y[i];
            gw += 2.0 * err * x[i];
            gb += 2.0 * err;
        }
        gw /= static_cast<double>(x.size());
        gb /= static_cast<double>(x.size());
        w -= lr * gw;
        b -= lr * gb;
    }
    assert(std::abs(w - 3.0) < 0.05);
    assert(std::abs(b - 5.0) < 0.2);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 07-库存与供应链管理: profit (报童利润)
// 来源: 07-库存与供应链管理/code/cpp/case04_newsvendor.cpp
// ──────────────────────────────────────────────────────────

double profit(int order, int demand, double price, double cost, double salvage) {
    const int sold = std::min(order, demand);
    const int leftover = std::max(0, order - demand);
    return price * sold + salvage * leftover - cost * order;
}

void test_profit_exact_match() {
    // 订货量 == 需求量 → 无剩余无缺货
    // profit = 10*100 + 2*0 - 6*100 = 400
    assert(std::abs(profit(100, 100, 10.0, 6.0, 2.0) - 400.0) < 1e-9);
    ++tests_passed;
}

void test_profit_excess_order() {
    // 多订：order=120, demand=100 → sold=100, leftover=20
    // profit = 10*100 + 2*20 - 6*120 = 1000+40-720 = 320
    assert(std::abs(profit(120, 100, 10.0, 6.0, 2.0) - 320.0) < 1e-9);
    ++tests_passed;
}

void test_profit_shortage() {
    // 少订：order=80, demand=100 → sold=80, leftover=0
    // profit = 10*80 + 0 - 6*80 = 800-480 = 320
    assert(std::abs(profit(80, 100, 10.0, 6.0, 2.0) - 320.0) < 1e-9);
    ++tests_passed;
}

void test_profit_zero_order() {
    assert(std::abs(profit(0, 100, 10.0, 6.0, 2.0) - 0.0) < 1e-9);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 08-随机规划与鲁棒优化: average_cost (SAA 平均成本)
// 来源: 08-随机规划与鲁棒优化/code/cpp/case03_saa_newsvendor.cpp
// ──────────────────────────────────────────────────────────

double average_cost(int order, const std::vector<int>& scenarios, double unit_cost,
                    double shortage_cost, double holding_cost) {
    double total = 0.0;
    for (int demand : scenarios) {
        total += unit_cost * order;
        total += shortage_cost * std::max(0, demand - order);
        total += holding_cost * std::max(0, order - demand);
    }
    return total / static_cast<double>(scenarios.size());
}

void test_average_cost_exact_match() {
    // 每个场景需求都等于订货量 → 无缺货无积压
    std::vector<int> scenarios = {100, 100, 100};
    // cost = 4*100 = 400
    assert(std::abs(average_cost(100, scenarios, 4.0, 9.0, 1.0) - 400.0) < 1e-9);
    ++tests_passed;
}

void test_average_cost_with_shortage() {
    // 需求高于订货：order=80, demand=100
    // cost = 4*80 + 9*(100-80) + 0 = 320 + 180 = 500
    std::vector<int> scenarios = {100};
    assert(std::abs(average_cost(80, scenarios, 4.0, 9.0, 1.0) - 500.0) < 1e-9);
    ++tests_passed;
}

void test_average_cost_with_holding() {
    // 需求低于订货：order=120, demand=80
    // cost = 4*120 + 0 + 1*(120-80) = 480 + 40 = 520
    std::vector<int> scenarios = {80};
    assert(std::abs(average_cost(120, scenarios, 4.0, 9.0, 1.0) - 520.0) < 1e-9);
    ++tests_passed;
}

void test_average_cost_multiple_scenarios() {
    // 两个场景的平均
    std::vector<int> scenarios = {100, 80};
    double c1 = 4.0 * 90 + 9.0 * 10 + 0;            // demand=100: 360+90=450
    double c2 = 4.0 * 90 + 0 + 1.0 * 10;             // demand=80: 360+10=370
    double expected = (c1 + c2) / 2.0;                // 410
    assert(std::abs(average_cost(90, scenarios, 4.0, 9.0, 1.0) - expected) < 1e-9);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 05-运筹学基础与实战: MinCostFlow (最小费用流)
// 来源: 05-运筹学基础与实战/code/cpp/case03_min_cost_flow.cpp
// ──────────────────────────────────────────────────────────

struct Edge {
    int to;
    int rev;
    int capacity;
    int cost;
};

class MinCostFlow {
public:
    explicit MinCostFlow(int n) : graph_(n) {}

    void add_edge(int from, int to, int capacity, int cost) {
        Edge forward{to, static_cast<int>(graph_[to].size()), capacity, cost};
        Edge backward{from, static_cast<int>(graph_[from].size()), 0, -cost};
        graph_[from].push_back(forward);
        graph_[to].push_back(backward);
    }

    std::pair<int, long long> solve(int source, int sink, int need) {
        int flow = 0;
        long long cost = 0;
        const int n = static_cast<int>(graph_.size());
        const int inf = std::numeric_limits<int>::max() / 4;

        while (flow < need) {
            std::vector<int> dist(n, inf), prev_v(n, -1), prev_e(n, -1);
            std::vector<bool> in_queue(n, false);
            std::queue<int> q;
            dist[source] = 0;
            q.push(source);
            in_queue[source] = true;

            while (!q.empty()) {
                int v = q.front(); q.pop(); in_queue[v] = false;
                for (int i = 0; i < static_cast<int>(graph_[v].size()); ++i) {
                    const Edge& e = graph_[v][i];
                    if (e.capacity > 0 && dist[e.to] > dist[v] + e.cost) {
                        dist[e.to] = dist[v] + e.cost;
                        prev_v[e.to] = v;
                        prev_e[e.to] = i;
                        if (!in_queue[e.to]) { q.push(e.to); in_queue[e.to] = true; }
                    }
                }
            }
            if (dist[sink] == inf) break;
            int add = need - flow;
            for (int v = sink; v != source; v = prev_v[v]) {
                add = std::min(add, graph_[prev_v[v]][prev_e[v]].capacity);
            }
            for (int v = sink; v != source; v = prev_v[v]) {
                Edge& e = graph_[prev_v[v]][prev_e[v]];
                e.capacity -= add;
                graph_[v][e.rev].capacity += add;
            }
            flow += add;
            cost += static_cast<long long>(add) * dist[sink];
        }
        return {flow, cost};
    }

private:
    std::vector<std::vector<Edge>> graph_;
};

void test_min_cost_flow_simple() {
    // 简单 s→a→t 网络
    MinCostFlow mcf(3);
    mcf.add_edge(0, 1, 10, 2);  // s→a, cap=10, cost=2
    mcf.add_edge(1, 2, 10, 3);  // a→t, cap=10, cost=3
    auto [flow, cost] = mcf.solve(0, 2, 5);
    assert(flow == 5);
    assert(cost == 25);  // 5 * (2+3) = 25
    ++tests_passed;
}

void test_min_cost_flow_two_paths() {
    // s→a→t (cap=5, cost=1+1=2) 和 s→b→t (cap=5, cost=2+2=4)
    MinCostFlow mcf(4);
    mcf.add_edge(0, 1, 5, 1);   // s→a
    mcf.add_edge(1, 3, 5, 1);   // a→t
    mcf.add_edge(0, 2, 5, 2);   // s→b
    mcf.add_edge(2, 3, 5, 2);   // b→t
    auto [flow, cost] = mcf.solve(0, 3, 8);
    assert(flow == 8);
    // 先走便宜路 5*2=10，再走贵路 3*4=12，total=22
    assert(cost == 22);
    ++tests_passed;
}

void test_min_cost_flow_course_example() {
    // 与课程示例完全相同的工厂-仓库网络
    MinCostFlow mcf(6);
    mcf.add_edge(0, 1, 70, 0);   // source→factory_a
    mcf.add_edge(0, 2, 50, 0);   // source→factory_b
    mcf.add_edge(1, 3, 60, 4);   // factory_a→warehouse_x
    mcf.add_edge(1, 4, 60, 7);   // factory_a→warehouse_y
    mcf.add_edge(2, 3, 50, 6);   // factory_b→warehouse_x
    mcf.add_edge(2, 4, 50, 3);   // factory_b→warehouse_y
    mcf.add_edge(3, 5, 60, 0);   // warehouse_x→sink
    mcf.add_edge(4, 5, 50, 0);   // warehouse_y→sink
    auto [flow, cost] = mcf.solve(0, 5, 110);
    assert(flow == 110);
    // 最优运输方案总费用
    assert(cost > 0);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 06-博弈论基础与实战: Nash 均衡判定
// 来源: 06-博弈论基础与实战/code/cpp/case03_prisoner_equilibrium.cpp
// ──────────────────────────────────────────────────────────

struct Payoff {
    int row;
    int col;
};

// 提取课程中纯策略纳什均衡搜索逻辑为可测试函数
std::vector<std::pair<int, int>> find_pure_nash(
    const std::vector<std::vector<Payoff>>& payoff, int n_strategies) {
    std::vector<std::pair<int, int>> result;
    for (int r = 0; r < n_strategies; ++r) {
        for (int c = 0; c < n_strategies; ++c) {
            bool row_best = true, col_best = true;
            for (int alt = 0; alt < n_strategies; ++alt) {
                if (payoff[alt][c].row > payoff[r][c].row) row_best = false;
                if (payoff[r][alt].col > payoff[r][c].col) col_best = false;
            }
            if (row_best && col_best) result.push_back({r, c});
        }
    }
    return result;
}

void test_prisoner_dilemma_nash() {
    // 经典囚徒困境：唯一纳什均衡是 (Defect, Defect) = (1, 1)
    std::vector<std::vector<Payoff>> payoff = {
        {{3, 3}, {0, 5}},
        {{5, 0}, {1, 1}}
    };
    auto eq = find_pure_nash(payoff, 2);
    assert(eq.size() == 1);
    assert(eq[0].first == 1 && eq[0].second == 1);
    ++tests_passed;
}

void test_coordination_game_nash() {
    // 协调博弈：两个纳什均衡 (0,0) 和 (1,1)
    std::vector<std::vector<Payoff>> payoff = {
        {{2, 2}, {0, 0}},
        {{0, 0}, {1, 1}}
    };
    auto eq = find_pure_nash(payoff, 2);
    assert(eq.size() == 2);
    ++tests_passed;
}

void test_matching_pennies_no_pure_nash() {
    // 猜硬币博弈：无纯策略纳什均衡
    std::vector<std::vector<Payoff>> payoff = {
        {{ 1, -1}, {-1,  1}},
        {{-1,  1}, { 1, -1}}
    };
    auto eq = find_pure_nash(payoff, 2);
    assert(eq.empty());
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 03-PDE物理方程求解: heat_step (一维热传导单步)
// 来源: 03-PDE物理方程求解/code/cpp/case03_heat_equation.cpp
// ──────────────────────────────────────────────────────────

// 提取为独立函数：一维热方程显式差分单步
void heat_step(std::vector<double>& u, std::vector<double>& next, double r) {
    const int nx = static_cast<int>(u.size());
    next[0] = 0.0;
    next[nx - 1] = 0.0;
    for (int i = 1; i < nx - 1; ++i) {
        next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
    }
    u.swap(next);
}

void test_heat_step_uniform() {
    // 均匀温度场（边界除外）→ 内部不变
    std::vector<double> u = {0.0, 1.0, 1.0, 1.0, 0.0};
    std::vector<double> next(5, 0.0);
    heat_step(u, next, 0.4);
    // 中心点不变: 1 + 0.4*(1 - 2 + 1) = 1
    assert(std::abs(u[2] - 1.0) < 1e-12);
    // 边界附近的点扩散：u[1] = 1 + 0.4*(0 - 2 + 1) = 0.6
    assert(std::abs(u[1] - 0.6) < 1e-12);
    ++tests_passed;
}

void test_heat_step_energy_conservation() {
    // 热量守恒（边界固定为 0 时能量单调不增）
    std::vector<double> u = {0.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<double> next(5, 0.0);
    double energy_before = 0.0;
    for (double v : u) energy_before += v;

    heat_step(u, next, 0.4);
    double energy_after = 0.0;
    for (double v : u) energy_after += v;

    // 由于边界吸收热量，总能量不增
    assert(energy_after <= energy_before + 1e-12);
    ++tests_passed;
}

void test_heat_step_stability() {
    // r=0.4 应保持数值稳定（无负值、无爆炸）
    const int nx = 21;
    std::vector<double> u(nx, 0.0);
    u[nx / 2] = 1.0;
    std::vector<double> next(nx, 0.0);
    for (int step = 0; step < 100; ++step) {
        heat_step(u, next, 0.4);
    }
    for (double v : u) {
        assert(v >= -1e-12 && v <= 1.0 + 1e-12);
    }
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// 课程 01-概率论与数理统计: Monte Carlo 估计 π
// 来源: 01-概率论与数理统计/code/cpp/case07_monte_carlo.cpp
// ──────────────────────────────────────────────────────────

double monte_carlo_pi(int samples, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int inside = 0;
    for (int i = 0; i < samples; ++i) {
        double x = unit(rng);
        double y = unit(rng);
        if (x * x + y * y <= 1.0) ++inside;
    }
    return 4.0 * static_cast<double>(inside) / samples;
}

void test_monte_carlo_pi_estimate() {
    double pi_hat = monte_carlo_pi(200000, 42);
    // 允许 0.02 的误差
    assert(std::abs(pi_hat - std::acos(-1.0)) < 0.02);
    ++tests_passed;
}

void test_monte_carlo_pi_deterministic() {
    // 相同种子应产生相同结果
    double a = monte_carlo_pi(10000, 123);
    double b = monte_carlo_pi(10000, 123);
    assert(a == b);
    ++tests_passed;
}

// ──────────────────────────────────────────────────────────
// main: 运行所有测试
// ──────────────────────────────────────────────────────────

int main() {
    std::cout << "Running C++ unit tests...\n";

    // 课程 02: 归并排序
    test_merge_sort_basic();
    test_merge_sort_empty_and_single();
    test_merge_sort_already_sorted();
    test_merge_sort_reverse_sorted();
    test_merge_sort_duplicates();

    // 课程 04: 分支定界上界
    test_fractional_bound_full_fit();
    test_fractional_bound_partial();
    test_fractional_bound_overweight_returns_zero();

    // 课程 09: Pareto 支配
    test_dominates_strict();
    test_dominates_one_better_one_equal();
    test_dominates_not_dominated_equal();
    test_dominates_incomparable();
    test_pareto_front_extraction();

    // 课程 10: 梯度下降 / MSE
    test_loss_perfect_fit();
    test_loss_known_value();
    test_loss_nonnegative();
    test_gradient_descent_converges();

    // 课程 07: 报童利润
    test_profit_exact_match();
    test_profit_excess_order();
    test_profit_shortage();
    test_profit_zero_order();

    // 课程 08: SAA 平均成本
    test_average_cost_exact_match();
    test_average_cost_with_shortage();
    test_average_cost_with_holding();
    test_average_cost_multiple_scenarios();

    // 课程 05: 最小费用流
    test_min_cost_flow_simple();
    test_min_cost_flow_two_paths();
    test_min_cost_flow_course_example();

    // 课程 06: 纳什均衡
    test_prisoner_dilemma_nash();
    test_coordination_game_nash();
    test_matching_pennies_no_pure_nash();

    // 课程 03: 热传导
    test_heat_step_uniform();
    test_heat_step_energy_conservation();
    test_heat_step_stability();

    // 课程 01: 蒙特卡洛
    test_monte_carlo_pi_estimate();
    test_monte_carlo_pi_deterministic();

    std::cout << "All " << tests_passed << " C++ tests passed.\n";
    return 0;
}
