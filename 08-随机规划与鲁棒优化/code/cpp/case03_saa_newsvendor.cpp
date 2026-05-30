#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// 教学注释：通过场景、概率权重和坏情况边界刻画不确定需求或收益。
// 对比不同假设下的解，理解随机性和鲁棒性如何改变决策。

double average_cost(int order, const std::vector<int>& scenarios, double unit_cost,
                    double shortage_cost, double holding_cost) {
    double total = 0.0;
    for (int demand : scenarios) {
        // 每个场景都评估同一个订货量：采购成本 + 缺货惩罚 + 库存持有成本。
        total += unit_cost * order;
        total += shortage_cost * std::max(0, demand - order);
        total += holding_cost * std::max(0, order - demand);
    }
    return total / scenarios.size();
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<double> demand_dist(100.0, 25.0);

    // SAA 用有限个随机场景替代真实需求分布，场景越多估计越稳定。
    std::vector<int> scenarios;
    for (int i = 0; i < 5000; ++i) {
        scenarios.push_back(std::max(0, static_cast<int>(demand_dist(rng) + 0.5)));
    }

    int best_order = 0;
    double best_cost = 1e100;
    // 小范围枚举订货量，选择样本平均成本最低的方案。
    for (int order = 40; order <= 180; ++order) {
        const double cost = average_cost(order, scenarios, 4.0, 9.0, 1.0);
        if (cost < best_cost) {
            best_cost = cost;
            best_order = order;
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "scenario count = " << scenarios.size() << "\n";
    std::cout << "SAA best order = " << best_order << "\n";
    std::cout << "estimated average cost = " << best_cost << "\n";
    return 0;
}
