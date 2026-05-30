#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// 教学注释：围绕订货、库存、契约和网络配置观察供应链决策变量。
// 重点比较成本、缺货风险与服务水平之间的权衡。

double profit(int order, int demand, double price, double cost, double salvage) {
    const int sold = std::min(order, demand);
    const int leftover = std::max(0, order - demand);
    // 利润 = 销售收入 + 残值回收 - 订货成本；缺货损失在未卖出的需求中隐含体现。
    return price * sold + salvage * leftover - cost * order;
}

int main() {
    const double price = 10.0;
    const double cost = 6.0;
    const double salvage = 2.0;
    // 临界比率是报童模型的核心：边际缺货损失 / (边际缺货损失 + 边际积压损失)。
    const double critical_ratio = (price - cost) / (price - salvage);

    std::mt19937 rng(42);
    std::normal_distribution<double> demand_dist(100.0, 20.0);
    std::vector<int> demand_samples;
    for (int i = 0; i < 20000; ++i) {
        demand_samples.push_back(std::max(0, static_cast<int>(demand_dist(rng) + 0.5)));
    }
    std::sort(demand_samples.begin(), demand_samples.end());

    // 经验分位数近似最优订货量：需求低于该量的概率约等于临界比率。
    const int quantile_index = static_cast<int>(critical_ratio * (demand_samples.size() - 1));
    const int order = demand_samples[quantile_index];

    double total_profit = 0.0;
    for (int demand : demand_samples) {
        total_profit += profit(order, demand, price, cost, salvage);
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "critical ratio = " << critical_ratio << "\n";
    std::cout << "recommended order = " << order << "\n";
    std::cout << "average profit = " << total_profit / demand_samples.size() << "\n";
    return 0;
}
