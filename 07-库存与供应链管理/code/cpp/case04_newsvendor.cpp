#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

double profit(int order, int demand, double price, double cost, double salvage) {
    const int sold = std::min(order, demand);
    const int leftover = std::max(0, order - demand);
    return price * sold + salvage * leftover - cost * order;
}

int main() {
    const double price = 10.0;
    const double cost = 6.0;
    const double salvage = 2.0;
    const double critical_ratio = (price - cost) / (price - salvage);

    std::mt19937 rng(42);
    std::normal_distribution<double> demand_dist(100.0, 20.0);
    std::vector<int> demand_samples;
    for (int i = 0; i < 20000; ++i) {
        demand_samples.push_back(std::max(0, static_cast<int>(demand_dist(rng) + 0.5)));
    }
    std::sort(demand_samples.begin(), demand_samples.end());

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
