#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// 教学注释：关注多个目标之间的冲突、Pareto 支配关系和权衡参数。
// 输出的解集用于筛选可解释、可落地的折中方案。

struct Point {
    double cost;
    double emission;
};

bool dominates(const Point& a, const Point& b) {
    // 两个目标都是越小越好；支配要求所有目标不差，且至少一个目标更好。
    const bool no_worse = a.cost <= b.cost && a.emission <= b.emission;
    const bool strictly_better = a.cost < b.cost || a.emission < b.emission;
    return no_worse && strictly_better;
}

int main() {
    // 每个点代表一个备选方案：(成本, 排放)。真实问题中它可能来自调度或供应链模型。
    std::vector<Point> population = {
        {100, 80}, {95, 90}, {110, 60}, {90, 110}, {105, 70},
        {120, 50}, {85, 130}, {115, 65}, {98, 88}, {130, 45}
    };

    std::vector<int> front;
    for (int i = 0; i < static_cast<int>(population.size()); ++i) {
        bool dominated = false;
        // 若存在另一个方案支配当前方案，当前方案就不在第一 Pareto 前沿上。
        for (int j = 0; j < static_cast<int>(population.size()); ++j) {
            if (i != j && dominates(population[j], population[i])) {
                dominated = true;
                break;
            }
        }
        if (!dominated) front.push_back(i);
    }

    std::sort(front.begin(), front.end(), [&](int a, int b) {
        return population[a].cost < population[b].cost;
    });

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "first Pareto front:\n";
    for (int idx : front) {
        std::cout << "  solution " << idx
                  << " cost=" << population[idx].cost
                  << " emission=" << population[idx].emission << "\n";
    }

    std::cout << "crowding-distance intuition: boundary solutions are preserved,\n";
    std::cout << "middle solutions are preferred when they cover sparse regions.\n";
    return 0;
}
