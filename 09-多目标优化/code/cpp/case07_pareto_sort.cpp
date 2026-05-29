#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

struct Point {
    double cost;
    double emission;
};

bool dominates(const Point& a, const Point& b) {
    const bool no_worse = a.cost <= b.cost && a.emission <= b.emission;
    const bool strictly_better = a.cost < b.cost || a.emission < b.emission;
    return no_worse && strictly_better;
}

int main() {
    std::vector<Point> population = {
        {100, 80}, {95, 90}, {110, 60}, {90, 110}, {105, 70},
        {120, 50}, {85, 130}, {115, 65}, {98, 88}, {130, 45}
    };

    std::vector<int> front;
    for (int i = 0; i < static_cast<int>(population.size()); ++i) {
        bool dominated = false;
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
