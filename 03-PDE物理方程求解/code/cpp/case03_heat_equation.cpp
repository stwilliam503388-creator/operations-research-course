#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const int nx = 41;
    const int steps = 200;
    const double alpha = 1.0;
    const double length = 1.0;
    const double dx = length / (nx - 1);
    const double dt = 0.4 * dx * dx / alpha;
    const double r = alpha * dt / (dx * dx);

    std::vector<double> u(nx, 0.0), next(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        const double x = i * dx;
        u[i] = (x > 0.4 && x < 0.6) ? 1.0 : 0.0;
    }

    for (int step = 0; step < steps; ++step) {
        next[0] = 0.0;
        next[nx - 1] = 0.0;
        for (int i = 1; i < nx - 1; ++i) {
            next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        }
        u.swap(next);
    }

    const auto max_it = std::max_element(u.begin(), u.end());
    double energy = 0.0;
    for (double value : u) energy += value * dx;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CFL r = " << r << "\n";
    std::cout << "max temperature = " << *max_it << "\n";
    std::cout << "integral temperature = " << energy << "\n";
    std::cout << "center temperature = " << u[nx / 2] << "\n";
    return 0;
}
