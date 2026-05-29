#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    const int samples = 200000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    int inside = 0;
    for (int i = 0; i < samples; ++i) {
        const double x = unit(rng);
        const double y = unit(rng);
        if (x * x + y * y <= 1.0) {
            ++inside;
        }
    }

    const double p_hat = static_cast<double>(inside) / samples;
    const double pi_hat = 4.0 * p_hat;
    const double se = 4.0 * std::sqrt(p_hat * (1.0 - p_hat) / samples);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "samples = " << samples << "\n";
    std::cout << "pi estimate = " << pi_hat << "\n";
    std::cout << "approx 95% CI = [" << pi_hat - 1.96 * se << ", "
              << pi_hat + 1.96 * se << "]\n";
    std::cout << "absolute error = " << std::abs(pi_hat - std::acos(-1.0)) << "\n";
    return 0;
}
