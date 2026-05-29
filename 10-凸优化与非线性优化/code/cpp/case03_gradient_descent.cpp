#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

double loss(const std::vector<double>& x, const std::vector<double>& y, double w, double b) {
    double total = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double err = w * x[i] + b - y[i];
        total += err * err;
    }
    return total / static_cast<double>(x.size());
}

int main() {
    std::vector<double> x;
    std::vector<double> y;
    for (int i = 0; i <= 20; ++i) {
        const double xi = static_cast<double>(i) / 2.0;
        x.push_back(xi);
        y.push_back(3.0 * xi + 5.0 + 0.2 * std::sin(xi));
    }

    double w = 0.0;
    double b = 0.0;
    const double lr = 0.01;
    const int steps = 5000;
    const double initial_loss = loss(x, y, w, b);

    for (int step = 0; step < steps; ++step) {
        double grad_w = 0.0;
        double grad_b = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double err = w * x[i] + b - y[i];
            grad_w += 2.0 * err * x[i];
            grad_b += 2.0 * err;
        }
        grad_w /= static_cast<double>(x.size());
        grad_b /= static_cast<double>(x.size());
        w -= lr * grad_w;
        b -= lr * grad_b;
    }

    const double final_loss = loss(x, y, w, b);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "initial_loss=" << initial_loss << "\n";
    std::cout << "final_loss=" << final_loss << "\n";
    std::cout << "loss_decreased=" << std::boolalpha << (final_loss < initial_loss) << "\n";
    std::cout << "w=" << w << ", b=" << b << "\n";
    std::cout << "w_error=" << std::abs(w - 3.0) << ", b_error=" << std::abs(b - 5.0) << "\n";

    return (final_loss < initial_loss && std::abs(w - 3.0) < 0.1 && std::abs(b - 5.0) < 0.4) ? 0 : 1;
}
