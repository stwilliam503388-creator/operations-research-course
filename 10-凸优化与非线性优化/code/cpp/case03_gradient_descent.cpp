#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// 教学注释：把梯度、凸性、约束和步长选择对应到优化迭代过程。
// 收敛指标和残差帮助判断算法是否稳定接近最优解。

double loss(const std::vector<double>& x, const std::vector<double>& y, double w, double b) {
    double total = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double err = w * x[i] + b - y[i];
        // 最小二乘把“拟合误差”平方后平均，凸二次目标只有一个全局最优解。
        total += err * err;
    }
    return total / static_cast<double>(x.size());
}

int main() {
    std::vector<double> x;
    std::vector<double> y;
    // 构造一组近似直线 y = 3x + 5 的教学数据，并加入轻微扰动。
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
        // 对 w 和 b 分别累积均方误差的梯度。
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double err = w * x[i] + b - y[i];
            grad_w += 2.0 * err * x[i];
            grad_b += 2.0 * err;
        }
        grad_w /= static_cast<double>(x.size());
        grad_b /= static_cast<double>(x.size());
        // 沿负梯度方向更新参数；步长过大会震荡，过小会收敛慢。
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
