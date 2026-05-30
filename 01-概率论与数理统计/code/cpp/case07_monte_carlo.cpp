#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// 教学注释：围绕随机分布、样本统计量和蒙特卡洛估计观察不确定性。
// 运行时重点比较理论量与模拟结果如何支撑决策判断。

int main() {
    const int samples = 200000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // 在单位正方形里随机撒点，统计落入四分之一圆的比例。
    // 这个比例约等于 pi / 4，因此可以反推 pi 的估计值。
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
    // 用二项比例的标准误差近似蒙特卡洛估计误差，帮助读者理解“样本越多越稳定”。
    const double se = 4.0 * std::sqrt(p_hat * (1.0 - p_hat) / samples);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "samples = " << samples << "\n";
    std::cout << "pi estimate = " << pi_hat << "\n";
    std::cout << "approx 95% CI = [" << pi_hat - 1.96 * se << ", "
              << pi_hat + 1.96 * se << "]\n";
    std::cout << "absolute error = " << std::abs(pi_hat - std::acos(-1.0)) << "\n";
    return 0;
}
