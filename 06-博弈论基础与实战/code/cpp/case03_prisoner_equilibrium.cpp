#include <iostream>
#include <string>
#include <vector>

// 教学注释：从参与者、策略和收益矩阵出发理解交互决策结构。
// 计算结果用于验证均衡、分配规则或机制设计是否符合预期。

struct Payoff {
    int row;
    int col;
};

int main() {
    const std::vector<std::string> actions = {"Cooperate", "Defect"};
    // payoff[r][c] 存放行玩家选择 r、列玩家选择 c 时双方的收益。
    const std::vector<std::vector<Payoff>> payoff = {
        {{3, 3}, {0, 5}},
        {{5, 0}, {1, 1}}
    };

    std::cout << "Pure strategy Nash equilibria:\n";
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            bool row_best_response = true;
            bool col_best_response = true;

            // 纳什均衡要求：固定对方策略后，任何一方单独改策略都不能更好。
            for (int alt = 0; alt < 2; ++alt) {
                if (payoff[alt][c].row > payoff[r][c].row) row_best_response = false;
                if (payoff[r][alt].col > payoff[r][c].col) col_best_response = false;
            }

            if (row_best_response && col_best_response) {
                std::cout << "  row=" << actions[r] << ", col=" << actions[c]
                          << " payoff=(" << payoff[r][c].row << ", "
                          << payoff[r][c].col << ")\n";
            }
        }
    }
    return 0;
}
