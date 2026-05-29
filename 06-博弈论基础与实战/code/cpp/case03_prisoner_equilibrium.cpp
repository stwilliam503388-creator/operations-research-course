#include <iostream>
#include <string>
#include <vector>

struct Payoff {
    int row;
    int col;
};

int main() {
    const std::vector<std::string> actions = {"Cooperate", "Defect"};
    const std::vector<std::vector<Payoff>> payoff = {
        {{3, 3}, {0, 5}},
        {{5, 0}, {1, 1}}
    };

    std::cout << "Pure strategy Nash equilibria:\n";
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            bool row_best_response = true;
            bool col_best_response = true;

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
