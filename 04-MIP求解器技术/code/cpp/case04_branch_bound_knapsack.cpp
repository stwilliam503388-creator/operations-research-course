#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>

struct Item {
    int value;
    int weight;
    double density;
};

struct Node {
    int level;
    int value;
    int weight;
    double bound;
};

double fractional_bound(const Node& node, const std::vector<Item>& items, int capacity) {
    if (node.weight >= capacity) return 0.0;
    double bound = node.value;
    int total_weight = node.weight;

    for (int i = node.level; i < static_cast<int>(items.size()); ++i) {
        if (total_weight + items[i].weight <= capacity) {
            total_weight += items[i].weight;
            bound += items[i].value;
        } else {
            const int remain = capacity - total_weight;
            bound += remain * items[i].density;
            break;
        }
    }
    return bound;
}

int main() {
    std::vector<Item> items = {
        {20, 2, 0.0}, {30, 5, 0.0}, {35, 7, 0.0}, {12, 3, 0.0}, {3, 1, 0.0}
    };
    const int capacity = 10;
    for (auto& item : items) item.density = static_cast<double>(item.value) / item.weight;
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.density > b.density;
    });

    auto by_bound = [](const Node& a, const Node& b) { return a.bound < b.bound; };
    std::priority_queue<Node, std::vector<Node>, decltype(by_bound)> pq(by_bound);

    Node root{0, 0, 0, 0.0};
    root.bound = fractional_bound(root, items, capacity);
    pq.push(root);

    int best = 0;
    int expanded = 0;

    while (!pq.empty()) {
        Node cur = pq.top();
        pq.pop();
        if (cur.bound <= best || cur.level == static_cast<int>(items.size())) continue;
        ++expanded;

        const Item& item = items[cur.level];
        Node take{cur.level + 1, cur.value + item.value, cur.weight + item.weight, 0.0};
        if (take.weight <= capacity) {
            best = std::max(best, take.value);
            take.bound = fractional_bound(take, items, capacity);
            if (take.bound > best) pq.push(take);
        }

        Node skip{cur.level + 1, cur.value, cur.weight, 0.0};
        skip.bound = fractional_bound(skip, items, capacity);
        if (skip.bound > best) pq.push(skip);
    }

    std::cout << "capacity = " << capacity << "\n";
    std::cout << "best integer value = " << best << "\n";
    std::cout << "expanded nodes = " << expanded << "\n";
    return 0;
}
