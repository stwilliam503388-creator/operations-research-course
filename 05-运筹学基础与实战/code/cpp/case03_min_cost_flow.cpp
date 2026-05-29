#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

struct Edge {
    int to;
    int rev;
    int capacity;
    int cost;
};

class MinCostFlow {
public:
    explicit MinCostFlow(int n) : graph_(n) {}

    void add_edge(int from, int to, int capacity, int cost) {
        Edge forward{to, static_cast<int>(graph_[to].size()), capacity, cost};
        Edge backward{from, static_cast<int>(graph_[from].size()), 0, -cost};
        graph_[from].push_back(forward);
        graph_[to].push_back(backward);
    }

    std::pair<int, int> solve(int source, int sink, int need) {
        int flow = 0;
        int cost = 0;
        const int n = static_cast<int>(graph_.size());
        const int inf = std::numeric_limits<int>::max() / 4;

        while (flow < need) {
            std::vector<int> dist(n, inf), prev_v(n, -1), prev_e(n, -1);
            std::vector<bool> in_queue(n, false);
            std::queue<int> q;
            dist[source] = 0;
            q.push(source);
            in_queue[source] = true;

            while (!q.empty()) {
                int v = q.front();
                q.pop();
                in_queue[v] = false;
                for (int i = 0; i < static_cast<int>(graph_[v].size()); ++i) {
                    const Edge& e = graph_[v][i];
                    if (e.capacity > 0 && dist[e.to] > dist[v] + e.cost) {
                        dist[e.to] = dist[v] + e.cost;
                        prev_v[e.to] = v;
                        prev_e[e.to] = i;
                        if (!in_queue[e.to]) {
                            q.push(e.to);
                            in_queue[e.to] = true;
                        }
                    }
                }
            }

            if (dist[sink] == inf) break;
            int add = need - flow;
            for (int v = sink; v != source; v = prev_v[v]) {
                add = std::min(add, graph_[prev_v[v]][prev_e[v]].capacity);
            }
            for (int v = sink; v != source; v = prev_v[v]) {
                Edge& e = graph_[prev_v[v]][prev_e[v]];
                e.capacity -= add;
                graph_[v][e.rev].capacity += add;
            }
            flow += add;
            cost += add * dist[sink];
        }
        return {flow, cost};
    }

private:
    std::vector<std::vector<Edge>> graph_;
};

int main() {
    const int source = 0;
    const int factory_a = 1;
    const int factory_b = 2;
    const int warehouse_x = 3;
    const int warehouse_y = 4;
    const int sink = 5;

    MinCostFlow mcf(6);
    mcf.add_edge(source, factory_a, 70, 0);
    mcf.add_edge(source, factory_b, 50, 0);
    mcf.add_edge(factory_a, warehouse_x, 60, 4);
    mcf.add_edge(factory_a, warehouse_y, 60, 7);
    mcf.add_edge(factory_b, warehouse_x, 50, 6);
    mcf.add_edge(factory_b, warehouse_y, 50, 3);
    mcf.add_edge(warehouse_x, sink, 60, 0);
    mcf.add_edge(warehouse_y, sink, 50, 0);

    const auto [flow, cost] = mcf.solve(source, sink, 110);
    std::cout << "delivered flow = " << flow << "\n";
    std::cout << "minimum transport cost = " << cost << "\n";
    return 0;
}
