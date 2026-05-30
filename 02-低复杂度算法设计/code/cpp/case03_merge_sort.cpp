#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// 教学注释：关注复杂度、状态设计与数据结构选择如何影响可运行规模。
// 阅读时对照搜索、剪枝或启发式步骤，理解它们减少计算量的方式。

void merge_sort(std::vector<int>& a, std::vector<int>& buffer, int left, int right) {
    if (right - left <= 1) return;
    const int mid = left + (right - left) / 2;
    // 分治：先把区间拆成两个更小的已排序子问题。
    merge_sort(a, buffer, left, mid);
    merge_sort(a, buffer, mid, right);

    // 合并：两个有序子数组线性扫描，整体复杂度保持 O(n log n)。
    int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        buffer[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    }
    while (i < mid) buffer[k++] = a[i++];
    while (j < right) buffer[k++] = a[j++];
    for (int t = left; t < right; ++t) a[t] = buffer[t];
}

template <typename Fn>
long long elapsed_microseconds(Fn fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (int n : {1000, 10000, 100000}) {
        std::vector<int> data(n);
        for (int& x : data) x = dist(rng);

        // 两个算法使用同一份随机数据，保证性能和正确性对比公平。
        auto a = data;
        std::vector<int> buffer(n);
        const auto merge_time = elapsed_microseconds([&] {
            merge_sort(a, buffer, 0, static_cast<int>(a.size()));
        });

        auto b = data;
        const auto std_time = elapsed_microseconds([&] {
            std::sort(b.begin(), b.end());
        });

        std::cout << "n=" << n
                  << " merge_sort=" << merge_time << "us"
                  << " std::sort=" << std_time << "us"
                  << " sorted=" << std::boolalpha << (a == b) << "\n";
    }
    return 0;
}
