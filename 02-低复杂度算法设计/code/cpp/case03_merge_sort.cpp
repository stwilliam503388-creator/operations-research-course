#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

void merge_sort(std::vector<int>& a, std::vector<int>& buffer, int left, int right) {
    if (right - left <= 1) return;
    const int mid = left + (right - left) / 2;
    merge_sort(a, buffer, left, mid);
    merge_sort(a, buffer, mid, right);

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
