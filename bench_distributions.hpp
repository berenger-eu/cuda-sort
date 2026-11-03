#ifndef BENCH_DISTRIBUTIONS_HPP
#define BENCH_DISTRIBUTIONS_HPP

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

// Starting from keys [0,n), move exactly m distinct positions in one cycle.
// A cycle avoids unchanged selected positions and preserves the key multiset.
// setKey(position, key) leaves record payloads attached to their input positions.
template <typename SetKey>
void perturbSortedKeys(int n, int percent, std::uint32_t seed, SetKey setKey) {
    if (n < 2) return;
    const int moved = std::min(n, std::max(2, int((std::int64_t(n) * percent + 50) / 100)));
    std::vector<int> positions(n);
    std::iota(positions.begin(), positions.end(), 0);
    std::mt19937 rng(seed);
    for (int j = 0; j < moved; ++j) {
        std::uniform_int_distribution<int> pick(j, n - 1);
        std::swap(positions[j], positions[pick(rng)]);
    }
    for (int j = 0; j < moved; ++j)
        setKey(positions[j], positions[(j + 1) % moved]);
}

#endif
