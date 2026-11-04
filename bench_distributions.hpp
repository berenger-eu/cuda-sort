#ifndef BENCH_DISTRIBUTIONS_HPP
#define BENCH_DISTRIBUTIONS_HPP

#include <cstdint>
#include <random>

// One simplified temporal update of sorted, regularly spaced integer keys.
// amplitude is measured in initial key spacings; it is not a percentage.
// No aligned windows: values may cross any warp/block boundary.
// setKey(position, key) preserves the caller's input-position record payloads.
template <typename SetKey>
void generateLocallyPerturbedKeys(int n, int amplitude, std::uint32_t seed, SetKey setKey) {
    constexpr int spacing = 4;
    const int delta = spacing * amplitude;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(-delta, delta);
    for (int i = 0; i < n; ++i)
        setKey(i, spacing * i + noise(rng));
}

#endif
