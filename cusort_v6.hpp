#ifndef CUSORT_V6_HPP
#define CUSORT_V6_HPP

#include "cusort.hpp"

namespace cusort_v6 {

template <int Begin, int End, typename Function>
__device__ __forceinline__ void ForItems(Function&& function) {
    if constexpr (Begin < End) {
        function(std::integral_constant<int, Begin>{});
        ForItems<Begin + 1, End>(function);
    }
}


// Compile-time member access avoids runtime indexing of a local array. Actual
// register residency must still be checked in the compiler's resource report.
template <typename T, unsigned Count> struct Registers {
    T value;
    Registers<T, Count - 1> tail;
    template <unsigned Index> __device__ __forceinline__ T& get() {
        static_assert(Index < Count, "Register index out of range");
        if constexpr (Index == 0) return value;
        else return tail.template get<Index - 1>();
    }
};
template <typename T> struct Registers<T, 0> {};

template <unsigned Threads, typename T, unsigned Items>
__device__ __forceinline__ void Publish(Registers<T, Items>& values, T* target) {
    ForItems<0, Items>([&](auto item) {
        constexpr unsigned Q = decltype(item)::value;
        target[threadIdx.x + Q * Threads] = values.template get<Q>();
    });
}

template <unsigned Threads, unsigned Mask, unsigned MaximumBit, bool Shared,
          typename T, unsigned Items>
__device__ __forceinline__ void Layer(Registers<T, Items>& values, T*& scratch, T*& spare) {
    constexpr unsigned ThreadMask = Mask & (Threads - 1);
    constexpr unsigned ItemMask = Mask / Threads;
    if constexpr (ThreadMask == 0) {
        // Each disjoint pair is compared once, entirely within this thread.
        ForItems<0, Items>([&](auto item) {
            constexpr unsigned Q = decltype(item)::value;
            constexpr unsigned Partner = Q ^ ItemMask;
            if constexpr (Q < Partner) {
                const T a = values.template get<Q>();
                const T b = values.template get<Partner>();
                if constexpr ((Q * Threads & MaximumBit) != 0) {
                    values.template get<Q>() = max(a, b);
                    values.template get<Partner>() = min(b, a);
                } else {
                    values.template get<Q>() = min(a, b);
                    values.template get<Partner>() = max(b, a);
                }
            }
        });
    } else if constexpr (Shared || ThreadMask >= 32) {
        // Alternate snapshots: a later publication cannot overwrite a snapshot
        // still being read by another warp. See the transition fence in Stages.
        Publish<Threads>(values, scratch);
        if constexpr (ThreadMask >= 32) __syncthreads();
        else __syncwarp(0xffffffffU);
        ForItems<0, Items>([&](auto item) {
            constexpr unsigned Q = decltype(item)::value;
            const unsigned index = threadIdx.x + Q * Threads;
            const T a = values.template get<Q>();
            const T b = scratch[index ^ Mask];
            values.template get<Q>() = (index & MaximumBit) ? max(a, b) : min(a, b);
        });
        swap(scratch, spare);
    } else {
        // Snapshot the register tile: reflection layers can exchange different
        // items as well as lanes. Updating in place would corrupt later items.
        Registers<T, Items> next;
        ForItems<0, Items>([&](auto item) {
            constexpr unsigned Q = decltype(item)::value;
            const T a = values.template get<Q>();
            const T b = ShuffleValue(0xffffffffU, values.template get<Q ^ ItemMask>(),
                                     (threadIdx.x & 31U) ^ ThreadMask);
            const unsigned index = threadIdx.x + Q * Threads;
            next.template get<Q>() = (index & MaximumBit) ? max(a, b) : min(a, b);
        });
        values = next;
    }
}

template <unsigned Distance, unsigned Threads, bool Shared, typename T, unsigned Items>
__device__ __forceinline__ void Merge(Registers<T, Items>& values, T*& scratch, T*& spare) {
    if constexpr (Distance != 0) {
        Layer<Threads, Distance, Distance, Shared>(values, scratch, spare);
        if constexpr (Shared && Threads > 32 && Distance == 32) {
            // Warp-local publications can now recycle both buffers without
            // overwriting another warp's last cross-warp snapshot read.
            __syncthreads();
        }
        Merge<Distance / 2, Threads, Shared>(values, scratch, spare);
    }
}

template <unsigned Threads, typename T, unsigned Items>
__device__ __forceinline__ bool IsSorted(Registers<T, Items>& values, int validCount,
                                        T*& scratch, T*& spare) {
    Publish<Threads>(values, scratch);
    __syncthreads();
    const bool sorted = TestIfSorted_1Block_v3(scratch, validCount);
    swap(scratch, spare);
    return sorted;
}

template <unsigned Stage, unsigned Working, unsigned Threads, bool Sorted, bool Shared,
          typename T, unsigned Items>
__device__ __forceinline__ void Stages(Registers<T, Items>& values, int validCount,
                                      T*& scratch, T*& spare) {
    if constexpr (Stage < Working) {
        Layer<Threads, 2 * Stage - 1, Stage, Shared>(values, scratch, spare);
        if constexpr (Shared && Threads > 32 && Stage == 32) __syncthreads();
        Merge<Stage / 2, Threads, Shared>(values, scratch, spare);
        // Same checkpoints as v5: before sorting, after the initial warp sort,
        // then after each complete block merge. No claim of adaptivity added.
        if constexpr (Sorted && Stage >= 16) {
            if (IsSorted<Threads>(values, validCount, scratch, spare)) return;
        }
        Stages<Stage * 2, Working, Threads, Sorted, Shared>(values, validCount, scratch, spare);
    }
}

template <typename T, unsigned Working, unsigned Threads, bool Sorted, bool Shared>
__device__ __forceinline__ void Core(Registers<T, Working / Threads>& values, int validCount,
                                    T*& scratch, T*& spare) {
    static_assert(Working >= 32 && (Working & (Working - 1)) == 0, "Power-of-two working length required");
    static_assert(Threads >= 32 && Threads <= 1024 && Threads <= Working &&
                  (Threads & (Threads - 1)) == 0, "Power-of-two block size required");
    static_assert(Working / Threads <= 16, "v6 limits register tiles to 16 values per thread");
    assert(validCount > 0 && unsigned(validCount) <= Working && blockDim.x == Threads);
    if constexpr (Sorted) {
        if (IsSorted<Threads>(values, validCount, scratch, spare)) return;
    }
    Stages<1, Working, Threads, Sorted, Shared>(values, validCount, scratch, spare);
}

template <typename T, unsigned Working, unsigned Threads, bool Sorted, bool Shared, bool Async>
__device__ __forceinline__ void Sort(T* input, T* output, int validCount) {
    static_assert(Working >= Threads && Threads >= 32, "Invalid static dimensions");
    assert(validCount > 0 && unsigned(validCount) <= Working && blockDim.x == Threads);
    extern __shared__ __align__(16) unsigned char v6_smem[];
    T* spare = reinterpret_cast<T*>(v6_smem);
    T* scratch = spare + Working;
    Registers<T, Working / Threads> values;
    if constexpr (Async) {
        const auto block = cooperative_groups::this_thread_block();
        cooperative_groups::memcpy_async(block, spare, input, sizeof(T) * validCount);
        cooperative_groups::wait(block);
    }
    ForItems<0, Working / Threads>([&](auto item) {
        constexpr unsigned Q = decltype(item)::value;
        const unsigned index = threadIdx.x + Q * Threads;
        values.template get<Q>() = index < unsigned(validCount)
            ? (Async ? spare[index] : input[index]) : my_numeric_limits_max<T>();
    });
    __syncthreads();
    Core<T, Working, Threads, Sorted, Shared>(values, validCount, scratch, spare);
    if constexpr (Async) {
        Publish<Threads>(values, scratch);
        __syncthreads();
        const auto block = cooperative_groups::this_thread_block();
        cooperative_groups::memcpy_async(block, output, scratch, sizeof(T) * validCount);
        cooperative_groups::wait(block);
    } else {
        ForItems<0, Working / Threads>([&](auto item) {
            constexpr unsigned Q = decltype(item)::value;
            const unsigned index = threadIdx.x + Q * Threads;
            if (index < unsigned(validCount)) output[index] = values.template get<Q>();
        });
    }
}
} // namespace cusort_v6

template <typename T, unsigned Working, unsigned Threads, bool Sorted, bool Shared, bool Async>
__global__ void Sort_1Block_v6_register(T* input, T* output, int validCount) {
    cusort_v6::Sort<T, Working, Threads, Sorted, Shared, Async>(input, output, validCount);
}

#endif
