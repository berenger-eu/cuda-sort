#ifndef CUSORT_EXPERIMENTS_HPP
#define CUSORT_EXPERIMENTS_HPP

#include "cusort.hpp"

// Experimental v3 descendants. The original warp/v1/v2/v3 kernels in cusort.hpp
// are unchanged. Both experiments retain v3's two shared arrays and barriers:
// v4: uint32 XOR partners and explicitly unrolled fixed warp-local loops;
// v5: the same operations with compile-time working length, block size and stages.
namespace cusort_experiment {

// One aligned warp group. In the shared path the fixed 15 (initial) or 5 (tail)
// layers finish in destination; the caller swaps the global pointers once.
template <typename T, bool FullSM, bool Initial>
__device__ __forceinline__ void WarpGroup(T* source, T* destination, unsigned index) {
    constexpr unsigned mask = 0xffffffffU;
    const unsigned lane = index & 31U;
    if constexpr (FullSM) {
        if constexpr (Initial) {
            #pragma unroll
            for (unsigned stage = 1; stage < 32; stage <<= 1) {
                const T a = source[index];
                const T b = source[index ^ (2U * stage - 1U)];
                destination[index] = (index & stage) ? max(a, b) : min(a, b);
                swap(source, destination);
                __syncwarp(mask);
                #pragma unroll
                for (unsigned distance = stage / 2; distance > 0; distance >>= 1) {
                    const T x = source[index];
                    const T y = source[index ^ distance];
                    destination[index] = (index & distance) ? max(x, y) : min(x, y);
                    swap(source, destination);
                    __syncwarp(mask);
                }
            }
        } else {
            #pragma unroll
            for (unsigned distance = 16; distance > 0; distance >>= 1) {
                const T a = source[index];
                const T b = source[index ^ distance];
                destination[index] = (index & distance) ? max(a, b) : min(a, b);
                swap(source, destination);
                __syncwarp(mask);
            }
        }
    } else {
        T a = source[index];
        if constexpr (Initial) {
            #pragma unroll
            for (unsigned stage = 1; stage < 32; stage <<= 1) {
                const T b = ShuffleValue(mask, a, lane ^ (2U * stage - 1U));
                a = (lane & stage) ? max(a, b) : min(a, b);
                #pragma unroll
                for (unsigned distance = stage / 2; distance > 0; distance >>= 1) {
                    const T b = ShuffleValue(mask, a, lane ^ distance);
                    a = (lane & distance) ? max(a, b) : min(a, b);
                }
            }
        } else {
            #pragma unroll
            for (unsigned distance = 16; distance > 0; distance >>= 1) {
                const T b = ShuffleValue(mask, a, lane ^ distance);
                a = (lane & distance) ? max(a, b) : min(a, b);
            }
        }
        source[index] = a;
    }
}

template <typename T, bool FullSM, bool Initial, unsigned WorkingValues = 0, unsigned BlockThreads = 0>
__device__ __forceinline__ void WarpGroups(T*& source, T*& destination, unsigned count) {
    if constexpr (WorkingValues != 0) {
        constexpr unsigned Items = WorkingValues / BlockThreads;
        constexpr unsigned Unroll = Items < 8 ? Items : 8;
        #pragma unroll Unroll
        for (unsigned item = 0; item < Items; ++item)
            WarpGroup<T, FullSM, Initial>(source, destination, threadIdx.x + item * BlockThreads);
    } else {
        #pragma unroll 1
        for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
            WarpGroup<T, FullSM, Initial>(source, destination, i);
    }
    // Includes idle warps in v4: all threads keep identical buffer identities.
    if constexpr (FullSM) swap(source, destination);
}

template <typename T>
__device__ __forceinline__ void Output(T* source, T* destination, unsigned i,
                                     unsigned partnerMask, unsigned maximumBit) {
    const T a = source[i];
    const T b = source[i ^ partnerMask];
    destination[i] = (i & maximumBit) ? max(a, b) : min(a, b);
}

template <typename T, unsigned WorkingValues = 0, unsigned BlockThreads = 0>
__device__ __forceinline__ void BlockLayer(T*& source, T*& destination, unsigned count,
                                         unsigned partnerMask, unsigned maximumBit) {
    if constexpr (WorkingValues != 0) {
        constexpr unsigned Items = WorkingValues / BlockThreads;
        // Bound expansion for small blocks processing a large working set.
        constexpr unsigned Unroll = Items < 8 ? Items : 8;
        #pragma unroll Unroll
        for (unsigned item = 0; item < Items; ++item)
            Output(source, destination, threadIdx.x + item * BlockThreads, partnerMask, maximumBit);
    } else {
        #pragma unroll 1
        for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
            Output(source, destination, i, partnerMask, maximumBit);
    }
    swap(source, destination);
    __syncthreads();
}

// Template recursion makes each network mask/distance constant in v5. These
// calls inline; there is no device recursion or device-side dynamic dispatch.
template <unsigned Distance, typename T, unsigned WorkingValues, unsigned BlockThreads>
__device__ __forceinline__ void StaticMerge(T*& source, T*& destination) {
    if constexpr (Distance >= 32) {
        BlockLayer<T, WorkingValues, BlockThreads>(source, destination, WorkingValues, Distance, Distance);
        StaticMerge<Distance / 2, T, WorkingValues, BlockThreads>(source, destination);
    }
}

template <unsigned Stage, typename T, bool TestIfSorted, bool FullSM,
          unsigned WorkingValues, unsigned BlockThreads>
__device__ __forceinline__ void StaticStages(T*& source, T*& destination, int validCount) {
    if constexpr (Stage < WorkingValues) {
        BlockLayer<T, WorkingValues, BlockThreads>(source, destination, WorkingValues, 2U * Stage - 1U, Stage);
        StaticMerge<Stage / 2, T, WorkingValues, BlockThreads>(source, destination);
        WarpGroups<T, FullSM, false, WorkingValues, BlockThreads>(source, destination, WorkingValues);
        __syncthreads();
        if constexpr (TestIfSorted) {
            if (TestIfSorted_1Block_v3(source, validCount)) return;
        }
        StaticStages<Stage * 2, T, TestIfSorted, FullSM, WorkingValues, BlockThreads>(source, destination, validCount);
    }
}

template <typename T, bool TestIfSorted, bool FullSM, unsigned WorkingValues = 0, unsigned BlockThreads = 0>
__device__ __forceinline__ T* Core(T* source, int validCount, T* destination, unsigned count) {
    static_assert((WorkingValues == 0) == (BlockThreads == 0), "Specify both static dimensions or neither");
    if constexpr (WorkingValues != 0) {
        static_assert(WorkingValues >= 32 && (WorkingValues & (WorkingValues - 1)) == 0, "Power-of-two working length required");
        static_assert(BlockThreads >= 32 && BlockThreads <= WorkingValues &&
                      (BlockThreads & (BlockThreads - 1)) == 0, "Power-of-two block size required");
        assert(count == WorkingValues && blockDim.x == BlockThreads);
    }
    assert(validCount > 0 && unsigned(validCount) <= count);
    assert(count >= 32 && (count & (count - 1)) == 0);
    assert(blockDim.x >= 32 && blockDim.x % 32 == 0);
    if constexpr (TestIfSorted) {
        if (TestIfSorted_1Block_v3(source, validCount)) return source;
    }
    WarpGroups<T, FullSM, true, WorkingValues, BlockThreads>(source, destination, count);
    __syncthreads();
    if constexpr (TestIfSorted) {
        if (TestIfSorted_1Block_v3(source, validCount)) return source;
    }
    if constexpr (WorkingValues != 0) {
        StaticStages<32, T, TestIfSorted, FullSM, WorkingValues, BlockThreads>(source, destination, validCount);
    } else {
        #pragma unroll 1
        for (unsigned stage = 32; stage < count; stage <<= 1) {
            BlockLayer(source, destination, count, 2U * stage - 1U, stage);
            #pragma unroll 1
            for (unsigned distance = stage / 2; distance >= 32; distance >>= 1)
                BlockLayer(source, destination, count, distance, distance);
            WarpGroups<T, FullSM, false>(source, destination, count);
            __syncthreads();
            if constexpr (TestIfSorted) {
                if (TestIfSorted_1Block_v3(source, validCount)) return source;
            }
        }
    }
    return source;
}

// Both wrappers support in-place or separate global output and use exactly
// two shared arrays, as v3 does. AsyncCopy is an independent template choice.
template <typename T, bool TestIfSorted, bool FullSM, bool AsyncCopy,
          unsigned WorkingValues = 0, unsigned BlockThreads = 0>
__device__ __forceinline__ void Sort(T* input, T* output, int validCount, unsigned runtimeWorking) {
    constexpr bool Specialized = WorkingValues != 0;
    const unsigned count = Specialized ? WorkingValues : runtimeWorking;
    const unsigned stride = Specialized ? BlockThreads : blockDim.x;
    assert(validCount > 0 && unsigned(validCount) <= count);
    if constexpr (Specialized) assert(runtimeWorking == WorkingValues && blockDim.x == BlockThreads);
    extern __shared__ __align__(16) unsigned char experiment_smem[];
    T* source = reinterpret_cast<T*>(experiment_smem);
    T* destination = source + count;
    if constexpr (AsyncCopy) {
        const auto block = cooperative_groups::this_thread_block();
        cooperative_groups::memcpy_async(block, source, input, sizeof(T) * validCount);
        for (unsigned i = unsigned(validCount) + threadIdx.x; i < count; i += stride)
            source[i] = my_numeric_limits_max<T>();
        cooperative_groups::wait(block);
    } else {
        for (unsigned i = threadIdx.x; i < count; i += stride)
            source[i] = i < unsigned(validCount) ? input[i] : my_numeric_limits_max<T>();
        __syncthreads();
    }
    T* result = Core<T, TestIfSorted, FullSM, WorkingValues, BlockThreads>(source, validCount, destination, count);
    if constexpr (AsyncCopy) {
        const auto block = cooperative_groups::this_thread_block();
        cooperative_groups::memcpy_async(block, output, result, sizeof(T) * validCount);
        cooperative_groups::wait(block);
    } else {
        for (unsigned i = threadIdx.x; i < unsigned(validCount); i += stride) output[i] = result[i];
        __syncthreads();
    }
}
} // namespace cusort_experiment

// Named core entry points also support direct composition with existing code.
template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v4_xor(T* source, int n, T* buffer, unsigned working) {
    return cusort_experiment::Core<T, TestIfSorted, FullSM>(source, n, buffer, working);
}

template <typename T, unsigned WorkingValues, unsigned BlockThreads, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v5_static(T* source, int n, T* buffer) {
    return cusort_experiment::Core<T, TestIfSorted, FullSM, WorkingValues, BlockThreads>(source, n, buffer, WorkingValues);
}

template <typename T, bool TestIfSorted, bool FullSM, bool AsyncCopy>
__global__ void Sort_1Block_v4_xor(T* input, T* output, int n, unsigned working) {
    cusort_experiment::Sort<T, TestIfSorted, FullSM, AsyncCopy>(input, output, n, working);
}

template <typename T, unsigned WorkingValues, unsigned BlockThreads, bool TestIfSorted, bool FullSM, bool AsyncCopy>
__global__ void Sort_1Block_v5_static(T* input, T* output, int n) {
    cusort_experiment::Sort<T, TestIfSorted, FullSM, AsyncCopy, WorkingValues, BlockThreads>(input, output, n, WorkingValues);
}
#endif
