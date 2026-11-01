#ifndef CUSORT_V3_HPP
#define CUSORT_V3_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <math_constants.h>

#include "SpTimer.hpp"

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

#include <fstream>


template <bool TBoo>
struct ConstexprBool{
    static constexpr bool value = TBoo;
};

template <typename Func>
void for_bool(Func&& func) {
    func(ConstexprBool<true>());
    func(ConstexprBool<false>());
}

// Template iterator to iterate on integers at compile time
template<int Start, int End, typename Func>
void for_int_step1(Func&& func) {
    if constexpr (Start < End) {
        func(std::integral_constant<int, Start>{});
        for_int_step1<Start + 1, End>(std::forward<Func>(func));
    }
}

template<int Start, int End, typename Func>
void for_int_pow2(Func&& func) {
    if constexpr (Start < End) {
        func(std::integral_constant<int, Start>{});
        for_int_pow2<Start*2, End>(std::forward<Func>(func));
    }
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

constexpr const int DefaultNbThreadsPerBlock = 512;
constexpr const int DefaultMaxBitoSortV2 = DefaultNbThreadsPerBlock;

inline void CUDA_CHECK(cudaError_t err);


// Reusable storage for the allocation-free sorting path.  Construct and
// reserve this object outside the timed region, then reuse it for every sort.
template <typename T, int MaxNbParts = 1024>
class CuSortWorkspace {
    static_assert(MaxNbParts >= 32 && MaxNbParts <= 1024 &&
                      ((MaxNbParts - 1) & MaxNbParts) == 0,
                  "MaxNbParts must be a power of two in [32, 1024]");

    T* valuesBuffer_ = nullptr;
    T* pivots_ = nullptr;
    int* counters_ = nullptr;
    int* blockOffsets_ = nullptr;
    int* blockUniform_ = nullptr;
    int* uniformParts_ = nullptr;
    int* activeNbParts_ = nullptr;
    size_t valuesCapacity_ = 0;
    int persistentBlocks_ = 1;
    std::vector<int> hostCounters_;
    std::vector<int> hostUniformParts_;
    std::vector<int> hostLargePartitions_;
    std::vector<cudaStream_t> sortStreams_;
    std::vector<cudaEvent_t> sortDone_;
    bool workerEventsRecorded_ = false;

public:
    explicit CuSortWorkspace(size_t initialCapacity = 0,
                             int blocksPerSM = 4,
                             int nbSortStreams = 8)
        : hostCounters_(MaxNbParts + 1),
          hostUniformParts_(MaxNbParts), hostLargePartitions_(MaxNbParts) {
        int device = 0;
        int multiprocessorCount = 1;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessorCount,
                                          cudaDevAttrMultiProcessorCount,
                                          device));
        persistentBlocks_ = std::max(1, blocksPerSM * multiprocessorCount);
        CUDA_CHECK(cudaMalloc(&pivots_, MaxNbParts * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&counters_, (MaxNbParts + 2) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&blockOffsets_,
                              static_cast<size_t>(persistentBlocks_) *
                                  MaxNbParts * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&blockUniform_,
                              static_cast<size_t>(persistentBlocks_) *
                                  MaxNbParts * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&uniformParts_, MaxNbParts * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&activeNbParts_, sizeof(int)));
        const int poolSize = std::max(1, nbSortStreams);
        sortStreams_.resize(poolSize);
        sortDone_.resize(poolSize);
        for(int idx = 0 ; idx < poolSize ; ++idx){
            CUDA_CHECK(cudaStreamCreateWithFlags(&sortStreams_[idx],
                                                  cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&sortDone_[idx],
                                                 cudaEventDisableTiming));
        }
        if (initialCapacity) reserve(initialCapacity);
    }

    ~CuSortWorkspace() {
        for(cudaStream_t worker : sortStreams_) cudaStreamSynchronize(worker);
        for(cudaEvent_t event : sortDone_) cudaEventDestroy(event);
        for(cudaStream_t worker : sortStreams_) cudaStreamDestroy(worker);
        if (valuesBuffer_) cudaFree(valuesBuffer_);
        if (pivots_) cudaFree(pivots_);
        if (counters_) cudaFree(counters_);
        if (blockOffsets_) cudaFree(blockOffsets_);
        if (blockUniform_) cudaFree(blockUniform_);
        if (uniformParts_) cudaFree(uniformParts_);
        if (activeNbParts_) cudaFree(activeNbParts_);
    }

    CuSortWorkspace(const CuSortWorkspace&) = delete;
    CuSortWorkspace& operator=(const CuSortWorkspace&) = delete;

    CuSortWorkspace(CuSortWorkspace&& other) noexcept
        : valuesBuffer_(other.valuesBuffer_), pivots_(other.pivots_),
          counters_(other.counters_), blockOffsets_(other.blockOffsets_),
          blockUniform_(other.blockUniform_), uniformParts_(other.uniformParts_),
          activeNbParts_(other.activeNbParts_),
          valuesCapacity_(other.valuesCapacity_),
          persistentBlocks_(other.persistentBlocks_),
          hostCounters_(std::move(other.hostCounters_)),
          hostUniformParts_(std::move(other.hostUniformParts_)),
          hostLargePartitions_(std::move(other.hostLargePartitions_)),
          sortStreams_(std::move(other.sortStreams_)),
          sortDone_(std::move(other.sortDone_)),
          workerEventsRecorded_(other.workerEventsRecorded_) {
        other.valuesBuffer_ = nullptr;
        other.pivots_ = nullptr;
        other.counters_ = nullptr;
        other.blockOffsets_ = nullptr;
        other.blockUniform_ = nullptr;
        other.uniformParts_ = nullptr;
        other.activeNbParts_ = nullptr;
        other.valuesCapacity_ = 0;
        other.workerEventsRecorded_ = false;
        other.sortStreams_.clear();
        other.sortDone_.clear();
    }

    void reserve(size_t capacity) {
        if (capacity <= valuesCapacity_) return;
        if(workerEventsRecorded_){
            for(cudaStream_t worker : sortStreams_){
                CUDA_CHECK(cudaStreamSynchronize(worker));
            }
        }
        if (valuesBuffer_) CUDA_CHECK(cudaFree(valuesBuffer_));
        CUDA_CHECK(cudaMalloc(&valuesBuffer_, capacity * sizeof(T)));
        valuesCapacity_ = capacity;
    }

    void setBlocksPerSM(int blocksPerSM) {
        int device = 0;
        int multiprocessorCount = 1;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessorCount,
                                          cudaDevAttrMultiProcessorCount,
                                          device));
        const int requestedBlocks = std::max(1, blocksPerSM * multiprocessorCount);
        if(requestedBlocks != persistentBlocks_){
            if(blockOffsets_) CUDA_CHECK(cudaFree(blockOffsets_));
            persistentBlocks_ = requestedBlocks;
            CUDA_CHECK(cudaMalloc(&blockOffsets_,
                                  static_cast<size_t>(persistentBlocks_) *
                                      MaxNbParts * sizeof(int)));
            if(blockUniform_) CUDA_CHECK(cudaFree(blockUniform_));
            CUDA_CHECK(cudaMalloc(&blockUniform_,
                                  static_cast<size_t>(persistentBlocks_) *
                                      MaxNbParts * sizeof(int)));
        }
    }

    T* valuesBuffer() { return valuesBuffer_; }
    T* pivots() { return pivots_; }
    int* counters() { return counters_; }
    int* skipScatterFlag() { return counters_ + MaxNbParts + 1; }
    int* blockOffsets() { return blockOffsets_; }
    int* blockUniform() { return blockUniform_; }
    int* uniformParts() { return uniformParts_; }
    int* activeNbParts() { return activeNbParts_; }
    size_t capacity() const { return valuesCapacity_; }
    int persistentBlocks() const { return persistentBlocks_; }
    std::vector<int>& hostCounters() { return hostCounters_; }
    std::vector<int>& hostUniformParts() { return hostUniformParts_; }
    std::vector<int>& hostLargePartitions() { return hostLargePartitions_; }
    int nbSortStreams() const { return static_cast<int>(sortStreams_.size()); }
    cudaStream_t sortStream(int idx) const { return sortStreams_[idx]; }
    cudaEvent_t sortDone(int idx) const { return sortDone_[idx]; }
    bool workerEventsRecorded() const { return workerEventsRecorded_; }
    void markWorkerEventsRecorded() { workerEventsRecorded_ = true; }
};



constexpr __forceinline__ __device__ __host__ int LowerPowerOf2EqualOrSmaller(int nbValuesToSort){
    int res = 1;
    while(res*2 <= nbValuesToSort){
        res *= 2;
    }
    return res;
}

template <typename T>
constexpr __forceinline__ __device__ __host__ int ComputeDefaultMaxBitoSort(){
    const int SharedMemPerBlock = 49152; // in bytes, for current GPU architecture
    const int MaxValuesBySharedMem = SharedMemPerBlock / sizeof(T);
    const int MaxValuesBySharedMemPow2 = LowerPowerOf2EqualOrSmaller(MaxValuesBySharedMem);
    return MaxValuesBySharedMemPow2/2;
}


inline void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__forceinline__ __device__ __host__ int GreaterPowerOf2EqualOrLarger(int nbValuesToSort){
    int res = 1;
    while(res < nbValuesToSort){
        res *= 2;
    }
    return res;
}



template <typename T>
__host__ __device__ __forceinline__  constexpr T my_numeric_limits_max(){
    if constexpr (std::is_same<T, int>::value){
        return 2147483647;
    }
    else if constexpr(std::is_same<T, float>::value){
        return std::numeric_limits<float>::infinity();
    }
    else if constexpr(std::is_same<T, double>::value){
        return std::numeric_limits<double>::infinity();
    }
    else{
        return T::max();
    }
}

template <typename T>
__host__ __device__ __forceinline__  constexpr T my_numeric_limits_min(){
    if constexpr (std::is_same<T, int>::value){
        return -2147483648;
    }
    else if constexpr(std::is_same<T, float>::value){
        return -std::numeric_limits<float>::infinity();
    }
    else if constexpr(std::is_same<T, double>::value){
        return -std::numeric_limits<double>::infinity();
    }
    else{
        return T::min();
    }
}

#ifdef __CUDA_ARCH__
template <typename T>
__device__ __forceinline__ T getNextVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val == my_numeric_limits_max<T>() ? val : val + 1;
    }
    else if constexpr(std::is_same<T, float>::value){
        return std::nextafter(val, CUDART_INF_F);
    }
    else if constexpr(std::is_same<T, double>::value){
        return std::nextafter(val, CUDART_INF);
    }
    else{
        return val.nextVal();
    }
}

template <typename T>
__device__ __forceinline__ T getPrevVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val == my_numeric_limits_min<T>() ? val : val - 1;
    }
    else if constexpr(std::is_same<T, float>::value){
        return std::nextafter(val, -CUDART_INF_F);
    }
    else if constexpr(std::is_same<T, double>::value){
        return std::nextafter(val, -CUDART_INF);
    }
    else{
        return val.prevVal();
    }
}

#else

template <typename T>
inline __host__ T getNextVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val == my_numeric_limits_max<T>() ? val : val + 1;
    }
    else if constexpr(std::is_same<T, float>::value){
        return std::nextafter(val, std::numeric_limits<float>::infinity());
    }
    else if constexpr(std::is_same<T, double>::value){
        return std::nextafter(val, std::numeric_limits<double>::infinity());
    }
    else{
        return val.nextVal();
    }
}

template <typename T>
inline __host__ T getPrevVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val == my_numeric_limits_min<T>() ? val : val - 1;
    }
    else if constexpr(std::is_same<T, float>::value){
        return std::nextafter(val, -std::numeric_limits<float>::infinity());
    }
    else if constexpr(std::is_same<T, double>::value){
        return std::nextafter(val, -std::numeric_limits<double>::infinity());
    }
    else{
        return val.prevVal();
    }
}

#endif // __CUDA_ARCH__

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

// CUDA shuffle intrinsics accept scalar operands, not arbitrary records.
// Exchange all object bytes so key/payload records remain intact. Padding in
// the last word is zeroed and is never copied into the returned object.
template <typename T>
__device__ __forceinline__ T ShuffleValue(unsigned mask, const T& value, int sourceLane) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Warp exchange requires a trivially copyable value");
    unsigned words[(sizeof(T) + sizeof(unsigned) - 1) / sizeof(unsigned)] = {};
    memcpy(words, &value, sizeof(T));
    #pragma unroll
    for (int idx = 0; idx < int(sizeof(words) / sizeof(unsigned)); ++idx) {
        words[idx] = __shfl_sync(mask, words[idx], sourceLane);
    }
    T result;
    memcpy(&result, words, sizeof(T));
    return result;
}

template <typename T>
__device__ __forceinline__ T ShuffleUpValue(unsigned mask, const T& value, unsigned delta) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Warp exchange requires a trivially copyable value");
    unsigned words[(sizeof(T) + sizeof(unsigned) - 1) / sizeof(unsigned)] = {};
    memcpy(words, &value, sizeof(T));
    #pragma unroll
    for (int idx = 0; idx < int(sizeof(words) / sizeof(unsigned)); ++idx) {
        words[idx] = __shfl_up_sync(mask, words[idx], delta);
    }
    T result;
    memcpy(&result, words, sizeof(T));
    return result;
}

// This function tests if values are sorted in a warp
// each thread holding one value to test
template <typename T>
__device__ __forceinline__ bool TestIfSorted_1Warp(const T& dataToSort) {
    // Get the active thread mask and the current lane ID.
    unsigned int mask = __activemask();
    int lane = threadIdx.x & 31;  // assuming warp size is 32

    // Each thread (except lane 0) compares its value with its left neighbor.
    bool isSorted = true;
    T prev = ShuffleUpValue(mask, dataToSort, 1);
    if (lane > 0) {
        isSorted = (prev <= dataToSort);
    }
    // Propagate the result to all threads in the warp.
    return __all_sync(mask, isSorted);
}


// This function check if the array given in parameter is sorted.
// It works for one block of thread (so the tests are performed for only the block)
// There must be at least nbValuesToSort threads in the block
template <typename T>
__device__ __forceinline__ bool TestIfSorted_1Block(const T* dataToSort, const int nbValuesToSort) {
    assert(nbValuesToSort <= blockDim.x);

    __shared__ volatile int isSorted;
    if (threadIdx.x == 0) {
        isSorted = 1;
    }
    __syncthreads();

    if (threadIdx.x > 0 && threadIdx.x < nbValuesToSort) {
        if (dataToSort[threadIdx.x] < dataToSort[threadIdx.x - 1]) {
            atomicExch((int*)&isSorted, 0);
        }
    }

    __syncthreads();

    return isSorted;
}

template <typename T>
__device__ __forceinline__ bool TestIfSorted_1Block_v3(const T* dataToSort, const int nbValuesToSort) {
    __shared__ volatile int isSorted;
    if (threadIdx.x == 0) {
        isSorted = 1;
    }
    __syncthreads();

    bool threadIsSorted = 1;
    for(int idx = threadIdx.x + 1 ; idx < nbValuesToSort ; idx += blockDim.x){
        if (dataToSort[idx] < dataToSort[idx - 1]) {
            threadIsSorted = 0;
        }
    }
    if(threadIsSorted == 0){
        atomicExch((int*)&isSorted, 0);
    }

    __syncthreads();

    return isSorted;
}

// Keep the local endpoint on ties. Besides preserving both equal-key records,
// this prevents maximum-key padding from swapping into the valid prefix.
// Get the min between two values
template <typename T>
__device__ __forceinline__ T min(const T& a, const T& b){
    return b < a ? b : a;
}

// Get the max between two values
template <typename T>
__device__ __forceinline__ T max(const T& a, const T& b){
    return a < b ? b : a;
}

template <typename T>
__device__ __forceinline__ void swap(T& a, T& b){
    T tmp = a;
    a = b;
    b = tmp;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

// Sort 32 values or less. The values are sorted in place.
// There might be more threads than 32.
// If FullSM is used, the shared memory is used to store the values while sorting.
// Otherwise the values are sorted using warp shuffle.
// The first warp sorts; all block threads participate in the final barrier.
template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ void Core_Sort_1Warp_SM_le32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort <= 32);
    assert(32 <= blockDim.x);

    // Only the first 32 threads are used
    if(threadIdx.x < 32){
        bool isSorted = false;
        if constexpr (TestIfSorted){
            if(TestIfSorted_1Warp(dataToSort[threadIdx.x])){
                isSorted = true;
            }
        }
        if(isSorted == false){
            unsigned laneId;
            asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
            int vecincout = 0;
            if constexpr (FullSM){
                for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
                    {
                        const int ffttout = (threadIdx.x & stepout);
                        vecincout = vecincout + (ffttout? -stepout : stepout);
                        const int vecpermute = threadIdx.x + vecincout;
                        const T a = dataToSort[threadIdx.x];
                        const T b = dataToSort[vecpermute];
                        __syncwarp(0xffffffff);
                        const T best = (ffttout ? max(a, b) : min(a, b));
                        dataToSort[threadIdx.x] = best;
                        __syncwarp(0xffffffff);
                    }
            
                    for(long int step = stepout/2 ; step >= 1 ; step/=2){
                        const int vecinc = step;
                        const int fftt = (threadIdx.x & step);
                        const int vecincout2 = (fftt? -vecinc  : vecinc);
                        const int vecpermute = (threadIdx.x + vecincout2);
                        const T a = dataToSort[threadIdx.x];
                        const T b = dataToSort[vecpermute];
                        __syncwarp(0xffffffff);
                        const T best = (fftt ? max(a, b) : min(a, b));
                        dataToSort[threadIdx.x] = best;
                        __syncwarp(0xffffffff);
                    }

                    if constexpr (TestIfSorted){
                        if(TestIfSorted_1Warp(dataToSort[threadIdx.x])){
                            break;
                        }
                    }
                }
            }
            else{
                const unsigned fullMask = 0xFFFFFFFF;
                T a = dataToSort[threadIdx.x];
                for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
                    {
                        const int ffttout = (threadIdx.x & stepout);
                        vecincout = vecincout + (ffttout? -stepout : stepout);
                        const int vecpermute = (vecincout+laneId);
                        const T b = ShuffleValue(fullMask, a, vecpermute);
                        a = (ffttout ? max(a, b) : min(a, b));
                    }
            
                    for(long int step = stepout/2 ; step >= 1 ; step/=2){
                        const int vecinc = step;
                        const int fftt = (threadIdx.x & step);
                        const int vecincout2 = (fftt? -vecinc  : vecinc);
                        const int vecpermute = (vecincout2+laneId);
                        const T b = ShuffleValue(fullMask, a, vecpermute);
                        a = (fftt ? max(a, b) : min(a, b));
                    }

                    if constexpr (TestIfSorted){
                        if(TestIfSorted_1Warp(a)){
                            break;
                        }
                    }
                }
                dataToSort[threadIdx.x] = a;
            }
        }
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Warp_SM_le32_d(T* dataToSort, T* output, const int nbValuesToSort){
    assert(nbValuesToSort <= 32);
    assert(32 <= blockDim.x);
    __shared__ T shared[32];
    // Copy
    if(threadIdx.x < nbValuesToSort){
        shared[threadIdx.x] = dataToSort[threadIdx.x];
    }
    else if(threadIdx.x < 32){
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }
    // Publish the initialized shared values before warp readers use them.
    if(threadIdx.x < 32) __syncwarp(0xffffffff);
    // Sort
    Core_Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(shared, nbValuesToSort);
    // Copy back
    if(threadIdx.x < nbValuesToSort){
        output[threadIdx.x] = shared[threadIdx.x];
    }
    // Ensure all threads see the new data
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Warp_SM_le32(T* dataToSort, const int nbValuesToSort){
    Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort, dataToSort, nbValuesToSort);
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Warp_SM_le32_outofplace(T* dataToSort, T* output, const int nbValuesToSort){
    Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort, output, nbValuesToSort);
}

// Sort at least 32 vales inplace.
// One padded value per thread: use a power-of-two block size of at least 32.
template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ void Core_Sort_1Block_SM_gt32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort > 0 && nbValuesToSort <= blockDim.x);
    assert(blockDim.x >= 32 && (blockDim.x & (blockDim.x - 1)) == 0);
    if constexpr(FullSM == false){
        assert(blockDim.x%32 == 0);
    }

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Block(dataToSort, nbValuesToSort)){
            return;
        }
    }

    unsigned laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    const unsigned fullMask = 0xFFFFFFFF;
    int vecincout = 0;
    if constexpr (FullSM){
        for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
            {
                const int ffttout = (threadIdx.x & stepout);
                vecincout = vecincout + (ffttout? -stepout : stepout);
                const int vecpermute = threadIdx.x + vecincout;
                const T a = dataToSort[threadIdx.x];
                const T b = dataToSort[vecpermute];
                const T best = (ffttout ? max(a, b) : min(a, b));
                __syncwarp(0xffffffff);
                dataToSort[threadIdx.x] = best;
                __syncwarp(0xffffffff);
            }
    
            for(long int step = stepout/2 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (threadIdx.x + vecincout2);
                const T a = dataToSort[threadIdx.x];
                const T b = dataToSort[vecpermute];
                const T best = (fftt ? max(a, b) : min(a, b));
                __syncwarp(0xffffffff);
                dataToSort[threadIdx.x] = best;
                __syncwarp(0xffffffff);
            }
        }
    }
    else{
        T a = dataToSort[threadIdx.x];
        for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
            {
                const int ffttout = (threadIdx.x & stepout);
                vecincout = vecincout + (ffttout? -stepout : stepout);
                const int vecpermute = (vecincout+laneId);
                const T b = ShuffleValue(fullMask, a, vecpermute);
                a = (ffttout ? max(a, b) : min(a, b));
            }
    
            for(long int step = stepout/2 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (vecincout2+laneId);
                const T b = ShuffleValue(fullMask, a, vecpermute);
                a = (fftt ? max(a, b) : min(a, b));
            }
        }
        dataToSort[threadIdx.x] = a;
    }
    __syncthreads();

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Block(dataToSort, nbValuesToSort)){
            return;
        }
    }

    for(long int stepout = 32 ; stepout < nbValuesToSort ; stepout *= 2){
        {
            const int ffttout = (threadIdx.x & stepout);
            vecincout = vecincout + (ffttout? -stepout : stepout);
            const int vecpermute = threadIdx.x + vecincout;
            const T a = dataToSort[threadIdx.x];
            const T b = dataToSort[vecpermute];
            const T best = (ffttout ? max(a, b) : min(a, b));
            __syncthreads();
            dataToSort[threadIdx.x] = best;
            __syncthreads();
        }

        for(long int step = stepout/2 ; step >= 32 ; step/=2){
            const int vecinc = step;
            const int fftt = (threadIdx.x & step);
            const int vecincout2 = (fftt? -vecinc  : vecinc);
            const int vecpermute = (threadIdx.x + vecincout2);
            const T a = dataToSort[threadIdx.x];
            const T b = dataToSort[vecpermute];
            const T best = (fftt ? max(a, b) : min(a, b));
            __syncthreads();
            dataToSort[threadIdx.x] = best;
            __syncthreads();
        }

        if constexpr (FullSM){
            for(long int step = 16 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (threadIdx.x + vecincout2);
                const T a = dataToSort[threadIdx.x];
                const T b = dataToSort[vecpermute];
                const T best = (fftt ? max(a, b) : min(a, b));
                __syncwarp(0xffffffff);
                dataToSort[threadIdx.x] = best;
                __syncwarp(0xffffffff);
            }
        }
        else{
            T a = dataToSort[threadIdx.x];
            for(long int step = 16 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (vecincout2+laneId);
                const T b = ShuffleValue(fullMask, a, vecpermute);
                a = (fftt ? max(a, b) : min(a, b));
            }
            dataToSort[threadIdx.x] = a;
        }
        __syncthreads();

        if constexpr (TestIfSorted){
            if(TestIfSorted_1Block(dataToSort, nbValuesToSort)){
                return;
            }
        }
    }
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_gt32_d_extsm(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ __align__(16) unsigned char smem[]; // untyped dynamic shared memory
    T* shared = reinterpret_cast<T*>(smem);
    assert(nbValuesToSort <= blockDim.x);
    // Copy
    if(threadIdx.x < nbValuesToSort){
        shared[threadIdx.x] = dataToSort[threadIdx.x];
    }
    else{
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }
    __syncthreads();
    // Sort
    Core_Sort_1Block_SM_gt32_d<T, TestIfSorted, FullSM>(shared, nbValuesToSort);
    // Copy back
    if(threadIdx.x < nbValuesToSort){
        dataToSort[threadIdx.x] = shared[threadIdx.x];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}

// Out-of-place form of the v1 kernel.  This is useful for larger element
// types, for which the benchmarks show that v1 is faster than the two-buffer
// v2 implementation.
template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_gt32_d_extsm_outofplace(
        const T* dataToSort, T* output, const int nbValuesToSort) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* shared = reinterpret_cast<T*>(smem);
    assert(nbValuesToSort <= blockDim.x);
    if(threadIdx.x < nbValuesToSort){
        shared[threadIdx.x] = dataToSort[threadIdx.x];
    }
    else{
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }
    __syncthreads();
    Core_Sort_1Block_SM_gt32_d<T, TestIfSorted, FullSM>(shared, nbValuesToSort);
    if(threadIdx.x < nbValuesToSort){
        output[threadIdx.x] = shared[threadIdx.x];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_outofplace(
        const T* dataToSort, T* output, const int nbValuesToSort) {
    Sort_1Block_gt32_d_extsm_outofplace<T, TestIfSorted, FullSM>(
        dataToSort, output, nbValuesToSort);
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d_extsm(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ __align__(16) unsigned char smem[]; // untyped dynamic shared memory
    T* shared_data = reinterpret_cast<T*>(smem);

    auto block = cooperative_groups::this_thread_block();

    // Cooperative Async Copy
    cooperative_groups::memcpy_async(block, shared_data, dataToSort, sizeof(T) * nbValuesToSort);
    if(nbValuesToSort <= threadIdx.x){
        shared_data[threadIdx.x] = my_numeric_limits_max<T>();
    }

    cooperative_groups::wait(block); // Ensure data is fully copied

    // Sort
    Core_Sort_1Block_SM_gt32_d<T, TestIfSorted, FullSM>(shared_data, nbValuesToSort);

    // Copy back
    cooperative_groups::memcpy_async(block, dataToSort, shared_data, sizeof(T) * nbValuesToSort);
    cooperative_groups::wait(block); // Ensure copy back is complete
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_async_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v2(T* dataToSort, const int nbValuesToSort,
                                                            T* buffer){
    assert(nbValuesToSort > 0 && nbValuesToSort <= blockDim.x);
    assert(blockDim.x >= 32 && (blockDim.x & (blockDim.x - 1)) == 0);
    if constexpr(FullSM == false){
        assert(blockDim.x%32 == 0);
    }

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Block(dataToSort, nbValuesToSort)){
            return dataToSort;
        }
    }

    unsigned laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    const unsigned fullMask = 0xFFFFFFFF;
    int vecincout = 0;
    if constexpr (FullSM){
        for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
            {
                const int ffttout = (threadIdx.x & stepout);
                vecincout = vecincout + (ffttout? -stepout : stepout);
                const int vecpermute = threadIdx.x + vecincout;
                const T a = dataToSort[threadIdx.x];
                const T b = dataToSort[vecpermute];
                const T best = (ffttout ? max(a, b) : min(a, b));
                buffer[threadIdx.x] = best;
                swap(dataToSort, buffer);
                __syncwarp(0xffffffff);
            }
    
            for(long int step = stepout/2 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (threadIdx.x + vecincout2);
                const T a = dataToSort[threadIdx.x];
                const T b = dataToSort[vecpermute];
                const T best = (fftt ? max(a, b) : min(a, b));
                buffer[threadIdx.x] = best;
                swap(dataToSort, buffer);
                __syncwarp(0xffffffff);
            }
        }
    }
    else{
        T a = dataToSort[threadIdx.x];
        for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
            {
                const int ffttout = (threadIdx.x & stepout);
                vecincout = vecincout + (ffttout? -stepout : stepout);
                const int vecpermute = (vecincout+laneId);
                const T b = ShuffleValue(fullMask, a, vecpermute);
                a = (ffttout ? max(a, b) : min(a, b));
            }
    
            for(long int step = stepout/2 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (vecincout2+laneId);
                const T b = ShuffleValue(fullMask, a, vecpermute);
                a = (fftt ? max(a, b) : min(a, b));
            }
        }
        dataToSort[threadIdx.x] = a;
    }
    __syncthreads();

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Block(dataToSort, nbValuesToSort)){
            return dataToSort;
        }
    }

    for(long int stepout = 32 ; stepout < nbValuesToSort ; stepout *= 2){
        {
            const int ffttout = (threadIdx.x & stepout);
            vecincout = vecincout + (ffttout? -stepout : stepout);
            const int vecpermute = threadIdx.x + vecincout;
            const T a = dataToSort[threadIdx.x];
            const T b = dataToSort[vecpermute];
            const T best = (ffttout ? max(a, b) : min(a, b));
            buffer[threadIdx.x] = best;
            swap(dataToSort, buffer);
            __syncthreads();
        }

        for(long int step = stepout/2 ; step >= 32 ; step/=2){
            const int vecinc = step;
            const int fftt = (threadIdx.x & step);
            const int vecincout2 = (fftt? -vecinc  : vecinc);
            const int vecpermute = (threadIdx.x + vecincout2);
            const T a = dataToSort[threadIdx.x];
            const T b = dataToSort[vecpermute];
            const T best = (fftt ? max(a, b) : min(a, b));
            buffer[threadIdx.x] = best;
            swap(dataToSort, buffer);
            __syncthreads();
        }

        if constexpr (FullSM){
            for(long int step = 16 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (threadIdx.x + vecincout2);
                const T a = dataToSort[threadIdx.x];
                const T b = dataToSort[vecpermute];
                const T best = (fftt ? max(a, b) : min(a, b));
                buffer[threadIdx.x] = best;
                swap(dataToSort, buffer);
                __syncwarp(0xffffffff);
            }
        }
        else{
            T a = dataToSort[threadIdx.x];
            for(long int step = 16 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (vecincout2+laneId);
                const T b = ShuffleValue(fullMask, a, vecpermute);
                a = (fftt ? max(a, b) : min(a, b));
            }
            dataToSort[threadIdx.x] = a;
        }
        __syncthreads();

        if constexpr (TestIfSorted){
            if(TestIfSorted_1Block(dataToSort, nbValuesToSort)){
                return dataToSort;
            }
        }
    }

    return dataToSort;
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_gt32_d_extsm_v2(T* dataToSort, T* output, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ __align__(16) unsigned char smem[]; // untyped dynamic shared memory
    T* shared = reinterpret_cast<T*>(smem);

    assert(nbValuesToSort <= blockDim.x);
    // Copy
    if(threadIdx.x < nbValuesToSort){
        shared[threadIdx.x] = dataToSort[threadIdx.x];
    }
    else{
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }
    __syncthreads();
    // Sort
    T* buffer = shared + blockDim.x;
    T* lastArray = Core_Sort_1Block_SM_gt32_d_v2<T, TestIfSorted, FullSM>(shared, nbValuesToSort, buffer);
    // Copy back
    if(threadIdx.x < nbValuesToSort){
        output[threadIdx.x] = lastArray[threadIdx.x];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_v2(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort, dataToSort, nbValuesToSort);
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_v2_outofplace(T* dataToSort, T* output, const int nbValuesToSort) {
    Sort_1Block_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort, output, nbValuesToSort);
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d_extsm_v2(T* dataToSort, T* output, const int nbValuesToSort) {
    // Declare shared memory ptr
        extern __shared__ __align__(16) unsigned char smem[]; // untyped dynamic shared memory
    T* shared_data = reinterpret_cast<T*>(smem);

    auto block = cooperative_groups::this_thread_block();

    // Cooperative Async Copy
    cooperative_groups::memcpy_async(block, shared_data, dataToSort, sizeof(T) * nbValuesToSort);
    if(nbValuesToSort <= threadIdx.x){
        shared_data[threadIdx.x] = my_numeric_limits_max<T>();
    }

    cooperative_groups::wait(block); // Ensure data is fully copied

    // Sort
    T* buffer = shared_data + blockDim.x;
    T* lastArray = Core_Sort_1Block_SM_gt32_d_v2<T, TestIfSorted, FullSM>(shared_data, nbValuesToSort, buffer);

    // Copy back
    cooperative_groups::memcpy_async(block, output, lastArray, sizeof(T) * nbValuesToSort);
    cooperative_groups::wait(block); // Ensure copy back is complete
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm_v2(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_async_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort, dataToSort, nbValuesToSort);
}

// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm_v2_outofplace(T* dataToSort, T* output, const int nbValuesToSort) {
    Sort_1Block_async_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort, output, nbValuesToSort);
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v3(T* dataToSort, const int nbValuesToSort,
                                                            T* buffer, const int nbWorkingValues){
    assert(nbValuesToSort > 0 && nbValuesToSort <= nbWorkingValues);
    assert(nbWorkingValues >= 32 && (nbWorkingValues & (nbWorkingValues - 1)) == 0);
    assert(blockDim.x >= 32 && blockDim.x % 32 == 0);
    if constexpr(FullSM == false){
        assert(blockDim.x%32 == 0);
    }

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Block_v3(dataToSort, nbValuesToSort)){
            return dataToSort;
        }
    }

    unsigned laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    const unsigned fullMask = 0xFFFFFFFF;
    if constexpr (FullSM){
        T* dataToSortOrig = dataToSort;
        T* bufferOrig = buffer;
        for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){ 
            int vecincout = 0;
            dataToSort = dataToSortOrig;
            buffer = bufferOrig;
            for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
                {
                    const int ffttout = (idxVal & stepout);
                    vecincout = vecincout + (ffttout? -stepout : stepout);
                    const int vecpermute = idxVal + vecincout;
                    const T a = dataToSort[idxVal];
                    const T b = dataToSort[vecpermute];
                    const T best = (ffttout ? max(a, b) : min(a, b));
                    buffer[idxVal] = best;
                    swap(dataToSort, buffer);
                    __syncwarp(0xffffffff);
                }
        
                for(long int step = stepout/2 ; step >= 1 ; step/=2){
                    const int vecinc = step;
                    const int fftt = (idxVal & step);
                    const int vecincout2 = (fftt? -vecinc  : vecinc);
                    const int vecpermute = (idxVal + vecincout2);
                    const T a = dataToSort[idxVal];
                    const T b = dataToSort[vecpermute];
                    const T best = (fftt ? max(a, b) : min(a, b));
                    buffer[idxVal] = best;
                    swap(dataToSort, buffer);
                    __syncwarp(0xffffffff);
                }
            }
        }
        // Fifteen warp-local layers exchange buffers an odd number of times.
        // Idle warps must keep the same current-buffer identity as active ones.
        if(threadIdx.x >= nbWorkingValues) swap(dataToSort, buffer);
    }
    else{
        for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){    
            int vecincout = 0; 
            T a = dataToSort[idxVal];
            for(long int stepout = 1 ; stepout < 32 ; stepout *= 2){
                {
                    const int ffttout = (idxVal & stepout);
                    vecincout = vecincout + (ffttout? -stepout : stepout);
                    const int vecpermute = (vecincout+laneId);
                    const T b = ShuffleValue(fullMask, a, vecpermute);
                    a = (ffttout ? max(a, b) : min(a, b));
                }
        
                for(long int step = stepout/2 ; step >= 1 ; step/=2){
                    const int vecinc = step;
                    const int fftt = (idxVal & step);
                    const int vecincout2 = (fftt? -vecinc  : vecinc);
                    const int vecpermute = (vecincout2+laneId);
                    const T b = ShuffleValue(fullMask, a, vecpermute);
                    a = (fftt ? max(a, b) : min(a, b));
                }
            }
            dataToSort[idxVal] = a;
        }
    }
    __syncthreads();

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Block_v3(dataToSort, nbValuesToSort)){
            return dataToSort;
        }
    }

    for(long int stepout = 32 ; stepout < nbWorkingValues ; stepout *= 2){  
        for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){ 
            int idxInStep = (idxVal % (stepout*2));
            int vecincout = 2 * (stepout - idxInStep) - 1;
            int vecpermute = idxVal + vecincout;  
            const T a = dataToSort[idxVal];
            const T b = dataToSort[vecpermute];
            const T best = (idxInStep >= stepout ? max(a, b) : min(a, b));
            buffer[idxVal] = best;
        }
        swap(dataToSort, buffer);
        __syncthreads();

        for(long int step = stepout/2 ; step >= 32 ; step/=2){
            for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){ 
                const int vecinc = step;
                const int fftt = (idxVal & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (idxVal + vecincout2);
                const T a = dataToSort[idxVal];
                const T b = dataToSort[vecpermute];
                const T best = (fftt ? max(a, b) : min(a, b));
                buffer[idxVal] = best;
            }
            swap(dataToSort, buffer);
            __syncthreads();
        }

        if constexpr (FullSM){
            T* dataToSortOrig = dataToSort;
            T* bufferOrig = buffer;
            for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){ 
                dataToSort = dataToSortOrig;
                buffer = bufferOrig;
                for(long int step = 16 ; step >= 1 ; step/=2){
                    const int vecinc = step;
                    const int fftt = (idxVal & step);
                    const int vecincout2 = (fftt? -vecinc  : vecinc);
                    const int vecpermute = (idxVal + vecincout2);
                    const T a = dataToSort[idxVal];
                    const T b = dataToSort[vecpermute];
                    const T best = (fftt ? max(a, b) : min(a, b));
                    buffer[idxVal] = best;
                    swap(dataToSort, buffer);
                    __syncwarp(0xffffffff);
                }
            }
            // The five warp-local tail layers also have odd buffer parity.
            if(threadIdx.x >= nbWorkingValues) swap(dataToSort, buffer);
        }
        else{
            for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){
                T a = dataToSort[idxVal];
                for(long int step = 16 ; step >= 1 ; step/=2){
                    const int vecinc = step;
                    const int fftt = (idxVal & step);
                    const int vecincout2 = (fftt? -vecinc  : vecinc);
                    const int vecpermute = (vecincout2+laneId);
                    const T b = ShuffleValue(fullMask, a, vecpermute);
                    a = (fftt ? max(a, b) : min(a, b));
                }
                dataToSort[idxVal] = a;
            }
        }
        __syncthreads();

        if constexpr (TestIfSorted){
            if(TestIfSorted_1Block_v3(dataToSort, nbValuesToSort)){
                return dataToSort;
            }
        }
    }

    return dataToSort;
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_gt32_d_extsm_v3(T* dataToSort, T* output, const int nbValuesToSort, 
                                            const int nbWorkingValues) {
    // Declare shared memory ptr
    extern __shared__ __align__(16) unsigned char smem[]; // untyped dynamic shared memory
    T* shared = reinterpret_cast<T*>(smem);

    // Copy
    for(int idxVal = threadIdx.x ; idxVal < nbValuesToSort ; idxVal += blockDim.x){
        shared[idxVal] = dataToSort[idxVal];
    }
    for(int idxVal = nbValuesToSort+threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){
        shared[idxVal] = my_numeric_limits_max<T>();
    }
    __syncthreads();
    // Sort
    T* buffer = shared + nbWorkingValues;
    T* lastArray = Core_Sort_1Block_SM_gt32_d_v3<T, TestIfSorted, FullSM>(shared, nbValuesToSort, buffer, nbWorkingValues);
    // Copy back
    for(int idxVal = threadIdx.x ; idxVal < nbValuesToSort ; idxVal += blockDim.x){
        output[idxVal] = lastArray[idxVal];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_v3(T* dataToSort, const int nbValuesToSort, const int nbWorkingValues) {
    Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, dataToSort, nbValuesToSort, nbWorkingValues);
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_v3_outofplace(T* dataToSort, T* output, const int nbValuesToSort, const int nbWorkingValues) {
    Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, output, nbValuesToSort, nbWorkingValues);
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d_extsm_v3(T* dataToSort, T* output, const int nbValuesToSort, 
                                                  const int nbWorkingValues) {
    // Declare shared memory ptr
    extern __shared__ __align__(16) unsigned char smem[]; // untyped dynamic shared memory
    T* shared_data = reinterpret_cast<T*>(smem);

    auto block = cooperative_groups::this_thread_block();

    // Cooperative Async Copy
    cooperative_groups::memcpy_async(block, shared_data, dataToSort, sizeof(T) * nbValuesToSort);
    for(int idxVal = nbValuesToSort+threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){
        shared_data[idxVal] = my_numeric_limits_max<T>();
    }

    cooperative_groups::wait(block); // Ensure data is fully copied

    // Sort
    T* buffer = shared_data + nbWorkingValues;
    T* lastArray = Core_Sort_1Block_SM_gt32_d_v3<T, TestIfSorted, FullSM>(shared_data, nbValuesToSort, buffer, nbWorkingValues);

    // Copy back
    cooperative_groups::memcpy_async(block, output, lastArray, sizeof(T) * nbValuesToSort);
    cooperative_groups::wait(block); // Ensure copy back is complete
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm_v3(T* dataToSort, const int nbValuesToSort, const int nbWorkingValues) {
    Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, dataToSort, nbValuesToSort, nbWorkingValues);
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm_v3_outofplace(T* dataToSort, T* output, const int nbValuesToSort, const int nbWorkingValues) {
    Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, output, nbValuesToSort, nbWorkingValues);
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


template <typename T>
__forceinline__ __device__ int binarySearch(const T pivots[], const int nbPivots, const T value) {
    int start = 0, end = nbPivots;
    while (start < end - 1) {
        int mid = start + (end - start) / 2;
        if (value < pivots[mid]) {
            end = mid;
        } else {
            start = mid;
        }
    }
    return start; 
}


template <typename T, int MaxParts>
__device__ void CountParts_Grid_core(const T* __restrict__ data, const int size, const T* __restrict__ pivots, int* __restrict__ counters,
                                                   int* __restrict__ localCounters, const int nbParts){
    assert(nbParts <= MaxParts);
    __shared__ T localPivots[MaxParts];

    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        localPivots[idx] = pivots[idx];
        localCounters[idx] = 0;
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;

    int currentBucket = 0;
    int currentCount = 0;
    
    for(int idxValue = id; idxValue < size; idxValue += numThreads){
        const T value = data[idxValue];

        // Binary search on pivots
        const int bucket = binarySearch(localPivots, nbParts, value);
        assert(0 <= bucket && bucket < nbParts);
        if(bucket != currentBucket){
            atomicAdd(&localCounters[currentBucket], currentCount);
            currentBucket = bucket;
            currentCount = 0;
        }
        currentCount += 1;
    }
    atomicAdd(&localCounters[currentBucket], currentCount);

    __syncthreads();
}


template <typename T, int MaxParts>
__global__ void CountParts_Grid(const T* __restrict__ data, const int size, const T* __restrict__ pivots, int* __restrict__ counters, const int nbParts){
    // Nb threads must be power of 2
    assert(((blockDim.x - 1) & blockDim.x) == 0);

    __shared__ int localCounters[MaxParts];

    CountParts_Grid_core<T, MaxParts>(data, size, pivots, counters, localCounters, nbParts);

    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        const int localCountersIdx = localCounters[idx];
        if(localCountersIdx){
            atomicAdd(&counters[idx], localCountersIdx);
        }
    }
    __syncthreads();
}

template <typename T, int MaxParts>
__global__ void MoveElements_Grid(T* __restrict__ data, const int size, const T* __restrict__ pivots, int* __restrict__ counters, T* __restrict__ dataBuffer, const int nbParts){
    // Nb threads must be power of 2
    assert(nbParts <= MaxParts);
    assert(((blockDim.x - 1) & blockDim.x) == 0);
    assert(blockDim.x <= DefaultNbThreadsPerBlock);

    __shared__ int localCounters[MaxParts];
    CountParts_Grid_core<T, MaxParts>(data, size, pivots, counters, localCounters, nbParts);

    __shared__ T localPivots[MaxParts];
    __shared__ int localGOffset[MaxParts];
    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        localPivots[idx] = pivots[idx];
        const int localCountersIdx = localCounters[idx];
        if(localCountersIdx){
            localGOffset[idx] = atomicAdd(&counters[idx], localCountersIdx);
        }
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;

    int bucket = -1;
    
    for(int idxValue = id; idxValue < size; idxValue += numThreads){
        const T value = data[idxValue];

        // Binary search on pivots
        const bool stillInBucket =
            bucket >= 0 && localPivots[bucket] <= value &&
            (bucket == nbParts-1 || value < localPivots[bucket+1]);
        if(!stillInBucket){
            bucket = binarySearch(localPivots, nbParts, value);
            assert(0 <= bucket && bucket < nbParts);
        }
        const int insertPosition = atomicAdd(&localGOffset[bucket], 1);
        dataBuffer[insertPosition] = value;
    }
}


// First pass of the persistent partitioner.  Each persistent block writes its
// complete histogram once; no global atomic operation is performed here.
template <typename T>
__host__ __device__ __forceinline__ bool EquivalentByOrderV3(
        const T& lhs, const T& rhs) {
    return !(lhs < rhs) && !(rhs < lhs);
}


// Performs the low-entropy pivot reduction entirely on the GPU.  Keeping the
// active count in device memory removes the first host synchronization of V2.
template <typename T, int MaxParts>
__global__ void AdaptPartitionCountDevice(
        T* pivots,
        const int candidateParts,
        int* activeNbParts) {
    if(blockIdx.x != 0 || threadIdx.x != 0) return;

    int distinctPivots = 0;
    for(int idx = 0 ; idx < candidateParts - 1 ; ++idx){
        if(idx == 0 || !EquivalentByOrderV3(pivots[idx-1], pivots[idx])){
            ++distinctPivots;
        }
    }

    int selectedParts = candidateParts;
    if(candidateParts > 32 && distinctPivots <= 8){
        for(int idx = 0 ; idx < 31 ; ++idx){
            const int source = (idx * (candidateParts - 2)) / 30;
            pivots[idx] = pivots[source];
        }
        pivots[31] = pivots[candidateParts-1];
        selectedParts = 32;
    }
    *activeNbParts = selectedParts;
}


template <typename T, int MaxParts>
__global__ void CountParts_Persistent(
        const T* __restrict__ data,
        const int size,
        const T* __restrict__ pivots,
        int* __restrict__ blockCounts,
        int* __restrict__ blockUniform,
        const int* __restrict__ activeNbParts) {
    __shared__ int nbParts;
    __shared__ int localCounters[MaxParts];
    __shared__ int localUniform[MaxParts];
    __shared__ T localPivots[MaxParts];

    if(threadIdx.x == 0) nbParts = *activeNbParts;
    __syncthreads();

    for(int idxPart = threadIdx.x ; idxPart < nbParts ; idxPart += blockDim.x){
        localCounters[idxPart] = 0;
        localUniform[idxPart] = 1;
        localPivots[idxPart] = pivots[idxPart];
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;
    int previousBucket = -1;
    int runCount = 0;

    for(int idxValue = id ; idxValue < size ; idxValue += numThreads){
        const T value = data[idxValue];
        const int bucket = binarySearch(localPivots, nbParts, value);

        if(bucket != previousBucket){
            if(previousBucket >= 0) atomicAdd(&localCounters[previousBucket], runCount);
            previousBucket = bucket;
            runCount = 0;
        }
        ++runCount;

        // This is conservative and exact: a bucket is copied only when every
        // value in it compares equivalent to its lower pivot.  Unlike V2 this
        // also proves that the final CONSTANT/BINARY bucket is mono-valued.
        if(!EquivalentByOrderV3(value, localPivots[bucket])){
            atomicExch(&localUniform[bucket], 0);
        }
    }
    if(previousBucket >= 0) atomicAdd(&localCounters[previousBucket], runCount);
    __syncthreads();

    for(int idxPart = threadIdx.x ; idxPart < nbParts ; idxPart += blockDim.x){
        const int matrixIdx = blockIdx.x * MaxParts + idxPart;
        blockCounts[matrixIdx] = localCounters[idxPart];
        blockUniform[matrixIdx] = localUniform[idxPart];
    }
}


// Reduces the persistent-block histograms into one count per partition.
template <int MaxParts, int ReductionThreads = 256>
__global__ void ReduceBlockCounts(
        const int* __restrict__ blockCounts,
        const int* __restrict__ blockUniform,
        int* __restrict__ partitionCounts,
        int* __restrict__ uniformParts,
        const int nbPersistentBlocks,
        const int* __restrict__ activeNbParts) {
    const int idxPart = blockIdx.x;
    const int nbParts = *activeNbParts;
    if(idxPart >= nbParts) return;

    int localSum = 0;
    int localUniform = 1;
    for(int idxBlock = threadIdx.x ; idxBlock < nbPersistentBlocks ;
        idxBlock += blockDim.x){
        const int matrixIdx = idxBlock * MaxParts + idxPart;
        localSum += blockCounts[matrixIdx];
        localUniform &= blockUniform[matrixIdx];
    }

    __shared__ int reduction[ReductionThreads];
    __shared__ int uniformReduction[ReductionThreads];
    reduction[threadIdx.x] = localSum;
    uniformReduction[threadIdx.x] = localUniform;
    __syncthreads();
    for(int step = blockDim.x / 2 ; step > 0 ; step /= 2){
        if(threadIdx.x < step){
            reduction[threadIdx.x] += reduction[threadIdx.x + step];
            uniformReduction[threadIdx.x] &= uniformReduction[threadIdx.x + step];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        partitionCounts[idxPart] = reduction[0];
        uniformParts[idxPart] = uniformReduction[0];
    }
}


// If the complete input belongs to one verified mono-value bucket, the input
// is already sorted.  The scatter kernel can then return immediately without
// writing the full array to the workspace and back.
__global__ void DetectSingleUniformBucket(
        const int* __restrict__ partitionCounts,
        const int* __restrict__ uniformParts,
        const int* __restrict__ activeNbParts,
        int* __restrict__ skipScatter) {
    if(blockIdx.x != 0 || threadIdx.x != 0) return;
    const int nbParts = *activeNbParts;
    int nonEmpty = 0;
    int allNonEmptyUniform = 1;
    for(int idx = 0 ; idx < nbParts ; ++idx){
        if(partitionCounts[idx] != 0){
            ++nonEmpty;
            allNonEmptyUniform &= uniformParts[idx];
        }
    }
    *skipScatter = (nonEmpty == 1 && allNonEmptyUniform) ? 1 : 0;
}


template <class ElementType, typename IndexType>
__device__ void BuildPrefix_SM_device(ElementType* prefix, IndexType N);


template <typename IndexType, int MaxParts>
__global__ void BuildPrefixDynamic(
        IndexType* prefixTree,
        const int* __restrict__ activeNbParts) {
    __shared__ IndexType buffer[MaxParts];
    const int nbParts = *activeNbParts;

    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        buffer[idx] = prefixTree[idx];
    }
    __syncthreads();
    BuildPrefix_SM_device(buffer, nbParts);
    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        prefixTree[idx] = buffer[idx];
    }
}


// Converts counts into the first output position owned by every
// (persistent block, partition) pair. partitionBoundaries contains
// [0, end(partition 0), ..., end(partition P-1)].
template <int MaxParts>
__global__ void BuildPersistentBlockOffsets(
        int* __restrict__ blockOffsets,
        const int* __restrict__ partitionBoundaries,
        const int nbPersistentBlocks,
        const int* __restrict__ activeNbParts) {
    const int idxPart = blockIdx.x;
    const int nbParts = *activeNbParts;
    if(idxPart >= nbParts || threadIdx.x != 0) return;

    int offset = partitionBoundaries[idxPart];
    for(int idxBlock = 0 ; idxBlock < nbPersistentBlocks ; ++idxBlock){
        const int index = idxBlock * MaxParts + idxPart;
        const int count = blockOffsets[index];
        blockOffsets[index] = offset;
        offset += count;
    }
}


// Second and final classification pass.  The precomputed per-block offsets
// remove both the recount and all global atomics from the scatter.
template <typename T, int MaxParts>
__global__ void MoveElements_Persistent(
        const T* __restrict__ data,
        const int size,
        const T* __restrict__ pivots,
        const int* __restrict__ blockOffsets,
        T* __restrict__ dataBuffer,
        const int* __restrict__ activeNbParts,
        const int* __restrict__ skipScatter) {
    __shared__ T localPivots[MaxParts];
    __shared__ int localOffsets[MaxParts];
    __shared__ int nbParts;

    if(threadIdx.x == 0) nbParts = *activeNbParts;
    __syncthreads();
    if(*skipScatter) return;

    for(int idxPart = threadIdx.x ; idxPart < nbParts ;
        idxPart += blockDim.x){
        localPivots[idxPart] = pivots[idxPart];
        localOffsets[idxPart] = blockOffsets[blockIdx.x * MaxParts + idxPart];
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;
    int bucket = -1;

    for(int idxValue = id ; idxValue < size ; idxValue += numThreads){
        const T value = data[idxValue];
        const bool stillInBucket =
            bucket >= 0 && localPivots[bucket] <= value &&
            (bucket == nbParts-1 || value < localPivots[bucket+1]);
        if(!stillInBucket){
            bucket = binarySearch(localPivots, nbParts, value);
        }
        if(nbParts <= 32){
            // Low-entropy inputs made every lane contend on one or two shared
            // counters in V2.  Aggregate equal buckets within a warp so that
            // CONSTANT needs one atomic per warp instead of one per value.
            const unsigned active = __activemask();
            unsigned remaining = active;
            const int lane = threadIdx.x & 31;
            const unsigned laneBit = 1u << lane;
            int outputPosition = 0;

            while(remaining){
                const int leader = __ffs(remaining) - 1;
                const int leaderBucket = __shfl_sync(active, bucket, leader);
                const unsigned group = __ballot_sync(active, bucket == leaderBucket);
                if(group & laneBit){
                    int base = 0;
                    if(lane == leader){
                        base = atomicAdd(&localOffsets[leaderBucket], __popc(group));
                    }
                    base = __shfl_sync(group, base, leader);
                    const unsigned lanesBefore = group & (laneBit - 1u);
                    outputPosition = base + __popc(lanesBefore);
                }
                remaining &= ~group;
            }
            dataBuffer[outputPosition] = value;
        }
        else{
            const int outputPosition = atomicAdd(&localOffsets[bucket], 1);
            dataBuffer[outputPosition] = value;
        }
    }
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


template <class ElementType, typename IndexType>
__device__ void BuildPrefix_SM_device(ElementType* prefix, IndexType N) {
    // Upward pass
    IndexType js = 2;
    while (js <= N) {
        IndexType jsd2 = js / 2;
        for (IndexType idN = threadIdx.x*js + js - 1; idN < N; idN += blockDim.x*js) {
            prefix[idN] += prefix[idN - jsd2];
        }
        js *= 2;
        __syncthreads();
    }

    // Downward pass
    js = max(4, js/2);
    while (js > 1) {
        IndexType jsd2 = js / 2;
        for (IndexType idN = threadIdx.x*js + js + jsd2 - 1; idN < N; idN += blockDim.x*js) {
            prefix[idN] += prefix[idN - jsd2];
        }
        js = jsd2;
        __syncthreads();
    }
}

template <typename IndexType, const int MaxNbParts>
__global__ void BuildPrefix_SM( IndexType* prefixTree, const IndexType inSize){
    __shared__ IndexType buffer[MaxNbParts];

    for(IndexType idx = threadIdx.x ; idx < inSize ; idx += blockDim.x){
        buffer[idx] = prefixTree[idx];
    }
    __syncthreads();

    BuildPrefix_SM_device(buffer, inSize);

    for(IndexType idx = threadIdx.x ; idx < inSize ; idx += blockDim.x){
        prefixTree[idx] = buffer[idx];
    }
    __syncthreads();
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T, int MaxParts, int kOver, bool TestIfSorted, bool FullSM>
__global__ void Sort_core_buildPivots(T* __restrict__ dataToSort, const int nbValuesToSort, T* __restrict__ pivots, const int nbParts){
    // We copy values from the array to build the partition
    assert(gridDim.x == 1);
    assert(nbParts <= MaxParts);
    assert(4 <= nbParts);
    assert(32 <= blockDim.x);
    assert(nbValuesToSort > 0);
    static_assert(MaxParts > 0 && ((MaxParts - 1) & MaxParts) == 0, "MaxParts must be a power of 2");

    const int nbSamples = nbParts * kOver;

    __shared__ T localPivots[MaxParts * kOver];
    
    if(threadIdx.x < 1){
        localPivots[threadIdx.x] = my_numeric_limits_min<T>();
    } 
    for(int idx = threadIdx.x + 1 ; idx < nbSamples ; idx += blockDim.x){
        // Deterministic hash sampling avoids pathological periodic inputs while
        // remaining reproducible. The multiply-high mapping avoids a costly
        // integer division and maps the 32-bit hash to [0, nbValuesToSort).
        uint32_t hash = static_cast<uint32_t>(idx) * 747796405u + 2891336453u;
        hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
        hash = (hash >> 22u) ^ hash;
        const int sampleIndex = static_cast<int>(
            (static_cast<uint64_t>(hash) *
             static_cast<uint64_t>(nbValuesToSort)) >> 32u);
        localPivots[idx] = dataToSort[sampleIndex];
    }
    __syncthreads();

    if(nbSamples <= 32){
        Core_Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(localPivots, nbSamples);
        // There is a syncthread in the function
    }
    else if(nbSamples <= 512){
        Core_Sort_1Block_SM_gt32_d<T, TestIfSorted, FullSM>(localPivots, nbSamples);
        // There is a syncthread in the function
    }
    else {
        __shared__ T buffer[MaxParts * kOver];
        Core_Sort_1Block_SM_gt32_d_v3<T, TestIfSorted, FullSM>(localPivots, nbSamples,
                                                            buffer, MaxParts * kOver);
        // There is a syncthread in the function
    }
    
    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        pivots[idx] = localPivots[idx * kOver];
    }
    __syncthreads();
    if(threadIdx.x == 0){
        pivots[nbParts-1] = getNextVal(localPivots[nbSamples-1]);
    }
    __syncthreads();

}

template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void SortPartitions(T* __restrict__ dataToSort, const int nbValuesToSort, const int* __restrict__ counterParts, const int nbParts){
    for(int idxPart = blockIdx.x ; idxPart < nbParts; idxPart += gridDim.x){
        const int start = counterParts[idxPart];
        const int end = counterParts[idxPart+1];
        const int size = end-start;
        if(size <= 32){
            Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort+start, dataToSort+start, size);
            __syncthreads();
        }
        else if(size <= ComputeDefaultMaxBitoSort<T>()){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= ComputeDefaultMaxBitoSort<T>());
            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, dataToSort+start, size, NOrPow2);
            }
            else{
                Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, dataToSort+start, size, NOrPow2);
            }
        }
    }
}

template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void SortPartitions_outofplace(T* __restrict__ dataToSort, T* __restrict__ dataToSortOut, const int nbValuesToSort, const int* __restrict__ counterParts, const int nbParts){
    for(int idxPart = blockIdx.x ; idxPart < nbParts; idxPart += gridDim.x){
        const int start = counterParts[idxPart];
        const int end = counterParts[idxPart+1];
        const int size = end-start;
        if(size <= 32){
            Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort+start, dataToSortOut+start, size);
            __syncthreads();
        }
        else if(size <= ComputeDefaultMaxBitoSort<T>()){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= ComputeDefaultMaxBitoSort<T>());
            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, dataToSortOut+start, size, NOrPow2);
            }
            else{
                Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, dataToSortOut+start, size, NOrPow2);
            }
        }
    }
}

template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void SortOnePartition(T* dataToSort, const int size){
    assert(size <= ComputeDefaultMaxBitoSort<T>());
    if(size <= 32){
        Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort, dataToSort, size);
        __syncthreads();
    }
    else if(size <= ComputeDefaultMaxBitoSort<T>()){
        const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
        assert(NOrPow2 <= ComputeDefaultMaxBitoSort<T>());
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, dataToSort, size, NOrPow2);
        }
        else{
            Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, dataToSort, size, NOrPow2);
        }
    }
}



template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts, int MaxNbParts>
__host__ void SortCore(T* dataToSort, const int nbValuesToSort, cudaStream_t stream, T* dataBuffer, bool storedInRightPlaced){
    if(nbValuesToSort <= 1){
        if(storedInRightPlaced == false){
            CUDA_CHECK(cudaMemcpyAsync(dataBuffer, dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }

    if(nbValuesToSort <= 32){
        if(storedInRightPlaced == false){
            Sort_1Warp_SM_le32_outofplace<T, TestIfSorted, FullSM><<<1,32,0,stream>>>(dataToSort, dataBuffer, nbValuesToSort);
        }
        else{
            Sort_1Warp_SM_le32<T, TestIfSorted, FullSM><<<1,32,0,stream>>>(dataToSort, nbValuesToSort);
        }
        return;
    }

    int MaxThreadsPerBlock = DefaultNbThreadsPerBlock;
    // TODO CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if(NOrPow2 <= DefaultMaxBitoSortV2){
        const int nbThreads = std::min(NOrPow2, MaxThreadsPerBlock);
        if constexpr(ASyncCpyBito){
            if(storedInRightPlaced == false){
                Sort_1Block_async_gt32_extsm_v2_outofplace<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, dataBuffer, nbValuesToSort);
            }
            else{
                Sort_1Block_async_gt32_extsm_v2<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort);
            }
        }
        else{
            if(storedInRightPlaced == false){
                Sort_1Block_gt32_extsm_v2_outofplace<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, dataBuffer, nbValuesToSort);
            }
            else{
                Sort_1Block_gt32_extsm_v2<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort);
            }
        }
        return;
    }

    if(NOrPow2 <= ComputeDefaultMaxBitoSort<T>()){
        const int nbThreads = std::min(NOrPow2, MaxThreadsPerBlock);
        if constexpr(ASyncCpyBito){
            if(storedInRightPlaced == false){
                Sort_1Block_async_gt32_extsm_v3_outofplace<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, dataBuffer, nbValuesToSort, NOrPow2);
            }
            else{
                Sort_1Block_async_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
            }
        }
        else{
            if(storedInRightPlaced == false){
                Sort_1Block_gt32_extsm_v3_outofplace<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, dataBuffer, nbValuesToSort, NOrPow2);
            }
            else{
                Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
            }
        }
        return;
    }

    const int NbParts = std::min(MaxNbParts, std::max(MinNbParts, nbValuesToSort/MaxThreadsPerBlock));
    const int NbPartsPow2 = GreaterPowerOf2EqualOrLarger(NbParts);

    const bool allocatedDataBuffer = (dataBuffer == nullptr);
    if(allocatedDataBuffer){
        CUDA_CHECK(cudaMallocAsync(&dataBuffer, nbValuesToSort * sizeof(T), stream));
    }

    T* pivots = nullptr;
    CUDA_CHECK(cudaMallocAsync(&pivots, NbParts * sizeof(T), stream));
    CUDA_CHECK(cudaMemsetAsync(pivots, 0, NbParts * sizeof(T), stream));

    const int NbPartsPow2Or32 = std::max(32, NbPartsPow2);
    const int kOver = 2;
    if(NbPartsPow2Or32 * kOver <= MaxThreadsPerBlock)
        Sort_core_buildPivots<T, MaxNbParts, kOver, TestIfSorted, FullSM><<<1, NbPartsPow2Or32*kOver, 0, stream>>>(dataToSort, nbValuesToSort, pivots, NbParts);
    else
        Sort_core_buildPivots<T, MaxNbParts, 1, TestIfSorted, FullSM><<<1, MaxThreadsPerBlock, 0, stream>>>(dataToSort, nbValuesToSort, pivots, NbParts);

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMallocAsync(&counterParts, (NbParts+2) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(counterParts, 0, (NbParts+2) * sizeof(int), stream));

    const int nbThreadBlock = (nbValuesToSort+MaxThreadsPerBlock-1)/MaxThreadsPerBlock;

    CountParts_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock, 0, stream >>>(dataToSort, nbValuesToSort, pivots, counterParts+2, NbParts);

    BuildPrefix_SM<int, MaxNbParts><<<1,NbParts,0,stream>>>(counterParts+2, NbParts);

    MoveElements_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock,0,stream >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts+1,
                                            dataBuffer, NbParts);

    T pivotsHost[NbParts];
    CUDA_CHECK(cudaMemcpyAsync(pivotsHost, pivots, NbParts * sizeof(T), cudaMemcpyDeviceToHost, stream));

    // Not needed anymore : CUDA_CHECK(cudaMemcpyAsync(dataToSort, dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFreeAsync(pivots, stream));

    int counterPartsHost[NbParts+1];
    CUDA_CHECK(cudaMemcpyAsync(counterPartsHost, counterParts, (NbParts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // We need counterPartsHost and that sortPartitions is over
    CUDA_CHECK(cudaStreamSynchronize(stream));  

    int maxSizePartToSort = 0;
    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if(size > maxSizePartToSort && size <= ComputeDefaultMaxBitoSort<T>()){
            maxSizePartToSort = size;
        }
    }
    if(maxSizePartToSort){
        const int maxSizePartToSortPow2 = GreaterPowerOf2EqualOrLarger(maxSizePartToSort);
        if(storedInRightPlaced == true){
            SortPartitions_outofplace<T, TestIfSorted, FullSM, ASyncCpyBito><<< NbParts, MaxThreadsPerBlock, maxSizePartToSortPow2*2*sizeof(T),stream>>>(dataBuffer, dataToSort, nbValuesToSort, counterParts, NbParts);
        }
        else{
            SortPartitions<T, TestIfSorted, FullSM, ASyncCpyBito><<< NbParts, MaxThreadsPerBlock, maxSizePartToSortPow2*2*sizeof(T),stream>>>(dataBuffer, nbValuesToSort, counterParts, NbParts);
        }
    }

    CUDA_CHECK(cudaFreeAsync(counterParts, stream));  

    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if(size > ComputeDefaultMaxBitoSort<T>()){
            if((idxPart == NbParts-1 || getNextVal(pivotsHost[idxPart]) < pivotsHost[idxPart+1])){
                SortCore<T, TestIfSorted, FullSM, ASyncCpyBito, MinNbParts, MaxNbParts>(dataBuffer+start, size, stream, dataToSort+start, !storedInRightPlaced);
            }
            else if(storedInRightPlaced){
                CUDA_CHECK(cudaMemcpyAsync(dataToSort+start, dataBuffer+start, size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    if(allocatedDataBuffer){
        CUDA_CHECK(cudaFreeAsync(dataBuffer, stream));
    }
}


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts = 32, int MaxNbParts = 1024>
__host__ void Sort(T* dataToSort, const int nbValuesToSort, cudaStream_t stream = 0){
    SortCore<T, TestIfSorted, FullSM, ASyncCpyBito, MinNbParts, MaxNbParts>(dataToSort, nbValuesToSort, stream, nullptr, true);
}

/////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ __forceinline__ int mergePathDiag(
    const T* A, int aCount,
    const T* B, int bCount,
    int diag
){
    // i ∈ [max(0, diag-bCount), min(diag, aCount)]
    int lo = max(0, diag - bCount);
    int hi = min(diag, aCount);

    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int j   = diag - mid;

        // On avance i si A[mid] est "trop petit" vs B[j-1]
        // (exactement ton test)
        if (mid < aCount && j > 0 && A[mid] < B[j - 1])
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo; // i, avec j = diag - i
}


template <typename T, int nbMergeValuesPerBlock>
__global__ void mergeKernel(
    const T* __restrict__ partValues,
    T*       __restrict__ mergeValues,
    int partSize,
    int totalNbValues
) {
    constexpr int TILE = nbMergeValuesPerBlock;

    const int nbMergeValues    = 2 * partSize;
    const int blocksPerMerge   = (nbMergeValues + TILE - 1) / TILE;

    const int mergeIdx         = blockIdx.x / blocksPerMerge;
    const int blockInMerge     = blockIdx.x - mergeIdx * blocksPerMerge;
    const int mergeStart       = mergeIdx * nbMergeValues;
    if (mergeStart >= totalNbValues) return;

    int sizeA = min(partSize, totalNbValues - mergeStart);
    int sizeB = min(partSize, max(0, totalNbValues - mergeStart - sizeA));
    int totalL = sizeA + sizeB;

    int D0_chunk = blockInMerge * TILE;
    if (D0_chunk >= totalL) return;
    int D1_chunk = min(D0_chunk + TILE, totalL);

    const T* runA = partValues + mergeStart;
    const T* runB = partValues + mergeStart + sizeA;
    T* out        = mergeValues + mergeStart + D0_chunk;

    __shared__ int a0_s, b0_s, a1_s, b1_s;

    if (threadIdx.x == 0) {
        int a0 = mergePathDiag(runA, sizeA, runB, sizeB, D0_chunk);
        int a1 = mergePathDiag(runA, sizeA, runB, sizeB, D1_chunk);
        a0_s = a0; b0_s = D0_chunk - a0;
        a1_s = a1; b1_s = D1_chunk - a1;
    }
    __syncthreads();

    const int a0 = a0_s, b0 = b0_s;
    const int a1 = a1_s, b1 = b1_s;

    const int localA = a1 - a0;
    const int localB = b1 - b0;
    const int localL = localA + localB; // == D1_chunk - D0_chunk <= TILE

    // A and B together contain at most TILE values.  Keeping two TILE-sized
    // arrays needlessly reduced occupancy (especially for double).
    __shared__ T shInputs[TILE];
    T* shA = shInputs;
    T* shB = shInputs + localA;

    for (int i = threadIdx.x; i < localA; i += blockDim.x) shA[i] = runA[a0 + i];
    for (int i = threadIdx.x; i < localB; i += blockDim.x) shB[i] = runB[b0 + i];
    __syncthreads();

    // Partition de [0..localL) entre threads
    int t0 = (threadIdx.x * localL) / blockDim.x;
    int t1 = ((threadIdx.x + 1) * localL) / blockDim.x;

    int iA = mergePathDiag(shA, localA, shB, localB, t0);
    int jB = t0 - iA;

    // Merge séquentiel du segment et STORE DIRECT en global
    // (pas de bufM, pas de sync finale)
    __shared__ T sout[TILE];
    // Petite optimisation : garde les valeurs en registres
    for (int k = t0; k < t1; ++k) {
        bool takeA = (jB >= localB) || (iA < localA && shA[iA] <= shB[jB]);
        sout[k] = takeA ? shA[iA++] : shB[jB++];
    }

    __syncthreads();
    for (int i = threadIdx.x; i < localL; i += blockDim.x) out[i] = sout[i];
}



template <typename T, const int MaxValuesToSort, bool TestIfSorted, bool FullSM>
__global__ void SortChunks(T* dataToSort, const int nbValuesToSort){
    const int nbChunks = (nbValuesToSort + MaxValuesToSort - 1) / MaxValuesToSort;
    for(int idxChunk = blockIdx.x ; idxChunk < nbChunks; idxChunk += gridDim.x){
        const int start = idxChunk*MaxValuesToSort;
        const int end = min(start+MaxValuesToSort, nbValuesToSort);
        const int size = end-start;
        if(size <= 32){
            Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort+start, dataToSort+start, size);
        }
        else if(size <= DefaultMaxBitoSortV2){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= DefaultMaxBitoSortV2);
            if constexpr (sizeof(T) <= 4) {
                Sort_1Block_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(
                    dataToSort+start, dataToSort+start, size);
            }
            else {
                Sort_1Block_gt32_d_extsm<T, TestIfSorted, FullSM>(
                    dataToSort+start, size);
            }
        }
        else if(size <= MaxValuesToSort){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= MaxValuesToSort);
            Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, dataToSort+start, size, NOrPow2);
        }
    }
}


template <typename T, int MaxValuesToSort = ComputeDefaultMaxBitoSort<T>()/2, bool TestIfSorted = false, bool FullSM = false>
__host__ void SortMerge(
    T* dataToSort,
    const int nbValuesToSort,
    const int nbThreadsPerBlockForMerge = std::min(DefaultNbThreadsPerBlock, MaxValuesToSort/16),
    cudaStream_t stream = 0,
    T* externalBuffer = nullptr,
    size_t externalBufferElems = 0,
    bool storedInRightPlaced = true
){
    // MaxValuesToSort must be power of 2
    static_assert(((MaxValuesToSort - 1) & MaxValuesToSort) == 0, "MaxValuesToSort must be a power of 2");

    auto ceil_div = [](int a, int b) -> int { return (a + b - 1) / b; };

    // When the caller requests an out-of-place result, externalBuffer is the
    // destination, not merely optional scratch storage.
    if (!storedInRightPlaced) {
        assert(externalBuffer != nullptr);
        assert(externalBufferElems >= static_cast<size_t>(nbValuesToSort));
    }

    if (nbValuesToSort <= 1){
        if(storedInRightPlaced == false){
            assert(externalBuffer != nullptr);
            CUDA_CHECK(cudaMemcpyAsync(externalBuffer, dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }

    // Petits cas
    if (nbValuesToSort <= 32) {
        if(storedInRightPlaced == false){
            Sort_1Warp_SM_le32_outofplace<T, TestIfSorted, FullSM>
                <<<1, 32, 0, stream>>>(dataToSort, externalBuffer, nbValuesToSort);
        }
        else{
            Sort_1Warp_SM_le32<T, TestIfSorted, FullSM>
                <<<1, 32, 0, stream>>>(dataToSort, nbValuesToSort);
        }
        return;
    }

    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if (NOrPow2 <= DefaultMaxBitoSortV2) {
        const int nbThreads = NOrPow2;
        if constexpr (sizeof(T) <= 4) {
            if(storedInRightPlaced == false){
                Sort_1Block_gt32_extsm_v2_outofplace<T, TestIfSorted, FullSM>
                    <<<1, nbThreads, 2 * NOrPow2 * sizeof(T), stream>>>(
                        dataToSort, externalBuffer, nbValuesToSort);
            }
            else{
                Sort_1Block_gt32_extsm_v2<T, TestIfSorted, FullSM>
                    <<<1, nbThreads, 2 * NOrPow2 * sizeof(T), stream>>>(
                        dataToSort, nbValuesToSort);
            }
        }
        else {
            if(storedInRightPlaced == false){
                Sort_1Block_gt32_extsm_outofplace<T, TestIfSorted, FullSM>
                    <<<1, nbThreads, NOrPow2 * sizeof(T), stream>>>(
                        dataToSort, externalBuffer, nbValuesToSort);
            }
            else{
                Sort_1Block_gt32_extsm<T, TestIfSorted, FullSM>
                    <<<1, nbThreads, NOrPow2 * sizeof(T), stream>>>(
                        dataToSort, nbValuesToSort);
            }
        }
        return;
    }

    if (NOrPow2 <= ComputeDefaultMaxBitoSort<T>()) {
        const int nbThreads = std::min(NOrPow2, DefaultNbThreadsPerBlock);
        if(storedInRightPlaced == false){
            Sort_1Block_gt32_extsm_v3_outofplace<T, TestIfSorted, FullSM>
                <<<1, nbThreads, 2 * NOrPow2 * sizeof(T), stream>>>(dataToSort, externalBuffer, nbValuesToSort, NOrPow2);
        }
        else{
            Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM>
                <<<1, nbThreads, 2 * NOrPow2 * sizeof(T), stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        }
        return;
    }

    // 1) Sort chunks (in-place)
    {
        const int NbBlocks = ceil_div(nbValuesToSort, MaxValuesToSort);
        SortChunks<T, MaxValuesToSort, TestIfSorted, FullSM>
            <<<NbBlocks, DefaultNbThreadsPerBlock, MaxValuesToSort * 2 * sizeof(T), stream>>>(
                dataToSort, nbValuesToSort
            );
    }

    // 2) Buffer (réutilisable si fourni)
    T*  dataBuffer  = externalBuffer;
    bool ownsBuffer = false;
    const size_t needElems = static_cast<size_t>(nbValuesToSort);

    if (!dataBuffer || externalBufferElems < needElems) {
        CUDA_CHECK(cudaMallocAsync(&dataBuffer, needElems * sizeof(T), stream));
        ownsBuffer = true;
    }

    T* in  = dataToSort;
    T* out = dataBuffer;

    for (int run_size = MaxValuesToSort;
         run_size < nbValuesToSort;
         run_size *= 2)
    {
        constexpr int TILE = MaxValuesToSort;

        const int numMerges      = ceil_div(nbValuesToSort, 2 * run_size);
        const int blocksPerMerge = ceil_div(2 * run_size, TILE);
        const int gridX          = numMerges * blocksPerMerge;

        mergeKernel<T, TILE>
            <<<gridX, nbThreadsPerBlockForMerge, 0, stream>>>(
                in, out, run_size, nbValuesToSort
            );

        std::swap(in, out);
    }

    T* requestedOutput = storedInRightPlaced ? dataToSort : externalBuffer;
    if (in != requestedOutput) {
        CUDA_CHECK(cudaMemcpyAsync(
            requestedOutput, in,
            nbValuesToSort * sizeof(T),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }

    if (ownsBuffer) {
        CUDA_CHECK(cudaFreeAsync(dataBuffer, stream));
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Sorts every small partition in one launch.  Partitions whose pivot interval
// can contain only one representable value are copied directly, including
// large constant buckets created by CONSTANT/BINARY inputs.
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void SortPartitionsAdaptive_outofplace(
        const T* __restrict__ input,
        T* __restrict__ output,
        const int* __restrict__ counterParts,
        const int* __restrict__ uniformParts,
        const int nbParts) {
    for(int idxPart = blockIdx.x ; idxPart < nbParts ; idxPart += gridDim.x){
        const int start = counterParts[idxPart];
        const int end = counterParts[idxPart+1];
        const int size = end-start;

        if(size == 0) continue;

        const bool singleValueRange = uniformParts[idxPart] != 0;

        if(singleValueRange){
            for(int idx = threadIdx.x ; idx < size ; idx += blockDim.x){
                output[start+idx] = input[start+idx];
            }
            __syncthreads();
        }
        else if(size <= 32){
            Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(
                const_cast<T*>(input)+start, output+start, size);
            __syncthreads();
        }
        else if(size <= DefaultMaxBitoSortV2){
            if constexpr (sizeof(T) <= 4) {
                Sort_1Block_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(
                    const_cast<T*>(input)+start, output+start, size);
            }
            else {
                Sort_1Block_gt32_d_extsm_outofplace<T, TestIfSorted, FullSM>(
                    input+start, output+start, size);
            }
        }
        else if(size <= ComputeDefaultMaxBitoSort<T>()){
            const int workingValues = GreaterPowerOf2EqualOrLarger(size);
            Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(
                const_cast<T*>(input)+start, output+start, size, workingValues);
        }
    }
}


template <typename T, int MaxValuesToSort = 2048, int MaxNbParts = 512, bool TestIfSorted = false, bool FullSM = false>
__host__ void HybridSortLegacy(T* dataToSort, const int nbValuesToSort, cudaStream_t stream = 0, T* dataBuffer = nullptr){
    static_assert(((MaxValuesToSort - 1) & MaxValuesToSort) == 0, "MaxValuesToSort must be a power of 2");

    constexpr int MaxThreadsPerBlock = DefaultNbThreadsPerBlock;
    constexpr int MergeThreadsPerBlock =
        std::min(DefaultNbThreadsPerBlock, MaxValuesToSort/16);

    // If the array is small enough, we use SortMerge
    if (nbValuesToSort <= MaxValuesToSort * 16) {
        SortMerge<T, MaxValuesToSort, TestIfSorted, FullSM>(dataToSort, nbValuesToSort, MergeThreadsPerBlock, stream, nullptr, 0, true);
        return;
    }

    // Otherwise, we multipartition it and then sort each partition with merge sort
    const int NbParts = std::min(MaxNbParts, (nbValuesToSort/MaxValuesToSort)+1);
    const int NbPartsPow2 = GreaterPowerOf2EqualOrLarger(NbParts);

    const bool allocatedDataBuffer = (dataBuffer == nullptr);
    if(allocatedDataBuffer){
        CUDA_CHECK(cudaMallocAsync(&dataBuffer, nbValuesToSort * sizeof(T), stream));
    }

    T* pivots = nullptr;
    CUDA_CHECK(cudaMallocAsync(&pivots, NbParts * sizeof(T), stream));
    CUDA_CHECK(cudaMemsetAsync(pivots, 0, NbParts * sizeof(T), stream));

    const int NbPartsPow2Or32 = std::max(32, NbPartsPow2);
    const int kOver = 2;
    if(NbPartsPow2Or32*kOver <= MaxThreadsPerBlock)
        Sort_core_buildPivots<T, MaxNbParts, kOver, TestIfSorted, FullSM><<<1, NbPartsPow2Or32*kOver, 0, stream>>>(dataToSort, nbValuesToSort, pivots, NbParts);
    else
        Sort_core_buildPivots<T, MaxNbParts, 1, TestIfSorted, FullSM><<<1, MaxThreadsPerBlock, 0, stream>>>(dataToSort, nbValuesToSort, pivots, NbParts);

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMallocAsync(&counterParts, (NbParts+2) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(counterParts, 0, (NbParts+2) * sizeof(int), stream));

    const int nbThreadBlock = (nbValuesToSort+MaxThreadsPerBlock-1)/MaxThreadsPerBlock;

    CountParts_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock, 0, stream >>>(dataToSort, nbValuesToSort, pivots, counterParts+2, NbParts);

    BuildPrefix_SM<int, MaxNbParts><<<1,NbParts,0,stream>>>(counterParts+2, NbParts);

    MoveElements_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock,0,stream >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts+1,
                                            dataBuffer, NbParts);

    CUDA_CHECK(cudaFreeAsync(pivots, stream));

    std::vector<int> counterPartsHost(NbParts+1);
    CUDA_CHECK(cudaMemcpyAsync(counterPartsHost.data(), counterParts,
                               (NbParts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    // The host needs partition boundaries. This remains the principal
    // synchronization point of the sample-sort path.
    CUDA_CHECK(cudaStreamSynchronize(stream));  

    int maxSmallPartition = 0;
    std::vector<int> largePartitions;
    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if (size <= ComputeDefaultMaxBitoSort<T>()) {
            maxSmallPartition = std::max(maxSmallPartition, size);
        }
        else {
            largePartitions.push_back(idxPart);
        }
    }

    // This replaces up to NbParts independent host-side SortMerge calls by a
    // single grid launch. With the chosen number of buckets, this is normally
    // the common path.
    if (maxSmallPartition > 0) {
        const int workingValues = GreaterPowerOf2EqualOrLarger(maxSmallPartition);
        SortPartitions_outofplace<T, TestIfSorted, FullSM, false>
            <<<NbParts, MaxThreadsPerBlock,
               2 * workingValues * sizeof(T), stream>>>(
                dataBuffer, dataToSort, nbValuesToSort, counterParts, NbParts);
    }

    // Only genuinely oversized buckets need merge sort. A small stream pool
    // lets those independent ranges overlap. Events reconnect all work to the
    // caller's stream, so neither the scratch buffer nor the output can race.
    const int nbSortStreams = std::min<int>(8, largePartitions.size());
    std::vector<cudaStream_t> asyncSorts(nbSortStreams);
    std::vector<cudaEvent_t> sortDone(nbSortStreams);
    for(int idxStream = 0 ; idxStream < nbSortStreams ; ++idxStream){
        CUDA_CHECK(cudaStreamCreateWithFlags(&asyncSorts[idxStream], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&sortDone[idxStream], cudaEventDisableTiming));
    }

    for(size_t idxLarge = 0 ; idxLarge < largePartitions.size() ; ++idxLarge){
        const int idxPart = largePartitions[idxLarge];
        const int start = counterPartsHost[idxPart];
        const int size = counterPartsHost[idxPart+1] - start;
        const int idxStream = static_cast<int>(idxLarge % nbSortStreams);
        SortMerge<T, MaxValuesToSort, TestIfSorted, FullSM>(
            dataBuffer+start, size, MergeThreadsPerBlock,
            asyncSorts[idxStream], dataToSort+start, size, false);
    }

    for(int idxStream = 0 ; idxStream < nbSortStreams ; ++idxStream){
        CUDA_CHECK(cudaEventRecord(sortDone[idxStream], asyncSorts[idxStream]));
        CUDA_CHECK(cudaStreamWaitEvent(stream, sortDone[idxStream], 0));
        CUDA_CHECK(cudaStreamDestroy(asyncSorts[idxStream]));
        CUDA_CHECK(cudaEventDestroy(sortDone[idxStream]));
    }

    CUDA_CHECK(cudaFreeAsync(counterParts, stream));

    if(allocatedDataBuffer){
        CUDA_CHECK(cudaFreeAsync(dataBuffer, stream));
    }
}


// The supplied measurements show no partitioning crossover for 32-bit values
// up to 2^28 elements.  Eight-byte structures cross over only around 2^27.
// This trait is deliberately specializable for another GPU or value type.
template <typename T>
struct CuSortTuning {
    static constexpr int PartitionThreshold =
        sizeof(T) <= 4 ? std::numeric_limits<int>::max() : (1 << 27);
};


// Explicit partitioned path.  It remains public so that the research variant
// can be benchmarked independently of the data-driven automatic dispatcher.
template <typename T,
          int MaxValuesToSort = 2048,
          int MaxNbParts = 1024,
          bool TestIfSorted = false,
          bool FullSM = true>
__host__ void HybridSortPartitionedV3(
        T* dataToSort,
        const int nbValuesToSort,
        CuSortWorkspace<T, MaxNbParts>& workspace,
        cudaStream_t stream = 0) {
    static_assert(((MaxValuesToSort - 1) & MaxValuesToSort) == 0,
                  "MaxValuesToSort must be a power of 2");
    if(nbValuesToSort <= 1) return;

    workspace.reserve(static_cast<size_t>(nbValuesToSort));

    // A workspace may be reused from a different caller stream.  These waits
    // prevent the next partition pass from overwriting ranges still consumed
    // by a worker from the previous invocation.
    if(workspace.workerEventsRecorded()){
        for(int idx = 0 ; idx < workspace.nbSortStreams() ; ++idx){
            CUDA_CHECK(cudaStreamWaitEvent(stream, workspace.sortDone(idx), 0));
        }
    }

    constexpr int MergeThreadsPerBlock =
        std::min(DefaultNbThreadsPerBlock, MaxValuesToSort/16);

    const int candidateParts = std::min(
        MaxNbParts,
        std::max(32, 1 + (nbValuesToSort - 1) / MaxValuesToSort));
    const int candidatePartsPow2 = GreaterPowerOf2EqualOrLarger(candidateParts);
    const int pivotThreads = std::max(32, candidatePartsPow2);

    if(pivotThreads * 2 <= DefaultNbThreadsPerBlock){
        Sort_core_buildPivots<T, MaxNbParts, 2, false, FullSM>
            <<<1, pivotThreads * 2, 0, stream>>>(
                dataToSort, nbValuesToSort, workspace.pivots(), candidateParts);
    }
    else{
        Sort_core_buildPivots<T, MaxNbParts, 1, false, FullSM>
            <<<1, DefaultNbThreadsPerBlock, 0, stream>>>(
                dataToSort, nbValuesToSort, workspace.pivots(), candidateParts);
    }

    AdaptPartitionCountDevice<T, MaxNbParts><<<1, 1, 0, stream>>>(
        workspace.pivots(), candidateParts, workspace.activeNbParts());

    CUDA_CHECK(cudaMemsetAsync(workspace.counters(), 0,
                               (candidateParts + 2) * sizeof(int), stream));

    const int allBlocks =
        (nbValuesToSort + DefaultNbThreadsPerBlock - 1) /
        DefaultNbThreadsPerBlock;
    const int persistentBlocks =
        std::max(1, std::min(allBlocks, workspace.persistentBlocks()));

    CountParts_Persistent<T, MaxNbParts>
        <<<persistentBlocks, DefaultNbThreadsPerBlock, 0, stream>>>(
            dataToSort, nbValuesToSort, workspace.pivots(),
            workspace.blockOffsets(), workspace.blockUniform(),
            workspace.activeNbParts());

    ReduceBlockCounts<MaxNbParts, 256>
        <<<candidateParts, 256, 0, stream>>>(
            workspace.blockOffsets(), workspace.blockUniform(),
            workspace.counters()+1, workspace.uniformParts(),
            persistentBlocks, workspace.activeNbParts());

    DetectSingleUniformBucket<<<1, 1, 0, stream>>>(
        workspace.counters()+1, workspace.uniformParts(),
        workspace.activeNbParts(), workspace.skipScatterFlag());

    BuildPrefixDynamic<int, MaxNbParts>
        <<<1, std::min(candidateParts, DefaultNbThreadsPerBlock), 0, stream>>>(
            workspace.counters()+1, workspace.activeNbParts());

    BuildPersistentBlockOffsets<MaxNbParts>
        <<<candidateParts, 1, 0, stream>>>(
            workspace.blockOffsets(), workspace.counters(),
            persistentBlocks, workspace.activeNbParts());

    MoveElements_Persistent<T, MaxNbParts>
        <<<persistentBlocks, DefaultNbThreadsPerBlock, 0, stream>>>(
            dataToSort, nbValuesToSort, workspace.pivots(),
            workspace.blockOffsets(), workspace.valuesBuffer(),
            workspace.activeNbParts(), workspace.skipScatterFlag());

    auto& hostCounters = workspace.hostCounters();
    auto& hostUniform = workspace.hostUniformParts();
    int nbParts = 0;
    int skippedScatter = 0;
    CUDA_CHECK(cudaMemcpyAsync(&nbParts, workspace.activeNbParts(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostCounters.data(), workspace.counters(),
                               (candidateParts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostUniform.data(), workspace.uniformParts(),
                               candidateParts * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&skippedScatter, workspace.skipScatterFlag(),
                               sizeof(int), cudaMemcpyDeviceToHost, stream));

    // This is now the only host synchronization in the partitioned path.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if(skippedScatter) return;

    auto& largePartitions = workspace.hostLargePartitions();
    int maxSmallPartition = 0;
    bool hasDirectPartition = false;
    int nbLargePartitions = 0;

    for(int idxPart = 0 ; idxPart < nbParts ; ++idxPart){
        const int size = hostCounters[idxPart+1] - hostCounters[idxPart];
        if(size == 0) continue;

        if(hostUniform[idxPart]){
            hasDirectPartition = true;
        }
        else if(size <= ComputeDefaultMaxBitoSort<T>()){
            maxSmallPartition = std::max(maxSmallPartition, size);
        }
        else{
            largePartitions[nbLargePartitions++] = idxPart;
        }
    }

    if(hasDirectPartition || maxSmallPartition > 0){
        size_t sharedBytes = 0;
        if(maxSmallPartition > 32){
            const int workingValues = std::max(
                DefaultMaxBitoSortV2,
                GreaterPowerOf2EqualOrLarger(maxSmallPartition));
            sharedBytes = 2 * static_cast<size_t>(workingValues) * sizeof(T);
        }
        SortPartitionsAdaptive_outofplace<T, TestIfSorted, FullSM>
            <<<nbParts, DefaultNbThreadsPerBlock, sharedBytes, stream>>>(
                workspace.valuesBuffer(), dataToSort,
                workspace.counters(), workspace.uniformParts(), nbParts);
    }

    // The streams and events are created once with the workspace.  This keeps
    // the useful overlap of the legacy path without recreating CUDA objects on
    // every call.
    const int usedStreams =
        std::min(nbLargePartitions, workspace.nbSortStreams());
    for(int idxLarge = 0 ; idxLarge < nbLargePartitions ; ++idxLarge){
        const int idxPart = largePartitions[idxLarge];
        const int start = hostCounters[idxPart];
        const int size = hostCounters[idxPart+1] - start;
        const int idxWorker = idxLarge % usedStreams;
        SortMerge<T, MaxValuesToSort, TestIfSorted, FullSM>(
            workspace.valuesBuffer()+start, size, MergeThreadsPerBlock,
            workspace.sortStream(idxWorker), dataToSort+start, size, false);
    }

    if(usedStreams > 0){
        for(int idxWorker = 0 ; idxWorker < workspace.nbSortStreams(); ++idxWorker){
            CUDA_CHECK(cudaEventRecord(workspace.sortDone(idxWorker),
                                       workspace.sortStream(idxWorker)));
            CUDA_CHECK(cudaStreamWaitEvent(stream, workspace.sortDone(idxWorker), 0));
        }
        workspace.markWorkerEventsRecorded();
    }
}


template <typename T,
          int MaxValuesToSort = 2048,
          int MaxNbParts = 1024,
          bool TestIfSorted = false,
          bool FullSM = true>
__host__ void HybridSortV3(
        T* dataToSort,
        const int nbValuesToSort,
        CuSortWorkspace<T, MaxNbParts>& workspace,
        cudaStream_t stream = 0) {
    if(nbValuesToSort <= 1) return;

    constexpr int MergeThreadsPerBlock =
        std::min(DefaultNbThreadsPerBlock, MaxValuesToSort/16);
    const bool usePartitioning =
        nbValuesToSort >= CuSortTuning<T>::PartitionThreshold;

    if(!usePartitioning){
        // One-block bitonic cases require no workspace at all.
        if(nbValuesToSort <= ComputeDefaultMaxBitoSort<T>()){
            SortMerge<T, MaxValuesToSort, TestIfSorted, FullSM>(
                dataToSort, nbValuesToSort, MergeThreadsPerBlock, stream,
                nullptr, 0, true);
            return;
        }
        if(workspace.workerEventsRecorded()){
            for(int idx = 0 ; idx < workspace.nbSortStreams() ; ++idx){
                CUDA_CHECK(cudaStreamWaitEvent(stream, workspace.sortDone(idx), 0));
            }
        }
        workspace.reserve(static_cast<size_t>(nbValuesToSort));
        SortMerge<T, MaxValuesToSort, TestIfSorted, FullSM>(
            dataToSort, nbValuesToSort, MergeThreadsPerBlock, stream,
            workspace.valuesBuffer(), workspace.capacity(), true);
        return;
    }

    HybridSortPartitionedV3<T, MaxValuesToSort, MaxNbParts,
                            TestIfSorted, FullSM>(
        dataToSort, nbValuesToSort, workspace, stream);
}


// Convenience overload: the fast merge/bitonic dispatch happens before any
// workspace construction.  Only the genuinely partitioned path allocates a
// temporary workspace and therefore has to synchronize before destruction.
template <typename T,
          int MaxValuesToSort = 2048,
          int MaxNbParts = 1024,
          bool TestIfSorted = false,
          bool FullSM = true>
__host__ void HybridSortV3(
        T* dataToSort,
        const int nbValuesToSort,
        cudaStream_t stream = 0) {
    constexpr int MergeThreadsPerBlock =
        std::min(DefaultNbThreadsPerBlock, MaxValuesToSort/16);
    if(nbValuesToSort < CuSortTuning<T>::PartitionThreshold){
        SortMerge<T, MaxValuesToSort, TestIfSorted, FullSM>(
            dataToSort, nbValuesToSort, MergeThreadsPerBlock, stream,
            nullptr, 0, true);
        return;
    }

    CuSortWorkspace<T, MaxNbParts> workspace(nbValuesToSort);
    HybridSortPartitionedV3<T, MaxValuesToSort, MaxNbParts,
                            TestIfSorted, FullSM>(
        dataToSort, nbValuesToSort, workspace, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


template <typename T,
          int MaxValuesToSort = 2048,
          int MaxNbParts = 1024,
          bool TestIfSorted = false,
          bool FullSM = true>
__host__ void HybridSort(
        T* dataToSort,
        const int nbValuesToSort,
        CuSortWorkspace<T, MaxNbParts>& workspace,
        cudaStream_t stream = 0) {
    HybridSortV3<T, MaxValuesToSort, MaxNbParts, TestIfSorted, FullSM>(
        dataToSort, nbValuesToSort, workspace, stream);
}


template <typename T,
          int MaxValuesToSort = 2048,
          int MaxNbParts = 1024,
          bool TestIfSorted = false,
          bool FullSM = true>
__host__ void HybridSort(
        T* dataToSort,
        const int nbValuesToSort,
        cudaStream_t stream = 0) {
    HybridSortV3<T, MaxValuesToSort, MaxNbParts, TestIfSorted, FullSM>(
        dataToSort, nbValuesToSort, stream);
}


#endif
