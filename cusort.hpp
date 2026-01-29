#ifndef CUSORT_HPP
#define CUSORT_HPP

#include <algorithm>
#include <iostream>
#include <random>
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

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

constexpr const int DefaultNbThreadsPerBlock = 512;
constexpr const int DefaultMaxBitoSortV2 = DefaultNbThreadsPerBlock;
constexpr const int DefaultMaxBitoSort = 4096;

void CUDA_CHECK(cudaError_t err) {
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
__device__ __forceinline__  constexpr T my_numeric_limits_max(){
    if constexpr (std::is_same<T, int>::value){
        return 2147483647;
    }
    else if(std::is_same<T, float>::value){
        return 3.40282e+38; 
    }
    else if(std::is_same<T, double>::value){
        return 1.79769e+308;
    }
    else{
        return 0;
    }
}

template <typename T>
__device__ __forceinline__  constexpr T my_numeric_limits_min(){
    if constexpr (std::is_same<T, int>::value){
        return -2147483648;
    }
    else if(std::is_same<T, float>::value){
        return -3.40282e+38;
    }
    else if(std::is_same<T, double>::value){
        return -1.79769e+308;
    }
    else{
        return 0;
    }
}

#ifdef __CUDA_ARCH__
template <typename T>
__device__ __forceinline__ T getNextVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val + 1;
    }
    else if(std::is_same<T, float>::value){
        return std::nextafter(val, CUDART_INF_F);
    }
    else if(std::is_same<T, double>::value){
        return std::nextafter(val, CUDART_INF);
    }
    else{
        return 0;
    }
}

template <typename T>
__device__ __forceinline__ T getPrevVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val - 1;
    }
    else if(std::is_same<T, float>::value){
        return std::nextafter(val, -CUDART_INF_F);
    }
    else if(std::is_same<T, double>::value){
        return std::nextafter(val, -CUDART_INF);
    }
    else{
        return 0;
    }
}

#else

template <typename T>
inline __host__ T getNextVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val + 1;
    }
    else if(std::is_same<T, float>::value){
        return std::nextafter(val, std::numeric_limits<float>::infinity());
    }
    else if(std::is_same<T, double>::value){
        return std::nextafter(val, std::numeric_limits<double>::infinity());
    }
    else{
        return 0;
    }
}

template <typename T>
inline __host__ T getPrevVal(const T& val){
    if constexpr (std::is_same<T, int>::value){
        return val - 1;
    }
    else if(std::is_same<T, float>::value){
        return std::nextafter(val, -std::numeric_limits<float>::infinity());
    }
    else if(std::is_same<T, double>::value){
        return std::nextafter(val, -std::numeric_limits<double>::infinity());
    }
    else{
        return 0;
    }
}

#endif // __CUDA_ARCH__

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

// This function tests if values are sorted in a warp
// each thread holding one value to test
template <typename T>
__device__ __forceinline__ bool TestIfSorted_1Warp(const T& dataToSort) {
    // Get the active thread mask and the current lane ID.
    unsigned int mask = __activemask();
    int lane = threadIdx.x & 31;  // assuming warp size is 32

    // Each thread (except lane 0) compares its value with its left neighbor.
    bool isSorted = true;
    T prev = __shfl_up_sync(mask, dataToSort, 1);
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

// Get the min between two values
template <typename T>
__device__ __forceinline__ T min(const T& a, const T& b){
    return a < b ? a : b;
}

// Get the max between two values
template <typename T>
__device__ __forceinline__ T max(const T& a, const T& b){
    return a > b ? a : b;
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
// 32 threads are used, the others are return immediately without syncthreads.
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
                        const T b = __shfl_sync(fullMask, a, vecpermute);
                        a = (ffttout ? max(a, b) : min(a, b));
                    }
            
                    for(long int step = stepout/2 ; step >= 1 ; step/=2){
                        const int vecinc = step;
                        const int fftt = (threadIdx.x & step);
                        const int vecincout2 = (fftt? -vecinc  : vecinc);
                        const int vecpermute = (vecincout2+laneId);
                        const T b = __shfl_sync(fullMask, a, vecpermute);
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
__device__ void Sort_1Warp_SM_le32_d(T* dataToSort, const int nbValuesToSort){
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
    // We do not sync because only the first 32 threads will be used
    // Sort
    Core_Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(shared, nbValuesToSort);
    // Copy back
    if(threadIdx.x < nbValuesToSort){
        dataToSort[threadIdx.x] = shared[threadIdx.x];
    }
    // Ensure all threads see the new data
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Warp_SM_le32(T* dataToSort, const int nbValuesToSort){
    Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}

// Sort at least 32 vales inplace.
// The number of threads must be multiple of 32 if FullSM is false and warp shuffle is used.
// Otherwise the number of threads can be any value.
template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ void Core_Sort_1Block_SM_gt32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort <= blockDim.x);
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
                const T b = __shfl_sync(fullMask, a, vecpermute);
                a = (ffttout ? max(a, b) : min(a, b));
            }
    
            for(long int step = stepout/2 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (vecincout2+laneId);
                const T b = __shfl_sync(fullMask, a, vecpermute);
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
                __syncthreads();
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
                const T b = __shfl_sync(fullMask, a, vecpermute);
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
    extern __shared__ T shared[];
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

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d_extsm(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ T shared_data[];

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
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v2(T* __restrict__ dataToSort, const int nbValuesToSort,
                                                            T* __restrict__ buffer){
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
                const T b = __shfl_sync(fullMask, a, vecpermute);
                a = (ffttout ? max(a, b) : min(a, b));
            }
    
            for(long int step = stepout/2 ; step >= 1 ; step/=2){
                const int vecinc = step;
                const int fftt = (threadIdx.x & step);
                const int vecincout2 = (fftt? -vecinc  : vecinc);
                const int vecpermute = (vecincout2+laneId);
                const T b = __shfl_sync(fullMask, a, vecpermute);
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
                const T b = __shfl_sync(fullMask, a, vecpermute);
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
__device__ void Sort_1Block_gt32_d_extsm_v2(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ T shared[];
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
        dataToSort[threadIdx.x] = lastArray[threadIdx.x];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_v2(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d_extsm_v2(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ T shared_data[];

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
    cooperative_groups::memcpy_async(block, dataToSort, lastArray, sizeof(T) * nbValuesToSort);
    cooperative_groups::wait(block); // Ensure copy back is complete
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm_v2(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_async_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

template <typename T, bool TestIfSorted, bool FullSM>
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v3(T* __restrict__ dataToSort, const int nbValuesToSort,
                                                            T* __restrict__ buffer, const int nbWorkingValues){
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
                    const T b = __shfl_sync(fullMask, a, vecpermute);
                    a = (ffttout ? max(a, b) : min(a, b));
                }
        
                for(long int step = stepout/2 ; step >= 1 ; step/=2){
                    const int vecinc = step;
                    const int fftt = (idxVal & step);
                    const int vecincout2 = (fftt? -vecinc  : vecinc);
                    const int vecpermute = (vecincout2+laneId);
                    const T b = __shfl_sync(fullMask, a, vecpermute);
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
        }
        else{
            for(int idxVal = threadIdx.x ; idxVal < nbWorkingValues ; idxVal += blockDim.x){
                T a = dataToSort[idxVal];
                for(long int step = 16 ; step >= 1 ; step/=2){
                    const int vecinc = step;
                    const int fftt = (idxVal & step);
                    const int vecincout2 = (fftt? -vecinc  : vecinc);
                    const int vecpermute = (vecincout2+laneId);
                    const T b = __shfl_sync(fullMask, a, vecpermute);
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
__device__ void Sort_1Block_gt32_d_extsm_v3(T* dataToSort, const int nbValuesToSort, 
                                            const int nbWorkingValues) {
    // Declare shared memory ptr
    extern __shared__ T shared[];
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
        dataToSort[idxVal] = lastArray[idxVal];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32_extsm_v3(T* dataToSort, const int nbValuesToSort, const int nbWorkingValues) {
    Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort, nbWorkingValues);
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d_extsm_v3(T* dataToSort, const int nbValuesToSort, 
                                                  const int nbWorkingValues) {
    // Declare shared memory ptr
    extern __shared__ T shared_data[];

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
    cooperative_groups::memcpy_async(block, dataToSort, lastArray, sizeof(T) * nbValuesToSort);
    cooperative_groups::wait(block); // Ensure copy back is complete
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32_extsm_v3(T* dataToSort, const int nbValuesToSort, const int nbWorkingValues) {
    Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort, nbWorkingValues);
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
        if(bucket == -1 || !(localPivots[bucket] <= value && value < localPivots[bucket+1])){
            bucket = binarySearch(localPivots, nbParts, value);
            assert(0 <= bucket && bucket < nbParts);
        }
        const int insertPosition = atomicAdd(&localGOffset[bucket], 1);
        dataBuffer[insertPosition] = value;
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

template <typename NumType, typename IndexType, const int MaxNbParts>
__global__ void BuildPrefix_SM( NumType* prefixTree, const IndexType inSize){
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

template <typename T, int MaxParts>
__global__ void Sort_core_buildPivots(T* __restrict__ dataToSort, const int nbValuesToSort, T* __restrict__ pivots, const int nbParts){
    // We copy values from the array to build the partition
    assert(gridDim.x == 1);
    assert(nbParts <= MaxParts);
    assert(4 <= nbParts);
    assert(32 <= blockDim.x);
    static_assert(((MaxParts - 1) & MaxParts) == 0, "MaxParts must be a power of 2");
    __shared__ T localPivots[MaxParts];

    const int nbSubParts = nbParts-3;
    
    if(threadIdx.x < 2){
        localPivots[threadIdx.x] = my_numeric_limits_min<T>();
    } 
    for(int idx = threadIdx.x + 2 ; idx < nbParts ; idx += blockDim.x){
        const int step = nbValuesToSort/(nbSubParts+1);
        localPivots[idx] = dataToSort[min(nbValuesToSort-1, step*idx-2)];
    }
    for(int idx = threadIdx.x + nbParts ; idx < MaxParts ; idx += blockDim.x){
        localPivots[idx] = my_numeric_limits_max<T>();
    } 
    __syncthreads();

    if(nbParts <= 32){
        Core_Sort_1Warp_SM_le32_d<T, false, false>(localPivots, nbParts);
        // There is a syncthread in the function
    }
    else if(nbParts <= 512){
        Core_Sort_1Block_SM_gt32_d<T, false, false>(localPivots, nbParts);
        // There is a syncthread in the function
    }
    else {
        __shared__ T buffer[MaxParts];
        Core_Sort_1Block_SM_gt32_d_v3<T, false, false>(localPivots, nbParts,
                                                            buffer, MaxParts);
        // There is a syncthread in the function
    }

    if(threadIdx.x == 0){
        localPivots[1] = getPrevVal(localPivots[2]);
        localPivots[nbParts-1] = getNextVal(localPivots[nbParts-2]);
    }   
    __syncthreads();
    
    for(int idx = threadIdx.x ; idx < nbParts ; idx += blockDim.x){
        pivots[threadIdx.x] = localPivots[threadIdx.x];
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
            Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort+start, size);
            __syncthreads();
        }
        else if(size <= DefaultMaxBitoSort){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= DefaultMaxBitoSort);
            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, size, NOrPow2);
            }
            else{
                Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort+start, size, NOrPow2);
            }
        }
    }
}

template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void SortOnePartition(T* dataToSort, const int size){
    assert(size <= DefaultMaxBitoSort);
    if(size <= 32){
        Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort, size);
        __syncthreads();
    }
    else if(size <= DefaultMaxBitoSort){
        const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
        assert(NOrPow2 <= DefaultMaxBitoSort);
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, size, NOrPow2);
        }
        else{
            Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort, size, NOrPow2);
        }
    }
}



template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts = 32, int MaxNbParts = 1024>
__host__ void Sort(T* dataToSort, const int nbValuesToSort, cudaStream_t stream = 0, T* dataBuffer = nullptr){
    if(nbValuesToSort <= 32){
        Sort_1Warp_SM_le32<T, TestIfSorted, FullSM><<<1,32,0,stream>>>(dataToSort, nbValuesToSort);
        return;
    }

    int MaxThreadsPerBlock = DefaultNbThreadsPerBlock;
    // TODO CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if(NOrPow2 <= DefaultMaxBitoSortV2){
        const int nbThreads = std::min(NOrPow2, MaxThreadsPerBlock);
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_extsm_v2<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort);
        }
        else{
            Sort_1Block_gt32_extsm_v2<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort);
        }
        return;
    }

    if(NOrPow2 <= DefaultMaxBitoSort){
        const int nbThreads = std::min(NOrPow2, MaxThreadsPerBlock);
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        }
        else{
            Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        }
        return;
    }

    const int NbParts = std::min(MaxNbParts, std::max(MinNbParts, nbValuesToSort/MaxThreadsPerBlock));
    const int NbPartsPow2 = GreaterPowerOf2EqualOrLarger(NbParts);

    const bool allocatedDataBuffer = (dataBuffer == nullptr);
    if(allocatedDataBuffer){
        CUDA_CHECK(cudaMallocAsync(&dataBuffer, nbValuesToSort * sizeof(T), stream));
    }

    int* pivots = nullptr;
    CUDA_CHECK(cudaMallocAsync(&pivots, NbParts * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(pivots, 0, NbParts * sizeof(int), stream));

    const int NbPartsPow2Or32 = std::max(32, NbPartsPow2);
    Sort_core_buildPivots<T, MaxNbParts><<<1, NbPartsPow2Or32, 0, stream>>>(dataToSort, nbValuesToSort, pivots, NbParts);

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMallocAsync(&counterParts, (NbParts+2) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(counterParts, 0, (NbParts+2) * sizeof(int), stream));

    const int nbThreadBlock = (nbValuesToSort+MaxThreadsPerBlock-1)/MaxThreadsPerBlock;

    CountParts_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock, 0, stream >>>(dataToSort, nbValuesToSort, pivots, counterParts+2, NbParts);

    BuildPrefix_SM<T, int, MaxNbParts><<<1,NbParts,0,stream>>>(counterParts+2, NbParts);

    MoveElements_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock,0,stream >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts+1,
                                            dataBuffer, NbParts);

    int pivotsHost[NbParts];
    CUDA_CHECK(cudaMemcpyAsync(pivotsHost, pivots, NbParts * sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyAsync(dataToSort, dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice, stream));
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
        if(size > maxSizePartToSort && size <= DefaultMaxBitoSort){
            maxSizePartToSort = size;
        }
    }
    const int maxSizePartToSortPow2 = GreaterPowerOf2EqualOrLarger(maxSizePartToSort);

    SortPartitions<T, TestIfSorted, FullSM, ASyncCpyBito><<< NbParts, MaxThreadsPerBlock, maxSizePartToSortPow2*2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, counterParts, NbParts);

    CUDA_CHECK(cudaFreeAsync(counterParts, stream));  

    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if(size > DefaultMaxBitoSort
                && (idxPart == NbParts-1 || getNextVal(pivotsHost[idxPart]) < pivotsHost[idxPart+1])){
            Sort<T, TestIfSorted, FullSM, ASyncCpyBito, MinNbParts, MaxNbParts>(dataToSort+start, size, stream, dataBuffer);
        }
    }

    if(allocatedDataBuffer){
        CUDA_CHECK(cudaFreeAsync(dataBuffer, stream));
    }
}


#include "tpool.hpp"

template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts = 32, int MaxNbParts = 1024>
__host__ void SortOnePar(ThreadPool& pool, T* dataToSort, const int nbValuesToSort, cudaStream_t stream = 0, T* dataBuffer = nullptr){
    if(nbValuesToSort <= 32){
        Sort_1Warp_SM_le32<T, TestIfSorted, FullSM><<<1,32,0,stream>>>(dataToSort, nbValuesToSort);
        return;
    }

    int MaxThreadsPerBlock = DefaultNbThreadsPerBlock;
    // TODO CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if(NOrPow2 <= DefaultMaxBitoSortV2){
        const int nbThreads = std::min(NOrPow2, MaxThreadsPerBlock);
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_extsm_v2<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort);
        }
        else{
            Sort_1Block_gt32_extsm_v2<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort);
        }
        return;
    }

    if(NOrPow2 <= DefaultMaxBitoSort){
        const int nbThreads = std::min(NOrPow2, MaxThreadsPerBlock);
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        }
        else{
            Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreads,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        }
        return;
    }

    const int NbParts = std::min(MaxNbParts, std::max(MinNbParts, nbValuesToSort/MaxThreadsPerBlock));
    const int NbPartsPow2 = GreaterPowerOf2EqualOrLarger(NbParts);

    const bool allocatedDataBuffer = (dataBuffer == nullptr);
    if(allocatedDataBuffer){
        CUDA_CHECK(cudaMallocAsync(&dataBuffer, nbValuesToSort * sizeof(T), stream));
    }

    int* pivots = nullptr;
    CUDA_CHECK(cudaMallocAsync(&pivots, NbParts * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(pivots, 0, NbParts * sizeof(int), stream));

    Sort_core_buildPivots<T, MaxNbParts><<<1, NbPartsPow2, 0, stream>>>(dataToSort, nbValuesToSort, pivots, NbParts);

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMallocAsync(&counterParts, (NbParts+2) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(counterParts, 0, (NbParts+2) * sizeof(int), stream));

    const int nbThreadBlock = (nbValuesToSort+MaxThreadsPerBlock-1)/MaxThreadsPerBlock;

    CountParts_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock, 0, stream >>>(dataToSort, nbValuesToSort, pivots, counterParts+2, NbParts);

    BuildPrefix_SM<T, int, MaxNbParts><<<1,NbParts,0,stream>>>(counterParts+2, NbParts);

    MoveElements_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock,0,stream >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts+1,
                                            dataBuffer, NbParts);

    int pivotsHost[NbParts];
    CUDA_CHECK(cudaMemcpyAsync(pivotsHost, pivots, NbParts * sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyAsync(dataToSort, dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFreeAsync(pivots, stream));

    int counterPartsHost[NbParts+1];
    CUDA_CHECK(cudaMemcpyAsync(counterPartsHost, counterParts, (NbParts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));

    cudaStream_t asyncSorts[NbParts];
    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        CUDA_CHECK(cudaStreamCreateWithFlags(&asyncSorts[idxPart], cudaStreamNonBlocking));
    }

    // We need counterPartsHost and that sortPartitions is over
    CUDA_CHECK(cudaFreeAsync(counterParts, stream));  
    CUDA_CHECK(cudaStreamSynchronize(stream));  

    auto g = pool.make_group();

    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if(size > DefaultMaxBitoSort
                && (idxPart == NbParts-1 || getNextVal(pivotsHost[idxPart]) < pivotsHost[idxPart+1])){
            g.run([&, idxPart, start, size]
            {
                SortOnePar<T, TestIfSorted, FullSM, ASyncCpyBito, MinNbParts, MaxNbParts>(pool, dataToSort+start, size, asyncSorts[idxPart], dataBuffer+start);
            });
        }
        else if(size <= DefaultMaxBitoSort && size > 0){
            g.run([&, idxPart, start, size]
            {
                const int NOrPow2Part = GreaterPowerOf2EqualOrLarger(size);
                SortOnePartition<T, TestIfSorted, FullSM, ASyncCpyBito><<< 1, MaxThreadsPerBlock, NOrPow2Part*2*sizeof(T), asyncSorts[idxPart]>>>(dataToSort+start, size);
            });
            // const int nbThreadsPart = std::min(NOrPow2Part, MaxThreadsPerBlock);
            // if constexpr(ASyncCpyBito){
            //     Sort_1Block_async_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreadsPart,2*NOrPow2Part*sizeof(T),stream>>>(dataToSort+start, size, NOrPow2Part);
            // }
            // else{
            //     Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,nbThreadsPart,2*NOrPow2Part*sizeof(T),stream>>>(dataToSort+start, size, NOrPow2Part);
            // }
        }
    }

    g.wait();

    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        CUDA_CHECK(cudaStreamSynchronize(asyncSorts[idxPart]));
        CUDA_CHECK(cudaStreamDestroy(asyncSorts[idxPart]));
    } 

    if(allocatedDataBuffer){
        CUDA_CHECK(cudaFreeAsync(dataBuffer, stream));
    }
}


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts = 32, int MaxNbParts = 1024>
__host__ void SortOne(ThreadPool& pool, T* dataToSort, const int nbValuesToSort, cudaStream_t stream = 0, T* dataBuffer = nullptr){
    SortOnePar<T, TestIfSorted, FullSM, ASyncCpyBito, MinNbParts, MaxNbParts>(pool, dataToSort, nbValuesToSort, stream, dataBuffer);
}

/////////////////////////////////////////////////////////////////////////////


template <typename T, int TILE>
__device__ __forceinline__ int mergePathDiag(
    const T* A, int aCount,
    const T* B, int bCount,
    int diag
){
    int lo = max(0, diag - bCount);
    int hi = min(aCount, diag);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int j   = diag - mid;
        // même logique que toi
        if (mid < aCount && j > 0 && A[mid] < B[j - 1]) lo = mid + 1;
        else                                           hi = mid;
    }
    return lo; // iA, et jB = diag - iA
}

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

    // Shared layout FIXE : A puis B (pas de bufM)
    __shared__ T shA[TILE];
    __shared__ T shB[TILE];

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
    int iA_end = mergePathDiag(shA, localA, shB, localB, t1);
    int jB_end = t1 - iA_end;

    // Petite optimisation : garde les valeurs en registres
    for (int k = t0; k < t1; ++k) {
        bool takeA = (jB >= localB) || (iA < localA && shA[iA] <= shB[jB]);
        out[k] = takeA ? shA[iA++] : shB[jB++];
    }
}



template <typename T, const int MaxValuesToSort>
__global__ void SortChunks(T* dataToSort, const int nbValuesToSort){
    const int nbChunks = (nbValuesToSort + MaxValuesToSort - 1) / MaxValuesToSort;
    for(int idxChunk = blockIdx.x ; idxChunk < nbChunks; idxChunk += gridDim.x){
        const int start = idxChunk*MaxValuesToSort;
        const int end = min(start+MaxValuesToSort, nbValuesToSort);
        const int size = end-start;
        if(size <= 32){
            Sort_1Warp_SM_le32_d<T, false, false>(dataToSort+start, size);
        }
        else if(size <= DefaultMaxBitoSortV2){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= DefaultMaxBitoSortV2);
            Sort_1Block_gt32_d_extsm_v2<T, false, false>(dataToSort+start, size);
        }
        else if(size <= MaxValuesToSort){
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(size);
            assert(NOrPow2 <= MaxValuesToSort);
            Sort_1Block_gt32_d_extsm_v3<T, false, false>(dataToSort+start, size, NOrPow2);
        }
    }
}


template <typename T, int MaxValuesToSort = 1024>
__host__ void SortMerge(
    T* dataToSort,
    int nbValuesToSort,
    cudaStream_t stream = 0,
    T* externalBuffer = nullptr,
    size_t externalBufferElems = 0
){
    constexpr bool TestIfSorted = false;
    constexpr bool FullSM       = false;

    auto ceil_div = [](int a, int b) -> int { return (a + b - 1) / b; };

    if (nbValuesToSort <= 1) return;

    // Petits cas
    if (nbValuesToSort <= 32) {
        Sort_1Warp_SM_le32<T, TestIfSorted, FullSM>
            <<<1, 32, 0, stream>>>(dataToSort, nbValuesToSort);
        return;
    }

    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if (NOrPow2 <= DefaultMaxBitoSort) {
        const int nbThreads = std::min(NOrPow2, DefaultNbThreadsPerBlock);
        Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM>
            <<<1, nbThreads, 2 * NOrPow2 * sizeof(T), stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        return;
    }

    // 1) Sort chunks (in-place)
    {
        const int NbBlocks = ceil_div(nbValuesToSort, MaxValuesToSort);
        SortChunks<T, MaxValuesToSort>
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

    // 3) Passes de merge
    constexpr int IdealMergeSize = 2048;

    T* in  = dataToSort;
    T* out = dataBuffer;

    // Phase 1 : TILE = MaxValuesToSort
    for (int run_size = MaxValuesToSort;
         run_size < nbValuesToSort && run_size < IdealMergeSize;
         run_size *= 2)
    {
        constexpr int TILE = MaxValuesToSort;

        const int numMerges      = ceil_div(nbValuesToSort, 2 * run_size);
        const int blocksPerMerge = ceil_div(2 * run_size, TILE);
        const int gridX          = numMerges * blocksPerMerge;

        mergeKernel<T, TILE>
            <<<gridX, DefaultNbThreadsPerBlock, 0, stream>>>(
                in, out, run_size, nbValuesToSort
            );

        std::swap(in, out);
    }

    // Phase 2 : TILE = IdealMergeSize
    for (int run_size = IdealMergeSize;
         run_size < nbValuesToSort;
         run_size *= 2)
    {
        constexpr int TILE = IdealMergeSize;

        const int numMerges      = ceil_div(nbValuesToSort, 2 * run_size);
        const int blocksPerMerge = ceil_div(2 * run_size, TILE);
        const int gridX          = numMerges * blocksPerMerge;

        mergeKernel<T, TILE>
            <<<gridX, DefaultNbThreadsPerBlock, 0, stream>>>(
                in, out, run_size, nbValuesToSort
            );

        std::swap(in, out);
    }

    // 4) Si le résultat est dans le buffer, copie finale (nécessaire, propre, correct)
    if (in != dataToSort) {
        CUDA_CHECK(cudaMemcpyAsync(
            dataToSort, in,
            nbValuesToSort * sizeof(T),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }

    if (ownsBuffer) {
        CUDA_CHECK(cudaFreeAsync(dataBuffer, stream));
    }
}




#endif
