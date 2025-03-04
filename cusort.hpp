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
constexpr const int DefaultMaxBitoSort = 1024;

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

    if constexpr (TestIfSorted){
        if(TestIfSorted_1Warp(dataToSort[threadIdx.x])){
            return;
        }
    }

    // Only the first 32 threads are used
    if(threadIdx.x < 32){
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
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Warp_SM_le32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort <= 32);
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
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v2(T* dataToSort, const int nbValuesToSort,
                                                            T* buffer){
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
__device__ __forceinline__ T* Core_Sort_1Block_SM_gt32_d_v3(T* dataToSort, const int nbValuesToSort,
                                                            T* buffer, const int nbWorkingValues){
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
__device__ void CountParts_Grid_core(const T* data, const int size, const T* pivots, int* counters,
                                                   int* localCounters, const int nbParts){
    assert(nbParts <= MaxParts);
    __shared__ T localPivots[MaxParts];

    if(threadIdx.x < nbParts){
        localPivots[threadIdx.x] = pivots[threadIdx.x];
        localCounters[threadIdx.x] = 0;
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;
    
    for(int idxValue = id; idxValue < size; idxValue += numThreads){
        const T value = data[idxValue];

        // Binary search on pivots
        const int bucket = binarySearch(localPivots, nbParts, value);
        assert(0 <= bucket && bucket < nbParts);
        atomicAdd(&localCounters[bucket], 1);
    }

    __syncthreads();
}


template <typename T, int MaxParts>
__global__ void CountParts_Grid(const T* data, const int size, const T* pivots, int* counters, const int nbParts){
    // Nb threads must be power of 2
    assert(((blockDim.x - 1) & blockDim.x) == 0);
    assert(blockDim.x >= nbParts);

    __shared__ int localCounters[MaxParts];

    CountParts_Grid_core<T, MaxParts>(data, size, pivots, counters, localCounters, nbParts);

    if(threadIdx.x < nbParts){
        atomicAdd(&counters[threadIdx.x], localCounters[threadIdx.x]);
    }
    __syncthreads();
}

template <typename T, int MaxParts>
__global__ void MoveElements_Grid(T* data, const int size, const T* pivots, int* counters, T* dataBuffer, const int nbParts){
    // Nb threads must be power of 2
    assert(((blockDim.x - 1) & blockDim.x) == 0);
    assert(blockDim.x >= nbParts);

    __shared__ int localCounters[MaxParts];
    CountParts_Grid_core<T, MaxParts>(data, size, pivots, counters, localCounters, nbParts);
    __syncthreads();

    __shared__ int localGOffset[MaxParts];
    if(threadIdx.x < nbParts){
        localGOffset[threadIdx.x] = atomicAdd(&counters[threadIdx.x], localCounters[threadIdx.x]);
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;
    
    for(int idxValue = id; idxValue < size; idxValue += numThreads){
        const T value = data[idxValue];

        // Binary search on pivots
        const int bucket = binarySearch(pivots, nbParts, value);
        assert(0 <= bucket && bucket < nbParts);
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
__global__ void Sort_core_buildPivots(T* dataToSort, const int nbValuesToSort, T* pivots, const int nbParts){
    // We copy values from the array to build the partition
    assert(gridDim.x == 1);
    assert(nbParts <= MaxParts);
    assert(4 <= nbParts);
    static_assert(((MaxParts - 1) & MaxParts) == 0, "MaxParts must be a power of 2");
    __shared__ T localPivots[MaxParts];

    const int nbSubParts = nbParts-3;
    
    if(threadIdx.x < 2){
        localPivots[threadIdx.x] = my_numeric_limits_min<T>();
    } 
    else if(threadIdx.x < nbParts){
        const int step = nbValuesToSort/(nbSubParts+1);
        localPivots[threadIdx.x] = dataToSort[min(nbValuesToSort-1, step*threadIdx.x-2)];
    }
    else if(threadIdx.x < MaxParts){
        localPivots[threadIdx.x] = my_numeric_limits_max<T>();
    } 
    __syncthreads();

    if(nbSubParts <= 32){
        Core_Sort_1Warp_SM_le32_d<T, false, false>(localPivots, nbSubParts);
        // There is a syncthread in the function
    }
    else{
        Core_Sort_1Block_SM_gt32_d<T, false, false>(localPivots, nbSubParts);
        // There is a syncthread in the function
    }

    if(threadIdx.x == 0){
        localPivots[1] = getPrevVal(localPivots[2]);
        localPivots[nbParts-1] = getNextVal(localPivots[nbParts-2]);
    }   
    __syncthreads();
    
    if(threadIdx.x < nbParts){
        pivots[threadIdx.x] = localPivots[threadIdx.x];
    }
    __syncthreads();
}

template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void SortPartitions(T* dataToSort, const int nbValuesToSort, const int counterParts[], const int nbParts){
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


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts = 32, int MaxNbParts = 64>
__host__ void Sort(T* dataToSort, const int nbValuesToSort, cudaStream_t stream = 0, T* dataBuffer = nullptr){
    static_assert(MaxNbParts <= DefaultNbThreadsPerBlock);

    if(nbValuesToSort <= 32){
        Sort_1Warp_SM_le32<T, TestIfSorted, FullSM><<<1,32,0,stream>>>(dataToSort, nbValuesToSort);
        return;
    }

    int MaxThreadsPerBlock = DefaultNbThreadsPerBlock;
    // TODO CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if(NOrPow2 <= DefaultMaxBitoSort){
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,NOrPow2,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
        }
        else{
            Sort_1Block_gt32_extsm_v3<T, TestIfSorted, FullSM><<<1,NOrPow2,2*NOrPow2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, NOrPow2);
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

    SortPartitions<T, TestIfSorted, FullSM, ASyncCpyBito><<< NbParts, MaxThreadsPerBlock, DefaultMaxBitoSort*2*sizeof(T),stream>>>(dataToSort, nbValuesToSort, counterParts, NbParts);

    int counterPartsHost[NbParts+1];
    CUDA_CHECK(cudaMemcpyAsync(counterPartsHost, counterParts, (NbParts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // We need counterPartsHost and that sortPartitions is over
    CUDA_CHECK(cudaStreamSynchronize(stream));
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



#endif