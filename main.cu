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

#include "SpTimer.hpp"

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

constexpr const int DefaultNbThreadsPerBlock = 512;

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int GreaterPowerOf2EqualOrLarger(int nbValuesToSort){
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

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

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

// Sort 32 values or less. The values are sorted in place.
// There might be more threads than 32.
// If FullSM is used, the shared memory is used to store the values while sorting.
// Otherwise the values are sorted using warp shuffle.
// 32 threads are used, the others are return immediately without syncthreads.
template <typename T, bool FullSM = true>
__device__ __forceinline__ void Core_Sort_1Warp_SM_le32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort <= 32);
    assert(32 <= blockDim.x);

    // Only the first 32 threads are used
    if(32 <= threadIdx.x){
        return;
    }

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
        }
        dataToSort[threadIdx.x] = a;
    }
}

template <typename T, bool FullSM = true>
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
    Core_Sort_1Warp_SM_le32_d<T, FullSM>(shared, nbValuesToSort);
    // Copy back
    if(threadIdx.x < nbValuesToSort){
        dataToSort[threadIdx.x] = shared[threadIdx.x];
    }
    // Ensure all threads see the new data
    __syncthreads();
}

template <typename T, bool FullSM = true>
__global__ void Sort_1Warp_SM_le32(T* dataToSort, const int nbValuesToSort){
    Sort_1Warp_SM_le32_d(dataToSort, nbValuesToSort);
}

// Sort at least 32 vales inplace.
// The number of threads must be multiple of 32 if FullSM is false and warp shuffle is used.
// Otherwise the number of threads can be any value.
template <typename T, bool TestIfSorted = false, bool FullSM = true>
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
    Sort_1Block_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);// TODO !!!
    return;

    // Ensure valid sorting range
    if (nbValuesToSort > blockDim.x) return;

    // Declare shared memory ptr
    extern __shared__ T shared_data[];

    auto block = cooperative_groups::this_thread_block();

    // Cooperative Async Copy
    cooperative_groups::memcpy_async(block, shared_data, dataToSort, sizeof(T) * nbValuesToSort);
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


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


template <typename T>
__forceinline__ __device__ int binarySearch(const T pivots[], const int nbPivots, const T value) {
    int start = 0, end = nbPivots;
    while (start < end - 1) {
        int mid = (start + end) / 2;
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
    __shared__ T localPivots[MaxParts];
    
    if(threadIdx.x < nbParts){
        const int step = nbValuesToSort/(nbParts+1);
        localPivots[threadIdx.x] = dataToSort[min(nbValuesToSort-1, step*threadIdx.x)];
    }
    else if(threadIdx.x < MaxParts){
        localPivots[threadIdx.x] = my_numeric_limits_max<T>();
    }
    if(threadIdx.x == 0){
        localPivots[0] = my_numeric_limits_min<T>();
    }    
    __syncthreads();

    if(nbParts <= 32){
        Core_Sort_1Warp_SM_le32_d<T, true>(localPivots, nbParts);
        __syncthreads();
    }
    else{
        Core_Sort_1Block_SM_gt32_d<T, false, false>(localPivots, nbParts);
        // There is a syncthread in the function
    }

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
            Sort_1Warp_SM_le32_d<T, FullSM>(dataToSort+start, size);
            __syncthreads();
        }
        else if(size <= blockDim.x){
            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort+start, size);
            }
            else{
                Sort_1Block_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort+start, size);
            }
        }
    }
}


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito, int MinNbParts = 32, int MaxNbParts = 64>
__host__ void Sort(T* dataToSort, const int nbValuesToSort){
    static_assert(MaxNbParts <= DefaultNbThreadsPerBlock);

    if(nbValuesToSort <= 32){
        Sort_1Warp_SM_le32<T, FullSM><<<1,32>>>(dataToSort, nbValuesToSort);
        return;
    }

    int MaxThreadsPerBlock = DefaultNbThreadsPerBlock;
    // TODO CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if(NOrPow2 <= MaxThreadsPerBlock){
        if constexpr(ASyncCpyBito){
            Sort_1Block_async_gt32_extsm<T, TestIfSorted, FullSM><<<1,NOrPow2,NOrPow2*sizeof(T)>>>(dataToSort, nbValuesToSort);
        }
        else{
            Sort_1Block_gt32_extsm<T, TestIfSorted, FullSM><<<1,NOrPow2,NOrPow2*sizeof(T)>>>(dataToSort, nbValuesToSort);
        }
        return;
    }

    const int NbParts = std::min(MaxNbParts, std::max(MinNbParts, nbValuesToSort/MaxThreadsPerBlock));
    const int NbPartsPow2 = GreaterPowerOf2EqualOrLarger(NbParts);

    int* dataBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dataBuffer, nbValuesToSort * sizeof(T)));

    int* pivots = nullptr;
    CUDA_CHECK(cudaMalloc(&pivots, NbParts * sizeof(int)));
    CUDA_CHECK(cudaMemset(pivots, 0, NbParts * sizeof(int)));

    Sort_core_buildPivots<T, MaxNbParts><<<1, NbPartsPow2 >>>(dataToSort, nbValuesToSort, pivots, NbParts);

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMalloc(&counterParts, (NbParts+2) * sizeof(int)));
    CUDA_CHECK(cudaMemset(counterParts, 0, (NbParts+2) * sizeof(int)));

    const int nbThreadBlock = (nbValuesToSort+MaxThreadsPerBlock-1)/MaxThreadsPerBlock;
    CountParts_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock >>>(dataToSort, nbValuesToSort, pivots, counterParts+2, NbParts);


    CUDA_CHECK(cudaDeviceSynchronize());

    BuildPrefix_SM<T, int, MaxNbParts><<<1,NbParts>>>(counterParts+2, NbParts);


    CUDA_CHECK(cudaDeviceSynchronize());

    MoveElements_Grid<T, MaxNbParts><<< nbThreadBlock, MaxThreadsPerBlock >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts+1,
                                            dataBuffer, NbParts);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dataToSort, dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(dataBuffer));
    CUDA_CHECK(cudaFree(pivots));

    SortPartitions<T, TestIfSorted, FullSM, ASyncCpyBito><<< NbParts, MaxThreadsPerBlock, MaxThreadsPerBlock*sizeof(T)>>>(dataToSort, nbValuesToSort, counterParts, NbParts);

    CUDA_CHECK(cudaDeviceSynchronize());
    int counterPartsHost[NbParts+1];
    CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, (NbParts+1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(counterParts));    
    
    for(int idxPart = 0 ; idxPart < NbParts ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if(size > MaxThreadsPerBlock){
            Sort<T, TestIfSorted, FullSM, ASyncCpyBito, MinNbParts, MaxNbParts>(dataToSort+start, size);
        }
    }
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template<bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
void testSmall() {
    // Size of the shared memory on the device, calling cuda api
    int MaxSharedMemSize;
    CUDA_CHECK(cudaDeviceGetAttribute(&MaxSharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    std::cout << "Shared memory size: " << MaxSharedMemSize << std::endl;
    // Max number of threads per block
    int MaxThreadsPerBlock;
    CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    std::cout << "Max threads per block: " << MaxThreadsPerBlock << std::endl;
    MaxThreadsPerBlock /= 2; // Does not work otherwise
    // Limit the size to this value
    const int MaxSize = std::min(int(MaxSharedMemSize / sizeof(int)), MaxThreadsPerBlock);
    std::cout << "Max size: " << MaxSize << std::endl;

    for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 2){
        for(const bool randValues : {false, true}){
            // Number of elements to sort
            const int N = idxSize;
            assert(N <= MaxSharedMemSize / sizeof(int));
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);
            assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

            std::cout << "Testing with " << N << " elements" << std::endl;
            std::cout << " - Number of threads: " << NOrPow2 << std::endl;
            std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

            std::vector<int> h_array(N);
            if(randValues){
                // Create a host vector and fill it with random values (using C++11 random)
                std::random_device rd;
                std::mt19937 gen(0);
                std::uniform_int_distribution<int> dis(0, 10000);
                for (int i = 0; i < N; ++i) {
                    h_array[i] = dis(gen);
                }
            }
            else{
                for (int i = 0; i < N; ++i) {
                    h_array[i] = N-i;
                }
            }

            // Make a copy of the host array and sort it using std::sort for later verification.
            std::vector<int> h_sorted = h_array;
            std::sort(h_sorted.begin(), h_sorted.end());

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            // Copy data from host to device
            CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

            // Define execution configuration (threads per block and blocks per grid)
            const int threadsPerBlock = NOrPow2;
            const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

            // Launch the sort kernel (template instantiated for int)
            const int sharedMemSize = NOrPow2 * sizeof(int);

            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_extsm<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }
            else{
                Sort_1Block_gt32_extsm<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }

            // Wait for GPU to finish before accessing on host
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy the (supposedly sorted) array back from device to host
            std::vector<int> h_result(N);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

            // Free device memory
            CUDA_CHECK(cudaFree(d_array));

            // Verify that the device result matches the sorted host array
            if (h_result != h_sorted) {
                for(int i = 0 ; i < N ; i++){
                    std::cout << h_result[i] << " " << h_sorted[i] << std::endl;
                }
                std::cout << "The array is NOT correctly sorted." << std::endl;
                assert(0);
            }
        }
    }


    for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 3){
        for(const bool randValues : {false, true}){
            // Number of elements to sort
            const int N = idxSize;
            assert(N <= MaxSharedMemSize / sizeof(int));
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);
            assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

            std::cout << "Testing with " << N << " elements" << std::endl;
            std::cout << " - Number of threads: " << NOrPow2 << std::endl;
            std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

            std::vector<int> h_array(N);
            if(randValues){
                // Create a host vector and fill it with random values (using C++11 random)
                std::random_device rd;
                std::mt19937 gen(0);
                std::uniform_int_distribution<int> dis(0, 10000);
                for (int i = 0; i < N; ++i) {
                    h_array[i] = dis(gen);
                }
            }
            else{
                for (int i = 0; i < N; ++i) {
                    h_array[i] = N-i;
                }
            }

            // Make a copy of the host array and sort it using std::sort for later verification.
            std::vector<int> h_sorted = h_array;
            std::sort(h_sorted.begin(), h_sorted.end());

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            // Copy data from host to device
            CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

            // Define execution configuration (threads per block and blocks per grid)
            const int threadsPerBlock = NOrPow2;
            const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

            // Launch the sort kernel (template instantiated for int)
            const int sharedMemSize = NOrPow2 * sizeof(int);

            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_extsm<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }
            else{
                Sort_1Block_gt32_extsm<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }

            // Wait for GPU to finish before accessing on host
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy the (supposedly sorted) array back from device to host
            std::vector<int> h_result(N);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

            // Free device memory
            CUDA_CHECK(cudaFree(d_array));

            // Verify that the device result matches the sorted host array
            if (h_result != h_sorted) {
                for(int i = 0 ; i < N ; i++){
                    std::cout << h_result[i] << " " << h_sorted[i] << std::endl;
                }
                std::cout << "The array is NOT correctly sorted." << std::endl;
                assert(0);
            }
        }
    }


    for(int idxSize = 1 ; idxSize <= 32 ; ++idxSize){
        for(const bool randValues : {false, true}){
            // Number of elements to sort
            const int N = idxSize;

            std::cout << "Testing with " << N << " elements" << std::endl;

            std::vector<int> h_array(N);
            if(randValues){
                // Create a host vector and fill it with random values (using C++11 random)
                std::random_device rd;
                std::mt19937 gen(0);
                std::uniform_int_distribution<int> dis(0, 10000);
                for (int i = 0; i < N; ++i) {
                    h_array[i] = dis(gen);
                }
            }
            else{
                for (int i = 0; i < N; ++i) {
                    h_array[i] = N-i;
                }
            }

            // Make a copy of the host array and sort it using std::sort for later verification.
            std::vector<int> h_sorted = h_array;
            std::sort(h_sorted.begin(), h_sorted.end());

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            // Copy data from host to device
            CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

            Sort_1Warp_SM_le32<int, FullSM><<<1,32>>>(d_array, N);

            // Wait for GPU to finish before accessing on host
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy the (supposedly sorted) array back from device to host
            std::vector<int> h_result(N);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

            // Free device memory
            CUDA_CHECK(cudaFree(d_array));

            // Verify that the device result matches the sorted host array
            if (h_result != h_sorted) {
                for(int i = 0 ; i < N ; i++){
                    std::cout << h_result[i] << " " << h_sorted[i] << std::endl;
                }
                std::cout << "The array is NOT correctly sorted." << std::endl;
                assert(0);
            }
        }
    }
}

template<bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
void testLarge() {
    // Limit the size to this value
    const int MaxSize = (1 << 20);
    std::cout << "Max size: " << MaxSize << std::endl;

    for(int idxSize = 32 ; idxSize < MaxSize ; idxSize *= 2){
        for(const bool randValues : {false, true}){
            // Number of elements to sort
            const int N = idxSize;
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);

            std::cout << "Testing with " << N << " elements" << std::endl;

            std::vector<int> h_array(N);
            if(randValues){
                // Create a host vector and fill it with random values (using C++11 random)
                std::random_device rd;
                std::mt19937 gen(0);
                std::uniform_int_distribution<int> dis(0, 10000);
                for (int i = 0; i < N; ++i) {
                    h_array[i] = dis(gen);
                }
            }
            else{
                for (int i = 0; i < N; ++i) {
                    h_array[i] = N-i;
                }
            }

            // Make a copy of the host array and sort it using std::sort for later verification.
            std::vector<int> h_sorted = h_array;
            std::sort(h_sorted.begin(), h_sorted.end());

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            // Copy data from host to device
            CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

            Sort<int, TestIfSorted, FullSM, ASyncCpyBito>(d_array, N);

            // Wait for GPU to finish before accessing on host
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy the (supposedly sorted) array back from device to host
            std::vector<int> h_result(N);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

            // Free device memory
            CUDA_CHECK(cudaFree(d_array));

            // Verify that the device result matches the sorted host array
            if (h_result != h_sorted) {
                for(int i = 0 ; i < N ; i++){
                    if(h_result[i] != h_sorted[i]){
                        std::cout << i << " " << h_result[i] << " " << h_sorted[i] << std::endl;
                    }
                }
                std::cout << "The array is NOT correctly sorted." << std::endl;
                assert(0);
            }
        }
    }
}


void testall() {
    if(true){
        testSmall<true, true, true>();
        testSmall<true, true, false>();
        testSmall<true, false, true>();
        testSmall<true, false, false>();
        testSmall<false, true, true>();
        testSmall<false, true, false>();
        testSmall<false, false, true>();
        testSmall<false, false, false>();
    }
    if(true){
        testLarge<true, true, true>();
        testLarge<true, true, false>();
        testLarge<true, false, true>();
        testLarge<true, false, false>();
        testLarge<false, true, true>();
        testLarge<false, true, false>();
        testLarge<false, false, true>();
        testLarge<false, false, false>();
    }
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

enum InitData{
    SORTED,
    REVERSED,
    RANDOM,
    NB_INITS
};

template <typename T, bool FullSM>
__global__ void benchmarkSortSmall(T* dataToSort, const int nbValuesToSort) {
    Sort_1Warp_SM_le32_d<T, FullSM>(dataToSort + blockIdx.x*32, nbValuesToSort);
}


void benmark32(){
    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / 32;
    std::vector<int> h_array(nbElements);

    for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
        for(int idxSizePerWarp = 1 ; idxSizePerWarp <= 32 ; ++idxSizePerWarp){
            for(int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
                {
                    std::cout << "idxInitData: " << idxInitData << " idxSizePerWarp: " << idxSizePerWarp << " idxLoop: " << idxLoop << std::endl;


                    if(idxInitData == RANDOM){
                        std::random_device rd;
                        std::mt19937 gen(0);
                        std::uniform_int_distribution<int> dis(0, 1000000);
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = dis(gen);
                        }
                    }
                    else if(idxInitData == SORTED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = i%32;
                        }
                    }
                    else if(idxInitData == REVERSED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = 31-i%32;
                        }
                    }
                    else{
                        assert(0);
                    }

                    // Allocate device memory for the array
                    int* d_array = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_array, h_array.size() * sizeof(int)));

                    // Copy data from host to device
                    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(int), cudaMemcpyHostToDevice));

                    // Define execution configuration (threads per block and blocks per grid)
                    const int threadsPerBlock = 32;
                    const int blocksPerGrid = nbBlocks;

                    CUDA_CHECK(cudaDeviceSynchronize());
                    SpTimer timer;

                    benchmarkSortSmall<int, true><<<blocksPerGrid, threadsPerBlock>>>(d_array, idxSizePerWarp);

                    CUDA_CHECK(cudaDeviceSynchronize());
                    timer.stop();

                    std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                    std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                    // Free device memory
                    CUDA_CHECK(cudaFree(d_array));
                }
                {
                    std::cout << "idxInitData: " << idxInitData << " idxSizePerWarp: " << idxSizePerWarp << " idxLoop: " << idxLoop << std::endl;


                    if(idxInitData == RANDOM){
                        std::random_device rd;
                        std::mt19937 gen(0);
                        std::uniform_int_distribution<int> dis(0, 1000000);
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = dis(gen);
                        }
                    }
                    else if(idxInitData == SORTED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = i%32;
                        }
                    }
                    else if(idxInitData == REVERSED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = 31-i%32;
                        }
                    }
                    else{
                        assert(0);
                    }

                    // Allocate device memory for the array
                    int* d_array = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_array, h_array.size() * sizeof(int)));

                    // Copy data from host to device
                    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(int), cudaMemcpyHostToDevice));

                    // Define execution configuration (threads per block and blocks per grid)
                    const int threadsPerBlock = 32;
                    const int blocksPerGrid = nbBlocks;

                    CUDA_CHECK(cudaDeviceSynchronize());
                    SpTimer timer;

                    benchmarkSortSmall<int, false><<<blocksPerGrid, threadsPerBlock>>>(d_array, idxSizePerWarp);

                    CUDA_CHECK(cudaDeviceSynchronize());
                    timer.stop();

                    std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                    std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                    // Free device memory
                    CUDA_CHECK(cudaFree(d_array));
                }
            }
        }
    }
}


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void benchmarkSort512(T* dataToSort, const int nbValuesToSort) {
    if constexpr(ASyncCpyBito){
        Sort_1Block_async_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
    }
    else{
        Sort_1Block_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort);
    }
}

void benmark512(){
    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / DefaultNbThreadsPerBlock;
    std::vector<int> h_array(nbElements);

    for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
        for(int idxSizePerWarp = 32 ; idxSizePerWarp <= DefaultNbThreadsPerBlock ; idxSizePerWarp += 32){
            for(int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
                {
                    std::cout << "idxInitData: " << idxInitData << " idxSizePerWarp: " << idxSizePerWarp << " idxLoop: " << idxLoop << std::endl;


                    if(idxInitData == RANDOM){
                        std::random_device rd;
                        std::mt19937 gen(0);
                        std::uniform_int_distribution<int> dis(0, 1000000);
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = dis(gen);
                        }
                    }
                    else if(idxInitData == SORTED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = i%32;
                        }
                    }
                    else if(idxInitData == REVERSED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = 31-i%32;
                        }
                    }
                    else{
                        assert(0);
                    }

                    // Allocate device memory for the array
                    int* d_array = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_array, h_array.size() * sizeof(int)));

                    // Copy data from host to device
                    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(int), cudaMemcpyHostToDevice));

                    // Define execution configuration (threads per block and blocks per grid)
                    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(idxSizePerWarp);
                    const int threadsPerBlock = NOrPow2;
                    const int blocksPerGrid = nbBlocks;

                    CUDA_CHECK(cudaDeviceSynchronize());
                    SpTimer timer;

                    benchmarkSort512<int, true, true, true><<<blocksPerGrid, threadsPerBlock, NOrPow2*sizeof(int) >>>(d_array, idxSizePerWarp);

                    CUDA_CHECK(cudaDeviceSynchronize());
                    timer.stop();

                    std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                    std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                    // Free device memory
                    CUDA_CHECK(cudaFree(d_array));
                }
            }
        }
    }
}


void benmarkSortLarge(){
    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    std::vector<int> h_array(nbElements);

    for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
        for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
            for(int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
                {
                    std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop << std::endl;


                    if(idxInitData == RANDOM){
                        std::random_device rd;
                        std::mt19937 gen(0);
                        std::uniform_int_distribution<int> dis(0, 1000000);
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = dis(gen);
                        }
                    }
                    else if(idxInitData == SORTED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = i%32;
                        }
                    }
                    else if(idxInitData == REVERSED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = 31-i%32;
                        }
                    }
                    else{
                        assert(0);
                    }

                    // Allocate device memory for the array
                    int* d_array = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_array, h_array.size() * sizeof(int)));

                    // Copy data from host to device
                    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(int), cudaMemcpyHostToDevice));

                    CUDA_CHECK(cudaDeviceSynchronize());
                    SpTimer timer;

                    Sort<int, true, true, true>(d_array, idxSize);

                    CUDA_CHECK(cudaDeviceSynchronize());
                    timer.stop();

                    std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;

                    // Free device memory
                    CUDA_CHECK(cudaFree(d_array));
                }
            }
        }
    }
}



void benchmarkAll(){
    //benmark32();
    //benmark512();
    benmarkSortLarge();
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

int main(){
    //testall();
    benchmarkAll();
    return 0;
}
