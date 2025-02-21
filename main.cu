#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>
#include <cuda/pipeline>
#include <cooperative_groups.h>

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

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

template <typename T>
__device__ __forceinline__ bool TestIfSorted_1Block(const T* dataToSort, const int nbValuesToSort){
    __shared__ bool isSorted;
    bool localIsSorted = true;
    if(threadIdx.x == 0){
        isSorted = true;
    }
    else if(threadIdx.x < nbValuesToSort){
        if(dataToSort[threadIdx.x] < dataToSort[threadIdx.x-1]){
            localIsSorted = false;
        }        
    }
    __syncthreads();
    if(!localIsSorted){
        isSorted = false;
    }
    __syncthreads();
    return isSorted;
}

template <typename T>
__device__ __forceinline__ T min(const T a, const T b){
    return a < b ? a : b;
}

template <typename T>
__device__ __forceinline__ T max(const T a, const T b){
    return a > b ? a : b;
}

template <typename T, bool FullSM = true>
__device__ __forceinline__ void Core_Sort_1Warp_SM_lt32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort <= 32);
    assert(32 <= blockDim.x);

    if(32 <= threadIdx.x){
        return;
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
}

template <typename T, bool FullSM = true>
__device__ void Sort_1Warp_SM_lt32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort <= 32);
    __shared__ T shared[32];
    // Copy
    if(threadIdx.x < nbValuesToSort){
        shared[threadIdx.x] = dataToSort[threadIdx.x];
    }
    else{
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }
    // Not needed __syncthreads();
    // Sort
    Core_Sort_1Warp_SM_lt32_d<T, FullSM>(shared, nbValuesToSort);
    // Copy back
    if(threadIdx.x < nbValuesToSort){
        dataToSort[threadIdx.x] = shared[threadIdx.x];
    }
}

template <typename T, bool FullSM = true>
__global__ void Sort_1Warp_SM_lt32(T* dataToSort, const int nbValuesToSort){
    Sort_1Warp_SM_lt32_d(dataToSort, nbValuesToSort);
}

template <typename T, bool TestIfSorted = false, bool FullSM = true>
__device__ __forceinline__ void Core_Sort_1Block_SM_gt32_d(T* dataToSort, const int nbValuesToSort){
    // Not neccessarly now, assert(nbValuesToSort == blockDim.x);
    assert(((blockDim.x-1) & blockDim.x) == 0);

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
__device__ void Sort_1Block_gt32_d(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ T shared[];
    assert(nbValuesToSort <= blockDim.x);
    assert(((blockDim.x-1) & blockDim.x) == 0);
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
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_gt32(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_gt32_d<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}

// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_async_gt32_d(T* dataToSort, const int nbValuesToSort) {
    // Declare shared memory ptr
    extern __shared__ T shared_data[];
    assert(nbValuesToSort <= blockDim.x);
    assert(((blockDim.x-1) & blockDim.x) == 0);

    auto block = cooperative_groups::this_thread_block();
    // Declare the barrier in shared memory.
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // Initialize the barrier using a friend function (or an appropriate initializer).
    if (threadIdx.x == 0) {
        init(&bar, block.size());
    }
    block.sync(); // Ensure all threads see the initialized barrier

    // Asynchronous copy from global to shared memory.
    if (threadIdx.x == 0) {
        cuda::memcpy_async(shared_data, dataToSort, sizeof(T)*nbValuesToSort, bar);
    } else if (threadIdx.x >= nbValuesToSort) {
        shared_data[threadIdx.x] = my_numeric_limits_max<T>();
    }
    bar.arrive_and_wait();

    // Sort
    Core_Sort_1Block_SM_gt32_d<T, TestIfSorted, FullSM>(shared_data, nbValuesToSort);
    // Copy back
    if(threadIdx.x == 0){
        cuda::memcpy_async(dataToSort, shared_data, sizeof(T)*nbValuesToSort, bar);
    }
    bar.arrive_and_wait();
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Sort_1Block_async_gt32(T* dataToSort, const int nbValuesToSort) {
    Sort_1Block_async_gt32_d<T, TestIfSorted, FullSM>(dataToSort, nbValuesToSort);
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


template <typename T>
__forceinline__ __device__ int binarySearch(const T pivots[], const int nbPivots, const T value) {
    int start = 0, end = nbPivots; // Use actual size
    while (start < end - 1) {
        int mid = (start + end) / 2;
        if (value < pivots[mid]) {
            end = mid;
        } else {
            start = mid;
        }
    }
    return start;  // `start` now gives the correct interval index
}


template <typename T>
__device__ void CountParts_Grid_32_core(const T* data, const int size, const T pivots[32], int* counters,
                                                   int localCounters[32]){
    __shared__ T localPivots[32];

    if(threadIdx.x < 32){
        localPivots[threadIdx.x] = pivots[threadIdx.x];
        localCounters[threadIdx.x] = 0;
    }
    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;
    
    for(int idxValue = id; idxValue < size; idxValue += numThreads){
        const T value = data[idxValue];

        // Binary search on pivots
        const int bucket = binarySearch(localPivots, 32, value);
        assert(0 <= bucket && bucket < 32);
        atomicAdd(&localCounters[bucket], 1);
    }

    __syncthreads();
}


template <typename T>
__global__ void CountParts_Grid_32(const T* data, const int size, const T* pivots, int* counters){
    // Nb threads must be power of 2
    assert(((blockDim.x - 1) & blockDim.x) == 0);
    assert(blockDim.x >= 32);

    __shared__ int localCounters[32];
    CountParts_Grid_32_core(data, size, pivots, counters, localCounters);

    if(threadIdx.x < 32){
        atomicAdd(&counters[threadIdx.x], localCounters[threadIdx.x]);
    }
}

template <typename T>
__global__ void MoveElements_Grid_32(T* data, const int size, const T* pivots, int* counters, T* dataBuffer){
    // Nb threads must be power of 2
    assert(((blockDim.x - 1) & blockDim.x) == 0);
    assert(blockDim.x >= 32);

    __shared__ int localCounters[32];
    CountParts_Grid_32_core(data, size, pivots, counters, localCounters);
    __syncthreads();

    __shared__ int localGOffset[32];
    if(threadIdx.x < 32){
        localGOffset[threadIdx.x] = atomicAdd(&counters[threadIdx.x], localCounters[threadIdx.x]);
    }

    __syncthreads();

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;
    
    for(int idxValue = id; idxValue < size; idxValue += numThreads){
        const T value = data[idxValue];

        // Binary search on pivots
        const int bucket = binarySearch(pivots, 32, value);
        assert(0 <= bucket && bucket < 32);
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

template <typename NumType, typename IndexType>
__global__ void BuildPrefix_SM( NumType* prefixTree, const IndexType inSize){
    extern __shared__ IndexType buffer[];

    for(IndexType idx = threadIdx.x ; idx < inSize ; idx += blockDim.x){
        buffer[idx] = prefixTree[idx];
    }

    __syncthreads();

    BuildPrefix_SM_device(buffer, inSize);

    for(IndexType idx = threadIdx.x ; idx < inSize ; idx += blockDim.x){
        prefixTree[idx] = buffer[idx];
    }
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void Sort_core_buildPivots(T* dataToSort, const int nbValuesToSort, T* pivots){
    // We copy values from the array to build the partition
    assert(gridDim.x == 1);
    {
        __shared__ T localPivots[32];
        
        if(threadIdx.x < 32){
            const int step = nbValuesToSort/33;
            localPivots[threadIdx.x] = dataToSort[min(nbValuesToSort-1, step*threadIdx.x)];
        }
        if(threadIdx.x == 0){
            localPivots[0] = my_numeric_limits_min<T>();
        }
        __syncwarp(0xffffffff);
        Core_Sort_1Warp_SM_lt32_d<T, true>(localPivots, 32);

        // Not needed __syncthreads();
    
        __syncwarp(0xffffffff);
        if(threadIdx.x < 32){
            pivots[threadIdx.x] = localPivots[threadIdx.x];
        }
    }
}

template <typename T, bool TestIfSorted, bool FullSM, bool async>
__global__ void SortPartitions(T* dataToSort, const int nbValuesToSort, const int counterParts[], const int nbParts){
    for(int idxPart = blockIdx.x ; idxPart < nbParts; idxPart += gridDim.x){
        const int start = counterParts[idxPart];
        const int end = counterParts[idxPart+1];
        const int size = end-start;
        if(size <= 32){
            Sort_1Warp_SM_lt32_d<T, FullSM>(dataToSort+start, size);
        }
        else if(size <= blockDim.x){
            if constexpr(async){
                Sort_1Block_async_gt32_d<T, TestIfSorted, FullSM>(dataToSort+start, size);
            }
            else{
                Sort_1Block_gt32_d<T, TestIfSorted, FullSM>(dataToSort+start, size);
            }
        }
    }
}


template <typename T, bool TestIfSorted, bool FullSM, bool async>
__host__ void Sort(T* dataToSort, const int nbValuesToSort){
    T* cpDataToSort = dataToSort;
    CUDA_CHECK(cudaMalloc(&dataToSort, nbValuesToSort * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dataToSort, cpDataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));

    std::cout << "===========================================" << std::endl;
    std::cout << " Sort " << nbValuesToSort << std::endl;

    std::vector<T> allDataToSortHost(nbValuesToSort);// TODO
    CUDA_CHECK(cudaMemcpy(allDataToSortHost.data(), dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));
    std::sort(allDataToSortHost.begin(), allDataToSortHost.end());

    if(nbValuesToSort <= 32){
        Sort_1Warp_SM_lt32<T, FullSM><<<1,32>>>(dataToSort, nbValuesToSort);

        {// TODO
            std::vector<T> dataSortHost(nbValuesToSort);
            CUDA_CHECK(cudaMemcpy(dataSortHost.data(), dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));
            for(int i = 0 ; i < nbValuesToSort ; i++){
                if(dataSortHost[i] != allDataToSortHost[i]){
                    std::cout << "Not equal" << std::endl;
                    assert(0);
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(cpDataToSort, dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));// TODO
        CUDA_CHECK(cudaFree(dataToSort));// TODO

        return;
    }

    int MaxThreadsPerBlock = 512;
    // TODO CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(nbValuesToSort);
    if(NOrPow2 <= MaxThreadsPerBlock){
        if constexpr(async){
            Sort_1Block_async_gt32<T, TestIfSorted, FullSM><<<1,NOrPow2,NOrPow2*sizeof(T)>>>(dataToSort, nbValuesToSort);
        }
        else{
            Sort_1Block_gt32<T, TestIfSorted, FullSM><<<1,NOrPow2,NOrPow2*sizeof(T)>>>(dataToSort, nbValuesToSort);
        }

        {// TODO
            std::vector<T> dataSortHost(nbValuesToSort);
            CUDA_CHECK(cudaMemcpy(dataSortHost.data(), dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));
            for(int i = 0 ; i < nbValuesToSort ; i++){
                if(dataSortHost[i] != allDataToSortHost[i]){
                    std::cout << "Not equal" << std::endl;
                    assert(0);
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(cpDataToSort, dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));// TODO
        CUDA_CHECK(cudaFree(dataToSort));// TODO

        return;
    }

    int* dataBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dataBuffer, nbValuesToSort * sizeof(T)));

    int* pivots = nullptr;
    CUDA_CHECK(cudaMalloc(&pivots, 32 * sizeof(int)));
    CUDA_CHECK(cudaMemset(pivots, 0, 32 * sizeof(int)));

    Sort_core_buildPivots<T><<<1, 32>>>(dataToSort, nbValuesToSort, pivots);

        {// TODO debug
    CUDA_CHECK(cudaDeviceSynchronize());
            int pivotsHost[32];
            CUDA_CHECK(cudaMemcpy(pivotsHost, pivots, 32 * sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "pivotsHost: ";
            for(int i = 0 ; i < 32 ; i++){
                std::cout << pivotsHost[i] << " ";
            }
            std::cout << std::endl;
            
        }

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMalloc(&counterParts, 34 * sizeof(int)));
    CUDA_CHECK(cudaMemset(counterParts, 0, 34 * sizeof(int)));

    CountParts_Grid_32<T><<< 512, MaxThreadsPerBlock >>>(dataToSort, nbValuesToSort, pivots, counterParts+2);

        {// TODO debug
    CUDA_CHECK(cudaDeviceSynchronize());
            int counterPartsHost[34];
            CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 34 * sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "  A CounterParts: ";
            int sum = 0;
            for(int i = 0 ; i < 34 ; i++){
                std::cout << "[" << i << "]" << counterPartsHost[i] << " ";
                sum += counterPartsHost[i];
            }
            std::cout << std::endl;
            assert(sum == nbValuesToSort);
        }

    BuildPrefix_SM<T, int><<<1,32,32*sizeof(int)>>>(counterParts+2, 32);

        {// TODO debug
    CUDA_CHECK(cudaDeviceSynchronize());
            int counterPartsHost[34];
            CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 34 * sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "  B CounterParts: ";
            int sum = 0;
            for(int i = 0 ; i < 34 ; i++){
                std::cout << "[" << i << "]" << counterPartsHost[i] << " ";
                sum += counterPartsHost[i];
            }
            std::cout << std::endl;
            assert(counterPartsHost[33] == nbValuesToSort);
        }

    MoveElements_Grid_32<T><<< 512, MaxThreadsPerBlock >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts+1,
                                            dataBuffer);

        {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<T> dataToSortHost(nbValuesToSort);
            CUDA_CHECK(cudaMemcpy(dataToSortHost.data(), dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));

            std::sort(dataToSortHost.begin(), dataToSortHost.end());

            if(dataToSortHost != allDataToSortHost){
                std::cout << "Vectors not equal" << std::endl;
                for(int i = 0 ; i < nbValuesToSort ; i++){
                    if(dataToSortHost[i] != allDataToSortHost[i]){
                        std::cout << "dataToSortHost[" << i << "]=" << dataToSortHost[i] << " != allDataToSortHost[" << i << "]=" << allDataToSortHost[i] << std::endl;
                    }
                }
                assert(0);
            }
        }

        {// Check values in partition
    CUDA_CHECK(cudaDeviceSynchronize());
            int counterPartsHost[34];
            CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 34 * sizeof(int), cudaMemcpyDeviceToHost));

            int pivotsHost[32];
            CUDA_CHECK(cudaMemcpy(pivotsHost, pivots, 32 * sizeof(int), cudaMemcpyDeviceToHost));

            std::vector<T> dataToSortHost(nbValuesToSort);
            CUDA_CHECK(cudaMemcpy(dataToSortHost.data(), dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));

            for(int idx = 0 ; idx < 32 ; ++idx){
                const int start = counterPartsHost[idx];
                const int end = counterPartsHost[idx+1];
                const int size = end-start;
                for(int idxval = start ; idxval < end ; ++idxval){
                    if(!(pivotsHost[idx] <= dataToSortHost[idxval] && 
                            (idx == 31 || dataToSortHost[idxval] < pivotsHost[idx+1]))){
                        std::cout << "out of bound" << std::endl;
                        assert(0);
                    }
                }
            }
        }

        {
            CUDA_CHECK(cudaDeviceSynchronize());
            int counterPartsHost[34];
            CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 34 * sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "  C CounterParts: ";
            for(int i = 0 ; i < 34 ; i++){
                std::cout << "[" << i << "]" << counterPartsHost[i] << " ";
            }
            std::cout << std::endl;
            assert(counterPartsHost[32] == nbValuesToSort);
    
            int host_pivots[32];
            CUDA_CHECK(cudaMemcpy(host_pivots, pivots, 32 * sizeof(int), cudaMemcpyDeviceToHost));
    
            std::vector<T> dataToSortHost(nbValuesToSort);
            CUDA_CHECK(cudaMemcpy(dataToSortHost.data(), dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));
            int idxPartition = 0;
            for(int idxData = 0 ; idxData < nbValuesToSort ; idxData++){
                while(idxData == counterPartsHost[idxPartition+1]){
                    idxPartition++;
                }
                // std::cout << "data " <<  idxData << " ";
                // std::cout << "idxPartition " <<  idxPartition << " ";
                // std::cout << "counterPartsHost[idxPartition] " <<  counterPartsHost[idxPartition] << " ";
                // std::cout << "counterPartsHost[idxPartition+1] " <<  counterPartsHost[idxPartition+1] << " ";
                // std::cout << " dataToSortHost " << dataToSortHost[idxData] << " ";
                // std::cout <<  " host_pivots " << host_pivots[idxPartition] << std::endl;
                // if(idxPartition != 31) std::cout << host_pivots[idxPartition+1] << std::endl;
                assert(host_pivots[idxPartition] <= dataToSortHost[idxData]);
                assert(idxPartition == 31 || dataToSortHost[idxData] <= host_pivots[idxPartition+1]);
            }
        }

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dataToSort, dataBuffer, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(dataBuffer));
    CUDA_CHECK(cudaFree(pivots));

    SortPartitions<T, TestIfSorted, FullSM, async><<< 32, MaxThreadsPerBlock, MaxThreadsPerBlock*sizeof(T)>>>(dataToSort, nbValuesToSort, counterParts, 32);

    {// TODO
    CUDA_CHECK(cudaDeviceSynchronize());
        int counterPartsHost[33];
        CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 33 * sizeof(int), cudaMemcpyDeviceToHost));

        for(int idxPart = 0 ; idxPart < 32 ; idxPart++){
            const int start = counterPartsHost[idxPart];
            const int end = counterPartsHost[idxPart+1];
            const int size = end-start;
            if(size <= MaxThreadsPerBlock){
                std::vector<T> dataToSortHost(size);
                CUDA_CHECK(cudaMemcpy(dataToSortHost.data(), dataToSort+start, size * sizeof(T), cudaMemcpyDeviceToHost));

                // Check it is sorted
                for(int i = 0 ; i < size ; i++){
                    if(i && dataToSortHost[i] < dataToSortHost[i-1]){
                        std::cout << "Not sorted" << std::endl;
                        assert(0);
                    }
                    if(allDataToSortHost[start+i] != dataToSortHost[i]){
                        std::cout << "Not equal" << std::endl;
                        assert(0);
                    }
                }
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    int counterPartsHost[33];
    CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 33 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(counterParts));    
    
    for(int idxPart = 0 ; idxPart < 32 ; idxPart++){
        const int start = counterPartsHost[idxPart];
        const int end = counterPartsHost[idxPart+1];
        const int size = end-start;
        if(size > MaxThreadsPerBlock){
            std::cout << "Run Sort start " << start  << " for " << size << " elements" << std::endl;
            Sort<T, TestIfSorted, FullSM, async>(dataToSort+start, size);
            cudaDeviceSynchronize();

            {// TODO
                std::vector<T> dataToSortHost(size);
                CUDA_CHECK(cudaMemcpy(dataToSortHost.data(), dataToSort+start, size * sizeof(T), cudaMemcpyDeviceToHost));

                // Check it is sorted
                for(int i = 0 ; i < size ; i++){
                    if(i && dataToSortHost[i] < dataToSortHost[i-1]){
                        std::cout << "Not sorted" << std::endl;
                        assert(0);
                    }
                    if(allDataToSortHost[start+i] != dataToSortHost[i]){
                        std::cout << "Not equal" << std::endl;
                        assert(0);
                    }
                }
            }
        }
    }

    {// TODO
        std::vector<T> dataToSortHost(nbValuesToSort);
        CUDA_CHECK(cudaMemcpy(dataToSortHost.data(), dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToHost));

        // Check it is sorted
        for(int i = 0 ; i < nbValuesToSort ; i++){
            if(i && dataToSortHost[i] < dataToSortHost[i-1]){
                std::cout << "Not sorted" << std::endl;
                assert(0);
            }
            if(allDataToSortHost[i] != dataToSortHost[i]){
                std::cout << "Not equal" << std::endl;
                assert(0);
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(cpDataToSort, dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));// TODO
    CUDA_CHECK(cudaFree(dataToSort));// TODO
    CUDA_CHECK(cudaDeviceSynchronize());
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template<bool TestIfSorted, bool FullSM, bool async>
void test() {
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

            if constexpr(async){
                Sort_1Block_async_gt32<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }
            else{
                Sort_1Block_gt32<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
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

            if constexpr(async){
                Sort_1Block_async_gt32<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }
            else{
                Sort_1Block_gt32<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
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

            Sort_1Warp_SM_lt32<int, FullSM><<<1,32>>>(d_array, N);

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

template<bool TestIfSorted, bool FullSM, bool async>
void testfull() {
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

            Sort<int, TestIfSorted, FullSM, async>(d_array, N);

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


int main() {
    if(false){
        test<true, true, true>();
        test<true, true, false>();
        test<true, false, true>();
        test<true, false, false>();
        test<false, true, true>();
        test<false, true, false>();
        test<false, false, true>();
        test<false, false, false>();
        return 0;
    }
    if(true){
        //testfull<true, true, true>();
        // testfull<true, true, false>();
        // testfull<true, false, true>();
        // testfull<true, false, false>();
        // testfull<false, true, true>();
        // testfull<false, true, false>();
        // testfull<false, false, true>();
        testfull<false, false, false>();
    }
    return 0;
}
