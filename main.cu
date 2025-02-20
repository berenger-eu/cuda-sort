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
    assert(nbValuesToSort == 32);

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
__global__ void Core_Sort_1Warp_SM_lt32(T* dataToSort, const int nbValuesToSort){
    Core_Sort_1Warp_SM_lt32_d<T, FullSM>(dataToSort, nbValuesToSort);
}


template <typename T, bool TestIfSorted = false, bool FullSM = true>
__device__ __forceinline__ void Core_Sort_1Block_SM_gt32_d(T* dataToSort, const int nbValuesToSort){
    assert(nbValuesToSort == blockDim.x);

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

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void Core_Sort_1Block_SM_gt32(T* dataToSort, const int nbValuesToSort){
    // Declare shared memory ptr
    __shared__ T shared[32];
    // Copy
    if(threadIdx.x < nbValuesToSort){
        shared[threadIdx.x] = dataToSort[threadIdx.x];
    }
    else{
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }

    Core_Sort_1Warp_SM_lt32_d<T, TestIfSorted, FullSM>(shared, nbValuesToSort);

    // Copy back
    if(threadIdx.x < nbValuesToSort){
        dataToSort[threadIdx.x] = shared[threadIdx.x];
    }
}


// Template CUDA kernel to sort an array
template <typename T, bool TestIfSorted, bool FullSM>
__device__ void Sort_1Block_gt32_d(T* dataToSort, const int nbValuesToSort) {
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
    }
    localCounters[threadIdx.x] = 0;
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

    // Warp-Level Reduction using Shared Memory
    for(int step = blockDim.x / 2; step >= 32; step /= 2){
        if(threadIdx.x < step){
            localCounters[threadIdx.x] += localCounters[threadIdx.x + step];
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void CountParts_Grid_32_d(const T* data, const int size, const T* pivots, int* counters){
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
__global__ void CountParts_Grid_32(const T* data, const int size, const T* pivots, int* counters){
    CountParts_Grid_32_d(data, size, pivots, counters);
}


template <typename T>
__device__ void MoveElements_Grid_32_d(T* data, const int size, const T* pivots, int* counters, T* dataBuffer){
    // Nb threads must be power of 2
    assert(((blockDim.x - 1) & blockDim.x) == 0);
    assert(blockDim.x >= 32);

    __shared__ int localCounters[32];
    CountParts_Grid_32_core(data, size, pivots, counters, localCounters);

    __shared__ int localGOffset[32];
    if(threadIdx.x < 32){
        localGOffset[threadIdx.x] = atomicAdd(&counters[threadIdx.x], localCounters[threadIdx.x]);
    }

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

template <typename T>
__global__ void MoveElements_Grid_32(T* data, const int size, const T* pivots, int* counters, T* dataBuffer){
    MoveElements_Grid_32_d(data, size, pivots, counters, dataBuffer);
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

template <typename T, bool TestIfSorted = false, bool FullSM = true>
__global__ void Sort_core_buildPivots(T* dataToSort, const int nbValuesToSort, T* pivots){
    // We copy values from the array to build the partition
    assert(gridDim.x == 1);
    {
        __shared__ T localPivots[32];
        
        if(threadIdx.x < 31){
            const int step = nbValuesToSort/32;
            localPivots[threadIdx.x] = dataToSort[min(nbValuesToSort, step*threadIdx.x)];
        }
        if(threadIdx.x == 0){
            localPivots[0] = my_numeric_limits_min<T>();
        }
        // Not needed __syncwarps();
        Core_Sort_1Warp_SM_lt32_d<T, true>(localPivots, 32);
    
        // Not needed __syncwarps();
        if(threadIdx.x < 32){
            pivots[threadIdx.x] = localPivots[threadIdx.x];
        }
    }
}

template <typename T, bool TestIfSorted = false, bool FullSM = true>
__global__ void SortPartitions(T* dataToSort, const int nbValuesToSort, const int counterParts[], const int nbParts){
    for(int idxPart = blockIdx.x ; idxPart < nbParts; idxPart += gridDim.x){
        const int start = counterParts[idxPart];
        const int end = counterParts[idxPart+1];
        const int size = end-start;
        if(size <= 32){
            Core_Sort_1Warp_SM_lt32_d<T, FullSM>(dataToSort+start, size);
        }
        else if(size <= blockDim.x){
            Sort_1Block_async_gt32_d<T, TestIfSorted, FullSM>(dataToSort+start, size);
        }
    }
}


template <typename T, bool TestIfSorted = false, bool FullSM = true>
__host__ void Sort(T* dataToSort, const int nbValuesToSort){
    if(nbValuesToSort <= 32){
        Core_Sort_1Warp_SM_lt32<T, FullSM><<<1,32>>>(dataToSort, nbValuesToSort);
        return;
    }
    if(nbValuesToSort <= 1024){
        Sort_1Block_async_gt32<T, TestIfSorted, FullSM><<<1,1024>>>(dataToSort, nbValuesToSort);
        return;
    }

    int* dataBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dataBuffer, nbValuesToSort * sizeof(T)));

    int* pivots = nullptr;
    CUDA_CHECK(cudaMalloc(&pivots, 34 * sizeof(int)));
    CUDA_CHECK(cudaMemset(pivots, 0, 34 * sizeof(int)));

    Sort_core_buildPivots<T><<<1, 32>>>(dataToSort, nbValuesToSort, pivots);

    int* counterParts = nullptr;
    CUDA_CHECK(cudaMalloc(&counterParts, 34 * sizeof(int)));
    CUDA_CHECK(cudaMemset(counterParts, 0, 34 * sizeof(int)));

    CountParts_Grid_32<T><<< 512, 512 >>>(dataToSort, nbValuesToSort, pivots, counterParts+1);

    BuildPrefix_SM<T, int><<<1,32,32*sizeof(int)>>>(counterParts+1, 32);

    MoveElements_Grid_32<T><<< 512, 512 >>>(dataToSort, nbValuesToSort,
                                            pivots,
                                            counterParts,
                                            dataBuffer);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaMemcpy(dataBuffer, dataToSort, nbValuesToSort * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(dataBuffer));
    CUDA_CHECK(cudaFree(pivots));

    SortPartitions<T, TestIfSorted, FullSM><<< 32, 1024>>>(dataToSort, nbValuesToSort, counterParts, 32);

    int counterPartsHost[34];
    CUDA_CHECK(cudaMemcpy(counterPartsHost, counterParts, 34 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(counterParts));    
    
    for(int idxPart = 0 ; idxPart < 32 ; idxPart++){
        const int start = counterParts[idxPart];
        const int end = counterParts[idxPart+1];
        const int size = end-start;
        if(size > 1024){
            Sort<T, TestIfSorted, FullSM>(dataBuffer+start, size);
        }
    }
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
    // Limit the size to this value
    const int MaxSize = std::min(int(MaxSharedMemSize / sizeof(int)), MaxThreadsPerBlock);
    std::cout << "Max size: " << MaxSize << std::endl;

    for(int idxSize = 32 ; idxSize < MaxSize ; idxSize *= 2){
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
                    h_array[i] = N-i;//dis(gen);
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
}

void testfull() {
    // Size of the shared memory on the device, calling cuda api
    int MaxSharedMemSize;
    CUDA_CHECK(cudaDeviceGetAttribute(&MaxSharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    std::cout << "Shared memory size: " << MaxSharedMemSize << std::endl;
    // Max number of threads per block
    int MaxThreadsPerBlock;
    CUDA_CHECK(cudaDeviceGetAttribute(&MaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
    std::cout << "Max threads per block: " << MaxThreadsPerBlock << std::endl;
    // Limit the size to this value
    const int MaxSize = std::min(int(MaxSharedMemSize / sizeof(int)), MaxThreadsPerBlock);
    std::cout << "Max size: " << MaxSize << std::endl;

    for(int idxSize = 32 ; idxSize < MaxSize ; idxSize *= 2){
        // Number of elements to sort
        const int N = idxSize;
        assert(N <= MaxSharedMemSize / sizeof(int));
        const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
        assert(((NOrPow2-1) & NOrPow2) == 0);
        assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

        std::cout << "Testing with " << N << " elements" << std::endl;
        std::cout << " - Number of threads: " << NOrPow2 << std::endl;
        std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

        // Create a host vector and fill it with random values (using C++11 random)
        std::vector<int> h_array(N);
        std::random_device rd;
        std::mt19937 gen(0);
        std::uniform_int_distribution<int> dis(0, 10000);
        for (int i = 0; i < N; ++i) {
            h_array[i] = N-i;//dis(gen);
        }

        // Make a copy of the host array and sort it using std::sort for later verification.
        std::vector<int> h_sorted = h_array;
        std::sort(h_sorted.begin(), h_sorted.end());

        // Allocate device memory for the array
        int* d_array = nullptr;
        CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        Sort<int>(d_array, N);

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


int main() {
    if(true){
        test<true, true, true>();
        test<true, true, false>();
        test<true, false, true>();
        test<true, false, false>();
        test<false, true, true>();
        test<false, true, false>();
        test<false, false, true>();
        test<false, false, false>();
    }
    //testfull();
    return 0;
}
