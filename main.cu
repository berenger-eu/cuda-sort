#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>

template <typename T>
__device__ __forceinline__ bool IsSorted(const T* input, const int nbValuesInVec){
    __shared__ bool isSorted;
    bool localIsSorted = true;
    if(threadIdx.x == 0){
        isSorted = true;
    }
    else if(threadIdx.x < nbValuesInVec){
        if(input[threadIdx.x] < input[threadIdx.x-1]){
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

template <typename T, bool TestIfSorted = false>
__device__ __forceinline__ void CoreSmallSort(T* input, const int nbValuesInVec){
    if constexpr (TestIfSorted){
        if(IsSorted(input, nbValuesInVec)){
            return;
        }
    }

    int vecincout = 0;
    {
        unsigned laneId;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
        const unsigned fullMask = 0xFFFFFFFF;

        T a = input[threadIdx.x];
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
        input[threadIdx.x] = a;
        __syncthreads();
    }

    if constexpr (TestIfSorted){
        if(IsSorted(input, nbValuesInVec)){
            return;
        }
    }

    for(long int stepout = 32 ; stepout < nbValuesInVec ; stepout *= 2){
        {
            const int ffttout = (threadIdx.x & stepout);
            vecincout = vecincout + (ffttout? -stepout : stepout);
            const int vecpermute = threadIdx.x + vecincout;
            const T a = input[threadIdx.x];
            const T b = input[vecpermute];
            const T best = (ffttout ? max(a, b) : min(a, b));
            __syncthreads();
            input[threadIdx.x] = best;
            __syncthreads();
        }

        for(long int step = stepout/2 ; step >= 1 ; step/=2){
            const int vecinc = step;
            const int fftt = (threadIdx.x & step);
            const int vecincout2 = (fftt? -vecinc  : vecinc);
            const int vecpermute = (threadIdx.x + vecincout2);
            const T a = input[threadIdx.x];
            const T b = input[vecpermute];
            const T best = (fftt ? max(a, b) : min(a, b));
            __syncthreads();
            input[threadIdx.x] = best;
            __syncthreads();
            if constexpr (TestIfSorted){
                if(IsSorted(input, nbValuesInVec)){
                    return;
                }
            }
        }
    }
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


// Template CUDA kernel to sort an array
template <typename T>
__global__ void sort_dv(T* d_arr, const int n) {
    // Declare shared memory ptr
    extern __shared__ T shared[];
    // Copy
    if(threadIdx.x < n){
        shared[threadIdx.x] = d_arr[threadIdx.x];
    }
    else{
        shared[threadIdx.x] = my_numeric_limits_max<T>();
    }
    __syncthreads();
    // Sort
    CoreSmallSort(shared, n);
    // Copy back
    if(threadIdx.x < n){
        d_arr[threadIdx.x] = shared[threadIdx.x];
    }
}

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int GreaterPowerOf2EqualOrLarger(int n){
    int res = 1;
    while(res < n){
        res *= 2;
    }
    return res;
}

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

        // Define execution configuration (threads per block and blocks per grid)
        const int threadsPerBlock = NOrPow2;
        const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the sort kernel (template instantiated for int)
        const int sharedMemSize = NOrPow2 * sizeof(int);
        sort_dv<int><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);

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
    test();
    return 0;
}
