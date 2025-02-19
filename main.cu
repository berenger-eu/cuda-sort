#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>

template <typename T>
__device__ __forceinline__ bool IsSorted(const T* input, const int nbValuesInVec){
    __shared__ bool isSorted;
    bool localIsSorted = true;
    if(threadIdx.x == 0){
        isSorted = true;
    }
    else if(threadIdx.x < nbValuesInVec){
        if(input[threadIdx.x] > input[threadIdx.x-1]){
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

template <typename T>
__device__ __forceinline__ void CoreSmallSort(T* input, const int nbValuesInVec){
    if(IsSorted(input, nbValuesInVec)){
        return;
    }

    int ffttout = (threadIdx.x & 1) ;//svzip1_b32(svpfalse_b(),svptrue_b32());

    int vecincout = (ffttout?-1:1);//svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

//     {// stepout == 1
//         const int vecpermute = threadIdx.x + vecincout;//svadd_s32_z(svptrue_b32(), vecindex, vecincout);

//         //const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
// printf("threadIdx.x %d <-> vecpermute %d\n", threadIdx.x, vecpermute);
//         // input[threadIdx.x] = (ffttout ?  // svsel_s32(ffttout,
//         //                       min(input[threadIdx.x], input[vecpermute])  // svmax_s32_z(svptrue_b32(), input, inputPerm),
//         //                       : max(input[threadIdx.x], input[vecpermute]));// svmin_s32_z(svptrue_b32(), input, inputPerm));
//         __syncthreads();
//     }
    vecincout = 0;
    for(long int stepout = 1 ; stepout < nbValuesInVec ; stepout *= 2){
        ffttout = (threadIdx.x & stepout); // svzip1_b32(ffttout,ffttout);
        vecincout = (ffttout? // svsel_s32(ffttout,
                     vecincout - stepout // svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                     : vecincout + stepout); // svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
        {
            const int vecpermute = threadIdx.x + vecincout;// svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            // const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            printf("threadIdx.x %d <-> vecpermute %d\n", threadIdx.x, vecpermute);
            // input[threadIdx.x] = (ffttout ?// svsel_s32(ffttout,
            //                       min(input[threadIdx.x], input[vecpermute]) // svmax_s32_z(svptrue_b32(), input, inputPerm),
            //                       : max(input[threadIdx.x], input[vecpermute])); // svmin_s32_z(svptrue_b32(), input, inputPerm));
            __syncthreads();
        }

        int fftt = (threadIdx.x ^ (stepout/2)); //svuzp2_b32(ffttout,ffttout);

        int vecinc = (stepout/2); // svdup_s32(stepout/2);

        for(long int step = stepout/2 ; step >= 1 ; step/=2){
            const int vecpermute = (threadIdx.x + (fftt? -vecinc  : vecinc)); // svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc), vecinc));

            // const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            printf("threadIdx.x %d <-> vecpermute %d\n", threadIdx.x, vecpermute);
            // input[threadIdx.x] = (fftt ?// svsel_s32(fftt,
            //     min(input[threadIdx.x], input[vecpermute]) //svmax_s32_z(svptrue_b32(), input, inputPerm),
            //     : max(input[threadIdx.x], input[vecpermute])); // svmin_s32_z(svptrue_b32(), input, inputPerm));

            fftt = (threadIdx.x ^ (step/2)); // svuzp2_b32(fftt,fftt);
            vecinc = vecinc/2; // svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            __syncthreads();
            if(IsSorted(input, nbValuesInVec)){
                return;
            }
        }
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


int main() {
    // Size of the shared memory on the device, calling cuda api
    int MaxSharedMemSize;
    CUDA_CHECK(cudaDeviceGetAttribute(&MaxSharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    std::cout << "Shared memory size: " << MaxSharedMemSize << std::endl;

    // Number of elements to sort
    const int N = 8;//256;
    assert(N <= MaxSharedMemSize / sizeof(int));

    // Create a host vector and fill it with random values (using C++11 random)
    std::vector<int> h_array(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 10000);
    for (int i = 0; i < N; ++i) {
        h_array[i] = dis(gen);
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
    const int threadsPerBlock = N;//256;
    const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the sort kernel (template instantiated for int)
    const int sharedMemSize = N * sizeof(int);
    sort_dv<int><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the (supposedly sorted) array back from device to host
    std::vector<int> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_array));

    // Verify that the device result matches the sorted host array
    if (h_result == h_sorted) {
        std::cout << "The array is correctly sorted." << std::endl;
    } else {
        std::cout << "The sorting is incorrect!" << std::endl;
    }

    return 0;
}



