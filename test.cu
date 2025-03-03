#include "cusort.hpp"

#include "valgen.hpp"

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

    CuRandManager<int> randManager;

    for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 2){
        for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
            // Number of elements to sort
            const int N = idxSize;
            assert(N <= MaxSharedMemSize / sizeof(int));
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);
            assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

            std::cout << "Testing with " << N << " elements" << std::endl;
            std::cout << " - Number of threads: " << NOrPow2 << std::endl;
            std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            randManager.initArray(d_array, idxSize, idxSize, idxInitData);

            std::vector<int> h_sorted(N);
            CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(h_sorted.begin(), h_sorted.end());

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

    for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 2){
        for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
            // Number of elements to sort
            const int N = idxSize;
            assert(N <= MaxSharedMemSize / sizeof(int));
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);
            assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

            std::cout << "Testing with " << N << " elements" << std::endl;
            std::cout << " - Number of threads: " << NOrPow2 << std::endl;
            std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
            std::vector<int> h_sorted(N);
            CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(h_sorted.begin(), h_sorted.end());

            // Define execution configuration (threads per block and blocks per grid)
            const int threadsPerBlock = NOrPow2;
            const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

            // Launch the sort kernel (template instantiated for int)
            const int sharedMemSize = 2 * NOrPow2 * sizeof(int);

            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_extsm_v2<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }
            else{
                Sort_1Block_gt32_extsm_v2<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
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
        for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
            // Number of elements to sort
            const int N = idxSize;
            assert(N <= MaxSharedMemSize / sizeof(int));
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);
            assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

            std::cout << "Testing with " << N << " elements" << std::endl;
            std::cout << " - Number of threads: " << NOrPow2 << std::endl;
            std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
            std::vector<int> h_sorted(N);
            CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(h_sorted.begin(), h_sorted.end());

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
        for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
            // Number of elements to sort
            const int N = idxSize;
            assert(N <= MaxSharedMemSize / sizeof(int));
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);
            assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

            std::cout << "Testing with " << N << " elements" << std::endl;
            std::cout << " - Number of threads: " << NOrPow2 << std::endl;
            std::cout << " - Shared memory size: " << NOrPow2 * sizeof(int) << std::endl;

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
            std::vector<int> h_sorted(N);
            CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(h_sorted.begin(), h_sorted.end());

            // Define execution configuration (threads per block and blocks per grid)
            const int threadsPerBlock = NOrPow2;
            const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

            // Launch the sort kernel (template instantiated for int)
            const int sharedMemSize = 2 * NOrPow2 * sizeof(int);

            if constexpr(ASyncCpyBito){
                Sort_1Block_async_gt32_extsm_v2<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
            }
            else{
                Sort_1Block_gt32_extsm_v2<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N);
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
        for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
            // Number of elements to sort
            const int N = idxSize;

            std::cout << "Testing with " << N << " elements" << std::endl;

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
            std::vector<int> h_sorted(N);
            CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(h_sorted.begin(), h_sorted.end());

            Sort_1Warp_SM_le32<int, TestIfSorted, FullSM><<<1,32>>>(d_array, N);

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
void testV3() {
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
    const int MaxSize = int(MaxSharedMemSize / sizeof(int) / 2);
    std::cout << "Max size: " << MaxSize << std::endl;

    CuRandManager<int> randManager;

    for(int nbThreads = 32 ; nbThreads <= MaxThreadsPerBlock ; nbThreads += 32){
        for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 2){
            for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                // Number of elements to sort
                const int N = idxSize;
                assert(N <= MaxSharedMemSize / sizeof(int));
                const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
                assert(((NOrPow2-1) & NOrPow2) == 0);
                assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

                std::cout << "Testing with " << N << " elements" << std::endl;
                std::cout << " - SizePow2: " << NOrPow2 << std::endl;
                std::cout << " - Shared memory size: " << 2 * NOrPow2 * sizeof(int) << std::endl;

                // Allocate device memory for the array
                int* d_array = nullptr;
                CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

                randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
                std::vector<int> h_sorted(N);
                CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
                std::sort(h_sorted.begin(), h_sorted.end());

                // Define execution configuration (threads per block and blocks per grid)
                const int threadsPerBlock = nbThreads;
                const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

                // Launch the sort kernel (template instantiated for int)
                const int sharedMemSize = 2 * NOrPow2 * sizeof(int);

                std::cout << " - threadsPerBlock: " << threadsPerBlock << std::endl;

                if constexpr(ASyncCpyBito){
                    Sort_1Block_async_gt32_extsm_v3<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N, NOrPow2);
                }
                else{
                    Sort_1Block_gt32_extsm_v3<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N, NOrPow2);
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

    for(int nbThreads = 32 ; nbThreads <= MaxThreadsPerBlock ; nbThreads += 32){
        for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 3){
            for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                // Number of elements to sort
                const int N = idxSize;
                assert(N <= MaxSharedMemSize / sizeof(int));
                const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
                assert(((NOrPow2-1) & NOrPow2) == 0);
                assert(NOrPow2 <= MaxSharedMemSize / sizeof(int));

                std::cout << "Testing with " << N << " elements" << std::endl;
                std::cout << " - SizePow2: " << NOrPow2 << std::endl;
                std::cout << " - Shared memory size: " << 2 * NOrPow2 * sizeof(int) << std::endl;

                // Allocate device memory for the array
                int* d_array = nullptr;
                CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

                randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
                std::vector<int> h_sorted(N);
                CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
                std::sort(h_sorted.begin(), h_sorted.end());

                // Define execution configuration (threads per block and blocks per grid)
                const int threadsPerBlock = nbThreads;
                const int blocksPerGrid = 1;//(N + threadsPerBlock - 1) / threadsPerBlock;

                // Launch the sort kernel (template instantiated for int)
                const int sharedMemSize = 2 * NOrPow2 * sizeof(int);

                std::cout << " - threadsPerBlock: " << threadsPerBlock << std::endl;

                if constexpr(ASyncCpyBito){
                    Sort_1Block_async_gt32_extsm_v3<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N, NOrPow2);
                }
                else{
                    Sort_1Block_gt32_extsm_v3<int, TestIfSorted, FullSM><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N, NOrPow2);
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
}


template<bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
void testLarge() {
    // Limit the size to this value
    const int MaxSize = (1 << 20);
    std::cout << "Max size: " << MaxSize << std::endl;

    CuRandManager<int> randManager;

    for(int idxSize = 32 ; idxSize < MaxSize ; idxSize *= 2){
        for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
            // Number of elements to sort
            const int N = idxSize;
            const int NOrPow2 = GreaterPowerOf2EqualOrLarger(N);
            assert(((NOrPow2-1) & NOrPow2) == 0);

            std::cout << "Testing with " << N << " elements" << std::endl;

            // Allocate device memory for the array
            int* d_array = nullptr;
            CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(int)));

            randManager.initArray(d_array, idxSize, idxSize, idxInitData);
            
            std::vector<int> h_sorted(N);
            CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost));
            std::sort(h_sorted.begin(), h_sorted.end());

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


int main() {
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
        testV3<true, true, true>();
        testV3<true, true, false>();
        testV3<true, false, true>();
        testV3<true, false, false>();
        testV3<false, true, true>();
        testV3<false, true, false>();
        testV3<false, false, true>();
        testV3<false, false, false>();
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
    return 0;
}
