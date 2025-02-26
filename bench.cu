#include "cusort.hpp"

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

#include <fstream>

enum InitData{
    SORTED,
    REVERSED,
    RANDOM,
    NORMAL,
    NB_INITS
};

std::string InitDataString(int inInitData){
    switch(inInitData){
        case SORTED:
            return "SORTED";
        case REVERSED:
            return "REVERSED";
        case RANDOM:
            return "RANDOM";
        case NORMAL:
            return "NORMAL";
        default:
            return "UNKNOWN";
    }
}

template <bool TBoo>
struct ConstexprBool{
    static constexpr bool value = TBoo;
};

template <typename Func>
constexpr void for_bool(Func&& func) {
    func(ConstexprBool<true>());
    func(ConstexprBool<false>());
}

template <typename T, bool TestIfSorted, bool FullSM>
__global__ void benchmarkSortSmall(T* dataToSort, const int nbValuesToSort) {
    Sort_1Warp_SM_le32_d<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*32, nbValuesToSort);
}


void benmark32(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_32.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    filecsv << "#testsorted,fullsm,datainit,size,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / 32;
    std::vector<int> h_array(nbElements);

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
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
                            else if(idxInitData == NORMAL){
                                std::random_device rd;
                                std::mt19937 gen(0);
                                std::normal_distribution<double> dis(0, 1 + idxSizePerWarp/4);
                                for (int i = 0; i < int(h_array.size()); ++i) {
                                    h_array[i] = std::max(0, std::min( idxSizePerWarp,int(dis(gen)+idxSizePerWarp/2)));
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

                            benchmarkSortSmall<int, TestSorted, FullSM><<<blocksPerGrid, threadsPerBlock>>>(d_array, idxSizePerWarp);

                            CUDA_CHECK(cudaDeviceSynchronize());
                            timer.stop();

                            std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                            std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                            // Free device memory
                            CUDA_CHECK(cudaFree(d_array));

                            filecsv << (TestSorted ? "TRUE":"FALSE") << ","
                                    << (FullSM?"TRUE":"FALSE") << "," 
                                    << InitDataString(idxInitData) << "," 
                                    << idxSizePerWarp << "," 
                                    << timer.getElapsed() << ","
                                    << timer.getElapsed()/nbBlocks << ","
                                    << timer.getElapsed()/nbBlocks/double(idxSizePerWarp*std::log2(idxSizePerWarp)) << std::endl;
                        }
                    }
                }
            }
        });
    });
}


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void benchmarkSort512(T* dataToSort, const int nbValuesToSort) {
    if constexpr(ASyncCpyBito){
        Sort_1Block_async_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort);
    }
    else{
        Sort_1Block_gt32_d_extsm<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort);
    }
}

void benmark512(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_512.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / DefaultNbThreadsPerBlock;
    std::vector<int> h_array(nbElements);
    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
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
                                else if(idxInitData == NORMAL){
                                    std::random_device rd;
                                    std::mt19937 gen(0);
                                    std::normal_distribution<double> dis(0, 1 + idxSizePerWarp/4);
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = std::max(0, std::min( idxSizePerWarp,int(dis(gen)+idxSizePerWarp/2)));
                                    }
                                }
                                else if(idxInitData == SORTED){
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = i%idxSizePerWarp;
                                    }
                                }
                                else if(idxInitData == REVERSED){
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = (idxSizePerWarp-1)-i%idxSizePerWarp;
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

                                benchmarkSort512<int, TestSorted, FullSM, AsynCpy><<<blocksPerGrid, threadsPerBlock, NOrPow2*sizeof(int) >>>(d_array, idxSizePerWarp);

                                CUDA_CHECK(cudaDeviceSynchronize());
                                timer.stop();

                                std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                                std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                                // Free device memory
                                CUDA_CHECK(cudaFree(d_array));

                                filecsv << (TestSorted?"TRUE":"FALSE") << "," 
                                        << (FullSM?"TRUE":"FALSE") << "," 
                                        << (AsynCpy?"TRUE":"FALSE") << "," 
                                        << InitDataString(idxInitData) << "," 
                                        << idxSizePerWarp << ","
                                        <<  timer.getElapsed() << ","
                                        << timer.getElapsed()/nbBlocks << "," 
                                        << timer.getElapsed()/nbBlocks/double(idxSizePerWarp*std::log2(idxSizePerWarp)) << std::endl;
                            }
                        }
                    }
                }
            });
        });
    });
}


template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void benchmarkSort512_v2(T* dataToSort, const int nbValuesToSort) {
    if constexpr(ASyncCpyBito){
        Sort_1Block_async_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort);
    }
    else{
        Sort_1Block_gt32_d_extsm_v2<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort);
    }
}

void benmark512_v2(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_512_v2.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / DefaultNbThreadsPerBlock;
    std::vector<int> h_array(nbElements);
    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
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
                                else if(idxInitData == NORMAL){
                                    std::random_device rd;
                                    std::mt19937 gen(0);
                                    std::normal_distribution<double> dis(0, 1 + idxSizePerWarp/4);
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = std::max(0, std::min( idxSizePerWarp,int(dis(gen)+idxSizePerWarp/2)));
                                    }
                                }
                                else if(idxInitData == SORTED){
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = i%idxSizePerWarp;
                                    }
                                }
                                else if(idxInitData == REVERSED){
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = (idxSizePerWarp-1)-i%idxSizePerWarp;
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

                                benchmarkSort512_v2<int, TestSorted, FullSM, AsynCpy><<<blocksPerGrid, threadsPerBlock, 2*NOrPow2*sizeof(int) >>>(d_array, idxSizePerWarp);

                                CUDA_CHECK(cudaDeviceSynchronize());
                                timer.stop();

                                std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                                std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                                // Free device memory
                                CUDA_CHECK(cudaFree(d_array));

                                filecsv << (TestSorted?"TRUE":"FALSE") << "," 
                                        << (FullSM?"TRUE":"FALSE") << "," 
                                        << (AsynCpy?"TRUE":"FALSE") << "," 
                                        << InitDataString(idxInitData) << "," 
                                        << idxSizePerWarp << ","
                                        <<  timer.getElapsed() << ","
                                        << timer.getElapsed()/nbBlocks << "," 
                                        << timer.getElapsed()/nbBlocks/double(idxSizePerWarp*std::log2(idxSizePerWarp)) << std::endl;
                            }
                        }
                    }
                }
            });
        });
    });
}



template <typename T, bool TestIfSorted, bool FullSM, bool ASyncCpyBito>
__global__ void benchmarkSort512_v3(T* dataToSort, const int nbValuesToSort, const int nbWorkingValues) {
    if constexpr(ASyncCpyBito){
        Sort_1Block_async_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort, nbWorkingValues);
    }
    else{
        Sort_1Block_gt32_d_extsm_v3<T, TestIfSorted, FullSM>(dataToSort + blockIdx.x*nbValuesToSort, nbValuesToSort, nbWorkingValues);
    }
}

void benmark512_v3(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_512_v3.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,nbthreads,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int maxNbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "maxNbElements: " << maxNbElements << std::endl;

    int MaxSharedMemSize;
    CUDA_CHECK(cudaDeviceGetAttribute(&MaxSharedMemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    std::cout << "Shared memory size: " << MaxSharedMemSize << std::endl;
    const int MaxSize = int(MaxSharedMemSize / sizeof(int) / 2);
    std::cout << "Max size: " << MaxSize << std::endl;


    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxThreads = 32 ; idxThreads <= DefaultNbThreadsPerBlock ; idxThreads += 32){
                        for(int idxSize = 32 ; idxSize <= MaxSize ; idxSize *= 2){
                            for(int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
                                const int nbBlocks = maxNbElements / DefaultNbThreadsPerBlock;
                                const int nbElements = idxSize*nbBlocks;
                                std::vector<int> h_array(nbElements);
                                {
                                    std::cout << "idxInitData: " << idxInitData 
                                                << " idxThreads: " << idxThreads 
                                                << " idxLoop: " << idxLoop
                                                << " idxSize " << idxSize
                                                << std::endl;


                                    if(idxInitData == RANDOM){
                                        std::random_device rd;
                                        std::mt19937 gen(0);
                                        std::uniform_int_distribution<int> dis(0, 1000000);
                                        for (int i = 0; i < int(h_array.size()); ++i) {
                                            h_array[i] = dis(gen);
                                        }
                                    }
                                    else if(idxInitData == NORMAL){
                                        std::random_device rd;
                                        std::mt19937 gen(0);
                                        std::normal_distribution<double> dis(0, 1 + idxSize/4);
                                        for (int i = 0; i < int(h_array.size()); ++i) {
                                            h_array[i] = std::max(0, std::min( idxSize,int(dis(gen)+idxSize/2)));
                                        }
                                    }
                                    else if(idxInitData == SORTED){
                                        for (int i = 0; i < int(h_array.size()); ++i) {
                                            h_array[i] = i%idxSize;
                                        }
                                    }
                                    else if(idxInitData == REVERSED){
                                        for (int i = 0; i < int(h_array.size()); ++i) {
                                            h_array[i] = (idxSize-1)-i%idxSize;
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
                                    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(idxSize);
                                    const int threadsPerBlock = NOrPow2;
                                    const int blocksPerGrid = nbBlocks;

                                    CUDA_CHECK(cudaDeviceSynchronize());
                                    SpTimer timer;

                                    benchmarkSort512_v3<int, TestSorted, FullSM, AsynCpy><<<blocksPerGrid, threadsPerBlock, 2*NOrPow2*sizeof(int) >>>(d_array, idxSize, NOrPow2);

                                    CUDA_CHECK(cudaDeviceSynchronize());
                                    timer.stop();

                                    std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;
                                    std::cout << "    - by sort : " << timer.getElapsed()/nbBlocks << "s" << std::endl;

                                    // Free device memory
                                    CUDA_CHECK(cudaFree(d_array));

                                    filecsv << (TestSorted?"TRUE":"FALSE") << "," 
                                            << (FullSM?"TRUE":"FALSE") << "," 
                                            << (AsynCpy?"TRUE":"FALSE") << "," 
                                            << InitDataString(idxInitData) << "," 
                                            << idxSize << ","
                                            << idxThreads << ","
                                            <<  timer.getElapsed() << ","
                                            << timer.getElapsed()/nbBlocks << "," 
                                            << timer.getElapsed()/nbBlocks/double(idxSize*std::log2(idxSize)) << std::endl;
                                }
                            }
                        }
                    }
                }
            });
        });
    });
}




void benmarkSortLarge(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_large.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1073741824UL, (freeMem / sizeof(int) / 2));
    std::cout << "nbElements: " << nbElements << std::endl;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
                        for(int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
                            std::vector<int> h_array(idxSize);
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
                                else if(idxInitData == NORMAL){
                                    std::random_device rd;
                                    std::mt19937 gen(0);
                                    std::normal_distribution<double> dis(0, 1 + h_array.size()/4);
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = std::max(0UL, std::min( h_array.size(),size_t(dis(gen)+h_array.size()/2)));
                                    }
                                }
                                else if(idxInitData == SORTED){
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = i;
                                    }
                                }
                                else if(idxInitData == REVERSED){
                                    for (int i = 0; i < int(h_array.size()); ++i) {
                                        h_array[i] = h_array.size()-i;
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

                                Sort<int, TestSorted, FullSM, AsynCpy>(d_array, idxSize);

                                CUDA_CHECK(cudaDeviceSynchronize());
                                timer.stop();

                                std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;

                                // Free device memory
                                CUDA_CHECK(cudaFree(d_array));

                                filecsv << (TestSorted?"TRUE":"FALSE") << "," 
                                        << (FullSM?"TRUE":"FALSE") << "," 
                                        << (AsynCpy?"TRUE":"FALSE") << "," 
                                        << InitDataString(idxInitData) << "," 
                                        << idxSize << "," 
                                        << timer.getElapsed() << ","
                                        << timer.getElapsed()/(idxSize * std::log2(idxSize))
                                        << std::endl;
                            }
                        }
                    }
                }
            });
        });
    });
}

// Use thrust
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void benmarkThrust(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_thrust.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    filecsv << "#datainit,size,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1073741824UL, (freeMem / sizeof(int) / 2));
    std::cout << "nbElements: " << nbElements << std::endl;

    std::vector<int> h_array(nbElements);

    for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
        for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
            for(int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
                {
                    std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop << std::endl;
                    {
                        size_t freeMem, totalMem;
                        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
                        std::cout << "Free memory: " << freeMem << std::endl;
                        std::cout << "Total memory: " << totalMem << std::endl;
                    }


                    if(idxInitData == RANDOM){
                        std::random_device rd;
                        std::mt19937 gen(0);
                        std::uniform_int_distribution<int> dis(0, 1000000);
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = dis(gen);
                        }
                    }
                    else if(idxInitData == NORMAL){
                        std::random_device rd;
                        std::mt19937 gen(0);
                        std::normal_distribution<double> dis(0, 1 + h_array.size()/4);
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = std::max(0UL, std::min( h_array.size(),size_t(dis(gen)+h_array.size()/2)));
                        }
                    }
                    else if(idxInitData == SORTED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = i;
                        }
                    }
                    else if(idxInitData == REVERSED){
                        for (int i = 0; i < int(h_array.size()); ++i) {
                            h_array[i] = h_array.size()-i;
                        }
                    }
                    else{
                        assert(0);
                    }

                    int* d_array = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_array, h_array.size() * sizeof(int)));

                    // Copy data from host to device
                    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(int), cudaMemcpyHostToDevice));

                    thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_array);

                    CUDA_CHECK(cudaDeviceSynchronize());
                    SpTimer timer;
                    thrust::sort(d_ptr, d_ptr + idxSize);

                    CUDA_CHECK(cudaDeviceSynchronize());
                    timer.stop();

                    std::cout << "Time: " << timer.getElapsed() << "s" << std::endl;

                    // Free device memory
                    CUDA_CHECK(cudaFree(d_array));

                    filecsv << InitDataString(idxInitData) << "," 
                            << idxSize << "," 
                            << timer.getElapsed()  << ","
                            << timer.getElapsed()/(idxSize * std::log2(idxSize))
                            << std::endl;
                }
            }
        }
    }
}

#include <ctime>

std::string GetDateTime(){
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d-%X", &tstruct);
    return buf;
}

void benchmarkAll(){
    const std::string filename = "benchmark_" + GetDateTime();
    benmark32(filename);
    benmark512(filename);
    benmark512_v2(filename);
    benmark512_v3(filename);
    benmarkSortLarge(filename);
    benmarkThrust(filename);
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

int main(){
    benchmarkAll();
    return 0;
}