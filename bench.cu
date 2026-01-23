#include "cusort.hpp"

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

#include <fstream>


template <bool TBoo>
struct ConstexprBool{
    static constexpr bool value = TBoo;
};

template <typename Func>
constexpr void for_bool(Func&& func) {
    func(ConstexprBool<true>());
    func(ConstexprBool<false>());
}


////////////////////////////////////////////////////////////////////

#include "valgen.hpp"

constexpr int NbLoops = 5;


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

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,datainit,size,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / 32;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                for(int idxSizePerWarp = 1 ; idxSizePerWarp <= 32 ; ++idxSizePerWarp){
                    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                        {
                            std::cout << "idxInitData: " << idxInitData << " idxSizePerWarp: " << idxSizePerWarp << " idxLoop: " << idxLoop << std::endl;

                            // Allocate device memory for the array
                            int* d_array = nullptr;
                            CUDA_CHECK(cudaMalloc(&d_array, nbElements * sizeof(int)));

                            SpTimer initTimer;
                            randManager.initArray(d_array, idxSizePerWarp, nbElements, idxInitData);
                            initTimer.stop();
                            std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;

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

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / DefaultNbThreadsPerBlock;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSizePerBlock = 32 ; idxSizePerBlock <= DefaultNbThreadsPerBlock ; idxSizePerBlock += 32){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            {
                                std::cout << "idxInitData: " << idxInitData 
                                            << " idxSizePerBlock: " << idxSizePerBlock 
                                            << " idxLoop: " << idxLoop << std::endl;


                                // Allocate device memory for the array
                                int* d_array = nullptr;
                                CUDA_CHECK(cudaMalloc(&d_array, nbElements * sizeof(int)));

                                SpTimer initTimer;
                                randManager.initArray(d_array, idxSizePerBlock, nbElements, idxInitData);
                                initTimer.stop();
                                std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;

                                // Define execution configuration (threads per block and blocks per grid)
                                const int NOrPow2 = GreaterPowerOf2EqualOrLarger(idxSizePerBlock);
                                const int threadsPerBlock = NOrPow2;
                                const int blocksPerGrid = nbBlocks;

                                CUDA_CHECK(cudaDeviceSynchronize());
                                SpTimer timer;

                                benchmarkSort512<int, TestSorted, FullSM, AsynCpy><<<blocksPerGrid, threadsPerBlock, NOrPow2*sizeof(int) >>>(d_array, idxSizePerBlock);

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
                                        << idxSizePerBlock << ","
                                        <<  timer.getElapsed() << ","
                                        << timer.getElapsed()/nbBlocks << "," 
                                        << timer.getElapsed()/nbBlocks/double(idxSizePerBlock*std::log2(idxSizePerBlock)) << std::endl;
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

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,fulltime,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(1UL << 29, (freeMem / sizeof(int)));
    std::cout << "nbElements: " << nbElements << std::endl;

    const int nbBlocks = nbElements / DefaultNbThreadsPerBlock;
    
    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSizePerBlock = 32 ; idxSizePerBlock <= DefaultNbThreadsPerBlock ; idxSizePerBlock += 32){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            {
                                std::cout << "idxInitData: " << idxInitData 
                                            << " idxSizePerBlock: " << idxSizePerBlock << " idxLoop: " << idxLoop << std::endl;


                                // Allocate device memory for the array
                                int* d_array = nullptr;
                                CUDA_CHECK(cudaMalloc(&d_array, nbElements * sizeof(int)));

                                SpTimer initTimer;
                                randManager.initArray(d_array, idxSizePerBlock, nbElements, idxInitData);
                                initTimer.stop();
                                std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;

                                // Define execution configuration (threads per block and blocks per grid)
                                const int NOrPow2 = GreaterPowerOf2EqualOrLarger(idxSizePerBlock);
                                const int threadsPerBlock = NOrPow2;
                                const int blocksPerGrid = nbBlocks;

                                CUDA_CHECK(cudaDeviceSynchronize());
                                SpTimer timer;

                                benchmarkSort512_v2<int, TestSorted, FullSM, AsynCpy><<<blocksPerGrid, threadsPerBlock, 2*NOrPow2*sizeof(int) >>>(d_array, idxSizePerBlock);

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
                                        << idxSizePerBlock << ","
                                        <<  timer.getElapsed() << ","
                                        << timer.getElapsed()/nbBlocks << "," 
                                        << timer.getElapsed()/nbBlocks/double(idxSizePerBlock*std::log2(idxSizePerBlock)) << std::endl;
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

    CuRandManager<int> randManager;

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
    const int MaxSize = std::min(DefaultMaxBitoSort, int(MaxSharedMemSize / sizeof(int) / 2));
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
                            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                                const int nbBlocks = maxNbElements / idxSize;
                                const int nbElements = idxSize*nbBlocks;
                                {
                                    std::cout << "idxInitData: " << idxInitData 
                                                << " idxThreads: " << idxThreads 
                                                << " idxLoop: " << idxLoop
                                                << " idxSize " << idxSize
                                                << " nbElements " << nbElements
                                                << " sharedMem " << 2*idxSize*sizeof(int)
                                                << std::endl;

                                    // Allocate device memory for the array
                                    int* d_array = nullptr;
                                    CUDA_CHECK(cudaMalloc(&d_array, nbElements * sizeof(int)));

                                    SpTimer initTimer;
                                    randManager.initArray(d_array, idxSize, nbElements, idxInitData);
                                    initTimer.stop();
                                    std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;

                                    // Define execution configuration (threads per block and blocks per grid)
                                    const int NOrPow2 = GreaterPowerOf2EqualOrLarger(idxSize);
                                    const int threadsPerBlock = std::min(NOrPow2, DefaultNbThreadsPerBlock);
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

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(268435456UL, (freeMem / sizeof(int) / 3));
    std::cout << "nbElements: " << nbElements << std::endl;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            {
                                std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop << std::endl;


                                // Allocate device memory for the array
                                int* d_array = nullptr;
                                CUDA_CHECK(cudaMalloc(&d_array, idxSize * sizeof(int)));

                                SpTimer initTimer;
                                randManager.initArray(d_array, idxSize, idxSize, idxInitData);
                                initTimer.stop();
                                std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;
                                
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


void benmarkSortLargeOne(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_largeone.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(268435456UL, (freeMem / sizeof(int) / 3));
    std::cout << "nbElements: " << nbElements << std::endl;

    ThreadPool pool;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            {
                                std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop << std::endl;


                                // Allocate device memory for the array
                                int* d_array = nullptr;
                                CUDA_CHECK(cudaMalloc(&d_array, idxSize * sizeof(int)));

                                SpTimer initTimer;
                                randManager.initArray(d_array, idxSize, idxSize, idxInitData);
                                initTimer.stop();
                                std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;
                                
                                CUDA_CHECK(cudaDeviceSynchronize());
                                SpTimer timer;

                                SortOne<int, TestSorted, FullSM, AsynCpy>(pool, d_array, idxSize);

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


void benmarkSortLargeMerge(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_largemerge.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(268435456UL, (freeMem / sizeof(int) / 3));
    std::cout << "nbElements: " << nbElements << std::endl;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            {
                                std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop << std::endl;


                                // Allocate device memory for the array
                                int* d_array = nullptr;
                                CUDA_CHECK(cudaMalloc(&d_array, idxSize * sizeof(int)));

                                SpTimer initTimer;
                                randManager.initArray(d_array, idxSize, idxSize, idxInitData);
                                initTimer.stop();
                                std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;
                                
                                CUDA_CHECK(cudaDeviceSynchronize());
                                SpTimer timer;

                                SortMerge<int>(d_array, idxSize);

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

    CuRandManager<int> randManager;

    filecsv << "#datainit,size,time,timelogn" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = std::min(268435456UL, (freeMem / sizeof(int) / 3));
    std::cout << "nbElements: " << nbElements << std::endl;

    for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
        for(int idxSize = 512 ; idxSize <= nbElements ; idxSize *= 2){
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                {
                    std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop << std::endl;
                    {
                        size_t freeMem, totalMem;
                        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
                        std::cout << "Free memory: " << freeMem << std::endl;
                        std::cout << "Total memory: " << totalMem << std::endl;
                    }


                    int* d_array = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_array, idxSize * sizeof(int)));

                    SpTimer initTimer;
                    randManager.initArray(d_array, idxSize, idxSize, idxInitData);
                    initTimer.stop();
                    std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;

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
    // Too slowbenmarkSortLargeOne(filename);
    benmarkSortLargeMerge(filename);
    benmarkThrust(filename);
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

int main(){
    benchmarkAll();
    return 0;
}
