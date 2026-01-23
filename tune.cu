#include "cusort.hpp"

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

#include <fstream>


template <bool TBoo>
struct ConstexprBool{
    static constexpr bool value = TBoo;
};

template <typename Func>
void for_bool(Func&& func) {
    func(ConstexprBool<true>());
    func(ConstexprBool<false>());
}

// Template iterator to iterate on integers at compile time
template<int Start, int End, typename Func>
void for_int_step1(Func&& func) {
    if constexpr (Start < End) {
        func(std::integral_constant<int, Start>{});
        for_int_step1<Start + 1, End>(std::forward<Func>(func));
    }
}

template<int Start, int End, typename Func>
void for_int_pow2(Func&& func) {
    if constexpr (Start < End) {
        func(std::integral_constant<int, Start>{});
        for_int_pow2<Start*2, End>(std::forward<Func>(func));
    }
}


////////////////////////////////////////////////////////////////////

#include "valgen.hpp"

constexpr int NbLoops = 5;


void benmarkSortLarge(const std::string& inBasename){
    std::ofstream filecsv(inBasename + "_large.csv");
    if(!filecsv.is_open()){
        std::cerr << "Cannot open file" << std::endl;
        return;
    }

    CuRandManager<int> randManager;

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,time,timelogn,minNbParts,maxNbParts" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = 32768;//std::min(268435456UL, (freeMem / sizeof(int) / 3));
    std::cout << "nbElements: " << nbElements << std::endl;

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSize = nbElements ; idxSize <= nbElements ; idxSize *= 2){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            for_int_pow2<16, 128>([&](auto MinNbParts)
                            {
                                for_int_pow2<MinNbParts.value*2, 1024+1>([&](auto MaxNbParts)
                                {
                                    std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop
                                            << " MinNbParts" << MinNbParts.value << " MaxNbParts" << MaxNbParts.value << std::endl;

                                    // Allocate device memory for the array
                                    int* d_array = nullptr;
                                    CUDA_CHECK(cudaMalloc(&d_array, idxSize * sizeof(int)));

                                    SpTimer initTimer;
                                    randManager.initArray(d_array, idxSize, idxSize, idxInitData);
                                    initTimer.stop();
                                    std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;
                                    
                                    CUDA_CHECK(cudaDeviceSynchronize());
                                    SpTimer timer;

                                    Sort<int, TestSorted, FullSM, AsynCpy, MinNbParts, MaxNbParts>(d_array, idxSize);

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
                                            << timer.getElapsed()/(idxSize * std::log2(idxSize)) << ","
                                            << MinNbParts.value << ","
                                            << MaxNbParts.value
                                            << std::endl;
                                });
                            });
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

    filecsv << "#testsorted,fullsm,asynccpy,datainit,size,time,timelogn,minNbParts,maxNbParts" << std::endl;

    // Get amount of free global memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem << std::endl;
    std::cout << "Total memory: " << totalMem << std::endl;

    const int nbElements = 32768;//std::min(268435456UL, (freeMem / sizeof(int) / 3));
    std::cout << "nbElements: " << nbElements << std::endl;

    ThreadPool pool(1);

    for_bool([&](auto bool_constant_testsorted) {
        constexpr bool TestSorted = bool_constant_testsorted.value;
        for_bool([&](auto bool_constant_fullsm) {
            constexpr bool FullSM = bool_constant_fullsm.value;
            for_bool([&](auto bool_constant_asynccpy) {
                constexpr bool AsynCpy = bool_constant_asynccpy.value;
                for(int idxInitData = 0 ; idxInitData < NB_INITS ; ++idxInitData){
                    for(int idxSize = nbElements ; idxSize <= nbElements ; idxSize *= 2){
                        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                            for_int_pow2<32, 1024+1>([&](auto MinNbParts)
                            {
                                for_int_pow2<MinNbParts.value*2, 1024+1>([&](auto MaxNbParts)
                            {
                                std::cout << "idxInitData: " << idxInitData << " idxSize: " << idxSize << " idxLoop: " << idxLoop <<
                                           " MinNbParts: " << MinNbParts.value << " MaxNbParts: " << MaxNbParts.value << std::endl;


                                // Allocate device memory for the array
                                int* d_array = nullptr;
                                CUDA_CHECK(cudaMalloc(&d_array, idxSize * sizeof(int)));

                                SpTimer initTimer;
                                randManager.initArray(d_array, idxSize, idxSize, idxInitData);
                                initTimer.stop();
                                std::cout << "Init time: " << initTimer.getElapsed() << "s" << std::endl;
                                
                                CUDA_CHECK(cudaDeviceSynchronize());
                                SpTimer timer;

                                SortOne<int, TestSorted, FullSM, AsynCpy, MinNbParts, MaxNbParts>(pool, d_array, idxSize);

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
                                        << timer.getElapsed()/(idxSize * std::log2(idxSize)) << ","
                                        << MinNbParts.value << ","
                                        << MaxNbParts.value
                                        << std::endl;
                            });
                            });
                        }
                    }
                }
            });
        });
    });
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

void tuneAll(){
    const std::string datetime = GetDateTime();
    const std::string filename = "tune_sort_" + datetime;
    benmarkSortLarge(filename);
    // Too slow benmarkSortLargeOne(filename);
}
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

int main(){
    tuneAll();
    return 0;
}
