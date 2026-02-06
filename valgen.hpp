#ifndef VALGEN_HPP
#define VALGEN_HPP


#include <curand_kernel.h>

enum InitData{
    SORTED,
    REVERSED,
    RANDOM,
    NORMAL,
    CONSTANT,
    BINARY,
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
        case CONSTANT:
            return "CONSTANT";
        case BINARY:
            return "BINARY";
        default:
            return "UNKNOWN";
    }
}


// Kernel to initialize one curandState per thread.
__global__ void InitRandStates(curandState_t* states, const int totalThreads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < totalThreads) {
        // Seed (here 1234) is fixed; the sequence number is unique per thread.
        curand_init(1234, tid, 0, &states[tid]);
    }
}

// Kernel to initialize the array using the random states (or arithmetic)
template <class T>
__global__ void InitArray(curandState_t* states, T *d_array,
                            const int idxSizePerBlock,
                            const int inNbElements,
                            const int idxInitData) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    
    // For RANDOM and NORMAL, load the state locally.
    curandState_t localState;
    if (idxInitData == RANDOM || idxInitData == NORMAL) {
        localState = states[tid];
    }
    
    // Each thread processes multiple elements.
    for (int i = tid; i < inNbElements; i += totalThreads) {
        if (idxInitData == RANDOM) {
            // curand_uniform returns a float in (0, 1]; scale it to [0, 1000000]
            d_array[i] = (int)(curand_uniform(&localState) * 1000000);
        }
        else if (idxInitData == NORMAL) {
            // Generate a normal value with mean 0 and standard deviation = 1 + idxSizePerBlock/4.
            // Then shift it by idxSizePerBlock/2 and clamp to [0, idxSizePerBlock].
            float sigma = 1.0f + idxSizePerBlock / 4.0f;
            int value = (int)(curand_normal(&localState) * sigma + idxSizePerBlock / 2.0f);
            if (value < 0) value = 0;
            if (value > idxSizePerBlock) value = idxSizePerBlock;
            d_array[i] = value;
        }
        else if (idxInitData == SORTED) {
            d_array[i] = i % idxSizePerBlock;
        }
        else if (idxInitData == REVERSED) {
            d_array[i] = (idxSizePerBlock - 1) - (i % idxSizePerBlock);
        }
        else if (idxInitData == CONSTANT) {
            d_array[i] = 99;
        }
        else if (idxInitData == BINARY) {
            d_array[i] = i%2;
        }
    }
    
    // For RANDOM and NORMAL, save the updated state back.
    if (idxInitData == RANDOM || idxInitData == NORMAL) {
        states[tid] = localState;
    }
}

template <class T>
class CuRandManager{
    const int nbThreadsPerBlock = 256;
    const int nbBlocks = 256;
    curandState_t* d_randStates = nullptr;
public:
    CuRandManager(){
        CUDA_CHECK(cudaMalloc(&d_randStates, nbThreadsPerBlock * nbBlocks * sizeof(curandState_t)));
        InitRandStates<<<nbBlocks, nbThreadsPerBlock>>>(d_randStates, nbThreadsPerBlock * nbBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CuRandManager(){
        if(d_randStates){
            CUDA_CHECK(cudaFree(d_randStates));
        }
    }

    CuRandManager(const CuRandManager&) = delete;
    CuRandManager& operator=(const CuRandManager&) = delete;

    void initArray(T *d_array, const int idxSizePerBlock, const int inNbElements, const int idxInitData){
        InitArray<<<nbBlocks, nbThreadsPerBlock>>>(d_randStates, d_array, idxSizePerBlock, inNbElements, idxInitData);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};


#endif // VALGEN_HPP
