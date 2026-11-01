#include "cusort.hpp"
#include "intanddata.hpp"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/version.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/version.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

// Default execution is restricted to independent, on-chip bitonic-sized arrays.
// --mode large explicitly selects the older whole-range sorting algorithms.
struct BenchOptions {
    std::string mode = "small";
    std::string type = "both";
    std::string prefix;
    int minSize = 1;
    int maxSize = 0; // Auto: bitonic capacity in small mode; 2^24 in large mode.
    int maxThreads = 1024;
    int arrays = 4096;
    int batchElements = 1 << 22;
    int repetitions = 5;
    bool checkOnly = false;
};

std::string timestamp() {
    const auto now = std::time(nullptr);
    char value[32];
    std::strftime(value, sizeof(value), "%Y-%m-%d-%H-%M-%S", std::localtime(&now));
    return value;
}

void usage() {
    std::cout << "Usage: bench [--mode small|large|both] [--type int|record|both]\n"
        "  --min-size N       Minimum array length (default 1)\n"
        "  --max-size N       Maximum array length (default: bitonic capacity)\n"
        "  --max-threads T    Sweep warp multiples up to T (default 1024)\n"
        "  --arrays A         Maximum independent arrays per small batch (default 4096)\n"
        "  --batch-elements M Bound N*A per batch (default 4194304)\n"
        "  --repetitions R    Timed repetitions after warm-up (default 5)\n"
        "  --output PREFIX   CSV path prefix\n"
        "  --check-only      Validate every applicable configuration, with 3 arrays\n"
        "                    and no timing; includes duplicate/extreme keys.\n"
        "Small CSV: E_s is CUDA-event batch time, X_s=E_s/(A*N*log2(N)).\n"
        "E_wall_s also includes host dispatch and completion. N=1 has X_s=nan.\n"
        "Thrust sorts each array separately with its CUDA backend. Its E_s includes\n"
        "host-induced gaps between calls; it is an API-level baseline.\n";
}

BenchOptions parseOptions(int argc, char** argv) {
    BenchOptions o;
    o.prefix = "benchmark_" + timestamp();
    for (int arg = 1; arg < argc; ++arg) {
        const std::string key = argv[arg];
        if (key == "--check-only") { o.checkOnly = true; continue; }
        if (arg + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++arg];
        if (key == "--mode") o.mode = value;
        else if (key == "--type") o.type = value;
        else if (key == "--output") o.prefix = value;
        else {
            size_t consumed = 0;
            const long long number = std::stoll(value, &consumed);
            if (consumed != value.size() || number < 1 || number > (1 << 28))
                throw std::runtime_error("Invalid positive integer for " + key);
            const int n = int(number);
            if (key == "--min-size") o.minSize = n;
            else if (key == "--max-size") o.maxSize = n;
            else if (key == "--max-threads") o.maxThreads = n;
            else if (key == "--arrays") o.arrays = n;
            else if (key == "--batch-elements") o.batchElements = n;
            else if (key == "--repetitions") o.repetitions = n;
            else throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (o.mode != "small" && o.mode != "large" && o.mode != "both")
        throw std::runtime_error("Invalid --mode");
    if (o.type != "int" && o.type != "record" && o.type != "both")
        throw std::runtime_error("Invalid --type");
    if (o.maxThreads < 32 || o.maxThreads > 1024 || o.maxThreads % 32)
        throw std::runtime_error("--max-threads must be a multiple of 32 in [32,1024]");
    if (o.maxSize && o.minSize > o.maxSize)
        throw std::runtime_error("--min-size exceeds --max-size");
    if (o.checkOnly) { o.arrays = std::min(o.arrays, 3); o.repetitions = 0; }
    return o;
}

template <typename T> struct DeviceArray {
    T* ptr = nullptr;
    explicit DeviceArray(size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

struct Events {
    cudaEvent_t begin, end;
    Events() { CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end)); }
    ~Events() { cudaEventDestroy(begin); cudaEventDestroy(end); }
};

__host__ __device__ unsigned mixBits(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

// Stateless initialization gives all methods exactly the same input, including
// defined, unique record payloads. Each repetition restores this original batch.
template <typename T>
__global__ void initializeBatch(T* data, int n, int count, int distribution) {
    for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < count;
         i += int(gridDim.x * blockDim.x)) {
        const int local = i % n;
        const unsigned random = mixBits(unsigned(i) + 1234U);
        int key = 0;
        switch (distribution) {
            case SORTED: key = local; break;
            case REVERSED: key = n - 1 - local; break;
            case RANDOM: key = int(random % 1000001U); break;
            case NORMAL: {
                // Box-Muller sample, clamped as in the original benchmark.
                const float u = (float(random & 0xffffffU) + 1.0f) / 16777217.0f;
                const float v = float(mixBits(random) & 0xffffffU) / 16777216.0f;
                const float z = sqrtf(-2.0f * logf(u)) * cosf(6.28318530718f * v);
                key = int(z * (1.0f + n / 4.0f) + n / 2.0f);
                key = key < 0 ? 0 : (key > n ? n : key);
                break;
            }
            case CONSTANT: key = 99; break;
            case BINARY: key = local % 2; break;
            default: { // Extreme integer keys expose padding and tie errors.
                const int extremes[] = {INT_MAX, INT_MIN, 0, INT_MAX, -1, 1};
                key = extremes[local % 6];
            }
        }
        if constexpr (std::is_same<T, IntAndData>::value) data[i] = {key, i};
        else data[i] = key;
    }
}

std::string distributionName(int d) {
    return d == NB_INITS ? "EXTREMES" : InitDataString(d);
}

template <typename T>
__device__ bool sameKey(const T& a, const T& b) {
    return !(a < b) && !(b < a);
}

// Scalar outputs must exactly equal a CPU-sorted reference. For records, check
// keys, original payload identity, segment membership, and uniqueness. This
// verifies the full permutation without requiring stability among equal keys.
template <typename T>
__global__ void validateBatch(const T* input, const T* result, const T* reference,
                             int n, int count, int* seen, int* error) {
    for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < count;
         i += int(gridDim.x * blockDim.x)) {
        bool valid = sameKey(result[i], reference[i]);
        if constexpr (std::is_same<T, IntAndData>::value) {
            const int origin = result[i].data;
            if (origin < 0 || origin >= count || origin / n != i / n) valid = false;
            else {
                if (input[origin] != result[i]) valid = false;
                if (atomicAdd(seen + origin, 1) != 0) valid = false;
            }
        }
        if (!valid) atomicExch(error, 1);
    }
}

struct Configuration {
    const char* method;
    int testSorted = -1, fullSM = -1, asyncCopy = -1;
    int threads = 0, itemsPerThread = 0, working = 0;
    size_t sharedBytes = 0;
};

struct Csv {
    std::ofstream stream;
    int runtimeVersion = 0;
    explicit Csv(const std::string& path) {
        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        stream.open(path);
        if (!stream) throw std::runtime_error("Cannot open " + path);
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        stream << "method,distribution,N,arrays,total_values,working_values,threads,items_per_thread,"
                  "testsorted,fullsm,asynccpy,shared_bytes,trial,E_s,X_s,E_wall_s,"
                  "wall_X_s,time_per_array_s,time_per_value_s,validated,cuda_runtime,thrust_version,cub_version\n";
        stream << std::setprecision(17);
    }
    void write(const Configuration& c, int distribution, int n, int arrays, int trial,
               double elapsed, double wall) {
        const double values = double(n) * arrays;
        const double denom = n > 1 ? values * std::log2(double(n)) : 0;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        stream << c.method << ',' << distributionName(distribution) << ',' << n << ','
            << arrays << ',' << size_t(n) * arrays << ',' << c.working << ','
            << c.threads << ',' << c.itemsPerThread << ',' << c.testSorted << ','
            << c.fullSM << ',' << c.asyncCopy << ',' << c.sharedBytes << ',' << trial << ','
            << elapsed << ',' << (denom ? elapsed / denom : nan) << ',' << wall << ','
            << (denom ? wall / denom : nan) << ',' << elapsed / arrays << ','
            << elapsed / values << ",1," << runtimeVersion << ',' << THRUST_VERSION
            << ',' << CUB_VERSION << '\n';
        stream.flush();
    }
};

template <typename T> struct Batch {
    int n, arrays, count, distribution;
    const BenchOptions& options;
    Csv& csv;
    DeviceArray<T> input, values, reference;
    DeviceArray<int> seen, error;
    Events events;
    Batch(int length, int batchArrays, int d, const BenchOptions& o, Csv& output)
        : n(length), arrays(batchArrays), count(n * arrays), distribution(d), options(o), csv(output),
          input(count), values(count), reference(count),
          seen(std::is_same<T, IntAndData>::value ? count : 1), error(1) {
        initializeBatch<<<256, 256>>>(input.ptr, n, count, distribution);
        CUDA_CHECK(cudaGetLastError());
        std::vector<T> expected(count);
        CUDA_CHECK(cudaMemcpy(expected.data(), input.ptr, size_t(count) * sizeof(T), cudaMemcpyDeviceToHost));
        for (int a = 0; a < arrays; ++a)
            std::sort(expected.begin() + size_t(a) * n, expected.begin() + size_t(a + 1) * n);
        CUDA_CHECK(cudaMemcpy(reference.ptr, expected.data(), size_t(count) * sizeof(T), cudaMemcpyHostToDevice));
    }
    void restore() {
        CUDA_CHECK(cudaMemcpyAsync(values.ptr, input.ptr, size_t(count) * sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void verify(const Configuration& config) {
        CUDA_CHECK(cudaMemset(seen.ptr, 0, sizeof(int) * (std::is_same<T, IntAndData>::value ? count : 1)));
        CUDA_CHECK(cudaMemset(error.ptr, 0, sizeof(int)));
        validateBatch<<<256, 256>>>(input.ptr, values.ptr, reference.ptr, n, count, seen.ptr, error.ptr);
        CUDA_CHECK(cudaGetLastError());
        int failed = 0;
        CUDA_CHECK(cudaMemcpy(&failed, error.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        if (failed) throw std::runtime_error(std::string("Validation failed: ") + config.method
            + " N=" + std::to_string(n) + " threads=" + std::to_string(config.threads)
            + " sorted=" + std::to_string(config.testSorted) + " fullsm=" + std::to_string(config.fullSM)
            + " async=" + std::to_string(config.asyncCopy) + " distribution=" + distributionName(distribution));
    }
    template <typename Launch> void run(const Configuration& config, Launch launch) {
        // Warm-up and complete correctness check are both outside timed samples.
        restore();
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify(config);
        if (options.checkOnly) {
            std::cout << "PASS " << config.method << " N=" << n << " threads=" << config.threads
                << " sorted=" << config.testSorted << " fullsm=" << config.fullSM
                << " async=" << config.asyncCopy << ' ' << distributionName(distribution) << '\n';
            return;
        }
        for (int trial = 0; trial < options.repetitions; ++trial) {
            restore(); // Never benchmark the sorted output of a preceding run.
            const auto start = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventRecord(events.begin));
            launch();
            CUDA_CHECK(cudaGetLastError());
            // Large sort APIs can use worker streams. Join them before recording
            // the stop; small kernels and Thrust use only the default stream.
            if (options.mode != "small" && arrays == 1 && n > ComputeDefaultMaxBitoSort<T>())
                CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(events.end));
            CUDA_CHECK(cudaEventSynchronize(events.end));
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, events.begin, events.end));
            verify(config); // Validate this measured output, outside both timers.
            csv.write(config, distribution, n, arrays, trial, double(ms) * 1e-3, wall);
        }
    }
};

template <typename T, bool Sorted, bool Shared>
__global__ void benchmarkSortSmall(T* data, int n) {
    T* segment = data + size_t(blockIdx.x) * n; // Packed stride is N, not 32.
    Sort_1Warp_SM_le32_d<T, Sorted, Shared>(segment, segment, n);
}

template <typename T, int Variant, bool Sorted, bool Shared, bool Async>
__global__ void benchmarkSortBlock(T* data, int n, int working) {
    T* segment = data + size_t(blockIdx.x) * n;
    if constexpr (Variant == 1) {
        if constexpr (Async) Sort_1Block_async_gt32_d_extsm<T, Sorted, Shared>(segment, n);
        else Sort_1Block_gt32_d_extsm<T, Sorted, Shared>(segment, n);
    } else if constexpr (Variant == 2) {
        if constexpr (Async) Sort_1Block_async_gt32_d_extsm_v2<T, Sorted, Shared>(segment, segment, n);
        else Sort_1Block_gt32_d_extsm_v2<T, Sorted, Shared>(segment, segment, n);
    } else {
        if constexpr (Async) Sort_1Block_async_gt32_d_extsm_v3<T, Sorted, Shared>(segment, segment, n, working);
        else Sort_1Block_gt32_d_extsm_v3<T, Sorted, Shared>(segment, segment, n, working);
    }
}

// CUB padding must be strictly after valid records, even for INT_MAX keys.
// Generated payloads are [0,count); INT_MAX is reserved for the padding record.
template <typename T> struct CubLess {
    __device__ bool operator()(const T& a, const T& b) const {
        if constexpr (std::is_same<T, IntAndData>::value) {
            if (a.val == b.val) return a.data != INT_MAX && b.data == INT_MAX;
        }
        return a < b;
    }
};

template <typename T, int Threads, int Items>
__global__ void benchmarkBlockMergeSort(T* data, int n) {
    using Load = cub::BlockLoad<T, Threads, Items, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Store = cub::BlockStore<T, Threads, Items, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using Sort = cub::BlockMergeSort<T, Threads, Items>;
    __shared__ union {
        typename Load::TempStorage load;
        typename Store::TempStorage store;
        typename Sort::TempStorage sort;
    } temp;
    T keys[Items];
    T* segment = data + size_t(blockIdx.x) * n;
    Load(temp.load).Load(segment, keys, n, my_numeric_limits_max<T>());
    __syncthreads();
    Sort(temp.sort).Sort(keys, CubLess<T>{}, n, my_numeric_limits_max<T>());
    __syncthreads();
    Store(temp.store).Store(segment, keys, n);
}

template <typename Kernel> bool supportedKernel(Kernel kernel, int threads, size_t dynamicBytes,
                                                const cudaDeviceProp& device, size_t* totalShared = nullptr) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    if (totalShared) *totalShared = attr.sharedSizeBytes + dynamicBytes;
    if (threads > device.maxThreadsPerBlock || threads > attr.maxThreadsPerBlock
        || attr.sharedSizeBytes + dynamicBytes > device.sharedMemPerBlock) return false;
    // Register allocation can also make an otherwise legal block size
    // unlaunchable. In particular, test this for the 1024-thread CUB variants.
    int activeBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, kernel, threads, dynamicBytes));
    return activeBlocks > 0;
}

template <typename T, int Variant, bool Sorted, bool Shared, bool Async>
void runBitonicBlock(Batch<T>& batch, int threads, int working, const cudaDeviceProp& device) {
    const size_t bytes = size_t(working) * sizeof(T) * (Variant == 1 ? 1 : 2);
    auto kernel = benchmarkSortBlock<T, Variant, Sorted, Shared, Async>;
    size_t shared = 0;
    if (!supportedKernel(kernel, threads, bytes, device, &shared)) return;
    const char* name = Variant == 1 ? "bitonic_v1" : (Variant == 2 ? "bitonic_v2" : "bitonic_v3");
    Configuration c{name, Sorted, Shared, Async, threads, (working + threads - 1) / threads, working, shared};
    batch.run(c, [&] { kernel<<<batch.arrays, threads, bytes>>>(batch.values.ptr, batch.n, working); });
}

template <typename T> void runSmallBitonic(Batch<T>& batch, const cudaDeviceProp& device) {
    const int working = std::max(32, GreaterPowerOf2EqualOrLarger(batch.n));
    const int maxThreads = std::min(batch.options.maxThreads, device.maxThreadsPerBlock);
    // Instantiate both values of all three configuration switches, for records
    // as well as scalars. ShuffleValue preserves all record bytes.
    for_bool([&](auto sorted) {
        for_bool([&](auto shared) {
            constexpr bool Sorted = decltype(sorted)::value;
            constexpr bool Shared = decltype(shared)::value;
            if (batch.n <= 32) {
                Configuration c{"bitonic_warp", Sorted, Shared, -1, 32, 1, 32, 32 * sizeof(T)};
                batch.run(c, [&] { benchmarkSortSmall<T, Sorted, Shared><<<batch.arrays, 32>>>(batch.values.ptr, batch.n); });
            }
            if (batch.n < 32) return;
            for_bool([&](auto async) {
                constexpr bool Async = decltype(async)::value;
                if (working <= maxThreads) {
                    runBitonicBlock<T, 1, Sorted, Shared, Async>(batch, working, working, device);
                    runBitonicBlock<T, 2, Sorted, Shared, Async>(batch, working, working, device);
                }
                for (int threads = 32; threads <= std::min(working, maxThreads); threads += 32)
                    runBitonicBlock<T, 3, Sorted, Shared, Async>(batch, threads, working, device);
            });
        });
    });
}

template <typename T> void runSmallCub(Batch<T>& batch, const cudaDeviceProp& device) {
    const int working = std::max(32, GreaterPowerOf2EqualOrLarger(batch.n));
    for_int_pow2<32, 1025>([&](auto threadsConstant) {
        constexpr int Threads = decltype(threadsConstant)::value;
        if (Threads > std::min(working, batch.options.maxThreads)) return;
        constexpr int MaxItems = ComputeDefaultMaxBitoSort<T>() / Threads;
        for_int_pow2<1, MaxItems + 1>([&](auto itemsConstant) {
            constexpr int Items = decltype(itemsConstant)::value;
            if (Threads * Items != working) return;
            auto kernel = benchmarkBlockMergeSort<T, Threads, Items>;
            size_t shared = 0;
            if (!supportedKernel(kernel, Threads, 0, device, &shared)) return;
            Configuration c{"cub_block_merge", -1, -1, -1, Threads, Items, working, shared};
            batch.run(c, [&] { kernel<<<batch.arrays, Threads>>>(batch.values.ptr, batch.n); });
        });
    });
}

template <typename T> void runThrust(Batch<T>& batch) {
    // One host CUDA-backend call per independent array. No global sort of the
    // concatenation, no thrust::seq fallback, and no change to key semantics.
    Configuration c{"thrust_segment_loop", -1, -1, -1, 0, 0, batch.n, 0};
    batch.run(c, [&] {
        for (int a = 0; a < batch.arrays; ++a) {
            auto first = thrust::device_pointer_cast(batch.values.ptr + size_t(a) * batch.n);
            thrust::sort(first, first + batch.n);
        }
    });
}

std::vector<int> smallSizes(int minSize, int maxSize) {
    std::set<int> sizes;
    for (int n = 1; n <= std::min(32, maxSize); ++n) sizes.insert(n);
    for (int n = 64; n <= std::min(1024, maxSize); n += 32) sizes.insert(n);
    for (int n = 32; n <= maxSize; n *= 2) {
        for (int delta : {-1, 0, 1}) if (n + delta >= 1 && n + delta <= maxSize) sizes.insert(n + delta);
    }
    sizes.insert(maxSize);
    std::vector<int> result;
    for (int n : sizes) if (n >= minSize) result.push_back(n);
    return result;
}

template <typename T> int safeArrays(int n, const BenchOptions& options) {
    size_t freeBytes, totalBytes;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Leave room for the reference, payload checks, temporary library storage,
    // and the host copy. The same array count is used by every method.
    const size_t capacity = std::min(size_t(options.batchElements), freeBytes / (8 * sizeof(T)));
    if (capacity < size_t(n)) throw std::runtime_error("Batch/memory budget cannot hold one array; increase --batch-elements");
    return int(std::min(size_t(options.arrays), capacity / size_t(n)));
}

template <typename T> int smallLimit(const cudaDeviceProp& device) {
    const int deviceCapacity = LowerPowerOf2EqualOrSmaller(int(device.sharedMemPerBlock / (2 * sizeof(T))));
    return std::min(ComputeDefaultMaxBitoSort<T>(), deviceCapacity);
}

template <typename T> void validateSmallRequest(const BenchOptions& options, const cudaDeviceProp& device,
                                               const std::string& type) {
    if (options.mode == "small" && options.maxSize > smallLimit<T>(device))
        throw std::runtime_error("--max-size exceeds the small bitonic capacity for " + type
            + " (" + std::to_string(smallLimit<T>(device)) + ")");
}

template <typename T> void benchmarkSmall(const BenchOptions& options, const cudaDeviceProp& device,
                                          const std::string& type) {
    const int limit = smallLimit<T>(device);
    const int maximum = options.maxSize ? std::min(options.maxSize, limit) : limit;
    if (options.minSize > maximum) return;
    Csv csv(options.prefix + "_" + type + "_small.csv");
    for (int n : smallSizes(options.minSize, maximum)) {
        const int arrays = safeArrays<T>(n, options);
        for (int distribution = 0; distribution < NB_INITS + (options.checkOnly ? 1 : 0); ++distribution) {
            std::cout << "small " << type << " N=" << n << " arrays=" << arrays << ' '
                << distributionName(distribution) << std::endl;
            Batch<T> batch(n, arrays, distribution, options, csv);
            runSmallBitonic(batch, device);
            runSmallCub(batch, device);
            runThrust(batch);
        }
    }
}

template <typename T> void benchmarkLarge(const BenchOptions& options, const cudaDeviceProp&,
                                          const std::string& type) {
    size_t freeBytes, totalBytes;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const int memoryLimit = int(std::min(size_t(1 << 28), freeBytes / (8 * sizeof(T))));
    const int maximum = std::min(options.maxSize ? options.maxSize : (1 << 24), memoryLimit);
    const int minimum = std::max(options.minSize, ComputeDefaultMaxBitoSort<T>() + 1);
    Csv csv(options.prefix + "_" + type + "_large.csv");
    for (int n = GreaterPowerOf2EqualOrLarger(minimum); n <= maximum; n *= 2) {
        for (int distribution = 0; distribution < NB_INITS + (options.checkOnly ? 1 : 0); ++distribution) {
            std::cout << "large " << type << " N=" << n << ' ' << distributionName(distribution) << std::endl;
            Batch<T> batch(n, 1, distribution, options, csv);
            for_bool([&](auto sorted) {
                for_bool([&](auto shared) {
                    constexpr bool Sorted = decltype(sorted)::value;
                    constexpr bool Shared = decltype(shared)::value;
                    for_bool([&](auto async) {
                        constexpr bool Async = decltype(async)::value;
                        Configuration c{"large_partition", Sorted, Shared, Async, 0, 0, n, 0};
                        batch.run(c, [&] { Sort<T, Sorted, Shared, Async>(batch.values.ptr, n); });
                    });
                    // These APIs have no asynchronous-copy template switch.
                    Configuration merge{"large_merge", Sorted, Shared, -1, 0, 0, n, 0};
                    batch.run(merge, [&] { SortMerge<T, ComputeDefaultMaxBitoSort<T>() / 2, Sorted, Shared>(batch.values.ptr, n); });
                    Configuration hybrid{"large_hybrid", Sorted, Shared, -1, 0, 0, n, 0};
                    batch.run(hybrid, [&] { HybridSort<T, ComputeDefaultMaxBitoSort<T>() / 2, 512, Sorted, Shared>(batch.values.ptr, n); });
                });
            });
            runThrust(batch);
        }
    }
}

template <typename T> void benchmarkType(const BenchOptions& options, const cudaDeviceProp& device,
                                         const std::string& type) {
    if (options.mode != "large") benchmarkSmall<T>(options, device, type);
    if (options.mode != "small") benchmarkLarge<T>(options, device, type);
}

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--help") { usage(); return 0; }
        const BenchOptions options = parseOptions(argc, argv);
        int deviceId;
        CUDA_CHECK(cudaGetDevice(&deviceId));
        cudaDeviceProp device;
        CUDA_CHECK(cudaGetDeviceProperties(&device, deviceId));
        // Reject incompatible limits before starting either type's long sweep.
        if (options.type != "record") validateSmallRequest<int>(options, device, "int");
        if (options.type != "int") validateSmallRequest<IntAndData>(options, device, "record");
        if (options.type != "record") benchmarkType<int>(options, device, "int");
        if (options.type != "int") benchmarkType<IntAndData>(options, device, "record");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        return 1;
    }
}
