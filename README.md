# CUDA sorting experiments

CUDA/C++ implementations and benchmarks for sorting small arrays on NVIDIA
GPUs. The main benchmark sorts batches of independent arrays, with one CUDA
block per array. It compares several bitonic implementations with CUB
`BlockMergeSort` and, optionally, Thrust.

The repository also contains earlier algorithms and experiments for sorting
large arrays. These are separate from the small-array study.

## Requirements

- An NVIDIA GPU and a compatible NVIDIA driver.
- CUDA Toolkit, including NVCC, CUDA Runtime, cuBLAS, Thrust, and CUB with
  `cub::BlockMergeSort`. The PlaFRIM configuration uses CUDA 13.0.2.
- CMake 3.20 or newer and a host C++ compiler supported by the selected toolkit.
- C++17 and CUDA C++17 support.

The build requires an explicit `CUDA_SM` architecture. The supplied targets are
`80` for A100 and `75` for Quadro RTX 8000. Use a separate build directory for
each architecture and configuration.

## Build

From the repository root, for an A100:

```bash
cmake -S . -B builda100 -DCMAKE_BUILD_TYPE=RELEASE -DCUDA_SM=80
cmake --build builda100 --target bench --parallel 1
```

For a Quadro RTX 8000:

```bash
cmake -S . -B buildrtx -DCMAKE_BUILD_TYPE=RELEASE -DCUDA_SM=75
cmake --build buildrtx --target bench --parallel 1
```

These commands follow the configuration used by the supplied batch scripts.
Compilation of `bench` can take substantial time and memory because it
instantiates many combinations of lengths, block sizes, types, and Boolean
options. Start with one compilation job; increase parallelism if the build
machine has sufficient memory. Runtime options such as `--methods` reduce the
benchmark workload, but do not reduce the instantiated kernels or build time.

Build the other executables explicitly when needed:

```bash
cmake --build builda100 --target test tune --parallel 1
```

Building without `--target` builds all three root-level `.cu` programs.
In-source builds are rejected. `extra/merge.cu` is not included by CMake.

### Compiler flags

The current CMake file explicitly sets CUDA Release flags to `-DSTANDALONE`;
it does not explicitly enable NVCC device-code optimization with `-O3`.
The host C++ Release flags are configured separately. To request an optimized
CUDA build, use a separate directory, for example:

```bash
cmake -S . -B builda100-o3 \
  -DCMAKE_BUILD_TYPE=RELEASE -DCUDA_SM=80 \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DSTANDALONE"
cmake --build builda100-o3 --target bench --parallel 1
```

Record the flags with each measurement campaign: this configuration differs
from the supplied batch scripts. Debug builds enable CUDA device debugging
(`-G`) and should not be used for performance measurements.

## Source files and executables

| File | Role |
| --- | --- |
| `cusort.hpp` | Reference warp/block bitonic implementations and the older large-array sorting algorithms. |
| `cusort_experiments.hpp` | V4 (XOR indexing and fixed warp-loop unrolling) and V5 (static specialization). |
| `cusort_v6.hpp` | V6: persistent register tiles with register, warp, and shared-memory exchanges. |
| `bench.cu` → `bench` | Configurable batched benchmark and correctness validation, including the experimental kernels and CUB baseline. |
| `test.cu` → `test` | Historical correctness tests for the original small, V3, and large-array/merge/hybrid implementations. Does not exercise V4–V6. |
| `tune.cu` → `tune` | Historical parameter sweeps for large-array partition, merge, and hybrid sorts. |
| `intanddata.hpp` | Integer key–payload record type, compared by key. |
| `valgen.hpp` | Original data generators used by the historical tests and tuning program. |
| `bench_distributions.hpp` | Seeded generator of bounded value perturbations for `bench`. |
| `SpTimer.hpp` | Timing utility used by the original code. |
| `extra/merge.cu` | Standalone merge experiment, outside the default build. |
| `BENCHMARKS.md` | Detailed benchmark protocol, configuration coverage, validation, and CSV definitions. |

The experimental kernels are benchmarked explicitly; they do not replace the
public legacy sorting dispatch automatically.

## Correctness checks

Run on a GPU, from the build directory. A restricted initial check is:

```bash
cd builda100
./bench --check-only --max-size 129 --max-threads 128 \
  --methods bitonic_warp,bitonic_v1,bitonic_v2,bitonic_v3,bitonic_v4_xor,bitonic_v5_static,bitonic_v6_register,cub_block_merge
```

For full supported-size validation of V5, V6, and CUB:

```bash
./bench --check-only \
  --methods bitonic_v5_static,bitonic_v6_register,cub_block_merge
```

`--check-only` validates up to three independent arrays, includes extreme and
duplicate keys, and does not collect timings. It prints `PASS` lines; generated
CSV files contain only their headers. Record validation also checks payload
preservation, association with the original key, and array membership.

The historical suite is run separately:

```bash
./test
```

It has no command-line selection and also runs large-array tests. For the
small-array experiments and newer kernels, use `bench --check-only`.

## Small-array benchmarks

From the build directory:

```bash
./bench --help
./bench
```

Plain `./bench` runs **small-array mode**, both integer keys and key–payload
records, all supported methods, all applicable configurations, nine data
distributions, and five timed repetitions. The defaults request up to 4096
arrays and 4194304 valid values per batch. The actual batch size also depends
on array length and available memory.

The default method set includes Thrust, which makes a separate host-side sort
call per array and can make the run much longer. To compare the bitonic
implementations with CUB alone:

```bash
./bench --mode small --arrays 4096 --batch-elements 4194304 --repetitions 5 \
  --methods bitonic_warp,bitonic_v1,bitonic_v2,bitonic_v3,bitonic_v4_xor,bitonic_v5_static,bitonic_v6_register,cub_block_merge \
  --output results/small
```

To restrict the type, lengths, or implementations:

```bash
./bench --type int --min-size 32 --max-size 1024 \
  --methods bitonic_v5_static,bitonic_v6_register,cub_block_merge \
  --output results/int_v5_v6
```

The maximum small-array length is determined from the implementation capacity,
element size, and device shared-memory limit. Unsupported lengths are rejected
in small-only mode. V1/V2 use one working value per thread; V3–V6 support
multiple values per thread, with different specialization constraints.

The Boolean options are swept internally: `testsorted` enables exact order
checking; `fullsm` selects shared-memory warp exchanges rather than shuffles;
`asynccpy` selects the asynchronous copy path where available. The warp method
has no separate asynchronous option. See `BENCHMARKS.md` for each sweep domain.

### Outputs and timing

Without `--output`, the prefix is `benchmark_YYYY-MM-DD-HH-MM-SS` in the current
directory. Small-mode files end in `_int_small.csv` and `_record_small.csv`.
`--output results/small` instead produces `results/small_int_small.csv` and
`results/small_record_small.csv`; parent directories are created automatically.

For valid array length `N` and `A` arrays:

- `E_s` is CUDA-event time for sorting the entire batch.
- `X_s = E_s / (A * N * log2(N))`; it is undefined (`nan`) for `N = 1`.
- `E_wall_s` also includes host dispatch and completion.
- Configuration, trial, validation, and kernel resource fields accompany times.

Input restoration, generation, and correctness checks are outside the timed
region. Every repetition starts from the same input. These are batched
throughput measurements: shared-memory and register requirements affect how
many blocks can reside concurrently. `active_blocks_per_sm` is an occupancy API
prediction, not a profiler measurement of achieved occupancy. `E_s / A` is an
amortized time per array rather than an isolated-sort latency.

## Historical large-array experiments

Large-array mode is opt-in:

```bash
./bench --mode large --max-size 1048576 --output results/large
```

`--mode both` runs both modes. The separate `./tune` executable performs its
older large-array parameter sweeps, with no command-line options; it is not
needed for the small-array study. Its merge and hybrid sweeps currently use
the same `_largemerge.csv` output name, so the latter overwrites the former.

## PlaFRIM jobs

The repository provides three platform files:

| File | Purpose |
| --- | --- |
| `modules` | Loads `git/2.54.0`, `gcc-toolchain/14.3.0`, `cuda-toolkit/13.0.2`, and `cmake/4.1.3`; sets `LD_PRELOAD=/usr/lib64/libcuda.so.1`. Source this file in a shell with the module environment initialized. |
| `a100.batch` | Slurm job constrained to `a100`; configures `CUDA_SM=80` in `builda100`, rebuilds, and runs `./bench`. |
| `rtx8000.batch` | Slurm job constrained to `rtx8000`; configures `CUDA_SM=75` in `buildrtx`, rebuilds, and runs `./bench`. |

The scripts currently assume the repository is at `/projets/schnaps/cuda-sort`.
Before using them in another checkout, adapt that path in both the `source`
and build-directory commands, and change the Slurm notification email. Both
request one exclusive node for up to 24 hours and compile with `make -j 4`.
Reduce compilation parallelism there if memory is insufficient.

Submit from the repository root on PlaFRIM:

```bash
sbatch a100.batch
sbatch rtx8000.batch
```

The jobs run the default small benchmark, including Thrust. To select methods
or other runtime options, edit the `./bench` line in the corresponding batch
file. CSVs are written inside `/projets/schnaps/cuda-sort/builda100` or
`buildrtx`. Slurm uses its default `slurm-JOBID.out` log filename because the
scripts do not specify an output file. `./tune` is commented out in both jobs.

For an interactive session on an allocated GPU node, the environment setup is:

```bash
source modules
```

Keep the module list, compiler flags, GPU model, benchmark command, and CSVs
together when recording an experiment.
