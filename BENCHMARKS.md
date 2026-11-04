# Small-array CUDA sorting benchmark

`bench` defaults to `--mode small`. It benchmarks independent arrays up to the
bitonic working-set limit, for `int` and `IntAndData`. It does not run the
multipartitioning/whole-range algorithms unless `--mode large` or `--mode both`
is explicitly supplied. Build with the project's CMake setup and a CUDA Toolkit
that provides `cub::BlockMergeSort` (CUDA 12.3 or newer).

Examples, from the build directory:

```sh
./bench --check-only --max-threads 128 --max-size 129
./bench --mode small --arrays 4096 --batch-elements 4194304 --repetitions 5
./bench --mode small --type int --min-size 1024 --max-size 4096
./bench --mode large --max-size 1048576
```

`--help` lists every option. `--output /path/prefix` chooses the CSV prefix.
Files are named `PREFIX_int_small.csv`, `PREFIX_record_small.csv`, etc.
The new unified schema replaces the former separate `_32`, `_512`, `_512_v2`,
and `_512_v3` files; older plotting scripts need their column mappings updated.

## Workload and configurations

For each array length `N`, allocate a packed array containing `A` independent
segments of exactly `N` valid elements. There are no gaps of 32 elements for
short inputs. Every small-sort kernel launches `A` blocks, one per segment.
The batch count is the minimum of `--arrays`, `--batch-elements / N`, and a
conservative GPU memory budget. All methods use the same `A` for a dataset.
The benchmark prints this actual count and saves it in each CSV row.

The small sweep includes every length 1 through 32, multiples of 32 through
1024, power-of-two boundaries and their neighbors, and the chosen maximum.
The default maximum is the implementation's `ComputeDefaultMaxBitoSort<T>()`,
clamped to the device's default shared-memory allowance for two arrays.
Out-of-range small-only requests are rejected rather than silently benchmarking
large-array algorithms. `--type int` can be used when requesting lengths beyond
the smaller record-type limit.

The warp, v1, v2, and v3 variants test both values of `TestIfSorted` and `FullSM`.
v1/v2/v3 also test both synchronous and cooperative asynchronous copy. The warp
wrapper has no asynchronous-copy option. v1/v2 use a power-of-two block with
one working element per thread. v3 sweeps **every multiple of 32** up to the
working length, device limit, and `--max-threads`, using that actual value in the
launch. Unsupported resource configurations are not launched.

CUB BlockMergeSort sweeps power-of-two block sizes and corresponding compile-time
items per thread. CUB's block size must be a power of two. Load/store primitives
transpose between coalesced global access and CUB's blocked register layout;
these loads/stores are part of the measured kernel. Its union temporary storage
is included in `shared_bytes`. Input records retain unique original indices as
payloads. A CUB-only comparator distinguishes the maximum-key padding record
from valid maximum-key records; it leaves the ordering of valid keys unchanged.

Thrust uses one host-side CUDA `thrust::sort` call per segment. This sorts exactly
the same collection of independent arrays. It does not sort the concatenation,
use a sequential device backend, or encode segment identifiers into keys.
It is an API-level baseline and may take much longer because of dispatch and
internal temporary-allocation overhead. Reduce `--arrays` for exploratory runs;
the resulting count still applies to every method.

## Timing and correctness

Initialization, allocations, CPU-reference construction, warm-up, validation,
and restoration of the input batch are outside timed regions. Each timed
repetition starts from the same input, including when testing early exit.
All nine distributions are reproducible across methods. In addition to SORTED,
REVERSED, RANDOM, NORMAL, CONSTANT and BINARY, every run now includes
`PERTURBED_2`, `PERTURBED_8` and `PERTURBED_32`, for both element types and
both benchmark modes. These replace the previous `ALMOST_SORTED_*` cases;
the old measurements are not renamed or reinterpreted.

For each independent array, start from regularly spaced sorted keys
`x[i] = s*i`, with `s = 4`, and generate one perturbed state:

```
x_new[i] = 4*i + epsilon[i]
epsilon[i] ~ discrete uniform {-4*r, ..., +4*r}
r = 2, 8, or 32
```

The suffix r is an amplitude in initial key spacings, not a percentage or an
exact displacement. Noise is sampled for every key; zero noise and ties are
allowed. Keys can be negative and are not clamped at the endpoints. The key
multiset changes. With stable tie-breaking by input position, each element's
rank displacement is at most `2*r` (respectively 4, 16, 64 positions): keys
initially separated by more than `2*r` positions cannot exchange order.
Even a very short movement can cross a warp or block boundary; the generator
has no aligned windows. For short arrays the bound is also limited by N-1.
The construction does not force an unsorted result and does not guarantee
that the sorting network's early-exit checks will succeed early.

A fixed seed `mixBits(array_index + 1234)` gives the same inputs to all methods
and configurations and provides different noise sequences across batch arrays.
Record payloads remain unique input-position identifiers. Host generation and
transfer occur outside timing; validation checks the perturbed keys, not the
unperturbed baseline. The supported maximum N (2^28), spacing and amplitudes
keep all generated keys within signed 32-bit range.

This is a simplified one-step model of temporal coherence, motivated by moving
object bounds in sweep-and-prune collision detection or particle depths in
transparent rendering. It is not an application trace or an end-to-end
simulation benchmark. Its performance must be measured before making any
claim of a benefit. The legacy `test`/`tune` distributions and sorting kernels
are unchanged.

Every configuration is checked after its warm-up and every timed repetition
against a CPU reference for
the **entire batch**. Records are checked for sorted keys, unchanged original
key/payload association, segment membership, and unique payload occurrence.
A failure stops the run before recording measurements for that configuration.
`--check-only` uses at most three arrays, adds an EXTREMES distribution containing
INT_MIN, INT_MAX, and duplicate keys, and validates without timing. Its CSV files
contain headers only; PASS lines are printed on standard output.

CSV fields include:

- `E_s`: CUDA-event elapsed seconds for the whole batch.
- `X_s = E_s / (A * N * log2(N))`.
- `E_wall_s`: steady-clock seconds including host dispatch and completion.
- `wall_X_s`: the same normalization applied to wall time.
- `time_per_array_s = E_s / A` and `time_per_value_s = E_s / (A*N)`.
- `working_values`, `threads`, `items_per_thread`, `shared_bytes`, the three
  configuration flags, trial index, CUDA runtime, Thrust and CUB versions.

A flag value of `-1` means not applicable. `threads=0` means library-managed
launch dimensions, not a zero-thread launch. `N=1` has `X_s=nan` and
`wall_X_s=nan`, because `log2(1)=0`; its unnormalized time fields remain valid.

The event interval for Thrust includes idle gaps while the host submits the
per-segment calls, and CUDA-backend temporary allocation is included. It is not
an isolated local-kernel timing. Whole-range large-mode functions also include
any internal allocations and worker-stream completion. `E_s/A` is an amortized
throughput measure, not the latency of one isolated sort. `X_s` uses N log N as
requested; a bitonic network itself performs O(N log-squared N) work. Check that
the batch is large enough to saturate the GPU before interpreting differences.


## Experimental kernels: v4 and v5

The reference `cusort.hpp` kernels are unchanged. New kernels live in
`cusort_experiments.hpp` and are included by `bench.cu`:

- `bitonic_v4_xor`: a v3 descendant using 32-bit unsigned indices,
  `i ^ (2*stage-1)` reflection partners and `i ^ distance` merge partners.
  The fixed 32-lane warp networks are explicitly unrolled. Runtime-size block
  layers retain their loops; there is no register-tiled redesign in this variant.
- `bitonic_v5_static`: the same operations with template working length and
  block size, compile-time stage/distance expansion, and constant per-thread
  traversal counts. Per-thread loop expansion is capped at eight iterations
  (larger counts are partially unrolled) to limit generated code size.

Both preserve the two shared arrays, cross-warp barrier schedule, comparison
semantics, padding, sortedness checkpoints, and the existing benchmark's full
batch validation. Both test all sortedness/shared-warp/copy flag combinations.
v4 sweeps every warp multiple like v3. v5 specializes power-of-two working
lengths and block sizes from 32 through the supported limit (up to 1024 threads),
so the item count per thread is exact. Compare v3/v4/v5 at matching block sizes
for a controlled comparison; their overall best envelopes tune independently.
They are additional benchmark methods, not replacements in the public Sort,
SortMerge, or HybridSort dispatch.

The CSV also records `registers_per_thread`, `active_blocks_per_sm`, and
`local_bytes_per_thread` for v1/v2/v3/v4/v5 and CUB. Occupancy is the runtime API's
resource prediction; local allocation is not itself a measurement of actual
spill traffic. Unknown/library-managed values use -1. The new columns are
compatible with the plotting script, and do not change E or X.

To validate or benchmark only selected methods, without repeating Thrust calls:

```sh
./bench --check-only --max-size 129 --max-threads 128 --methods bitonic_v3,bitonic_v4_xor,bitonic_v5_static,cub_block_merge
./bench --type int --min-size 32 --methods bitonic_v3,bitonic_v4_xor,bitonic_v5_static,cub_block_merge --output comparison_int
```

Without `--methods`, all methods (including references and Thrust) are retained.
Static specialization increases build time and binary size. A GPU run is still
needed to determine whether any experiment improves performance; no speedup is
assumed from the source transformation alone. The companion plotting script in
the paper folder recognizes both new method names and includes their individual
configuration plots and thread heatmaps when their measurements are present.
