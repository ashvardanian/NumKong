# NumKong: Mixed Precision for All

Portable mixed-precision math, linear-algebra, & retrieval library with 2'000+ SIMD kernels for x86, Arm, RISC-V, LoongArch, Power, & WebAssembly, leveraging rare algebraic transforms with both 1D & 2D registers like AMX & SME, covering 15+ numeric types from 4-bit integers & 6-bit floats to 128-bit complex numbers, validated against 118-bit extended-precision baselines with saturation, casting, & rounding edge-case coverage, in a 5-100x smaller binary than other BLAS-like alternatives, co-designed with Tensor abstractions in C++, Python, Rust, JavaScript, GoLang, & Swift.

![NumKong banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/NumKong-v7.png?raw=true)

## Latency, Throughput, & Numerical Stability

Most libraries return dot products in the __same type as the input__ — Float16 × Float16 → Float16, Int8 × Int8 → Int8.
This leads to quiet overflow: a 2048-dimensional `i8` dot product can reach ±10 million, but `i8` maxes out at 127.
NumKong promotes to wider accumulators — Float16 → Float32, BFloat16 → Float32, Int8 → Int32, Float32 → Float64 — so results stay in range.

| Input  |     NumPy + OpenBLAS |        PyTorch + MKL |                  JAX |               NumKong |
| :----- | -------------------: | -------------------: | -------------------: | --------------------: |
|        |       ░░░░░░░░░░░░░░ |       ░░░░░░░░░░░░░░ |       ░░░░░░░░░░░░░░ |        ░░░░░░░░░░░░░░ |
| `f64`  | 2.0 gso/s, 1e-15 err | 0.6 gso/s, 1e-15 err | 0.4 gso/s, 1e-14 err |  5.8 gso/s, 1e-16 err |
| `f32`  |  1.5 gso/s, 2e-6 err |  0.6 gso/s, 2e-6 err |  0.4 gso/s, 5e-6 err |   7.1 gso/s, 2e-7 err |
| `bf16` |                    — |  0.5 gso/s, 1.9% err |  0.5 gso/s, 1.9% err |   9.7 gso/s, 1.8% err |
| `f16`  | 0.2 gso/s, 0.25% err | 0.5 gso/s, 0.25% err | 0.4 gso/s, 0.25% err | 11.5 gso/s, 0.24% err |
| `e5m2` |                    — |  0.7 gso/s, 4.6% err |  0.5 gso/s, 4.6% err |     7.1 gso/s, 0% err |
| `i8`   |  1.1 gso/s, overflow |  0.5 gso/s, overflow |  0.5 gso/s, overflow |    14.8 gso/s, 0% err |

> Single 2048-d dot product on Intel Sapphire Rapids, single-threaded.
> Each cell shows __gso/s, mean relative error__ vs higher-precision reference.
> gso/s = Giga Scalar Operations per Second — a more suitable name than GFLOP/s when counting both integer and floating-point work.
> NumPy 2.4, PyTorch 2.10, JAX 0.9.

A fair objection: PyTorch and JAX are designed for throughput, not single-call latency.
They lower execution graphs through [XLA](https://openxla.org/) or vendored BLAS libraries like [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) and Nvidia [cuBLAS](https://developer.nvidia.com/cublas).
So here's the same comparison on a throughput-oriented workload — matrix multiplication:

| Input  |      NumPy + OpenBLAS |         PyTorch + MKL |                    JAX |              NumKong |
| :----- | --------------------: | --------------------: | ---------------------: | -------------------: |
|        |        ░░░░░░░░░░░░░░ |        ░░░░░░░░░░░░░░ |         ░░░░░░░░░░░░░░ |       ░░░░░░░░░░░░░░ |
| `f64`  | 65.5 gso/s, 1e-15 err | 68.2 gso/s, 1e-15 err | ~14.3 gso/s, 1e-15 err | 8.6 gso/s, 1e-16 err |
| `f32`  |   140 gso/s, 9e-7 err |   145 gso/s, 1e-6 err |  ~60.5 gso/s, 1e-6 err | 37.7 gso/s, 4e-7 err |
| `bf16` |                     — |   851 gso/s, 1.8% err |  ~25.8 gso/s, 3.4% err |  458 gso/s, 3.6% err |
| `f16`  |  0.3 gso/s, 0.25% err |  140 gso/s, 0.37% err | ~26.1 gso/s, 0.35% err | 103 gso/s, 0.26% err |
| `e5m2` |                     — |   0.4 gso/s, 4.6% err |  ~26.4 gso/s, 4.6% err |    398 gso/s, 0% err |
| `i8`   |   0.4 gso/s, overflow |  50.0 gso/s, overflow |   ~0.0 gso/s, overflow |   1279 gso/s, 0% err |

> Matrix multiplication (2048 × 2048) × (2048 × 2048) on Intel Sapphire Rapids, single-threaded.
> gso/s = Giga Scalar Operations per Second, same format.
> NumPy 2.4, PyTorch 2.10, JAX 0.9, same versions.

For `f64`, compensated "Dot2" summation reduces error by 10–50× compared to naive Float64 accumulation, depending on vector length.
For `f32`, widening to Float64 gives 5–10× lower error.
The library ships as a relatively small binary:

| Package          |   Size | Parallelism & Memory                              | Available For     |
| :--------------- | -----: | :------------------------------------------------ | :---------------- |
| PyTorch + MKL    | 705 MB | Vector & Tile SIMD, OpenMP Threads, Hidden Allocs | Python, C++, Java |
| JAX + jaxlib     | 357 MB | Vector SIMD, XLA Threads, Hidden Allocs           | Python            |
| NumPy + OpenBLAS |  30 MB | Vector SIMD, Built-in Threads, Hidden Allocs      | Python            |
| mathjs           |   9 MB | No SIMD, No Threads, Many Allocs                  | JS                |
| NumKong          |   5 MB | Vector & Tile SIMD, Your Threads, Your Allocs     | 7 languages       |

Every kernel is validated against 118-bit extended-precision baselines with per-type ULP budgets across log-normal, uniform, and Cauchy input distributions.
Tests check triangle inequality, Cauchy-Schwarz bounds, NaN propagation, overflow detection, and probability-simplex constraints for each ISA variant.
Results are cross-validated against OpenBLAS, Intel MKL, and Apple Accelerate.
A broader throughput comparison is maintained in [NumWars](https://github.com/ashvardanian/NumWars).

## Quick Start

| Language | Install                    | Compatible with                | Guide                                        |
| :------- | :------------------------- | :----------------------------- | :------------------------------------------- |
| C / C++  | CMake, headers, & prebuilt | Linux, macOS, Windows, Android | [include/README.md](include/README.md)       |
| Python   | `pip install`              | Linux, macOS, Windows          | [python/README.md](python/README.md)         |
| Rust     | `cargo add`                | Linux, macOS, Windows          | [rust/README.md](rust/README.md)             |
| JS       | `npm install` & `import`   | Node.js, Bun, Deno & browsers  | [javascript/README.md](javascript/README.md) |
| Swift    | Swift Package Manager      | macOS, iOS, tvOS, watchOS      | [swift/README.md](swift/README.md)           |
| Go       | `go get`                   | Linux, macOS, Windows via cGo  | [golang/README.md](golang/README.md)         |

## What's Inside

NumKong covers 17 numeric types — from 6-bit floats to 64-bit complex numbers — across dozens of operations and 30+ SIMD backends, with hardware-aware defaults: Arm prioritizes `f16`, x86 prioritizes `bf16`.

### Language Bindings

| Operation                   | [C 99 & C++ 23][c] | [Python][py] | [Rust][rs] | [JavaScript][js] | [Swift][swift] | [GoLang][go] |
| :-------------------------- | :----------------: | :----------: | :--------: | :--------------: | :------------: | :----------: |
| __Vector Ops__              |                    |              |            |                  |                |              |
| [Dot] Product               |         ●          |      ●       |     ●      |        ●         |       ●        |      ●       |
| [Spatial] Metric            |         ●          |      ●       |     ●      |        ●         |       ●        |      ●       |
| [Set] Similarity            |         ●          |      ●       |     ●      |        ●         |       ●        |      ●       |
| [Geo]spatial                |         ●          |      ●       |     ●      |        ·         |       ●        |      ●       |
| [Mesh] Alignment            |         ●          |      ●       |     ●      |        ·         |       ·        |      ·       |
| [Sparse] Products           |         ●          |      ●       |     ●      |        ·         |       ·        |      ·       |
| [Probability] Divergences   |         ●          |      ●       |     ●      |        ●         |       ·        |      ●       |
| [Curved] Spaces             |         ●          |      ●       |     ●      |        ·         |       ·        |      ·       |
| __Many-to-Many Vector Ops__ |                    |              |            |                  |                |              |
| "[Dots]" Products           |         ●          |      ●       |     ●      |        ●         |       ●        |      ●       |
| "[Spatials]" Metrics        |         ●          |      ●       |     ●      |        ●         |       ●        |      ●       |
| "[Sets]" Similarities       |         ●          |      ●       |     ●      |        ·         |       ●        |      ●       |
| [MaxSim] Scoring            |         ●          |      ●       |     ●      |        ·         |       ●        |      ●       |
| __Scalar Ops__              |                    |              |            |                  |                |              |
| [Cast]                      |         ●          |      ●       |     ●      |        ●         |       ·        |      ·       |
| [Reduce]                    |         ●          |      ●       |     ●      |        ·         |       ·        |      ·       |
| [Each]                      |         ●          |      ●       |     ●      |        ·         |       ·        |      ·       |
| [Trigonometry]              |         ●          |      ●       |     ●      |        ·         |       ·        |      ·       |

[Dot]: include/numkong/dot/README.md
[Dots]: include/numkong/dots/README.md
[Spatial]: include/numkong/spatial/README.md
[Spatials]: include/numkong/spatials/README.md
[Set]: include/numkong/set/README.md
[Sets]: include/numkong/sets/README.md
[Cast]: include/numkong/cast/README.md
[Reduce]: include/numkong/reduce/README.md
[Trigonometry]: include/numkong/trigonometry/README.md
[MaxSim]: include/numkong/maxsim/README.md
[Mesh]: include/numkong/mesh/README.md
[Each]: include/numkong/each/README.md
[Sparse]: include/numkong/sparse/README.md
[Probability]: include/numkong/probability/README.md
[Curved]: include/numkong/curved/README.md
[Geo]: include/numkong/geospatial/README.md
[c]: include/README.md
[py]: python/README.md
[js]: javascript/README.md
[rs]: rust/README.md
[swift]: swift/README.md
[go]: golang/README.md


## Design Decisions

- Avoid loop unrolling and scalar tails.
- Don't manage threads and be compatible with any parallelism models.
- Don't manage memory and be compatible with arbitrary allocators & alignment.
- Don't constrain ourselves to traditional BLAS-like Matrix Multiplication APIs.
- Don't throw exceptions and pass values by pointers.
- Prefer saturated arithmetic and avoid overflows, where needed.
- Cover most modern CPUs with flexible dispatch and wait for them to converge with GPUs.

The rest of this document unpacks the functionality and the logic behind the design decisions.

### Auto-Vectorization & Loop Unrolling

Most "optimized SIMD code" is a 2–4x unrolled data-parallel `for`-loop over `f32` arrays with a serial scalar tail for the last few elements:

```c
float boring_dot_product_f32(float const *a, float const *b, size_t n) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    float result = _mm256_reduce_add_ps(_mm256_add_ps(sum0, sum1));
    for (; i < n; i++) result += a[i] * b[i]; // serial tail
    return result;
}
```

This kind of unrolling has been a common request for NumKong, but the library avoids it by design.

__Modern CPUs already "unroll" in hardware.__
Out-of-order engines with reorder buffers of 320–630 entries (Zen 4: 320, Golden Cove: 512, Apple Firestorm: ~630) can keep a dozen of loop iterations in-flight simultaneously.
The physical register file is much larger than the ISA-visible architectural registers — Skylake has ~180 physical integer registers behind 16 architectural GPRs, and ~168 physical vector registers behind 32 architectural ZMMs.
The register renaming unit maps the same `zmm0` in iteration N and iteration N+1 to different physical registers, extracting cross-iteration parallelism automatically — exactly the benefit that source-level unrolling was historically supposed to provide.

__Unrolling works against NumKong's goals.__
Every unrolled copy is a distinct instruction in the binary.
With 1,500+ kernel endpoints across 30+ backends, even 2x unrolling would inflate the `.text` section by megabytes — directly impacting install size for Python wheels, NPM packages, and Rust crates.
Larger loop bodies also increase instruction-cache and micro-op-cache pressure; [Agner Fog](https://www.agner.org/optimize/) also recommends:

> _"avoid loop unrolling where possible in order to economize the use of the micro-op cache"_.

A loop that spills out of the uop cache falls back to the slower legacy decoder, making the "optimized" version slower than the compact original.
For a header-only library, unrolling also compounds __compilation time__: register allocation is NP-hard (reducible to graph coloring), and unrolling multiplies the number of simultaneously live ranges the allocator must consider, increasing compile time super-linearly across every translation unit that includes the headers.

__Serial tails are a correctness hazard.__
The leftover elements after the last full SIMD chunk run through a scalar loop that silently drops FMA fusion, compensated accumulation, and saturating arithmetic — producing results with different numerical properties than the SIMD body.
NumKong often uses masked loads instead (`_mm512_maskz_loadu_ps` on AVX-512, predicated `svld1_f32` on SVE), processing every element through the same arithmetic path regardless of alignment.
It's not exactly orthogonal to loop-unrolling, but makes a different kernel layout more compatible.

__The gains come from elsewhere.__
On Intel Sapphire Rapids, NumKong was benchmarked against auto-vectorized code compiled with GCC 12.
GCC handles single-precision `float` well, but struggles with `_Float16` and other mixed-precision paths:

| Kind                      | GCC 12 `f32` | GCC 12 `f16` | NumKong `f16` | `f16` improvement |
| :------------------------ | -----------: | -----------: | ------------: | ----------------: |
| Inner Product             |    3,810 K/s |      192 K/s |     5,990 K/s |          __31 x__ |
| Cosine Distance           |    3,280 K/s |      336 K/s |     6,880 K/s |          __20 x__ |
| Euclidean Distance ²      |    4,620 K/s |      147 K/s |     5,320 K/s |          __36 x__ |
| Jensen-Shannon Divergence |    1,180 K/s |       18 K/s |     2,140 K/s |         __118 x__ |

NumKong's `f16` kernels are faster than GCC's `f32` output — not because of unrolling, but because they use [F16C](https://en.wikipedia.org/wiki/F16C) conversion instructions, widening FMA pipelines, and compensated accumulation that compilers do not synthesize from a plain `for` loop.
The same story repeats for `bf16`, `e4m3`, `i8`, and `i4`: these types require algorithmic transformations — lookup tables, algebraic domain shifts, asymmetric [VNNI](https://en.wikipedia.org/wiki/AVX-512#VNNI) tricks — that live beyond the reach of auto-vectorization.

### Parallelism & Multi-Threading

BLAS libraries traditionally manage their own thread pools.
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md) spawns threads controlled by `OPENBLAS_NUM_THREADS`, [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-1/techniques-to-set-the-number-of-threads.html) forks its own OpenMP runtime via `MKL_NUM_THREADS`, and [Apple Accelerate](https://developer.apple.com/documentation/accelerate/blas) delegates to [GCD](https://developer.apple.com/documentation/dispatch) (Grand Central Dispatch).
This works in isolation — but the moment your application adds its own parallelism (joblib, std::thread, Tokio, GCD, OpenMP), you get __thread oversubscription__: MKL spawns 8 threads inside each of your 8 joblib workers, producing 64 threads on 8 cores, thrashing caches and stalling on context switches.
The Python ecosystem has built [entire libraries](https://github.com/joblib/threadpoolctl) just to work around this problem, and [scikit-learn's documentation](https://scikit-learn.org/stable/computing/parallelism.html) devotes a full page to managing the interaction between joblib parallelism and BLAS thread pools.

NumKong takes a different position: __the numerics layer should not own threads__.
Modern hardware makes the "spawn N threads and split evenly" model increasingly untenable:

- __Server-grade CPUs__ have hundreds of cores split across sockets, chiplets, and tiles, resulting in dozens of physical [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access) domains with vastly different memory access latencies.
  A thread pool that ignores NUMA topology will spend more time on remote memory stalls than on arithmetic.
- __Consumer-grade CPUs__ pack heterogeneous Quality-of-Service core types on the same die — Intel P-cores and E-cores run at different frequencies and sometimes support different ISA extensions.
  A naive work-split gives equal chunks to fast and slow cores, and the whole task stalls waiting for the slowest partition.
- __Real-time operating systems__ in robotics and edge AI cannot afford to yield the main thread to a BLAS-managed pool.
  These systems need deterministic latency, not maximum throughput.

Instead, NumKong exposes __row-range parameters__ that let the caller partition work across any threading model.
For GEMM-shaped `dots_packed`, this is straightforward — pass a slice of A's rows and the full packed B to compute the corresponding slice of C.
For SYRK-shaped `dots_symmetric`, explicit `start_row` / `end_row` parameters control which rows of the symmetric output matrix a given thread computes.
The [GIL](https://docs.python.org/3/glossary.html#term-global-interpreter-lock) (Global Interpreter Lock) is released around every kernel call, making NumKong compatible with `concurrent.futures`, `multiprocessing`, or any other parallelism model:

```python
import concurrent.futures, numkong as nk, numpy as np

vectors, num_threads = np.random.randn(1000, 768).astype(np.float32), 4
output = nk.zeros((1000, 1000), dtype="float32")

def compute_slice(t):
    start = t * (len(vectors) // num_threads)
    end = start + len(vectors) // num_threads if t < num_threads - 1 else len(vectors)
    nk.dots_symmetric(vectors, out=output, start_row=start, end_row=end)

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
    list(pool.map(compute_slice, range(num_threads)))
```

For users who want a ready-made low-latency thread pool without the oversubscription baggage of OpenMP, we built [ForkUnion](https://github.com/ashvardanian/ForkUnion) — a minimalist fork-join library for C, C++, and Rust that avoids mutexes, CAS atomics, and dynamic allocations on the critical path, with optional NUMA pinning on Linux.

### Memory Allocation & Management

BLAS libraries typically allocate internal buffers during GEMM — [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) packs matrices into L2/L3-sized panels via per-thread buffer pools backed by `mmap` or `shmget`.
This hidden allocation has caused real problems: [14 lock/unlock pairs per small GEMM call](https://github.com/OpenMathLib/OpenBLAS/issues/478) throttling 12-thread scaling to 2x, [silently incorrect results](https://github.com/OpenMathLib/OpenBLAS/issues/1844) from thread-unsafe allocation in `np.dot`, and [deadlocks after `fork()`](https://github.com/numpy/numpy/issues/30092) due to mutex state not being reset in child processes.
The [BLASFEO](https://github.com/giaf/blasfeo) library was created specifically for embedded model-predictive control where `malloc` during computation is unacceptable.

NumKong __never allocates memory__.
Following the same philosophy as [Intel MKL's packed GEMM API](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html) (`cblas_sgemm_pack_get_size` → `cblas_sgemm_pack` → `cblas_sgemm_compute`), NumKong exposes typed three-phase interfaces — `nk_dots_packed_size_*` → `nk_dots_pack_*` → `nk_dots_packed_*` — where the caller owns the buffer and NumKong only fills it.

The owning containers follow the same caller-controls-allocation discipline across every binding.
In C++, Rust, Python, JavaScript, and Swift an owning tensor or vector is _fixed-capacity resizable_: `capacity` is the allocated ceiling, `try_resize`/`resize` reshapes within it without moving storage so handed-out views stay valid, `reserve` is the explicit opt-in that may reallocate to grow, and `clear` empties the shape while keeping capacity.

The reason GEMM libraries repack matrices at all is that every hardware target has a different preferred layout — Intel AMX expects B in a [VNNI-interleaved](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html) tile format (pairs of BFloat16 values packed into DWORDs across the K dimension), while Arm SME wants column vectors for its [FMOPA outer-product](https://developer.arm.com/documentation/ddi0602/latest/SME-Instructions) instructions.
Since GEMM is $O(N^3)$ and repacking is $O(N^2)$, the cost is asymptotically free — but the allocation and locking overhead is not.

NumKong's `nk_dots_pack_*` family performs five transformations beyond simple reordering:

- __Type pre-conversion__ — mini-floats (E4M3, BFloat16, etc.) are upcast to the compute type once during packing, not on every GEMM call.
  This amortizes the conversion cost across all rows of A that will be multiplied against the packed B.
- __SIMD depth padding__ — rows are zero-padded to the SIMD vector width (16 for AVX-512 Float32, 64 for AVX-512 Int8), allowing inner loops to load without boundary checks.
- __Per-column norm precomputation__ — squared norms ($\|b_j\|^2$) are computed and stored alongside the packed data, so distance kernels (`angulars_packed`, `euclideans_packed`) can reuse them without a separate pass.
- __ISA-specific tile layout__ — AMX packing interleaves BFloat16 pairs into 16×32 tiles matching `TDPBF16PS` expectations; SME packing arranges vectors at SVE granularity for `FMOPA` outer products; generic backends use simple column-major with depth padding.
- __Power-of-2 stride breaking__ — when the padded row stride is a power of 2, one extra SIMD step of padding is added.
  Power-of-2 strides cause [cache set aliasing](https://en.algorithmica.org/hpc/cpu-cache/associativity/) where consecutive rows map to the same cache sets, effectively shrinking usable L1/L2 capacity — stride-256 traversals can be [~10x slower](https://en.algorithmica.org/hpc/cpu-cache/associativity/) than stride-257.

```python
import numkong as nk, numpy as np

right_matrix = np.random.randn(1000, 768).astype(np.float16)
right_packed = nk.dots_pack(right_matrix, dtype=nk.float16)                        # pack once
for query_batch in stream: results = nk.dots_packed(query_batch, right_packed)    # reuse many times
```

### Why Not Just GEMM? The Evolution of Matrix Multiplication APIs

The classic BLAS GEMM computes $C = \alpha A B + \beta C$ for Float32/Float64 matrices.
It covers many use cases, but LLM inference, vector search, and quantum simulation expose three ways in which the traditional interface falls short.

__Frozen weights justify separating packing from computation.__
During LLM inference, a very large share of GEMM calls use a static weight matrix — weights don't change after loading.
This makes offline repacking a one-time cost amortized over the entire serving lifetime: [NVIDIA's TurboMind](https://arxiv.org/pdf/2508.15601) explicitly splits GEMM into offline weight packing (hardware-aware layout conversion) and online mixed-precision computation, and [Intel MKL's packed GEMM API](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html) exposes the same two-phase pattern.
NumKong's `nk_dots_pack_*` → `nk_dots_packed_*` path follows this philosophy — pack the weight matrix once, reuse it across all queries.

__Mixed precision demands more than an epilogue addition.__
Modern transformer layers operate in a precision sandwich: weights stored in BFloat16/Float8, [GEMM accumulated in Float32](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/), output downcast back to BFloat16 for the next layer.
Between GEMM calls, [LayerNorm or RMSNorm](https://arxiv.org/html/2409.12951v2) re-normalizes hidden states, so the next layer is often much closer to an angular or normalized similarity computation than to a plain raw dot product.
[nGPT](https://arxiv.org/html/2410.01131v1) takes this to its logical conclusion: all vectors live on the unit hypersphere, and every matrix-vector product is a pure angular distance.
This means many "GEMM" workloads in production are semantically closer to many-to-many angular distance computation — which is exactly what NumKong's `angulars_packed` and `euclideans_packed` kernels compute directly, fusing norm handling and type conversion into a single pass.

__The GEMM-for-distances trick has real costs.__
A common shortcut in vector search is to decompose pairwise Euclidean distance as $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \langle a, b \rangle$, precompute norms, and call `sgemm` for the inner-product matrix.
Both [FAISS](https://github.com/facebookresearch/faiss/wiki/Implementation-notes) and [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html) use this approach — and both document its limitations.
Scikit-learn's docs warn of _"catastrophic cancellation"_ in the subtraction; this has caused [real bugs](https://github.com/scikit-learn/scikit-learn/issues/9354) with ~37% error on near-identical Float32 vectors.
The $O(N^2)$ postprocessing pass (adding norms, square roots, divisions) is not free either — [NVIDIA's RAFT](https://github.com/rapidsai/raft/pull/339) measured a __20–25% speedup__ from fusing it into the GEMM epilogue.
Even [FAISS switches to direct SIMD](https://github.com/facebookresearch/faiss/wiki/Implementation-notes) when the query count drops below 20.
The standard BLAS interface was never designed for sub-byte types either — [no vendor supports Int4](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/), and sub-byte types cannot even be strided without bit-level repacking.

__Some operations need more than GEMM + postprocessing.__
NumKong implements several GEMM-shaped operations where the "epilogue" is too complex for a simple addition:

- __Bilinear forms__ ($a^T C b$) in quantum computing compute a [scalar expectation value](<https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Advanced_Quantum_Mechanics_(Kok)/10:_Pauli_Spin_Matrices/10.2:_Expectation_Values>) — the naive approach materializes an $N$-dimensional intermediate vector $Cb$, but NumKong's typed `nk_bilinear_*` kernels stream through rows of $C$ with nested compensated dot products, never allocating beyond registers.
  For complex-valued quantum states, where the intermediate would be a 2N-element complex vector, the savings double.
- __MaxSim scoring__ for [ColBERT-style late-interaction retrieval](https://github.com/stanford-futuredata/ColBERT) computes $\sum_i \min_j \text{angular}(q_i, d_j)$ — a sum-of-min-distances across token pairs.
  A GEMM would produce the full $M \times N$ similarity matrix, but NumKong's typed `nk_maxsim_packed_*` kernels fuse a coarse Int8-quantized screening with full-precision angular refinement on winning pairs only, packing both query and document matrices to use all 4 SME tiles as accumulators.
  [PLAID](https://ar5iv.labs.arxiv.org/html/2205.09707) and [maxsim-cpu](https://www.mixedbread.com/blog/maxsim-cpu) have independently shown that dedicated MaxSim kernels can outperform the GEMM decomposition by 5–10x.

NumKong treats these as first-class operations — `dots_packed`, `euclideans_packed`, `angulars_packed`, typed `nk_bilinear_*` kernels, and typed `nk_maxsim_packed_*` kernels — rather than decomposing everything into GEMM + postprocessing.

### Precision by Design: Saturation, Rounding, & Float6 Over Float8

Floating-point arithmetic on computers [is not associative](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html): $(a + b) + c \neq a + (b + c)$ in general, and upcasting to wider types is not always sufficient.
NumKong makes operation-specific decisions about where to spend precision and where to economize, rather than applying one rule uniformly.

__Saturation depends on the operation.__
A reduction over a 4 GB array of `i8` values contains ~4 billion elements — but [Int32 wrapping overflow](https://cedardb.com/blog/overflow_handling/) occurs after just ~17 million Int8 summands ($127 \times 16.9\text{M} > 2^{31}$).
Reductions in NumKong use saturating arithmetic because the input can be arbitrarily long.
Matrix multiplications don't need saturation because GEMM depth rarely exceeds tens of thousands — well within Int32 range.
x86 provides no saturating 32-bit SIMD add ([only byte/word variants](https://www.felixcloutier.com/x86/paddb:paddw:paddd:paddq)), so NumKong implements saturation via overflow detection with XOR-based unsigned comparison on platforms that lack native support.

__Square roots & special math ops are platform-specific.__
Angular distance requires $1/\sqrt{\|a\|^2 \cdot \|b\|^2}$ — but the cost of computing this normalization varies dramatically across hardware.
x86 `VSQRTPS` takes [~12 cycles](https://uops.info/html-lat/SKX/VSQRTPS_XMM_XMM-Measurements.html), followed by `VDIVPS` at ~11 cycles — totalling ~23 cycles for a precise `1/sqrt(x)`.
The `VRSQRT14PS` alternative starts with a [14-bit estimate in ~4 cycles](https://www.intel.com/content/www/us/en/developer/articles/code-sample/reference-implementations-for-ia-approximation-instructions-vrcp14-vrsqrt14-vrcp28-vrsqrt28-vexp2.html), then one Newton-Raphson iteration ($y = y \cdot (1.5 - 0.5 x y^2)$, ~4 more cycles) reaches full Float32 precision — roughly 3x faster.
ARM's `FRSQRTE` provides only [~8 bits](https://github.com/DLTcollab/sse2neon/issues/526), requiring __two__ Newton-Raphson iterations to match.
NumKong selects the iteration count per platform so the final ULP bound is consistent across ISAs, rather than exposing different precision to different users.

__E2M3 and E3M2 can outperform E4M3 and E5M2.__
6-bit [MX formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) can be scaled to exact integers, enabling integer accumulation that avoids E5M2's catastrophic cancellation risk.
This works because E2M3's narrower exponent range means every representable value maps to an integer after a fixed shift — no rounding, no cancellation.
See [Mini-Floats](#mini-floats-e4m3-e5m2-e3m2--e2m3) for a worked example.

Every such decision — saturation thresholds, Newton-Raphson iteration counts, integer vs floating-point paths — is documented per operation and per type in the [module-specific READMEs](include/numkong/).

### Calling Convention & Error Handling

NumKong never throws exceptions, never sets `errno`, and never calls `setjmp`/`longjmp` — [exceptions bloat call sites with unwind tables](https://monkeywritescode.blogspot.com/p/c-exceptions-under-hood.html) and are invisible to C, Python, Rust, Swift, Go, and JavaScript FFI; `errno` is thread-local state whose [storage model varies across C runtimes](https://en.cppreference.com/w/c/error/errno).
Instead, every function takes inputs as `const` pointers, writes outputs through caller-provided pointers, and returns `void`:

```c
void nk_dot_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
void nk_dot_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
```

Pointers eliminate implicit casts for types with platform-dependent storage — this is why they matter for half-precision types.
`nk_f16_t` and `nk_bf16_t` resolve to native `__fp16` / `__bf16` when available but fall back to `unsigned short` otherwise — if passed by value, the compiler would silently apply integer promotion instead of preserving the bit pattern.
Passing by pointer keeps the representation opaque: kernels read raw and convert explicitly when needed, so the same binary works regardless of whether the compiler understands `_Float16`.

The only place that requires error signaling is [dynamic dispatch](#compile-time-and-run-time-dispatch) — looking up the best kernel for the current CPU at runtime.
When no kernel matches, the dispatcher sets the [capabilities mask](c/dispatch.h) to zero and fills the function pointer with a family-specific error stub such as `nk_error_dense_` from [c/dispatch.h](c/dispatch.h) and [c/numkong.c](c/numkong.c) that writes `0xFF` into the output — `NaN` for floats, `−1` for signed integers, `TYPE_MAX` for unsigned.

### Compile-Time and Run-Time Dispatch

NumKong provides two dispatch mechanisms.
__Compile-time dispatch__ selects the fastest kernel supported by the target platform at build time — thinner binaries, no indirection overhead, but requires knowing your deployment hardware.
__Run-time dispatch__ compiles every supported kernel into the binary and picks the best one on the target machine via `nk_capabilities()` — one pointer indirection per call, but a single binary runs everywhere.
The run-time path is common in DBMS products (ClickHouse), web browsers (Chromium), and other upstream projects that ship to heterogeneous fleets.
Distributed artifacts (Rust crate, Python wheels, JS native modules, shared libs from the default CMake build) pin the translation-unit baseline to each architecture's ABI floor so the library runs on any CPU matching the ABI, not just the build host — see [CONTRIBUTING.md](CONTRIBUTING.md#target-baseline-policy) for the per-arch table and the `NK_MARCH_NATIVE` override used for host-tuned local builds.

All kernel names follow the pattern `nk_{operation}_{type}_{backend}`.
If you need to resolve the best kernel manually, use `nk_find_kernel_punned` with a `nk_kernel_kind_t`, `nk_dtype_t`, and a viable capabilities mask:

```c
nk_metric_dense_punned_t angular = 0;
nk_capability_t used = nk_cap_serial_k;
nk_find_kernel_punned(
    nk_kernel_angular_k, nk_f32_k,            // what functionality? for which input type?
    nk_capabilities(),                        // which capabilities are viable?
    (nk_kernel_punned_t *)&angular, &used);   // the kernel found and capabilties used!
```

The first call to `nk_capabilities()` initializes the dispatch table; all subsequent calls are lock-free.

## Numeric Types

### Float64 & Float32: IEEE Precision

__Float64__ — NumKong uses __compensated summation__ that tracks numerical errors separately.
On serial paths, we use __[Neumaier's algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements)__ (1974), an improvement over Kahan-Babuška that correctly handles cases where added terms are larger than the running sum, achieving $O(1)$ error growth instead of $O(n)$.
On SIMD paths with FMA support, we implement the __Dot2 algorithm__ (Ogita-Rump-Oishi, 2005), maintaining separate error compensators for both multiplication and accumulation via `TwoProd` and `TwoSum` operations.
The accuracy differences are visible in the [benchmark tables above](#latency-throughput--numerical-stability) — compensated Float64 suits scientific computing where numerical stability matters more than raw speed.

__Float32__ — SIMD implementations load Float32 values, upcast to Float64 for full-precision multiplication and accumulation, then downcast only during finalization.
This avoids catastrophic cancellation at minimal cost since modern CPUs have dedicated Float64 vector units operating at nearly the same throughput as Float32.
The same compensated accumulation strategy applies to Mahalanobis distance, bilinear forms, and KL/JS divergences.

```c
// Dot2 TwoProd: Capture multiplication rounding error
h = a * b;
r = fma(a, b, -h);  // Extracts rounding error

// Dot2 TwoSum: Capture addition rounding error
t = sum + product;
e = (sum - t) + product;  // Compensator term
```

### BFloat16 & Float16: Half Precision

__BFloat16__ — not an IEEE 754 standard type, but widely adopted for AI workloads.
BFloat16 shares Float32's 8-bit exponent but truncates the mantissa to 7 bits, prioritizing __dynamic range over precision__ (±3.4×10³⁸ with coarser granularity).
On old CPUs, upcasting BFloat16 to Float32 requires just an unpack and left-shift by 16 bits (essentially free); on newer CPUs, both Arm and x86 provide widening mixed-precision dot products via __DPBF16PS__ (AVX-512 on Genoa/Sapphire Rapids) and __BFDOT__ (NEON on ARMv8.6-A Graviton 3+).
NumKong's Float8 types (E4M3/E5M2) upcast to BFloat16 before using DPBF16PS, creating a three-tier precision hierarchy: Float8 for storage, BFloat16 for compute, Float32 for accumulation.

__Float16__ — IEEE 754 half-precision with 1 sign bit, 5 exponent bits (bias=15), and 10 mantissa bits, giving a range of ±65504.
Float16 prioritizes __precision over range__ (10 vs 7 mantissa bits), making it better suited for values near zero and gradients during training.
On x86, older CPUs use __F16C extensions__ (Ivy Bridge+) for fast Float16 → Float32 conversion; Sapphire Rapids+ adds native __AVX-512-FP16__ with dedicated Float16 arithmetic.
On Arm, ARMv8.4-A adds __FMLAL/FMLAL2__ instructions for fused Float16 → Float32 widening multiply-accumulate, reducing the total latency from 7 cycles to 4 cycles and achieving 20–48% speedup over the separate convert-then-FMA path.

| Platform         | BFloat16 Path              | Step | Float16 Path           | Step |
| :--------------- | :------------------------- | ---: | :--------------------- | ---: |
| __x86__          |                            |      |                        |      |
| Diamond, '26     | ↓ Genoa                    |   32 | `VDPPHPS` widening dot |   32 |
| Sapphire, '23    | ↓ Genoa                    |   32 | ↓ Skylake              |   16 |
| Genoa, '22       | `VDPBF16PS` widening dot   |   32 | ↓ Skylake              |   16 |
| Skylake, '15     | `SLLI` + `VFMADD`          |   16 | `VCVTPH2PS` + `VFMADD` |   16 |
| Haswell, '13     | `SLLI` + `VFMADD`          |    8 | `VCVTPH2PS` + `VFMADD` |    8 |
| __Arm__          |                            |      |                        |      |
| Apple M2+, '22   | `BFDOT` widening dot       |    8 | ↓ FP16FML              |    8 |
| Graviton 3+, '21 | `SVBFDOT` widening dot     | 4–32 | `SVCVT` → `SVFMLA`     | 4–32 |
| Apple M1, '20    | ↓ NEON                     |    8 | `FMLAL` widening FMA   |    8 |
| Graviton 2, '19  | ↓ NEON                     |    8 | `FCVTL` + `FMLA`       |    4 |
| Graviton 1, '18  | `SHLL` + `FMLA`            |    8 | bit-manip → `FMLA`     |    8 |
| __RISC-V__       |                            |      |                        |      |
| RVV+Zvfbfwma     | `VFWMACCBF16` widening FMA | 4–32 | ↓ RVV                  | 4–32 |
| RVV+Zvfh         | ↓ RVV                      | 4–32 | `VFWMACC` widening FMA | 4–32 |
| RVV              | shift + `VFMACC`           | 4–32 | convert + `VFMACC`     | 4–32 |

> BFloat16 shares Float32's 8-bit exponent, so upcasting is a 16-bit left shift (`SLLI` on x86, `SHLL` on Arm) that zero-pads the truncated mantissa — essentially free.
> Float16 has a different exponent width (5 vs 8 bits), requiring a dedicated convert: `VCVTPH2PS` (x86 F16C) or `FCVTL` (Arm NEON).
> Widening dot products (`VDPBF16PS`, `BFDOT`, `FMLAL`) fuse the conversion and multiply-accumulate into one instruction.
> Sapphire Rapids has native `VFMADDPH` for Float16 arithmetic, but NumKong does not use it for general dot products — Float16 accumulation loses precision.
> It is only used for mini-float (E2M3/E3M2) paths where periodic flush-to-Float32 windows keep error bounded.
> The table above covers only vector dot-product paths - GEMMs also leverage Arm SME and Intel AMX instructions.
> Beyond x86, Arm, and RISC-V, NumKong also ships LoongArch, WebAssembly, and PowerPC backends, also excluded from the table.

### Mini-Floats: E4M3, E5M2, E3M2, & E2M3

| Format       |  Bits |  Range | NumKong Promotion Rules               | Support in GPUs   |
| :----------- | ----: | -----: | ------------------------------------- | ----------------- |
| E5M2FN       |     8 | ±57344 | BFloat16 → Float32                    | H100+, MI300+     |
| E4M3FN       |     8 |   ±448 | BFloat16 → Float32                    | H100+, MI300+     |
| E3M2FN       | 6 → 8 |    ±28 | B- & Float16 → Float32, Int16 → Int32 | only block-scaled |
| E2M3FN       | 6 → 8 |   ±7.5 | B- & Float16 → Float32, Int8 → Int32  | only block-scaled |
| Scaled NVFP4 |     4 |     ±6 | —                                     | B200+             |
| Scaled MXFP4 |     4 |     ±6 | —                                     | B200+, MI325+     |

> __Block scaling.__
> NumKong does not implement block-scaled variants (MXFP4, NVFP4, or block-scaled E3M2/E2M3).
> Block scaling couples elements through a shared exponent per block, introducing structural bias into a fundamentally uniform operation.
> NumKong treats each element independently; block-scaled inputs should be dequantized before processing.

> __FNUZ variants.__
> AMD MI300 (CDNA 3) uses FNUZ encoding (negative-zero-is-NaN) rather than the OCP standard.
> MI350+ and NVIDIA H100/B200 both use OCP-standard E4M3FN/E5M2FN.
> NumKong follows the OCP convention; FNUZ inputs require conversion before processing.

__8-bit floats (E4M3 & E5M2)__ follow the [OCP FP8 standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).
E4M3FN (no infinities, NaN only) is preferred for __training__ where precision near zero matters; E5M2FN (with infinities) provides wider dynamic range for __inference__.
On x86 Genoa/Sapphire Rapids, E4M3/E5M2 values upcast to BFloat16 via lookup tables, then use native __DPBF16PS__ for 2-per-lane dot products accumulating to Float32.
On Arm Graviton 3+, the same BFloat16 upcast happens via NEON table lookups, then __BFDOT__ instructions complete the computation.

| Platform          | E5M2 Path                      | Step | E4M3 Path                      | Step |
| :---------------- | :----------------------------- | ---: | :----------------------------- | ---: |
| __x86__           |                                |      |                                |      |
| Diamond, '26      | `VCVTBF82PH` → F16 + `VDPPHPS` |   32 | `VCVTHF82PH` → F16 + `VDPPHPS` |   32 |
| Genoa, '22        | → BF16 + `VDPBF16PS`           |   32 | ↓ Ice Lake                     |   64 |
| Ice Lake, '19     | ↓ Skylake                      |   16 | octave LUT + `VPDPBUSD`        |   64 |
| Skylake, '15      | rebias → F32 FMA               |   16 | rebias → F32 FMA               |   16 |
| Haswell, '13      | rebias → F32 FMA               |    8 | rebias → F32 FMA               |    8 |
| __Arm__           |                                |      |                                |      |
| NEON+FP8DOT, '26  | native `FDOT`                  |   16 | native `FDOT`                  |   16 |
| NEON+FP16FML, '20 | SHL → F16 + `FMLAL`            |   16 | LUT → F16 + `FMLAL`            |   16 |
| NEON, '18         | SHL + `FCVTL` + FMA            |    8 | → F16 + `FCVTL` + FMA          |    8 |
| __RISC-V__        |                                |      |                                |      |
| RVV+Zvfbfwma      | rebias → BF16 + `VFWMACCBF16`  | 4–32 | LUT → BF16 + `VFWMACCBF16`     | 4–32 |
| RVV+Zvfh          | SHL → F16 + `VFWMACC`          | 4–32 | LUT → F16 + `VFWMACC`          | 4–32 |
| RVV               | rebias → F32 + `VFMACC`        | 4–32 | LUT → F32 + `VFMACC`           | 4–32 |

> E5M2 shares Float16's exponent bias (15), so E5M2 → Float16 conversion is a single left-shift by 8 bits (`SHL 8`).
> E4M3 on Ice Lake uses "octave decomposition": the 4-bit exponent splits into 2 octave + 2 remainder bits, yielding 7 integer accumulators post-scaled by powers of 2.

__6-bit floats (E3M2 & E2M3)__ follow the [OCP MX v1.0 standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
Their smaller range allows scaling to exact integers that fit in `i8`/`i16`, enabling integer `VPDPBUSD`/`SDOT` accumulation instead of the floating-point pipeline.
Float16 can also serve as an accumulator, accurately representing ~50 products of E3M2FN pairs or ~20 products of E2M3FN pairs before overflow.
On Arm, NEON FHM extensions bring widening `FMLAL` dot-products for Float16 — both faster and more widely available than `BFDOT` for BFloat16.

| Platform           | E3M2 Path                 | Step | E2M3 Path                | Step |
| :----------------- | :------------------------ | ---: | :----------------------- | ---: |
| __x86__            |                           |      |                          |      |
| Sierra Forest, '24 | ↓ Haswell                 |   32 | `VPSHUFB` + `VPDPBSSD`   |   32 |
| Alder Lake, '21    | ↓ Haswell                 |   32 | `VPSHUFB` + `VPDPBUSD`   |   32 |
| Ice Lake, '19      | `VPERMW` + `VPMADDWD`     |   32 | `VPERMB` + `VPDPBUSD`    |   64 |
| Skylake, '15       | `VPSHUFB` + `VPMADDWD`    |   64 | `VPSHUFB` + `VPMADDUBSW` |   64 |
| Haswell, '13       | `VPSHUFB` + `VPMADDWD`    |   32 | `VPSHUFB` + `VPMADDUBSW` |   32 |
| __Arm__            |                           |      |                          |      |
| NEON+FP8DOT, '26   | → E5M2 + `FDOT`           |   16 | → E4M3 + `FDOT`          |   16 |
| NEON+DotProd, '19  | `VQTBL2` + `SMLAL`        |   16 | `VQTBL2` + `SDOT`        |   16 |
| NEON, '18          | → F16 + `FCVTL` + FMA     |   16 | → F16 + `FCVTL` + FMA    |   16 |
| __RISC-V__         |                           |      |                          |      |
| RVV                | I16 gather LUT + `VWMACC` | 4–32 | U8 gather LUT + `VWMACC` | 4–32 |

> E3M2/E2M3 values map to exact integers via 32-entry LUTs (magnitudes up to 448 for E3M2, 120 for E2M3), enabling integer accumulation with no rounding error.
> On NEON+FP8DOT, E3M2 is first promoted to E5M2 and E2M3 to E4M3 before the hardware `FDOT` instruction.
> Sierra Forest and Alder Lake use native `VPDPBSSD` (signed×signed) and `VPDPBUSD` (unsigned×signed) respectively for E2M3.

E4M3 and E5M2 cannot use the integer path.
E4M3 scaled by 16 reaches 7,680 — too large for Int8, barely fitting Int16 with a 128-entry table.
E5M2's range (±57,344) makes the scaled product exceed Int32 entirely.
Without the integer path, E5M2 falls back to Float32 accumulation — where its [2-bit mantissa (only 4 values per binade)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) creates a [catastrophic cancellation risk](https://www.ac.uma.es/arith2024/papers/Fused%20FP8%204-Way%20Dot%20Product%20with%20Scaling%20and%20FP32%20Accumulation.pdf) that E2M3's integer path avoids completely:

|         |  _i_ = 0 | _i_ = 1 |  _i_ = 2 |   _i_ = 3 |  _i_ = 4 |  _i_ = 5 |  _i_ = 6 |
| :------ | -------: | ------: | -------: | --------: | -------: | -------: | -------: |
| _aᵢ_    |  0.00122 |   20480 | −0.00122 |       1.5 |    −3072 |     −640 |  0.00146 |
| _bᵢ_    |      −40 |     320 |    −1280 |  −7.63e⁻⁵ | 0.000427 |    10240 | −4.58e⁻⁵ |
| _aᵢ·bᵢ_ | −0.04883 | 6553600 |   1.5625 | −0.000114 |  −1.3125 | −6553600 |      ≈ 0 |

> __Why Float32 accumulation fails here.__
> The accurate sum of these 7 products is ≈ 0.201.
> A `vfmaq_f32` call accumulates 4 lanes at a time; the first batch already carries values around ±6.5 M.
> At that magnitude the Float32 ULP is 0.5 — so the small meaningful terms (−0.049, 1.563, −1.313, −0.0001) are all below one ULP and get absorbed during lane reduction.
> The large terms then cancel exactly to zero, and the information is gone.
> Final Float32 result: __0.0__ instead of __0.201__.

### Int8 & Int4: Integer Types

Both signed and unsigned 8-bit and 4-bit integers are supported with __Int32 accumulation__ to prevent overflow.
A notable optimization is the __VNNI algebraic transform__: on Ice Lake+ with AVX-512 VNNI, the native __DPBUSD__ instruction is asymmetric (unsigned × signed → signed), but NumKong uses it for both Int8×Int8 and UInt8×UInt8.
For __signed Int8×Int8__, we convert the signed operand to unsigned via XOR with `0x80`, compute `DPBUSD(a⊕0x80, b) = (a+128)×b`, then subtract a correction term `128×sum(b)` to recover the true result.
For __unsigned UInt8×UInt8__, we XOR the second operand to make it signed, compute `DPBUSD(a, b⊕0x80) = a×(b-128)`, then add correction `128×sum(a)` via the fast SAD instruction.

__Int4__ values pack two nibbles per byte, requiring bitmask extraction: low nibbles `(byte & 0x0F)` and high nibbles `(byte >> 4)`.
For signed Int4, the transformation `(nibble ⊕ 8) - 8` maps the unsigned range [0,15] to signed range [−8,7].
Separate accumulators for low and high nibbles avoid expensive nibble-interleaving and allow SIMD lanes to work in parallel.

```c
// Asymmetric transform for i8×i8 using DPBUSD (unsigned×signed)
a_unsigned = a XOR 0x80;           // Convert signed→unsigned
result = DPBUSD(a_unsigned, b);    // Computes (a+128)×b
correction = 128 * sum(b);         // Parallel on different port
final = result - correction;       // True a×b value
```

### Binary: Packed Bits

The `u1x8` type packs 8 binary values per byte, enabling __Hamming distance__ and __Jaccard similarity__ via population-count instructions.
On x86, `VPOPCNTDQ` (Ice Lake+) counts set bits in 512-bit registers directly; on Arm, `CNT` (NEON) operates on 8-bit lanes with a horizontal add.
Results accumulate into `u32` — sufficient for vectors up to 4 billion bits.
Binary representations are the most compact option for locality-sensitive hashing and binary neural network inference.

### Complex Types

NumKong supports four complex types — `f16c`, `bf16c`, `f32c`, and `f64c` — stored as interleaved real/imaginary pairs.
Complex types are essential in quantum simulation (state vectors, density matrices), signal processing (FFT coefficients, filter design), and electromagnetic modeling.
The `dot` operation computes the unconjugated dot product $\sum a_k b_k$, while `vdot` computes the conjugated inner product $\sum \bar{a}_k b_k$ standard in physics and signal processing.

For complex dot products, NumKong defers sign flips until after the accumulation loop: instead of using separate FMA and FMS (fused multiply-subtract) instructions for the real component, we compute $a_r b_r + a_i b_i$ treating all products as positive, then apply a single bitwise XOR with `0x80000000` to flip the sign bits.
This avoids execution port contention between FMA and FMS, letting dual FMA units stay occupied.

```c
for (...) { // Complex multiply optimization: XOR sign flip after the loop
    sum_real = fma(a, b, sum_real);   // No sign flip in loop
    sum_imag = fma(a, b_swapped, sum_imag);
}
sum_real = xor(sum_real, 0x80000000);  // Single XOR after loop
```

## Reading Materials

Beyond the READMEs in this repository, there are several standalone articles covering different evolution steps and features of this library.

- [NumKong: 2'000 Mixed Precision Kernels For All](https://ashvardanian.com/posts/numkong/)
- [Hiding x86 Port Latency for 330 GB/s/core Reductions](https://ashvardanian.com/posts/cpu-ports/)
- [Understanding SIMD: Infinite Complexity of Trivial Problems](https://ashvardanian.com/posts/understanding-simd-complexity/)
- [NumPy vs BLAS: Losing 90% of Throughput](https://ashvardanian.com/posts/numpy-vs-blas-costs/)
- [5x Faster Set Intersections: SVE2, AVX-512, & NEON](https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/)
- [Python, C, Assembly - 2'500x Faster Cosine Similarity](https://ashvardanian.com/posts/python-c-assembly-comparison/)
- [GCC Compiler vs Human - 119x Faster Assembly](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/)
- [Accelerating JavaScript arrays by 10x for Vector Search](https://ashvardanian.com/posts/javascript-ai-vector-search/)
- [SciPy distances... up to 200x faster with AVX-512 & SVE](https://ashvardanian.com/posts/simsimd-faster-scipy/)

## License

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.
