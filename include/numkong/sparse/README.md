# Sparse Vector Operations in NumKong

NumKong implements set intersection and weighted dot products for sparse vectors stored as sorted arrays of unique indices with optional associated weights.
Set intersection counts common elements between two sorted index arrays; sparse dot product sums the products of weights at matching indices.
Used in inverted-index search, sparse feature matching, and graph intersection queries.
The separate index/weight stream design makes these primitives composable into batched sparse operations and future sparse GEMM workloads.

Set intersection:

$$
|A \cap B| = |\{i : i \in A \land i \in B\}|
$$

Sparse dot product:

$$
\text{dot}(a, b) = \sum_{i \in A \cap B} w_a(i) \cdot w_b(i)
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def intersect(a_indices: np.ndarray, b_indices: np.ndarray) -> int:
    return len(np.intersect1d(a_indices, b_indices))

def sparse_dot(a_indices: np.ndarray, a_weights: np.ndarray,
               b_indices: np.ndarray, b_weights: np.ndarray) -> float:
    common = np.intersect1d(a_indices, b_indices, return_indices=True)
    return np.dot(a_weights[common[1]], b_weights[common[2]])
```

## Input & Output Types

| Input Type | Output Type | Description                                  |
| ---------- | ----------- | -------------------------------------------- |
| `u16`      | `u64`       | 16-bit index intersection count              |
| `u32`      | `u64`       | 32-bit index intersection count              |
| `u64`      | `u64`       | 64-bit index intersection count              |
| `u32+f32`  | `f32`       | Sparse dot with 32-bit indices, f32 weights  |
| `u16+bf16` | `f32`       | Sparse dot with 16-bit indices, bf16 weights |

## Optimizations

### Adaptive Merge vs Galloping Search

`nk_sparse_intersect_u32_serial` selects between linear merge and galloping (exponential) search based on length ratio: when `longer_length > 64 * shorter_length`, galloping search over the longer array is used.
Linear merge advances two pointers in lockstep at $O(|A| + |B|)$ using branch-free conditional increments: `i += ai < bj; j += ai >= bj` — no branch misprediction penalty.
Galloping binary-searches the longer array for each element of the shorter at $O(|A| \cdot \log |B|)$: an exponential probe doubles the search range until the target is bracketed, then binary search narrows within.
The crossover at 64x length ratio balances the per-element cost of binary search ($\log_2 |B|$ comparisons) against linear scan's single comparison per advance — the threshold was chosen empirically, as cache locality favors linear merge at moderate ratios.

### Broadcast-Compare SIMD Intersection on x86

`nk_sparse_intersect_u32_icelake` loads 16 indices from each array into ZMM registers, then rotates one register through multiple positions, comparing each rotation against the other with `VPCMPEQD` to test all 16x16 = 256 pairs.
The rotation approach uses `_mm512_shuffle_epi32` with permutation constants (`_MM_PERM_ADCB`, etc.) to cycle elements through comparison positions — contending for port 5 (~1cy per shuffle, ~3cy for `_mm512_alignr_epi32`).
Match counts are extracted via `_mm_popcnt_u32` on the comparison masks, accumulating intersection size without materializing matched elements.
Before each 16x16 comparison block, a fast overlap check (`a_max < b_min || b_max < a_min`) skips non-overlapping register loads entirely — critical for sparse workloads where most pairs have disjoint index ranges.
No native `_mm512_2intersect_epi16` instruction exists in any x86 ISA — UInt16 intersection must convert indices to UInt32 before comparison, halving effective throughput.

### VP2INTERSECT on AMD Turin

`nk_sparse_intersect_u32_turin` uses the `VP2INTERSECT` instruction (Zen5), which produces _two_ 16-bit masks in a single operation — one indicating which elements of A matched any element of B, and vice versa.
This replaces the entire shuffle-rotate-compare sequence from Ice Lake with a single instruction, eliminating port-5 contention entirely.
Even on Turin, UInt16 intersection requires zero-extending to UInt32 first — no `VP2INTERSECT` variant operates on 16-bit elements.
For UInt64, `_mm512_2intersect_epi64` processes 8x8 = 64 pairs per instruction — half the throughput of UInt32 but still far faster than the Ice Lake shuffle approach.

### SVE2 Set Membership via svmatch and svhistcnt

`nk_sparse_intersect_u16_sve2` uses the `svmatch_u16` instruction — true hardware set membership testing that matches each element against a 128-bit lane of candidates.
However, `svmatch` only operates on UInt8 and UInt16 — no UInt32 or UInt64 variant exists in SVE2.
For UInt32/UInt64, `nk_sparse_intersect_u32_sve2` uses `svhistcnt_u32_z` (histogram count): this computes a prefix-match count for each element against preceding elements in the combined register, and a reverse pass captures the upper triangle — ORing both halves yields the full intersection mask.
NEON (`nk_sparse_intersect_u32_neon`) lacks compress-store entirely — when intersection results must be materialized (not just counted), the kernel falls back to serial extraction, using `vclz_u32` (count leading zeros) to compute pointer advance steps from comparison masks.

### BFloat16 Weights in Sparse Dot Products

`nk_sparse_dot_u16bf16_sve2` loads BFloat16 weights alongside UInt16 indices, selecting matching weights via `svsel_s16` after intersection detection, then accumulating with `svbfdot_f32` — a single instruction that multiplies BFloat16 pairs and adds to a Float32 accumulator.
`nk_sparse_dot_u16bf16_turin` zero-extends UInt16 indices to UInt32 for `VP2INTERSECT`, then compresses matching BFloat16 weights with `VPCOMPRESSW` and accumulates via `_mm256_dpbf16_ps` (6cy latency on Genoa).
BFloat16 weights halve memory traffic compared to Float32 (16-bit vs 32-bit per weight) while preserving sufficient precision for learned sparse attention weights and embedding lookups.
The index/weight stream separation enables type-independent intersection (UInt16/UInt32/UInt64 indices) with type-specific accumulation (BFloat16 or Float32 weights) — the same intersection code path serves both weight types.

### Implications for Sparse GEMM

Current sparse operations handle inner-product dot products — one pair of sparse vectors at a time.
Extending to batched sparse GEMM (SpMM, SpGEMM) would require simultaneous intersection of multiple sparse vectors — the broadcast-compare pattern scales naturally, since one document vector's indices can be broadcast against multiple query vectors' indices in the same ZMM/SVE registers.
The 64x galloping threshold is tuned for individual vector pairs; batched workloads with different sparsity patterns per row would benefit from adaptive per-pair threshold selection.
Hardware support remains the bottleneck: no ISA provides native sparse outer-product instructions, and `VP2INTERSECT` exists only on AMD Zen5+ — Intel Tiger Lake's implementation had 36-41cy latency, making it slower than the manual shuffle approach on Ice Lake.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_SPARSE_FIRST_LENGTH`, `NK_SPARSE_SECOND_LENGTH`, and `NK_SPARSE_INTERSECTION` environment variables.
Columns show throughput at 1%, 50%, and 95% intersection ratio with both set lengths fixed at 4096.
The throughput is measured in GB/s as the number of input bytes per second.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.
Accuracy is reported as mean ULP (units in last place) averaged over all test pairs — the average number of representable floating-point values between the computed result and the exact answer.

### Intel Sapphire Rapids

#### Native

| Kernel                            |                       1% |                      50% |                      95% |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u64_serial`  |                2.96 gb/s |                3.06 gb/s |                3.27 gb/s |
| `nk_sparse_intersect_u64_icelake` |                3.64 gb/s |                3.83 gb/s |                3.74 gb/s |
| __u32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u32_serial`  |                1.51 gb/s |                1.55 gb/s |                1.69 gb/s |
| `nk_sparse_intersect_u32_icelake` |                4.15 gb/s |                4.29 gb/s |                4.25 gb/s |
| __u16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u16_serial`  |               0.747 gb/s |               0.793 gb/s |               0.824 gb/s |
| `nk_sparse_intersect_u16_icelake` |                3.06 gb/s |                3.09 gb/s |                3.10 gb/s |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_dot_u32f32_serial`     |       2.78 gb/s, 6.6 ulp |       2.78 gb/s, 5.9 ulp |       2.79 gb/s, 6.4 ulp |
| `nk_sparse_dot_u32f32_icelake`    |       7.80 gb/s, 3.8 ulp |         6.46 gb/s, 4 ulp |       5.86 gb/s, 3.8 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_dot_u16bf16_serial`    |        0.366 gb/s, 0 ulp |        0.364 gb/s, 0 ulp |        0.366 gb/s, 0 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                           |                       1% |                      50% |                      95% |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u64_serial` |                1.74 gb/s |                1.74 gb/s |                1.74 gb/s |
| __u32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u32_serial` |               0.492 gb/s |               0.492 gb/s |               0.492 gb/s |
| __u16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u16_serial` |               0.309 gb/s |               0.309 gb/s |               0.309 gb/s |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_dot_u32f32_serial`    |       1.87 gb/s, 9.1 ulp |       1.87 gb/s, 9.1 ulp |       1.87 gb/s, 9.1 ulp |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_dot_u16bf16_serial`   |        0.927 gb/s, 0 ulp |        0.927 gb/s, 0 ulp |        0.927 gb/s, 0 ulp |

### Apple M4

#### Native

| Kernel                           |                       1% |                      50% |                      95% |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u64_serial` |                3.74 gb/s |                3.96 gb/s |                4.17 gb/s |
| `nk_sparse_intersect_u64_neon`   |                4.91 gb/s |                4.61 gb/s |                4.46 gb/s |
| __u32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u32_serial` |                1.88 gb/s |                1.98 gb/s |                2.09 gb/s |
| `nk_sparse_intersect_u32_neon`   |                2.81 gb/s |                2.60 gb/s |                2.46 gb/s |
| __u16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_intersect_u16_serial` |               0.951 gb/s |               0.995 gb/s |                1.05 gb/s |
| `nk_sparse_intersect_u16_neon`   |                1.86 gb/s |                1.85 gb/s |                1.85 gb/s |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_dot_u32f32_serial`    |       3.73 gb/s, 7.1 ulp |       3.74 gb/s, 7.1 ulp |       3.79 gb/s, 7.0 ulp |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sparse_dot_u16bf16_serial`   |         1.73 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |
