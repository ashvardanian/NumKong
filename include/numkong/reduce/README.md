# Horizontal Reductions in NumKong

NumKong implements single-pass horizontal reductions over dense vectors: statistical moments (sum + sum-of-squares) and extrema (min + max with argmin + argmax).
Both reductions traverse the input once, producing scalar outputs with compensated arithmetic for numerical stability.
The only module with full stride support — `stride_bytes` controls the byte distance between consecutive logical elements, enabling column extraction from row-major matrices and strided array views without copying.
Used internally by packing routines for norm precomputation and by distance kernels for normalization.

Moments:

$$
\text{sum} = \sum a_i, \quad \text{sumsq} = \sum a_i^2
$$

Min-max:

$$
\text{min} = \min_i a_i, \quad \text{argmin} = \arg\min_i a_i
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def moments(a: np.ndarray) -> tuple[float, float]:
    return np.sum(a), np.sum(a ** 2)

def minmax(a: np.ndarray) -> tuple[float, int, float, int]:
    return np.min(a), np.argmin(a), np.max(a), np.argmax(a)
```

## Input & Output Types

The Output Type column lists the accumulator type of `nk_reduce_moments_*`.
`nk_reduce_minmax_*` instead reports extrema in the input type — `i4`, `u4`, and `u1` unpack to `i8`, `u8`, and `u8` — alongside two `nk_size_t` indices.

Float reductions:

| Input Type | Output Type | Description                             |
| ---------- | ----------- | --------------------------------------- |
| `f64`      | `f64`       | 64-bit double precision                 |
| `f32`      | `f64`       | 32-bit single precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output      |
| `f16`      | `f32`       | 16-bit half precision, widened output   |

Mini-float reductions:

| Input Type | Output Type | Description                                  |
| ---------- | ----------- | -------------------------------------------- |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits    |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits    |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits |

Integer reductions:

| Input Type | Output Type | Description                        |
| ---------- | ----------- | ---------------------------------- |
| `i64`      | `i64`       | 64-bit signed                      |
| `i32`      | `i64`       | 32-bit signed, widened to 64-bit   |
| `i16`      | `i64`       | 16-bit signed, widened to 64-bit   |
| `i8`       | `i64`       | 8-bit signed, widened to 64-bit    |
| `u64`      | `u64`       | 64-bit unsigned                    |
| `u32`      | `u64`       | 32-bit unsigned, widened to 64-bit |
| `u16`      | `u64`       | 16-bit unsigned, widened to 64-bit |
| `u8`       | `u64`       | 8-bit unsigned, widened to 64-bit  |

Sub-byte reductions:

| Input Type | Output Type | Description                               |
| ---------- | ----------- | ----------------------------------------- |
| `i4`       | `i64`       | 4-bit signed nibbles, widened to 64-bit   |
| `u4`       | `u64`       | 4-bit unsigned nibbles, widened to 64-bit |
| `u1`       | `u64`       | 1-bit binary packed octets                |

## Optimizations

### Strided Access Across Backends

Reductions accept a `stride_bytes` parameter specifying the byte distance between consecutive logical elements — the only NumKong module where loads far outnumber stores (N loads, 2-4 scalar stores), making arbitrary strides practical.
Serial iterates with byte-pointer arithmetic: `ptr += stride_bytes` per element.
NEON uses hardware de-interleaving loads (`vld2q_f32`, `vld3q_f32`, `vld4q_f32`) for small integer strides (2-4 elements apart), extracting column 0 from interleaved data in a single instruction.
Haswell/Skylake use blend masks for small strides and `_mm256_i32gather_ps` / `_mm512_i32gather_pd` hardware gathers for larger strides — 8cy per gather on Haswell, ~5cy on Skylake for 16-element gathers.
RVV uses native strided loads (`__riscv_vlse32_v_f32m1`) that accept arbitrary byte strides directly in the load instruction — no gather overhead, no stride-dependent branching.

### Kahan-Neumaier Compensated Summation

`nk_reduce_moments_f32_serial`, `nk_reduce_moments_f64_serial` use Neumaier's variant of Kahan summation — maintaining a running compensation term that captures rounding errors.
Standard pairwise summation accumulates $O(\sqrt{n})$ ULP error for n elements; Neumaier compensation bounds error to $O(1)$ ULP regardless of vector length.
The serial path uses Neumaier's adaptive branch: `if (abs(sum) >= abs(val))` selects the larger summand first, minimizing relative error in the compensation term.
SIMD backends (`nk_reduce_moments_f64_haswell`, `nk_reduce_moments_f64_skylake`) carry 4 or 8 independent compensation lanes in a YMM or ZMM register — computing `round_error = tentative - sum; correction = (sum - (tentative - round_error)) + (val - round_error)` without branches, folding all lanes into a single scalar correction at the end.
The Float32 SIMD kernels skip compensation: they widen every input to Float64 first, and the wider accumulator already absorbs the rounding error.

### Fused Moments in a Single Pass

`nk_reduce_moments_f32_haswell`, `nk_reduce_moments_f64_skylake` compute sum and sum-of-squares simultaneously — one load feeds both a sum accumulator and a square accumulator.
On Haswell the Float32 inputs are widened with `VCVTPS2PD`, then folded in with `VADDPD` and `VFMADD231PD`; on Skylake the Float64 path squares with `VMULPD` so the product can go through the compensated add.
Two accumulators share the same loaded data, halving memory bandwidth compared to separate sum + norm passes.
The squared-norm $\|a\|^2 = \sum a_i^2$ is a self-dot-product, reused by packing routines (`nk_dots_pack_f32_haswell`) to precompute per-vector norms during layout transformation.
For Float16/BFloat16/Float8 inputs, all backends widen to Float32 before accumulation — NEON FHM (`nk_reduce_moments_e4m3_neonfhm`) converts e4m3->f16 by bit manipulation, then uses `vfmlalq_low_f16` and `vfmlalq_high_f16` to fuse the Float16 → Float32 widening with the FMA into the Float32 accumulator.

### Integer Saturation in Sum-of-Squares

Integer moments accumulate sums in the widest available type: Int8/UInt8/Int16 inputs produce Int64/UInt64 outputs.
Sums use widening addition chains — NEON uses pairwise widening (`vpaddlq_s16` -> UInt32 -> UInt64 stages); Haswell biases Int8 inputs with 0x80 and uses unsigned SAD (`_mm256_sad_epu8`) for the sum, correcting by subtracting $128 \times \text{count}$ at the end.
Sum-of-squares can overflow UInt64 when squaring Int64/UInt64 inputs — only `nk_reduce_moments_i64_*` and `nk_reduce_moments_u64_*` need an explicit saturating multiply: it checks whether `abs(val) < 2^32` (so the square fits in UInt64) and otherwise saturates to I64_MAX for signed inputs, U64_MAX for unsigned ones.
Narrower inputs cannot overflow the product itself, so they only need the saturating accumulate.
Haswell emulates UInt64 saturating add via XOR-based unsigned comparison: flip sign bits to convert unsigned overflow detection into a signed comparison, then OR with the overflow mask to produce all-ones on saturation.

### Recursive Blocking for Counter Overflow

All SIMD backends use loop iteration counters narrower than `nk_size_t` to save register pressure — UInt8 for Int8 minmax lanes, UInt16 for Float32 moments lanes.
When `count` exceeds the counter's range x lane count (e.g., Haswell Float32 moments: $65536 \times 8 = 524288$ elements for UInt16 counters), the reduction splits recursively: process the left half, process the right half, combine results with saturating arithmetic.
Block caps vary by backend and element width: Haswell Int8 minmax uses UInt8 loop counters (cap = $256 \times 32 = 8192$); Skylake Float32 moments uses UInt16 counters (cap = $65536 \times 16 = 1048576$).
The recursive split is invisible to the caller — the public API accepts arbitrary `count` values; internal dispatch chooses between single-pass and recursive based on the cap.

### Index Tracking at Different Register Scales

Argmin/argmax requires tracking both values and their positions — but indices need wider storage than values (UInt64 for arbitrary-length vectors, vs UInt8/UInt16/Float32 for data).
Haswell Int8 minmax tracks iteration counters in UInt8 lanes (same width as data) — after the loop, the winning lane's counter is multiplied by the lane count and added to the lane index within the register to reconstruct the global position.
RVV uses u64m2 registers (LMUL=2) for indices alongside f32m1 for values — the wider index register holds one 64-bit position per Float32 lane, enabling direct merge without post-loop reconstruction.
NEON uses same-width counters (u8x16 for i8x16 minmax), limiting block size to $256 \times 16 = 4096$ elements before recursive splitting.

### NaN-Aware Extrema Tracking

`nk_reduce_minmax_f32_haswell`, `nk_reduce_minmax_f64_skylake` use IEEE ordered-quiet comparisons (`_CMP_LT_OQ`, `_CMP_GT_OQ`) — returning false when either operand is NaN, so NaN inputs never replace the running extremum.
Tail elements beyond the vector-aligned portion are blended into a quiet-NaN register — `nk_reduce_minmax_f64_haswell` partial-loads the tail and `_mm256_blendv_pd`-s it over a NaN vector, so the padding lanes cannot win any comparison; Skylake instead predicates the tail comparison with the same `__mmask` used for the load, and `nk_reduce_minmax_f32_haswell` finishes its tail in scalar code.
If all inputs are NaN, the sentinels remain (min = F32_MAX, max = F32_MIN) and indices are set to NK_SIZE_MAX, signaling no valid extremum.
The final horizontal reduction across lanes uses pairwise `VSHUFPS` + `VMINPS` chains — 3 shuffles for a 256-bit register, $O(\log_2 w)$ for width $w$.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                           |                      256 |                     1024 |                     4096 |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`   |         1.47 gb/s, 0 ulp |         1.73 gb/s, 0 ulp |         1.95 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`    |         6.59 gb/s, 0 ulp |         5.95 gb/s, 0 ulp |         5.82 gb/s, 0 ulp |
| `nk_reduce_moments_f64_haswell`  |       10.8 gb/s, 0.1 ulp |         9.18 gb/s, 0 ulp |         6.05 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_haswell`   |         8.11 gb/s, 0 ulp |         9.45 gb/s, 0 ulp |         6.59 gb/s, 0 ulp |
| `nk_reduce_moments_f64_skylake`  |       14.7 gb/s, 0.3 ulp |       13.9 gb/s, 0.1 ulp |         11.4 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_skylake`   |         9.02 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |         9.93 gb/s, 0 ulp |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`   |        0.458 gb/s, 0 ulp |        0.437 gb/s, 0 ulp |        0.449 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`    |         3.35 gb/s, 0 ulp |         3.04 gb/s, 0 ulp |         3.27 gb/s, 0 ulp |
| `nk_reduce_moments_f32_haswell`  |       18.4 gb/s, 0.8 ulp |       17.8 gb/s, 4.2 ulp |       11.7 gb/s, 7.7 ulp |
| `nk_reduce_minmax_f32_haswell`   |         8.18 gb/s, 0 ulp |         8.92 gb/s, 0 ulp |         8.24 gb/s, 0 ulp |
| `nk_reduce_moments_f32_skylake`  |       20.7 gb/s, 0.4 ulp |       20.3 gb/s, 3.1 ulp |       17.1 gb/s, 8.8 ulp |
| `nk_reduce_minmax_f32_skylake`   |         7.35 gb/s, 0 ulp |         15.9 gb/s, 0 ulp |         21.8 gb/s, 0 ulp |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`  |        0.208 gb/s, 0 ulp |        0.245 gb/s, 0 ulp |        0.239 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_serial`   |        0.935 gb/s, 0 ulp |        0.984 gb/s, 0 ulp |         1.00 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_haswell` |         11.4 gb/s, 0 ulp |         12.2 gb/s, 0 ulp |       10.8 gb/s, 1.6 ulp |
| `nk_reduce_minmax_bf16_haswell`  |         4.98 gb/s, 0 ulp |         7.54 gb/s, 0 ulp |         9.30 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_skylake` |         18.2 gb/s, 0 ulp |         27.0 gb/s, 0 ulp |       17.9 gb/s, 0.7 ulp |
| `nk_reduce_minmax_bf16_skylake`  |         6.53 gb/s, 0 ulp |         18.2 gb/s, 0 ulp |         13.7 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_genoa`   |         18.1 gb/s, 0 ulp |         20.5 gb/s, 0 ulp |       19.3 gb/s, 0.8 ulp |
| __f16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`   |        0.391 gb/s, 0 ulp |        0.354 gb/s, 0 ulp |        0.407 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_serial`    |        0.901 gb/s, 0 ulp |        0.877 gb/s, 0 ulp |        0.974 gb/s, 0 ulp |
| `nk_reduce_moments_f16_haswell`  |         13.5 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |       11.0 gb/s, 0.3 ulp |
| `nk_reduce_minmax_f16_haswell`   |         6.61 gb/s, 0 ulp |         9.19 gb/s, 0 ulp |         8.10 gb/s, 0 ulp |
| `nk_reduce_moments_f16_skylake`  |         17.7 gb/s, 0 ulp |       29.1 gb/s, 0.1 ulp |         18.6 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_skylake`   |         10.2 gb/s, 0 ulp |         20.8 gb/s, 0 ulp |         22.0 gb/s, 0 ulp |
| __e5m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`  |        0.157 gb/s, 0 ulp |        0.296 gb/s, 0 ulp |        0.229 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`   |        0.418 gb/s, 0 ulp |        0.417 gb/s, 0 ulp |        0.451 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_haswell` |         2.40 gb/s, 0 ulp |         2.69 gb/s, 0 ulp |         2.61 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_haswell`  |         4.48 gb/s, 0 ulp |         6.80 gb/s, 0 ulp |         7.21 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_skylake` |         4.66 gb/s, 0 ulp |         2.83 gb/s, 0 ulp |         4.04 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_skylake`  |         3.90 gb/s, 0 ulp |         11.8 gb/s, 0 ulp |         19.1 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_genoa`   |         4.76 gb/s, 0 ulp |         6.08 gb/s, 0 ulp |         5.88 gb/s, 0 ulp |
| __e4m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`  |        0.121 gb/s, 0 ulp |        0.129 gb/s, 0 ulp |        0.158 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`   |        0.460 gb/s, 0 ulp |        0.473 gb/s, 0 ulp |        0.464 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_haswell` |         1.82 gb/s, 0 ulp |         1.90 gb/s, 0 ulp |         1.77 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_haswell`  |         4.42 gb/s, 0 ulp |         7.00 gb/s, 0 ulp |         8.10 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_skylake` |         2.77 gb/s, 0 ulp |         3.53 gb/s, 0 ulp |         2.74 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_skylake`  |         3.79 gb/s, 0 ulp |         9.57 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_genoa`   |         4.67 gb/s, 0 ulp |         5.87 gb/s, 0 ulp |         5.67 gb/s, 0 ulp |
| __e3m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`  |        0.158 gb/s, 0 ulp |        0.279 gb/s, 0 ulp |        0.348 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`   |        0.464 gb/s, 0 ulp |        0.416 gb/s, 0 ulp |        0.470 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_haswell` |         2.37 gb/s, 0 ulp |         2.55 gb/s, 0 ulp |         2.53 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_haswell`  |         5.36 gb/s, 0 ulp |         7.89 gb/s, 0 ulp |         9.56 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_skylake` |         2.77 gb/s, 0 ulp |         3.32 gb/s, 0 ulp |         3.58 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_skylake`  |         9.85 gb/s, 0 ulp |         20.1 gb/s, 0 ulp |         14.6 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_icelake` |         8.82 gb/s, 0 ulp |         9.02 gb/s, 0 ulp |         13.4 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_alder`   |         4.80 gb/s, 0 ulp |         7.11 gb/s, 0 ulp |         7.89 gb/s, 0 ulp |
| __e2m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`  |        0.157 gb/s, 0 ulp |        0.294 gb/s, 0 ulp |        0.301 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`   |        0.465 gb/s, 0 ulp |        0.421 gb/s, 0 ulp |        0.453 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_haswell` |         2.43 gb/s, 0 ulp |         2.45 gb/s, 0 ulp |         2.58 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_haswell`  |         5.31 gb/s, 0 ulp |         7.90 gb/s, 0 ulp |         9.36 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_skylake` |         3.49 gb/s, 0 ulp |         3.02 gb/s, 0 ulp |         3.66 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_skylake`  |         6.14 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_icelake` |         12.7 gb/s, 0 ulp |         22.7 gb/s, 0 ulp |         21.7 gb/s, 0 ulp |
| __i8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`    |                2.21 gb/s |                2.40 gb/s |                2.29 gb/s |
| `nk_reduce_minmax_i8_serial`     |               0.806 gb/s |               0.973 gb/s |                1.09 gb/s |
| `nk_reduce_moments_i8_haswell`   |                9.35 gb/s |                11.9 gb/s |                12.7 gb/s |
| `nk_reduce_minmax_i8_haswell`    |                7.11 gb/s |                11.7 gb/s |                13.2 gb/s |
| `nk_reduce_moments_i8_skylake`   |                10.4 gb/s |                16.6 gb/s |                20.1 gb/s |
| `nk_reduce_minmax_i8_skylake`    |                2.96 gb/s |                14.4 gb/s |                15.5 gb/s |
| `nk_reduce_moments_i8_icelake`   |                14.0 gb/s |                28.3 gb/s |                28.4 gb/s |
| __u8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`    |                2.40 gb/s |                2.49 gb/s |                2.15 gb/s |
| `nk_reduce_minmax_u8_serial`     |               0.776 gb/s |               0.931 gb/s |                1.05 gb/s |
| `nk_reduce_moments_u8_haswell`   |                10.3 gb/s |                12.9 gb/s |                13.6 gb/s |
| `nk_reduce_minmax_u8_haswell`    |                7.08 gb/s |                11.2 gb/s |                12.0 gb/s |
| `nk_reduce_moments_u8_skylake`   |                13.2 gb/s |                20.1 gb/s |                19.6 gb/s |
| `nk_reduce_minmax_u8_skylake`    |                4.45 gb/s |                14.0 gb/s |                20.4 gb/s |
| `nk_reduce_moments_u8_icelake`   |                14.6 gb/s |                21.7 gb/s |                30.4 gb/s |
| `nk_reduce_moments_u8_alder`     |                11.5 gb/s |                13.3 gb/s |                13.7 gb/s |
| __i4__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i4_serial`    |               0.345 gb/s |               0.757 gb/s |               0.752 gb/s |
| `nk_reduce_minmax_i4_serial`     |               0.313 gb/s |               0.285 gb/s |               0.357 gb/s |
| `nk_reduce_moments_i4_haswell`   |                6.36 gb/s |                9.17 gb/s |                10.3 gb/s |
| `nk_reduce_moments_i4_skylake`   |                7.67 gb/s |                8.85 gb/s |                15.4 gb/s |
| __u4__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u4_serial`    |               0.438 gb/s |               0.799 gb/s |                1.00 gb/s |
| `nk_reduce_minmax_u4_serial`     |               0.352 gb/s |               0.292 gb/s |               0.397 gb/s |
| `nk_reduce_moments_u4_haswell`   |                7.40 gb/s |                10.7 gb/s |                10.8 gb/s |
| `nk_reduce_moments_u4_skylake`   |                9.45 gb/s |                15.0 gb/s |                18.3 gb/s |
| __u1__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u1_serial`    |                1.36 gb/s |                1.96 gb/s |                2.04 gb/s |
| `nk_reduce_minmax_u1_serial`     |                5.44 gb/s |                14.7 gb/s |                84.1 gb/s |
| `nk_reduce_moments_u1_haswell`   |                4.29 gb/s |                9.69 gb/s |                12.0 gb/s |
| `nk_reduce_moments_u1_skylake`   |                2.90 gb/s |                12.3 gb/s |                20.6 gb/s |
| __i16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`   |                2.54 gb/s |                2.68 gb/s |                2.80 gb/s |
| `nk_reduce_minmax_i16_serial`    |                1.60 gb/s |                1.75 gb/s |                2.07 gb/s |
| `nk_reduce_moments_i16_haswell`  |                13.7 gb/s |                14.7 gb/s |                12.5 gb/s |
| `nk_reduce_minmax_i16_haswell`   |                8.56 gb/s |                10.9 gb/s |                10.0 gb/s |
| `nk_reduce_moments_i16_skylake`  |                16.8 gb/s |                21.0 gb/s |                20.5 gb/s |
| `nk_reduce_minmax_i16_skylake`   |                6.74 gb/s |                15.9 gb/s |                19.1 gb/s |
| `nk_reduce_moments_i16_icelake`  |                19.0 gb/s |                24.9 gb/s |                28.2 gb/s |
| `nk_reduce_moments_i16_alder`    |                10.0 gb/s |                12.1 gb/s |                10.5 gb/s |
| __u16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`   |                2.62 gb/s |                2.55 gb/s |                2.54 gb/s |
| `nk_reduce_minmax_u16_serial`    |                1.28 gb/s |                1.41 gb/s |                1.62 gb/s |
| `nk_reduce_moments_u16_haswell`  |                6.82 gb/s |                6.95 gb/s |                6.60 gb/s |
| `nk_reduce_minmax_u16_haswell`   |                8.25 gb/s |                10.5 gb/s |                11.6 gb/s |
| `nk_reduce_moments_u16_skylake`  |                10.2 gb/s |                13.9 gb/s |                12.6 gb/s |
| `nk_reduce_minmax_u16_skylake`   |                16.0 gb/s |                22.6 gb/s |                16.9 gb/s |
| `nk_reduce_moments_u16_alder`    |                7.17 gb/s |                8.10 gb/s |                7.57 gb/s |
| __i32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`   |                2.39 gb/s |                2.25 gb/s |                2.32 gb/s |
| `nk_reduce_minmax_i32_serial`    |                2.99 gb/s |                3.67 gb/s |                4.48 gb/s |
| `nk_reduce_moments_i32_haswell`  |                5.43 gb/s |                5.37 gb/s |                4.41 gb/s |
| `nk_reduce_minmax_i32_haswell`   |                11.1 gb/s |                10.2 gb/s |                10.4 gb/s |
| `nk_reduce_moments_i32_skylake`  |                6.87 gb/s |                11.1 gb/s |                10.6 gb/s |
| `nk_reduce_minmax_i32_skylake`   |                23.8 gb/s |                24.7 gb/s |                17.6 gb/s |
| __u32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`   |                3.46 gb/s |                3.53 gb/s |                3.41 gb/s |
| `nk_reduce_minmax_u32_serial`    |                2.81 gb/s |                3.34 gb/s |                4.05 gb/s |
| `nk_reduce_moments_u32_haswell`  |                6.10 gb/s |                5.79 gb/s |                5.27 gb/s |
| `nk_reduce_minmax_u32_haswell`   |                10.6 gb/s |                11.2 gb/s |                9.95 gb/s |
| `nk_reduce_moments_u32_skylake`  |                15.9 gb/s |                9.96 gb/s |                15.3 gb/s |
| `nk_reduce_minmax_u32_skylake`   |                23.6 gb/s |                25.3 gb/s |                21.7 gb/s |
| __i64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`   |                2.43 gb/s |                2.43 gb/s |                2.43 gb/s |
| `nk_reduce_minmax_i64_serial`    |                4.90 gb/s |                5.54 gb/s |                6.10 gb/s |
| `nk_reduce_moments_i64_haswell`  |                7.16 gb/s |                6.54 gb/s |                5.38 gb/s |
| `nk_reduce_minmax_i64_haswell`   |                9.50 gb/s |                9.87 gb/s |                7.63 gb/s |
| `nk_reduce_moments_i64_skylake`  |                13.0 gb/s |                8.29 gb/s |                10.5 gb/s |
| `nk_reduce_minmax_i64_skylake`   |                11.6 gb/s |                23.1 gb/s |                22.0 gb/s |
| __u64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`   |                1.94 gb/s |                1.92 gb/s |                1.83 gb/s |
| `nk_reduce_minmax_u64_serial`    |                5.99 gb/s |                7.20 gb/s |                7.33 gb/s |
| `nk_reduce_moments_u64_haswell`  |                8.60 gb/s |                8.45 gb/s |                5.96 gb/s |
| `nk_reduce_minmax_u64_haswell`   |                8.93 gb/s |                9.81 gb/s |                7.55 gb/s |
| `nk_reduce_moments_u64_skylake`  |                15.6 gb/s |                19.3 gb/s |                8.87 gb/s |
| `nk_reduce_minmax_u64_skylake`   |                9.90 gb/s |                23.1 gb/s |                21.6 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                      256 |                     1024 |                     4096 |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`       |          1.2 gb/s, 0 ulp |         1.22 gb/s, 0 ulp |         1.19 gb/s, 0 ulp |
| `nk_reduce_moments_f64_v128relaxed`  |         6.97 gb/s, 0 ulp |         6.91 gb/s, 0 ulp |         7.06 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`        |         5.23 gb/s, 0 ulp |         5.49 gb/s, 0 ulp |         5.71 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_v128relaxed`   |          4.9 gb/s, 0 ulp |         5.16 gb/s, 0 ulp |         5.11 gb/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`       |        0.606 gb/s, 0 ulp |        0.622 gb/s, 0 ulp |        0.618 gb/s, 0 ulp |
| `nk_reduce_moments_f32_v128relaxed`  |       8.84 gb/s, 0.1 ulp |       6.78 gb/s, 0.4 ulp |         6.49 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`        |         2.63 gb/s, 0 ulp |         2.73 gb/s, 0 ulp |         2.86 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_v128relaxed`   |         4.59 gb/s, 0 ulp |         5.02 gb/s, 0 ulp |         5.05 gb/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`      |        0.274 gb/s, 0 ulp |         0.28 gb/s, 0 ulp |        0.288 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_v128relaxed` |          7.7 gb/s, 0 ulp |       6.06 gb/s, 0.1 ulp |       5.26 gb/s, 1.6 ulp |
| `nk_reduce_minmax_bf16_serial`       |        0.873 gb/s, 0 ulp |        0.977 gb/s, 0 ulp |         1.04 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_v128relaxed`  |         4.88 gb/s, 0 ulp |         5.14 gb/s, 0 ulp |         5.55 gb/s, 0 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`       |        0.234 gb/s, 0 ulp |        0.235 gb/s, 0 ulp |        0.247 gb/s, 0 ulp |
| `nk_reduce_moments_f16_v128relaxed`  |         1.91 gb/s, 0 ulp |         1.95 gb/s, 0 ulp |       1.92 gb/s, 0.2 ulp |
| `nk_reduce_minmax_f16_serial`        |        0.859 gb/s, 0 ulp |        0.967 gb/s, 0 ulp |         1.06 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_v128relaxed`   |          1.3 gb/s, 0 ulp |         1.38 gb/s, 0 ulp |         1.39 gb/s, 0 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`      |        0.116 gb/s, 0 ulp |        0.118 gb/s, 0 ulp |        0.122 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_v128relaxed` |         1.55 gb/s, 0 ulp |         1.59 gb/s, 0 ulp |         1.56 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`       |        0.471 gb/s, 0 ulp |        0.499 gb/s, 0 ulp |        0.527 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_v128relaxed`  |         1.27 gb/s, 0 ulp |         2.76 gb/s, 0 ulp |         3.07 gb/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`      |        0.109 gb/s, 0 ulp |        0.109 gb/s, 0 ulp |        0.111 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_v128relaxed` |         1.15 gb/s, 0 ulp |         1.16 gb/s, 0 ulp |         1.17 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`       |        0.451 gb/s, 0 ulp |        0.495 gb/s, 0 ulp |        0.535 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_v128relaxed`  |         1.43 gb/s, 0 ulp |         2.61 gb/s, 0 ulp |         3.06 gb/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`      |        0.121 gb/s, 0 ulp |        0.117 gb/s, 0 ulp |         0.12 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_v128relaxed` |         2.04 gb/s, 0 ulp |          2.1 gb/s, 0 ulp |         2.09 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`       |        0.431 gb/s, 0 ulp |        0.433 gb/s, 0 ulp |        0.437 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_v128relaxed`  |         1.45 gb/s, 0 ulp |         3.75 gb/s, 0 ulp |         4.32 gb/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`      |        0.117 gb/s, 0 ulp |        0.118 gb/s, 0 ulp |         0.12 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_v128relaxed` |          3.1 gb/s, 0 ulp |         3.26 gb/s, 0 ulp |         3.28 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`       |        0.434 gb/s, 0 ulp |         0.43 gb/s, 0 ulp |        0.439 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_v128relaxed`  |         2.84 gb/s, 0 ulp |         3.77 gb/s, 0 ulp |         4.32 gb/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`        |                2.04 gb/s |                2.07 gb/s |                2.09 gb/s |
| `nk_reduce_moments_i8_v128relaxed`   |                6.69 gb/s |                7.37 gb/s |                7.43 gb/s |
| `nk_reduce_minmax_i8_serial`         |               0.928 gb/s |                0.92 gb/s |               0.935 gb/s |
| `nk_reduce_minmax_i8_v128relaxed`    |                5.08 gb/s |                5.55 gb/s |                   7 gb/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`        |                2.04 gb/s |                2.09 gb/s |                 2.1 gb/s |
| `nk_reduce_moments_u8_v128relaxed`   |                6.38 gb/s |                6.96 gb/s |                7.04 gb/s |
| `nk_reduce_minmax_u8_serial`         |               0.851 gb/s |               0.851 gb/s |               0.858 gb/s |
| `nk_reduce_minmax_u8_v128relaxed`    |                3.97 gb/s |                4.52 gb/s |                5.45 gb/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`       |                4.09 gb/s |                4.12 gb/s |                4.18 gb/s |
| `nk_reduce_moments_i16_v128relaxed`  |                7.03 gb/s |                7.37 gb/s |                7.34 gb/s |
| `nk_reduce_minmax_i16_serial`        |                1.86 gb/s |                1.85 gb/s |                1.87 gb/s |
| `nk_reduce_minmax_i16_v128relaxed`   |                6.67 gb/s |                7.53 gb/s |                 8.2 gb/s |
| __u16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`       |                4.09 gb/s |                4.17 gb/s |                4.23 gb/s |
| `nk_reduce_moments_u16_v128relaxed`  |                6.74 gb/s |                7.05 gb/s |                6.91 gb/s |
| `nk_reduce_minmax_u16_serial`        |                1.67 gb/s |                1.67 gb/s |                1.68 gb/s |
| `nk_reduce_minmax_u16_v128relaxed`   |                4.78 gb/s |                5.43 gb/s |                5.83 gb/s |
| __i32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`       |                3.65 gb/s |                3.74 gb/s |                3.77 gb/s |
| `nk_reduce_moments_i32_v128relaxed`  |                1.58 gb/s |                 1.6 gb/s |                1.58 gb/s |
| `nk_reduce_minmax_i32_serial`        |                4.27 gb/s |                 4.3 gb/s |                4.32 gb/s |
| `nk_reduce_minmax_i32_v128relaxed`   |                6.91 gb/s |                7.77 gb/s |                8.06 gb/s |
| __u32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`       |                3.59 gb/s |                 3.6 gb/s |                3.65 gb/s |
| `nk_reduce_moments_u32_v128relaxed`  |                1.22 gb/s |                1.21 gb/s |                1.21 gb/s |
| `nk_reduce_minmax_u32_serial`        |                3.81 gb/s |                3.81 gb/s |                3.88 gb/s |
| `nk_reduce_minmax_u32_v128relaxed`   |                 5.1 gb/s |                5.62 gb/s |                5.89 gb/s |
| __i64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`       |                3.09 gb/s |                3.14 gb/s |                3.28 gb/s |
| `nk_reduce_moments_i64_v128relaxed`  |                2.65 gb/s |                2.66 gb/s |                2.65 gb/s |
| `nk_reduce_minmax_i64_serial`        |                8.49 gb/s |                8.38 gb/s |                8.65 gb/s |
| `nk_reduce_minmax_i64_v128relaxed`   |                6.14 gb/s |                6.37 gb/s |                 6.5 gb/s |
| __u64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`       |                2.78 gb/s |                2.79 gb/s |                2.87 gb/s |
| `nk_reduce_moments_u64_v128relaxed`  |                2.46 gb/s |                2.48 gb/s |                2.47 gb/s |
| `nk_reduce_minmax_u64_serial`        |                7.83 gb/s |                7.59 gb/s |                7.74 gb/s |
| `nk_reduce_minmax_u64_v128relaxed`   |                2.54 gb/s |                2.59 gb/s |                2.64 gb/s |

### Apple M5

#### Native

| Kernel                             |                      256 |                     1024 |                     4096 |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`     |         6.41 gb/s, 0 ulp |         7.19 gb/s, 0 ulp |         6.71 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`      |         16.4 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |
| `nk_reduce_moments_f64_neon`       |         16.4 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_neon`        |         18.0 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         16.2 gb/s, 0 ulp |
| __f32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`     |         3.06 gb/s, 0 ulp |         3.40 gb/s, 0 ulp |         3.15 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`      |         8.29 gb/s, 0 ulp |         7.97 gb/s, 0 ulp |         8.00 gb/s, 0 ulp |
| `nk_reduce_moments_f32_neon`       |       16.6 gb/s, 0.4 ulp |       10.8 gb/s, 1.8 ulp |       9.78 gb/s, 1.1 ulp |
| `nk_reduce_minmax_f32_neon`        |         19.7 gb/s, 0 ulp |         17.3 gb/s, 0 ulp |         16.5 gb/s, 0 ulp |
| __bf16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`    |         1.49 gb/s, 0 ulp |         1.62 gb/s, 0 ulp |         1.55 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_serial`     |         2.19 gb/s, 0 ulp |         2.49 gb/s, 0 ulp |         2.69 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_neonbfdot` |         26.1 gb/s, 0 ulp |       28.2 gb/s, 0.4 ulp |       29.9 gb/s, 0.3 ulp |
| __f16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`     |         1.46 gb/s, 0 ulp |         1.55 gb/s, 0 ulp |         1.51 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_serial`      |         1.62 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |         2.02 gb/s, 0 ulp |
| `nk_reduce_moments_f16_neonhalf`   |       21.2 gb/s, 0.1 ulp |       15.4 gb/s, 0.1 ulp |       10.6 gb/s, 0.8 ulp |
| __e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`    |        0.750 gb/s, 0 ulp |        0.830 gb/s, 0 ulp |        0.758 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`     |         1.10 gb/s, 0 ulp |         1.34 gb/s, 0 ulp |         1.39 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_neon`      |         10.4 gb/s, ? ulp |         7.47 gb/s, ? ulp |         5.22 gb/s, ? ulp |
| `nk_reduce_moments_e5m2_neonfhm`   |         12.4 gb/s, 0 ulp |         7.31 gb/s, 0 ulp |         4.78 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_neon`       |         15.5 gb/s, 0 ulp |         17.4 gb/s, 0 ulp |         18.1 gb/s, 0 ulp |
| __e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`    |        0.569 gb/s, 0 ulp |        0.640 gb/s, 0 ulp |        0.599 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`     |         1.14 gb/s, 0 ulp |         1.39 gb/s, 0 ulp |         1.42 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_neon`      |         6.52 gb/s, ? ulp |         5.52 gb/s, ? ulp |         4.88 gb/s, ? ulp |
| `nk_reduce_moments_e4m3_neonfhm`   |         4.21 gb/s, 0 ulp |         4.23 gb/s, 0 ulp |         4.11 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_neon`       |         15.5 gb/s, 0 ulp |         17.4 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |
| __e3m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`    |        0.750 gb/s, 0 ulp |        0.806 gb/s, 0 ulp |        0.773 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`     |        0.640 gb/s, 0 ulp |        0.641 gb/s, 0 ulp |        0.634 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_neon`      |         8.64 gb/s, ? ulp |         6.96 gb/s, ? ulp |         5.70 gb/s, ? ulp |
| `nk_reduce_minmax_e3m2_neon`       |         16.6 gb/s, 0 ulp |         19.1 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |
| __e2m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`    |        0.753 gb/s, 0 ulp |        0.821 gb/s, 0 ulp |        0.775 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`     |        0.640 gb/s, 0 ulp |        0.643 gb/s, 0 ulp |        0.618 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_neon`      |         17.1 gb/s, ? ulp |         15.2 gb/s, ? ulp |         11.9 gb/s, ? ulp |
| `nk_reduce_moments_e2m3_neonsdot`  |         29.4 gb/s, 0 ulp |         29.3 gb/s, 0 ulp |         24.4 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_neon`       |         17.0 gb/s, 0 ulp |         19.3 gb/s, 0 ulp |         19.6 gb/s, 0 ulp |
| __i8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`      |                3.00 gb/s |                3.50 gb/s |                3.21 gb/s |
| `nk_reduce_minmax_i8_serial`       |                1.81 gb/s |                1.94 gb/s |                1.89 gb/s |
| `nk_reduce_moments_i8_neon`        |                27.9 gb/s |                17.7 gb/s |                13.3 gb/s |
| `nk_reduce_minmax_i8_neon`         |                25.7 gb/s |                30.3 gb/s |                27.8 gb/s |
| `nk_reduce_moments_i8_neonsdot`    |                44.4 gb/s |                46.8 gb/s |                33.4 gb/s |
| __u8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`      |                3.08 gb/s |                3.51 gb/s |                3.23 gb/s |
| `nk_reduce_minmax_u8_serial`       |                1.84 gb/s |                1.94 gb/s |                1.90 gb/s |
| `nk_reduce_moments_u8_neon`        |                29.2 gb/s |                18.0 gb/s |                13.6 gb/s |
| `nk_reduce_minmax_u8_neon`         |                27.0 gb/s |                28.3 gb/s |                29.2 gb/s |
| `nk_reduce_moments_u8_neonsdot`    |                45.5 gb/s |                43.5 gb/s |                33.3 gb/s |
| __i4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i4_serial`      |                2.06 gb/s |                2.57 gb/s |                2.39 gb/s |
| `nk_reduce_minmax_i4_serial`       |               0.701 gb/s |               0.794 gb/s |               0.803 gb/s |
| __u4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u4_serial`      |                2.20 gb/s |                2.75 gb/s |                2.62 gb/s |
| `nk_reduce_minmax_u4_serial`       |               0.741 gb/s |               0.845 gb/s |               0.856 gb/s |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u1_serial`      |                1.79 gb/s |                1.99 gb/s |                2.03 gb/s |
| `nk_reduce_minmax_u1_serial`       |                9.74 gb/s |                38.5 gb/s |                 121 gb/s |
| __i16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`     |                6.02 gb/s |                7.03 gb/s |                6.70 gb/s |
| `nk_reduce_minmax_i16_serial`      |                3.66 gb/s |                3.88 gb/s |                3.82 gb/s |
| `nk_reduce_moments_i16_neon`       |                23.0 gb/s |                16.2 gb/s |                11.9 gb/s |
| `nk_reduce_minmax_i16_neon`        |                26.8 gb/s |                28.5 gb/s |                25.9 gb/s |
| __u16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`     |                6.13 gb/s |                6.79 gb/s |                6.45 gb/s |
| `nk_reduce_minmax_u16_serial`      |                2.72 gb/s |                2.68 gb/s |                2.67 gb/s |
| `nk_reduce_moments_u16_neon`       |                23.1 gb/s |                16.3 gb/s |                11.6 gb/s |
| `nk_reduce_minmax_u16_neon`        |                26.6 gb/s |                28.5 gb/s |                25.8 gb/s |
| __i32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`     |                7.70 gb/s |                8.09 gb/s |                7.52 gb/s |
| `nk_reduce_minmax_i32_serial`      |                7.35 gb/s |                7.49 gb/s |                7.65 gb/s |
| `nk_reduce_moments_i32_neon`       |                8.88 gb/s |                6.74 gb/s |                6.60 gb/s |
| `nk_reduce_minmax_i32_neon`        |                27.3 gb/s |                27.9 gb/s |                25.8 gb/s |
| __u32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`     |                7.51 gb/s |                8.10 gb/s |                7.73 gb/s |
| `nk_reduce_minmax_u32_serial`      |                7.28 gb/s |                7.69 gb/s |                7.68 gb/s |
| `nk_reduce_moments_u32_neon`       |                19.0 gb/s |                12.8 gb/s |                11.5 gb/s |
| `nk_reduce_minmax_u32_neon`        |                27.6 gb/s |                28.3 gb/s |                26.4 gb/s |
| __i64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`     |                10.4 gb/s |                12.9 gb/s |                11.9 gb/s |
| `nk_reduce_minmax_i64_serial`      |                13.7 gb/s |                14.1 gb/s |                14.0 gb/s |
| `nk_reduce_moments_i64_neon`       |                15.6 gb/s |                13.1 gb/s |                12.9 gb/s |
| `nk_reduce_minmax_i64_neon`        |                18.0 gb/s |                16.6 gb/s |                15.6 gb/s |
| __u64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`     |                10.4 gb/s |                11.9 gb/s |                11.2 gb/s |
| `nk_reduce_minmax_u64_serial`      |                13.4 gb/s |                14.1 gb/s |                13.8 gb/s |
| `nk_reduce_moments_u64_neon`       |                29.1 gb/s |                22.5 gb/s |                22.2 gb/s |
| `nk_reduce_minmax_u64_neon`        |                18.3 gb/s |                16.7 gb/s |                16.1 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                               |                      256 |                     1024 |                     4096 |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`       |         7.05 gb/s, 0 ulp |         6.89 gb/s, 0 ulp |         7.04 gb/s, 0 ulp |
| `nk_reduce_moments_f64_v128relaxed`  |         17.9 gb/s, 0 ulp |         17.9 gb/s, 0 ulp |         17.9 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`        |         11.8 gb/s, 0 ulp |         11.5 gb/s, 0 ulp |         11.4 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_v128relaxed`   |         16.4 gb/s, 0 ulp |         16.8 gb/s, 0 ulp |         16.9 gb/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`       |         3.34 gb/s, 0 ulp |         3.32 gb/s, 0 ulp |         3.34 gb/s, 0 ulp |
| `nk_reduce_moments_f32_v128relaxed`  |       16.7 gb/s, 0.1 ulp |       11.9 gb/s, 0.5 ulp |         10.3 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`        |         4.40 gb/s, 0 ulp |         4.31 gb/s, 0 ulp |         4.28 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_v128relaxed`   |         14.5 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`      |         1.76 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_v128relaxed` |         20.8 gb/s, 0 ulp |       14.8 gb/s, 0.2 ulp |       11.1 gb/s, 1.6 ulp |
| `nk_reduce_minmax_bf16_serial`       |         1.80 gb/s, 0 ulp |         2.01 gb/s, 0 ulp |         2.10 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_v128relaxed`  |         9.22 gb/s, 0 ulp |         9.53 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`       |         1.77 gb/s, 0 ulp |         1.73 gb/s, 0 ulp |         1.73 gb/s, 0 ulp |
| `nk_reduce_moments_f16_v128relaxed`  |         11.8 gb/s, 0 ulp |         10.8 gb/s, 0 ulp |       9.84 gb/s, 0.3 ulp |
| `nk_reduce_minmax_f16_serial`        |         1.79 gb/s, 0 ulp |         1.99 gb/s, 0 ulp |         2.09 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_v128relaxed`   |         6.04 gb/s, 0 ulp |         7.62 gb/s, 0 ulp |         8.11 gb/s, 0 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`      |        0.886 gb/s, 0 ulp |        0.865 gb/s, 0 ulp |        0.870 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_v128relaxed` |         3.53 gb/s, 0 ulp |         3.55 gb/s, 0 ulp |         3.56 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`       |         1.12 gb/s, 0 ulp |         1.18 gb/s, 0 ulp |         1.22 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_v128relaxed`  |         5.50 gb/s, 0 ulp |         10.8 gb/s, 0 ulp |         14.6 gb/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`      |        0.617 gb/s, 0 ulp |        0.606 gb/s, 0 ulp |        0.610 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_v128relaxed` |         3.01 gb/s, 0 ulp |         3.04 gb/s, 0 ulp |         3.05 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`       |         1.09 gb/s, 0 ulp |         1.21 gb/s, 0 ulp |         1.27 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_v128relaxed`  |         6.46 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`      |        0.881 gb/s, 0 ulp |        0.873 gb/s, 0 ulp |        0.868 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_v128relaxed` |         5.41 gb/s, 0 ulp |         4.67 gb/s, 0 ulp |         4.34 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`       |        0.671 gb/s, 0 ulp |        0.667 gb/s, 0 ulp |        0.666 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_v128relaxed`  |         7.95 gb/s, 0 ulp |         13.0 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`      |        0.884 gb/s, 0 ulp |        0.877 gb/s, 0 ulp |        0.857 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_v128relaxed` |         9.54 gb/s, 0 ulp |         9.49 gb/s, 0 ulp |         8.72 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`       |        0.670 gb/s, 0 ulp |        0.667 gb/s, 0 ulp |        0.666 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_v128relaxed`  |         9.35 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |         16.4 gb/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`        |                3.78 gb/s |                3.68 gb/s |                3.62 gb/s |
| `nk_reduce_moments_i8_v128relaxed`   |                21.6 gb/s |                22.3 gb/s |                18.7 gb/s |
| `nk_reduce_minmax_i8_serial`         |                1.47 gb/s |                1.49 gb/s |                1.49 gb/s |
| `nk_reduce_minmax_i8_v128relaxed`    |                13.0 gb/s |                19.1 gb/s |                23.1 gb/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`        |                3.71 gb/s |                3.68 gb/s |                3.62 gb/s |
| `nk_reduce_moments_u8_v128relaxed`   |                21.6 gb/s |                22.3 gb/s |                18.8 gb/s |
| `nk_reduce_minmax_u8_serial`         |                1.51 gb/s |                1.55 gb/s |                1.55 gb/s |
| `nk_reduce_minmax_u8_v128relaxed`    |                12.6 gb/s |                16.6 gb/s |                22.5 gb/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`       |                7.52 gb/s |                7.35 gb/s |                7.33 gb/s |
| `nk_reduce_moments_i16_v128relaxed`  |                13.3 gb/s |                9.97 gb/s |                8.83 gb/s |
| `nk_reduce_minmax_i16_serial`        |                2.96 gb/s |                2.99 gb/s |                2.99 gb/s |
| `nk_reduce_minmax_i16_v128relaxed`   |                18.8 gb/s |                14.5 gb/s |                16.1 gb/s |
| __u16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`       |                7.51 gb/s |                7.28 gb/s |                7.23 gb/s |
| `nk_reduce_moments_u16_v128relaxed`  |                13.2 gb/s |                9.98 gb/s |                8.84 gb/s |
| `nk_reduce_minmax_u16_serial`        |                3.04 gb/s |                3.11 gb/s |                3.11 gb/s |
| `nk_reduce_minmax_u16_v128relaxed`   |                16.5 gb/s |                14.5 gb/s |                16.2 gb/s |
| __i32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`       |                5.73 gb/s |                5.66 gb/s |                5.58 gb/s |
| `nk_reduce_moments_i32_v128relaxed`  |                6.40 gb/s |                6.35 gb/s |                6.39 gb/s |
| `nk_reduce_minmax_i32_serial`        |                6.85 gb/s |                6.88 gb/s |                6.90 gb/s |
| `nk_reduce_minmax_i32_v128relaxed`   |                14.0 gb/s |                15.9 gb/s |                16.7 gb/s |
| __u32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`       |                6.01 gb/s |                5.70 gb/s |                5.54 gb/s |
| `nk_reduce_moments_u32_v128relaxed`  |                1.65 gb/s |                1.58 gb/s |                1.56 gb/s |
| `nk_reduce_minmax_u32_serial`        |                6.80 gb/s |                6.87 gb/s |                6.90 gb/s |
| `nk_reduce_minmax_u32_v128relaxed`   |                14.4 gb/s |                15.9 gb/s |                16.7 gb/s |
| __i64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`       |                9.52 gb/s |                9.73 gb/s |                9.62 gb/s |
| `nk_reduce_moments_i64_v128relaxed`  |                7.42 gb/s |                7.40 gb/s |                7.50 gb/s |
| `nk_reduce_minmax_i64_serial`        |                13.6 gb/s |                13.7 gb/s |                13.7 gb/s |
| `nk_reduce_minmax_i64_v128relaxed`   |                16.4 gb/s |                16.9 gb/s |                17.0 gb/s |
| __u64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`       |                10.2 gb/s |                10.3 gb/s |                10.5 gb/s |
| `nk_reduce_moments_u64_v128relaxed`  |                3.04 gb/s |                2.98 gb/s |                2.94 gb/s |
| `nk_reduce_minmax_u64_serial`        |                13.6 gb/s |                13.7 gb/s |                13.7 gb/s |
| `nk_reduce_minmax_u64_v128relaxed`   |                3.31 gb/s |                3.33 gb/s |                3.35 gb/s |
