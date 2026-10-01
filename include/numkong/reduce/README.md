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
| :--------- | :---------- | :-------------------------------------- |
| `f64`      | `f64`       | 64-bit double precision                 |
| `f32`      | `f64`       | 32-bit single precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output      |
| `f16`      | `f32`       | 16-bit half precision, widened output   |

Mini-float reductions:

| Input Type | Output Type | Description                                  |
| :--------- | :---------- | :------------------------------------------- |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits    |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits    |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits |
| `e2m1`     | `f32`       | 4-bit MX format: 2 exponent, 1 mantissa bit  |

Integer reductions:

| Input Type | Output Type | Description                        |
| :--------- | :---------- | :--------------------------------- |
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
| :--------- | :---------- | :---------------------------------------- |
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
For Float16/BFloat16/Float8 inputs, all backends widen to Float32 before accumulation — NEON FHM (`nk_reduce_moments_e4m3_neonfhm`) converts e4m3 → f16 by bit manipulation, then uses `vfmlalq_low_f16` and `vfmlalq_high_f16` to fuse the Float16 → Float32 widening with the FMA into the Float32 accumulator.

### Integer Saturation in Sum-of-Squares

Integer moments accumulate sums in the widest available type: Int8/UInt8/Int16 inputs produce Int64/UInt64 outputs.
Sums use widening addition chains — NEON uses pairwise widening (`vpaddlq_s16` → UInt32 → UInt64 stages); Haswell biases Int8 inputs with 0x80 and uses unsigned SAD (`_mm256_sad_epu8`) for the sum, correcting by subtracting $128 \times \text{count}$ at the end.
Sum-of-squares can overflow UInt64 when squaring Int64/UInt64 inputs — only `nk_reduce_moments_i64_*` and `nk_reduce_moments_u64_*` need an explicit saturating multiply: it checks whether `abs(val)` < 2³² (so the square fits in UInt64) and otherwise saturates to I64_MAX for signed inputs, U64_MAX for unsigned ones.
Narrower inputs cannot overflow the product itself, so they only need the saturating accumulate.
Haswell emulates UInt64 saturating add via XOR-based unsigned comparison: flip sign bits to convert unsigned overflow detection into a signed comparison, then OR with the overflow mask to produce all-ones on saturation.

### Recursive Blocking for Counter Overflow

All SIMD backends use loop iteration counters narrower than `nk_size_t` to save register pressure — UInt8 for Int8 minmax lanes, UInt16 for Float32 moments lanes.
When `count` exceeds the counter's range × lane count (e.g., Haswell Float32 moments: $65536 \times 8 = 524288$ elements for UInt16 counters), the reduction splits recursively: process the left half, process the right half, combine results with saturating arithmetic.
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
The sentinels are ±infinity, so any finite extremum replaces them.
A side still at its sentinel saw only that infinity and NaNs, so a serial scan returns the first such infinity, and only an all-NaN input reports `NUMKONG_SIZE_MAX`.
The final horizontal reduction across lanes uses pairwise `VSHUFPS` + `VMINPS` chains — 3 shuffles for a 256-bit register, $O(\log_2 w)$ for width $w$.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_BATCH_PER_CORE` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.

### Intel Sapphire Rapids

#### Native

| Kernel                           |                      256 |                     1024 |                     4096 |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`   |         1.37 gb/s, 0 ulp |         1.61 gb/s, 0 ulp |         1.82 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`    |         6.14 gb/s, 0 ulp |         5.54 gb/s, 0 ulp |         5.42 gb/s, 0 ulp |
| `nk_reduce_moments_f64_haswell`  |       10.1 gb/s, 0.1 ulp |         8.55 gb/s, 0 ulp |         5.63 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_haswell`   |         7.55 gb/s, 0 ulp |         8.80 gb/s, 0 ulp |         6.14 gb/s, 0 ulp |
| `nk_reduce_moments_f64_skylake`  |       13.7 gb/s, 0.3 ulp |       12.9 gb/s, 0.1 ulp |         10.6 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_skylake`   |         8.40 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         9.25 gb/s, 0 ulp |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`   |        0.427 gb/s, 0 ulp |        0.407 gb/s, 0 ulp |        0.418 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`    |         3.12 gb/s, 0 ulp |         2.83 gb/s, 0 ulp |         3.05 gb/s, 0 ulp |
| `nk_reduce_moments_f32_haswell`  |       17.1 gb/s, 0.8 ulp |       16.6 gb/s, 4.2 ulp |       10.9 gb/s, 7.7 ulp |
| `nk_reduce_minmax_f32_haswell`   |         7.62 gb/s, 0 ulp |         8.31 gb/s, 0 ulp |         7.67 gb/s, 0 ulp |
| `nk_reduce_moments_f32_skylake`  |       19.3 gb/s, 0.4 ulp |       18.9 gb/s, 3.1 ulp |       15.9 gb/s, 8.8 ulp |
| `nk_reduce_minmax_f32_skylake`   |         6.85 gb/s, 0 ulp |         14.8 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`  |        0.194 gb/s, 0 ulp |        0.228 gb/s, 0 ulp |        0.223 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_serial`   |        0.871 gb/s, 0 ulp |        0.916 gb/s, 0 ulp |        0.931 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_haswell` |         10.6 gb/s, 0 ulp |         11.4 gb/s, 0 ulp |       10.1 gb/s, 1.6 ulp |
| `nk_reduce_minmax_bf16_haswell`  |         4.64 gb/s, 0 ulp |         7.02 gb/s, 0 ulp |         8.66 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_skylake` |         17.0 gb/s, 0 ulp |         25.1 gb/s, 0 ulp |       16.7 gb/s, 0.7 ulp |
| `nk_reduce_minmax_bf16_skylake`  |         6.08 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         12.8 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_genoa`   |         16.9 gb/s, 0 ulp |         19.1 gb/s, 0 ulp |       18.0 gb/s, 0.8 ulp |
| __f16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`   |        0.364 gb/s, 0 ulp |        0.330 gb/s, 0 ulp |        0.379 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_serial`    |        0.839 gb/s, 0 ulp |        0.817 gb/s, 0 ulp |        0.907 gb/s, 0 ulp |
| `nk_reduce_moments_f16_haswell`  |         12.6 gb/s, 0 ulp |         11.7 gb/s, 0 ulp |       10.2 gb/s, 0.3 ulp |
| `nk_reduce_minmax_f16_haswell`   |         6.16 gb/s, 0 ulp |         8.56 gb/s, 0 ulp |         7.54 gb/s, 0 ulp |
| `nk_reduce_moments_f16_skylake`  |         16.5 gb/s, 0 ulp |       27.1 gb/s, 0.1 ulp |         17.3 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_skylake`   |         9.50 gb/s, 0 ulp |         19.4 gb/s, 0 ulp |         20.5 gb/s, 0 ulp |
| __e5m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`  |        0.146 gb/s, 0 ulp |        0.276 gb/s, 0 ulp |        0.213 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`   |        0.389 gb/s, 0 ulp |        0.388 gb/s, 0 ulp |        0.420 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_haswell` |         2.24 gb/s, 0 ulp |         2.51 gb/s, 0 ulp |         2.43 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_haswell`  |         4.17 gb/s, 0 ulp |         6.33 gb/s, 0 ulp |         6.71 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_skylake` |         4.34 gb/s, 0 ulp |         2.64 gb/s, 0 ulp |         3.76 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_skylake`  |         3.63 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_genoa`   |         4.43 gb/s, 0 ulp |         5.66 gb/s, 0 ulp |         5.48 gb/s, 0 ulp |
| __e4m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`  |        0.113 gb/s, 0 ulp |        0.120 gb/s, 0 ulp |        0.147 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`   |        0.428 gb/s, 0 ulp |        0.441 gb/s, 0 ulp |        0.432 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_haswell` |         1.70 gb/s, 0 ulp |         1.77 gb/s, 0 ulp |         1.65 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_haswell`  |         4.12 gb/s, 0 ulp |         6.52 gb/s, 0 ulp |         7.54 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_skylake` |         2.58 gb/s, 0 ulp |         3.29 gb/s, 0 ulp |         2.55 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_skylake`  |         3.53 gb/s, 0 ulp |         8.91 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_genoa`   |         4.35 gb/s, 0 ulp |         5.47 gb/s, 0 ulp |         5.28 gb/s, 0 ulp |
| __e3m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`  |        0.147 gb/s, 0 ulp |        0.260 gb/s, 0 ulp |        0.324 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`   |        0.432 gb/s, 0 ulp |        0.387 gb/s, 0 ulp |        0.438 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_haswell` |         2.21 gb/s, 0 ulp |         2.37 gb/s, 0 ulp |         2.36 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_haswell`  |         4.99 gb/s, 0 ulp |         7.35 gb/s, 0 ulp |         8.90 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_skylake` |         2.58 gb/s, 0 ulp |         3.09 gb/s, 0 ulp |         3.33 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_skylake`  |         9.17 gb/s, 0 ulp |         18.7 gb/s, 0 ulp |         13.6 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_icelake` |         8.21 gb/s, 0 ulp |         8.40 gb/s, 0 ulp |         12.5 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_alder`   |         4.47 gb/s, 0 ulp |         6.62 gb/s, 0 ulp |         7.35 gb/s, 0 ulp |
| __e2m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`  |        0.146 gb/s, 0 ulp |        0.274 gb/s, 0 ulp |        0.280 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`   |        0.433 gb/s, 0 ulp |        0.392 gb/s, 0 ulp |        0.422 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_haswell` |         2.26 gb/s, 0 ulp |         2.28 gb/s, 0 ulp |         2.40 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_haswell`  |         4.95 gb/s, 0 ulp |         7.36 gb/s, 0 ulp |         8.72 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_skylake` |         3.25 gb/s, 0 ulp |         2.81 gb/s, 0 ulp |         3.41 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_skylake`  |         5.72 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_icelake` |         11.8 gb/s, 0 ulp |         21.1 gb/s, 0 ulp |         20.2 gb/s, 0 ulp |
| __i8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`    |                2.06 gb/s |                2.24 gb/s |                2.13 gb/s |
| `nk_reduce_minmax_i8_serial`     |               0.751 gb/s |               0.906 gb/s |                1.02 gb/s |
| `nk_reduce_moments_i8_haswell`   |                8.71 gb/s |                11.1 gb/s |                11.8 gb/s |
| `nk_reduce_minmax_i8_haswell`    |                6.62 gb/s |                10.9 gb/s |                12.3 gb/s |
| `nk_reduce_moments_i8_skylake`   |                9.69 gb/s |                15.5 gb/s |                18.7 gb/s |
| `nk_reduce_minmax_i8_skylake`    |                2.76 gb/s |                13.4 gb/s |                14.4 gb/s |
| `nk_reduce_moments_i8_icelake`   |                13.0 gb/s |                26.4 gb/s |                26.4 gb/s |
| __u8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`    |                2.24 gb/s |                2.32 gb/s |                2.00 gb/s |
| `nk_reduce_minmax_u8_serial`     |               0.723 gb/s |               0.867 gb/s |               0.978 gb/s |
| `nk_reduce_moments_u8_haswell`   |                9.59 gb/s |                12.0 gb/s |                12.7 gb/s |
| `nk_reduce_minmax_u8_haswell`    |                6.59 gb/s |                10.4 gb/s |                11.2 gb/s |
| `nk_reduce_moments_u8_skylake`   |                12.3 gb/s |                18.7 gb/s |                18.3 gb/s |
| `nk_reduce_minmax_u8_skylake`    |                4.14 gb/s |                13.0 gb/s |                19.0 gb/s |
| `nk_reduce_moments_u8_icelake`   |                13.6 gb/s |                20.2 gb/s |                28.3 gb/s |
| `nk_reduce_moments_u8_alder`     |                10.7 gb/s |                12.4 gb/s |                12.8 gb/s |
| __i4__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i4_serial`    |               0.321 gb/s |               0.705 gb/s |               0.700 gb/s |
| `nk_reduce_minmax_i4_serial`     |               0.292 gb/s |               0.265 gb/s |               0.332 gb/s |
| `nk_reduce_moments_i4_haswell`   |                5.92 gb/s |                8.54 gb/s |                9.59 gb/s |
| `nk_reduce_moments_i4_skylake`   |                7.14 gb/s |                8.24 gb/s |                14.3 gb/s |
| __u4__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u4_serial`    |               0.408 gb/s |               0.744 gb/s |               0.931 gb/s |
| `nk_reduce_minmax_u4_serial`     |               0.328 gb/s |               0.272 gb/s |               0.370 gb/s |
| `nk_reduce_moments_u4_haswell`   |                6.89 gb/s |                9.97 gb/s |                10.1 gb/s |
| `nk_reduce_moments_u4_skylake`   |                8.80 gb/s |                14.0 gb/s |                17.0 gb/s |
| __u1__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u1_serial`    |                1.27 gb/s |                1.83 gb/s |                1.90 gb/s |
| `nk_reduce_minmax_u1_serial`     |                5.07 gb/s |                13.7 gb/s |                78.3 gb/s |
| `nk_reduce_moments_u1_haswell`   |                4.00 gb/s |                9.02 gb/s |                11.2 gb/s |
| `nk_reduce_moments_u1_skylake`   |                2.70 gb/s |                11.5 gb/s |                19.2 gb/s |
| __i16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`   |                2.37 gb/s |                2.50 gb/s |                2.61 gb/s |
| `nk_reduce_minmax_i16_serial`    |                1.49 gb/s |                1.63 gb/s |                1.93 gb/s |
| `nk_reduce_moments_i16_haswell`  |                12.8 gb/s |                13.7 gb/s |                11.6 gb/s |
| `nk_reduce_minmax_i16_haswell`   |                7.97 gb/s |                10.2 gb/s |                9.31 gb/s |
| `nk_reduce_moments_i16_skylake`  |                15.6 gb/s |                19.6 gb/s |                19.1 gb/s |
| `nk_reduce_minmax_i16_skylake`   |                6.28 gb/s |                14.8 gb/s |                17.8 gb/s |
| `nk_reduce_moments_i16_icelake`  |                17.7 gb/s |                23.2 gb/s |                26.3 gb/s |
| `nk_reduce_moments_i16_alder`    |                9.31 gb/s |                11.3 gb/s |                9.78 gb/s |
| __u16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`   |                2.44 gb/s |                2.37 gb/s |                2.37 gb/s |
| `nk_reduce_minmax_u16_serial`    |                1.19 gb/s |                1.31 gb/s |                1.51 gb/s |
| `nk_reduce_moments_u16_haswell`  |                6.35 gb/s |                6.47 gb/s |                6.15 gb/s |
| `nk_reduce_minmax_u16_haswell`   |                7.68 gb/s |                9.78 gb/s |                10.8 gb/s |
| `nk_reduce_moments_u16_skylake`  |                9.50 gb/s |                12.9 gb/s |                11.7 gb/s |
| `nk_reduce_minmax_u16_skylake`   |                14.9 gb/s |                21.0 gb/s |                15.7 gb/s |
| `nk_reduce_moments_u16_alder`    |                6.68 gb/s |                7.54 gb/s |                7.05 gb/s |
| __i32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`   |                2.23 gb/s |                2.10 gb/s |                2.16 gb/s |
| `nk_reduce_minmax_i32_serial`    |                2.78 gb/s |                3.42 gb/s |                4.17 gb/s |
| `nk_reduce_moments_i32_haswell`  |                5.06 gb/s |                5.00 gb/s |                4.11 gb/s |
| `nk_reduce_minmax_i32_haswell`   |                10.3 gb/s |                9.50 gb/s |                9.69 gb/s |
| `nk_reduce_moments_i32_skylake`  |                6.40 gb/s |                10.3 gb/s |                9.87 gb/s |
| `nk_reduce_minmax_i32_skylake`   |                22.2 gb/s |                23.0 gb/s |                16.4 gb/s |
| __u32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`   |                3.22 gb/s |                3.29 gb/s |                3.18 gb/s |
| `nk_reduce_minmax_u32_serial`    |                2.62 gb/s |                3.11 gb/s |                3.77 gb/s |
| `nk_reduce_moments_u32_haswell`  |                5.68 gb/s |                5.39 gb/s |                4.91 gb/s |
| `nk_reduce_minmax_u32_haswell`   |                9.87 gb/s |                10.4 gb/s |                9.27 gb/s |
| `nk_reduce_moments_u32_skylake`  |                14.8 gb/s |                9.28 gb/s |                14.2 gb/s |
| `nk_reduce_minmax_u32_skylake`   |                22.0 gb/s |                23.6 gb/s |                20.2 gb/s |
| __i64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`   |                2.26 gb/s |                2.26 gb/s |                2.26 gb/s |
| `nk_reduce_minmax_i64_serial`    |                4.56 gb/s |                5.16 gb/s |                5.68 gb/s |
| `nk_reduce_moments_i64_haswell`  |                6.67 gb/s |                6.09 gb/s |                5.01 gb/s |
| `nk_reduce_minmax_i64_haswell`   |                8.85 gb/s |                9.19 gb/s |                7.11 gb/s |
| `nk_reduce_moments_i64_skylake`  |                12.1 gb/s |                7.72 gb/s |                9.78 gb/s |
| `nk_reduce_minmax_i64_skylake`   |                10.8 gb/s |                21.5 gb/s |                20.5 gb/s |
| __u64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`   |                1.81 gb/s |                1.79 gb/s |                1.70 gb/s |
| `nk_reduce_minmax_u64_serial`    |                5.58 gb/s |                6.71 gb/s |                6.83 gb/s |
| `nk_reduce_moments_u64_haswell`  |                8.01 gb/s |                7.87 gb/s |                5.55 gb/s |
| `nk_reduce_minmax_u64_haswell`   |                8.32 gb/s |                9.14 gb/s |                7.03 gb/s |
| `nk_reduce_moments_u64_skylake`  |                14.5 gb/s |                18.0 gb/s |                8.26 gb/s |
| `nk_reduce_minmax_u64_skylake`   |                9.22 gb/s |                21.5 gb/s |                20.1 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                      256 |                     1024 |                     4096 |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`       |          1.1 gb/s, 0 ulp |         1.14 gb/s, 0 ulp |         1.11 gb/s, 0 ulp |
| `nk_reduce_moments_f64_v128`         |         6.49 gb/s, 0 ulp |         6.44 gb/s, 0 ulp |         6.58 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`        |         4.87 gb/s, 0 ulp |         5.11 gb/s, 0 ulp |         5.32 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_v128relaxed`   |          4.6 gb/s, 0 ulp |         4.81 gb/s, 0 ulp |         4.76 gb/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`       |        0.564 gb/s, 0 ulp |        0.579 gb/s, 0 ulp |        0.576 gb/s, 0 ulp |
| `nk_reduce_moments_f32_v128relaxed`  |       8.23 gb/s, 0.1 ulp |       6.31 gb/s, 0.4 ulp |         6.04 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`        |         2.45 gb/s, 0 ulp |         2.54 gb/s, 0 ulp |         2.66 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_v128relaxed`   |         4.27 gb/s, 0 ulp |         4.68 gb/s, 0 ulp |         4.70 gb/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`      |        0.255 gb/s, 0 ulp |         0.26 gb/s, 0 ulp |        0.268 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_v128`        |                        … |                        … |                        … |
| `nk_reduce_minmax_bf16_serial`       |        0.813 gb/s, 0 ulp |        0.910 gb/s, 0 ulp |        0.969 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_v128relaxed`  |         4.54 gb/s, 0 ulp |         4.79 gb/s, 0 ulp |         5.17 gb/s, 0 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`       |        0.218 gb/s, 0 ulp |        0.219 gb/s, 0 ulp |        0.230 gb/s, 0 ulp |
| `nk_reduce_moments_f16_v128relaxed`  |         1.78 gb/s, 0 ulp |         1.82 gb/s, 0 ulp |       1.79 gb/s, 0.2 ulp |
| `nk_reduce_minmax_f16_serial`        |        0.800 gb/s, 0 ulp |        0.901 gb/s, 0 ulp |        0.987 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_v128relaxed`   |          1.2 gb/s, 0 ulp |         1.29 gb/s, 0 ulp |         1.29 gb/s, 0 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`      |        0.108 gb/s, 0 ulp |        0.110 gb/s, 0 ulp |        0.114 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_v128relaxed` |         1.44 gb/s, 0 ulp |         1.48 gb/s, 0 ulp |         1.45 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`       |        0.439 gb/s, 0 ulp |        0.465 gb/s, 0 ulp |        0.491 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_v128relaxed`  |         1.18 gb/s, 0 ulp |         2.57 gb/s, 0 ulp |         2.86 gb/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`      |        0.102 gb/s, 0 ulp |        0.102 gb/s, 0 ulp |        0.103 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_v128relaxed` |         1.07 gb/s, 0 ulp |         1.08 gb/s, 0 ulp |         1.09 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`       |        0.420 gb/s, 0 ulp |        0.461 gb/s, 0 ulp |        0.498 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_v128relaxed`  |         1.33 gb/s, 0 ulp |         2.43 gb/s, 0 ulp |         2.85 gb/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`      |        0.113 gb/s, 0 ulp |        0.109 gb/s, 0 ulp |         0.11 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_v128relaxed` |         1.90 gb/s, 0 ulp |          2.0 gb/s, 0 ulp |         1.95 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`       |        0.401 gb/s, 0 ulp |        0.403 gb/s, 0 ulp |        0.407 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_v128relaxed`  |         1.35 gb/s, 0 ulp |         3.49 gb/s, 0 ulp |         4.02 gb/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`      |        0.109 gb/s, 0 ulp |        0.110 gb/s, 0 ulp |         0.11 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_v128relaxed` |          2.9 gb/s, 0 ulp |         3.04 gb/s, 0 ulp |         3.05 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`       |        0.404 gb/s, 0 ulp |         0.40 gb/s, 0 ulp |        0.409 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_v128relaxed`  |         2.64 gb/s, 0 ulp |         3.51 gb/s, 0 ulp |         4.02 gb/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`        |                1.90 gb/s |                1.93 gb/s |                1.95 gb/s |
| `nk_reduce_moments_i8_v128`          |                6.23 gb/s |                6.86 gb/s |                6.92 gb/s |
| `nk_reduce_minmax_i8_serial`         |               0.864 gb/s |                0.86 gb/s |               0.871 gb/s |
| `nk_reduce_minmax_i8_v128relaxed`    |                4.73 gb/s |                5.17 gb/s |                   7 gb/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`        |                1.90 gb/s |                1.95 gb/s |                 2.0 gb/s |
| `nk_reduce_moments_u8_v128`          |                5.94 gb/s |                6.48 gb/s |                6.56 gb/s |
| `nk_reduce_minmax_u8_serial`         |               0.793 gb/s |               0.793 gb/s |               0.799 gb/s |
| `nk_reduce_minmax_u8_v128relaxed`    |                3.70 gb/s |                4.21 gb/s |                5.08 gb/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`       |                3.81 gb/s |                3.84 gb/s |                3.89 gb/s |
| `nk_reduce_moments_i16_v128`         |                6.55 gb/s |                6.86 gb/s |                6.84 gb/s |
| `nk_reduce_minmax_i16_serial`        |                1.73 gb/s |                1.72 gb/s |                1.74 gb/s |
| `nk_reduce_minmax_i16_v128relaxed`   |                6.21 gb/s |                7.01 gb/s |                 7.6 gb/s |
| __u16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`       |                3.81 gb/s |                3.88 gb/s |                3.94 gb/s |
| `nk_reduce_moments_u16_v128`         |                6.28 gb/s |                6.57 gb/s |                6.44 gb/s |
| `nk_reduce_minmax_u16_serial`        |                1.56 gb/s |                1.56 gb/s |                1.56 gb/s |
| `nk_reduce_minmax_u16_v128relaxed`   |                4.45 gb/s |                5.06 gb/s |                5.43 gb/s |
| __i32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`       |                3.40 gb/s |                3.48 gb/s |                3.51 gb/s |
| `nk_reduce_moments_i32_v128`         |                1.47 gb/s |                 1.5 gb/s |                1.47 gb/s |
| `nk_reduce_minmax_i32_serial`        |                3.98 gb/s |                 4.0 gb/s |                4.02 gb/s |
| `nk_reduce_minmax_i32_v128relaxed`   |                6.44 gb/s |                7.24 gb/s |                7.51 gb/s |
| __u32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`       |                3.34 gb/s |                 3.4 gb/s |                3.40 gb/s |
| `nk_reduce_moments_u32_v128`         |                1.14 gb/s |                1.13 gb/s |                1.13 gb/s |
| `nk_reduce_minmax_u32_serial`        |                3.55 gb/s |                3.55 gb/s |                3.61 gb/s |
| `nk_reduce_minmax_u32_v128relaxed`   |                 4.7 gb/s |                5.23 gb/s |                5.49 gb/s |
| __i64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`       |                2.88 gb/s |                2.92 gb/s |                3.05 gb/s |
| `nk_reduce_moments_i64_v128relaxed`  |                2.47 gb/s |                2.48 gb/s |                2.47 gb/s |
| `nk_reduce_minmax_i64_serial`        |                7.91 gb/s |                7.80 gb/s |                8.06 gb/s |
| `nk_reduce_minmax_i64_v128relaxed`   |                5.72 gb/s |                5.93 gb/s |                 6.1 gb/s |
| __u64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`       |                2.59 gb/s |                2.60 gb/s |                2.67 gb/s |
| `nk_reduce_moments_u64_v128relaxed`  |                2.29 gb/s |                2.31 gb/s |                2.30 gb/s |
| `nk_reduce_minmax_u64_serial`        |                7.29 gb/s |                7.07 gb/s |                7.21 gb/s |
| `nk_reduce_minmax_u64_v128relaxed`   |                2.37 gb/s |                2.41 gb/s |                2.46 gb/s |

### Apple M5

#### Native

| Kernel                             |                      256 |                     1024 |                     4096 |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`     |         5.97 gb/s, 0 ulp |         6.70 gb/s, 0 ulp |         6.25 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`      |         15.3 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |
| `nk_reduce_moments_f64_neon`       |         15.3 gb/s, 0 ulp |         15.2 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_neon`        |         16.8 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |         15.1 gb/s, 0 ulp |
| __f32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`     |         2.85 gb/s, 0 ulp |         3.17 gb/s, 0 ulp |         2.93 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`      |         7.72 gb/s, 0 ulp |         7.42 gb/s, 0 ulp |         7.45 gb/s, 0 ulp |
| `nk_reduce_moments_f32_neon`       |       15.5 gb/s, 0.4 ulp |       10.1 gb/s, 1.8 ulp |       9.11 gb/s, 1.1 ulp |
| `nk_reduce_minmax_f32_neon`        |         18.3 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |         15.4 gb/s, 0 ulp |
| __bf16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`    |         1.39 gb/s, 0 ulp |         1.51 gb/s, 0 ulp |         1.44 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_serial`     |         2.04 gb/s, 0 ulp |         2.32 gb/s, 0 ulp |         2.51 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_neonbfdot` |         24.3 gb/s, 0 ulp |       26.3 gb/s, 0.4 ulp |       27.8 gb/s, 0.3 ulp |
| __f16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`     |         1.36 gb/s, 0 ulp |         1.44 gb/s, 0 ulp |         1.41 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_serial`      |         1.51 gb/s, 0 ulp |         1.74 gb/s, 0 ulp |         1.88 gb/s, 0 ulp |
| `nk_reduce_moments_f16_neon`       |       19.7 gb/s, 0.1 ulp |       14.3 gb/s, 0.1 ulp |       9.87 gb/s, 0.8 ulp |
| __e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`    |        0.698 gb/s, 0 ulp |        0.773 gb/s, 0 ulp |        0.706 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`     |         1.02 gb/s, 0 ulp |         1.25 gb/s, 0 ulp |         1.29 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_neon`      |         9.69 gb/s, ? ulp |         6.96 gb/s, ? ulp |         4.86 gb/s, ? ulp |
| `nk_reduce_moments_e5m2_neonfhm`   |         11.5 gb/s, 0 ulp |         6.81 gb/s, 0 ulp |         4.45 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_neon`       |         14.4 gb/s, 0 ulp |         16.2 gb/s, 0 ulp |         16.9 gb/s, 0 ulp |
| __e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`    |        0.530 gb/s, 0 ulp |        0.596 gb/s, 0 ulp |        0.558 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`     |         1.06 gb/s, 0 ulp |         1.29 gb/s, 0 ulp |         1.32 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_neon`      |         6.07 gb/s, ? ulp |         5.14 gb/s, ? ulp |         4.54 gb/s, ? ulp |
| `nk_reduce_moments_e4m3_neonfhm`   |         3.92 gb/s, 0 ulp |         3.94 gb/s, 0 ulp |         3.83 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_neon`       |         14.4 gb/s, 0 ulp |         16.2 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |
| __e3m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`    |        0.698 gb/s, 0 ulp |        0.751 gb/s, 0 ulp |        0.720 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`     |        0.596 gb/s, 0 ulp |        0.597 gb/s, 0 ulp |        0.590 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_neon`      |         8.05 gb/s, ? ulp |         6.48 gb/s, ? ulp |         5.31 gb/s, ? ulp |
| `nk_reduce_minmax_e3m2_neon`       |         15.5 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |
| __e2m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`    |        0.701 gb/s, 0 ulp |        0.765 gb/s, 0 ulp |        0.722 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`     |        0.596 gb/s, 0 ulp |        0.599 gb/s, 0 ulp |        0.576 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_neon`      |         15.9 gb/s, ? ulp |         14.2 gb/s, ? ulp |         11.1 gb/s, ? ulp |
| `nk_reduce_moments_e2m3_neonsdot`  |         27.4 gb/s, 0 ulp |         27.3 gb/s, 0 ulp |         22.7 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_neon`       |         15.8 gb/s, 0 ulp |         18.0 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| __i8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`      |                2.79 gb/s |                3.26 gb/s |                2.99 gb/s |
| `nk_reduce_minmax_i8_serial`       |                1.69 gb/s |                1.81 gb/s |                1.76 gb/s |
| `nk_reduce_moments_i8_neon`        |                26.0 gb/s |                16.5 gb/s |                12.4 gb/s |
| `nk_reduce_minmax_i8_neon`         |                23.9 gb/s |                28.2 gb/s |                25.9 gb/s |
| `nk_reduce_moments_i8_neonsdot`    |                41.4 gb/s |                43.6 gb/s |                31.1 gb/s |
| __u8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`      |                2.87 gb/s |                3.27 gb/s |                3.01 gb/s |
| `nk_reduce_minmax_u8_serial`       |                1.71 gb/s |                1.81 gb/s |                1.77 gb/s |
| `nk_reduce_moments_u8_neon`        |                27.2 gb/s |                16.8 gb/s |                12.7 gb/s |
| `nk_reduce_minmax_u8_neon`         |                25.1 gb/s |                26.4 gb/s |                27.2 gb/s |
| `nk_reduce_moments_u8_neonsdot`    |                42.4 gb/s |                40.5 gb/s |                31.0 gb/s |
| __i4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i4_serial`      |                1.92 gb/s |                2.39 gb/s |                2.23 gb/s |
| `nk_reduce_minmax_i4_serial`       |               0.653 gb/s |               0.739 gb/s |               0.748 gb/s |
| __u4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u4_serial`      |                2.05 gb/s |                2.56 gb/s |                2.44 gb/s |
| `nk_reduce_minmax_u4_serial`       |               0.690 gb/s |               0.787 gb/s |               0.797 gb/s |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u1_serial`      |                1.67 gb/s |                1.85 gb/s |                1.89 gb/s |
| `nk_reduce_minmax_u1_serial`       |                9.07 gb/s |                35.9 gb/s |                 113 gb/s |
| __i16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`     |                5.61 gb/s |                6.55 gb/s |                6.24 gb/s |
| `nk_reduce_minmax_i16_serial`      |                3.41 gb/s |                3.61 gb/s |                3.56 gb/s |
| `nk_reduce_moments_i16_neon`       |                21.4 gb/s |                15.1 gb/s |                11.1 gb/s |
| `nk_reduce_minmax_i16_neon`        |                25.0 gb/s |                26.5 gb/s |                24.1 gb/s |
| __u16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`     |                5.71 gb/s |                6.32 gb/s |                6.01 gb/s |
| `nk_reduce_minmax_u16_serial`      |                2.53 gb/s |                2.50 gb/s |                2.49 gb/s |
| `nk_reduce_moments_u16_neon`       |                21.5 gb/s |                15.2 gb/s |                10.8 gb/s |
| `nk_reduce_minmax_u16_neon`        |                24.8 gb/s |                26.5 gb/s |                24.0 gb/s |
| __i32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`     |                7.17 gb/s |                7.53 gb/s |                7.00 gb/s |
| `nk_reduce_minmax_i32_serial`      |                6.85 gb/s |                6.98 gb/s |                7.12 gb/s |
| `nk_reduce_moments_i32_neon`       |                8.27 gb/s |                6.28 gb/s |                6.15 gb/s |
| `nk_reduce_minmax_i32_neon`        |                25.4 gb/s |                26.0 gb/s |                24.0 gb/s |
| __u32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`     |                6.99 gb/s |                7.54 gb/s |                7.20 gb/s |
| `nk_reduce_minmax_u32_serial`      |                6.78 gb/s |                7.16 gb/s |                7.15 gb/s |
| `nk_reduce_moments_u32_neon`       |                17.7 gb/s |                11.9 gb/s |                10.7 gb/s |
| `nk_reduce_minmax_u32_neon`        |                25.7 gb/s |                26.4 gb/s |                24.6 gb/s |
| __i64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`     |                9.69 gb/s |                12.0 gb/s |                11.1 gb/s |
| `nk_reduce_minmax_i64_serial`      |                12.8 gb/s |                13.1 gb/s |                13.0 gb/s |
| `nk_reduce_moments_i64_neon`       |                14.5 gb/s |                12.2 gb/s |                12.0 gb/s |
| `nk_reduce_minmax_i64_neon`        |                16.8 gb/s |                15.5 gb/s |                14.5 gb/s |
| __u64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`     |                9.69 gb/s |                11.1 gb/s |                10.4 gb/s |
| `nk_reduce_minmax_u64_serial`      |                12.5 gb/s |                13.1 gb/s |                12.9 gb/s |
| `nk_reduce_moments_u64_neon`       |                27.1 gb/s |                21.0 gb/s |                20.7 gb/s |
| `nk_reduce_minmax_u64_neon`        |                17.0 gb/s |                15.6 gb/s |                15.0 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                               |                      256 |                     1024 |                     4096 |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`       |         6.57 gb/s, 0 ulp |         6.42 gb/s, 0 ulp |         6.56 gb/s, 0 ulp |
| `nk_reduce_moments_f64_v128`         |         16.7 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`        |         11.0 gb/s, 0 ulp |         10.7 gb/s, 0 ulp |         10.6 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_v128relaxed`   |         15.3 gb/s, 0 ulp |         15.6 gb/s, 0 ulp |         15.7 gb/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`       |         3.11 gb/s, 0 ulp |         3.09 gb/s, 0 ulp |         3.11 gb/s, 0 ulp |
| `nk_reduce_moments_f32_v128relaxed`  |       15.6 gb/s, 0.1 ulp |       11.1 gb/s, 0.5 ulp |         9.59 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`        |         4.10 gb/s, 0 ulp |         4.01 gb/s, 0 ulp |         3.99 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_v128relaxed`   |         13.5 gb/s, 0 ulp |         14.7 gb/s, 0 ulp |         15.6 gb/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`      |         1.64 gb/s, 0 ulp |         1.60 gb/s, 0 ulp |         1.60 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_v128`        |                        … |                        … |                        … |
| `nk_reduce_minmax_bf16_serial`       |         1.68 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |         1.96 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_v128relaxed`  |         8.59 gb/s, 0 ulp |         8.88 gb/s, 0 ulp |         9.78 gb/s, 0 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`       |         1.65 gb/s, 0 ulp |         1.61 gb/s, 0 ulp |         1.61 gb/s, 0 ulp |
| `nk_reduce_moments_f16_v128relaxed`  |         11.0 gb/s, 0 ulp |         10.1 gb/s, 0 ulp |       9.16 gb/s, 0.3 ulp |
| `nk_reduce_minmax_f16_serial`        |         1.67 gb/s, 0 ulp |         1.85 gb/s, 0 ulp |         1.95 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_v128relaxed`   |         5.63 gb/s, 0 ulp |         7.10 gb/s, 0 ulp |         7.55 gb/s, 0 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`      |        0.825 gb/s, 0 ulp |        0.806 gb/s, 0 ulp |        0.810 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_v128relaxed` |         3.29 gb/s, 0 ulp |         3.31 gb/s, 0 ulp |         3.32 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`       |         1.04 gb/s, 0 ulp |         1.10 gb/s, 0 ulp |         1.14 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_v128relaxed`  |         5.12 gb/s, 0 ulp |         10.1 gb/s, 0 ulp |         13.6 gb/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`      |        0.575 gb/s, 0 ulp |        0.564 gb/s, 0 ulp |        0.568 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_v128relaxed` |         2.80 gb/s, 0 ulp |         2.83 gb/s, 0 ulp |         2.84 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`       |         1.02 gb/s, 0 ulp |         1.13 gb/s, 0 ulp |         1.18 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_v128relaxed`  |         6.02 gb/s, 0 ulp |         9.78 gb/s, 0 ulp |         14.0 gb/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`      |        0.820 gb/s, 0 ulp |        0.813 gb/s, 0 ulp |        0.808 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_v128relaxed` |         5.04 gb/s, 0 ulp |         4.35 gb/s, 0 ulp |         4.04 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`       |        0.625 gb/s, 0 ulp |        0.621 gb/s, 0 ulp |        0.620 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_v128relaxed`  |         7.40 gb/s, 0 ulp |         12.1 gb/s, 0 ulp |         15.2 gb/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`      |        0.823 gb/s, 0 ulp |        0.817 gb/s, 0 ulp |        0.798 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_v128relaxed` |         8.88 gb/s, 0 ulp |         8.84 gb/s, 0 ulp |         8.12 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`       |        0.624 gb/s, 0 ulp |        0.621 gb/s, 0 ulp |        0.620 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_v128relaxed`  |         8.71 gb/s, 0 ulp |         12.0 gb/s, 0 ulp |         15.3 gb/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`        |                3.52 gb/s |                3.43 gb/s |                3.37 gb/s |
| `nk_reduce_moments_i8_v128`          |                20.1 gb/s |                20.8 gb/s |                17.4 gb/s |
| `nk_reduce_minmax_i8_serial`         |                1.37 gb/s |                1.39 gb/s |                1.39 gb/s |
| `nk_reduce_minmax_i8_v128relaxed`    |                12.1 gb/s |                17.8 gb/s |                21.5 gb/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`        |                3.46 gb/s |                3.43 gb/s |                3.37 gb/s |
| `nk_reduce_moments_u8_v128`          |                20.1 gb/s |                20.8 gb/s |                17.5 gb/s |
| `nk_reduce_minmax_u8_serial`         |                1.41 gb/s |                1.44 gb/s |                1.44 gb/s |
| `nk_reduce_minmax_u8_v128relaxed`    |                11.7 gb/s |                15.5 gb/s |                21.0 gb/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`       |                7.00 gb/s |                6.85 gb/s |                6.83 gb/s |
| `nk_reduce_moments_i16_v128`         |                12.4 gb/s |                9.29 gb/s |                8.22 gb/s |
| `nk_reduce_minmax_i16_serial`        |                2.76 gb/s |                2.78 gb/s |                2.78 gb/s |
| `nk_reduce_minmax_i16_v128relaxed`   |                17.5 gb/s |                13.5 gb/s |                15.0 gb/s |
| __u16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`       |                6.99 gb/s |                6.78 gb/s |                6.73 gb/s |
| `nk_reduce_moments_u16_v128`         |                12.3 gb/s |                9.29 gb/s |                8.23 gb/s |
| `nk_reduce_minmax_u16_serial`        |                2.83 gb/s |                2.90 gb/s |                2.90 gb/s |
| `nk_reduce_minmax_u16_v128relaxed`   |                15.4 gb/s |                13.5 gb/s |                15.1 gb/s |
| __i32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`       |                5.34 gb/s |                5.27 gb/s |                5.20 gb/s |
| `nk_reduce_moments_i32_v128`         |                5.96 gb/s |                5.91 gb/s |                5.95 gb/s |
| `nk_reduce_minmax_i32_serial`        |                6.38 gb/s |                6.41 gb/s |                6.43 gb/s |
| `nk_reduce_minmax_i32_v128relaxed`   |                13.0 gb/s |                14.8 gb/s |                15.6 gb/s |
| __u32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`       |                5.60 gb/s |                5.31 gb/s |                5.16 gb/s |
| `nk_reduce_moments_u32_v128`         |                1.54 gb/s |                1.47 gb/s |                1.45 gb/s |
| `nk_reduce_minmax_u32_serial`        |                6.33 gb/s |                6.40 gb/s |                6.43 gb/s |
| `nk_reduce_minmax_u32_v128relaxed`   |                13.4 gb/s |                14.8 gb/s |                15.6 gb/s |
| __i64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`       |                8.87 gb/s |                9.06 gb/s |                8.96 gb/s |
| `nk_reduce_moments_i64_v128relaxed`  |                6.91 gb/s |                6.89 gb/s |                6.98 gb/s |
| `nk_reduce_minmax_i64_serial`        |                12.7 gb/s |                12.8 gb/s |                12.8 gb/s |
| `nk_reduce_minmax_i64_v128relaxed`   |                15.3 gb/s |                15.7 gb/s |                15.8 gb/s |
| __u64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`       |                9.50 gb/s |                9.59 gb/s |                9.78 gb/s |
| `nk_reduce_moments_u64_v128relaxed`  |                2.83 gb/s |                2.78 gb/s |                2.74 gb/s |
| `nk_reduce_minmax_u64_serial`        |                12.7 gb/s |                12.8 gb/s |                12.8 gb/s |
| `nk_reduce_minmax_u64_v128relaxed`   |                3.08 gb/s |                3.10 gb/s |                3.12 gb/s |
