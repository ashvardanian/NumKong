# Vector-Vector Dot Products in NumKong

NumKong implements dot products for every numeric type supported by the library, as a core building block of higher-level functionality for vectors and higher rank tensors.

Dot product for real numbers and integers is defined as:

$$
\text{dot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot b_i
$$

For complex numbers, the dot product expands via the distributive property of complex multiplication:

$$
\text{dot}(a, b) = \sum_{i=0}^{n-1} (a_{i,re} \cdot b_{i,re} - a_{i,im} \cdot b_{i,im}) + j \sum_{i=0}^{n-1} (a_{i,re} \cdot b_{i,im} + a_{i,im} \cdot b_{i,re})
$$

The conjugate dot product negates the imaginary part of $b$:

$$
\text{vdot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot \bar{b_i} = \sum_{i=0}^{n-1} (a_{i,re} \cdot b_{i,re} + a_{i,im} \cdot b_{i,im}) + j \sum_{i=0}^{n-1} (a_{i,im} \cdot b_{i,re} - a_{i,re} \cdot b_{i,im})
$$

Where $\bar{b_i}$ is the complex conjugate of $b_i$.
Reformulating as Python pseudocode for interleaved real/imaginary scalar arrays:

```python
def dot_real(a: List[number], b: List[number]) -> number:
    return sum(ai * bi for ai, bi in zip(a, b))

def dot_complex(a: List[number], b: List[number]) -> Tuple[number, number]:
    a_re, a_im = a[0::2], a[1::2]
    b_re, b_im = b[0::2], b[1::2]
    ab_re = sum(ar * br - ai * bi for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    ab_im = sum(ar * bi + ai * br for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    return ab_re, ab_im

def vdot_complex(a: List[number], b: List[number]) -> Tuple[number, number]:
    a_re, a_im = a[0::2], a[1::2]
    b_re, b_im = b[0::2], b[1::2]
    ab_re = sum(ar * br + ai * bi for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    ab_im = sum(ai * br - ar * bi for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    return ab_re, ab_im
```

## Input & Output Types

Real and integer dot products:

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits      |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits      |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits   |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits   |
| `i8`       | `i32`       | 8-bit signed integers                          |
| `u8`       | `u32`       | 8-bit unsigned integers                        |
| `i4`       | `i32`       | 4-bit signed integers, packed nibble pairs     |
| `u4`       | `u32`       | 4-bit unsigned integers, packed nibble pairs   |
| `u1`       | `u32`       | 1-bit binary packed octets, popcount of AND    |

Complex dot products (both `dot` and `vdot`):

| Input Type | Output Type | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `f64c`     | `f64c`      | 64-bit complex pairs                       |
| `f32c`     | `f32c`      | 32-bit complex pairs                       |
| `f16c`     | `f32c`      | 16-bit complex pairs, widened output       |
| `bf16c`    | `f32c`      | 16-bit brain complex pairs, widened output |

## Optimizations

### Compensated Arithmetic for Large Floats

`nk_dot_f64_serial` uses Neumaier compensated summation — tracking a correction term adjusted by magnitude comparison at each step.
`nk_dot_f64_haswell`, `nk_dot_f64_skylake`, `nk_dot_f64_sve` implement the Dot2 algorithm by Ogita, Rump, and Oishi: TwoProd via FMA captures the rounding error of each product exactly, and a TwoSum chain propagates it through the accumulator.
On SVE, the final horizontal reduction uses `svtbl` to extract upper halves at each tree level, applying TwoSum at every stage.
The serial path uses Neumaier because it processes one element at a time and can cheaply branch on magnitudes.
Dot2 avoids those branches entirely — TwoProd and TwoSum are pure arithmetic with no comparisons, mapping naturally to wide SIMD where branching per lane is impossible.

### Lookup Tables for Mini-Floats

`nk_dot_e2m3_haswell`, `nk_dot_e3m2_haswell`, `nk_dot_e2m3_skylake`, `nk_dot_e3m2_skylake` encode 32 MX format values into scaled integers via dual 16-entry LUTs loaded into vector registers.
The low 4 magnitude bits index `VPSHUFB`, bit 4 selects between the lower and upper table via blending, and the results feed into `VPMADDUBSW` + `VPMADDWD` chains with a final $\div 256$ scaling.
The Sapphire-specific MX implementation in `sapphire.h` replaces this with a single 64-entry signed LUT via `VPERMUTEX2VAR`, where the sign bit naturally selects between positive and negative tables.
That path accumulates in native Float16 via `VFMADD_PH` and flushes to Float32 every 128 elements to avoid overflow.

### Algebraic Domain Shifting

`nk_dot_i8_icelake`, `nk_dot_u8_icelake` work around `VPDPBUSD` requiring UInt8 × Int8 operands.
For Int8 × Int8, one operand is XORed with `0x80` to shift to unsigned, and the correction $128 \cdot \sum b_i$ is computed via `VPSADBW`, which runs on port 5 and avoids contention with `DPBUSD` on ports 0-1.
`nk_dot_i4_icelake` extends this to packed nibbles using the identity $(a'-8)(b'-8) = a' b' - 8(a'+b') + 64$ — two `VPDPBUSD` calls handle low and high nibbles separately, with SAD-based correction.
`nk_dot_i8_v128relaxed`, `nk_dot_u8_v128relaxed` face an even tighter constraint: WASM's `i32x4_relaxed_dot_i8x16_i7x16_add` computes Int8 × Int7, so the sign bit of one operand must be masked off entirely.
For Int8 × Int8, the sign bit of $b$ is cleared to produce a 7-bit value, and a windowed correction $-128 \cdot \sum_{b_i < 0} a_i$ is accumulated in Int16 and flushed every 127 iterations to prevent overflow.
For UInt8 × UInt8, $b$ is XORed with `0x80` to shift into signed range, same as Ice Lake, with the correction $128 \cdot \sum a_i$ computed via pairwise widening adds.

### Octave Decomposition for E4M3 via VNNI

`nk_dot_e4m3_icelake` splits the 4-bit E4M3 exponent into 2 "octave" bits (top) and 2 "remainder" bits (bottom).
The bottom 5 bits (2 remainder + 3 mantissa) map via `VPERMB` to u8 integers in [0, 120] — identical structure to the E2M3 $\times 16$ LUT.
A subnormal fixup replaces LUT entries for magnitude < 8 with $2 \times \text{mantissa}$ via a second masked `VPERMB`, avoiding `VPADDB` on the VPDPBUSD execution ports.
Sign is computed via `VPTERNLOGD` with immediate 0x14, fusing `(a \oplus b) \wedge \lnot \text{0x7F}` in one instruction.
The 4 octave bins per operand produce $4 \times 4 = 16$ `VPDPBUSD` cross-products accumulated into 7 registers grouped by octave sum $k = o_a + o_b \in [0, 6]$.
Each accumulator is scaled by $2^{4k-20}$ — an exact power of two, introducing no rounding.
This processes 64 E4M3 bytes per iteration in u8, doubling the element density of the BF16 upcast path.

### Widening Fusion Through BFloat16 on x86

`nk_dot_e5m2_genoa` converts FP8 values to BF16, then accumulates via `VDPBF16PS`, reusing Genoa's BF16 dot-product instruction for FP8 types.
Each `VDPBF16PS` fuses two BF16 multiply-adds per 32-bit lane at 6-cycle throughput.
On Skylake-X–class CPUs without BF16 dot-product hardware, `nk_dot_e4m3_skylake` / `nk_dot_e5m2_skylake` (and their Haswell twins `nk_dot_e4m3_haswell` / `nk_dot_e5m2_haswell`) instead route through the Giesen-style FP8 → F16 fake-bit-pattern cast, widen via `VCVTPH2PS`, and accumulate in F32 with two independent FMA chains reducing into a single register — avoiding the 3-chain scheduler-stall of the BF16 algebraic form on kernels without native BF16 FMA.
`nk_dot_bf16c_genoa` uses the same instruction for complex BF16, preparing operands with `VPSHUFB` for lane swapping and `VPXORD` with `0x80000000` for sign flips before feeding into `VDPBF16PS`.

### Deferred Sign-Flip in Complex Dot Products

The Haswell BFloat16Complex/Float16Complex/Float32Complex kernels compute $\sum (a_r b_r - a_i b_i)$ without per-pair subtraction.
Instead, two accumulators collect interleaved products $[a_r b_r, a_i b_i, \ldots]$ and $[a_r b_i, a_i b_r, \ldots]$, and a post-loop XOR flips the sign of every odd lane to produce the subtraction.
This gives one FMA per accumulator per iteration, but each lane grows $O(n)$ while the true result is $O(\sqrt{n})$.
The Float32Complex kernel absorbs this via Float64 accumulators; Genoa's `VDPBF16PS` and ARM's `FMLSL` pair terms naturally.
For BFloat16Complex/Float16Complex on Haswell the accumulator is Float32, so the $O(\log n)$ precision loss from lane separation is visible in max ULP at large $n$, though mean ULP remains low.

### Widening Fusion Through Float16 on Arm

`nk_dot_f16_neonfhm`, `nk_dot_f16c_neonfhm` use the ARMv8.4-FHM instructions `FMLAL`/`FMLSL`, which fuse FP16-to-FP32 conversion with multiply-accumulate in a single operation.
`vfmlalq_low_f16` and `vfmlalq_high_f16` process the lower and upper 4 elements of an 8-wide FP16 vector respectively.
For complex dot products, `FMLSL` provides the subtraction path $a_{re} b_{im} - a_{im} b_{re}$ without a separate negate step.

### Widening Chains on RISC-V

`nk_dot_i8_rvv`, `nk_dot_u8_rvv` use `vwmul` for Int8 × Int8 → Int16 widening multiply followed by `vwadd` to widen-accumulate into Int32 — a two-stage chain that naturally prevents overflow.
`nk_dot_bf16_rvvbf16` uses the Zvfbfwma extension's `vfwmaccbf16` for fused BFloat16 × BFloat16 → Float32 widening multiply-accumulate.
`nk_dot_e4m3_rvvbf16`, `nk_dot_e5m2_rvvbf16` convert Float8 to BFloat16 via 256-entry LUTs, then feed the same `vfwmaccbf16` path.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in gb/s as the number of bytes read per second amortized for a large batch of vector pairs.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Rows marked `🧩` use external BLAS baselines rather than NumKong kernels.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                  |                      256 |                     1024 |                     4096 |
| :---------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f64c_with_blas` 🧩  |        29.1 gb/s, 25 ulp |        27.6 gb/s, 97 ulp |        13.6 gb/s, 32 ulp |
| `vdot_f64c_with_blas` 🧩 |        29.1 gb/s, 18 ulp |        27.9 gb/s, 17 ulp |        15.3 gb/s, 25 ulp |
| `nk_dot_f64c_serial`    |       5.45 gb/s, 3.9 ulp |       6.49 gb/s, 9.0 ulp |       6.84 gb/s, 2.9 ulp |
| `nk_vdot_f64c_serial`   |       5.47 gb/s, 4.6 ulp |       6.41 gb/s, 1.6 ulp |       6.76 gb/s, 2.2 ulp |
| `nk_dot_f64c_skylake`   |         23.8 gb/s, 0 ulp |         23.4 gb/s, 0 ulp |         11.8 gb/s, 0 ulp |
| `nk_vdot_f64c_skylake`  |         23.6 gb/s, 0 ulp |         23.7 gb/s, 0 ulp |         11.6 gb/s, 0 ulp |
| __f32c__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f32c_with_blas` 🧩  |       28.7 gb/s, 8.6 ulp |        29.8 gb/s, 13 ulp |        15.8 gb/s, 19 ulp |
| `vdot_f32c_with_blas` 🧩 |        29.2 gb/s, 11 ulp |        30.2 gb/s, 14 ulp |        15.7 gb/s, 21 ulp |
| `nk_dot_f32c_serial`    |         9.46 gb/s, 0 ulp |         9.82 gb/s, 0 ulp |         9.71 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`   |         9.64 gb/s, 0 ulp |         9.95 gb/s, 0 ulp |         10.1 gb/s, 0 ulp |
| `nk_dot_f32c_haswell`   |         22.4 gb/s, 0 ulp |         22.2 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |
| `nk_vdot_f32c_haswell`  |         22.4 gb/s, 0 ulp |         21.8 gb/s, 0 ulp |         14.7 gb/s, 0 ulp |
| `nk_dot_f32c_skylake`   |         25.6 gb/s, 0 ulp |         27.2 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |
| `nk_vdot_f32c_skylake`  |         27.8 gb/s, 0 ulp |         27.4 gb/s, 0 ulp |         18.8 gb/s, 0 ulp |
| __bf16c__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`   |      0.628 gb/s, 0.1 ulp |      0.627 gb/s, 2.3 ulp |      0.626 gb/s, 7.9 ulp |
| `nk_vdot_bf16c_serial`  |      0.622 gb/s, 0.2 ulp |      0.624 gb/s, 2.1 ulp |     0.627 gb/s, 11.2 ulp |
| `nk_dot_bf16c_haswell`  |       21.5 gb/s, 0.1 ulp |       18.5 gb/s, 1.3 ulp |       18.4 gb/s, 3.4 ulp |
| `nk_vdot_bf16c_haswell` |       21.9 gb/s, 0.8 ulp |       19.0 gb/s, 2.0 ulp |       18.5 gb/s, 4.5 ulp |
| `nk_dot_bf16c_genoa`    |         37.9 gb/s, 0 ulp |       30.3 gb/s, 1.1 ulp |       29.5 gb/s, 2.8 ulp |
| `nk_vdot_bf16c_genoa`   |       36.1 gb/s, 0.7 ulp |       30.2 gb/s, 1.2 ulp |       30.2 gb/s, 3.3 ulp |
| __f16c__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`    |      2.02 gb/s, 14.4 ulp |      2.00 gb/s, 27.3 ulp |      2.02 gb/s, 34.0 ulp |
| `nk_vdot_f16c_serial`   |      1.67 gb/s, 15.0 ulp |      1.64 gb/s, 26.3 ulp |      1.64 gb/s, 34.2 ulp |
| `nk_dot_f16c_haswell`   |      23.9 gb/s, 12.7 ulp |      19.4 gb/s, 22.3 ulp |      19.3 gb/s, 40.1 ulp |
| `nk_vdot_f16c_haswell`  |      24.0 gb/s, 11.1 ulp |      20.0 gb/s, 17.4 ulp |      17.1 gb/s, 29.2 ulp |
| __f64__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f64_with_blas` 🧩   |       27.8 gb/s, 6.9 ulp |       30.1 gb/s, 9.3 ulp |        15.7 gb/s, 20 ulp |
| `nk_dot_f64_serial`     |       4.28 gb/s, 2.2 ulp |       4.39 gb/s, 2.0 ulp |       4.42 gb/s, 3.3 ulp |
| `nk_dot_f64_haswell`    |         24.2 gb/s, 0 ulp |         25.7 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| `nk_dot_f64_skylake`    |         29.0 gb/s, 0 ulp |         28.6 gb/s, 0 ulp |         24.9 gb/s, 0 ulp |
| __f32__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f32_with_blas` 🧩   |        47.8 gb/s, 14 ulp |        30.7 gb/s, 14 ulp |        29.7 gb/s, 15 ulp |
| `nk_dot_f32_serial`     |         11.0 gb/s, 0 ulp |         11.2 gb/s, 0 ulp |         11.5 gb/s, 0 ulp |
| `nk_dot_f32_haswell`    |         30.5 gb/s, 0 ulp |         23.9 gb/s, 0 ulp |         24.4 gb/s, 0 ulp |
| `nk_dot_f32_skylake`    |         44.2 gb/s, 0 ulp |         29.8 gb/s, 0 ulp |         30.0 gb/s, 0 ulp |
| __bf16__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`    |        0.633 gb/s, 0 ulp |      0.630 gb/s, 0.5 ulp |      0.638 gb/s, 5.4 ulp |
| `nk_dot_bf16_haswell`   |         39.3 gb/s, 0 ulp |       25.5 gb/s, 0.2 ulp |      20.2 gb/s, 25.3 ulp |
| `nk_dot_bf16_skylake`   |         62.7 gb/s, 0 ulp |       30.2 gb/s, 0.2 ulp |       29.5 gb/s, 2.3 ulp |
| `nk_dot_bf16_genoa`     |         88.8 gb/s, 0 ulp |       29.7 gb/s, 0.2 ulp |       31.2 gb/s, 2.2 ulp |
| __f16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`     |      1.31 gb/s, 11.5 ulp |      1.32 gb/s, 33.7 ulp |      1.30 gb/s, 59.7 ulp |
| `nk_dot_f16_haswell`    |       31.3 gb/s, 7.0 ulp |      22.8 gb/s, 14.0 ulp |      19.8 gb/s, 29.8 ulp |
| `nk_dot_f16_skylake`    |       54.9 gb/s, 6.2 ulp |       31.7 gb/s, 8.6 ulp |      30.9 gb/s, 22.8 ulp |
| __e5m2__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`    |         1.90 gb/s, 0 ulp |         1.07 gb/s, 0 ulp |         1.08 gb/s, 0 ulp |
| `nk_dot_e5m2_haswell`   |         4.92 gb/s, 0 ulp |         4.95 gb/s, 0 ulp |         4.80 gb/s, 0 ulp |
| `nk_dot_e5m2_skylake`   |         6.20 gb/s, 0 ulp |         6.36 gb/s, 0 ulp |         6.25 gb/s, 0 ulp |
| `nk_dot_e5m2_genoa`     |         12.1 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |
| __e4m3__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`    |        0.762 gb/s, 0 ulp |        0.424 gb/s, 0 ulp |        0.420 gb/s, 0 ulp |
| `nk_dot_e4m3_haswell`   |         3.78 gb/s, 0 ulp |         3.77 gb/s, 0 ulp |         3.75 gb/s, 0 ulp |
| `nk_dot_e4m3_skylake`   |         5.10 gb/s, 0 ulp |         5.16 gb/s, 0 ulp |         5.21 gb/s, 0 ulp |
| `nk_dot_e4m3_icelake`   |         13.2 gb/s, 0 ulp |         14.9 gb/s, 0 ulp |         14.7 gb/s, 0 ulp |
| __e3m2__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`    |         1.47 gb/s, 0 ulp |         1.05 gb/s, 0 ulp |         1.04 gb/s, 0 ulp |
| `nk_dot_e3m2_haswell`   |         12.0 gb/s, 0 ulp |         12.2 gb/s, 0 ulp |         12.2 gb/s, 0 ulp |
| `nk_dot_e3m2_skylake`   |         21.6 gb/s, 0 ulp |         23.1 gb/s, 0 ulp |         23.2 gb/s, 0 ulp |
| `nk_dot_e3m2_icelake`   |         23.1 gb/s, 0 ulp |         24.3 gb/s, 0 ulp |         23.9 gb/s, 0 ulp |
| __e2m3__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`    |         1.87 gb/s, 0 ulp |         1.25 gb/s, 0 ulp |         1.96 gb/s, 0 ulp |
| `nk_dot_e2m3_haswell`   |         20.5 gb/s, 0 ulp |         20.4 gb/s, 0 ulp |         19.3 gb/s, 0 ulp |
| `nk_dot_e2m3_skylake`   |         35.7 gb/s, 0 ulp |         33.2 gb/s, 0 ulp |         30.7 gb/s, 0 ulp |
| `nk_dot_e2m3_icelake`   |         58.0 gb/s, 0 ulp |         46.0 gb/s, 0 ulp |         31.5 gb/s, 0 ulp |
| `nk_dot_e2m3_alder`     |         29.9 gb/s, 0 ulp |         30.8 gb/s, 0 ulp |         29.1 gb/s, 0 ulp |
| __i8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`      |                16.9 gb/s |                16.8 gb/s |                15.6 gb/s |
| `nk_dot_i8_haswell`     |                43.2 gb/s |                35.8 gb/s |                29.1 gb/s |
| `nk_dot_i8_skylake`     |                52.9 gb/s |                36.5 gb/s |                28.5 gb/s |
| `nk_dot_i8_icelake`     |                64.0 gb/s |                46.2 gb/s |                26.8 gb/s |
| `nk_dot_i8_alder`       |                42.8 gb/s |                40.4 gb/s |                31.1 gb/s |
| __u8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`      |                16.9 gb/s |                16.5 gb/s |                15.8 gb/s |
| `nk_dot_u8_haswell`     |                47.7 gb/s |                37.7 gb/s |                29.1 gb/s |
| `nk_dot_u8_skylake`     |                48.7 gb/s |                32.6 gb/s |                27.5 gb/s |
| `nk_dot_u8_icelake`     |                68.4 gb/s |                46.9 gb/s |                30.2 gb/s |
| `nk_dot_u8_alder`       |                42.1 gb/s |                41.8 gb/s |                31.6 gb/s |
| __i4__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`      |                9.37 gb/s |                11.8 gb/s |                11.8 gb/s |
| `nk_dot_i4_haswell`     |                8.22 gb/s |                8.53 gb/s |                8.23 gb/s |
| `nk_dot_i4_icelake`     |                24.3 gb/s |                36.3 gb/s |                25.5 gb/s |
| __u4__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`      |                10.6 gb/s |                12.0 gb/s |                11.9 gb/s |
| `nk_dot_u4_haswell`     |                15.0 gb/s |                16.4 gb/s |                14.3 gb/s |
| `nk_dot_u4_icelake`     |                48.1 gb/s |                64.4 gb/s |                30.9 gb/s |
| __u1__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`      |                3.92 gb/s |                5.04 gb/s |                4.97 gb/s |
| `nk_dot_u1_haswell`     |                14.7 gb/s |                43.2 gb/s |                69.4 gb/s |
| `nk_dot_u1_icelake`     |                17.9 gb/s |                68.8 gb/s |                 110 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`       |       1.76 gb/s, 5.3 ulp |       2.43 gb/s, 3.3 ulp |       0.28 gb/s, 2.9 ulp |
| `nk_vdot_f64c_serial`      |       1.02 gb/s, 3.5 ulp |       2.44 gb/s, 5.5 ulp |       0.15 gb/s, 2.2 ulp |
| `nk_dot_f64c_v128relaxed`  |      2.80 gb/s, 37.8 ulp |      3.01 gb/s, 34.9 ulp |       0.21 gb/s, 167 ulp |
| `nk_vdot_f64c_v128relaxed` |      2.06 gb/s, 20.1 ulp |      2.87 gb/s, 51.4 ulp |      0.04 gb/s, 57.2 ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`       |         1.26 gb/s, 0 ulp |         1.74 gb/s, 0 ulp |         0.08 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`      |         1.13 gb/s, 0 ulp |         1.78 gb/s, 0 ulp |         0.21 gb/s, 0 ulp |
| `nk_dot_f32c_v128relaxed`  |         1.62 gb/s, 0 ulp |         1.92 gb/s, 0 ulp |         0.20 gb/s, 0 ulp |
| `nk_vdot_f32c_v128relaxed` |         1.66 gb/s, 0 ulp |         1.69 gb/s, 0 ulp |         0.13 gb/s, 0 ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`      |       1.00 gb/s, 0.1 ulp |       2.29 gb/s, 1.7 ulp |       0.08 gb/s, 7.9 ulp |
| `nk_vdot_bf16c_serial`     |      0.581 gb/s, 0.1 ulp |      0.919 gb/s, 2.9 ulp |      0.30 gb/s, 11.2 ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`       |      1.39 gb/s, 12.6 ulp |     0.759 gb/s, 22.2 ulp |        0.23 gb/s, 34 ulp |
| `nk_vdot_f16c_serial`      |      1.11 gb/s, 14.4 ulp |     0.828 gb/s, 41.8 ulp |      0.02 gb/s, 34.2 ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`        |       1.53 gb/s, 3.1 ulp |       1.72 gb/s, 2.5 ulp |       0.20 gb/s, 3.3 ulp |
| `nk_dot_f64_v128relaxed`   |       2.62 gb/s, 3.2 ulp |       2.11 gb/s, 3.6 ulp |       0.28 gb/s, 3.8 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`        |         1.95 gb/s, 0 ulp |         1.89 gb/s, 0 ulp |         0.28 gb/s, 0 ulp |
| `nk_dot_f32_v128relaxed`   |        0.083 gb/s, 0 ulp |         1.61 gb/s, 0 ulp |         1.37 gb/s, 0 ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`       |         2.90 gb/s, 0 ulp |       2.27 gb/s, 0.5 ulp |       0.22 gb/s, 5.2 ulp |
| `nk_dot_bf16_v128relaxed`  |        0.521 gb/s, 0 ulp |       2.30 gb/s, 0.3 ulp |       0.30 gb/s, 2.4 ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`        |     0.648 gb/s, 13.5 ulp |       0.712 gb/s, 32 ulp |      0.08 gb/s, 59.7 ulp |
| `nk_dot_f16_v128relaxed`   |       1.58 gb/s, 7.0 ulp |      1.05 gb/s, 30.8 ulp |      0.09 gb/s, 65.1 ulp |
| __e5m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`       |         1.27 gb/s, 0 ulp |        0.679 gb/s, 0 ulp |         0.10 gb/s, 0 ulp |
| `nk_dot_e5m2_v128relaxed`  |        0.970 gb/s, 0 ulp |        0.955 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| __e4m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`       |        0.312 gb/s, 0 ulp |        0.342 gb/s, 0 ulp |         0.12 gb/s, 0 ulp |
| `nk_dot_e4m3_v128relaxed`  |         1.05 gb/s, 0 ulp |        0.721 gb/s, 0 ulp |         0.30 gb/s, 0 ulp |
| __e3m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`       |        0.565 gb/s, 0 ulp |        0.552 gb/s, 0 ulp |         0.06 gb/s, 0 ulp |
| `nk_dot_e3m2_v128relaxed`  |        0.670 gb/s, 0 ulp |         2.91 gb/s, 0 ulp |         0.24 gb/s, 0 ulp |
| __e2m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`       |        0.584 gb/s, 0 ulp |        0.661 gb/s, 0 ulp |         0.07 gb/s, 0 ulp |
| `nk_dot_e2m3_v128relaxed`  |         2.69 gb/s, 0 ulp |        0.131 gb/s, 0 ulp |         0.09 gb/s, 0 ulp |
| __i8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`         |                1.17 gb/s |                1.17 gb/s |                0.29 gb/s |
| `nk_dot_i8_v128relaxed`    |                1.71 gb/s |               0.896 gb/s |                0.24 gb/s |
| __u8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`         |                1.16 gb/s |               0.658 gb/s |                0.30 gb/s |
| `nk_dot_u8_v128relaxed`    |               0.873 gb/s |               0.997 gb/s |                0.15 gb/s |
| __i4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`         |               0.217 gb/s |               0.226 gb/s |                0.28 gb/s |
| `nk_dot_i4_v128relaxed`    |                1.53 gb/s |                2.87 gb/s |                0.24 gb/s |
| __u4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`         |               0.303 gb/s |               0.250 gb/s |               0.003 gb/s |
| `nk_dot_u4_v128relaxed`    |               0.126 gb/s |                2.70 gb/s |                0.08 gb/s |
| __u1__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`         |                1.95 gb/s |                1.53 gb/s |                0.09 gb/s |
| `nk_dot_u1_v128relaxed`    |               0.548 gb/s |                1.88 gb/s |                0.13 gb/s |

### Apple M5

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`      |         8.02 gb/s, 5 ulp |         7.30 gb/s, 3 ulp |       7.25 gb/s, 9.7 ulp |
| `nk_vdot_f64c_serial`     |       8.29 gb/s, 4.2 ulp |       7.53 gb/s, 3.3 ulp |       7.38 gb/s, 3.3 ulp |
| `nk_dot_f64c_neon`        |         23.7 gb/s, 0 ulp |         21.6 gb/s, 0 ulp |         21.3 gb/s, 0 ulp |
| `nk_vdot_f64c_neon`       |         23.6 gb/s, 0 ulp |         21.8 gb/s, 0 ulp |         20.9 gb/s, 0 ulp |
| __f32c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`      |         27.8 gb/s, 0 ulp |         24.6 gb/s, 0 ulp |         23.2 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`     |         27.2 gb/s, 0 ulp |         24.0 gb/s, 0 ulp |         22.6 gb/s, 0 ulp |
| `nk_dot_f32c_neon`        |         22.8 gb/s, 0 ulp |         18.2 gb/s, 0 ulp |         16.9 gb/s, 0 ulp |
| `nk_vdot_f32c_neon`       |         22.7 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |
| __bf16c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`     |       15.6 gb/s, 0.2 ulp |       12.5 gb/s, 2.8 ulp |      12.5 gb/s, 15.8 ulp |
| `nk_vdot_bf16c_serial`    |       15.9 gb/s, 0.2 ulp |       12.9 gb/s, 2.6 ulp |      11.7 gb/s, 11.4 ulp |
| `nk_dot_bf16c_neonbfdot`  |       26.3 gb/s, 0.1 ulp |         18.5 gb/s, 2 ulp |       17.6 gb/s, 8.8 ulp |
| `nk_vdot_bf16c_neonbfdot` |       26.5 gb/s, 0.1 ulp |       18.2 gb/s, 1.8 ulp |       17.3 gb/s, 8.8 ulp |
| __f16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`      |      15.8 gb/s, 20.8 ulp |      13.0 gb/s, 64.1 ulp |      12.3 gb/s, 73.1 ulp |
| `nk_vdot_f16c_serial`     |      15.8 gb/s, 24.8 ulp |      13.0 gb/s, 31.9 ulp |       12.3 gb/s, 137 ulp |
| `nk_dot_f16c_neonhalf`    |       26.1 gb/s, 3.0 ulp |       18.4 gb/s, 6.5 ulp |      16.8 gb/s, 20.5 ulp |
| `nk_vdot_f16c_neonhalf`   |      26.1 gb/s, 34.9 ulp |      18.5 gb/s, 40.7 ulp |      17.0 gb/s, 73.1 ulp |
| `nk_dot_f16c_neonfhm`     |       25.3 gb/s, 3.0 ulp |       17.1 gb/s, 6.5 ulp |      15.9 gb/s, 20.5 ulp |
| `nk_vdot_f16c_neonfhm`    |      25.0 gb/s, 31.4 ulp |      17.0 gb/s, 38.6 ulp |      15.8 gb/s, 67.6 ulp |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`       |       8.11 gb/s, 2.4 ulp |       8.13 gb/s, 175 ulp |       8.09 gb/s, 2.7 ulp |
| `nk_dot_f64_neon`         |         44.2 gb/s, 0 ulp |         42.3 gb/s, 0 ulp |         38.4 gb/s, 0 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`       |         23.3 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |         14.6 gb/s, 0 ulp |
| `nk_dot_f32_neon`         |         46.4 gb/s, 0 ulp |         38.0 gb/s, 0 ulp |         34.8 gb/s, 0 ulp |
| __bf16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`      |         12.4 gb/s, 0 ulp |       8.59 gb/s, 0.9 ulp |         7.36 gb/s, 6 ulp |
| `nk_dot_bf16_neon`        |       39.0 gb/s, 3.7 ulp |       27.2 gb/s, 3.7 ulp |       19.9 gb/s, 3.7 ulp |
| `nk_dot_bf16_neonbfdot`   |         70.8 gb/s, 0 ulp |       60.8 gb/s, 0.6 ulp |       47.8 gb/s, 4.5 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`       |        12.0 gb/s, 19 ulp |      8.33 gb/s, 31.1 ulp |      7.11 gb/s, 57.8 ulp |
| `nk_dot_f16_neon`         |      35.7 gb/s, 33.4 ulp |      25.8 gb/s, 37.4 ulp |      21.3 gb/s, 23.1 ulp |
| `nk_dot_f16_neonfhm`      |      48.7 gb/s, 14.9 ulp |      27.5 gb/s, 26.7 ulp |      18.8 gb/s, 39.9 ulp |
| __e5m2__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`      |         3.80 gb/s, 0 ulp |         3.41 gb/s, 0 ulp |         3.41 gb/s, 0 ulp |
| `nk_dot_e5m2_neon`        |         19.0 gb/s, 0 ulp |         13.2 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |
| `nk_dot_e5m2_neonfhm`     |         25.6 gb/s, 0 ulp |         15.3 gb/s, 0 ulp |         9.55 gb/s, 0 ulp |
| `nk_dot_e5m2_neonbfdot`   |         3.65 gb/s, 0 ulp |         3.82 gb/s, 0 ulp |         3.68 gb/s, 0 ulp |
| __e4m3__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`      |         1.74 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |         1.71 gb/s, 0 ulp |
| `nk_dot_e4m3_neon`        |         4.44 gb/s, 0 ulp |         4.51 gb/s, 0 ulp |         4.57 gb/s, 0 ulp |
| `nk_dot_e4m3_neonfhm`     |         10.1 gb/s, 0 ulp |         8.51 gb/s, 0 ulp |         7.96 gb/s, 0 ulp |
| `nk_dot_e4m3_neonbfdot`   |         3.59 gb/s, 0 ulp |         3.68 gb/s, 0 ulp |         3.64 gb/s, 0 ulp |
| __e3m2__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`      |         2.51 gb/s, 0 ulp |         2.33 gb/s, 0 ulp |         2.24 gb/s, 0 ulp |
| `nk_dot_e3m2_neonsdot`    |         20.5 gb/s, 0 ulp |         20.7 gb/s, 0 ulp |         20.1 gb/s, 0 ulp |
| __e2m3__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`      |         2.54 gb/s, 0 ulp |         2.27 gb/s, 0 ulp |         2.29 gb/s, 0 ulp |
| `nk_dot_e2m3_neonsdot`    |         47.3 gb/s, 0 ulp |         47.5 gb/s, 0 ulp |         43.4 gb/s, 0 ulp |
| __i8__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`        |                 115 gb/s |                 102 gb/s |                92.3 gb/s |
| `nk_dot_i8_neonsdot`      |                92.8 gb/s |                87.4 gb/s |                59.9 gb/s |
| __u8__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`        |                 110 gb/s |                99.2 gb/s |                94.9 gb/s |
| `nk_dot_u8_neonsdot`      |                92.5 gb/s |                86.6 gb/s |                59.5 gb/s |
| __i4__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`        |                23.0 gb/s |                24.4 gb/s |                24.2 gb/s |
| `nk_dot_i4_neonsdot`      |                58.2 gb/s |                44.7 gb/s |                30.4 gb/s |
| __u4__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`        |                25.3 gb/s |                27.2 gb/s |                26.9 gb/s |
| `nk_dot_u4_neonsdot`      |                67.3 gb/s |                47.4 gb/s |                29.4 gb/s |
| __u1__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`        |                7.01 gb/s |                7.63 gb/s |                7.19 gb/s |
| `nk_dot_u1_neon`          |                33.3 gb/s |                64.6 gb/s |                88.0 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`       |       4.17 gb/s, 3.8 ulp |       6.03 gb/s, 3.9 ulp |       6.37 gb/s, 3.2 ulp |
| `nk_vdot_f64c_serial`      |       6.00 gb/s, 3.8 ulp |       6.55 gb/s, 3.4 ulp |      6.83 gb/s, 15.1 ulp |
| `nk_dot_f64c_v128relaxed`  |        46.7 gb/s, 26 ulp |        38.2 gb/s, 42 ulp |        40.5 gb/s, 88 ulp |
| `nk_vdot_f64c_v128relaxed` |      46.1 gb/s, 22.8 ulp |      39.8 gb/s, 37.3 ulp |      39.9 gb/s, 43.6 ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`       |         20.9 gb/s, 0 ulp |         21.3 gb/s, 0 ulp |         22.5 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`      |         19.9 gb/s, 0 ulp |         21.5 gb/s, 0 ulp |         22.4 gb/s, 0 ulp |
| `nk_dot_f32c_v128relaxed`  |         22.4 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |         19.8 gb/s, 0 ulp |
| `nk_vdot_f32c_v128relaxed` |         21.9 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |         19.9 gb/s, 0 ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`      |       10.3 gb/s, 0.1 ulp |       11.6 gb/s, 2.5 ulp |        11.6 gb/s, 10 ulp |
| `nk_vdot_bf16c_serial`     |       10.7 gb/s, 0.2 ulp |       11.7 gb/s, 2.1 ulp |      11.6 gb/s, 11.4 ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`       |        3.76 gb/s, 13 ulp |        3.83 gb/s, 20 ulp |        3.86 gb/s, 90 ulp |
| `nk_vdot_f16c_serial`      |      3.81 gb/s, 13.9 ulp |      3.89 gb/s, 35.5 ulp |      3.85 gb/s, 42.4 ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`        |       7.26 gb/s, 2.4 ulp |       7.45 gb/s, 2.6 ulp |       7.96 gb/s, 2.2 ulp |
| `nk_dot_f64_v128relaxed`   |       38.7 gb/s, 2.6 ulp |       42.0 gb/s, 3.2 ulp |       43.9 gb/s, 2.6 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`        |        19.0 gb/s, 16 ulp |        14.6 gb/s, 69 ulp |       14.0 gb/s, 104 ulp |
| `nk_dot_f32_v128relaxed`   |         20.4 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |         18.7 gb/s, 0 ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`       |         9.53 gb/s, 0 ulp |       7.42 gb/s, 0.6 ulp |       7.20 gb/s, 5.9 ulp |
| `nk_dot_bf16_v128relaxed`  |         41.9 gb/s, 0 ulp |       28.3 gb/s, 0.4 ulp |       21.5 gb/s, 3.7 ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`        |        3.31 gb/s, 16 ulp |        3.63 gb/s, 26 ulp |        3.66 gb/s, 53 ulp |
| `nk_dot_f16_v128relaxed`   |       11.4 gb/s, 9.0 ulp |        11.2 gb/s, 23 ulp |        12.0 gb/s, 39 ulp |
| __e5m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`       |         3.02 gb/s, 0 ulp |         2.95 gb/s, 0 ulp |         3.16 gb/s, 0 ulp |
| `nk_dot_e5m2_v128relaxed`  |         3.47 gb/s, 0 ulp |         3.45 gb/s, 0 ulp |         3.48 gb/s, 0 ulp |
| __e4m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`       |        0.978 gb/s, 0 ulp |        0.893 gb/s, 0 ulp |        0.936 gb/s, 0 ulp |
| `nk_dot_e4m3_v128relaxed`  |         2.78 gb/s, 0 ulp |         2.75 gb/s, 0 ulp |         2.78 gb/s, 0 ulp |
| __e3m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`       |         3.13 gb/s, 0 ulp |         2.95 gb/s, 0 ulp |         3.16 gb/s, 0 ulp |
| `nk_dot_e3m2_v128relaxed`  |         12.1 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |
| __e2m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`       |         2.99 gb/s, 0 ulp |         3.00 gb/s, 0 ulp |         3.17 gb/s, 0 ulp |
| `nk_dot_e2m3_v128relaxed`  |         20.4 gb/s, 0 ulp |         20.6 gb/s, 0 ulp |         21.7 gb/s, 0 ulp |
| __i8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`         |                22.2 gb/s |                19.6 gb/s |                17.8 gb/s |
| `nk_dot_i8_v128relaxed`    |                42.0 gb/s |                49.0 gb/s |                49.7 gb/s |
| __u8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`         |                23.0 gb/s |                19.9 gb/s |                17.8 gb/s |
| `nk_dot_u8_v128relaxed`    |                29.3 gb/s |                33.0 gb/s |                35.1 gb/s |
| __i4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`         |               0.990 gb/s |               0.923 gb/s |               0.985 gb/s |
| `nk_dot_i4_v128relaxed`    |                15.2 gb/s |                17.9 gb/s |                19.2 gb/s |
| __u4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`         |               0.992 gb/s |               0.933 gb/s |               0.988 gb/s |
| `nk_dot_u4_v128relaxed`    |                30.2 gb/s |                32.1 gb/s |                33.8 gb/s |
| __u1__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`         |                5.26 gb/s |                5.80 gb/s |                6.48 gb/s |
| `nk_dot_u1_v128relaxed`    |                21.2 gb/s |                47.4 gb/s |                67.3 gb/s |
