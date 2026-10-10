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

| Input Type | Output Type | Description                                      |
| :--------- | :---------- | :----------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output   |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits        |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits        |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits     |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits     |
| `e2m1`     | `f32`       | 4-bit MX format: 2 exponent, 1 mantissa bit      |
| `i8`       | `i32`       | 8-bit signed integers                            |
| `u8`       | `u32`       | 8-bit unsigned integers                          |
| `i4`       | `i32`       | 4-bit signed integers, packed nibble pairs       |
| `u4`       | `u32`       | 4-bit unsigned integers, packed nibble pairs     |
| `u1`       | `u32`       | 1-bit binary packed octets, popcount of AND      |

Complex dot products (both `dot` and `vdot`):

| Input Type | Output Type | Description                                |
| :--------- | :---------- | :----------------------------------------- |
| `f64c`     | `f64c`      | 64-bit complex pairs                       |
| `f32c`     | `f64c`      | 32-bit complex pairs, widened output       |
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

### Algebraic Domain Shifting

`nk_dot_i8_icelake`, `nk_dot_u8_icelake` work around `VPDPBUSD` requiring UInt8 × Int8 operands.
For Int8 × Int8, one operand is XORed with `0x80` to shift to unsigned, and the correction $128 \cdot \sum b_i$ is computed via `VPSADBW`, which runs on port 5 and avoids contention with `DPBUSD` on ports 0-1.
`nk_dot_i4_icelake` extends this to packed nibbles using the identity $(a'-8)(b'-8) = a' b' - 8(a'+b') + 64$ — two `VPDPBUSD` calls handle low and high nibbles separately, with SAD-based correction.
`nk_dot_i8_v128relaxed`, `nk_dot_u8_v128relaxed` face an even tighter constraint: WASM's `i32x4_relaxed_dot_i8x16_i7x16_add` computes Int8 × Int7, so the sign bit of one operand must be masked off entirely.
For Int8 × Int8, the sign bit of $b$ is cleared to produce a 7-bit value, and a windowed correction $-128 \cdot \sum_{b_i < 0} a_i$ is accumulated in Int16 and flushed every 127 iterations to prevent overflow.
For UInt8 × UInt8, $b$ is XORed with `0x80` to shift into signed range, same as Ice Lake, with the correction $128 \cdot \sum a_i$ computed via pairwise widening adds.
`nk_dot_i8_v128`, `nk_dot_u8_v128` are the SIMD128 twins for engines without Relaxed SIMD: both operands widen to Int16 and `i32x4.dot_i16x8_s` multiplies adjacent pairs exactly, so no sign correction is needed and the window may run 32767 iterations before its Int32 lanes are drained.

### Octave Decomposition for E4M3 via VNNI

`nk_dot_e4m3_icelake` splits the 4-bit E4M3 exponent into 2 "octave" bits (top) and 2 "remainder" bits (bottom).
The bottom 5 bits (2 remainder + 3 mantissa) map via `VPERMB` to u8 integers in [0, 120] — identical structure to the E2M3 $\times 16$ LUT.
A subnormal fixup replaces LUT entries for magnitude < 8 with $2 \times \text{mantissa}$ via a second masked `VPERMB`, avoiding `VPADDB` on the VPDPBUSD execution ports.
Sign is computed via `VPTERNLOGD` with immediate 0x14, fusing (a ⊕ b) ∧ ¬`0x7F` in one instruction.
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

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_DIMS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in gb/s as the number of bytes read per second amortized for a large batch of vector pairs.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                   |                      256 |                     1024 |                     4096 |
| :----------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f64c_with_blas` 🧩  |        25.9 gb/s, 25 ulp |        25.3 gb/s, 97 ulp |        20.8 gb/s, 32 ulp |
| `vdot_f64c_with_blas` 🧩 |        25.8 gb/s, 18 ulp |        21.9 gb/s, 17 ulp |        21.5 gb/s, 25 ulp |
| `nk_dot_f64c_serial`     |       1.74 gb/s, 3.9 ulp |       1.89 gb/s, 9.0 ulp |       1.54 gb/s, 2.9 ulp |
| `nk_vdot_f64c_serial`    |       1.74 gb/s, 4.6 ulp |       1.70 gb/s, 1.6 ulp |       1.52 gb/s, 2.2 ulp |
| `nk_dot_f64c_haswell`    |                8.18 gb/s |                16.8 gb/s |                5.43 gb/s |
| `nk_vdot_f64c_haswell`   |                15.3 gb/s |                16.7 gb/s |                5.05 gb/s |
| `nk_dot_f64c_skylake`    |         18.3 gb/s, 0 ulp |         20.9 gb/s, 0 ulp |         6.88 gb/s, 0 ulp |
| `nk_vdot_f64c_skylake`   |         18.8 gb/s, 0 ulp |         20.8 gb/s, 0 ulp |         7.08 gb/s, 0 ulp |
| __f32c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f32c_with_blas` 🧩  |       25.4 gb/s, 8.6 ulp |        21.6 gb/s, 13 ulp |        26.5 gb/s, 19 ulp |
| `vdot_f32c_with_blas` 🧩 |        25.1 gb/s, 11 ulp |        21.7 gb/s, 14 ulp |        26.4 gb/s, 21 ulp |
| `nk_dot_f32c_serial`     |         8.13 gb/s, 0 ulp |         9.54 gb/s, 0 ulp |         5.05 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`    |         7.79 gb/s, 0 ulp |         9.41 gb/s, 0 ulp |         5.30 gb/s, 0 ulp |
| `nk_dot_f32c_haswell`    |         17.5 gb/s, 0 ulp |         16.0 gb/s, 0 ulp |         20.7 gb/s, 0 ulp |
| `nk_vdot_f32c_haswell`   |         17.3 gb/s, 0 ulp |         16.4 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |
| `nk_dot_f32c_skylake`    |         20.4 gb/s, 0 ulp |         24.7 gb/s, 0 ulp |         25.6 gb/s, 0 ulp |
| `nk_vdot_f32c_skylake`   |         20.4 gb/s, 0 ulp |         25.3 gb/s, 0 ulp |         24.4 gb/s, 0 ulp |
| __bf16c__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`    |       3.83 gb/s, 0.1 ulp |       4.75 gb/s, 2.3 ulp |       4.45 gb/s, 7.9 ulp |
| `nk_vdot_bf16c_serial`   |       3.52 gb/s, 0.2 ulp |       4.70 gb/s, 2.1 ulp |      4.51 gb/s, 11.2 ulp |
| `nk_dot_bf16c_haswell`   |       16.1 gb/s, 0.1 ulp |       15.2 gb/s, 1.3 ulp |       17.0 gb/s, 3.4 ulp |
| `nk_vdot_bf16c_haswell`  |       16.2 gb/s, 0.8 ulp |       16.5 gb/s, 2.0 ulp |       16.9 gb/s, 4.5 ulp |
| `nk_dot_bf16c_genoa`     |         22.8 gb/s, 0 ulp |       25.1 gb/s, 1.1 ulp |       24.8 gb/s, 2.8 ulp |
| `nk_vdot_bf16c_genoa`    |       22.7 gb/s, 0.7 ulp |       25.1 gb/s, 1.2 ulp |       25.4 gb/s, 3.3 ulp |
| __f16c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`     |      1.02 gb/s, 14.4 ulp |      1.44 gb/s, 27.3 ulp |      1.32 gb/s, 34.0 ulp |
| `nk_vdot_f16c_serial`    |     0.992 gb/s, 15.0 ulp |      1.44 gb/s, 26.3 ulp |      1.32 gb/s, 34.2 ulp |
| `nk_dot_f16c_haswell`    |      16.1 gb/s, 12.7 ulp |      17.4 gb/s, 22.3 ulp |      17.6 gb/s, 40.1 ulp |
| `nk_vdot_f16c_haswell`   |      16.1 gb/s, 11.1 ulp |      17.5 gb/s, 17.4 ulp |      17.5 gb/s, 29.2 ulp |
| __f64__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f64_with_blas` 🧩   |       25.5 gb/s, 6.9 ulp |       22.4 gb/s, 9.3 ulp |        26.4 gb/s, 20 ulp |
| `nk_dot_f64_serial`      |       2.75 gb/s, 2.2 ulp |       3.51 gb/s, 2.0 ulp |       1.96 gb/s, 3.3 ulp |
| `nk_dot_f64_haswell`     |         17.0 gb/s, 0 ulp |         23.8 gb/s, 0 ulp |         24.1 gb/s, 0 ulp |
| `nk_dot_f64_skylake`     |         20.4 gb/s, 0 ulp |         24.6 gb/s, 0 ulp |         24.9 gb/s, 0 ulp |
| __f32__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dot_f32_with_blas` 🧩   |        28.1 gb/s, 14 ulp |        19.8 gb/s, 14 ulp |        26.4 gb/s, 15 ulp |
| `nk_dot_f32_serial`      |         7.19 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |
| `nk_dot_f32_haswell`     |         22.6 gb/s, 0 ulp |         20.5 gb/s, 0 ulp |         22.1 gb/s, 0 ulp |
| `nk_dot_f32_skylake`     |         28.6 gb/s, 0 ulp |         25.8 gb/s, 0 ulp |         26.0 gb/s, 0 ulp |
| __bf16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`     |         3.75 gb/s, 0 ulp |       5.71 gb/s, 0.5 ulp |       5.61 gb/s, 5.4 ulp |
| `nk_dot_bf16_haswell`    |         35.5 gb/s, 0 ulp |       19.4 gb/s, 0.2 ulp |      18.9 gb/s, 25.3 ulp |
| `nk_dot_bf16_skylake`    |         56.5 gb/s, 0 ulp |       25.1 gb/s, 0.2 ulp |       26.1 gb/s, 2.3 ulp |
| `nk_dot_bf16_genoa`      |         55.3 gb/s, 0 ulp |       25.1 gb/s, 0.2 ulp |       25.9 gb/s, 2.2 ulp |
| __f16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`      |      1.04 gb/s, 11.5 ulp |      1.47 gb/s, 33.7 ulp |      1.46 gb/s, 59.7 ulp |
| `nk_dot_f16_haswell`     |       28.4 gb/s, 7.0 ulp |      19.8 gb/s, 14.0 ulp |      19.2 gb/s, 29.8 ulp |
| `nk_dot_f16_skylake`     |       48.8 gb/s, 6.2 ulp |       24.4 gb/s, 8.6 ulp |      25.9 gb/s, 22.8 ulp |
| __e5m2__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`     |         1.36 gb/s, 0 ulp |         1.90 gb/s, 0 ulp |         1.89 gb/s, 0 ulp |
| `nk_dot_e5m2_haswell`    |         16.1 gb/s, 0 ulp |         13.2 gb/s, 0 ulp |         15.3 gb/s, 0 ulp |
| `nk_dot_e5m2_skylake`    |         19.3 gb/s, 0 ulp |         15.9 gb/s, 0 ulp |         17.3 gb/s, 0 ulp |
| `nk_dot_e5m2_genoa`      |         7.30 gb/s, 0 ulp |         8.55 gb/s, 0 ulp |         8.65 gb/s, 0 ulp |
| __e4m3__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`     |        0.761 gb/s, 0 ulp |        0.783 gb/s, 0 ulp |        0.785 gb/s, 0 ulp |
| `nk_dot_e4m3_haswell`    |         4.71 gb/s, 0 ulp |         4.67 gb/s, 0 ulp |         4.78 gb/s, 0 ulp |
| `nk_dot_e4m3_skylake`    |         7.15 gb/s, 0 ulp |         7.17 gb/s, 0 ulp |         7.21 gb/s, 0 ulp |
| `nk_dot_e4m3_icelake`    |         10.8 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |         12.3 gb/s, 0 ulp |
| __e3m2__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`     |         1.36 gb/s, 0 ulp |         1.86 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |
| `nk_dot_e3m2_haswell`    |         9.53 gb/s, 0 ulp |         6.17 gb/s, 0 ulp |         8.37 gb/s, 0 ulp |
| `nk_dot_e3m2_skylake`    |         18.8 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |         18.1 gb/s, 0 ulp |
| `nk_dot_e3m2_icelake`    |         13.5 gb/s, 0 ulp |         15.3 gb/s, 0 ulp |         19.1 gb/s, 0 ulp |
| __e2m3__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`     |         1.21 gb/s, 0 ulp |         1.88 gb/s, 0 ulp |         1.90 gb/s, 0 ulp |
| `nk_dot_e2m3_haswell`    |         13.8 gb/s, 0 ulp |         8.05 gb/s, 0 ulp |         11.5 gb/s, 0 ulp |
| `nk_dot_e2m3_skylake`    |         30.7 gb/s, 0 ulp |         22.7 gb/s, 0 ulp |         23.9 gb/s, 0 ulp |
| `nk_dot_e2m3_icelake`    |         28.2 gb/s, 0 ulp |         32.5 gb/s, 0 ulp |         25.0 gb/s, 0 ulp |
| `nk_dot_e2m3_alder`      |         15.7 gb/s, 0 ulp |         15.3 gb/s, 0 ulp |         16.8 gb/s, 0 ulp |
| __e2m1__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m1_serial`     |                1.53 gb/s |                1.18 gb/s |                1.18 gb/s |
| `nk_dot_e2m1_haswell`    |                14.4 gb/s |                23.6 gb/s |                14.1 gb/s |
| `nk_dot_e2m1_skylake`    |                21.1 gb/s |                23.2 gb/s |                15.3 gb/s |
| `nk_dot_e2m1_alder`      |                20.8 gb/s |                18.0 gb/s |                14.4 gb/s |
| __i8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`       |                3.13 gb/s |                4.38 gb/s |                4.52 gb/s |
| `nk_dot_i8_haswell`      |                35.6 gb/s |                15.3 gb/s |                23.0 gb/s |
| `nk_dot_i8_skylake`      |                43.7 gb/s |                26.3 gb/s |                25.5 gb/s |
| `nk_dot_i8_icelake`      |                52.2 gb/s |                37.0 gb/s |                25.3 gb/s |
| `nk_dot_i8_alder`        |                29.7 gb/s |                28.5 gb/s |                24.7 gb/s |
| __u8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`       |                2.98 gb/s |                5.24 gb/s |                5.02 gb/s |
| `nk_dot_u8_haswell`      |                36.1 gb/s |                19.9 gb/s |                25.5 gb/s |
| `nk_dot_u8_skylake`      |                41.8 gb/s |                25.8 gb/s |                25.3 gb/s |
| `nk_dot_u8_icelake`      |                43.7 gb/s |                38.4 gb/s |                25.2 gb/s |
| `nk_dot_u8_alder`        |                30.3 gb/s |                31.1 gb/s |                23.7 gb/s |
| __i4__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`       |               0.872 gb/s |                1.29 gb/s |                1.32 gb/s |
| `nk_dot_i4_haswell`      |                5.73 gb/s |                4.51 gb/s |                5.50 gb/s |
| `nk_dot_i4_icelake`      |                15.4 gb/s |                27.8 gb/s |                16.5 gb/s |
| __u4__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`       |                1.30 gb/s |                2.07 gb/s |                2.11 gb/s |
| `nk_dot_u4_haswell`      |                11.4 gb/s |                11.0 gb/s |                9.88 gb/s |
| `nk_dot_u4_icelake`      |                27.1 gb/s |                48.1 gb/s |                23.4 gb/s |
| __u1__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`       |                2.18 gb/s |                3.55 gb/s |                4.16 gb/s |
| `nk_dot_u1_haswell`      |                11.6 gb/s |                23.5 gb/s |                18.4 gb/s |
| `nk_dot_u1_icelake`      |                10.6 gb/s |                50.8 gb/s |                71.7 gb/s |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`       |       1.67 gb/s, 5.3 ulp |       1.54 gb/s, 3.3 ulp |       1.64 gb/s, 2.9 ulp |
| `nk_vdot_f64c_serial`      |       1.69 gb/s, 3.5 ulp |       1.73 gb/s, 5.5 ulp |       1.72 gb/s, 2.2 ulp |
| `nk_dot_f64c_v128relaxed`  |      5.10 gb/s, 37.8 ulp |      4.93 gb/s, 34.9 ulp |       4.79 gb/s, 167 ulp |
| `nk_vdot_f64c_v128relaxed` |      4.92 gb/s, 20.1 ulp |      4.86 gb/s, 51.4 ulp |      4.48 gb/s, 57.2 ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`       |         4.76 gb/s, 0 ulp |         7.23 gb/s, 0 ulp |         6.01 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`      |         4.78 gb/s, 0 ulp |         7.28 gb/s, 0 ulp |         6.38 gb/s, 0 ulp |
| `nk_dot_f32c_v128relaxed`  |         9.18 gb/s, 0 ulp |         10.7 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |
| `nk_vdot_f32c_v128relaxed` |         9.33 gb/s, 0 ulp |         10.8 gb/s, 0 ulp |         9.99 gb/s, 0 ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`      |       2.71 gb/s, 0.1 ulp |       4.34 gb/s, 1.7 ulp |       3.33 gb/s, 7.9 ulp |
| `nk_vdot_bf16c_serial`     |       2.71 gb/s, 0.1 ulp |       4.31 gb/s, 2.9 ulp |      4.13 gb/s, 11.2 ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`       |     0.906 gb/s, 12.6 ulp |      1.48 gb/s, 22.2 ulp |        1.40 gb/s, 34 ulp |
| `nk_vdot_f16c_serial`      |     0.906 gb/s, 14.4 ulp |      1.48 gb/s, 41.8 ulp |      1.44 gb/s, 34.2 ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`        |       2.94 gb/s, 3.1 ulp |       3.30 gb/s, 2.5 ulp |       2.98 gb/s, 3.3 ulp |
| `nk_dot_f64_v128relaxed`   |       6.42 gb/s, 3.2 ulp |       7.44 gb/s, 3.6 ulp |       7.25 gb/s, 3.8 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`        |         5.11 gb/s, 0 ulp |         8.37 gb/s, 0 ulp |         7.93 gb/s, 0 ulp |
| `nk_dot_f32_v128relaxed`   |         4.74 gb/s, 0 ulp |         8.49 gb/s, 0 ulp |         6.95 gb/s, 0 ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`       |         2.94 gb/s, 0 ulp |       4.94 gb/s, 0.5 ulp |       4.49 gb/s, 5.2 ulp |
| `nk_dot_bf16_v128relaxed`  |         10.4 gb/s, 0 ulp |       9.84 gb/s, 0.3 ulp |       9.44 gb/s, 2.4 ulp |
| `nk_dot_bf16_v128`         |                13.4 gb/s |                14.7 gb/s |                18.3 gb/s |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`        |     0.873 gb/s, 13.5 ulp |        1.48 gb/s, 32 ulp |      1.27 gb/s, 59.7 ulp |
| `nk_dot_f16_v128relaxed`   |       1.88 gb/s, 7.0 ulp |      2.91 gb/s, 30.8 ulp |      2.96 gb/s, 65.1 ulp |
| __e5m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`       |        0.697 gb/s, 0 ulp |         1.17 gb/s, 0 ulp |         1.06 gb/s, 0 ulp |
| `nk_dot_e5m2_v128relaxed`  |         1.10 gb/s, 0 ulp |         1.27 gb/s, 0 ulp |         1.28 gb/s, 0 ulp |
| __e4m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`       |        0.302 gb/s, 0 ulp |        0.418 gb/s, 0 ulp |        0.381 gb/s, 0 ulp |
| `nk_dot_e4m3_v128relaxed`  |         1.35 gb/s, 0 ulp |         1.59 gb/s, 0 ulp |         1.62 gb/s, 0 ulp |
| __e3m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`       |         1.15 gb/s, 0 ulp |         1.18 gb/s, 0 ulp |         1.20 gb/s, 0 ulp |
| `nk_dot_e3m2_v128relaxed`  |         3.28 gb/s, 0 ulp |         3.89 gb/s, 0 ulp |         4.38 gb/s, 0 ulp |
| __e2m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`       |         1.16 gb/s, 0 ulp |         1.16 gb/s, 0 ulp |         1.20 gb/s, 0 ulp |
| `nk_dot_e2m3_v128relaxed`  |         6.09 gb/s, 0 ulp |         6.90 gb/s, 0 ulp |         8.05 gb/s, 0 ulp |
| __e2m1__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m1_serial`       |                1.16 gb/s |                1.26 gb/s |                1.26 gb/s |
| `nk_dot_e2m1_v128relaxed`  |                8.71 gb/s |                11.9 gb/s |                9.83 gb/s |
| __i8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`         |                3.38 gb/s |                3.75 gb/s |                3.92 gb/s |
| `nk_dot_i8_v128relaxed`    |                9.35 gb/s |                13.1 gb/s |                15.9 gb/s |
| `nk_dot_i8_v128`           |                10.8 gb/s |                14.2 gb/s |                15.9 gb/s |
| __u8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`         |                3.33 gb/s |                3.69 gb/s |                3.72 gb/s |
| `nk_dot_u8_v128relaxed`    |                7.60 gb/s |                11.7 gb/s |                13.3 gb/s |
| `nk_dot_u8_v128`           |                10.4 gb/s |                15.0 gb/s |                15.6 gb/s |
| __i4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`         |                1.42 gb/s |                1.58 gb/s |                1.61 gb/s |
| `nk_dot_i4_v128relaxed`    |                5.14 gb/s |                9.68 gb/s |                8.56 gb/s |
| __u4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`         |                1.54 gb/s |                1.76 gb/s |                1.79 gb/s |
| `nk_dot_u4_v128relaxed`    |                8.53 gb/s |                16.7 gb/s |                12.1 gb/s |
| __u1__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`         |                1.95 gb/s |                2.51 gb/s |                2.88 gb/s |
| `nk_dot_u1_v128`           |                3.88 gb/s |                17.1 gb/s |                15.2 gb/s |

### Apple M5

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`      |         7.47 gb/s, 5 ulp |         6.80 gb/s, 3 ulp |       6.75 gb/s, 9.7 ulp |
| `nk_vdot_f64c_serial`     |       7.72 gb/s, 4.2 ulp |       7.01 gb/s, 3.3 ulp |       6.87 gb/s, 3.3 ulp |
| `nk_dot_f64c_neon`        |         22.1 gb/s, 0 ulp |         20.1 gb/s, 0 ulp |         19.8 gb/s, 0 ulp |
| `nk_vdot_f64c_neon`       |         22.0 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |         19.5 gb/s, 0 ulp |
| __f32c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`      |         25.9 gb/s, 0 ulp |         22.9 gb/s, 0 ulp |         21.6 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`     |         25.3 gb/s, 0 ulp |         22.4 gb/s, 0 ulp |         21.0 gb/s, 0 ulp |
| `nk_dot_f32c_neon`        |         21.2 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         15.7 gb/s, 0 ulp |
| `nk_vdot_f32c_neon`       |         21.1 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |         15.6 gb/s, 0 ulp |
| __bf16c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`     |       14.5 gb/s, 0.2 ulp |       11.6 gb/s, 2.8 ulp |      11.6 gb/s, 15.8 ulp |
| `nk_vdot_bf16c_serial`    |       14.8 gb/s, 0.2 ulp |       12.0 gb/s, 2.6 ulp |      10.9 gb/s, 11.4 ulp |
| `nk_dot_bf16c_neonbfdot`  |       24.5 gb/s, 0.1 ulp |         17.2 gb/s, 2 ulp |       16.4 gb/s, 8.8 ulp |
| `nk_vdot_bf16c_neonbfdot` |       24.7 gb/s, 0.1 ulp |       17.0 gb/s, 1.8 ulp |       16.1 gb/s, 8.8 ulp |
| __f16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`      |      14.7 gb/s, 20.8 ulp |      12.1 gb/s, 64.1 ulp |      11.5 gb/s, 73.1 ulp |
| `nk_vdot_f16c_serial`     |      14.7 gb/s, 24.8 ulp |      12.1 gb/s, 31.9 ulp |       11.5 gb/s, 137 ulp |
| `nk_dot_f16c_neon`        |       24.3 gb/s, 3.0 ulp |       17.1 gb/s, 6.5 ulp |      15.6 gb/s, 20.5 ulp |
| `nk_vdot_f16c_neon`       |      24.3 gb/s, 34.9 ulp |      17.2 gb/s, 40.7 ulp |      15.8 gb/s, 73.1 ulp |
| `nk_dot_f16c_neonfhm`     |       23.6 gb/s, 3.0 ulp |       15.9 gb/s, 6.5 ulp |      14.8 gb/s, 20.5 ulp |
| `nk_vdot_f16c_neonfhm`    |      23.3 gb/s, 31.4 ulp |      15.8 gb/s, 38.6 ulp |      14.7 gb/s, 67.6 ulp |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`       |       7.55 gb/s, 2.4 ulp |       7.57 gb/s, 175 ulp |       7.53 gb/s, 2.7 ulp |
| `nk_dot_f64_neon`         |         41.2 gb/s, 0 ulp |         39.4 gb/s, 0 ulp |         35.8 gb/s, 0 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`       |         21.7 gb/s, 0 ulp |         14.7 gb/s, 0 ulp |         13.6 gb/s, 0 ulp |
| `nk_dot_f32_neon`         |         43.2 gb/s, 0 ulp |         35.4 gb/s, 0 ulp |         32.4 gb/s, 0 ulp |
| __bf16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`      |         11.5 gb/s, 0 ulp |       8.00 gb/s, 0.9 ulp |         6.85 gb/s, 6 ulp |
| `nk_dot_bf16_neon`        |       36.3 gb/s, 3.7 ulp |       25.3 gb/s, 3.7 ulp |       18.5 gb/s, 3.7 ulp |
| `nk_dot_bf16_neonbfdot`   |         65.9 gb/s, 0 ulp |       56.6 gb/s, 0.6 ulp |       44.5 gb/s, 4.5 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`       |        11.2 gb/s, 19 ulp |      7.76 gb/s, 31.1 ulp |      6.62 gb/s, 57.8 ulp |
| `nk_dot_f16_neon`         |      33.2 gb/s, 33.4 ulp |      24.0 gb/s, 37.4 ulp |      19.8 gb/s, 23.1 ulp |
| `nk_dot_f16_neonfhm`      |      45.4 gb/s, 14.9 ulp |      25.6 gb/s, 26.7 ulp |      17.5 gb/s, 39.9 ulp |
| __e5m2__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`      |         3.54 gb/s, 0 ulp |         3.18 gb/s, 0 ulp |         3.18 gb/s, 0 ulp |
| `nk_dot_e5m2_neon`        |         17.7 gb/s, 0 ulp |         12.3 gb/s, 0 ulp |         9.78 gb/s, 0 ulp |
| `nk_dot_e5m2_neonfhm`     |         23.8 gb/s, 0 ulp |         14.2 gb/s, 0 ulp |         8.89 gb/s, 0 ulp |
| `nk_dot_e5m2_neonbfdot`   |         3.40 gb/s, 0 ulp |         3.56 gb/s, 0 ulp |         3.43 gb/s, 0 ulp |
| __e4m3__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`      |         1.62 gb/s, 0 ulp |         1.60 gb/s, 0 ulp |         1.59 gb/s, 0 ulp |
| `nk_dot_e4m3_neon`        |         4.14 gb/s, 0 ulp |         4.20 gb/s, 0 ulp |         4.26 gb/s, 0 ulp |
| `nk_dot_e4m3_neonfhm`     |         9.41 gb/s, 0 ulp |         7.93 gb/s, 0 ulp |         7.41 gb/s, 0 ulp |
| `nk_dot_e4m3_neonbfdot`   |         3.34 gb/s, 0 ulp |         3.43 gb/s, 0 ulp |         3.39 gb/s, 0 ulp |
| __e3m2__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`      |         2.34 gb/s, 0 ulp |         2.17 gb/s, 0 ulp |         2.09 gb/s, 0 ulp |
| `nk_dot_e3m2_neonsdot`    |         19.1 gb/s, 0 ulp |         19.3 gb/s, 0 ulp |         18.7 gb/s, 0 ulp |
| __e2m3__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`      |         2.37 gb/s, 0 ulp |         2.11 gb/s, 0 ulp |         2.13 gb/s, 0 ulp |
| `nk_dot_e2m3_neonsdot`    |         44.1 gb/s, 0 ulp |         44.2 gb/s, 0 ulp |         40.4 gb/s, 0 ulp |
| __i8__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`        |                 107 gb/s |                95.0 gb/s |                86.0 gb/s |
| `nk_dot_i8_neonsdot`      |                86.4 gb/s |                81.4 gb/s |                55.8 gb/s |
| __u8__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`        |                 102 gb/s |                92.4 gb/s |                88.4 gb/s |
| `nk_dot_u8_neonsdot`      |                86.1 gb/s |                80.7 gb/s |                55.4 gb/s |
| __i4__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`        |                21.4 gb/s |                22.7 gb/s |                22.5 gb/s |
| `nk_dot_i4_neonsdot`      |                54.2 gb/s |                41.6 gb/s |                28.3 gb/s |
| __u4__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`        |                23.6 gb/s |                25.3 gb/s |                25.1 gb/s |
| `nk_dot_u4_neonsdot`      |                62.7 gb/s |                44.1 gb/s |                27.4 gb/s |
| __u1__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`        |                6.53 gb/s |                7.11 gb/s |                6.70 gb/s |
| `nk_dot_u1_neon`          |                31.0 gb/s |                60.2 gb/s |                82.0 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`       |       3.88 gb/s, 3.8 ulp |       5.62 gb/s, 3.9 ulp |       5.93 gb/s, 3.2 ulp |
| `nk_vdot_f64c_serial`      |       5.59 gb/s, 3.8 ulp |       6.10 gb/s, 3.4 ulp |      6.36 gb/s, 15.1 ulp |
| `nk_dot_f64c_v128relaxed`  |        43.5 gb/s, 26 ulp |        35.6 gb/s, 42 ulp |        37.7 gb/s, 88 ulp |
| `nk_vdot_f64c_v128relaxed` |      42.9 gb/s, 22.8 ulp |      37.1 gb/s, 37.3 ulp |      37.2 gb/s, 43.6 ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`       |         19.5 gb/s, 0 ulp |         19.8 gb/s, 0 ulp |         21.0 gb/s, 0 ulp |
| `nk_vdot_f32c_serial`      |         18.5 gb/s, 0 ulp |         20.0 gb/s, 0 ulp |         20.9 gb/s, 0 ulp |
| `nk_dot_f32c_v128relaxed`  |         20.9 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |         18.4 gb/s, 0 ulp |
| `nk_vdot_f32c_v128relaxed` |         20.4 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |         18.5 gb/s, 0 ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`      |       9.59 gb/s, 0.1 ulp |       10.8 gb/s, 2.5 ulp |        10.8 gb/s, 10 ulp |
| `nk_vdot_bf16c_serial`     |       9.97 gb/s, 0.2 ulp |       10.9 gb/s, 2.1 ulp |      10.8 gb/s, 11.4 ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`       |        3.50 gb/s, 13 ulp |        3.57 gb/s, 20 ulp |        3.59 gb/s, 90 ulp |
| `nk_vdot_f16c_serial`      |      3.55 gb/s, 13.9 ulp |      3.62 gb/s, 35.5 ulp |      3.59 gb/s, 42.4 ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`        |       6.76 gb/s, 2.4 ulp |       6.94 gb/s, 2.6 ulp |       7.41 gb/s, 2.2 ulp |
| `nk_dot_f64_v128relaxed`   |       36.0 gb/s, 2.6 ulp |       39.1 gb/s, 3.2 ulp |       40.9 gb/s, 2.6 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`        |        17.7 gb/s, 16 ulp |        13.6 gb/s, 69 ulp |       13.0 gb/s, 104 ulp |
| `nk_dot_f32_v128relaxed`   |         19.0 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |         17.4 gb/s, 0 ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`       |         8.88 gb/s, 0 ulp |       6.91 gb/s, 0.6 ulp |       6.71 gb/s, 5.9 ulp |
| `nk_dot_bf16_v128relaxed`  |         39.0 gb/s, 0 ulp |       26.4 gb/s, 0.4 ulp |       20.0 gb/s, 3.7 ulp |
| `nk_dot_bf16_v128`         |                        … |                        … |                        … |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`        |        3.08 gb/s, 16 ulp |        3.38 gb/s, 26 ulp |        3.41 gb/s, 53 ulp |
| `nk_dot_f16_v128relaxed`   |       10.6 gb/s, 9.0 ulp |        10.4 gb/s, 23 ulp |        11.2 gb/s, 39 ulp |
| __e5m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`       |         2.81 gb/s, 0 ulp |         2.75 gb/s, 0 ulp |         2.94 gb/s, 0 ulp |
| `nk_dot_e5m2_v128relaxed`  |         3.23 gb/s, 0 ulp |         3.21 gb/s, 0 ulp |         3.24 gb/s, 0 ulp |
| __e4m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`       |        0.911 gb/s, 0 ulp |        0.832 gb/s, 0 ulp |        0.872 gb/s, 0 ulp |
| `nk_dot_e4m3_v128relaxed`  |         2.59 gb/s, 0 ulp |         2.56 gb/s, 0 ulp |         2.59 gb/s, 0 ulp |
| __e3m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`       |         2.92 gb/s, 0 ulp |         2.75 gb/s, 0 ulp |         2.94 gb/s, 0 ulp |
| `nk_dot_e3m2_v128relaxed`  |         11.3 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |         11.7 gb/s, 0 ulp |
| __e2m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`       |         2.78 gb/s, 0 ulp |         2.79 gb/s, 0 ulp |         2.95 gb/s, 0 ulp |
| `nk_dot_e2m3_v128relaxed`  |         19.0 gb/s, 0 ulp |         19.2 gb/s, 0 ulp |         20.2 gb/s, 0 ulp |
| __i8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`         |                20.7 gb/s |                18.3 gb/s |                16.6 gb/s |
| `nk_dot_i8_v128relaxed`    |                39.1 gb/s |                45.6 gb/s |                46.3 gb/s |
| `nk_dot_i8_v128`           |                        … |                        … |                        … |
| __u8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`         |                21.4 gb/s |                18.5 gb/s |                16.6 gb/s |
| `nk_dot_u8_v128relaxed`    |                27.3 gb/s |                30.7 gb/s |                32.7 gb/s |
| `nk_dot_u8_v128`           |                        … |                        … |                        … |
| __i4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`         |               0.922 gb/s |               0.860 gb/s |               0.917 gb/s |
| `nk_dot_i4_v128relaxed`    |                14.2 gb/s |                16.7 gb/s |                17.9 gb/s |
| __u4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`         |               0.924 gb/s |               0.869 gb/s |               0.920 gb/s |
| `nk_dot_u4_v128relaxed`    |                28.1 gb/s |                29.9 gb/s |                31.5 gb/s |
| __u1__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`         |                4.90 gb/s |                5.40 gb/s |                6.03 gb/s |
| `nk_dot_u1_v128`           |                19.7 gb/s |                44.1 gb/s |                62.7 gb/s |
