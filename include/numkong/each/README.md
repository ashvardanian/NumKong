# Element-Wise Arithmetic in NumKong

NumKong implements element-wise vector arithmetic: addition, scaling, blending, and fused multiply-add across all supported numeric types.
Each operation reads one to three input vectors and writes one output vector of the same length, with scalar coefficients $\alpha$ and $\beta$ controlling linear combinations.
Mixed-precision workflows use narrower input types (Float16, BFloat16, Float8) with Float32 intermediate computation and narrowed output.

Sum (addition):

$$
\text{result}_i = a_i + b_i
$$

Scale:

$$
\text{result}_i = \alpha \cdot a_i + \beta
$$

Blend:

$$
\text{result}_i = \alpha \cdot a_i + \beta \cdot b_i
$$

Fused multiply-add:

$$
\text{result}_i = \alpha \cdot a_i \cdot b_i + \beta \cdot c_i
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def fma(a: np.ndarray, b: np.ndarray, c: np.ndarray,
        alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    return alpha * a * b + beta * c

def scale(a: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return alpha * a + beta

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def blend(a: np.ndarray, b: np.ndarray,
          alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    return alpha * a + beta * b
```

## Input & Output Types

Real and integer element-wise operations:

| Input Type | Output Type | Description                               |
| :--------- | :---------- | :---------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision          |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision          |
| `f16`      | `f16`       | 16-bit IEEE 754 half precision            |
| `bf16`     | `bf16`      | 16-bit brain float                        |
| `e4m3`     | `e4m3`      | 8-bit Float8: 4 exponent, 3 mantissa bits |
| `e5m2`     | `e5m2`      | 8-bit Float8: 5 exponent, 2 mantissa bits |
| `e3m2`     | `e3m2`      | 6-bit Float6: 3 exponent, 2 mantissa bits |
| `e2m3`     | `e2m3`      | 6-bit Float6: 2 exponent, 3 mantissa bits |
| `i8`       | `i8`        | 8-bit signed integers, saturating         |
| `u8`       | `u8`        | 8-bit unsigned integers, saturating       |
| `i16`      | `i16`       | 16-bit signed integers                    |
| `u16`      | `u16`       | 16-bit unsigned integers                  |
| `i32`      | `i32`       | 32-bit signed integers                    |
| `u32`      | `u32`       | 32-bit unsigned integers                  |
| `i64`      | `i64`       | 64-bit signed integers                    |
| `u64`      | `u64`       | 64-bit unsigned integers                  |

Complex element-wise operations:

| Input Type | Output Type | Description          |
| :--------- | :---------- | :------------------- |
| `f64c`     | `f64c`      | 64-bit complex pairs |
| `f32c`     | `f32c`      | 32-bit complex pairs |

## Optimizations

### Widening-Narrowing Pipeline for Sub-32-bit Types

`nk_each_fma_f16_haswell`, `nk_each_blend_bf16_neonbfdot`, `nk_each_scale_e4m3_haswell` widen inputs to Float32 before arithmetic, then narrow the result back to the original type.
The widen-compute-narrow pipeline costs 2 extra conversion instructions per element but guarantees Float32-precision intermediate results — critical for FMA where naive Float16 multiplication would lose 5+ bits of mantissa.
Haswell processes 8 Float16 elements per cycle: `VCVTPH2PS` (widen) → `VFMADD231PS` (FMA) → `VCVTPS2PH` (narrow), fully pipelined across 3 execution ports.

### Saturating Integer Arithmetic

`nk_each_sum_i8_haswell`, `nk_each_sum_u8_neon` use saturating addition — clamping to type bounds instead of wrapping on overflow.
Haswell uses `VPADDSB` / `VPADDUSB` for signed/unsigned 8-bit saturation in a single instruction (32 elements per cycle at YMM width).
Serial fallback implements saturation via branch-free min/max: `result = min(max(a + b, TYPE_MIN), TYPE_MAX)` with overflow detection through sign-bit comparison.

### Complex Number Layout

`nk_each_fma_f32c_serial`, `nk_each_blend_f64c_serial` operate on interleaved real/imaginary pairs: `[re0, im0, re1, im1, ...]`.
Addition and scaling treat complex vectors as 2N-length real vectors — no special handling needed.
FMA requires cross-lane operations for the imaginary part: `re(a*b) = re(a)*re(b) - im(a)*im(b)`, implemented via `VFMADDSUB231PS` which alternates add/subtract across even/odd lanes.

Fused SwiGLU and grouped RMSNorm support F32, F16, BF16, and E4M3 on serial and NEON backends.
The downcasting `nk_each_rmscast_<type>` reverses a dot product of `<type>`: it reads the F32 of BF16, F16, E4M3, E5M2, E2M3 and E3M2 dots, the F64 of F32 dots, or the I32 and U32 of I8 and U8 dots, and normalizes it into `<type>`.
Integer outputs round to nearest even and saturate, with γ carrying the quantization scale.
Every type has serial, CUDA and ROCm kernels, Haswell, Skylake and NEON follow their RMSNorm types, Ampere narrows BF16 and Ada narrows E4M3 and E5M2.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_BATCH_PER_CORE` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes read per second.

### Intel Sapphire Rapids

#### Native

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `sum_f64_with_blas` 🧩        |                11.3 gb/s |                8.04 gb/s |                6.36 gb/s |
| `each_blend_f64_with_blas` 🧩 |                10.5 gb/s |                8.20 gb/s |                6.25 gb/s |
| `nk_each_sum_f64_serial`      |         15.6 gb/s, 0 ulp |         16.0 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |
| `nk_each_sum_f64_haswell`     |         13.7 gb/s, 0 ulp |         9.50 gb/s, 0 ulp |         7.44 gb/s, 0 ulp |
| `nk_each_sum_f64_skylake`     |         15.3 gb/s, 0 ulp |         15.6 gb/s, 0 ulp |         8.00 gb/s, 0 ulp |
| `nk_each_scale_f64_serial`    |         10.6 gb/s, 0 ulp |         11.6 gb/s, 0 ulp |         7.44 gb/s, 0 ulp |
| `nk_each_scale_f64_haswell`   |         8.95 gb/s, 0 ulp |         8.59 gb/s, 0 ulp |         4.73 gb/s, 0 ulp |
| `nk_each_scale_f64_skylake`   |         10.4 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |         5.87 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`    |       15.3 gb/s, 1.4 ulp |       15.3 gb/s, 1.1 ulp |       11.0 gb/s, 1.1 ulp |
| `nk_each_blend_f64_haswell`   |       12.5 gb/s, 1.5 ulp |       10.4 gb/s, 1.5 ulp |       7.31 gb/s, 1.1 ulp |
| `nk_each_blend_f64_skylake`   |       15.4 gb/s, 1.7 ulp |       14.8 gb/s, 1.5 ulp |       7.96 gb/s, 1.1 ulp |
| `nk_each_fma_f64_serial`      |       18.7 gb/s, 1.5 ulp |       19.4 gb/s, 1.5 ulp |       10.9 gb/s, 1.3 ulp |
| `nk_each_fma_f64_haswell`     |       15.6 gb/s, 1.5 ulp |       10.3 gb/s, 1.5 ulp |       8.77 gb/s, 2.8 ulp |
| `nk_each_fma_f64_skylake`     |       18.4 gb/s, 1.4 ulp |       19.0 gb/s, 1.5 ulp |       10.8 gb/s, 2.7 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `sum_f32_with_blas` 🧩        |                10.8 gb/s |                11.9 gb/s |                7.05 gb/s |
| `each_blend_f32_with_blas` 🧩 |                9.59 gb/s |                8.94 gb/s |                6.23 gb/s |
| `nk_each_sum_f32_serial`      |         15.7 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |         14.6 gb/s, 0 ulp |
| `nk_each_sum_f32_haswell`     |         13.6 gb/s, 0 ulp |         13.3 gb/s, 0 ulp |         7.91 gb/s, 0 ulp |
| `nk_each_sum_f32_skylake`     |         15.6 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |         15.4 gb/s, 0 ulp |
| `nk_each_scale_f32_serial`    |         10.1 gb/s, 0 ulp |         12.1 gb/s, 0 ulp |         11.3 gb/s, 0 ulp |
| `nk_each_scale_f32_haswell`   |         8.62 gb/s, 0 ulp |         9.50 gb/s, 0 ulp |         5.79 gb/s, 0 ulp |
| `nk_each_scale_f32_skylake`   |         11.5 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |         12.0 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`    |       15.1 gb/s, 351 ulp |       17.4 gb/s, 2.0 ulp |       16.2 gb/s, 1.4 ulp |
| `nk_each_blend_f32_haswell`   |       13.5 gb/s, 2.3 ulp |       13.2 gb/s, 2.1 ulp |       7.60 gb/s, 1.3 ulp |
| `nk_each_blend_f32_skylake`   |       14.6 gb/s, 1.9 ulp |       16.0 gb/s, 1.8 ulp |       14.8 gb/s, 1.3 ulp |
| `nk_each_fma_f32_serial`      |       18.7 gb/s, 1.4 ulp |       17.4 gb/s, 2.1 ulp |       17.7 gb/s, 1.6 ulp |
| `nk_each_fma_f32_haswell`     |       17.1 gb/s, 1.4 ulp |       14.2 gb/s, 1.8 ulp |       8.87 gb/s, 1.5 ulp |
| `nk_each_fma_f32_skylake`     |       19.4 gb/s, 1.4 ulp |       18.0 gb/s, 1.7 ulp |       15.4 gb/s, 1.5 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`     |        0.163 gb/s, 0 ulp |        0.166 gb/s, 0 ulp |        0.161 gb/s, 0 ulp |
| `nk_each_sum_bf16_haswell`    |         7.42 gb/s, 0 ulp |         8.69 gb/s, 0 ulp |         9.59 gb/s, 0 ulp |
| `nk_each_sum_bf16_skylake`    |         10.6 gb/s, 0 ulp |         10.4 gb/s, 0 ulp |         13.1 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`   |        0.119 gb/s, 0 ulp |        0.111 gb/s, 0 ulp |        0.123 gb/s, 0 ulp |
| `nk_each_scale_bf16_haswell`  |         5.66 gb/s, 0 ulp |         6.10 gb/s, 0 ulp |         6.44 gb/s, 0 ulp |
| `nk_each_scale_bf16_skylake`  |         6.92 gb/s, 0 ulp |         7.49 gb/s, 0 ulp |         7.87 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`   |        0.197 gb/s, 0 ulp |        0.190 gb/s, 0 ulp |        0.209 gb/s, 0 ulp |
| `nk_each_blend_bf16_haswell`  |       7.99 gb/s, 2.2 ulp |       8.79 gb/s, 1.5 ulp |       9.50 gb/s, 1.5 ulp |
| `nk_each_blend_bf16_skylake`  |       9.59 gb/s, 2.3 ulp |       11.1 gb/s, 1.3 ulp |       12.5 gb/s, 1.5 ulp |
| `nk_each_fma_bf16_serial`     |        0.246 gb/s, 0 ulp |        0.242 gb/s, 0 ulp |        0.238 gb/s, 0 ulp |
| `nk_each_fma_bf16_haswell`    |       10.2 gb/s, 1.5 ulp |       9.59 gb/s, 0.9 ulp |       10.6 gb/s, 1.0 ulp |
| `nk_each_fma_bf16_skylake`    |       13.1 gb/s, 1.2 ulp |       12.1 gb/s, 0.7 ulp |       14.7 gb/s, 1.1 ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`      |         31.4 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |
| `nk_each_sum_f16_haswell`     |         13.4 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |         9.16 gb/s, 0 ulp |
| `nk_each_sum_f16_sapphire`    |         36.5 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`    |        0.394 gb/s, 0 ulp |        0.263 gb/s, 0 ulp |        0.381 gb/s, 0 ulp |
| `nk_each_scale_f16_haswell`   |         8.31 gb/s, 0 ulp |         8.00 gb/s, 0 ulp |         7.59 gb/s, 0 ulp |
| `nk_each_scale_f16_skylake`   |         15.8 gb/s, 0 ulp |         9.97 gb/s, 0 ulp |         11.3 gb/s, 0 ulp |
| `nk_each_blend_f16_serial`    |      0.716 gb/s, 1.3 ulp |      0.623 gb/s, 1.6 ulp |      0.738 gb/s, 1.5 ulp |
| `nk_each_blend_f16_haswell`   |       12.6 gb/s, 1.2 ulp |       10.2 gb/s, 1.4 ulp |       10.9 gb/s, 1.5 ulp |
| `nk_each_blend_f16_skylake`   |       15.7 gb/s, 1.3 ulp |       12.9 gb/s, 1.4 ulp |       13.2 gb/s, 1.1 ulp |
| `nk_each_fma_f16_serial`      |      0.899 gb/s, 1.0 ulp |      0.733 gb/s, 1.1 ulp |      0.887 gb/s, 1.2 ulp |
| `nk_each_fma_f16_haswell`     |       14.2 gb/s, 1.4 ulp |       12.7 gb/s, 1.0 ulp |       14.6 gb/s, 1.1 ulp |
| `nk_each_fma_f16_skylake`     |       15.2 gb/s, 1.3 ulp |       15.1 gb/s, 1.3 ulp |       14.2 gb/s, 1.1 ulp |
| __e4m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`     |       0.0900 gb/s, 0 ulp |       0.0878 gb/s, 0 ulp |       0.0881 gb/s, 0 ulp |
| `nk_each_sum_e4m3_haswell`    |        0.834 gb/s, 0 ulp |        0.719 gb/s, 0 ulp |        0.767 gb/s, 0 ulp |
| `nk_each_sum_e4m3_skylake`    |         1.38 gb/s, 0 ulp |         1.25 gb/s, 0 ulp |         1.30 gb/s, 0 ulp |
| `nk_each_sum_e4m3_sapphire`   |         1.97 gb/s, 0 ulp |         1.76 gb/s, 0 ulp |         1.98 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`   |       0.0512 gb/s, 0 ulp |       0.0506 gb/s, 0 ulp |       0.0531 gb/s, 0 ulp |
| `nk_each_scale_e4m3_haswell`  |        0.461 gb/s, 0 ulp |        0.495 gb/s, 0 ulp |        0.503 gb/s, 0 ulp |
| `nk_each_scale_e4m3_skylake`  |        0.978 gb/s, 0 ulp |        0.950 gb/s, 0 ulp |         1.02 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`   |       0.0828 gb/s, 0 ulp |       0.0863 gb/s, 0 ulp |       0.0816 gb/s, 0 ulp |
| `nk_each_blend_e4m3_haswell`  |      0.752 gb/s, 0.6 ulp |        0.704 gb/s, 0 ulp |        0.735 gb/s, 0 ulp |
| `nk_each_blend_e4m3_skylake`  |         1.40 gb/s, 0 ulp |         1.34 gb/s, 0 ulp |         1.38 gb/s, 0 ulp |
| `nk_each_fma_e4m3_serial`     |        0.112 gb/s, 0 ulp |      0.110 gb/s, 0.9 ulp |        0.107 gb/s, 0 ulp |
| `nk_each_fma_e4m3_haswell`    |        0.921 gb/s, 0 ulp |        0.847 gb/s, 0 ulp |      0.901 gb/s, 0.5 ulp |
| `nk_each_fma_e4m3_skylake`    |         1.76 gb/s, 0 ulp |         1.62 gb/s, 0 ulp |         1.68 gb/s, 0 ulp |
| __e5m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`     |        0.107 gb/s, 0 ulp |        0.106 gb/s, 0 ulp |        0.102 gb/s, 0 ulp |
| `nk_each_sum_e5m2_haswell`    |        0.969 gb/s, 0 ulp |        0.919 gb/s, 0 ulp |        0.905 gb/s, 0 ulp |
| `nk_each_sum_e5m2_skylake`    |         1.59 gb/s, 0 ulp |         1.64 gb/s, 0 ulp |         1.67 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`   |       0.0552 gb/s, 0 ulp |       0.0587 gb/s, 0 ulp |       0.0587 gb/s, 0 ulp |
| `nk_each_scale_e5m2_haswell`  |        0.560 gb/s, 0 ulp |        0.569 gb/s, 0 ulp |        0.548 gb/s, 0 ulp |
| `nk_each_scale_e5m2_skylake`  |         1.03 gb/s, 0 ulp |         1.03 gb/s, 0 ulp |         1.04 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`   |        0.101 gb/s, 0 ulp |        0.105 gb/s, 0 ulp |       0.106 gb/s, 50 ulp |
| `nk_each_blend_e5m2_haswell`  |        0.930 gb/s, 0 ulp |        0.834 gb/s, 0 ulp |        0.886 gb/s, 0 ulp |
| `nk_each_blend_e5m2_skylake`  |         1.65 gb/s, 0 ulp |         1.54 gb/s, 0 ulp |         1.60 gb/s, 0 ulp |
| `nk_each_fma_e5m2_serial`     |      0.144 gb/s, 5.1 ulp |        0.136 gb/s, 0 ulp |        0.139 gb/s, 0 ulp |
| `nk_each_fma_e5m2_haswell`    |         1.15 gb/s, 0 ulp |         1.11 gb/s, 0 ulp |         1.16 gb/s, 0 ulp |
| `nk_each_fma_e5m2_skylake`    |         2.20 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |         1.97 gb/s, 0 ulp |
| __e2m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`     |        0.102 gb/s, 0 ulp |       0.0978 gb/s, 0 ulp |        0.102 gb/s, 0 ulp |
| `nk_each_scale_e2m3_serial`   |       0.0466 gb/s, 0 ulp |       0.0441 gb/s, 0 ulp |       0.0461 gb/s, 0 ulp |
| `nk_each_blend_e2m3_serial`   |       0.0805 gb/s, 0 ulp |       0.0827 gb/s, 0 ulp |       0.0846 gb/s, 0 ulp |
| `nk_each_fma_e2m3_serial`     |        0.124 gb/s, 0 ulp |        0.119 gb/s, 0 ulp |        0.119 gb/s, 0 ulp |
| __e3m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`     |        0.110 gb/s, 0 ulp |        0.107 gb/s, 0 ulp |       0.0987 gb/s, 0 ulp |
| `nk_each_scale_e3m2_serial`   |       0.0535 gb/s, 0 ulp |       0.0517 gb/s, 0 ulp |       0.0510 gb/s, 0 ulp |
| `nk_each_blend_e3m2_serial`   |        0.102 gb/s, 0 ulp |       0.0978 gb/s, 0 ulp |       0.0890 gb/s, 0 ulp |
| `nk_each_fma_e3m2_serial`     |        0.141 gb/s, 0 ulp |        0.139 gb/s, 0 ulp |        0.129 gb/s, 0 ulp |
| __i8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`       |                20.2 gb/s |                14.8 gb/s |                11.6 gb/s |
| `nk_each_sum_i8_haswell`      |                35.1 gb/s |                14.6 gb/s |                15.4 gb/s |
| `nk_each_sum_i8_icelake`      |                44.0 gb/s |                16.5 gb/s |                16.5 gb/s |
| `nk_each_scale_i8_serial`     |                2.18 gb/s |                1.40 gb/s |                1.58 gb/s |
| `nk_each_scale_i8_haswell`    |                3.64 gb/s |                3.66 gb/s |                3.39 gb/s |
| `nk_each_scale_i8_skylake`    |                6.28 gb/s |                6.24 gb/s |                6.45 gb/s |
| `nk_each_blend_i8_serial`     |                3.41 gb/s |                2.08 gb/s |                2.42 gb/s |
| `nk_each_blend_i8_haswell`    |                5.54 gb/s |                5.00 gb/s |                5.93 gb/s |
| `nk_each_fma_i8_serial`       |                4.18 gb/s |                2.45 gb/s |                2.78 gb/s |
| `nk_each_fma_i8_haswell`      |                6.85 gb/s |                6.37 gb/s |                6.66 gb/s |
| `nk_each_fma_i8_skylake`      |                10.4 gb/s |                8.80 gb/s |                9.41 gb/s |
| __u8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`       |                15.8 gb/s |                12.8 gb/s |                11.6 gb/s |
| `nk_each_sum_u8_haswell`      |                39.7 gb/s |                14.6 gb/s |                14.3 gb/s |
| `nk_each_sum_u8_icelake`      |                42.7 gb/s |                16.1 gb/s |                16.2 gb/s |
| `nk_each_scale_u8_serial`     |                1.97 gb/s |                1.89 gb/s |                1.73 gb/s |
| `nk_each_scale_u8_haswell`    |                3.64 gb/s |                3.62 gb/s |                3.98 gb/s |
| `nk_each_scale_u8_skylake`    |                6.45 gb/s |                5.58 gb/s |                6.24 gb/s |
| `nk_each_blend_u8_serial`     |                3.01 gb/s |                2.44 gb/s |                3.19 gb/s |
| `nk_each_blend_u8_haswell`    |                4.54 gb/s |                4.75 gb/s |                5.22 gb/s |
| `nk_each_fma_u8_serial`       |                2.97 gb/s |                3.65 gb/s |                4.23 gb/s |
| `nk_each_fma_u8_haswell`      |                6.50 gb/s |                5.86 gb/s |                7.10 gb/s |
| `nk_each_fma_u8_skylake`      |                9.00 gb/s |                8.58 gb/s |                9.59 gb/s |
| __i16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`      |                11.2 gb/s |                10.8 gb/s |                13.2 gb/s |
| `nk_each_sum_i16_haswell`     |                24.3 gb/s |                15.1 gb/s |                15.6 gb/s |
| `nk_each_sum_i16_icelake`     |                36.6 gb/s |                17.0 gb/s |                16.8 gb/s |
| `nk_each_scale_i16_serial`    |                2.97 gb/s |                3.96 gb/s |                3.51 gb/s |
| `nk_each_scale_i16_haswell`   |                6.58 gb/s |                6.91 gb/s |                7.16 gb/s |
| `nk_each_scale_i16_skylake`   |                11.9 gb/s |                8.33 gb/s |                9.78 gb/s |
| `nk_each_blend_i16_serial`    |                5.59 gb/s |                5.63 gb/s |                5.93 gb/s |
| `nk_each_fma_i16_serial`      |                7.80 gb/s |                7.12 gb/s |                6.58 gb/s |
| `nk_each_fma_i16_haswell`     |                9.97 gb/s |                11.6 gb/s |                12.5 gb/s |
| `nk_each_fma_i16_skylake`     |                17.0 gb/s |                14.6 gb/s |                17.7 gb/s |
| __u16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`      |                11.4 gb/s |                11.0 gb/s |                12.8 gb/s |
| `nk_each_sum_u16_haswell`     |                22.4 gb/s |                13.4 gb/s |                16.5 gb/s |
| `nk_each_sum_u16_icelake`     |                37.0 gb/s |                17.0 gb/s |                17.1 gb/s |
| `nk_each_scale_u16_serial`    |                4.14 gb/s |                4.92 gb/s |                5.10 gb/s |
| `nk_each_scale_u16_haswell`   |                7.28 gb/s |                6.71 gb/s |                7.13 gb/s |
| `nk_each_scale_u16_skylake`   |                14.4 gb/s |                10.8 gb/s |                8.42 gb/s |
| `nk_each_blend_u16_serial`    |                7.30 gb/s |                6.59 gb/s |                8.52 gb/s |
| `nk_each_fma_u16_serial`      |                8.39 gb/s |                7.63 gb/s |                8.73 gb/s |
| `nk_each_fma_u16_haswell`     |                10.4 gb/s |                11.4 gb/s |                12.9 gb/s |
| `nk_each_fma_u16_skylake`     |                21.0 gb/s |                19.4 gb/s |                16.3 gb/s |
| __i32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`      |                11.7 gb/s |                11.8 gb/s |                9.41 gb/s |
| `nk_each_sum_i32_haswell`     |                13.6 gb/s |                14.2 gb/s |                12.9 gb/s |
| `nk_each_sum_i32_icelake`     |                15.9 gb/s |                17.2 gb/s |                14.3 gb/s |
| `nk_each_scale_i32_serial`    |                2.83 gb/s |                3.05 gb/s |                3.20 gb/s |
| `nk_each_scale_i32_haswell`   |                8.33 gb/s |                8.12 gb/s |                7.53 gb/s |
| `nk_each_scale_i32_skylake`   |                10.3 gb/s |                11.7 gb/s |                10.5 gb/s |
| `nk_each_blend_i32_serial`    |                4.65 gb/s |                5.58 gb/s |                4.71 gb/s |
| `nk_each_fma_i32_serial`      |                5.90 gb/s |                5.94 gb/s |                5.96 gb/s |
| `nk_each_fma_i32_haswell`     |                12.6 gb/s |                15.3 gb/s |                10.4 gb/s |
| `nk_each_fma_i32_skylake`     |                18.7 gb/s |                19.8 gb/s |                14.9 gb/s |
| __u32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`      |                12.7 gb/s |                10.8 gb/s |                9.26 gb/s |
| `nk_each_sum_u32_haswell`     |                14.9 gb/s |                16.5 gb/s |                11.8 gb/s |
| `nk_each_sum_u32_icelake`     |                16.2 gb/s |                17.7 gb/s |                13.8 gb/s |
| `nk_each_scale_u32_serial`    |                1.98 gb/s |                3.00 gb/s |                2.55 gb/s |
| `nk_each_scale_u32_haswell`   |                7.64 gb/s |                8.75 gb/s |                8.55 gb/s |
| `nk_each_scale_u32_skylake`   |                9.87 gb/s |                12.0 gb/s |                10.3 gb/s |
| `nk_each_blend_u32_serial`    |                3.50 gb/s |                4.86 gb/s |                5.38 gb/s |
| `nk_each_fma_u32_serial`      |                4.45 gb/s |                5.24 gb/s |                7.85 gb/s |
| `nk_each_fma_u32_haswell`     |                12.8 gb/s |                15.0 gb/s |                11.3 gb/s |
| `nk_each_fma_u32_skylake`     |                18.8 gb/s |                19.7 gb/s |                14.2 gb/s |
| __i64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`      |                14.8 gb/s |                14.7 gb/s |                9.59 gb/s |
| `nk_each_sum_i64_icelake`     |                16.5 gb/s |                17.9 gb/s |                10.3 gb/s |
| `nk_each_scale_i64_serial`    |                6.93 gb/s |                8.54 gb/s |                8.12 gb/s |
| `nk_each_scale_i64_skylake`   |                11.0 gb/s |                12.9 gb/s |                8.48 gb/s |
| `nk_each_blend_i64_serial`    |                10.2 gb/s |                13.3 gb/s |                9.78 gb/s |
| `nk_each_fma_i64_serial`      |                12.6 gb/s |                17.7 gb/s |                11.0 gb/s |
| `nk_each_fma_i64_skylake`     |                20.2 gb/s |                20.7 gb/s |                11.0 gb/s |
| __u64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`      |                13.7 gb/s |                15.7 gb/s |                9.78 gb/s |
| `nk_each_sum_u64_icelake`     |                16.9 gb/s |                17.9 gb/s |                11.5 gb/s |
| `nk_each_scale_u64_serial`    |                7.44 gb/s |                9.12 gb/s |                8.21 gb/s |
| `nk_each_scale_u64_skylake`   |                10.8 gb/s |                12.9 gb/s |                6.74 gb/s |
| `nk_each_blend_u64_serial`    |                11.1 gb/s |                15.4 gb/s |                12.9 gb/s |
| `nk_each_fma_u64_serial`      |                14.2 gb/s |                20.1 gb/s |                13.0 gb/s |
| `nk_each_fma_u64_skylake`     |                20.1 gb/s |                20.3 gb/s |                10.6 gb/s |
| __f64c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f64c_serial`   |       9.87 gb/s, 3.4 ulp |       6.56 gb/s, 2.6 ulp |       5.24 gb/s, 2.0 ulp |
| `nk_each_scale_f64c_haswell`  |       9.21 gb/s, 3.5 ulp |       6.01 gb/s, 2.0 ulp |       5.49 gb/s, 2.1 ulp |
| `nk_each_scale_f64c_skylake`  |       8.65 gb/s, 3.5 ulp |       5.60 gb/s, 2.0 ulp |       5.28 gb/s, 2.0 ulp |
| `nk_each_blend_f64c_serial`   |       15.1 gb/s, 2.5 ulp |       8.20 gb/s, 2.4 ulp |       7.66 gb/s, 2.6 ulp |
| `nk_each_blend_f64c_haswell`  |       13.0 gb/s, 2.5 ulp |       7.69 gb/s, 2.4 ulp |       8.00 gb/s, 2.7 ulp |
| `nk_each_blend_f64c_skylake`  |       13.3 gb/s, 2.5 ulp |       8.21 gb/s, 2.7 ulp |       6.83 gb/s, 2.7 ulp |
| `nk_each_fma_f64c_serial`     |       16.2 gb/s, 4.5 ulp |       9.41 gb/s, 3.2 ulp |       8.80 gb/s, 2.8 ulp |
| `nk_each_fma_f64c_haswell`    |       13.5 gb/s, 4.9 ulp |       8.25 gb/s, 3.1 ulp |       8.67 gb/s, 2.7 ulp |
| `nk_each_fma_f64c_skylake`    |       15.0 gb/s, 4.0 ulp |       8.32 gb/s, 3.4 ulp |       9.69 gb/s, 2.8 ulp |
| __f32c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f32c_serial`   |       9.41 gb/s, 2.1 ulp |       8.64 gb/s, 1.8 ulp |       5.77 gb/s, 2.1 ulp |
| `nk_each_scale_f32c_haswell`  |       8.20 gb/s, 2.1 ulp |       8.26 gb/s, 1.8 ulp |       7.20 gb/s, 2.1 ulp |
| `nk_each_scale_f32c_skylake`  |       9.01 gb/s, 2.2 ulp |       7.09 gb/s, 1.8 ulp |       6.60 gb/s, 2.1 ulp |
| `nk_each_blend_f32c_serial`   |       13.6 gb/s, 6.9 ulp |       10.1 gb/s, 2.8 ulp |       7.85 gb/s, 8.7 ulp |
| `nk_each_blend_f32c_haswell`  |       12.1 gb/s, 8.7 ulp |       11.3 gb/s, 2.9 ulp |       9.78 gb/s, 9.7 ulp |
| `nk_each_blend_f32c_skylake`  |       13.0 gb/s, 7.7 ulp |       8.64 gb/s, 2.7 ulp |      8.82 gb/s, 10.2 ulp |
| `nk_each_fma_f32c_serial`     |       14.6 gb/s, 8.8 ulp |      11.14 gb/s, 3.9 ulp |       8.69 gb/s, 8.5 ulp |
| `nk_each_fma_f32c_haswell`    |       13.1 gb/s, 6.6 ulp |       9.59 gb/s, 2.9 ulp |       9.10 gb/s, 7.2 ulp |
| `nk_each_fma_f32c_skylake`    |       14.4 gb/s, 9.2 ulp |       9.59 gb/s, 3.8 ulp |       9.27 gb/s, 8.3 ulp |

### Apple M5

#### Native

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64_serial`       |         50.9 gb/s, 0 ulp |         62.1 gb/s, 0 ulp |         48.2 gb/s, 0 ulp |
| `nk_each_sum_f64_neon`         |         50.1 gb/s, 0 ulp |         66.4 gb/s, 0 ulp |         52.2 gb/s, 0 ulp |
| `nk_each_scale_f64_serial`     |         43.2 gb/s, 0 ulp |         54.2 gb/s, 0 ulp |         44.9 gb/s, 0 ulp |
| `nk_each_scale_f64_neon`       |         39.2 gb/s, 0 ulp |         47.3 gb/s, 0 ulp |         48.4 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`     |         69.9 gb/s, 0 ulp |         66.9 gb/s, 0 ulp |         56.3 gb/s, 0 ulp |
| `nk_each_blend_f64_neon`       |       60.4 gb/s, 1.9 ulp |       65.0 gb/s, 2.7 ulp |       60.1 gb/s, 1.8 ulp |
| `nk_each_fma_f64_serial`       |         78.0 gb/s, 0 ulp |         71.1 gb/s, 0 ulp |         61.5 gb/s, 0 ulp |
| `nk_each_fma_f64_neon`         |       66.6 gb/s, 1.4 ulp |       66.7 gb/s, 1.6 ulp |       64.8 gb/s, 1.6 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32_serial`       |         83.7 gb/s, 0 ulp |         63.1 gb/s, 0 ulp |         55.5 gb/s, 0 ulp |
| `nk_each_sum_f32_neon`         |         70.3 gb/s, 0 ulp |         61.7 gb/s, 0 ulp |         51.4 gb/s, 0 ulp |
| `nk_each_scale_f32_serial`     |         61.5 gb/s, 0 ulp |         57.3 gb/s, 0 ulp |         50.7 gb/s, 0 ulp |
| `nk_each_scale_f32_neon`       |         46.3 gb/s, 0 ulp |         41.6 gb/s, 0 ulp |         45.8 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`     |        81.1 gb/s, 26 ulp |        62.3 gb/s, 26 ulp |       48.5 gb/s, 2.0 ulp |
| `nk_each_blend_f32_neon`       |       69.3 gb/s, 1.7 ulp |       62.1 gb/s, 1.6 ulp |       46.7 gb/s, 1.6 ulp |
| `nk_each_fma_f32_serial`       |       83.3 gb/s, 2.1 ulp |       47.8 gb/s, 2.5 ulp |       52.3 gb/s, 2.2 ulp |
| `nk_each_fma_f32_neon`         |       76.7 gb/s, 2.1 ulp |        51.4 gb/s, 21 ulp |       52.2 gb/s, 1.8 ulp |
| __bf16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`      |         30.5 gb/s, 0 ulp |         29.4 gb/s, 0 ulp |         31.5 gb/s, 0 ulp |
| `nk_each_sum_bf16_neonbfdot`   |         39.1 gb/s, 0 ulp |         38.8 gb/s, 0 ulp |         40.3 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`    |         18.0 gb/s, 0 ulp |         16.5 gb/s, 0 ulp |         18.4 gb/s, 0 ulp |
| `nk_each_scale_bf16_neonbfdot` |         25.1 gb/s, 0 ulp |         22.2 gb/s, 0 ulp |         24.9 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`    |        23.6 gb/s, 28 ulp |        22.4 gb/s, 26 ulp |       24.6 gb/s, 2.2 ulp |
| `nk_each_blend_bf16_neonbfdot` |        32.4 gb/s, 29 ulp |        33.6 gb/s, 29 ulp |       32.3 gb/s, 2.2 ulp |
| `nk_each_fma_bf16_serial`      |       29.5 gb/s, 2.1 ulp |       27.3 gb/s, 2.0 ulp |        30.8 gb/s, 33 ulp |
| `nk_each_fma_bf16_neonbfdot`   |       39.9 gb/s, 1.2 ulp |       37.1 gb/s, 1.2 ulp |       36.6 gb/s, 1.5 ulp |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`       |         85.2 gb/s, 0 ulp |         63.5 gb/s, 0 ulp |         61.6 gb/s, 0 ulp |
| `nk_each_sum_f16_neonhalf`     |         77.5 gb/s, 0 ulp |         62.7 gb/s, 0 ulp |         57.0 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`     |         35.0 gb/s, 0 ulp |         33.4 gb/s, 0 ulp |         33.7 gb/s, 0 ulp |
| `nk_each_scale_f16_neon`       |         35.9 gb/s, 0 ulp |         34.2 gb/s, 0 ulp |         30.2 gb/s, 0 ulp |
| `nk_each_blend_f16_serial`     |       35.7 gb/s, 2.0 ulp |       33.3 gb/s, 2.0 ulp |       36.4 gb/s, 2.3 ulp |
| `nk_each_blend_f16_neon`       |       42.3 gb/s, 2.0 ulp |       40.0 gb/s, 2.0 ulp |       36.9 gb/s, 2.3 ulp |
| `nk_each_fma_f16_serial`       |       40.5 gb/s, 2.1 ulp |       34.6 gb/s, 1.8 ulp |       40.4 gb/s, 2.2 ulp |
| `nk_each_fma_f16_neon`         |       47.3 gb/s, 2.1 ulp |       43.3 gb/s, 1.8 ulp |       45.0 gb/s, 2.2 ulp |
| __e4m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`      |        0.333 gb/s, 0 ulp |        0.309 gb/s, 0 ulp |        0.344 gb/s, 0 ulp |
| `nk_each_sum_e4m3_neon`        |         1.48 gb/s, 0 ulp |         1.63 gb/s, 0 ulp |         1.60 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`    |        0.138 gb/s, 0 ulp |        0.128 gb/s, 0 ulp |        0.142 gb/s, 0 ulp |
| `nk_each_scale_e4m3_neon`      |        0.888 gb/s, 0 ulp |        0.997 gb/s, 0 ulp |        0.959 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`    |      0.236 gb/s, 0.4 ulp |      0.223 gb/s, 1.1 ulp |      0.241 gb/s, 2.8 ulp |
| `nk_each_blend_e4m3_neon`      |       1.45 gb/s, 0.1 ulp |         1.62 gb/s, 0 ulp |         1.56 gb/s, 0 ulp |
| `nk_each_fma_e4m3_serial`      |      0.311 gb/s, 0.7 ulp |      0.295 gb/s, 0.6 ulp |      0.319 gb/s, 2.2 ulp |
| `nk_each_fma_e4m3_neon`        |       1.79 gb/s, 0.1 ulp |       2.05 gb/s, 0.6 ulp |       1.94 gb/s, 1.0 ulp |
| __e5m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`      |        0.407 gb/s, 0 ulp |        0.385 gb/s, 0 ulp |        0.414 gb/s, 0 ulp |
| `nk_each_sum_e5m2_neon`        |         2.90 gb/s, 0 ulp |         3.24 gb/s, 0 ulp |         3.13 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`    |        0.168 gb/s, 0 ulp |        0.161 gb/s, 0 ulp |        0.166 gb/s, 0 ulp |
| `nk_each_scale_e5m2_neon`      |         1.53 gb/s, 0 ulp |         1.70 gb/s, 0 ulp |         1.63 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`    |        0.295 gb/s, 0 ulp |        0.300 gb/s, 0 ulp |      0.294 gb/s, 4.9 ulp |
| `nk_each_blend_e5m2_neon`      |       2.81 gb/s, 0.7 ulp |         3.06 gb/s, 0 ulp |         2.98 gb/s, 0 ulp |
| `nk_each_fma_e5m2_serial`      |      0.443 gb/s, 1.9 ulp |      0.424 gb/s, 0.9 ulp |      0.422 gb/s, 4.0 ulp |
| `nk_each_fma_e5m2_neon`        |         3.86 gb/s, 0 ulp |       4.12 gb/s, 1.3 ulp |         4.01 gb/s, 0 ulp |
| __e2m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`      |        0.314 gb/s, ? ulp |        0.277 gb/s, ? ulp |        0.270 gb/s, ? ulp |
| `nk_each_scale_e2m3_serial`    |       0.0950 gb/s, ? ulp |       0.0861 gb/s, ? ulp |       0.0875 gb/s, ? ulp |
| `nk_each_blend_e2m3_serial`    |        0.210 gb/s, ? ulp |        0.187 gb/s, ? ulp |        0.187 gb/s, ? ulp |
| `nk_each_fma_e2m3_serial`      |        0.343 gb/s, ? ulp |        0.308 gb/s, ? ulp |        0.308 gb/s, ? ulp |
| __e3m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`      |        0.507 gb/s, ? ulp |        0.440 gb/s, ? ulp |        0.427 gb/s, ? ulp |
| `nk_each_scale_e3m2_serial`    |        0.164 gb/s, ? ulp |        0.141 gb/s, ? ulp |        0.133 gb/s, ? ulp |
| `nk_each_blend_e3m2_serial`    |        0.331 gb/s, ? ulp |        0.281 gb/s, ? ulp |        0.272 gb/s, ? ulp |
| `nk_each_fma_e3m2_serial`      |        0.471 gb/s, ? ulp |        0.403 gb/s, ? ulp |        0.406 gb/s, ? ulp |
| __i8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`        |                88.1 gb/s |                51.7 gb/s |                39.8 gb/s |
| `nk_each_sum_i8_neon`          |                79.6 gb/s |                56.3 gb/s |                36.9 gb/s |
| `nk_each_scale_i8_serial`      |               0.159 gb/s |               0.135 gb/s |               0.138 gb/s |
| `nk_each_scale_i8_neon`        |                10.2 gb/s |                10.3 gb/s |                9.78 gb/s |
| `nk_each_blend_i8_serial`      |               0.281 gb/s |               0.237 gb/s |               0.253 gb/s |
| `nk_each_blend_i8_neon`        |                11.9 gb/s |                11.9 gb/s |                11.8 gb/s |
| `nk_each_fma_i8_serial`        |               0.406 gb/s |               0.334 gb/s |               0.350 gb/s |
| __u8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`        |                12.3 gb/s |                11.5 gb/s |                11.5 gb/s |
| `nk_each_sum_u8_neon`          |                79.4 gb/s |                79.0 gb/s |                62.7 gb/s |
| `nk_each_scale_u8_serial`      |               0.126 gb/s |               0.112 gb/s |               0.111 gb/s |
| `nk_each_scale_u8_neon`        |                10.1 gb/s |                10.2 gb/s |                10.6 gb/s |
| `nk_each_blend_u8_serial`      |               0.238 gb/s |               0.203 gb/s |               0.205 gb/s |
| `nk_each_blend_u8_neon`        |                11.1 gb/s |                12.0 gb/s |                11.3 gb/s |
| `nk_each_fma_u8_serial`        |               0.332 gb/s |               0.310 gb/s |               0.301 gb/s |
| __i16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`       |                93.1 gb/s |                64.8 gb/s |                73.8 gb/s |
| `nk_each_sum_i16_neon`         |                70.1 gb/s |                61.1 gb/s |                62.3 gb/s |
| `nk_each_scale_i16_serial`     |               0.333 gb/s |               0.277 gb/s |               0.264 gb/s |
| `nk_each_scale_i16_neon`       |                16.4 gb/s |                17.2 gb/s |                14.4 gb/s |
| `nk_each_blend_i16_serial`     |               0.600 gb/s |               0.487 gb/s |               0.458 gb/s |
| `nk_each_fma_i16_serial`       |               0.828 gb/s |               0.681 gb/s |               0.663 gb/s |
| `nk_each_fma_i16_neon`         |                26.0 gb/s |                25.8 gb/s |                26.2 gb/s |
| __u16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`       |                23.3 gb/s |                20.9 gb/s |                21.6 gb/s |
| `nk_each_sum_u16_neon`         |                73.8 gb/s |                62.2 gb/s |                63.4 gb/s |
| `nk_each_scale_u16_serial`     |               0.249 gb/s |               0.228 gb/s |               0.224 gb/s |
| `nk_each_scale_u16_neon`       |                15.8 gb/s |                17.6 gb/s |                17.3 gb/s |
| `nk_each_blend_u16_serial`     |               0.484 gb/s |               0.402 gb/s |               0.414 gb/s |
| `nk_each_fma_u16_serial`       |               0.693 gb/s |               0.618 gb/s |               0.594 gb/s |
| `nk_each_fma_u16_neon`         |                26.2 gb/s |                26.0 gb/s |                26.9 gb/s |
| __i32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`       |                63.9 gb/s |                57.7 gb/s |                66.5 gb/s |
| `nk_each_sum_i32_neon`         |                61.7 gb/s |                62.3 gb/s |                58.1 gb/s |
| `nk_each_scale_i32_serial`     |               0.626 gb/s |               0.571 gb/s |               0.549 gb/s |
| `nk_each_scale_i32_neon`       |                17.0 gb/s |                18.0 gb/s |                17.3 gb/s |
| `nk_each_blend_i32_serial`     |                1.18 gb/s |                1.03 gb/s |               0.925 gb/s |
| `nk_each_fma_i32_serial`       |                1.59 gb/s |                1.43 gb/s |                1.37 gb/s |
| `nk_each_fma_i32_neon`         |                26.4 gb/s |                26.8 gb/s |                25.7 gb/s |
| __u32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`       |                52.3 gb/s |                42.8 gb/s |                46.6 gb/s |
| `nk_each_sum_u32_neon`         |                79.1 gb/s |                64.9 gb/s |                51.3 gb/s |
| `nk_each_scale_u32_serial`     |               0.557 gb/s |               0.517 gb/s |               0.514 gb/s |
| `nk_each_scale_u32_neon`       |                17.1 gb/s |                17.5 gb/s |                17.2 gb/s |
| `nk_each_blend_u32_serial`     |               0.950 gb/s |               0.915 gb/s |               0.888 gb/s |
| `nk_each_fma_u32_serial`       |                1.40 gb/s |                1.33 gb/s |                1.31 gb/s |
| `nk_each_fma_u32_neon`         |                26.3 gb/s |                26.9 gb/s |                25.1 gb/s |
| __i64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`       |                51.4 gb/s |                38.4 gb/s |                50.2 gb/s |
| `nk_each_sum_i64_neon`         |                49.3 gb/s |                38.8 gb/s |                47.9 gb/s |
| `nk_each_scale_i64_serial`     |                7.36 gb/s |                7.12 gb/s |                7.57 gb/s |
| `nk_each_scale_i64_neon`       |                36.7 gb/s |                21.2 gb/s |                32.3 gb/s |
| `nk_each_blend_i64_serial`     |                12.9 gb/s |                12.9 gb/s |                13.1 gb/s |
| `nk_each_fma_i64_serial`       |                12.4 gb/s |                9.26 gb/s |                11.9 gb/s |
| `nk_each_fma_i64_neon`         |                53.7 gb/s |                31.9 gb/s |                46.2 gb/s |
| __u64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`       |                49.0 gb/s |                29.3 gb/s |                47.4 gb/s |
| `nk_each_sum_u64_neon`         |                50.6 gb/s |                38.5 gb/s |                45.4 gb/s |
| `nk_each_scale_u64_serial`     |                8.65 gb/s |                8.83 gb/s |                8.48 gb/s |
| `nk_each_scale_u64_neon`       |                35.8 gb/s |                32.7 gb/s |                32.7 gb/s |
| `nk_each_blend_u64_serial`     |                13.3 gb/s |                15.0 gb/s |                13.1 gb/s |
| `nk_each_fma_u64_serial`       |                13.2 gb/s |                12.9 gb/s |                14.2 gb/s |
| `nk_each_fma_u64_neon`         |                48.8 gb/s |                48.4 gb/s |                49.5 gb/s |
| __f64c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f64c_serial`    |       26.2 gb/s, 2.2 ulp |       36.5 gb/s, 1.9 ulp |       35.2 gb/s, 2.2 ulp |
| `nk_each_scale_f64c_neon`      |       23.8 gb/s, 1.5 ulp |       47.5 gb/s, 1.5 ulp |       39.2 gb/s, 1.3 ulp |
| `nk_each_blend_f64c_serial`    |       41.8 gb/s, 4.2 ulp |       33.1 gb/s, 3.0 ulp |       45.4 gb/s, 2.6 ulp |
| `nk_each_blend_f64c_neon`      |       40.7 gb/s, 3.2 ulp |       43.1 gb/s, 3.0 ulp |       45.0 gb/s, 2.2 ulp |
| `nk_each_fma_f64c_serial`      |       54.2 gb/s, 3.4 ulp |       40.9 gb/s, 5.2 ulp |       48.6 gb/s, 2.5 ulp |
| `nk_each_fma_f64c_neon`        |       52.2 gb/s, 3.2 ulp |       48.6 gb/s, 3.0 ulp |       49.5 gb/s, 2.4 ulp |
| __f32c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f32c_serial`    |        41.7 gb/s, 17 ulp |       35.1 gb/s, 2.4 ulp |        41.3 gb/s, 17 ulp |
| `nk_each_scale_f32c_neon`      |       47.3 gb/s, 1.8 ulp |       49.1 gb/s, 1.6 ulp |       42.6 gb/s, 1.6 ulp |
| `nk_each_blend_f32c_serial`    |       46.8 gb/s, 2.4 ulp |       45.6 gb/s, 2.6 ulp |        49.8 gb/s, 28 ulp |
| `nk_each_blend_f32c_neon`      |       45.6 gb/s, 2.2 ulp |       54.3 gb/s, 2.2 ulp |       47.5 gb/s, 3.2 ulp |
| `nk_each_fma_f32c_serial`      |       52.5 gb/s, 3.1 ulp |        46.4 gb/s, 78 ulp |       48.0 gb/s, 3.5 ulp |
| `nk_each_fma_f32c_neon`        |       58.4 gb/s, 2.9 ulp |       58.1 gb/s, 4.1 ulp |        54.0 gb/s, 81 ulp |
