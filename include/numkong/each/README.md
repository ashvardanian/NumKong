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

Fused SwiGLU and grouped RMSNorm support F32, F16, BF16, and E4M3.
The downcasting `nk_each_rmscast_<type>` reverses a dot product of `<type>`: it reads the F32 of BF16, F16, E4M3, E5M2, E2M3 and E3M2 dots, the F64 of F32 dots, or the I32 and U32 of I8 and U8 dots, and normalizes it into `<type>`.
Integer outputs round to nearest even and saturate, with γ carrying the quantization scale.
Every type has serial, CUDA, ROCm and Metal kernels, Haswell, Skylake and NEON follow their RMSNorm types, Ampere narrows BF16 and Ada narrows E4M3 and E5M2.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_BATCH_PER_CORE` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes read per second.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `sum_f64_with_blas` 🧩         |                15.7 gb/s |                17.4 gb/s |                15.2 gb/s |
| `each_blend_f64_with_blas` 🧩  |                15.2 gb/s |                16.7 gb/s |                15.1 gb/s |
| `nk_each_sum_f64_serial`       |         10.4 gb/s, 0 ulp |         17.3 gb/s, 0 ulp |         9.88 gb/s, 0 ulp |
| `nk_each_sum_f64_haswell`      |         16.5 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |
| `nk_each_sum_f64_skylake`      |         16.6 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |
| `nk_each_scale_f64_serial`     |         7.50 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |         7.39 gb/s, 0 ulp |
| `nk_each_scale_f64_haswell`    |         12.0 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |         12.8 gb/s, 0 ulp |
| `nk_each_scale_f64_skylake`    |         12.0 gb/s, 0 ulp |         12.7 gb/s, 0 ulp |         13.1 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`     |       9.86 gb/s, 1.4 ulp |       17.1 gb/s, 1.1 ulp |       7.97 gb/s, 1.1 ulp |
| `nk_each_blend_f64_haswell`    |       16.7 gb/s, 1.5 ulp |       17.4 gb/s, 1.5 ulp |       16.6 gb/s, 1.1 ulp |
| `nk_each_blend_f64_skylake`    |       15.6 gb/s, 1.7 ulp |       17.6 gb/s, 1.5 ulp |       17.5 gb/s, 1.1 ulp |
| `nk_each_fma_f64_serial`       |       10.3 gb/s, 1.5 ulp |       19.2 gb/s, 1.5 ulp |       8.24 gb/s, 1.3 ulp |
| `nk_each_fma_f64_haswell`      |       19.0 gb/s, 1.5 ulp |       19.7 gb/s, 1.5 ulp |       17.6 gb/s, 2.8 ulp |
| `nk_each_fma_f64_skylake`      |       18.8 gb/s, 1.4 ulp |       19.7 gb/s, 1.5 ulp |       18.9 gb/s, 2.7 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `sum_f32_with_blas` 🧩         |                14.3 gb/s |                16.2 gb/s |                16.4 gb/s |
| `each_blend_f32_with_blas` 🧩  |                13.7 gb/s |                16.2 gb/s |                17.0 gb/s |
| `nk_each_sum_f32_serial`       |         6.88 gb/s, 0 ulp |         14.6 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |
| `nk_each_sum_f32_haswell`      |         17.1 gb/s, 0 ulp |         16.9 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |
| `nk_each_sum_f32_skylake`      |         16.7 gb/s, 0 ulp |         17.1 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |
| `nk_each_scale_f32_serial`     |         5.13 gb/s, 0 ulp |         8.52 gb/s, 0 ulp |         5.79 gb/s, 0 ulp |
| `nk_each_scale_f32_haswell`    |         15.4 gb/s, 0 ulp |         12.7 gb/s, 0 ulp |         13.1 gb/s, 0 ulp |
| `nk_each_scale_f32_skylake`    |         14.9 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |         13.1 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`     |       6.16 gb/s, 351 ulp |       12.4 gb/s, 2.0 ulp |       8.41 gb/s, 1.4 ulp |
| `nk_each_blend_f32_haswell`    |       17.2 gb/s, 2.3 ulp |       17.2 gb/s, 2.1 ulp |       17.5 gb/s, 1.3 ulp |
| `nk_each_blend_f32_skylake`    |       16.6 gb/s, 1.9 ulp |       17.1 gb/s, 1.8 ulp |       17.6 gb/s, 1.3 ulp |
| `nk_each_fma_f32_serial`       |       6.84 gb/s, 1.4 ulp |       14.9 gb/s, 2.1 ulp |       10.3 gb/s, 1.6 ulp |
| `nk_each_fma_f32_haswell`      |       18.2 gb/s, 1.4 ulp |       19.4 gb/s, 1.8 ulp |       19.5 gb/s, 1.5 ulp |
| `nk_each_fma_f32_skylake`      |       18.1 gb/s, 1.4 ulp |       19.4 gb/s, 1.7 ulp |       19.7 gb/s, 1.5 ulp |
| `nk_each_swiglu_f32_serial`    |                1.27 gb/s |                1.26 gb/s |                1.28 gb/s |
| `nk_each_swiglu_f32_haswell`   |                8.44 gb/s |                12.5 gb/s |                13.5 gb/s |
| `nk_each_swiglu_f32_skylake`   |                9.91 gb/s |                14.0 gb/s |                14.6 gb/s |
| `nk_each_rmsnorm_f32_serial`   |                2.04 gb/s |                2.57 gb/s |                2.28 gb/s |
| `nk_each_rmsnorm_f32_haswell`  |                9.72 gb/s |                11.0 gb/s |                10.1 gb/s |
| `nk_each_rmsnorm_f32_skylake`  |                12.1 gb/s |                12.3 gb/s |                11.4 gb/s |
| __bf16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`      |         1.79 gb/s, 0 ulp |         2.73 gb/s, 0 ulp |         1.51 gb/s, 0 ulp |
| `nk_each_sum_bf16_haswell`     |         9.47 gb/s, 0 ulp |         9.48 gb/s, 0 ulp |         12.5 gb/s, 0 ulp |
| `nk_each_sum_bf16_skylake`     |         13.0 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`    |        0.999 gb/s, 0 ulp |         1.36 gb/s, 0 ulp |        0.815 gb/s, 0 ulp |
| `nk_each_scale_bf16_haswell`   |         7.59 gb/s, 0 ulp |         5.85 gb/s, 0 ulp |         7.12 gb/s, 0 ulp |
| `nk_each_scale_bf16_skylake`   |         10.3 gb/s, 0 ulp |         5.43 gb/s, 0 ulp |         11.3 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`    |         1.64 gb/s, 0 ulp |         2.35 gb/s, 0 ulp |         1.35 gb/s, 0 ulp |
| `nk_each_blend_bf16_haswell`   |       9.18 gb/s, 2.2 ulp |       9.21 gb/s, 1.5 ulp |       11.9 gb/s, 1.5 ulp |
| `nk_each_blend_bf16_skylake`   |       11.6 gb/s, 2.3 ulp |       7.28 gb/s, 1.3 ulp |       16.5 gb/s, 1.5 ulp |
| `nk_each_fma_bf16_serial`      |         2.09 gb/s, 0 ulp |         3.19 gb/s, 0 ulp |         1.94 gb/s, 0 ulp |
| `nk_each_fma_bf16_haswell`     |       9.51 gb/s, 1.5 ulp |       11.5 gb/s, 0.9 ulp |       14.5 gb/s, 1.0 ulp |
| `nk_each_fma_bf16_skylake`     |       8.98 gb/s, 1.2 ulp |       8.40 gb/s, 0.7 ulp |       19.0 gb/s, 1.1 ulp |
| `nk_each_swiglu_bf16_serial`   |               0.500 gb/s |               0.497 gb/s |               0.490 gb/s |
| `nk_each_swiglu_bf16_haswell`  |                3.64 gb/s |                3.65 gb/s |                3.79 gb/s |
| `nk_each_swiglu_bf16_skylake`  |                5.45 gb/s |                5.33 gb/s |                5.58 gb/s |
| `nk_each_rmsnorm_bf16_serial`  |               0.978 gb/s |                1.01 gb/s |                1.03 gb/s |
| `nk_each_rmsnorm_bf16_haswell` |                3.87 gb/s |                4.20 gb/s |                4.94 gb/s |
| `nk_each_rmsnorm_bf16_skylake` |                5.12 gb/s |                5.81 gb/s |                6.91 gb/s |
| `nk_each_rmsnorm_bf16_genoa`   |                1.33 gb/s |                1.38 gb/s |                1.45 gb/s |
| `nk_each_rmscast_bf16_haswell` |                6.39 gb/s |                7.84 gb/s |                7.56 gb/s |
| `nk_each_rmscast_bf16_skylake` |                8.45 gb/s |                10.4 gb/s |                12.0 gb/s |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`       |        0.315 gb/s, 0 ulp |        0.451 gb/s, 0 ulp |        0.322 gb/s, 0 ulp |
| `nk_each_sum_f16_haswell`      |         17.0 gb/s, 0 ulp |         15.3 gb/s, 0 ulp |         17.3 gb/s, 0 ulp |
| `nk_each_sum_f16_sapphire`     |         20.1 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |         12.4 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`     |        0.170 gb/s, 0 ulp |        0.233 gb/s, 0 ulp |        0.130 gb/s, 0 ulp |
| `nk_each_scale_f16_haswell`    |         17.7 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |
| `nk_each_scale_f16_skylake`    |         21.7 gb/s, 0 ulp |         12.0 gb/s, 0 ulp |         12.8 gb/s, 0 ulp |
| `nk_each_blend_f16_serial`     |      0.266 gb/s, 1.3 ulp |      0.388 gb/s, 1.6 ulp |      0.277 gb/s, 1.5 ulp |
| `nk_each_blend_f16_haswell`    |       15.3 gb/s, 1.2 ulp |       14.6 gb/s, 1.4 ulp |       17.4 gb/s, 1.5 ulp |
| `nk_each_blend_f16_skylake`    |       18.9 gb/s, 1.3 ulp |       15.9 gb/s, 1.4 ulp |       17.2 gb/s, 1.1 ulp |
| `nk_each_fma_f16_serial`       |      0.267 gb/s, 1.0 ulp |      0.356 gb/s, 1.1 ulp |      0.239 gb/s, 1.2 ulp |
| `nk_each_fma_f16_haswell`      |       15.3 gb/s, 1.4 ulp |       16.5 gb/s, 1.0 ulp |       19.3 gb/s, 1.1 ulp |
| `nk_each_fma_f16_skylake`      |       16.2 gb/s, 1.3 ulp |       18.0 gb/s, 1.3 ulp |       19.5 gb/s, 1.1 ulp |
| `nk_each_swiglu_f16_serial`    |               0.287 gb/s |               0.273 gb/s |               0.248 gb/s |
| `nk_each_rmsnorm_f16_serial`   |               0.314 gb/s |               0.327 gb/s |               0.203 gb/s |
| __e4m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`      |        0.199 gb/s, 0 ulp |        0.250 gb/s, 0 ulp |       0.0958 gb/s, 0 ulp |
| `nk_each_sum_e4m3_haswell`     |         1.30 gb/s, 0 ulp |         1.28 gb/s, 0 ulp |         1.31 gb/s, 0 ulp |
| `nk_each_sum_e4m3_skylake`     |         1.56 gb/s, 0 ulp |         1.34 gb/s, 0 ulp |         2.41 gb/s, 0 ulp |
| `nk_each_sum_e4m3_sapphire`    |         2.12 gb/s, 0 ulp |         2.78 gb/s, 0 ulp |         2.16 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`    |        0.114 gb/s, 0 ulp |        0.168 gb/s, 0 ulp |       0.0543 gb/s, 0 ulp |
| `nk_each_scale_e4m3_haswell`   |        0.700 gb/s, 0 ulp |        0.719 gb/s, 0 ulp |        0.709 gb/s, 0 ulp |
| `nk_each_scale_e4m3_skylake`   |        0.980 gb/s, 0 ulp |        0.850 gb/s, 0 ulp |         1.38 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`    |        0.192 gb/s, 0 ulp |        0.240 gb/s, 0 ulp |        0.105 gb/s, 0 ulp |
| `nk_each_blend_e4m3_haswell`   |       1.21 gb/s, 0.6 ulp |         1.23 gb/s, 0 ulp |         1.26 gb/s, 0 ulp |
| `nk_each_blend_e4m3_skylake`   |         1.64 gb/s, 0 ulp |         1.37 gb/s, 0 ulp |         2.36 gb/s, 0 ulp |
| `nk_each_fma_e4m3_serial`      |        0.209 gb/s, 0 ulp |      0.280 gb/s, 0.9 ulp |        0.138 gb/s, 0 ulp |
| `nk_each_fma_e4m3_haswell`     |         1.62 gb/s, 0 ulp |         1.65 gb/s, 0 ulp |       1.69 gb/s, 0.5 ulp |
| `nk_each_fma_e4m3_skylake`     |         1.95 gb/s, 0 ulp |         1.69 gb/s, 0 ulp |         3.04 gb/s, 0 ulp |
| `nk_each_swiglu_e4m3_serial`   |               0.132 gb/s |               0.129 gb/s |              0.0793 gb/s |
| `nk_each_swiglu_e4m3_haswell`  |               0.671 gb/s |               0.679 gb/s |               0.684 gb/s |
| `nk_each_swiglu_e4m3_skylake`  |                1.23 gb/s |                1.21 gb/s |                1.23 gb/s |
| `nk_each_rmsnorm_e4m3_serial`  |               0.140 gb/s |               0.133 gb/s |               0.126 gb/s |
| `nk_each_rmsnorm_e4m3_haswell` |               0.563 gb/s |               0.602 gb/s |               0.614 gb/s |
| `nk_each_rmsnorm_e4m3_skylake` |                1.01 gb/s |                1.07 gb/s |                1.10 gb/s |
| `nk_each_rmsnorm_e4m3_genoa`   |               0.176 gb/s |               0.175 gb/s |              0.0956 gb/s |
| __e5m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`      |        0.187 gb/s, 0 ulp |        0.260 gb/s, 0 ulp |        0.167 gb/s, 0 ulp |
| `nk_each_sum_e5m2_haswell`     |         1.44 gb/s, 0 ulp |         1.47 gb/s, 0 ulp |         1.44 gb/s, 0 ulp |
| `nk_each_sum_e5m2_skylake`     |         1.77 gb/s, 0 ulp |         1.46 gb/s, 0 ulp |         2.77 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`    |        0.130 gb/s, 0 ulp |        0.178 gb/s, 0 ulp |       0.0838 gb/s, 0 ulp |
| `nk_each_scale_e5m2_haswell`   |        0.812 gb/s, 0 ulp |        0.844 gb/s, 0 ulp |        0.849 gb/s, 0 ulp |
| `nk_each_scale_e5m2_skylake`   |       0.0997 gb/s, 0 ulp |        0.880 gb/s, 0 ulp |         1.55 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`    |        0.167 gb/s, 0 ulp |        0.203 gb/s, 0 ulp |       0.143 gb/s, 50 ulp |
| `nk_each_blend_e5m2_haswell`   |         1.43 gb/s, 0 ulp |         1.42 gb/s, 0 ulp |         1.42 gb/s, 0 ulp |
| `nk_each_blend_e5m2_skylake`   |        0.166 gb/s, 0 ulp |         1.50 gb/s, 0 ulp |         2.66 gb/s, 0 ulp |
| `nk_each_fma_e5m2_serial`      |      0.233 gb/s, 5.1 ulp |        0.276 gb/s, 0 ulp |        0.199 gb/s, 0 ulp |
| `nk_each_fma_e5m2_haswell`     |         1.96 gb/s, 0 ulp |         1.90 gb/s, 0 ulp |         1.94 gb/s, 0 ulp |
| `nk_each_fma_e5m2_skylake`     |        0.185 gb/s, 0 ulp |         2.09 gb/s, 0 ulp |         3.42 gb/s, 0 ulp |
| __e2m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`      |        0.221 gb/s, 0 ulp |        0.256 gb/s, 0 ulp |        0.160 gb/s, 0 ulp |
| `nk_each_scale_e2m3_serial`    |        0.079 gb/s, 0 ulp |       0.0931 gb/s, 0 ulp |       0.0674 gb/s, 0 ulp |
| `nk_each_blend_e2m3_serial`    |        0.155 gb/s, 0 ulp |        0.185 gb/s, 0 ulp |        0.131 gb/s, 0 ulp |
| `nk_each_fma_e2m3_serial`      |        0.244 gb/s, 0 ulp |        0.289 gb/s, 0 ulp |        0.211 gb/s, 0 ulp |
| __e3m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`      |        0.170 gb/s, 0 ulp |        0.203 gb/s, 0 ulp |        0.131 gb/s, 0 ulp |
| `nk_each_scale_e3m2_serial`    |       0.0765 gb/s, 0 ulp |       0.0909 gb/s, 0 ulp |       0.0659 gb/s, 0 ulp |
| `nk_each_blend_e3m2_serial`    |        0.152 gb/s, 0 ulp |        0.169 gb/s, 0 ulp |        0.122 gb/s, 0 ulp |
| `nk_each_fma_e3m2_serial`      |        0.212 gb/s, 0 ulp |        0.179 gb/s, 0 ulp |        0.181 gb/s, 0 ulp |
| __i8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`        |                1.85 gb/s |                1.55 gb/s |                1.58 gb/s |
| `nk_each_sum_i8_haswell`       |                51.0 gb/s |                16.9 gb/s |                17.4 gb/s |
| `nk_each_sum_i8_icelake`       |                39.3 gb/s |                16.8 gb/s |                16.1 gb/s |
| `nk_each_scale_i8_serial`      |              0.0781 gb/s |              0.0671 gb/s |              0.0662 gb/s |
| `nk_each_scale_i8_haswell`     |               0.411 gb/s |               0.416 gb/s |               0.406 gb/s |
| `nk_each_scale_i8_skylake`     |               0.492 gb/s |                4.17 gb/s |                6.82 gb/s |
| `nk_each_blend_i8_serial`      |               0.153 gb/s |               0.133 gb/s |               0.133 gb/s |
| `nk_each_blend_i8_haswell`     |               0.755 gb/s |               0.745 gb/s |               0.753 gb/s |
| `nk_each_blend_i8_skylake`     |                11.8 gb/s |                8.44 gb/s |                11.1 gb/s |
| `nk_each_fma_i8_serial`        |               0.510 gb/s |               0.417 gb/s |               0.317 gb/s |
| `nk_each_fma_i8_haswell`       |               0.746 gb/s |               0.704 gb/s |               0.743 gb/s |
| `nk_each_fma_i8_skylake`       |               0.914 gb/s |                5.55 gb/s |                11.7 gb/s |
| `nk_each_rmscast_i8_serial`    |                1.03 gb/s |                1.06 gb/s |               0.884 gb/s |
| __u8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`        |                2.58 gb/s |                2.22 gb/s |                2.38 gb/s |
| `nk_each_sum_u8_haswell`       |                50.6 gb/s |                17.1 gb/s |                17.3 gb/s |
| `nk_each_sum_u8_icelake`       |                38.8 gb/s |                16.7 gb/s |                16.7 gb/s |
| `nk_each_scale_u8_serial`      |              0.0867 gb/s |              0.0734 gb/s |              0.0626 gb/s |
| `nk_each_scale_u8_haswell`     |               0.415 gb/s |               0.399 gb/s |               0.409 gb/s |
| `nk_each_scale_u8_skylake`     |               0.772 gb/s |                3.71 gb/s |                6.54 gb/s |
| `nk_each_blend_u8_serial`      |               0.156 gb/s |               0.161 gb/s |               0.117 gb/s |
| `nk_each_blend_u8_haswell`     |               0.763 gb/s |               0.714 gb/s |               0.775 gb/s |
| `nk_each_blend_u8_skylake`     |                12.0 gb/s |                8.32 gb/s |                11.1 gb/s |
| `nk_each_fma_u8_serial`        |               0.276 gb/s |               0.342 gb/s |               0.240 gb/s |
| `nk_each_fma_u8_haswell`       |               0.751 gb/s |               0.716 gb/s |               0.749 gb/s |
| `nk_each_fma_u8_skylake`       |                2.24 gb/s |                6.19 gb/s |                11.2 gb/s |
| __i16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`       |                3.03 gb/s |                4.01 gb/s |                2.85 gb/s |
| `nk_each_sum_i16_haswell`      |                30.2 gb/s |                16.8 gb/s |                17.8 gb/s |
| `nk_each_sum_i16_icelake`      |                21.1 gb/s |                16.6 gb/s |                16.3 gb/s |
| `nk_each_scale_i16_serial`     |               0.146 gb/s |               0.188 gb/s |               0.130 gb/s |
| `nk_each_scale_i16_haswell`    |                11.2 gb/s |                7.02 gb/s |                10.1 gb/s |
| `nk_each_scale_i16_skylake`    |                3.35 gb/s |                6.07 gb/s |                11.1 gb/s |
| `nk_each_blend_i16_serial`     |               0.306 gb/s |               0.375 gb/s |               0.308 gb/s |
| `nk_each_fma_i16_serial`       |                1.25 gb/s |                1.46 gb/s |                1.58 gb/s |
| `nk_each_fma_i16_haswell`      |                10.8 gb/s |                12.6 gb/s |                17.4 gb/s |
| `nk_each_fma_i16_skylake`      |                4.84 gb/s |                9.75 gb/s |                18.1 gb/s |
| __u16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`       |                4.64 gb/s |                5.62 gb/s |                6.68 gb/s |
| `nk_each_sum_u16_haswell`      |                29.9 gb/s |                16.4 gb/s |                17.4 gb/s |
| `nk_each_sum_u16_icelake`      |                21.4 gb/s |                16.7 gb/s |                9.37 gb/s |
| `nk_each_scale_u16_serial`     |               0.165 gb/s |               0.199 gb/s |               0.186 gb/s |
| `nk_each_scale_u16_haswell`    |                12.4 gb/s |                7.53 gb/s |                10.4 gb/s |
| `nk_each_scale_u16_skylake`    |                7.17 gb/s |                6.41 gb/s |                11.2 gb/s |
| `nk_each_blend_u16_serial`     |               0.287 gb/s |               0.365 gb/s |               0.338 gb/s |
| `nk_each_fma_u16_serial`       |               0.631 gb/s |               0.761 gb/s |               0.769 gb/s |
| `nk_each_fma_u16_haswell`      |                11.0 gb/s |                12.5 gb/s |                17.6 gb/s |
| `nk_each_fma_u16_skylake`      |                7.95 gb/s |                10.7 gb/s |                17.4 gb/s |
| __i32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`       |                4.39 gb/s |                9.85 gb/s |                10.7 gb/s |
| `nk_each_sum_i32_haswell`      |                13.3 gb/s |                16.7 gb/s |                17.6 gb/s |
| `nk_each_sum_i32_icelake`      |                11.8 gb/s |                17.3 gb/s |                9.91 gb/s |
| `nk_each_scale_i32_serial`     |               0.801 gb/s |                1.14 gb/s |                1.23 gb/s |
| `nk_each_scale_i32_haswell`    |                7.99 gb/s |                10.4 gb/s |                11.6 gb/s |
| `nk_each_scale_i32_skylake`    |                6.81 gb/s |                8.59 gb/s |                12.0 gb/s |
| `nk_each_blend_i32_serial`     |               0.623 gb/s |               0.669 gb/s |               0.791 gb/s |
| `nk_each_fma_i32_serial`       |                2.08 gb/s |                2.52 gb/s |                2.48 gb/s |
| `nk_each_fma_i32_haswell`      |                11.2 gb/s |                16.9 gb/s |                19.2 gb/s |
| `nk_each_fma_i32_skylake`      |                9.22 gb/s |                15.0 gb/s |                18.5 gb/s |
| __u32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`       |                6.13 gb/s |                5.57 gb/s |                9.12 gb/s |
| `nk_each_sum_u32_haswell`      |                17.1 gb/s |                14.6 gb/s |                17.6 gb/s |
| `nk_each_sum_u32_icelake`      |                12.0 gb/s |                17.3 gb/s |                10.4 gb/s |
| `nk_each_scale_u32_serial`     |                1.17 gb/s |               0.688 gb/s |               0.646 gb/s |
| `nk_each_scale_u32_haswell`    |                1.12 gb/s |                1.12 gb/s |                1.10 gb/s |
| `nk_each_scale_u32_skylake`    |                7.82 gb/s |                10.0 gb/s |                11.5 gb/s |
| `nk_each_blend_u32_serial`     |               0.607 gb/s |               0.496 gb/s |               0.345 gb/s |
| `nk_each_fma_u32_serial`       |                2.02 gb/s |                1.74 gb/s |                2.35 gb/s |
| `nk_each_fma_u32_haswell`      |                1.60 gb/s |                1.63 gb/s |                1.57 gb/s |
| `nk_each_fma_u32_skylake`      |                10.1 gb/s |                15.7 gb/s |                18.8 gb/s |
| __i64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`       |                7.55 gb/s |                5.37 gb/s |                5.03 gb/s |
| `nk_each_sum_i64_icelake`      |                11.4 gb/s |                17.6 gb/s |                10.7 gb/s |
| `nk_each_scale_i64_serial`     |                2.08 gb/s |                1.38 gb/s |                1.28 gb/s |
| `nk_each_scale_i64_skylake`    |                8.19 gb/s |                12.8 gb/s |                12.9 gb/s |
| `nk_each_blend_i64_serial`     |                1.32 gb/s |                1.11 gb/s |                1.53 gb/s |
| `nk_each_fma_i64_serial`       |                4.24 gb/s |                3.53 gb/s |                4.58 gb/s |
| `nk_each_fma_i64_skylake`      |                13.7 gb/s |                19.5 gb/s |                15.6 gb/s |
| __u64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`       |                12.0 gb/s |                12.0 gb/s |                14.8 gb/s |
| `nk_each_sum_u64_icelake`      |                11.9 gb/s |                17.3 gb/s |                10.9 gb/s |
| `nk_each_scale_u64_serial`     |                1.42 gb/s |               0.719 gb/s |                1.09 gb/s |
| `nk_each_scale_u64_skylake`    |                9.53 gb/s |                12.7 gb/s |                12.7 gb/s |
| `nk_each_blend_u64_serial`     |                1.06 gb/s |               0.939 gb/s |                1.30 gb/s |
| `nk_each_fma_u64_serial`       |                3.49 gb/s |                2.02 gb/s |                2.72 gb/s |
| `nk_each_fma_u64_skylake`      |                15.4 gb/s |                19.5 gb/s |                17.0 gb/s |
| __f64c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64c_serial`      |                16.9 gb/s |                17.3 gb/s |                8.52 gb/s |
| `nk_each_scale_f64c_serial`    |       10.7 gb/s, 3.4 ulp |       7.90 gb/s, 2.6 ulp |       2.16 gb/s, 2.0 ulp |
| `nk_each_scale_f64c_haswell`   |       12.9 gb/s, 3.5 ulp |       13.2 gb/s, 2.0 ulp |       9.60 gb/s, 2.1 ulp |
| `nk_each_scale_f64c_skylake`   |       12.6 gb/s, 3.5 ulp |       13.0 gb/s, 2.0 ulp |       10.4 gb/s, 2.0 ulp |
| `nk_each_blend_f64c_serial`    |       14.3 gb/s, 2.5 ulp |       10.4 gb/s, 2.4 ulp |       6.31 gb/s, 2.6 ulp |
| `nk_each_blend_f64c_haswell`   |       17.3 gb/s, 2.5 ulp |       17.5 gb/s, 2.4 ulp |       10.7 gb/s, 2.7 ulp |
| `nk_each_blend_f64c_skylake`   |       13.4 gb/s, 2.5 ulp |       17.6 gb/s, 2.7 ulp |       7.77 gb/s, 2.7 ulp |
| `nk_each_fma_f64c_serial`      |       15.7 gb/s, 4.5 ulp |       10.8 gb/s, 3.2 ulp |       8.47 gb/s, 2.8 ulp |
| `nk_each_fma_f64c_haswell`     |       19.5 gb/s, 4.9 ulp |       19.9 gb/s, 3.1 ulp |       10.5 gb/s, 2.7 ulp |
| `nk_each_fma_f64c_skylake`     |       11.9 gb/s, 4.0 ulp |       17.3 gb/s, 3.4 ulp |       7.68 gb/s, 2.8 ulp |
| __f32c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32c_serial`      |                12.8 gb/s |                15.2 gb/s |                13.2 gb/s |
| `nk_each_scale_f32c_serial`    |       6.51 gb/s, 2.1 ulp |       3.70 gb/s, 1.8 ulp |       2.96 gb/s, 2.1 ulp |
| `nk_each_scale_f32c_haswell`   |       11.9 gb/s, 2.1 ulp |       12.9 gb/s, 1.8 ulp |       12.8 gb/s, 2.1 ulp |
| `nk_each_scale_f32c_skylake`   |       10.9 gb/s, 2.2 ulp |       12.8 gb/s, 1.8 ulp |       12.6 gb/s, 2.1 ulp |
| `nk_each_blend_f32c_serial`    |       7.68 gb/s, 6.9 ulp |       5.18 gb/s, 2.8 ulp |       3.66 gb/s, 8.7 ulp |
| `nk_each_blend_f32c_haswell`   |       16.7 gb/s, 8.7 ulp |       17.3 gb/s, 2.9 ulp |       16.5 gb/s, 9.7 ulp |
| `nk_each_blend_f32c_skylake`   |       16.5 gb/s, 7.7 ulp |       17.5 gb/s, 2.7 ulp |      16.6 gb/s, 10.2 ulp |
| `nk_each_fma_f32c_serial`      |       8.44 gb/s, 8.8 ulp |       6.45 gb/s, 3.9 ulp |       5.07 gb/s, 8.5 ulp |
| `nk_each_fma_f32c_haswell`     |       18.8 gb/s, 6.6 ulp |       19.4 gb/s, 2.9 ulp |       17.0 gb/s, 7.2 ulp |
| `nk_each_fma_f32c_skylake`     |       18.8 gb/s, 9.2 ulp |       19.5 gb/s, 3.8 ulp |       16.8 gb/s, 8.3 ulp |

#### CUDA

Rows ran on one `1g.34gb` MIG slice of a B300 with 18 SMs, over a 4096 × 2048 shape per call.

| Kernel                        |   4K × 2K |
| :---------------------------- | --------: |
| __f64__                       | ░░░░░░░░░ |
| `nk_each_sum_f64_cuda`        |  501 gb/s |
| `nk_each_scale_f64_cuda`      |  362 gb/s |
| `nk_each_blend_f64_cuda`      |  474 gb/s |
| `nk_each_fma_f64_cuda`        |  471 gb/s |
| __f32__                       | ░░░░░░░░░ |
| `nk_each_sum_f32_cuda`        |  500 gb/s |
| `nk_each_scale_f32_cuda`      |  359 gb/s |
| `nk_each_blend_f32_cuda`      |  499 gb/s |
| `nk_each_fma_f32_cuda`        |  581 gb/s |
| `nk_each_rmsnorm_f32_cuda`    |  309 gb/s |
| `nk_each_rmscast_f32_cuda`    |  179 gb/s |
| `nk_each_swiglu_f32_cuda`     |  463 gb/s |
| __bf16__                      | ░░░░░░░░░ |
| `nk_each_sum_bf16_cuda`       |  490 gb/s |
| `nk_each_scale_bf16_cuda`     |  333 gb/s |
| `nk_each_blend_bf16_cuda`     |  489 gb/s |
| `nk_each_fma_bf16_cuda`       |  565 gb/s |
| `nk_each_rmsnorm_bf16_cuda`   |  213 gb/s |
| `nk_each_rmscast_bf16_cuda`   |  338 gb/s |
| `nk_each_swiglu_bf16_cuda`    |  354 gb/s |
| `nk_each_sum_bf16_ampere`     |  491 gb/s |
| `nk_each_scale_bf16_ampere`   |  349 gb/s |
| `nk_each_blend_bf16_ampere`   |  490 gb/s |
| `nk_each_fma_bf16_ampere`     |  568 gb/s |
| `nk_each_rmsnorm_bf16_ampere` |  229 gb/s |
| `nk_each_rmscast_bf16_ampere` |  349 gb/s |
| `nk_each_swiglu_bf16_ampere`  |  377 gb/s |
| __e4m3__                      | ░░░░░░░░░ |
| `nk_each_sum_e4m3_cuda`       |  105 gb/s |
| `nk_each_scale_e4m3_cuda`     | 58.4 gb/s |
| `nk_each_blend_e4m3_cuda`     |  104 gb/s |
| `nk_each_fma_e4m3_cuda`       |  128 gb/s |
| `nk_each_rmsnorm_e4m3_cuda`   | 31.8 gb/s |
| `nk_each_rmscast_e4m3_cuda`   |  161 gb/s |
| `nk_each_swiglu_e4m3_cuda`    | 70.7 gb/s |
| `nk_each_sum_e4m3_ada`        |  244 gb/s |
| `nk_each_scale_e4m3_ada`      |  178 gb/s |
| `nk_each_blend_e4m3_ada`      |  241 gb/s |
| `nk_each_fma_e4m3_ada`        |  267 gb/s |
| `nk_each_rmsnorm_e4m3_ada`    | 74.5 gb/s |
| `nk_each_rmscast_e4m3_ada`    |  326 gb/s |
| `nk_each_swiglu_e4m3_ada`     |  162 gb/s |
| __e5m2__                      | ░░░░░░░░░ |
| `nk_each_rmscast_e5m2_cuda`   |  170 gb/s |
| `nk_each_rmscast_e5m2_ada`    |  280 gb/s |
| __e2m3__                      | ░░░░░░░░░ |
| `nk_each_rmscast_e2m3_cuda`   |  132 gb/s |
| __e3m2__                      | ░░░░░░░░░ |
| `nk_each_rmscast_e3m2_cuda`   |  132 gb/s |
| __i8__                        | ░░░░░░░░░ |
| `nk_each_sum_i8_cuda`         |  479 gb/s |
| `nk_each_scale_i8_cuda`       |  233 gb/s |
| `nk_each_blend_i8_cuda`       |  387 gb/s |
| `nk_each_fma_i8_cuda`         |  426 gb/s |
| `nk_each_rmscast_i8_cuda`     |  321 gb/s |
| __u8__                        | ░░░░░░░░░ |
| `nk_each_sum_u8_cuda`         |  486 gb/s |
| `nk_each_scale_u8_cuda`       |  212 gb/s |
| `nk_each_blend_u8_cuda`       |  360 gb/s |
| `nk_each_fma_u8_cuda`         |  437 gb/s |
| `nk_each_rmscast_u8_cuda`     |  320 gb/s |
| __i32__                       | ░░░░░░░░░ |
| `nk_each_sum_i32_cuda`        |  498 gb/s |
| `nk_each_scale_i32_cuda`      | 78.5 gb/s |
| `nk_each_blend_i32_cuda`      | 94.3 gb/s |
| `nk_each_fma_i32_cuda`        |  101 gb/s |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                           |                      256 |                     1024 |                     4096 |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64_serial`         |                9.61 gb/s |                14.1 gb/s |                10.6 gb/s |
| `nk_each_scale_f64_serial`       |                7.39 gb/s |                9.50 gb/s |                6.65 gb/s |
| `nk_each_blend_f64_serial`       |                10.4 gb/s |                14.1 gb/s |                9.19 gb/s |
| `nk_each_fma_f64_serial`         |                12.2 gb/s |                16.1 gb/s |                11.8 gb/s |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32_serial`         |                5.96 gb/s |                7.46 gb/s |                7.42 gb/s |
| `nk_each_sum_f32_v128`           |                14.5 gb/s |                16.6 gb/s |                16.3 gb/s |
| `nk_each_scale_f32_serial`       |                4.46 gb/s |                4.85 gb/s |                3.38 gb/s |
| `nk_each_scale_f32_v128relaxed`  |                9.20 gb/s |                11.9 gb/s |                12.0 gb/s |
| `nk_each_blend_f32_serial`       |                5.52 gb/s |                6.12 gb/s |                4.62 gb/s |
| `nk_each_blend_f32_v128relaxed`  |                9.46 gb/s |                17.2 gb/s |                16.1 gb/s |
| `nk_each_fma_f32_serial`         |                7.30 gb/s |                8.15 gb/s |                5.96 gb/s |
| `nk_each_fma_f32_v128relaxed`    |                11.3 gb/s |                19.7 gb/s |                19.0 gb/s |
| `nk_each_swiglu_f32_serial`      |               0.998 gb/s |                1.08 gb/s |               0.837 gb/s |
| `nk_each_rmsnorm_f32_serial`     |                1.50 gb/s |                1.68 gb/s |                1.23 gb/s |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`        |                1.85 gb/s |                2.29 gb/s |                1.55 gb/s |
| `nk_each_sum_bf16_v128`          |                5.86 gb/s |                5.86 gb/s |                4.88 gb/s |
| `nk_each_scale_bf16_serial`      |                1.17 gb/s |                1.29 gb/s |               0.901 gb/s |
| `nk_each_scale_bf16_v128relaxed` |                2.73 gb/s |                3.29 gb/s |                3.09 gb/s |
| `nk_each_blend_bf16_serial`      |                1.67 gb/s |                2.00 gb/s |                1.41 gb/s |
| `nk_each_blend_bf16_v128relaxed` |                4.19 gb/s |                5.39 gb/s |                4.57 gb/s |
| `nk_each_fma_bf16_serial`        |                2.01 gb/s |                2.60 gb/s |                1.73 gb/s |
| `nk_each_fma_bf16_v128relaxed`   |                5.13 gb/s |                6.91 gb/s |                6.05 gb/s |
| `nk_each_swiglu_bf16_serial`     |               0.388 gb/s |               0.422 gb/s |               0.314 gb/s |
| `nk_each_rmsnorm_bf16_serial`    |               0.608 gb/s |               0.762 gb/s |               0.551 gb/s |
| __f16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`         |               0.734 gb/s |               0.716 gb/s |               0.558 gb/s |
| `nk_each_sum_f16_v128relaxed`    |                1.09 gb/s |                1.32 gb/s |                1.10 gb/s |
| `nk_each_scale_f16_serial`       |               0.331 gb/s |               0.336 gb/s |               0.199 gb/s |
| `nk_each_scale_f16_v128relaxed`  |               0.684 gb/s |               0.795 gb/s |               0.692 gb/s |
| `nk_each_blend_f16_serial`       |               0.625 gb/s |               0.756 gb/s |               0.531 gb/s |
| `nk_each_blend_f16_v128relaxed`  |                1.02 gb/s |                1.18 gb/s |                1.02 gb/s |
| `nk_each_fma_f16_serial`         |               0.444 gb/s |               0.430 gb/s |               0.322 gb/s |
| `nk_each_fma_f16_v128relaxed`    |                1.28 gb/s |                1.50 gb/s |                1.28 gb/s |
| `nk_each_swiglu_f16_serial`      |               0.251 gb/s |               0.268 gb/s |               0.159 gb/s |
| `nk_each_rmsnorm_f16_serial`     |               0.142 gb/s |               0.140 gb/s |               0.118 gb/s |
| __e4m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`        |               0.164 gb/s |               0.190 gb/s |               0.130 gb/s |
| `nk_each_scale_e4m3_serial`      |               0.117 gb/s |               0.121 gb/s |              0.0617 gb/s |
| `nk_each_blend_e4m3_serial`      |               0.194 gb/s |               0.193 gb/s |               0.128 gb/s |
| `nk_each_fma_e4m3_serial`        |               0.216 gb/s |               0.195 gb/s |               0.123 gb/s |
| `nk_each_swiglu_e4m3_serial`     |               0.102 gb/s |               0.111 gb/s |              0.0762 gb/s |
| `nk_each_rmsnorm_e4m3_serial`    |              0.0959 gb/s |              0.0948 gb/s |              0.0575 gb/s |
| __e5m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`        |               0.309 gb/s |               0.350 gb/s |               0.214 gb/s |
| `nk_each_scale_e5m2_serial`      |               0.163 gb/s |               0.153 gb/s |              0.0891 gb/s |
| `nk_each_blend_e5m2_serial`      |               0.314 gb/s |               0.332 gb/s |               0.220 gb/s |
| `nk_each_fma_e5m2_serial`        |               0.428 gb/s |               0.410 gb/s |               0.223 gb/s |
| __e2m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`        |               0.453 gb/s |               0.445 gb/s |               0.231 gb/s |
| `nk_each_scale_e2m3_serial`      |               0.186 gb/s |               0.182 gb/s |               0.113 gb/s |
| `nk_each_blend_e2m3_serial`      |               0.270 gb/s |               0.264 gb/s |               0.177 gb/s |
| `nk_each_fma_e2m3_serial`        |               0.478 gb/s |               0.512 gb/s |               0.297 gb/s |
| __e3m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`        |               0.319 gb/s |               0.308 gb/s |               0.191 gb/s |
| `nk_each_scale_e3m2_serial`      |               0.169 gb/s |               0.165 gb/s |              0.0862 gb/s |
| `nk_each_blend_e3m2_serial`      |               0.295 gb/s |               0.283 gb/s |               0.181 gb/s |
| `nk_each_fma_e3m2_serial`        |               0.418 gb/s |               0.405 gb/s |               0.228 gb/s |
| __i8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`          |                1.61 gb/s |                1.62 gb/s |                1.01 gb/s |
| `nk_each_sum_i8_v128`            |                25.7 gb/s |                14.4 gb/s |                15.5 gb/s |
| `nk_each_scale_i8_serial`        |               0.192 gb/s |               0.193 gb/s |               0.125 gb/s |
| `nk_each_scale_i8_v128relaxed`   |               0.577 gb/s |               0.648 gb/s |               0.583 gb/s |
| `nk_each_blend_i8_serial`        |               0.339 gb/s |               0.345 gb/s |               0.175 gb/s |
| `nk_each_blend_i8_v128relaxed`   |                1.09 gb/s |                1.12 gb/s |                1.01 gb/s |
| `nk_each_fma_i8_serial`          |               0.475 gb/s |               0.482 gb/s |               0.319 gb/s |
| `nk_each_fma_i8_v128relaxed`     |                1.47 gb/s |                1.46 gb/s |                1.30 gb/s |
| `nk_each_rmscast_i8_serial`      |               0.579 gb/s |               0.624 gb/s |               0.353 gb/s |
| __u8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`          |                1.80 gb/s |                1.83 gb/s |                1.89 gb/s |
| `nk_each_sum_u8_v128`            |                25.0 gb/s |                13.7 gb/s |                15.5 gb/s |
| `nk_each_scale_u8_serial`        |               0.189 gb/s |               0.170 gb/s |              0.0794 gb/s |
| `nk_each_scale_u8_v128relaxed`   |               0.606 gb/s |               0.615 gb/s |               0.549 gb/s |
| `nk_each_blend_u8_serial`        |               0.398 gb/s |               0.403 gb/s |               0.199 gb/s |
| `nk_each_blend_u8_v128relaxed`   |                1.07 gb/s |                1.06 gb/s |               0.965 gb/s |
| `nk_each_fma_u8_serial`          |               0.458 gb/s |               0.448 gb/s |               0.407 gb/s |
| `nk_each_fma_u8_v128relaxed`     |                1.43 gb/s |                1.40 gb/s |                1.24 gb/s |
| __i16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`         |                3.10 gb/s |                2.01 gb/s |                3.28 gb/s |
| `nk_each_scale_i16_serial`       |               0.398 gb/s |               0.402 gb/s |               0.376 gb/s |
| `nk_each_blend_i16_serial`       |               0.685 gb/s |               0.698 gb/s |               0.471 gb/s |
| `nk_each_fma_i16_serial`         |                1.02 gb/s |                1.04 gb/s |                1.02 gb/s |
| __u16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`         |                3.36 gb/s |                3.69 gb/s |                3.79 gb/s |
| `nk_each_scale_u16_serial`       |               0.426 gb/s |               0.396 gb/s |               0.153 gb/s |
| `nk_each_blend_u16_serial`       |               0.806 gb/s |               0.806 gb/s |               0.391 gb/s |
| `nk_each_fma_u16_serial`         |                1.09 gb/s |                1.11 gb/s |                1.08 gb/s |
| __i32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`         |                4.98 gb/s |                5.55 gb/s |                4.15 gb/s |
| `nk_each_scale_i32_serial`       |               0.755 gb/s |               0.784 gb/s |               0.458 gb/s |
| `nk_each_blend_i32_serial`       |                1.48 gb/s |                1.55 gb/s |                1.00 gb/s |
| `nk_each_fma_i32_serial`         |                2.09 gb/s |                2.13 gb/s |                2.17 gb/s |
| __u32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`         |                5.54 gb/s |                7.01 gb/s |                7.35 gb/s |
| `nk_each_scale_u32_serial`       |               0.843 gb/s |               0.579 gb/s |               0.252 gb/s |
| `nk_each_blend_u32_serial`       |                1.37 gb/s |                1.04 gb/s |               0.467 gb/s |
| `nk_each_fma_u32_serial`         |                2.29 gb/s |                2.46 gb/s |                1.48 gb/s |
| __i64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`         |                8.40 gb/s |                10.5 gb/s |                6.64 gb/s |
| `nk_each_scale_i64_serial`       |                1.65 gb/s |                1.71 gb/s |                1.10 gb/s |
| `nk_each_blend_i64_serial`       |                2.61 gb/s |                3.23 gb/s |                2.01 gb/s |
| `nk_each_fma_i64_serial`         |                5.07 gb/s |                6.79 gb/s |                4.18 gb/s |
| __u64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`         |                10.3 gb/s |                13.7 gb/s |                9.06 gb/s |
| `nk_each_scale_u64_serial`       |               0.847 gb/s |               0.843 gb/s |               0.600 gb/s |
| `nk_each_blend_u64_serial`       |                1.49 gb/s |                1.44 gb/s |                1.06 gb/s |
| `nk_each_fma_u64_serial`         |                3.36 gb/s |                3.62 gb/s |                2.17 gb/s |
| __f64c__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64c_serial`        |                13.2 gb/s |                10.6 gb/s |                10.7 gb/s |
| `nk_each_scale_f64c_serial`      |                8.41 gb/s |                7.42 gb/s |                7.43 gb/s |
| `nk_each_blend_f64c_serial`      |                12.6 gb/s |                9.64 gb/s |                9.77 gb/s |
| `nk_each_fma_f64c_serial`        |                13.8 gb/s |                10.4 gb/s |                10.5 gb/s |
| __f32c__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32c_serial`        |                5.50 gb/s |                5.41 gb/s |                5.46 gb/s |
| `nk_each_scale_f32c_serial`      |                4.73 gb/s |                3.84 gb/s |                3.84 gb/s |
| `nk_each_blend_f32c_serial`      |                6.68 gb/s |                4.83 gb/s |                4.95 gb/s |
| `nk_each_fma_f32c_serial`        |                7.25 gb/s |                5.23 gb/s |                5.32 gb/s |

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
