# Type Conversions in NumKong

NumKong implements bidirectional type conversions between all supported numeric formats through Float32 as a hub type.
Conversions cover IEEE 754 floats (Float16, Float32, Float64), brain float (BFloat16), Float8 formats (e4m3, e5m2, e2m3, e3m2), and integers (Int8–Int64, UInt8–UInt64, packed i4x2/u4x2).
All conversions use round-to-nearest-even (RNE) for narrowing and exact widening where the target format has sufficient range and precision.

BFloat16 relates to Float32 by truncation with rounding:

$$
\text{bf16} \approx \text{f32} \gg 16
$$

With RNE tie-breaking to preserve the least significant bit of the truncated result.

Float16 range and precision:

$$
\text{f16} \in [-65504, 65504], \quad \text{min positive normal} = 2^{-14}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def cast(a: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    return a.astype(target_dtype)
```

## Input & Output Types

Float-to-float conversions:

| Input Type | Output Type | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `f64`      | `f32`       | 64-bit to 32-bit, narrowing with RNE       |
| `f32`      | `f64`       | 32-bit to 64-bit, exact widening           |
| `f32`      | `f16`       | 32-bit to 16-bit half precision            |
| `f16`      | `f32`       | 16-bit half to 32-bit, exact widening      |
| `f32`      | `bf16`      | 32-bit to brain float, truncation with RNE |
| `bf16`     | `f32`       | Brain float to 32-bit, exact widening      |

Float-to-Float8 conversions:

| Input Type | Output Type | Description                                   |
| ---------- | ----------- | --------------------------------------------- |
| `f32`      | `e4m3`      | 32-bit to Float8: 4 exponent, 3 mantissa bits |
| `e4m3`     | `f32`       | Float8 to 32-bit, exact via lookup table      |
| `f32`      | `e5m2`      | 32-bit to Float8: 5 exponent, 2 mantissa bits |
| `e5m2`     | `f32`       | Float8 to 32-bit, exact via lookup table      |
| `f32`      | `e2m3`      | 32-bit to MX: 2 exponent, 3 mantissa bits     |
| `e2m3`     | `f32`       | MX to 32-bit, exact via lookup table          |
| `f32`      | `e3m2`      | 32-bit to MX: 3 exponent, 2 mantissa bits     |
| `e3m2`     | `f32`       | MX to 32-bit, exact via lookup table          |

Float-to-integer conversions:

| Input Type | Output Type | Description                         |
| ---------- | ----------- | ----------------------------------- |
| `f32`      | `i8`        | Clamped to [-128, 127], rounded     |
| `f32`      | `u8`        | Clamped to [0, 255], rounded        |
| `f32`      | `i16`       | Clamped to [-32768, 32767], rounded |
| `f32`      | `u16`       | Clamped to [0, 65535], rounded      |
| `f64`      | `i32`       | Clamped to Int32 range, rounded     |
| `f64`      | `u32`       | Clamped to UInt32 range, rounded    |
| `f64`      | `i64`       | Clamped to Int64 range, rounded     |
| `f64`      | `u64`       | Clamped to UInt64 range, rounded    |

Packed sub-byte conversions:

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| `i4x2`     | `i8`        | Signed 4-bit pair to two signed 8-bit values     |
| `u4x2`     | `u8`        | Unsigned 4-bit pair to two unsigned 8-bit values |

## Optimizations

### Lookup Tables for Mini-Floats

`nk_e4m3_to_f32_serial`, `nk_e5m2_to_f32_serial`, `nk_e2m3_to_f32_serial`, `nk_e3m2_to_f32_serial` use 256-entry precomputed lookup tables — each 8-bit input indexes directly into a Float32 result array.
The reverse direction (`nk_f32_to_e4m3_serial`) uses clamping + rounding: clamp to format range, multiply by scale, round-to-nearest, cast to UInt8.
SIMD backends (`nk_cast_haswell`, `nk_cast_skylake`) use `VPGATHERDD` to perform 8 or 16 simultaneous table lookups from the same 256-entry table.
AVX-512 gathers on Skylake achieve ~3cy throughput per 16-element lookup vs ~8cy on Haswell for 8-element gathers.

### BFloat16 as Truncated Float32

`nk_bf16_to_f32_serial` zero-extends by left-shifting 16 bits — exact, no rounding error, single-cycle on all platforms.
`nk_f32_to_bf16_serial` right-shifts with round-to-nearest-even: adds a rounding bias of `0x7FFF + ((bits >> 16) & 1)` before truncating, matching the IEEE 754 RNE tie-breaking rule.
NEON backend uses `vreinterpretq_u16_u8` + `vzip` for zero-extension; Haswell uses `VPSLLD` / `VPSRLD` shifts.

### F16C Hardware Conversion

`nk_f16_to_f32_haswell`, `nk_f32_to_f16_haswell` use the F16C extension instructions `VCVTPH2PS` / `VCVTPS2PH` — single-instruction conversion of 8 elements with correct denormal handling, NaN propagation, and RNE rounding.
The serial fallback (`nk_f16_to_f32_serial`) must handle denormals via explicit exponent/mantissa extraction and conditional re-normalization — ~15 integer ops per element vs 1 instruction with F16C.
AVX-512 (`nk_cast_skylake`) doubles throughput to 16 elements per instruction.
F16C also unlocks a cheaper FP8 → F32 path that bypasses i32-lane bit math: `nk_e5m2x16_to_f32x16_skylake_` and `nk_e5m2x8_to_f32x8_haswell_` widen u8 → u16 and left-shift by 8 (E5M2 shares F16's bias 15, so the result is a bit-exact F16 encoding of every input including subnormals and NaN), then feed `VCVTPH2PS` — three ops total.
E4M3 can't use a plain shift (bias 7 vs 15), but the Giesen-style fake-F16 `((byte & 0x7F) << 7) | ((byte & 0x80) << 8)` gives an F16 whose value differs from the E4M3 magnitude by exactly 2⁸; `nk_e4m3x16_to_f32x16_skylake_` and `nk_e4m3x8_to_f32x8_haswell_` widen through `VCVTPH2PS`, multiply by 256 in F32 to correct, and blend in F32 NaN for the lone `|byte|==0x7F` encoding.
For E4M3 GEMM specifically, `nk_e4m3x16_to_f16x16_skylake_` produces TRUE F16 (bias-corrected, with a small subnormal LUT and NaN blend) so the packed buffer stores 2 bytes/element instead of 4 — the inner loop reads F16 and widens to F32 once per B-load, trading ~10% compute for 50% pack memory.

## Performance

The following performance tables are produced by manually running `nk_bench` included internal tools to measure the throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of bytes read and written per second, with ↓ for downcasts and ↑ for upcasts.
Each kernel runs for at least 5 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel             |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :----------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.542 gb/s |   0.521 gb/s |   0.553 gb/s |    1.10 gb/s |    1.12 gb/s |    1.17 gb/s |
| `nk_cast_haswell`  |    40.8 gb/s |    52.4 gb/s |    55.1 gb/s |    27.7 gb/s |    43.2 gb/s |    46.3 gb/s |
| `nk_cast_skylake`  |    23.6 gb/s |    44.8 gb/s |    46.8 gb/s |    37.6 gb/s |    60.1 gb/s |    61.3 gb/s |
| `nk_cast_icelake`  |    21.4 gb/s |    26.0 gb/s |    27.2 gb/s |    32.6 gb/s |    39.4 gb/s |    44.3 gb/s |
| `nk_cast_sapphire` |    21.5 gb/s |    21.1 gb/s |    49.5 gb/s |    39.2 gb/s |    38.3 gb/s |    56.3 gb/s |
| __f32 ↔ f16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    6.28 gb/s |    6.36 gb/s |    6.20 gb/s |    2.93 gb/s |    2.95 gb/s |    2.89 gb/s |
| `nk_cast_haswell`  |    50.2 gb/s |     106 gb/s |     105 gb/s |    31.7 gb/s |    60.2 gb/s |    66.1 gb/s |
| `nk_cast_skylake`  |    38.0 gb/s |    56.6 gb/s |    39.4 gb/s |    39.7 gb/s |    58.3 gb/s |    43.7 gb/s |
| `nk_cast_icelake`  |    51.8 gb/s |    60.2 gb/s |    54.3 gb/s |    52.2 gb/s |    57.7 gb/s |    60.6 gb/s |
| `nk_cast_sapphire` |    31.8 gb/s |    33.8 gb/s |    38.8 gb/s |    35.0 gb/s |    33.6 gb/s |    51.5 gb/s |
| __f32 ↔ e5m2__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.785 gb/s |   0.725 gb/s |   0.569 gb/s |    2.62 gb/s |    2.57 gb/s |    2.69 gb/s |
| `nk_cast_haswell`  |    7.93 gb/s |    8.39 gb/s |    5.44 gb/s |    12.6 gb/s |    17.9 gb/s |    10.6 gb/s |
| `nk_cast_skylake`  |    10.3 gb/s |    10.8 gb/s |    10.0 gb/s |    27.2 gb/s |    28.6 gb/s |    28.0 gb/s |
| `nk_cast_icelake`  |    5.07 gb/s |    4.96 gb/s |    6.08 gb/s |    14.9 gb/s |    13.7 gb/s |    14.5 gb/s |
| `nk_cast_sapphire` |    7.81 gb/s |    5.25 gb/s |    10.7 gb/s |    24.7 gb/s |    15.2 gb/s |    25.0 gb/s |
| __f32 ↔ e4m3__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.653 gb/s |   0.623 gb/s |   0.445 gb/s |    1.51 gb/s |    1.43 gb/s |    1.44 gb/s |
| `nk_cast_haswell`  |    6.74 gb/s |    7.35 gb/s |    6.68 gb/s |    10.4 gb/s |    12.1 gb/s |    7.47 gb/s |
| `nk_cast_skylake`  |    7.70 gb/s |    9.83 gb/s |    9.79 gb/s |    17.3 gb/s |    23.2 gb/s |    22.2 gb/s |
| `nk_cast_icelake`  |    8.51 gb/s |    9.01 gb/s |    9.43 gb/s |    17.8 gb/s |    20.5 gb/s |    21.4 gb/s |
| `nk_cast_sapphire` |    4.98 gb/s |    4.90 gb/s |    8.56 gb/s |    15.7 gb/s |    11.0 gb/s |    17.1 gb/s |
| __f32 ↔ e3m2__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.863 gb/s |    1.44 gb/s |    1.21 gb/s |    2.46 gb/s |    4.20 gb/s |    4.14 gb/s |
| `nk_cast_haswell`  |    4.70 gb/s |    5.04 gb/s |    5.00 gb/s |    7.47 gb/s |    7.82 gb/s |    8.03 gb/s |
| `nk_cast_skylake`  |    6.34 gb/s |    6.37 gb/s |    6.46 gb/s |    14.7 gb/s |    17.6 gb/s |    17.1 gb/s |
| `nk_cast_icelake`  |    5.34 gb/s |    5.10 gb/s |    6.36 gb/s |    13.3 gb/s |    14.2 gb/s |    21.3 gb/s |
| `nk_cast_sapphire` |    8.78 gb/s |    9.93 gb/s |    7.02 gb/s |    23.0 gb/s |    18.5 gb/s |    20.8 gb/s |
| __f32 ↔ e2m3__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.941 gb/s |    1.39 gb/s |   0.688 gb/s |    2.68 gb/s |    4.79 gb/s |    2.70 gb/s |
| `nk_cast_haswell`  |    4.76 gb/s |    4.51 gb/s |    5.00 gb/s |    8.26 gb/s |    8.92 gb/s |    9.02 gb/s |
| `nk_cast_skylake`  |    6.55 gb/s |    6.54 gb/s |    6.42 gb/s |    13.4 gb/s |    15.9 gb/s |    16.1 gb/s |
| `nk_cast_icelake`  |    5.03 gb/s |    6.41 gb/s |    6.44 gb/s |    12.4 gb/s |    14.8 gb/s |    16.2 gb/s |
| `nk_cast_sapphire` |    9.95 gb/s |    8.90 gb/s |    9.17 gb/s |    19.7 gb/s |    24.1 gb/s |    16.8 gb/s |
| __f32 ↔ i16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    1.99 gb/s |    2.02 gb/s |    2.04 gb/s |    4.59 gb/s |    4.63 gb/s |    4.68 gb/s |
| `nk_cast_haswell`  |    46.4 gb/s |    51.8 gb/s |    53.0 gb/s |    19.8 gb/s |    21.0 gb/s |    21.9 gb/s |
| `nk_cast_skylake`  |    31.0 gb/s |    34.2 gb/s |    36.7 gb/s |    48.7 gb/s |    58.5 gb/s |    61.1 gb/s |
| __f32 ↔ u16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.19 gb/s |    3.13 gb/s |    3.14 gb/s |    4.60 gb/s |    4.82 gb/s |    4.75 gb/s |
| `nk_cast_haswell`  |    36.4 gb/s |    43.6 gb/s |    48.4 gb/s |    19.1 gb/s |    20.6 gb/s |    21.2 gb/s |
| `nk_cast_skylake`  |    32.0 gb/s |    36.1 gb/s |    37.3 gb/s |    48.4 gb/s |    55.0 gb/s |    59.5 gb/s |
| __f32 ↔ i8__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.22 gb/s |    3.62 gb/s |    3.40 gb/s |    5.41 gb/s |    5.65 gb/s |    5.73 gb/s |
| `nk_cast_haswell`  |    21.6 gb/s |    25.5 gb/s |    27.5 gb/s |    12.8 gb/s |    13.6 gb/s |    14.0 gb/s |
| `nk_cast_skylake`  |    13.0 gb/s |    13.2 gb/s |    13.9 gb/s |    22.1 gb/s |    23.4 gb/s |    22.9 gb/s |
| `nk_cast_icelake`  |    14.2 gb/s |    16.4 gb/s |    21.5 gb/s |    25.4 gb/s |    29.4 gb/s |    34.8 gb/s |
| `nk_cast_sapphire` |    26.0 gb/s |    27.3 gb/s |    19.5 gb/s |    33.1 gb/s |    48.9 gb/s |    49.4 gb/s |
| __f32 ↔ u8__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    4.44 gb/s |    4.58 gb/s |    5.84 gb/s |    7.45 gb/s |    7.20 gb/s |    4.24 gb/s |
| `nk_cast_haswell`  |    41.2 gb/s |    42.2 gb/s |    41.4 gb/s |    17.9 gb/s |    19.2 gb/s |    20.8 gb/s |
| `nk_cast_skylake`  |    27.8 gb/s |    31.1 gb/s |    33.4 gb/s |    39.8 gb/s |    48.7 gb/s |    51.5 gb/s |
| __f64 ↔ f32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    11.6 gb/s |    12.2 gb/s |    12.3 gb/s |    12.1 gb/s |    12.9 gb/s |    13.2 gb/s |
| `nk_cast_skylake`  |    52.1 gb/s |    59.4 gb/s |    53.8 gb/s |    54.4 gb/s |    65.9 gb/s |    60.6 gb/s |
| __f64 ↔ i64__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    5.30 gb/s |    5.21 gb/s |    5.21 gb/s |    15.4 gb/s |    16.1 gb/s |    14.0 gb/s |
| `nk_cast_skylake`  |    8.73 gb/s |    9.81 gb/s |    9.03 gb/s |    25.3 gb/s |    26.8 gb/s |    20.3 gb/s |
| __f64 ↔ u64__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    9.17 gb/s |    8.55 gb/s |    8.57 gb/s |    16.3 gb/s |    15.1 gb/s |    15.0 gb/s |
| `nk_cast_skylake`  |    13.8 gb/s |    14.5 gb/s |    15.4 gb/s |    25.5 gb/s |    28.1 gb/s |    19.6 gb/s |
| __f64 ↔ i32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.71 gb/s |    3.97 gb/s |    3.71 gb/s |    11.6 gb/s |    12.3 gb/s |    12.6 gb/s |
| `nk_cast_skylake`  |    38.7 gb/s |    48.1 gb/s |    45.9 gb/s |    54.1 gb/s |    64.2 gb/s |    60.8 gb/s |
| __f64 ↔ u32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    6.37 gb/s |    6.16 gb/s |    6.08 gb/s |    10.9 gb/s |    11.9 gb/s |    10.3 gb/s |
| `nk_cast_skylake`  |    46.6 gb/s |    48.9 gb/s |    49.5 gb/s |    50.2 gb/s |    60.5 gb/s |    62.3 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel           |        ↓ 256 |         ↓ 1K |            ↓ 4K |        ↑ 256 |         ↑ 1K |            ↑ 4K |
| :--------------- | -----------: | -----------: | --------------: | -----------: | -----------: | --------------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       1.63 gb/s |       ? gb/s |       ? gb/s |       2.21 gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |      0.436 gb/s |       ? gb/s |       ? gb/s |       1.19 gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |      0.294 gb/s |       ? gb/s |       ? gb/s |       1.45 gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |      0.239 gb/s |       ? gb/s |       ? gb/s |      0.746 gb/s |

### Apple M5

#### Native

| Kernel           |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :--------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    1.37 gb/s |    1.35 gb/s |    1.41 gb/s |    1.37 gb/s |    1.34 gb/s |    1.38 gb/s |
| `nk_cast_neon`   |    19.3 gb/s |    23.7 gb/s |    23.2 gb/s |    59.4 gb/s |    58.9 gb/s |    57.3 gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    1.37 gb/s |    1.31 gb/s |    1.32 gb/s |    1.37 gb/s |    1.31 gb/s |    1.40 gb/s |
| `nk_cast_neon`   |    20.1 gb/s |    21.9 gb/s |    25.0 gb/s |    52.1 gb/s |    60.2 gb/s |    70.2 gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.681 gb/s |   0.621 gb/s |   0.600 gb/s |    1.17 gb/s |    1.17 gb/s |    1.23 gb/s |
| `nk_cast_neon`   |    8.50 gb/s |    8.45 gb/s |    8.35 gb/s |    40.6 gb/s |    46.5 gb/s |    46.5 gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.683 gb/s |   0.618 gb/s |   0.586 gb/s |    1.02 gb/s |    1.01 gb/s |    1.02 gb/s |
| `nk_cast_neon`   |    7.85 gb/s |    7.91 gb/s |    7.66 gb/s |    18.9 gb/s |    19.2 gb/s |    18.3 gb/s |
| __f32 ↔ e3m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.702 gb/s |   0.632 gb/s |   0.596 gb/s |    1.17 gb/s |    1.13 gb/s |    1.15 gb/s |
| `nk_cast_neon`   |    8.94 gb/s |    9.02 gb/s |    8.91 gb/s |    24.9 gb/s |    25.0 gb/s |    24.4 gb/s |
| __f32 ↔ e2m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.921 gb/s |   0.843 gb/s |   0.715 gb/s |    1.21 gb/s |    1.21 gb/s |    1.26 gb/s |
| `nk_cast_neon`   |    8.89 gb/s |    9.03 gb/s |    8.82 gb/s |    24.9 gb/s |    25.1 gb/s |    24.6 gb/s |
| __f32 ↔ i16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.785 gb/s |   0.679 gb/s |   0.678 gb/s |    1.44 gb/s |    1.39 gb/s |    1.49 gb/s |
| `nk_cast_neon`   |    19.4 gb/s |    22.6 gb/s |    23.9 gb/s |    19.9 gb/s |    23.2 gb/s |    25.9 gb/s |
| __f32 ↔ u16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.916 gb/s |   0.822 gb/s |   0.726 gb/s |    1.37 gb/s |    1.36 gb/s |    1.48 gb/s |
| `nk_cast_neon`   |    20.3 gb/s |    20.6 gb/s |    22.1 gb/s |    15.6 gb/s |    18.5 gb/s |    17.4 gb/s |
| __f32 ↔ i8__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.725 gb/s |   0.616 gb/s |   0.578 gb/s |    1.21 gb/s |    1.21 gb/s |    1.28 gb/s |
| `nk_cast_neon`   |    18.2 gb/s |    24.5 gb/s |    21.7 gb/s |    16.3 gb/s |    18.9 gb/s |    19.8 gb/s |
| __f32 ↔ u8__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.967 gb/s |   0.795 gb/s |   0.723 gb/s |    1.29 gb/s |    1.25 gb/s |    1.40 gb/s |
| `nk_cast_neon`   |    17.5 gb/s |    19.8 gb/s |    19.4 gb/s |    13.8 gb/s |    17.8 gb/s |    15.1 gb/s |
| __f64 ↔ f32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    2.65 gb/s |    2.60 gb/s |    2.70 gb/s |    2.59 gb/s |    2.55 gb/s |    2.65 gb/s |
| `nk_cast_neon`   |    2.87 gb/s |    2.60 gb/s |    2.73 gb/s |    2.64 gb/s |    2.63 gb/s |    2.57 gb/s |
| __f64 ↔ i64__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    2.42 gb/s |    2.00 gb/s |    1.86 gb/s |    3.79 gb/s |    3.61 gb/s |    4.03 gb/s |
| `nk_cast_neon`   |    2.51 gb/s |    1.94 gb/s |    1.78 gb/s |    3.83 gb/s |    3.68 gb/s |    3.79 gb/s |
| __f64 ↔ u64__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    2.56 gb/s |    2.19 gb/s |    2.06 gb/s |    3.71 gb/s |    3.50 gb/s |    3.87 gb/s |
| `nk_cast_neon`   |    2.68 gb/s |    2.10 gb/s |    1.97 gb/s |    3.68 gb/s |    3.61 gb/s |    3.58 gb/s |
| __f64 ↔ i32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    1.58 gb/s |    1.32 gb/s |    1.29 gb/s |    2.65 gb/s |    2.58 gb/s |    2.84 gb/s |
| `nk_cast_neon`   |    1.61 gb/s |    1.33 gb/s |    1.24 gb/s |    2.73 gb/s |    2.63 gb/s |    2.66 gb/s |
| __f64 ↔ u32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    1.83 gb/s |    1.53 gb/s |    1.47 gb/s |    2.55 gb/s |    2.48 gb/s |    2.69 gb/s |
| `nk_cast_neon`   |    1.89 gb/s |    1.53 gb/s |    1.38 gb/s |    2.56 gb/s |    2.54 gb/s |    2.59 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel           |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :--------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.514 gb/s |   0.522 gb/s |   0.538 gb/s |   0.511 gb/s |   0.526 gb/s |   0.519 gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.368 gb/s |   0.363 gb/s |   0.360 gb/s |   0.490 gb/s |   0.480 gb/s |   0.489 gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.323 gb/s |   0.312 gb/s |   0.304 gb/s |   0.423 gb/s |   0.425 gb/s |   0.425 gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.315 gb/s |   0.304 gb/s |   0.295 gb/s |   0.396 gb/s |   0.396 gb/s |   0.397 gb/s |
