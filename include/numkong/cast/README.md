# Type Conversions in NumKong

NumKong implements bidirectional type conversions between all supported numeric formats through a wider hub type: `u64` when both sides are unsigned integers, `i64` when both sides are integers and at least one is signed, and `f64c` for everything else.
The SIMD backends use a narrower Float32 hub for the subset of types they cover.
Conversions cover IEEE 754 floats (Float16, Float32, Float64), brain float (BFloat16), Float8 formats (e4m3, e5m2, e2m3, e3m2), and integers (Int8–Int64, UInt8–UInt64, packed i4x2/u4x2, packed u1x8).
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
| :--------- | :---------- | :----------------------------------------- |
| `f64`      | `f32`       | 64-bit to 32-bit, narrowing with RNE       |
| `f32`      | `f64`       | 32-bit to 64-bit, exact widening           |
| `f32`      | `f16`       | 32-bit to 16-bit half precision            |
| `f16`      | `f32`       | 16-bit half to 32-bit, exact widening      |
| `f32`      | `bf16`      | 32-bit to brain float, truncation with RNE |
| `bf16`     | `f32`       | Brain float to 32-bit, exact widening      |

Float-to-Float8 conversions:

| Input Type | Output Type | Description                                   |
| :--------- | :---------- | :-------------------------------------------- |
| `f32`      | `e4m3`      | 32-bit to Float8: 4 exponent, 3 mantissa bits |
| `e4m3`     | `f32`       | Float8 to 32-bit, exact via lookup table      |
| `f32`      | `e5m2`      | 32-bit to Float8: 5 exponent, 2 mantissa bits |
| `e5m2`     | `f32`       | Float8 to 32-bit, exact via lookup table      |
| `f32`      | `e2m3`      | 32-bit to MX: 2 exponent, 3 mantissa bits     |
| `e2m3`     | `f32`       | MX to 32-bit, exact via lookup table          |
| `f32`      | `e3m2`      | 32-bit to MX: 3 exponent, 2 mantissa bits     |
| `e3m2`     | `f32`       | MX to 32-bit, exact via lookup table          |
| `f32`      | `e2m1`      | 32-bit to MX: 2 exponent, 1 mantissa bit      |
| `e2m1`     | `f32`       | MX to 32-bit, exact                           |

Float-to-integer conversions:

| Input Type | Output Type | Description                         |
| :--------- | :---------- | :---------------------------------- |
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
| :--------- | :---------- | :----------------------------------------------- |
| `i4x2`     | `i8`        | Signed 4-bit pair to two signed 8-bit values     |
| `u4x2`     | `u8`        | Unsigned 4-bit pair to two unsigned 8-bit values |

Block-scaled conversions go through the same `nk_cast_best`, with the composite dtypes of `nk_block_scaled_format_of_dtype`.
A block-scaled side passes an `nk_<format>_cref_t` as its source or an `nk_<format>_ref_t` as its destination, holding the codes, the block scales and, for NVFP4, the tensor scale.
Encodes derive the scales and decodes apply them:

| Format      | Elements | Scale per Block | Tensor Scale |
| :---------- | :------- | :-------------- | :----------- |
| `nvfp4`     | `e2m1`   | `ue4m3` per 16  | `f32`        |
| `mxfp4`     | `e2m1`   | `ue8m0` per 32  | none         |
| `mxfp6e2m3` | `e2m3`   | `ue8m0` per 32  | none         |
| `mxfp6e3m2` | `e3m2`   | `ue8m0` per 32  | none         |
| `mxfp8e4m3` | `e4m3`   | `ue8m0` per 32  | none         |
| `mxfp8e5m2` | `e5m2`   | `ue8m0` per 32  | none         |
| `mxint8`    | `i8`     | `ue8m0` per 32  | none         |

## Optimizations

### Lookup Tables for Mini-Floats

`nk_e5m2_to_f32_serial` indexes a 128-entry table with the 7-bit magnitude, and `nk_e2m3_to_f32_serial` and `nk_e3m2_to_f32_serial` index 32-entry tables with their 5-bit magnitudes, OR-ing the sign bit back in afterwards.
`nk_e4m3_to_f32_serial` needs no table: the exponent and mantissa fields map onto Float32 with a bias shift of 120, and only the subnormal and the single NaN encoding are special-cased.
The reverse direction (`nk_f32_to_e4m3_serial`) is bit manipulation with RNE rounding: NaN and infinity are mapped explicitly, subnormals below 2⁻⁶ are rounded through a × 512 scale, and normals get their 23-bit mantissa rounded down to 3 bits.
The SIMD backends do not gather from these tables.
On x86 the Float8 paths ride the F16C converters described below, and NEON looks up Float16 high bytes with `VQTBL4` over a 128-byte table in `nk_e4m3x16_to_f16x8x2_neon_` and its siblings.

NEON E4M3 narrowing rounds and rebiases the Float32 magnitude as one integer encoding, clamps finite overflow to 448, and handles subnormals on their fixed 2⁻⁹ grid.
The shared converter preserves signed zero, ties-to-even, and the E4M3 NaN encoding across casts and fused token kernels.

### BFloat16 as Truncated Float32

`nk_bf16_to_f32_serial` zero-extends by left-shifting 16 bits — exact, no rounding error, single-cycle on all platforms.
`nk_f32_to_bf16_serial` right-shifts with round-to-nearest-even: adds a rounding bias of `0x7FFF + ((bits >> 16) & 1)` before truncating, matching the IEEE 754 RNE tie-breaking rule.
NEON backend uses `vmovl_u16` + `vshlq_n_u32` for zero-extension; Haswell uses `VPSLLD` / `VPSRLD` shifts.

### F16C Hardware Conversion

`nk_f16_to_f32_haswell`, `nk_f32_to_f16_haswell` use the F16C extension instructions `VCVTPH2PS` / `VCVTPS2PH` — single-instruction conversion of 8 elements with correct denormal handling, NaN propagation, and RNE rounding.
The serial fallback (`nk_f16_to_f32_serial`) must handle denormals via explicit exponent/mantissa extraction and conditional re-normalization — ~15 integer ops per element vs 1 instruction with F16C.
AVX-512 (`nk_cast_skylake`) doubles throughput to 16 elements per instruction.
F16C also unlocks a cheaper FP8 → F32 path that bypasses i32-lane bit math: `nk_e5m2x16_to_f32x16_skylake_` and `nk_e5m2x8_to_f32x8_haswell_` widen u8 → u16 and left-shift by 8 (E5M2 shares F16's bias 15, so the result is a bit-exact F16 encoding of every input including subnormals and NaN), then feed `VCVTPH2PS` — three ops total.
E4M3 can't use a plain shift (bias 7 vs 15), but the Giesen-style fake-F16 `((byte & 0x7F) << 7) | ((byte & 0x80) << 8)` gives an F16 whose value differs from the E4M3 magnitude by exactly 2⁸; `nk_e4m3x16_to_f32x16_skylake_` and `nk_e4m3x8_to_f32x8_haswell_` widen through `VCVTPH2PS`, multiply by 256 in F32 to correct, and blend in F32 NaN for the lone `|byte|==0x7F` encoding.
For E4M3 GEMM specifically, `nk_e4m3x16_to_f16x16_skylake_` produces TRUE F16 (bias-corrected, with a small subnormal LUT and NaN blend) so the packed buffer stores 2 bytes/element instead of 4 — the inner loop reads F16 and widens to F32 once per B-load, trading ~10% compute for 50% pack memory.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology), measuring throughput alone.
The input size is controlled by the `NUMWARS_BATCH_PER_CORE` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of bytes read and written per second, with ↓ for downcasts and ↑ for upcasts.
Each kernel runs for at least 3 seconds per configuration.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel              |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :------------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    5.03 gb/s |    7.25 gb/s |    7.61 gb/s |   0.705 gb/s |   0.796 gb/s |   0.801 gb/s |
| `nk_cast_haswell`   |    26.6 gb/s |    23.4 gb/s |    23.8 gb/s |    21.9 gb/s |    17.3 gb/s |    44.1 gb/s |
| `nk_cast_skylake`   |    22.0 gb/s |    25.4 gb/s |    21.0 gb/s |    18.6 gb/s |    25.0 gb/s |    25.4 gb/s |
| `nk_cast_icelake`   |    17.1 gb/s |    20.1 gb/s |    21.2 gb/s |    18.7 gb/s |    20.9 gb/s |    22.1 gb/s |
| `nk_cast_sapphire`  |    18.5 gb/s |    20.5 gb/s |    25.9 gb/s |    16.8 gb/s |    22.5 gb/s |    26.0 gb/s |
| __f32 ↔ f16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.442 gb/s |   0.392 gb/s |   0.354 gb/s |   0.543 gb/s |   0.552 gb/s |   0.554 gb/s |
| `nk_cast_haswell`   |    38.6 gb/s |    70.5 gb/s |    89.8 gb/s |    29.4 gb/s |    51.8 gb/s |    44.2 gb/s |
| `nk_cast_skylake`   |    22.5 gb/s |    24.9 gb/s |    23.1 gb/s |    17.7 gb/s |    23.9 gb/s |    22.1 gb/s |
| `nk_cast_icelake`   |    19.6 gb/s |    21.6 gb/s |    25.2 gb/s |    15.9 gb/s |    24.3 gb/s |    26.8 gb/s |
| `nk_cast_sapphire`  |    21.8 gb/s |    22.8 gb/s |    23.4 gb/s |    15.9 gb/s |    21.9 gb/s |    25.3 gb/s |
| __f32 ↔ e5m2__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.402 gb/s |   0.393 gb/s |   0.232 gb/s |   0.529 gb/s |   0.610 gb/s |   0.522 gb/s |
| `nk_cast_haswell`   |    5.61 gb/s |    5.63 gb/s |    5.39 gb/s |    9.16 gb/s |    9.77 gb/s |    10.2 gb/s |
| `nk_cast_skylake`   |    9.49 gb/s |    10.2 gb/s |    7.84 gb/s |    13.9 gb/s |    18.0 gb/s |    19.7 gb/s |
| `nk_cast_icelake`   |    7.16 gb/s |    7.88 gb/s |    7.85 gb/s |    11.7 gb/s |    13.7 gb/s |    14.4 gb/s |
| `nk_cast_sapphire`  |    7.36 gb/s |    7.99 gb/s |    10.2 gb/s |    13.1 gb/s |    13.9 gb/s |    20.8 gb/s |
| __f32 ↔ e4m3__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.398 gb/s |   0.362 gb/s |   0.235 gb/s |   0.460 gb/s |   0.409 gb/s |   0.414 gb/s |
| `nk_cast_haswell`   |    5.57 gb/s |    4.88 gb/s |    6.08 gb/s |    9.98 gb/s |    10.2 gb/s |    16.4 gb/s |
| `nk_cast_skylake`   |    8.71 gb/s |    9.43 gb/s |    8.12 gb/s |    13.8 gb/s |    18.2 gb/s |    14.1 gb/s |
| `nk_cast_icelake`   |    6.76 gb/s |    7.98 gb/s |    7.39 gb/s |    11.3 gb/s |    17.9 gb/s |    13.8 gb/s |
| `nk_cast_sapphire`  |    6.87 gb/s |    7.63 gb/s |    8.87 gb/s |    11.8 gb/s |    14.4 gb/s |    18.6 gb/s |
| __f32 ↔ e3m2__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.401 gb/s |   0.364 gb/s |   0.221 gb/s |   0.512 gb/s |   0.489 gb/s |   0.468 gb/s |
| `nk_cast_haswell`   |    5.59 gb/s |    6.51 gb/s |    7.15 gb/s |    7.58 gb/s |    11.3 gb/s |    12.0 gb/s |
| `nk_cast_skylake`   |    6.42 gb/s |    9.44 gb/s |    7.80 gb/s |    9.11 gb/s |    12.6 gb/s |    10.5 gb/s |
| `nk_cast_icelake`   |    6.44 gb/s |    7.44 gb/s |    9.64 gb/s |    28.6 gb/s |    44.4 gb/s |    49.7 gb/s |
| `nk_cast_sapphire`  |    6.48 gb/s |    7.20 gb/s |    9.72 gb/s |    30.6 gb/s |    37.5 gb/s |    51.2 gb/s |
| __f32 ↔ e2m3__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.412 gb/s |   0.470 gb/s |   0.291 gb/s |   0.528 gb/s |   0.529 gb/s |   0.484 gb/s |
| `nk_cast_haswell`   |    5.69 gb/s |    5.80 gb/s |    5.47 gb/s |    8.49 gb/s |    8.49 gb/s |    12.9 gb/s |
| `nk_cast_skylake`   |    10.1 gb/s |    11.0 gb/s |    10.3 gb/s |    10.0 gb/s |    13.2 gb/s |    11.2 gb/s |
| `nk_cast_icelake`   |    7.77 gb/s |    8.47 gb/s |    8.55 gb/s |    28.6 gb/s |    36.2 gb/s |    44.2 gb/s |
| `nk_cast_sapphire`  |    10.2 gb/s |    8.59 gb/s |    11.3 gb/s |    37.6 gb/s |    37.8 gb/s |    50.6 gb/s |
| __f32 ↔ e2m1__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.548 gb/s |   0.538 gb/s |   0.415 gb/s |   0.706 gb/s |   0.661 gb/s |   0.497 gb/s |
| __f32 ↔ i32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.739 gb/s |   0.725 gb/s |   0.529 gb/s |   0.915 gb/s |   0.911 gb/s |   0.822 gb/s |
| __f32 ↔ i16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.523 gb/s |   0.489 gb/s |   0.380 gb/s |   0.690 gb/s |   0.837 gb/s |   0.663 gb/s |
| `nk_cast_haswell`   |    16.9 gb/s |    30.8 gb/s |    24.6 gb/s |    10.9 gb/s |    21.3 gb/s |    12.5 gb/s |
| `nk_cast_skylake`   |    14.0 gb/s |    21.6 gb/s |    18.4 gb/s |    14.6 gb/s |    21.7 gb/s |    16.8 gb/s |
| __f32 ↔ u16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.582 gb/s |   0.515 gb/s |   0.431 gb/s |   0.704 gb/s |   0.684 gb/s |   0.676 gb/s |
| `nk_cast_haswell`   |    16.9 gb/s |    31.4 gb/s |    24.9 gb/s |    14.6 gb/s |    20.1 gb/s |    11.9 gb/s |
| `nk_cast_skylake`   |    15.4 gb/s |    22.2 gb/s |    17.9 gb/s |    13.6 gb/s |    22.0 gb/s |    18.2 gb/s |
| __f32 ↔ i8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.450 gb/s |   0.468 gb/s |   0.311 gb/s |   0.585 gb/s |   0.654 gb/s |   0.562 gb/s |
| `nk_cast_haswell`   |    18.4 gb/s |    34.5 gb/s |    37.8 gb/s |    9.97 gb/s |    20.6 gb/s |    22.5 gb/s |
| `nk_cast_skylake`   |            ⋯ |            ⋯ |            ⋯ |            ⋯ |            ⋯ |            ⋯ |
| `nk_cast_icelake`   |    11.7 gb/s |    15.7 gb/s |    16.5 gb/s |    12.6 gb/s |    15.0 gb/s |    15.5 gb/s |
| `nk_cast_sapphire`  |    13.0 gb/s |    16.1 gb/s |    20.0 gb/s |    13.7 gb/s |    15.9 gb/s |    21.7 gb/s |
| __f32 ↔ u8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.425 gb/s |   0.479 gb/s |   0.332 gb/s |   0.556 gb/s |   0.653 gb/s |   0.549 gb/s |
| `nk_cast_haswell`   |    15.5 gb/s |    29.1 gb/s |    21.9 gb/s |    9.35 gb/s |    18.9 gb/s |    11.2 gb/s |
| `nk_cast_skylake`   |    12.2 gb/s |    18.9 gb/s |    16.6 gb/s |    12.8 gb/s |    18.4 gb/s |    15.0 gb/s |
| __f32 ↔ i4__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.709 gb/s |   0.677 gb/s |   0.627 gb/s |   0.785 gb/s |   0.798 gb/s |   0.768 gb/s |
| __f32 ↔ u4__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.745 gb/s |   0.749 gb/s |   0.674 gb/s |   0.800 gb/s |   0.813 gb/s |   0.775 gb/s |
| __f64 ↔ f32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.815 gb/s |   0.918 gb/s |   0.833 gb/s |    1.11 gb/s |    1.21 gb/s |    1.05 gb/s |
| `nk_cast_skylake`   |    18.4 gb/s |    25.4 gb/s |    25.3 gb/s |    20.6 gb/s |    26.5 gb/s |    26.9 gb/s |
| __f64 ↔ i64__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.31 gb/s |   0.925 gb/s |   0.775 gb/s |    1.39 gb/s |    1.04 gb/s |    1.40 gb/s |
| `nk_cast_skylake`   |   0.970 gb/s |    1.33 gb/s |   0.836 gb/s |    1.08 gb/s |    1.42 gb/s |    1.06 gb/s |
| __f64 ↔ u64__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.26 gb/s |    1.18 gb/s |   0.960 gb/s |    1.48 gb/s |    1.22 gb/s |   0.927 gb/s |
| `nk_cast_skylake`   |   0.811 gb/s |    1.25 gb/s |    1.07 gb/s |   0.765 gb/s |    1.52 gb/s |    1.24 gb/s |
| __f64 ↔ i32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.05 gb/s |   0.745 gb/s |   0.542 gb/s |    1.59 gb/s |    1.22 gb/s |    1.27 gb/s |
| `nk_cast_skylake`   |    17.6 gb/s |    24.3 gb/s |    18.4 gb/s |    20.3 gb/s |    27.0 gb/s |    21.4 gb/s |
| __f64 ↔ u32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.979 gb/s |   0.715 gb/s |   0.747 gb/s |    1.41 gb/s |    1.10 gb/s |    1.16 gb/s |
| `nk_cast_skylake`   |    17.1 gb/s |    22.3 gb/s |    17.4 gb/s |    17.2 gb/s |    25.9 gb/s |    20.6 gb/s |
| __f64 ↔ i8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |            ⋯ |            ⋯ |            ⋯ |    1.17 gb/s |    1.16 gb/s |    1.16 gb/s |
| __f64 ↔ u8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |            ⋯ |            ⋯ |            ⋯ |    1.17 gb/s |    1.15 gb/s |    1.16 gb/s |
| __i32 ↔ i8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    4.79 gb/s |    4.76 gb/s |    4.66 gb/s |    12.8 gb/s |    15.5 gb/s |    16.1 gb/s |
| `nk_cast_skylake`   |    8.21 gb/s |    9.34 gb/s |    9.78 gb/s |    9.31 gb/s |    10.9 gb/s |    11.4 gb/s |
| __i64 ↔ i16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |            ⋯ |            ⋯ |            ⋯ |    18.1 gb/s |    19.9 gb/s |    20.5 gb/s |
| __f32 ↔ nvfp4__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.464 gb/s |   0.464 gb/s |   0.308 gb/s |    1.02 gb/s |   0.983 gb/s |   0.750 gb/s |
| __f32 ↔ mxfp4__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.532 gb/s |   0.495 gb/s |   0.424 gb/s |    1.18 gb/s |    1.15 gb/s |    1.12 gb/s |
| __f32 ↔ mxfp6e2m3__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.490 gb/s |   0.470 gb/s |   0.314 gb/s |   0.953 gb/s |   0.956 gb/s |   0.927 gb/s |
| __f32 ↔ mxfp6e3m2__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.433 gb/s |   0.427 gb/s |   0.259 gb/s |   0.746 gb/s |   0.762 gb/s |   0.731 gb/s |
| __f32 ↔ mxfp8e4m3__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.448 gb/s |   0.442 gb/s |   0.279 gb/s |   0.771 gb/s |   0.752 gb/s |   0.584 gb/s |
| __f32 ↔ mxfp8e5m2__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.484 gb/s |   0.468 gb/s |   0.266 gb/s |   0.899 gb/s |   0.901 gb/s |   0.873 gb/s |

#### CUDA

Rows ran on one `1g.34gb` MIG slice of a B300 with 18 SMs, converting 4096 rows of 4096 values per call.
Block-scaled rows convert `f32` into elements plus per-block scales and back.

| Kernel              |   ↓ 4K × 4K |   ↑ 4K × 4K |
| :------------------ | ----------: | ----------: |
| __f32 ↔ bf16__      | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |    738 gb/s |    682 gb/s |
| `nk_cast_ampere`    |    741 gb/s |           ⋯ |
| `nk_cast_ada`       |    741 gb/s |           ⋯ |
| __f32 ↔ f16__       | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |    178 gb/s |           ⋯ |
| __f32 ↔ e4m3__      | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |    355 gb/s |    580 gb/s |
| `nk_cast_ada`       |    752 gb/s |    581 gb/s |
| __bf16 ↔ e4m3__     | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |   69.3 gb/s |   82.3 gb/s |
| __f32 ↔ i8__        | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |    159 gb/s |           ⋯ |
| __f64 ↔ f32__       | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |    326 gb/s |           ⋯ |
| __f32 ↔ nvfp4__     | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |    100 gb/s |   98.7 gb/s |
| __f32 ↔ mxfp4__     | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |   85.2 gb/s |           ⋯ |
| __f32 ↔ mxfp8e4m3__ | ░░░░░░░░░░░ | ░░░░░░░░░░░ |
| `nk_cast_cuda`      |   71.0 gb/s |   89.9 gb/s |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                |        ↓ 256 |         ↓ 1K |            ↓ 4K |        ↑ 256 |         ↑ 1K |            ↑ 4K |
| :-------------------- | -----------: | -----------: | --------------: | -----------: | -----------: | --------------: |
| __f32 ↔ bf16__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |    4.65 gb/s |    4.83 gb/s |       3.58 gb/s |   0.389 gb/s |   0.386 gb/s |      0.339 gb/s |
| `nk_cast_v128relaxed` |    4.96 gb/s |    5.52 gb/s |       6.25 gb/s |    5.66 gb/s |    6.19 gb/s |       7.29 gb/s |
| __f32 ↔ f16__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.350 gb/s |   0.318 gb/s |      0.259 gb/s |   0.336 gb/s |   0.336 gb/s |      0.297 gb/s |
| `nk_cast_v128relaxed` |    3.07 gb/s |    2.70 gb/s |       3.42 gb/s |    4.23 gb/s |    3.81 gb/s |       5.32 gb/s |
| __f32 ↔ e5m2__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.273 gb/s |   0.208 gb/s |      0.224 gb/s |   0.271 gb/s |   0.277 gb/s |      0.276 gb/s |
| `nk_cast_v128relaxed` |    2.36 gb/s |    2.15 gb/s |       2.50 gb/s |    3.38 gb/s |    2.91 gb/s |       3.60 gb/s |
| __f32 ↔ e4m3__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.265 gb/s |   0.231 gb/s |      0.197 gb/s |   0.294 gb/s |   0.251 gb/s |      0.282 gb/s |
| `nk_cast_v128relaxed` |    2.12 gb/s |    1.98 gb/s |       2.25 gb/s |    3.79 gb/s |    3.55 gb/s |       4.06 gb/s |
| __f32 ↔ e3m2__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.268 gb/s |   0.256 gb/s |      0.224 gb/s |   0.271 gb/s |   0.289 gb/s |      0.273 gb/s |
| `nk_cast_v128relaxed` |    2.44 gb/s |    2.53 gb/s |       2.54 gb/s |    2.83 gb/s |    2.48 gb/s |       2.60 gb/s |
| __f32 ↔ e2m3__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.270 gb/s |   0.291 gb/s |      0.244 gb/s |   0.273 gb/s |   0.288 gb/s |      0.282 gb/s |
| `nk_cast_v128relaxed` |    2.37 gb/s |    2.28 gb/s |       2.58 gb/s |    1.01 gb/s |    1.10 gb/s |       1.06 gb/s |
| __f32 ↔ e2m1__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.334 gb/s |   0.324 gb/s |      0.305 gb/s |   0.647 gb/s |   0.647 gb/s |      0.466 gb/s |
| __f32 ↔ i32__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.455 gb/s |   0.493 gb/s |      0.405 gb/s |   0.615 gb/s |   0.633 gb/s |      0.582 gb/s |
| __f32 ↔ i16__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.328 gb/s |   0.331 gb/s |      0.292 gb/s |   0.406 gb/s |   0.448 gb/s |      0.435 gb/s |
| __f32 ↔ u16__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.377 gb/s |   0.336 gb/s |      0.260 gb/s |   0.409 gb/s |   0.442 gb/s |      0.416 gb/s |
| __f32 ↔ i8__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.280 gb/s |   0.295 gb/s |      0.250 gb/s |   0.354 gb/s |   0.385 gb/s |      0.348 gb/s |
| `nk_cast_v128relaxed` |    3.08 gb/s |    3.75 gb/s |       3.89 gb/s |    4.30 gb/s |    5.34 gb/s |       5.41 gb/s |
| __f32 ↔ u8__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.310 gb/s |   0.296 gb/s |      0.234 gb/s |   0.356 gb/s |   0.379 gb/s |      0.358 gb/s |
| `nk_cast_v128relaxed` |    2.94 gb/s |    3.10 gb/s |       3.60 gb/s |    4.07 gb/s |    4.72 gb/s |       5.51 gb/s |
| __f32 ↔ i4__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.372 gb/s |   0.371 gb/s |      0.372 gb/s |   0.696 gb/s |   0.722 gb/s |      0.676 gb/s |
| __f32 ↔ u4__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.369 gb/s |   0.368 gb/s |      0.380 gb/s |   0.707 gb/s |   0.719 gb/s |      0.698 gb/s |
| __f64 ↔ f32__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.959 gb/s |    1.04 gb/s |      0.968 gb/s |   0.788 gb/s |   0.848 gb/s |      0.788 gb/s |
| __f64 ↔ i64__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |    1.00 gb/s |   0.922 gb/s |      0.591 gb/s |    1.01 gb/s |    1.14 gb/s |      0.879 gb/s |
| __f64 ↔ u64__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |    1.02 gb/s |   0.979 gb/s |      0.646 gb/s |   0.914 gb/s |   0.800 gb/s |      0.572 gb/s |
| __f64 ↔ i32__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.751 gb/s |   0.843 gb/s |      0.645 gb/s |   0.789 gb/s |   0.935 gb/s |      0.851 gb/s |
| __f64 ↔ u32__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.880 gb/s |   0.747 gb/s |      0.465 gb/s |   0.830 gb/s |   0.932 gb/s |      0.744 gb/s |
| __f64 ↔ i8__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |            ⋯ |            ⋯ |               ⋯ |   0.603 gb/s |   0.675 gb/s |      0.632 gb/s |
| __f64 ↔ u8__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |            ⋯ |            ⋯ |               ⋯ |   0.583 gb/s |   0.682 gb/s |      0.636 gb/s |
| __i32 ↔ i8__          | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |    3.36 gb/s |    4.63 gb/s |       2.93 gb/s |    3.53 gb/s |    3.80 gb/s |       2.92 gb/s |
| __i64 ↔ i16__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |            ⋯ |            ⋯ |               ⋯ |    9.13 gb/s |    13.5 gb/s |       8.18 gb/s |
| __f32 ↔ nvfp4__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.344 gb/s |   0.401 gb/s |      0.250 gb/s |   0.914 gb/s |    1.01 gb/s |      0.979 gb/s |
| __f32 ↔ mxfp4__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.375 gb/s |   0.368 gb/s |      0.265 gb/s |    1.02 gb/s |    1.10 gb/s |       1.06 gb/s |
| __f32 ↔ mxfp6e2m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.321 gb/s |   0.317 gb/s |      0.200 gb/s |   0.321 gb/s |   0.323 gb/s |      0.326 gb/s |
| __f32 ↔ mxfp6e3m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.343 gb/s |   0.325 gb/s |      0.207 gb/s |   0.323 gb/s |   0.324 gb/s |      0.325 gb/s |
| __f32 ↔ mxfp8e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.306 gb/s |   0.281 gb/s |      0.211 gb/s |   0.351 gb/s |   0.347 gb/s |      0.334 gb/s |
| __f32 ↔ mxfp8e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial`      |   0.327 gb/s |   0.315 gb/s |      0.229 gb/s |   0.307 gb/s |   0.318 gb/s |      0.318 gb/s |

### Apple M5

Refreshed rates on Apple M5 Pro use a 1-second warm-up and at least 2 seconds of timed calls, with CPU and GPU measurement windows isolated from other benchmark runs.

#### Native

| Kernel              |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :------------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.28 gb/s |    1.26 gb/s |    1.31 gb/s |    1.28 gb/s |    1.25 gb/s |    1.29 gb/s |
| `nk_cast_neon`      |    18.0 gb/s |    22.1 gb/s |    21.6 gb/s |    55.3 gb/s |    54.9 gb/s |    53.4 gb/s |
| __f32 ↔ f16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.28 gb/s |    1.22 gb/s |    1.23 gb/s |    1.28 gb/s |    1.22 gb/s |    1.30 gb/s |
| `nk_cast_neon`      |    18.7 gb/s |    20.4 gb/s |    23.3 gb/s |    48.5 gb/s |    56.1 gb/s |    65.4 gb/s |
| __f32 ↔ e5m2__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.634 gb/s |   0.578 gb/s |   0.559 gb/s |    1.09 gb/s |    1.09 gb/s |    1.15 gb/s |
| `nk_cast_neon`      |    7.92 gb/s |    7.87 gb/s |    7.78 gb/s |    37.8 gb/s |    43.3 gb/s |    43.3 gb/s |
| __f32 ↔ e4m3__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.636 gb/s |   0.576 gb/s |   0.546 gb/s |   0.950 gb/s |   0.941 gb/s |   0.950 gb/s |
| `nk_cast_neon`      |    10.7 gb/s |    12.6 gb/s |    11.7 gb/s |    17.6 gb/s |    17.9 gb/s |    17.0 gb/s |
| __f32 ↔ e3m2__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.654 gb/s |   0.589 gb/s |   0.555 gb/s |    1.09 gb/s |    1.05 gb/s |    1.07 gb/s |
| `nk_cast_neon`      |    8.33 gb/s |    8.40 gb/s |    8.30 gb/s |    23.2 gb/s |    23.3 gb/s |    22.7 gb/s |
| __f32 ↔ e2m3__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.858 gb/s |   0.785 gb/s |   0.666 gb/s |    1.13 gb/s |    1.13 gb/s |    1.17 gb/s |
| `nk_cast_neon`      |    8.28 gb/s |    8.41 gb/s |    8.21 gb/s |    23.2 gb/s |    23.4 gb/s |    22.9 gb/s |
| __f32 ↔ i16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.731 gb/s |   0.632 gb/s |   0.631 gb/s |    1.34 gb/s |    1.29 gb/s |    1.39 gb/s |
| `nk_cast_neon`      |    18.1 gb/s |    21.0 gb/s |    22.3 gb/s |    18.5 gb/s |    21.6 gb/s |    24.1 gb/s |
| __f32 ↔ u16__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.853 gb/s |   0.766 gb/s |   0.676 gb/s |    1.28 gb/s |    1.27 gb/s |    1.38 gb/s |
| `nk_cast_neon`      |    18.9 gb/s |    19.2 gb/s |    20.6 gb/s |    14.5 gb/s |    17.2 gb/s |    16.2 gb/s |
| __f32 ↔ i8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.675 gb/s |   0.574 gb/s |   0.538 gb/s |    1.13 gb/s |    1.13 gb/s |    1.19 gb/s |
| `nk_cast_neon`      |    17.0 gb/s |    22.8 gb/s |    20.2 gb/s |    15.2 gb/s |    17.6 gb/s |    18.4 gb/s |
| __f32 ↔ u8__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.901 gb/s |   0.740 gb/s |   0.673 gb/s |    1.20 gb/s |    1.16 gb/s |    1.30 gb/s |
| `nk_cast_neon`      |    16.3 gb/s |    18.4 gb/s |    18.1 gb/s |    12.9 gb/s |    16.6 gb/s |    14.1 gb/s |
| __f64 ↔ f32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    2.47 gb/s |    2.42 gb/s |    2.51 gb/s |    2.41 gb/s |    2.37 gb/s |    2.47 gb/s |
| `nk_cast_neon`      |    2.67 gb/s |    2.42 gb/s |    2.54 gb/s |    2.46 gb/s |    2.45 gb/s |    2.39 gb/s |
| __f64 ↔ i64__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    2.25 gb/s |    1.86 gb/s |    1.73 gb/s |    3.53 gb/s |    3.36 gb/s |    3.75 gb/s |
| `nk_cast_neon`      |    2.34 gb/s |    1.81 gb/s |    1.66 gb/s |    3.57 gb/s |    3.43 gb/s |    3.53 gb/s |
| __f64 ↔ u64__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    2.38 gb/s |    2.04 gb/s |    1.92 gb/s |    3.46 gb/s |    3.26 gb/s |    3.60 gb/s |
| `nk_cast_neon`      |    2.50 gb/s |    1.96 gb/s |    1.83 gb/s |    3.43 gb/s |    3.36 gb/s |    3.33 gb/s |
| __f64 ↔ i32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.47 gb/s |    1.23 gb/s |    1.20 gb/s |    2.47 gb/s |    2.40 gb/s |    2.64 gb/s |
| `nk_cast_neon`      |    1.50 gb/s |    1.24 gb/s |    1.15 gb/s |    2.54 gb/s |    2.45 gb/s |    2.48 gb/s |
| __f64 ↔ u32__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.70 gb/s |    1.42 gb/s |    1.37 gb/s |    2.37 gb/s |    2.31 gb/s |    2.51 gb/s |
| `nk_cast_neon`      |    1.76 gb/s |    1.42 gb/s |    1.29 gb/s |    2.38 gb/s |    2.37 gb/s |    2.41 gb/s |
| __f32 ↔ i4__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.65 gb/s |    1.77 gb/s |    1.95 gb/s |    1.79 gb/s |    1.78 gb/s |    1.83 gb/s |
| `nk_cast_neon`      |    12.8 gb/s |    13.8 gb/s |    12.0 gb/s |    12.0 gb/s |    11.4 gb/s |    12.1 gb/s |
| __f32 ↔ u4__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.63 gb/s |    1.66 gb/s |    1.92 gb/s |    1.76 gb/s |    1.83 gb/s |    1.84 gb/s |
| `nk_cast_neon`      |    11.9 gb/s |    12.9 gb/s |    11.0 gb/s |    12.1 gb/s |    11.9 gb/s |    12.6 gb/s |
| __f32 ↔ e2m1__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.18 gb/s |    1.21 gb/s |    1.12 gb/s |    1.74 gb/s |    1.77 gb/s |    1.80 gb/s |
| `nk_cast_neon`      |    9.42 gb/s |    9.80 gb/s |    9.65 gb/s |    10.1 gb/s |    10.3 gb/s |    10.4 gb/s |
| __f32 ↔ mxfp8e4m3__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.897 gb/s |   0.861 gb/s |   0.778 gb/s |    1.42 gb/s |    1.46 gb/s |    1.34 gb/s |
| `nk_cast_neon`      |    3.47 gb/s |    3.57 gb/s |    3.59 gb/s |    11.6 gb/s |    13.1 gb/s |    13.4 gb/s |
| __f32 ↔ mxfp8e5m2__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.910 gb/s |   0.883 gb/s |   0.819 gb/s |    1.94 gb/s |    1.91 gb/s |    1.75 gb/s |
| `nk_cast_neon`      |    3.75 gb/s |    3.86 gb/s |    3.88 gb/s |    13.2 gb/s |    15.0 gb/s |    15.2 gb/s |
| __f32 ↔ mxfp6e2m3__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.883 gb/s |   0.892 gb/s |   0.878 gb/s |    1.85 gb/s |    1.91 gb/s |    1.77 gb/s |
| `nk_cast_neon`      |    3.84 gb/s |    3.96 gb/s |    3.98 gb/s |    12.6 gb/s |    14.1 gb/s |    14.4 gb/s |
| __f32 ↔ mxfp6e3m2__ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |   0.912 gb/s |   0.937 gb/s |   0.873 gb/s |    1.83 gb/s |    1.94 gb/s |    1.85 gb/s |
| `nk_cast_neon`      |    3.91 gb/s |    4.03 gb/s |    4.04 gb/s |    12.0 gb/s |    13.3 gb/s |    13.6 gb/s |
| __f32 ↔ mxfp4__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.22 gb/s |    1.19 gb/s |    1.12 gb/s |    4.09 gb/s |    4.28 gb/s |    4.24 gb/s |
| `nk_cast_neon`      |    3.94 gb/s |    4.03 gb/s |    4.02 gb/s |    7.94 gb/s |    8.65 gb/s |    8.73 gb/s |
| __f32 ↔ nvfp4__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`    |    1.38 gb/s |    1.26 gb/s |    1.21 gb/s |    3.28 gb/s |    3.39 gb/s |    3.37 gb/s |
| `nk_cast_neon`      |    2.10 gb/s |    2.09 gb/s |    1.88 gb/s |    6.15 gb/s |    6.55 gb/s |    6.47 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel           |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :--------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.479 gb/s |   0.486 gb/s |   0.501 gb/s |   0.476 gb/s |   0.490 gb/s |   0.483 gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.343 gb/s |   0.338 gb/s |   0.335 gb/s |   0.456 gb/s |   0.447 gb/s |   0.455 gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.301 gb/s |   0.291 gb/s |   0.283 gb/s |   0.394 gb/s |   0.396 gb/s |   0.396 gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.293 gb/s |   0.283 gb/s |   0.275 gb/s |   0.369 gb/s |   0.369 gb/s |   0.370 gb/s |
