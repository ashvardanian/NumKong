# Trigonometric Functions in NumKong

NumKong implements element-wise trigonometric functions — sine, cosine, and arc tangent — with ~3 ulp error bounds for f32 and faithful rounding for f64.
Each function operates on dense vectors, reading input angles (radians) and writing output values of the same length.
The implementations derive from SLEEF (SIMD Library for Evaluating Elementary Functions), adapted for NumKong's ISA dispatch and type system.

Sine:

$$
\text{sin}: \mathbb{R} \to [-1, 1]
$$

Cosine:

$$
\text{cos}: \mathbb{R} \to [-1, 1]
$$

Arc tangent:

$$
\text{atan}: \mathbb{R} \to \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def sin(a: np.ndarray) -> np.ndarray:
    return np.sin(a)

def cos(a: np.ndarray) -> np.ndarray:
    return np.cos(a)

def atan(a: np.ndarray) -> np.ndarray:
    return np.arctan(a)
```

## Input & Output Types

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision                 |
| `f16`      | `f16`       | 16-bit half precision, widened to f32 internally |

## Optimizations

### Cody-Waite Range Reduction

All trigonometric kernels reduce the input angle to $[-\pi/4, \pi/4]$ before polynomial evaluation using Cody-Waite argument reduction.
The constant $\pi$ is split into high and low parts ($\pi_{\text{hi}} + \pi_{\text{lo}}$) to maintain precision during the subtraction $x - n\pi$: `reduced = (x - n * pi_hi) - n * pi_lo`.
Single-part subtraction would lose ~3 bits of precision for large multiples of $\pi$; the two-part split preserves the full mantissa.
The quadrant index $n = \text{round}(x / \pi)$ selects which trigonometric identity to apply (sin-cos swap, sign flip) via a 2-bit branch.

### Minimax Polynomial Approximation

`nk_trig_sin_f32_serial`, `nk_trig_cos_f32_serial` evaluate degree-9 minimax polynomials via Horner's method after range reduction.
The polynomial coefficients are precomputed to minimize maximum error over $[-\pi/4, \pi/4]$ — Chebyshev-optimal, not Taylor truncation.
Horner evaluation: `p = c9*x^2 + c7; p = p*x^2 + c5; p = p*x^2 + c3; p = p*x^2 + c1; p = p*x` — 4 FMA operations plus 1 multiply for the final odd-power term.
`nk_trig_sin_f64_serial` uses degree-19 polynomials for 52-bit mantissa coverage.

### Vectorized Polynomial Evaluation

`nk_trig_sin_f32_haswell`, `nk_trig_cos_f32_skylake` evaluate the same polynomial on 8 (AVX2) or 16 (AVX-512) elements simultaneously.
Range reduction, quadrant selection, and polynomial evaluation all operate on packed vectors — the only scalar operation is the final sign correction via `VBLENDVPS` with the quadrant mask.
`nk_trig_sin_f32_neon` processes 4 elements per iteration using `vfmaq_f32` for the Horner chain.
WASM v128relaxed (`nk_trig_sin_f32_v128relaxed`) uses `f32x4.relaxed_madd` for the FMA steps, achieving ~2x throughput over strict `f32x4.mul` + `f32x4.add` sequences.

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

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f64_serial`   |        0.994 gb/s, 0 ulp |        0.783 gb/s, 0 ulp |        0.827 gb/s, 0 ulp |
| `nk_trig_cos_f64_serial`   |        0.906 gb/s, 0 ulp |        0.784 gb/s, 0 ulp |        0.824 gb/s, 0 ulp |
| `nk_trig_atan_f64_serial`  |        0.307 gb/s, 0 ulp |        0.291 gb/s, 0 ulp |        0.291 gb/s, 0 ulp |
| `nk_trig_sin_f64_haswell`  |         4.59 gb/s, 0 ulp |         4.19 gb/s, 0 ulp |         4.04 gb/s, 0 ulp |
| `nk_trig_cos_f64_haswell`  |         4.25 gb/s, 0 ulp |         4.14 gb/s, 0 ulp |         3.92 gb/s, 0 ulp |
| `nk_trig_atan_f64_haswell` |         3.83 gb/s, 0 ulp |         3.21 gb/s, 0 ulp |         3.49 gb/s, 0 ulp |
| `nk_trig_sin_f64_skylake`  |         7.65 gb/s, 0 ulp |         6.55 gb/s, 0 ulp |         4.70 gb/s, 0 ulp |
| `nk_trig_cos_f64_skylake`  |         7.88 gb/s, 0 ulp |         5.76 gb/s, 0 ulp |         5.01 gb/s, 0 ulp |
| `nk_trig_atan_f64_skylake` |         5.08 gb/s, 0 ulp |         4.72 gb/s, 0 ulp |         4.58 gb/s, 0 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f32_serial`   |         6.29 gb/s, 5 ulp |         6.07 gb/s, 5 ulp |         5.41 gb/s, 5 ulp |
| `nk_trig_cos_f32_serial`   |        7.03 gb/s, 15 ulp |        6.24 gb/s, 15 ulp |        5.16 gb/s, 15 ulp |
| `nk_trig_atan_f32_serial`  |      0.642 gb/s, 0.4 ulp |      0.541 gb/s, 0.4 ulp |      0.567 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_haswell`  |         10.0 gb/s, 5 ulp |         7.36 gb/s, 5 ulp |         5.63 gb/s, 5 ulp |
| `nk_trig_cos_f32_haswell`  |        7.82 gb/s, 15 ulp |        7.11 gb/s, 15 ulp |        5.09 gb/s, 15 ulp |
| `nk_trig_atan_f32_haswell` |       7.63 gb/s, 0.4 ulp |       5.94 gb/s, 0.4 ulp |       5.38 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_skylake`  |         11.9 gb/s, 5 ulp |         9.14 gb/s, 5 ulp |         5.43 gb/s, 5 ulp |
| `nk_trig_cos_f32_skylake`  |        10.4 gb/s, 15 ulp |        8.26 gb/s, 15 ulp |        5.40 gb/s, 15 ulp |
| `nk_trig_atan_f32_skylake` |       9.07 gb/s, 0.4 ulp |       7.80 gb/s, 0.4 ulp |       5.75 gb/s, 0.4 ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f16_serial`   |      0.112 gb/s, 0.9 ulp |      0.102 gb/s, 1.1 ulp |      0.110 gb/s, 0.9 ulp |
| `nk_trig_cos_f16_serial`   |       0.105 gb/s, 12 ulp |      0.0962 gb/s, 12 ulp |      0.0976 gb/s, 12 ulp |
| `nk_trig_atan_f16_serial`  |     0.0208 gb/s, 6.4 ulp |     0.0201 gb/s, 6.7 ulp |     0.0204 gb/s, 6.6 ulp |
| `nk_trig_sin_f16_skylake`  |     6.05 gb/s, 8.41K ulp |     5.81 gb/s, 8.43K ulp |     5.24 gb/s, 8.41K ulp |
| `nk_trig_cos_f16_skylake`  |     6.05 gb/s, 8.34K ulp |     5.20 gb/s, 8.34K ulp |     5.09 gb/s, 8.35K ulp |
| `nk_trig_atan_f16_skylake` |     4.86 gb/s, 16.5K ulp |     5.25 gb/s, 16.6K ulp |     4.76 gb/s, 16.5K ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f64_serial`       |       0.34 gb/s, 0.2 ulp |       0.38 gb/s, 0.2 ulp |       0.08 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_serial`       |       0.36 gb/s, 0.3 ulp |       0.39 gb/s, 0.3 ulp |       0.08 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_serial`      |       0.11 gb/s, 0.3 ulp |       0.12 gb/s, 0.3 ulp |       0.11 gb/s, 0.3 ulp |
| `nk_trig_sin_f64_v128relaxed`  |       0.59 gb/s, 0.2 ulp |       0.26 gb/s, 0.2 ulp |       0.05 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_v128relaxed`  |       0.29 gb/s, 0.3 ulp |       0.50 gb/s, 0.3 ulp |       0.03 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_v128relaxed` |       0.11 gb/s, 0.3 ulp |       0.48 gb/s, 0.3 ulp |       0.21 gb/s, 0.3 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f32_serial`       |       0.17 gb/s, 4.9 ulp |       0.51 gb/s, 4.9 ulp |       0.07 gb/s, 4.9 ulp |
| `nk_trig_cos_f32_serial`       |      0.05 gb/s, 14.4 ulp |      0.41 gb/s, 14.4 ulp |      0.10 gb/s, 14.4 ulp |
| `nk_trig_atan_f32_serial`      |       0.08 gb/s, 0.4 ulp |       0.08 gb/s, 0.4 ulp |       0.09 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_v128relaxed`  |      0.13 gb/s, 20.7 ulp |      0.01 gb/s, 20.7 ulp |      0.10 gb/s, 20.7 ulp |
| `nk_trig_cos_f32_v128relaxed`  |      0.15 gb/s, 21.9 ulp |      0.32 gb/s, 21.9 ulp |      0.05 gb/s, 21.9 ulp |
| `nk_trig_atan_f32_v128relaxed` |       0.45 gb/s, 0.4 ulp |       0.39 gb/s, 0.4 ulp |       0.15 gb/s, 0.4 ulp |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f16_serial`       |       0.07 gb/s, 1.1 ulp |       0.07 gb/s, 1.1 ulp |       0.07 gb/s, 1.1 ulp |
| `nk_trig_cos_f16_serial`       |      0.07 gb/s, 11.8 ulp |      0.07 gb/s, 11.8 ulp |      0.07 gb/s, 11.8 ulp |
| `nk_trig_atan_f16_serial`      |       0.03 gb/s, 6.5 ulp |       0.03 gb/s, 6.5 ulp |       0.03 gb/s, 6.5 ulp |

### Apple M4

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f64_serial`  |      0.627 gb/s, 0.2 ulp |      0.634 gb/s, 0.2 ulp |      0.639 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_serial`  |      0.621 gb/s, 0.3 ulp |      0.632 gb/s, 0.3 ulp |      0.619 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_serial` |      0.153 gb/s, 0.3 ulp |      0.154 gb/s, 0.3 ulp |      0.153 gb/s, 0.3 ulp |
| `nk_trig_sin_f64_neon`    |       5.94 gb/s, 0.2 ulp |       5.75 gb/s, 0.2 ulp |       5.82 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_neon`    |       5.15 gb/s, 0.3 ulp |       5.36 gb/s, 0.3 ulp |       5.37 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_neon`   |       3.53 gb/s, 0.3 ulp |       3.50 gb/s, 0.3 ulp |       3.50 gb/s, 0.3 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f32_serial`  |       7.94 gb/s, 4.9 ulp |       7.94 gb/s, 4.9 ulp |       7.23 gb/s, 4.9 ulp |
| `nk_trig_cos_f32_serial`  |        7.26 gb/s, 14 ulp |        6.41 gb/s, 14 ulp |        6.52 gb/s, 14 ulp |
| `nk_trig_atan_f32_serial` |      0.128 gb/s, 0.4 ulp |      0.129 gb/s, 0.4 ulp |      0.126 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_neon`    |       9.75 gb/s, 4.9 ulp |       9.44 gb/s, 4.9 ulp |       8.13 gb/s, 4.9 ulp |
| `nk_trig_cos_f32_neon`    |        8.68 gb/s, 18 ulp |        7.77 gb/s, 18 ulp |        7.84 gb/s, 18 ulp |
| `nk_trig_atan_f32_neon`   |       5.57 gb/s, 0.4 ulp |       5.00 gb/s, 0.4 ulp |       5.10 gb/s, 0.4 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f16_serial`  |       3.66 gb/s, 1.3 ulp |       3.71 gb/s, 1.3 ulp |       3.38 gb/s, 1.3 ulp |
| `nk_trig_cos_f16_serial`  |        3.28 gb/s, 12 ulp |        3.29 gb/s, 12 ulp |        3.15 gb/s, 12 ulp |
| `nk_trig_atan_f16_serial` |     0.0639 gb/s, 6.5 ulp |     0.0626 gb/s, 6.5 ulp |     0.0627 gb/s, 6.5 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f64_serial`       |      0.771 gb/s, 0.2 ulp |      0.733 gb/s, 0.2 ulp |      0.798 gb/s, 0.2 ulp |
| `nk_trig_sin_f64_v128relaxed`  |       8.65 gb/s, 0.2 ulp |       8.40 gb/s, 0.2 ulp |       9.45 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_serial`       |      0.746 gb/s, 0.3 ulp |      0.703 gb/s, 0.3 ulp |      0.735 gb/s, 0.3 ulp |
| `nk_trig_cos_f64_v128relaxed`  |       9.07 gb/s, 0.3 ulp |       8.80 gb/s, 0.3 ulp |       9.58 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_serial`      |      0.247 gb/s, 0.3 ulp |      0.234 gb/s, 0.3 ulp |      0.232 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_v128relaxed` |       5.51 gb/s, 0.3 ulp |       5.21 gb/s, 0.3 ulp |       5.96 gb/s, 0.3 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f32_serial`       |       9.94 gb/s, 4.9 ulp |       9.44 gb/s, 4.9 ulp |       10.2 gb/s, 4.9 ulp |
| `nk_trig_sin_f32_v128relaxed`  |        17.8 gb/s, 20 ulp |        17.5 gb/s, 20 ulp |        17.9 gb/s, 20 ulp |
| `nk_trig_cos_f32_serial`       |        9.13 gb/s, 14 ulp |        8.96 gb/s, 14 ulp |        9.55 gb/s, 14 ulp |
| `nk_trig_cos_f32_v128relaxed`  |        16.7 gb/s, 21 ulp |        14.7 gb/s, 21 ulp |        16.9 gb/s, 21 ulp |
| `nk_trig_atan_f32_serial`      |      0.191 gb/s, 0.4 ulp |      0.185 gb/s, 0.4 ulp |      0.199 gb/s, 0.4 ulp |
| `nk_trig_atan_f32_v128relaxed` |       11.2 gb/s, 0.4 ulp |       10.7 gb/s, 0.4 ulp |       11.6 gb/s, 0.4 ulp |
