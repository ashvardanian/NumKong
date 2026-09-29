# Trigonometric Functions in NumKong

NumKong implements element-wise trigonometric functions — sine, cosine, and arc tangent — within 2 ulp for f64, 2 ulp for f32 sine and cosine, 3 ulp for f32 arc tangent, and 1 ulp for f16, as measured in [Accuracy](#accuracy).
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

| Input Type | Output Type | Description                                                      |
| :--------- | :---------- | :--------------------------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                                 |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision                                 |
| `f16`      | `f16`       | 16-bit half precision, in f16 with FP16 arithmetic, else via f32 |

## Optimizations

### Cody-Waite Range Reduction

All trigonometric kernels reduce the input angle to $[-\pi/2, \pi/2]$ before polynomial evaluation using Cody-Waite argument reduction.
The multiple $n = \text{round}(x / \pi)$ decides the sign of the result through its parity alone.
Cosine reuses the sine polynomial by reducing against the odd multiple $(n + \frac{1}{2})\pi$ of $\pi/2$ in one pass.
Subtracting a rounded $\pi/2$ on its own would leave an absolute error of $1.6 \cdot 10^{-7}$ next to the zeros of cosine, millions of ulp of results that small.

The f32 kernels with a fused multiply-add split $\pi$ into three f32 parts, $\pi_{\text{hi}} + \pi_{\text{lo}} + \pi_{\text{lowest}}$, and subtract each with one FMA: `reduced = fma(-n, pi_lowest, fma(-n, pi_lo, fma(-n, pi_hi, x)))`.
The first FMA is exact while the reduced angle stays below 2, and the three parts leave $2 \cdot 10^{-23} n$ of $\pi$ unrepresented.
A two-part split leaves $3.4 \cdot 10^{-15} n$, which is 6 ulp next to the zero of cosine at $3\pi/2$ and 688 ulp by $|x| = 10^4$.
The serial and WASM Relaxed SIMD kernels, whose multiply-adds may not be fused, split $\pi$ into four parts of 12 significant bits instead, so that every product $n \cdot \pi_i$ is exact on its own up to $|x| \approx 1.28 \cdot 10^4$.
Neither split replaces Payne-Hanek reduction, so the error grows past those ranges.

### Minimax Polynomial Approximation

The f32 kernels evaluate one odd degree-9 polynomial via Horner's method after range reduction.
Its coefficients are a Remez fit of the relative error over $[-\pi/2, \pi/2]$, at most $6 \cdot 10^{-9}$ before rounding to f32.
Taylor coefficients of the same degree would leave $3.6 \cdot 10^{-6}$ at $|x| = \pi/2$, which is 59 ulp next to $\pm 1$.
Horner evaluation: `p = c9; p = p*x^2 + c7; p = p*x^2 + c5; p = p*x^2 + c3; result = p*x^3 + x` — 3 FMA operations for the polynomial plus one more against the cubed angle.
The f16 kernels widen to f32 and evaluate shorter polynomials that f16 cannot tell apart from the f32 ones, then round back to f16.
Sine and cosine use an odd degree-5 minimax polynomial on $[-\pi/2, \pi/2]$ with free linear term, at most $1.1 \cdot 10^{-4}$ relative error.
Their reduction splits $\pi$ into $3.140625$, whose 8 bits keep $n \cdot \pi_{\text{hi}}$ exact for every f16 input, and one more f32 part, one FMA each.
The serial kernels, lacking a fused multiply-add, split the remainder once more into an 8-bit part and an f32 tail, because $|x|$ reaches 65504 and the rounded product $n \cdot \pi_{\text{lo}}$ alone costs 5 ulp next to cosine's zeros.
Arc tangent folds $|x| > 1$ through $\pi/2 - \text{atan}(1/x)$ and evaluates an odd degree-9 minimax polynomial on $[0, 1]$, at most $3 \cdot 10^{-5}$ relative error.
`nk_trig_sin_f64_serial` uses degree-19 polynomials for 52-bit mantissa coverage.

### Native F16 Evaluation

The NEON FP16, SVE FP16 and Sapphire Rapids kernels evaluate in f16, on twice the lanes of f32 and without conversions.
The f32-fitted polynomials above, with coefficients rounded to f16, reach 2 ulp there, so these kernels use their own.
Sine and cosine evaluate `r + r³·(c3 + r²·(c5 + r²·c7))`, whose unit linear term keeps tiny angles exact.
Its f16 coefficients, $-0.16650390625$, $0.00824737548828125$ and $-0.00018310546875$, are the f16 neighbours of a minimax fit that hold every f16 input within 1 ulp.
Their reduction splits $\pi$ into three f16 parts, $3.140625$, $9.675 \cdot 10^{-4}$ and a third scaled by $2^{12}$.
Unscaled, that third part would be an f16 subnormal with 2 significant bits, costing 27 ulp next to the zero of cosine at $x = 177.5$, and dropping it costs 143 ulp there.
The f16 reduction holds 1 ulp only up to $|x| \le 256$, as the rounded f16 quotient $x / \pi$ drifts from the nearest multiple past it.
Past $2048\pi$ that multiple is no longer an f16 integer, and over all finite inputs the f16 reduction reaches 47104 ulp.
So a vector with any lane beyond 256 reduces in f32 with the two-part split of the f32-evaluated kernels, and narrows the reduced angle back for the same f16 polynomial.
Arc tangent divides in f16 and evaluates an odd degree-7 polynomial with coefficients $-0.328125$, $0.1600341796875$ and $-0.046722412109375$.
Its folded lanes add the low part of $\pi/2$ before the high part, because $\pi/2$ rounded to f16 alone is half an ulp off.

### Vectorized Polynomial Evaluation

`nk_trig_sin_f32_haswell`, `nk_trig_cos_f32_skylake` evaluate the same polynomial on 8 (AVX2) or 16 (AVX-512) elements simultaneously.
Range reduction, quadrant selection, and polynomial evaluation all operate on packed vectors — the only scalar operation is the final sign correction via `VBLENDVPS` with the quadrant mask.
`nk_trig_sin_f32_neon` processes 4 elements per iteration using `vfmaq_f32` for the Horner chain.
WASM v128relaxed (`nk_trig_sin_f32_v128relaxed`) uses `f32x4.relaxed_madd` for the FMA steps, achieving ~2x throughput over strict `f32x4.mul` + `f32x4.add` sequences.

## Accuracy

The f32 bounds come from every f32 input in the range, both signs, against the correctly rounded f64 result.
The f16 bounds come from every f16 input in the range.
The f64 bounds come from 180 thousand sampled inputs against a 200-bit reference.

| Kernel      | Input range      | Max error                            |
| :---------- | :--------------- | :----------------------------------- |
| f64 sin/cos | $[-2\pi, 2\pi]$  | 1.6 ulp of the exact value           |
| f64 atan    | $[-10, 10]$      | 1.6 ulp of the exact value           |
| f32 sin/cos | $\|x\| \le 10^4$ | 2 ulp, $1.25 \cdot 10^{-7}$ absolute |
| f32 atan    | all finite       | 3 ulp                                |
| f16 all     | all finite       | 1 ulp                                |

F64 results are not faithfully rounded: the worst of them sit 2 ulp from the correctly rounded value.
The f32 sine and cosine bound holds next to the zeros too, where the result is as small as $5 \cdot 10^{-14}$.
The FMA capabilities keep it up to $|x| \le 10^5$, and 3 ulp up to $10^6$.
F16 polynomial errors stay below half an f16 ulp, so the rounded result is at most one ulp from the correctly rounded value: 5012 of the 63488 finite inputs for sine, 4934 for cosine, and 544 for arc tangent.
The native f16 kernels hold the same 1 ulp bound over all finite inputs, with more results at one ulp: 7902 for sine, 11238 for cosine, and 2766 for arc tangent.
Those counts were measured on NEON FP16.
SVE FP16 and Sapphire Rapids run the same sequence of correctly rounded operations, so they should return the same results, but they have only been compiled.

The ulp figures in the tables below are mean errors recorded with earlier kernels, whose f32 sine and cosine lacked the reduction and coefficients above.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_DIMS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.

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
