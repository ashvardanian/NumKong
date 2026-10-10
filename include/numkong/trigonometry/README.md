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
Sine and cosine evaluate r + r³ · (c3 + r² · (c5 + r² · c7)), whose unit linear term keeps tiny angles exact.
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

| Kernel      | Input range     | Max error                  |           |                                      |
| :---------- | :-------------- | :------------------------- | --------- | ------------------------------------ |
| f64 sin/cos | $[-2\pi, 2\pi]$ | 1.6 ulp of the exact value |           |                                      |
| f64 atan    | $[-10, 10]$     | 1.6 ulp of the exact value |           |                                      |
| f32 sin/cos | $\              | x\                         | \le 10^4$ | 2 ulp, $1.25 \cdot 10^{-7}$ absolute |
| f32 atan    | all finite      | 3 ulp                      |           |                                      |
| f16 all     | all finite      | 1 ulp                      |           |                                      |

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
The input size is controlled by the `NUMWARS_BATCH_PER_CORE` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                      |                      256 |                     1024 |                     4096 |
| :-------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `trig_sin_f64_stl` 🧩       |               0.413 gb/s |               0.408 gb/s |               0.410 gb/s |
| `trig_cos_f64_stl` 🧩       |               0.447 gb/s |               0.446 gb/s |               0.442 gb/s |
| `trig_atan_f64_stl` 🧩      |               0.742 gb/s |               0.738 gb/s |               0.726 gb/s |
| `nk_trig_sin_f64_serial`    |        0.205 gb/s, 0 ulp |        0.206 gb/s, 0 ulp |        0.190 gb/s, 0 ulp |
| `nk_trig_cos_f64_serial`    |        0.202 gb/s, 0 ulp |        0.201 gb/s, 0 ulp |        0.193 gb/s, 0 ulp |
| `nk_trig_atan_f64_serial`   |       0.0572 gb/s, 0 ulp |       0.0572 gb/s, 0 ulp |       0.0558 gb/s, 0 ulp |
| `nk_trig_sin_f64_haswell`   |         5.26 gb/s, 0 ulp |         6.59 gb/s, 0 ulp |         5.68 gb/s, 0 ulp |
| `nk_trig_cos_f64_haswell`   |         5.04 gb/s, 0 ulp |         6.26 gb/s, 0 ulp |         5.34 gb/s, 0 ulp |
| `nk_trig_atan_f64_haswell`  |         3.75 gb/s, 0 ulp |         4.30 gb/s, 0 ulp |         3.50 gb/s, 0 ulp |
| `nk_trig_sin_f64_skylake`   |         8.04 gb/s, 0 ulp |         10.3 gb/s, 0 ulp |         9.29 gb/s, 0 ulp |
| `nk_trig_cos_f64_skylake`   |         7.75 gb/s, 0 ulp |         9.96 gb/s, 0 ulp |         8.80 gb/s, 0 ulp |
| `nk_trig_atan_f64_skylake`  |         5.60 gb/s, 0 ulp |         7.18 gb/s, 0 ulp |         5.78 gb/s, 0 ulp |
| __f32__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `trig_sin_f32_stl` 🧩       |               0.416 gb/s |               0.413 gb/s |               0.408 gb/s |
| `trig_cos_f32_stl` 🧩       |               0.404 gb/s |               0.402 gb/s |               0.398 gb/s |
| `trig_atan_f32_stl` 🧩      |               0.382 gb/s |               0.373 gb/s |               0.380 gb/s |
| `nk_trig_sin_f32_serial`    |        0.337 gb/s, 5 ulp |        0.321 gb/s, 5 ulp |        0.311 gb/s, 5 ulp |
| `nk_trig_cos_f32_serial`    |       0.333 gb/s, 15 ulp |       0.272 gb/s, 15 ulp |       0.313 gb/s, 15 ulp |
| `nk_trig_atan_f32_serial`   |     0.0854 gb/s, 0.4 ulp |     0.0741 gb/s, 0.4 ulp |     0.0809 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_haswell`   |         8.36 gb/s, 5 ulp |         9.30 gb/s, 5 ulp |         9.78 gb/s, 5 ulp |
| `nk_trig_cos_f32_haswell`   |        7.92 gb/s, 15 ulp |        8.78 gb/s, 15 ulp |        8.48 gb/s, 15 ulp |
| `nk_trig_atan_f32_haswell`  |       6.46 gb/s, 0.4 ulp |       7.08 gb/s, 0.4 ulp |       6.96 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_skylake`   |         12.2 gb/s, 5 ulp |         11.8 gb/s, 5 ulp |         11.6 gb/s, 5 ulp |
| `nk_trig_cos_f32_skylake`   |        11.6 gb/s, 15 ulp |        11.4 gb/s, 15 ulp |        11.5 gb/s, 15 ulp |
| `nk_trig_atan_f32_skylake`  |       10.5 gb/s, 0.4 ulp |       10.7 gb/s, 0.4 ulp |       10.6 gb/s, 0.4 ulp |
| __f16__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f16_serial`    |     0.0776 gb/s, 0.9 ulp |     0.0783 gb/s, 1.1 ulp |     0.0747 gb/s, 0.9 ulp |
| `nk_trig_cos_f16_serial`    |      0.0756 gb/s, 12 ulp |      0.0755 gb/s, 12 ulp |      0.0722 gb/s, 12 ulp |
| `nk_trig_atan_f16_serial`   |     0.0944 gb/s, 6.4 ulp |      0.093 gb/s, 6.7 ulp |     0.0886 gb/s, 6.6 ulp |
| `nk_trig_sin_f16_skylake`   |     9.14 gb/s, 8.41K ulp |     6.75 gb/s, 8.43K ulp |     8.28 gb/s, 8.41K ulp |
| `nk_trig_cos_f16_skylake`   |     8.46 gb/s, 8.34K ulp |     6.38 gb/s, 8.34K ulp |     7.38 gb/s, 8.35K ulp |
| `nk_trig_atan_f16_skylake`  |     7.10 gb/s, 16.5K ulp |     5.80 gb/s, 16.6K ulp |     6.63 gb/s, 16.5K ulp |
| `nk_trig_sin_f16_sapphire`  |                15.5 gb/s |                9.35 gb/s |                12.0 gb/s |
| `nk_trig_cos_f16_sapphire`  |                11.8 gb/s |                8.20 gb/s |                10.7 gb/s |
| `nk_trig_atan_f16_sapphire` |                5.36 gb/s |                5.35 gb/s |                5.32 gb/s |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `trig_sin_f64_stl` 🧩          |               0.266 gb/s |               0.249 gb/s |               0.246 gb/s |
| `trig_cos_f64_stl` 🧩          |               0.309 gb/s |               0.247 gb/s |               0.247 gb/s |
| `trig_atan_f64_stl` 🧩         |               0.518 gb/s |               0.502 gb/s |               0.478 gb/s |
| `nk_trig_sin_f64_serial`       |      0.197 gb/s, 0.2 ulp |      0.198 gb/s, 0.2 ulp |      0.197 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_serial`       |      0.194 gb/s, 0.3 ulp |      0.195 gb/s, 0.3 ulp |      0.193 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_serial`      |     0.0543 gb/s, 0.3 ulp |     0.0544 gb/s, 0.3 ulp |     0.0543 gb/s, 0.3 ulp |
| `nk_trig_sin_f64_v128relaxed`  |       3.17 gb/s, 0.2 ulp |       3.19 gb/s, 0.2 ulp |       3.23 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_v128relaxed`  |       3.29 gb/s, 0.3 ulp |       3.32 gb/s, 0.3 ulp |       3.34 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_v128relaxed` |       1.74 gb/s, 0.3 ulp |       1.79 gb/s, 0.3 ulp |       1.76 gb/s, 0.3 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `trig_sin_f32_stl` 🧩          |               0.159 gb/s |               0.158 gb/s |               0.158 gb/s |
| `trig_cos_f32_stl` 🧩          |               0.161 gb/s |               0.157 gb/s |               0.155 gb/s |
| `trig_atan_f32_stl` 🧩         |               0.276 gb/s |               0.274 gb/s |               0.273 gb/s |
| `nk_trig_sin_f32_serial`       |      0.306 gb/s, 4.9 ulp |      0.305 gb/s, 4.9 ulp |      0.304 gb/s, 4.9 ulp |
| `nk_trig_cos_f32_serial`       |     0.305 gb/s, 14.4 ulp |     0.304 gb/s, 14.4 ulp |     0.303 gb/s, 14.4 ulp |
| `nk_trig_atan_f32_serial`      |     0.0806 gb/s, 0.4 ulp |     0.0809 gb/s, 0.4 ulp |     0.0811 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_v128relaxed`  |      4.56 gb/s, 20.7 ulp |      4.73 gb/s, 20.7 ulp |      4.83 gb/s, 20.7 ulp |
| `nk_trig_cos_f32_v128relaxed`  |      4.33 gb/s, 21.9 ulp |      4.39 gb/s, 21.9 ulp |      4.46 gb/s, 21.9 ulp |
| `nk_trig_atan_f32_v128relaxed` |       3.22 gb/s, 0.4 ulp |       3.44 gb/s, 0.4 ulp |       3.49 gb/s, 0.4 ulp |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f16_serial`       |     0.0736 gb/s, 1.1 ulp |     0.0738 gb/s, 1.1 ulp |     0.0734 gb/s, 1.1 ulp |
| `nk_trig_cos_f16_serial`       |    0.0732 gb/s, 11.8 ulp |    0.0734 gb/s, 11.8 ulp |    0.0731 gb/s, 11.8 ulp |
| `nk_trig_atan_f16_serial`      |     0.0832 gb/s, 6.5 ulp |     0.0824 gb/s, 6.5 ulp |     0.0823 gb/s, 6.5 ulp |

### Apple M4

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f64_serial`  |      0.584 gb/s, 0.2 ulp |      0.590 gb/s, 0.2 ulp |      0.595 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_serial`  |      0.578 gb/s, 0.3 ulp |      0.589 gb/s, 0.3 ulp |      0.576 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_serial` |      0.142 gb/s, 0.3 ulp |      0.143 gb/s, 0.3 ulp |      0.142 gb/s, 0.3 ulp |
| `nk_trig_sin_f64_neon`    |       5.53 gb/s, 0.2 ulp |       5.36 gb/s, 0.2 ulp |       5.42 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_neon`    |       4.80 gb/s, 0.3 ulp |       4.99 gb/s, 0.3 ulp |       5.00 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_neon`   |       3.29 gb/s, 0.3 ulp |       3.26 gb/s, 0.3 ulp |       3.26 gb/s, 0.3 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f32_serial`  |       7.39 gb/s, 4.9 ulp |       7.39 gb/s, 4.9 ulp |       6.73 gb/s, 4.9 ulp |
| `nk_trig_cos_f32_serial`  |        6.76 gb/s, 14 ulp |        5.97 gb/s, 14 ulp |        6.07 gb/s, 14 ulp |
| `nk_trig_atan_f32_serial` |      0.119 gb/s, 0.4 ulp |      0.120 gb/s, 0.4 ulp |      0.117 gb/s, 0.4 ulp |
| `nk_trig_sin_f32_neon`    |       9.08 gb/s, 4.9 ulp |       8.79 gb/s, 4.9 ulp |       7.57 gb/s, 4.9 ulp |
| `nk_trig_cos_f32_neon`    |        8.08 gb/s, 18 ulp |        7.24 gb/s, 18 ulp |        7.30 gb/s, 18 ulp |
| `nk_trig_atan_f32_neon`   |       5.19 gb/s, 0.4 ulp |       4.66 gb/s, 0.4 ulp |       4.75 gb/s, 0.4 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f16_serial`  |       3.41 gb/s, 1.3 ulp |       3.46 gb/s, 1.3 ulp |       3.15 gb/s, 1.3 ulp |
| `nk_trig_cos_f16_serial`  |        3.05 gb/s, 12 ulp |        3.06 gb/s, 12 ulp |        2.93 gb/s, 12 ulp |
| `nk_trig_atan_f16_serial` |     0.0595 gb/s, 6.5 ulp |     0.0583 gb/s, 6.5 ulp |     0.0584 gb/s, 6.5 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f64_serial`       |      0.718 gb/s, 0.2 ulp |      0.683 gb/s, 0.2 ulp |      0.743 gb/s, 0.2 ulp |
| `nk_trig_sin_f64_v128relaxed`  |       8.06 gb/s, 0.2 ulp |       7.82 gb/s, 0.2 ulp |       8.80 gb/s, 0.2 ulp |
| `nk_trig_cos_f64_serial`       |      0.695 gb/s, 0.3 ulp |      0.655 gb/s, 0.3 ulp |      0.685 gb/s, 0.3 ulp |
| `nk_trig_cos_f64_v128relaxed`  |       8.45 gb/s, 0.3 ulp |       8.20 gb/s, 0.3 ulp |       8.92 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_serial`      |      0.230 gb/s, 0.3 ulp |      0.218 gb/s, 0.3 ulp |      0.216 gb/s, 0.3 ulp |
| `nk_trig_atan_f64_v128relaxed` |       5.13 gb/s, 0.3 ulp |       4.85 gb/s, 0.3 ulp |       5.55 gb/s, 0.3 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_trig_sin_f32_serial`       |       9.26 gb/s, 4.9 ulp |       8.79 gb/s, 4.9 ulp |       9.50 gb/s, 4.9 ulp |
| `nk_trig_sin_f32_v128relaxed`  |        16.6 gb/s, 20 ulp |        16.3 gb/s, 20 ulp |        16.7 gb/s, 20 ulp |
| `nk_trig_cos_f32_serial`       |        8.50 gb/s, 14 ulp |        8.34 gb/s, 14 ulp |        8.89 gb/s, 14 ulp |
| `nk_trig_cos_f32_v128relaxed`  |        15.6 gb/s, 21 ulp |        13.7 gb/s, 21 ulp |        15.7 gb/s, 21 ulp |
| `nk_trig_atan_f32_serial`      |      0.178 gb/s, 0.4 ulp |      0.172 gb/s, 0.4 ulp |      0.185 gb/s, 0.4 ulp |
| `nk_trig_atan_f32_v128relaxed` |       10.4 gb/s, 0.4 ulp |       9.97 gb/s, 0.4 ulp |       10.8 gb/s, 0.4 ulp |
