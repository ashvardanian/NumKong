# Divergence Measures for Probability Distributions in NumKong

NumKong implements divergence functions between discrete probability distributions: Kullback-Leibler divergence measures the information lost when one distribution approximates another, while Jensen-Shannon distance provides a symmetric and bounded alternative.
These are used in variational inference, topic modeling, and distribution comparison tasks.

Kullback-Leibler divergence from $P$ to $Q$:

$$
\text{KLD}(P \| Q) = \sum_{i=0}^{n-1} P(i) \ln \frac{P(i)}{Q(i)}
$$

Jensen-Shannon distance is the square root of the symmetrized KLD through a mixture:

$$
\text{JSD}(P, Q) = \frac{1}{2} \text{KLD}(P \| M) + \frac{1}{2} \text{KLD}(Q \| M)
$$

where $M = \frac{P + Q}{2}$, yielding the distance:

$$
d_{JS}(P, Q) = \sqrt{\text{JSD}(P, Q)}
$$

Unlike the raw divergence, $d_{JS}$ is a true metric satisfying the triangle inequality.

Reformulating as Python pseudocode:

```python
import numpy as np

def kld(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-7) -> float:
    return np.sum(p * np.log(np.maximum(p, epsilon) / np.maximum(q, epsilon)))

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    return np.sqrt((kld(p, m) + kld(q, m)) / 2)
```

Operands are clamped to at least `epsilon`, so terms at or above it follow the exact formula, while zero entries contribute nothing instead of hitting $\log 0$ or $0/0$.

## Use Cases

__Kullback-Leibler divergence__ is widely used in variational inference (ELBO objective), knowledge distillation between neural networks, information gain in decision trees, and measuring fit between a model and observed data.

__Jensen-Shannon distance__ is commonly used in microbiome community comparison (enterotyping), where its metric property enables clustering with standard algorithms.
It also appears in distribution drift detection, topic model evaluation, and as the theoretical foundation of the original GAN objective — though in practice GAN training uses proxy losses rather than computing JSD directly.

## Input & Output Types

| Input Type | Output Type | Description                                      |
| :--------- | :---------- | :----------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output   |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |

## Optimizations

### SIMD Log2 Approximation

`nk_kld_f32_skylake`, `nk_jsd_f32_skylake` use `VGETEXP` and `VGETMANT` to decompose floating-point values into exponent and mantissa components, then apply a polynomial approximation to the mantissa to compute $\log_2$.
The pipeline on Skylake is:

```
exponent = VGETEXPPS(x)
mantissa = VGETMANTPS(x, normalize_to_[1,2))
s        = (mantissa - 1) / (mantissa + 1)
log2(x)  ≈ exponent + (2/ln2) · s · (1 + s²/3 + s⁴/5 + s⁶/7 + s⁸/9)
```

`VGETEXP` extracts the unbiased exponent as a float, while `VGETMANT` normalizes the mantissa to $[1, 2)$.
A five-term odd series in $s$ — the $\operatorname{atanh}$ expansion, which converges quickly because $s \in [0, 1/3]$ — completes the approximation.
These instructions handle subnormals correctly without extra integer bit manipulation.

`nk_kld_f32_neon`, `nk_jsd_f32_neon`, `nk_kld_f16_haswell`, `nk_jsd_f16_haswell` use integer bit extraction instead:

```
exponent = (reinterpret_as_int(x) >> 23) - 127
mantissa = reinterpret_as_float((reinterpret_as_int(x) & 0x7FFFFF) | 0x3F800000)
log2(x) ≈ exponent + series_or_polynomial(mantissa)
```

This approach reinterprets the float as an integer, shifts out the mantissa bits to obtain the exponent, then masks and recombines to produce a normalized mantissa in $[1, 2)$.
The Haswell kernels then reuse the same $s$-series as Skylake, while the NEON kernels evaluate a degree-5 polynomial in the mantissa scaled by $m - 1$.
It works on any ISA with integer-float reinterpretation and avoids the need for specialized exponent/mantissa instructions.
The SIMD kernels scale the accumulated $\log_2$ sum by $\ln 2$ before returning, while the serial ones call a natural-log helper directly — either way the reported divergence is in nats, not bits.

### Kahan Compensated Summation for Float64

`nk_kld_f64_haswell`, `nk_jsd_f64_haswell` use Kahan compensated summation to maintain a running correction term alongside the accumulator.
The Kahan update for each divergence term is:

```
compensated_term = divergence_term - correction
tentative_sum    = accumulator + compensated_term
correction       = (tentative_sum - accumulator) - compensated_term
accumulator      = tentative_sum
```

After each $P(i) \log_2(P(i) / Q(i))$ term is computed, `correction` captures the low-order bits lost in the addition, and the next iteration subtracts this correction from the new term before adding it to the accumulator.
This keeps the accumulated error bounded by $O(1)$ ULP regardless of vector length, rather than the $O(n)$ ULP growth of naive summation.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_DIMS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
The published tables below summarize mean ULP (units in last place) across all test pairs — the average number of representable floating-point values between the computed result and the exact answer.
The current `numkong_cpu_test` family also reports max/mean absolute and relative divergence error for detailed inspection.

### Intel Sapphire Rapids

#### Native

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |    0.645 gb/s, 5.65K ulp |    0.651 gb/s, 24.5K ulp |    0.701 gb/s, 98.9K ulp |
| `nk_jsd_f64_serial`  |      0.302 gb/s, 0.5 ulp |      0.325 gb/s, 0.3 ulp |      0.364 gb/s, 0.6 ulp |
| `nk_kld_f64_haswell` |     4.97 gb/s, 5.64K ulp |     5.21 gb/s, 24.6K ulp |     5.36 gb/s, 99.1K ulp |
| `nk_jsd_f64_haswell` |       2.82 gb/s, 1.7 ulp |       2.84 gb/s, 1.4 ulp |       3.03 gb/s, 1.2 ulp |
| `nk_kld_f64_skylake` |     6.53 gb/s, 5.64K ulp |     6.38 gb/s, 24.4K ulp |     6.39 gb/s, 98.9K ulp |
| `nk_jsd_f64_skylake` |       3.41 gb/s, 1.6 ulp |       3.59 gb/s, 1.4 ulp |       4.00 gb/s, 1.2 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |    0.492 gb/s, 1.04K ulp |    0.481 gb/s, 4.54K ulp |    0.491 gb/s, 18.2K ulp |
| `nk_jsd_f32_serial`  |      0.254 gb/s, 0.4 ulp |      0.253 gb/s, 0.4 ulp |      0.250 gb/s, 4.5 ulp |
| `nk_kld_f32_skylake` |     11.0 gb/s, 1.04K ulp |     9.69 gb/s, 4.55K ulp |     8.13 gb/s, 18.3K ulp |
| `nk_jsd_f32_skylake` |       5.82 gb/s, 6.6 ulp |       5.55 gb/s, 7.0 ulp |      5.63 gb/s, 11.1 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |    0.129 gb/s, 1.04K ulp |    0.132 gb/s, 4.53K ulp |    0.127 gb/s, 18.3K ulp |
| `nk_jsd_bf16_serial` |     0.0798 gb/s, 1.5 ulp |     0.0784 gb/s, 3.4 ulp |    0.0783 gb/s, 10.7 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |    0.155 gb/s, 1.05K ulp |    0.152 gb/s, 4.53K ulp |    0.152 gb/s, 18.2K ulp |
| `nk_jsd_f16_serial`  |      0.141 gb/s, 1.5 ulp |      0.138 gb/s, 2.3 ulp |      0.142 gb/s, 9.4 ulp |
| `nk_kld_f16_haswell` |     6.51 gb/s, 1.05K ulp |     5.67 gb/s, 4.54K ulp |     6.49 gb/s, 18.2K ulp |
| `nk_jsd_f16_haswell` |       2.62 gb/s, 6.4 ulp |       2.60 gb/s, 6.8 ulp |      2.53 gb/s, 11.5 ulp |
| `nk_kld_f16_skylake` |     5.74 gb/s, 1.05K ulp |     5.26 gb/s, 4.54K ulp |     5.38 gb/s, 18.3K ulp |
| `nk_jsd_f16_skylake` |       3.27 gb/s, 6.5 ulp |       3.00 gb/s, 6.9 ulp |      3.12 gb/s, 11.4 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |    0.223 gb/s, 5.64K ulp |    0.208 gb/s, 24.6K ulp |     0.12 gb/s, 99.6K ulp |
| `nk_jsd_f64_serial`  |      0.293 gb/s, 0.5 ulp |      0.374 gb/s, 0.3 ulp |       0.27 gb/s, 0.5 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |    0.281 gb/s, 1.04K ulp |    0.319 gb/s, 4.52K ulp |    0.258 gb/s, 18.3K ulp |
| `nk_jsd_f32_serial`  |      0.142 gb/s, 0.4 ulp |      0.153 gb/s, 0.4 ulp |      0.149 gb/s, 4.7 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |    0.129 gb/s, 1.05K ulp |    0.133 gb/s, 4.53K ulp |    0.140 gb/s, 18.3K ulp |
| `nk_jsd_bf16_serial` |     0.0807 gb/s, 1.5 ulp |     0.0722 gb/s, 3.1 ulp |     0.0632 gb/s, 9.8 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |    0.110 gb/s, 1.04K ulp |    0.118 gb/s, 4.53K ulp |    0.103 gb/s, 18.3K ulp |
| `nk_jsd_f16_serial`  |     0.0697 gb/s, 1.4 ulp |     0.0634 gb/s, 2.6 ulp |     0.0798 gb/s, 9.7 ulp |

### Apple M5

#### Native

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |      3.00 gb/s, 5.6K ulp |       3.13 gb/s, 25K ulp |       3.09 gb/s, 99K ulp |
| `nk_jsd_f64_serial`  |       1.92 gb/s, 0.4 ulp |       2.02 gb/s, 0.4 ulp |       2.02 gb/s, 0.5 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |      8.62 gb/s, 1.0K ulp |      8.13 gb/s, 4.5K ulp |       8.48 gb/s, 18K ulp |
| `nk_jsd_f32_serial`  |       1.94 gb/s, 0.4 ulp |       2.01 gb/s, 0.4 ulp |       1.98 gb/s, 4.6 ulp |
| `nk_kld_f32_neon`    |      17.7 gb/s, 1.0K ulp |      16.2 gb/s, 4.5K ulp |       16.9 gb/s, 18K ulp |
| `nk_jsd_f32_neon`    |        9.08 gb/s, 15 ulp |        8.68 gb/s, 14 ulp |       8.96 gb/s, 9.9 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |      4.27 gb/s, 1.0K ulp |      4.16 gb/s, 4.5K ulp |       4.33 gb/s, 18K ulp |
| `nk_jsd_bf16_serial` |       1.01 gb/s, 1.4 ulp |      0.997 gb/s, 2.9 ulp |       1.02 gb/s, 9.7 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |      4.31 gb/s, 1.0K ulp |      4.14 gb/s, 4.5K ulp |       4.24 gb/s, 18K ulp |
| `nk_jsd_f16_serial`  |      0.959 gb/s, 1.4 ulp |      0.896 gb/s, 2.7 ulp |      0.909 gb/s, 8.7 ulp |
| `nk_kld_f16_neon`    |      9.50 gb/s, 1.0K ulp |      9.01 gb/s, 4.5K ulp |       9.30 gb/s, 18K ulp |
| `nk_jsd_f16_neon`    |        4.66 gb/s, 15 ulp |        4.46 gb/s, 14 ulp |       4.60 gb/s, 9.9 ulp |
