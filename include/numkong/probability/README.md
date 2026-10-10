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

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |    0.758 gb/s, 5.65K ulp |    0.871 gb/s, 24.5K ulp |    0.849 gb/s, 98.9K ulp |
| `nk_jsd_f64_serial`  |      0.437 gb/s, 0.5 ulp |      0.448 gb/s, 0.3 ulp |      0.449 gb/s, 0.6 ulp |
| `nk_kld_f64_haswell` |     5.55 gb/s, 5.64K ulp |     5.33 gb/s, 24.6K ulp |     2.82 gb/s, 99.1K ulp |
| `nk_jsd_f64_haswell` |       3.20 gb/s, 1.7 ulp |       2.90 gb/s, 1.4 ulp |       1.98 gb/s, 1.2 ulp |
| `nk_kld_f64_skylake` |     7.84 gb/s, 5.64K ulp |     7.95 gb/s, 24.4K ulp |     5.65 gb/s, 98.9K ulp |
| `nk_jsd_f64_skylake` |       4.40 gb/s, 1.6 ulp |       4.18 gb/s, 1.4 ulp |       1.97 gb/s, 1.2 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |    0.840 gb/s, 1.04K ulp |    0.838 gb/s, 4.54K ulp |    0.829 gb/s, 18.2K ulp |
| `nk_jsd_f32_serial`  |      0.448 gb/s, 0.4 ulp |      0.454 gb/s, 0.4 ulp |      0.449 gb/s, 4.5 ulp |
| `nk_kld_f32_skylake` |     13.0 gb/s, 1.04K ulp |     13.3 gb/s, 4.55K ulp |     13.8 gb/s, 18.3K ulp |
| `nk_jsd_f32_skylake` |       7.10 gb/s, 6.6 ulp |       6.90 gb/s, 7.0 ulp |      6.59 gb/s, 11.1 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |    0.406 gb/s, 1.04K ulp |    0.360 gb/s, 4.53K ulp |    0.401 gb/s, 18.3K ulp |
| `nk_jsd_bf16_serial` |      0.222 gb/s, 1.5 ulp |      0.196 gb/s, 3.4 ulp |     0.219 gb/s, 10.7 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |    0.307 gb/s, 1.05K ulp |    0.266 gb/s, 4.53K ulp |    0.308 gb/s, 18.2K ulp |
| `nk_jsd_f16_serial`  |      0.182 gb/s, 1.5 ulp |      0.182 gb/s, 2.3 ulp |      0.181 gb/s, 9.4 ulp |
| `nk_kld_f16_haswell` |     6.96 gb/s, 1.05K ulp |     6.55 gb/s, 4.54K ulp |     6.66 gb/s, 18.2K ulp |
| `nk_jsd_f16_haswell` |       3.33 gb/s, 6.4 ulp |       3.38 gb/s, 6.8 ulp |      3.37 gb/s, 11.5 ulp |
| `nk_kld_f16_skylake` |     7.77 gb/s, 1.05K ulp |     6.58 gb/s, 4.54K ulp |     7.52 gb/s, 18.3K ulp |
| `nk_jsd_f16_skylake` |       4.07 gb/s, 6.5 ulp |       3.75 gb/s, 6.9 ulp |      4.01 gb/s, 11.4 ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |    0.878 gb/s, 5.64K ulp |    0.872 gb/s, 24.6K ulp |    0.881 gb/s, 99.6K ulp |
| `nk_jsd_f64_serial`  |      0.464 gb/s, 0.5 ulp |      0.467 gb/s, 0.3 ulp |      0.457 gb/s, 0.5 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |    0.756 gb/s, 1.04K ulp |    0.752 gb/s, 4.52K ulp |    0.753 gb/s, 18.3K ulp |
| `nk_jsd_f32_serial`  |      0.401 gb/s, 0.4 ulp |      0.404 gb/s, 0.4 ulp |      0.401 gb/s, 4.7 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |    0.373 gb/s, 1.05K ulp |    0.367 gb/s, 4.53K ulp |    0.362 gb/s, 18.3K ulp |
| `nk_jsd_bf16_serial` |      0.207 gb/s, 1.5 ulp |      0.206 gb/s, 3.1 ulp |      0.204 gb/s, 9.8 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |    0.285 gb/s, 1.04K ulp |    0.280 gb/s, 4.53K ulp |    0.281 gb/s, 18.3K ulp |
| `nk_jsd_f16_serial`  |      0.174 gb/s, 1.4 ulp |      0.174 gb/s, 2.6 ulp |      0.174 gb/s, 9.7 ulp |

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
