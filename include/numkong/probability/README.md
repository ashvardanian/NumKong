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

def kld(p: np.ndarray, q: np.ndarray, eps: float = 1e-7) -> float:
    return np.sum(p * np.log((p + eps) / (q + eps)))

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    return np.sqrt((kld(p, m) + kld(q, m)) / 2)
```

## Use Cases

__Kullback-Leibler divergence__ is widely used in variational inference (ELBO objective), knowledge distillation between neural networks, information gain in decision trees, and measuring fit between a model and observed data.

__Jensen-Shannon distance__ is commonly used in microbiome community comparison (enterotyping), where its metric property enables clustering with standard algorithms. It also appears in distribution drift detection, topic model evaluation, and as the theoretical foundation of the original GAN objective — though in practice GAN training uses proxy losses rather than computing JSD directly.

## Input & Output Types

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
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

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
The published tables below summarize mean ULP (units in last place) across all test pairs — the average number of representable floating-point values between the computed result and the exact answer. The current `nk_test` family also reports max/mean absolute and relative divergence error for detailed inspection.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |    0.693 gb/s, 5.65K ulp |    0.699 gb/s, 24.5K ulp |    0.753 gb/s, 98.9K ulp |
| `nk_jsd_f64_serial`  |      0.324 gb/s, 0.5 ulp |      0.349 gb/s, 0.3 ulp |      0.391 gb/s, 0.6 ulp |
| `nk_kld_f64_haswell` |     5.34 gb/s, 5.64K ulp |     5.59 gb/s, 24.6K ulp |     5.76 gb/s, 99.1K ulp |
| `nk_jsd_f64_haswell` |       3.03 gb/s, 1.7 ulp |       3.05 gb/s, 1.4 ulp |       3.25 gb/s, 1.2 ulp |
| `nk_kld_f64_skylake` |     7.01 gb/s, 5.64K ulp |     6.85 gb/s, 24.4K ulp |     6.86 gb/s, 98.9K ulp |
| `nk_jsd_f64_skylake` |       3.66 gb/s, 1.6 ulp |       3.85 gb/s, 1.4 ulp |       4.30 gb/s, 1.2 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |    0.528 gb/s, 1.04K ulp |    0.516 gb/s, 4.54K ulp |    0.527 gb/s, 18.2K ulp |
| `nk_jsd_f32_serial`  |      0.273 gb/s, 0.4 ulp |      0.272 gb/s, 0.4 ulp |      0.268 gb/s, 4.5 ulp |
| `nk_kld_f32_skylake` |     11.8 gb/s, 1.04K ulp |     10.4 gb/s, 4.55K ulp |     8.73 gb/s, 18.3K ulp |
| `nk_jsd_f32_skylake` |       6.25 gb/s, 6.6 ulp |       5.96 gb/s, 7.0 ulp |      6.05 gb/s, 11.1 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |    0.138 gb/s, 1.04K ulp |    0.142 gb/s, 4.53K ulp |    0.136 gb/s, 18.3K ulp |
| `nk_jsd_bf16_serial` |     0.0857 gb/s, 1.5 ulp |     0.0842 gb/s, 3.4 ulp |    0.0841 gb/s, 10.7 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |    0.166 gb/s, 1.05K ulp |    0.163 gb/s, 4.53K ulp |    0.163 gb/s, 18.2K ulp |
| `nk_jsd_f16_serial`  |      0.151 gb/s, 1.5 ulp |      0.148 gb/s, 2.3 ulp |      0.152 gb/s, 9.4 ulp |
| `nk_kld_f16_haswell` |     6.99 gb/s, 1.05K ulp |     6.09 gb/s, 4.54K ulp |     6.97 gb/s, 18.2K ulp |
| `nk_jsd_f16_haswell` |       2.81 gb/s, 6.4 ulp |       2.79 gb/s, 6.8 ulp |      2.72 gb/s, 11.5 ulp |
| `nk_kld_f16_skylake` |     6.16 gb/s, 1.05K ulp |     5.65 gb/s, 4.54K ulp |     5.78 gb/s, 18.3K ulp |
| `nk_jsd_f16_skylake` |       3.51 gb/s, 6.5 ulp |       3.22 gb/s, 6.9 ulp |      3.35 gb/s, 11.4 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |    0.239 gb/s, 5.64K ulp |    0.223 gb/s, 24.6K ulp |     0.13 gb/s, 99.6K ulp |
| `nk_jsd_f64_serial`  |      0.315 gb/s, 0.5 ulp |      0.402 gb/s, 0.3 ulp |       0.29 gb/s, 0.5 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |    0.302 gb/s, 1.04K ulp |    0.342 gb/s, 4.52K ulp |    0.277 gb/s, 18.3K ulp |
| `nk_jsd_f32_serial`  |      0.152 gb/s, 0.4 ulp |      0.164 gb/s, 0.4 ulp |      0.160 gb/s, 4.7 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |    0.139 gb/s, 1.05K ulp |    0.143 gb/s, 4.53K ulp |    0.150 gb/s, 18.3K ulp |
| `nk_jsd_bf16_serial` |     0.0867 gb/s, 1.5 ulp |     0.0775 gb/s, 3.1 ulp |     0.0679 gb/s, 9.8 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |    0.118 gb/s, 1.04K ulp |    0.127 gb/s, 4.53K ulp |    0.111 gb/s, 18.3K ulp |
| `nk_jsd_f16_serial`  |     0.0748 gb/s, 1.4 ulp |     0.0681 gb/s, 2.6 ulp |     0.0857 gb/s, 9.7 ulp |

### Apple M5

#### Native

| Kernel                |                      256 |                     1024 |                     4096 |
| :-------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`   |      3.22 gb/s, 5.6K ulp |       3.36 gb/s, 25K ulp |       3.32 gb/s, 99K ulp |
| `nk_jsd_f64_serial`   |       2.06 gb/s, 0.4 ulp |       2.17 gb/s, 0.4 ulp |       2.17 gb/s, 0.5 ulp |
| __f32__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`   |      9.26 gb/s, 1.0K ulp |      8.73 gb/s, 4.5K ulp |       9.10 gb/s, 18K ulp |
| `nk_jsd_f32_serial`   |       2.08 gb/s, 0.4 ulp |       2.16 gb/s, 0.4 ulp |       2.13 gb/s, 4.6 ulp |
| `nk_kld_f32_neon`     |      19.0 gb/s, 1.0K ulp |      17.4 gb/s, 4.5K ulp |       18.1 gb/s, 18K ulp |
| `nk_jsd_f32_neon`     |        9.75 gb/s, 15 ulp |        9.32 gb/s, 14 ulp |       9.62 gb/s, 9.9 ulp |
| __bf16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial`  |      4.58 gb/s, 1.0K ulp |      4.47 gb/s, 4.5K ulp |       4.65 gb/s, 18K ulp |
| `nk_jsd_bf16_serial`  |       1.08 gb/s, 1.4 ulp |       1.07 gb/s, 2.9 ulp |       1.09 gb/s, 9.7 ulp |
| __f16__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`   |      4.63 gb/s, 1.0K ulp |      4.45 gb/s, 4.5K ulp |       4.55 gb/s, 18K ulp |
| `nk_jsd_f16_serial`   |       1.03 gb/s, 1.4 ulp |      0.962 gb/s, 2.7 ulp |      0.976 gb/s, 8.7 ulp |
| `nk_kld_f16_neonhalf` |      10.2 gb/s, 1.0K ulp |      9.67 gb/s, 4.5K ulp |       9.99 gb/s, 18K ulp |
| `nk_jsd_f16_neonhalf` |        5.00 gb/s, 15 ulp |        4.79 gb/s, 14 ulp |       4.94 gb/s, 9.9 ulp |
