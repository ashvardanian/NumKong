# MaxSim Late-Interaction Scoring in NumKong

NumKong implements ColBERT-style late-interaction scoring: the MaxSim score sums, over each query token, the minimum angular distance to any document token. A two-stage coarse-to-fine strategy uses i8-quantized screening to find the best document per query, then full-precision refinement computes the final angular distance.

MaxSim score:

$$
\text{MaxSim}(Q, D) = \sum_{i=0}^{m-1} \min_{j=0}^{n-1} \text{angular}(q_i, d_j)
$$

Coarse screening finds the best document via i8 dot products as a proxy for argmin angular:

$$
j^* = \arg\max_j \text{dot}_{\text{i8}}(q_i, d_j)
$$

Full-precision refinement:

$$
\text{angular}(q_i, d_{j^*}) = 1 - \frac{\text{dot}(q_i, d_{j^*})}{\|q_i\| \cdot \|d_{j^*}\|}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def maxsim(queries: np.ndarray, documents: np.ndarray) -> float:
    score = 0.0
    for q in queries:
        dots = documents @ q
        best = np.argmax(dots)
        d = documents[best]
        angular = 1 - np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
        score += angular
    return score
```

## Input & Output Types

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision                   |

## Optimizations

### Dual Pre-Packing

`nk_maxsim_packed_bf16_sme`, `nk_maxsim_packed_f32_sme` benefit from having _both_ query and document matrices pre-packed into identical contiguous formats, unlike the `nk_dots_packed_*` family where only B is pre-packed and A is accessed with arbitrary stride.
In the dots GEMM, one ZA tile must be reserved for A-side staging (loading unpacked A rows into the tile array), leaving 3 ZA tiles for accumulation.
With both sides pre-packed, all 4 ZA tiles (ZA0–ZA3) serve as accumulators — a +33% increase in MOPA throughput.
No output matrix materialization: dots_packed writes a full M×N f32 result matrix, while maxsim reduces each query row to a single argmax index in-flight, eliminating the M×N memory round-trip.
Benchmark data (Apple M4, SVL=512):

| Dimensions           | dots_packed GEMM | maxsim fused | GEMM speedup | End-to-end |
| -------------------- | ---------------: | -----------: | -----------: | ---------: |
| 32×128×128 (ColBERT) |       840 GFLOPS |  1516 GFLOPS |        1.81× |      5.10× |
| 32×256×128           |      1037 GFLOPS |  1591 GFLOPS |        1.53× |      5.17× |
| 64×512×128           |      1016 GFLOPS |  1651 GFLOPS |        1.62× |      5.42× |
| 32×128×256           |       859 GFLOPS |  1725 GFLOPS |        2.01× |      4.06× |
| 32×1024×768 (BERT)   |      1124 GFLOPS |  1932 GFLOPS |        1.72× |      2.61× |

End-to-end speedup (5×) exceeds GEMM-only speedup (1.5–2×) because maxsim eliminates output materialization and fuses argmax+angular refinement into the tile extraction loop.

### Two-Stage Coarse-to-Fine Scoring

All backends use i8-quantized coarse screening at O(m·n·k) with 1 byte/element instead of 2–4, followed by full-precision refinement at O(m·k) for only the winning pairs.
Break-even at ~4 documents per query — beyond that, coarse screening dominates and the i8 bandwidth advantage compounds.

### ISA-Specific Quantization Ranges

Haswell uses [-79, 79] — `VPMADDUBSW` produces i16 intermediates, must avoid saturation (2×depth×79 < 32767).
Alder Lake and Ice Lake use [-127, 127] — `VPDPBUSD` accumulates directly to i32, no i16 bottleneck.
WASM v128relaxed uses [-63, 63] — `i32x4_relaxed_dot_i8x16_i7x16_add` requires 7-bit operands.
Serial uses [-127, 127].

### XOR-0x80 Bias Correction

`nk_maxsim_packed_bf16_haswell`, `nk_maxsim_packed_f32_alder`, `nk_maxsim_packed_f32_icelake` work around the unsigned×signed operand requirement of `VPMADDUBSW` and `VPDPBUSD`.
Both query and document are signed after quantization, so queries are XOR'd with `0x80` to shift to unsigned range.
Post-multiply correction subtracts $128 \cdot \text{sum\_i8}(d_j)$ per document, where sums are precomputed in packed metadata.

### Vertical Column Extraction on SME

`nk_maxsim_packed_bf16_sme`, `nk_maxsim_packed_f32_sme` accumulate Q×D dot products into ZA tiles (4 tiles ZA0–ZA3, each SVL×SVL).
The argmax operation needs to find the best document for each query.
The naive approach reads rows horizontally (`svread_hor_za32`) and reduces each row with `svmaxv` — but `svmaxv` is a horizontal reduction costing ~8 cycles on typical SVE implementations.
Vertical column extraction flips the access pattern: `svread_ver_za32_f32_m` reads one _column_ of ZA, returning one dot-product score per query for a single document.
Element-wise `svcmpgt_f32` + `svsel_f32` (~1 cycle each) update the running maximum across all queries simultaneously.
For 32 queries × 256 documents: horizontal approach = 32 × 256 × `svmaxv` = 8,192 horizontal reductions; vertical approach = 256 column reads × 1 element-wise `svmax` = 256 vertical reads + 256 comparisons (~270 cycles vs ~2,048 cycles for the argmax phase alone).
The argmax index is tracked in-flight using `svsel` to conditionally update an index vector alongside the maximum values — no separate argmax pass needed.
After finding the best document index per query, full-precision angular refinement uses the originals stored in the packed buffer's third region.

### Three-Region Packed Buffer

All backends use a three-region packed buffer layout: [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals, 64B-aligned].
Per-vector metadata (12 bytes) stores quantization scale, i8 sum (for bias correction), and inverse norm (for angular finalization).
The originals region stores full-precision vectors for refinement via existing `nk_dot_*` primitives.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for late-interaction scoring over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga Scalar Operations per Second, with $\text{ops} = 2 \cdot M \cdot N \cdot K$ complexity for scoring $M$ query tokens against $N$ document tokens of dimension $K$.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                              |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f32__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f32_serial`       |    15.7 gso/s, 48.9K ulp |    15.2 gso/s, 48.9K ulp |    16.3 gso/s, 48.9K ulp |
| `nk_maxsim_packed_f32_haswell`      |    77.2 gso/s, 49.3K ulp |    70.7 gso/s, 49.3K ulp |    74.5 gso/s, 49.3K ulp |
| `nk_maxsim_packed_f32_alder`        |    99.7 gso/s, 48.9K ulp |    97.7 gso/s, 48.9K ulp |    94.5 gso/s, 48.9K ulp |
| `nk_maxsim_packed_f32_icelake`      |     131 gso/s, 48.9K ulp |     124 gso/s, 48.9K ulp |     136 gso/s, 48.9K ulp |
| `nk_maxsim_packed_f32_sapphireamx`  |     273 gso/s, 48.9K ulp |     293 gso/s, 48.9K ulp |     285 gso/s, 48.9K ulp |
| __bf16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_bf16_serial`      |    15.9 gso/s, 49.0K ulp |    17.0 gso/s, 49.0K ulp |    15.3 gso/s, 49.0K ulp |
| `nk_maxsim_packed_bf16_haswell`     |    79.2 gso/s, 49.3K ulp |    85.0 gso/s, 49.3K ulp |    81.0 gso/s, 49.3K ulp |
| `nk_maxsim_packed_bf16_alder`       |     114 gso/s, 49.0K ulp |     110 gso/s, 49.0K ulp |     115 gso/s, 49.0K ulp |
| `nk_maxsim_packed_bf16_genoa`       |     163 gso/s, 49.0K ulp |     165 gso/s, 49.0K ulp |     174 gso/s, 49.0K ulp |
| `nk_maxsim_packed_bf16_sapphireamx` |       418 gso/s, 994 ulp |       418 gso/s, 994 ulp |       445 gso/s, 994 ulp |
| __f16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f16_serial`       |    15.5 gso/s, 49.4K ulp |    15.6 gso/s, 49.4K ulp |    16.9 gso/s, 49.4K ulp |
| `nk_maxsim_packed_f16_haswell`      |    79.1 gso/s, 49.8K ulp |    78.1 gso/s, 49.8K ulp |    79.1 gso/s, 49.8K ulp |
| `nk_maxsim_packed_f16_alder`        |     113 gso/s, 49.4K ulp |     112 gso/s, 49.4K ulp |     107 gso/s, 49.4K ulp |
| `nk_maxsim_packed_f16_icelake`      |     154 gso/s, 49.4K ulp |     164 gso/s, 49.4K ulp |     163 gso/s, 49.4K ulp |
| `nk_maxsim_packed_f16_sapphireamx`  |     339 gso/s, 49.5K ulp |     395 gso/s, 49.5K ulp |     381 gso/s, 49.5K ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                              |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f32__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f32_serial`       |       ? gso/s, 46.8K ulp |       ? gso/s, 46.8K ulp |       ? gso/s, 46.8K ulp |
| `nk_maxsim_packed_f32_v128relaxed`  |       ? gso/s, 1.58M ulp |       ? gso/s, 1.58M ulp |       ? gso/s, 1.58M ulp |
| __bf16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_bf16_serial`      |       ? gso/s, 47.0K ulp |       ? gso/s, 47.0K ulp |       ? gso/s, 47.0K ulp |
| `nk_maxsim_packed_bf16_v128relaxed` |       ? gso/s, 1.58M ulp |       ? gso/s, 1.58M ulp |       ? gso/s, 1.58M ulp |
| __f16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f16_serial`       |       ? gso/s, 46.4K ulp |       ? gso/s, 46.4K ulp |       ? gso/s, 46.4K ulp |
| `nk_maxsim_packed_f16_v128relaxed`  |       ? gso/s, 1.58M ulp |       ? gso/s, 1.58M ulp |       ? gso/s, 1.58M ulp |

### Apple M4

#### Native

| Kernel                           |                     256³ |                    1024³ |                    4096³ |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f32_serial`    |      124 gso/s, 166K ulp |      136 gso/s, 104K ulp |     130 gso/s, 55.1K ulp |
| `nk_maxsim_packed_f32_neonsdot`  |      170 gso/s, 167K ulp |      240 gso/s, 104K ulp |     167 gso/s, 55.1K ulp |
| `nk_maxsim_packed_f32_sme`       |      291 gso/s, 200K ulp |   1,800 gso/s, 64.6K ulp |           ? gso/s, ? ulp |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_bf16_serial`   |      135 gso/s, 167K ulp |      139 gso/s, 105K ulp |     132 gso/s, 54.8K ulp |
| `nk_maxsim_packed_bf16_neonsdot` |      192 gso/s, 167K ulp |      257 gso/s, 105K ulp |     161 gso/s, 54.8K ulp |
| `nk_maxsim_packed_bf16_sme`      |     580 gso/s, 16.1K ulp |     1,620 gso/s, 735 ulp |           ? gso/s, ? ulp |
| __f16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f16_serial`    |      136 gso/s, 169K ulp |      140 gso/s, 104K ulp |     134 gso/s, 55.1K ulp |
| `nk_maxsim_packed_f16_neonsdot`  |      193 gso/s, 166K ulp |      255 gso/s, 104K ulp |     172 gso/s, 55.1K ulp |
| `nk_maxsim_packed_f16_sme`       |     573 gso/s, 16.0K ulp |     1,620 gso/s, 725 ulp |           ? gso/s, ? ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                              |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f32__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f32_serial`       |    33.7 gso/s, 46.8K ulp |    35.0 gso/s, 46.8K ulp |    35.8 gso/s, 46.8K ulp |
| `nk_maxsim_packed_f32_v128relaxed`  |    88.5 gso/s, 46.0K ulp |    98.1 gso/s, 46.0K ulp |    82.7 gso/s, 46.0K ulp |
| __bf16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_bf16_serial`      |    34.4 gso/s, 49.2K ulp |    35.1 gso/s, 49.2K ulp |    35.7 gso/s, 49.2K ulp |
| `nk_maxsim_packed_bf16_v128relaxed` |    92.3 gso/s, 49.4K ulp |     100 gso/s, 49.4K ulp |    83.2 gso/s, 49.4K ulp |
| __f16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f16_serial`       |    33.8 gso/s, 49.5K ulp |    35.0 gso/s, 49.5K ulp |    35.7 gso/s, 49.5K ulp |
| `nk_maxsim_packed_f16_v128relaxed`  |    87.0 gso/s, 49.3K ulp |    95.8 gso/s, 49.3K ulp |    82.3 gso/s, 49.3K ulp |
