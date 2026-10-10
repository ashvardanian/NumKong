# MaxSim Late-Interaction Scoring in NumKong

NumKong implements ColBERT-style late-interaction scoring: the MaxSim score sums, over each query token, the minimum angular distance to any document token.
A two-stage coarse-to-fine strategy uses i8-quantized screening to rule out documents per query, then full-precision refinement of every document the screen cannot rule out computes the exact minimum angular distance.

MaxSim score:

$$
\text{MaxSim}(Q, D) = \sum_{i=0}^{m-1} \min_{j=0}^{n-1} \text{angular}(q_i, d_j)
$$

Coarse screening weights each i8 dot product by the document's quantization scale $s_j$ over its norm, $w_j = s_j / \|d_j\|$, which ranks documents by cosine:

$$
\hat{c}_{ij} = \text{dot}_{\text{i8}}(q_i, d_j) \cdot w_j
$$

Rounding leaves each quantized vector within $r = \tfrac{1}{2}\sqrt{k}$ of its scaled original in L2, so $\hat{c}_{ij}$ misses the exact $\text{dot}(q_i, d_j) / (s_{q_i} \|d_j\|)$ by at most

$$
e_{ij} = r + w_j \left( \frac{r}{w_{q_i}} + r^2 \right)
$$

Refinement then covers every candidate $C_i = \{ j : \hat{c}_{ij} + e_{ij} \ge \max_l (\hat{c}_{il} - e_{il}) \}$, so near-tied documents are all refined and the result matches an exhaustive search:

$$
\min_{j \in C_i} \text{angular}(q_i, d_j) = \min_{j \in C_i} \left( 1 - \frac{\text{dot}(q_i, d_j)}{\|q_i\| \cdot \|d_j\|} \right)
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def maxsim(queries: np.ndarray, documents: np.ndarray) -> float:
    score = 0.0
    norms = np.linalg.norm(documents, axis=1)
    for q in queries:
        screened, error = coarse_screen(q, documents)  # i8 scores and their error bounds
        candidates = screened + error >= np.max(screened - error)
        cosines = documents[candidates] @ q / (norms[candidates] * np.linalg.norm(q))
        score += max(1 - cosines.max(), 0)
    return score
```

## Input & Output Types

| Input Type | Output Type | Description                                      |
| :--------- | :---------- | :----------------------------------------------- |
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

All backends use i8-quantized coarse screening at O(m·n·k) with 1 byte/element instead of 2–4, followed by full-precision refinement at O(m·|C|·k) for only the candidate pairs.
On uniform random vectors the candidate set averages ~2.2 of 32 documents at depth 256, ~2.4 of 128 and ~4.2 of 300 at depth 128, and ~11 of 128 at depth 1536.
Refinement sums the angular distances with compensation, and f32 inputs keep f64 inverse norms, so f32 results land within an f64 ULP or two of an exhaustive f64 search.
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
The bf16 and f16 kernels keep that running maximum of exact cosines, while the f32 kernel screens in i8 and reads each 4-tile group twice.
The first pass raises every query's lower bound with element-wise `svmaxnm`, and the second refines, from the originals in the packed buffer, only the queries whose column score plus error reaches that bound.

### Three-Region Packed Buffer

All backends use a three-region packed buffer layout: [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals, 64B-aligned].
Per-vector metadata (16 bytes) stores the screening weight (quantization scale over norm), i8 sum (for bias correction), and the f64 inverse norm (for angular finalization).
The originals region stores full-precision vectors for refinement via existing `nk_dot_*` primitives.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by `NUMWARS_DIMS_HEIGHT`, `NUMWARS_DIMS_WIDTH`, and `NUMWARS_DIMS_DEPTH` environment variables, all set to the same value for late-interaction scoring over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga Scalar Operations per Second, with $\text{ops} = 2 \cdot M \cdot N \cdot K$ complexity for scoring $M$ query tokens against $N$ document tokens of dimension $K$.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                              |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f32__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f32_serial`       |    5.67 gso/s, 48.9K ulp |    5.71 gso/s, 48.9K ulp |    4.85 gso/s, 48.9K ulp |
| `nk_maxsim_packed_f32_haswell`      |    39.1 gso/s, 49.3K ulp |    38.1 gso/s, 49.3K ulp |    11.4 gso/s, 49.3K ulp |
| `nk_maxsim_packed_f32_alder`        |    52.7 gso/s, 48.9K ulp |    77.3 gso/s, 48.9K ulp |    31.4 gso/s, 48.9K ulp |
| `nk_maxsim_packed_f32_icelake`      |    65.7 gso/s, 48.9K ulp |     102 gso/s, 48.9K ulp |    44.3 gso/s, 48.9K ulp |
| `nk_maxsim_packed_f32_sapphireamx`  |    92.4 gso/s, 48.9K ulp |     158 gso/s, 48.9K ulp |    54.3 gso/s, 48.9K ulp |
| __bf16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_bf16_serial`      |    5.67 gso/s, 49.0K ulp |    5.77 gso/s, 49.0K ulp |    5.07 gso/s, 49.0K ulp |
| `nk_maxsim_packed_bf16_haswell`     |    43.1 gso/s, 49.3K ulp |    51.3 gso/s, 49.3K ulp |    17.8 gso/s, 49.3K ulp |
| `nk_maxsim_packed_bf16_alder`       |    60.0 gso/s, 49.0K ulp |    93.2 gso/s, 49.0K ulp |    47.2 gso/s, 49.0K ulp |
| `nk_maxsim_packed_bf16_genoa`       |    73.2 gso/s, 49.0K ulp |     126 gso/s, 49.0K ulp |    84.8 gso/s, 49.0K ulp |
| `nk_maxsim_packed_bf16_sapphireamx` |       486 gso/s, 994 ulp |       532 gso/s, 994 ulp |       431 gso/s, 994 ulp |
| __f16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f16_serial`       |    5.29 gso/s, 49.4K ulp |    5.14 gso/s, 49.4K ulp |    2.87 gso/s, 49.4K ulp |
| `nk_maxsim_packed_f16_haswell`      |    43.9 gso/s, 49.8K ulp |    52.2 gso/s, 49.8K ulp |    18.3 gso/s, 49.8K ulp |
| `nk_maxsim_packed_f16_alder`        |    58.4 gso/s, 49.4K ulp |    93.3 gso/s, 49.4K ulp |    48.0 gso/s, 49.4K ulp |
| `nk_maxsim_packed_f16_icelake`      |    72.7 gso/s, 49.4K ulp |     123 gso/s, 49.4K ulp |    77.8 gso/s, 49.4K ulp |
| `nk_maxsim_packed_f16_sapphireamx`  |     111 gso/s, 49.5K ulp |     231 gso/s, 49.5K ulp |     128 gso/s, 49.5K ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                              |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f32__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f32_serial`       |               4.23 gso/s |               4.36 gso/s |               3.67 gso/s |
| `nk_maxsim_packed_f32_v128relaxed`  |               19.4 gso/s |               12.0 gso/s |               3.02 gso/s |
| __bf16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_bf16_serial`      |               3.21 gso/s |               4.26 gso/s |               3.63 gso/s |
| `nk_maxsim_packed_bf16_v128relaxed` |               27.2 gso/s |               20.5 gso/s |               4.33 gso/s |
| __f16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_maxsim_packed_f16_serial`       |               4.04 gso/s |               3.97 gso/s |               1.77 gso/s |
| `nk_maxsim_packed_f16_v128relaxed`  |               15.7 gso/s |               9.18 gso/s |               2.13 gso/s |

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
