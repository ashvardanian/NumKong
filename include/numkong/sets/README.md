# Batched Set Distances in NumKong

NumKong implements batched M×N Hamming and Jaccard distance matrices for binary vectors. The module reuses the dots u1 packing and GEMM infrastructure, converting popcount-of-AND dot products to set distances via precomputed norms.

Hamming distance from batched dot products:

$$
D_{ij} = \|A_i\|_1 + \|B_j\|_1 - 2 \cdot \text{dot}(A_i, B_j)
$$

Where dot = popcount(AND), measuring intersection size.

Jaccard distance from batched dot products:

$$
D_{ij} = 1 - \frac{\text{dot}(A_i, B_j)}{\|A_i\|_1 + \|B_j\|_1 - \text{dot}(A_i, B_j)}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def hammings_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.array([[np.unpackbits(np.bitwise_and(ai, bj)).sum()
                      for bj in b] for ai in a])
    a_pop = np.array([np.unpackbits(ai).sum() for ai in a])[:, None]
    b_pop = np.array([np.unpackbits(bj).sum() for bj in b])[None, :]
    return a_pop + b_pop - 2 * dots

def jaccards_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.array([[np.unpackbits(np.bitwise_and(ai, bj)).sum()
                      for bj in b] for ai in a])
    a_pop = np.array([np.unpackbits(ai).sum() for ai in a])[:, None]
    b_pop = np.array([np.unpackbits(bj).sum() for bj in b])[None, :]
    union = a_pop + b_pop - dots
    return np.where(union > 0, 1.0 - dots / union, 0.0)
```

## Input & Output Types

| Input Type | Output Type | Description                            |
| ---------- | ----------- | -------------------------------------- |
| `u1`       | `u32`       | Binary Hamming distance, packed octets |
| `u1`       | `f32`       | Binary Jaccard distance, packed octets |

## Optimizations

### Hamming and Jaccard from Intersection Counts

`nk_hammings_packed_u1_serial`, `nk_hammings_packed_u1_haswell`, `nk_jaccards_packed_u1_serial`, `nk_jaccards_packed_u1_haswell` reuse the dots u1 GEMM output where each dot product $\text{dot}(a, b) = \text{popcount}(a \mathbin{\&} b) = |A \cap B|$ counts intersection bits.
The L1 norm of a binary vector is its popcount: $|A| = \text{popcount}(a) = \|a\|_1$.
By inclusion-exclusion, $|A \cup B| = |A| + |B| - |A \cap B|$.
Hamming distance counts positions where exactly one bit is set: $D_H = |A| + |B| - 2|A \cap B| = \text{popcount}(a \oplus b)$.
Finalizer `nk_hamming_u32x4_from_dot_serial_` computes `pop_a + pop_b - 2 * dot` in pure UInt32 arithmetic — no division, no float conversion, no sqrt.
Jaccard distance: $D_J = 1 - \frac{|A \cap B|}{|A \cup B|} = 1 - \frac{\text{dot}}{\text{pop}_a + \text{pop}_b - \text{dot}}$.
Finalizer `nk_jaccard_f32x4_from_dot_serial_` requires UInt32 → Float32 cast plus Float32 division (~11cy latency on Haswell), making it ~3× more expensive per element than Hamming's integer subtraction chain.
Per-column popcount norms ($\|a\|_1$, $\|b\|_1$) are precomputed during packing and stored in packed buffer metadata, avoiding per-pair recomputation.

### SME Binary Outer-Product Accumulation

`nk_hammings_packed_u1_smebi32`, `nk_jaccards_packed_u1_smebi32` use the `BMOPA` instruction which computes $\text{popcount}(\text{XNOR}(a, b))$ — counting _matching_ bits in a single outer-product operation over 16×16 output tiles with 512-bit depth chunks.
This is fundamentally different from the AND+POPCNT used by scalar/NEON/x86 kernels, which count _intersection_ bits.
Hamming from `BMOPA`: $D_H = \text{depth\_bits} - \text{popcount}(\text{XNOR})$, since XOR popcount (differing bits) is the Hamming distance directly — no per-vector norm correction needed.
Jaccard from `BMOPA`: must convert matching-bit counts to intersection via $|A \cap B| = (\text{popcount}(\text{XNOR}) - (\text{depth\_bits} - |A| - |B|)) / 2$, then apply the Jaccard formula — more arithmetic than the AND-based path.
Streaming mode overhead (~50–100 cycles for `SMSTART`/`SMSTOP`) is amortized across the full M×N output.

## Performance

The following performance tables are produced by manually re-running `numkong_test` and `numkong_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for batched set operations over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga Scalar Operations per Second.
Accuracy is reported where applicable as exact distance in the result representation; floating Jaccard rows are shown as mean ULP (units in last place).
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`     |                109 gso/s |                162 gso/s |                284 gso/s |
| `nk_hammings_symmetric_u1_serial`  |               39.7 gso/s |                133 gso/s |                325 gso/s |
| `nk_jaccards_packed_u1_serial`     |        54.8 gso/s, 0 ulp |         128 gso/s, 0 ulp |         259 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`  |        29.8 gso/s, 0 ulp |         110 gso/s, 0 ulp |         292 gso/s, 0 ulp |
| `nk_hammings_packed_u1_haswell`    |                100 gso/s |                126 gso/s |                168 gso/s |
| `nk_hammings_symmetric_u1_haswell` |               58.5 gso/s |                132 gso/s |                328 gso/s |
| `nk_jaccards_packed_u1_haswell`    |      84.2 gso/s, 0.3 ulp |       124 gso/s, 0.3 ulp |       165 gso/s, 0.3 ulp |
| `nk_jaccards_symmetric_u1_haswell` |      57.6 gso/s, 0.3 ulp |       131 gso/s, 0.3 ulp |       324 gso/s, 0.3 ulp |
| `nk_hammings_packed_u1_icelake`    |                110 gso/s |                340 gso/s |                604 gso/s |
| `nk_hammings_symmetric_u1_icelake` |               76.2 gso/s |                258 gso/s |              1,040 gso/s |
| `nk_jaccards_packed_u1_icelake`    |      89.2 gso/s, 0.3 ulp |       312 gso/s, 0.3 ulp |       601 gso/s, 0.3 ulp |
| `nk_jaccards_symmetric_u1_icelake` |      66.9 gso/s, 0.3 ulp |       260 gso/s, 0.3 ulp |       965 gso/s, 0.3 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                            |                     256³ |                    1024³ |                    4096³ |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`    |               43.7 gso/s |               68.0 gso/s |               74.7 gso/s |
| `nk_hammings_packed_u1_v128`      |               75.3 gso/s |                134 gso/s |                144 gso/s |
| `nk_hammings_symmetric_u1_serial` |               3.72 gso/s |               13.5 gso/s |               41.0 gso/s |
| `nk_hammings_symmetric_u1_v128`   |               3.64 gso/s |               13.9 gso/s |               42.2 gso/s |
| `nk_jaccards_packed_u1_serial`    |        33.7 gso/s, 0 ulp |        61.3 gso/s, 0 ulp |        73.2 gso/s, 0 ulp |
| `nk_jaccards_packed_u1_v128`      |        66.4 gso/s, 0 ulp |         129 gso/s, 0 ulp |         143 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial` |        3.57 gso/s, 0 ulp |        13.3 gso/s, 0 ulp |        40.6 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_v128`   |        3.65 gso/s, 0 ulp |        13.9 gso/s, 0 ulp |        42.2 gso/s, 0 ulp |

### Apple M5

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`     |                156 gso/s |                231 gso/s |                262 gso/s |
| `nk_hammings_symmetric_u1_serial`  |                106 gso/s |                196 gso/s |                246 gso/s |
| `nk_jaccards_packed_u1_serial`     |         136 gso/s, 0 ulp |         221 gso/s, 0 ulp |         262 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`  |        96.5 gso/s, 0 ulp |         183 gso/s, 0 ulp |         244 gso/s, 0 ulp |
| `nk_hammings_packed_u1_neon`       |                321 gso/s |                436 gso/s |                508 gso/s |
| `nk_hammings_symmetric_u1_neon`    |                126 gso/s |                239 gso/s |                318 gso/s |
| `nk_jaccards_packed_u1_neon`       |         271 gso/s, 0 ulp |         423 gso/s, 0 ulp |         503 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_neon`    |         120 gso/s, 0 ulp |         233 gso/s, 0 ulp |         316 gso/s, 0 ulp |
| `nk_hammings_packed_u1_smebi32`    |              3,286 gso/s |              7,303 gso/s |             11,269 gso/s |
| `nk_hammings_symmetric_u1_smebi32` |              1,872 gso/s |              5,332 gso/s |              4,079 gso/s |
| `nk_jaccards_packed_u1_smebi32`    |         371 gso/s, 0 ulp |       1,735 gso/s, 0 ulp |       4,348 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_smebi32` |        83.1 gso/s, 0 ulp |         358 gso/s, 0 ulp |       1,005 gso/s, 0 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                            |                     256³ |                    1024³ |                    4096³ |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`    |               99.3 gso/s |                127 gso/s |                154 gso/s |
| `nk_hammings_symmetric_u1_serial` |               63.7 gso/s |                142 gso/s |                210 gso/s |
| `nk_jaccards_packed_u1_serial`    |        92.2 gso/s, 0 ulp |         123 gso/s, 0 ulp |         153 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial` |        59.3 gso/s, 0 ulp |         142 gso/s, 0 ulp |         207 gso/s, 0 ulp |
| `nk_hammings_packed_u1_v128`      |                266 gso/s |                378 gso/s |                426 gso/s |
| `nk_hammings_symmetric_u1_v128`   |               72.2 gso/s |                185 gso/s |                259 gso/s |
| `nk_jaccards_packed_u1_v128`      |         243 gso/s, 0 ulp |         370 gso/s, 0 ulp |         424 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_v128`   |        72.9 gso/s, 0 ulp |         183 gso/s, 0 ulp |         257 gso/s, 0 ulp |
