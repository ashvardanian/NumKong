# Set Similarity Measures in NumKong

NumKong implements set similarity functions for binary and integer vectors: Hamming distance measures the number of differing elements, while Jaccard distance measures the complement of the intersection-over-union ratio.
These are fundamental to locality-sensitive hashing, MinHash sketches, and binary feature matching.

Hamming distance counts the number of positions where elements differ.
For binary vectors packed as octets, this is the popcount of the XOR.
For byte-level vectors, it counts the number of mismatched bytes:

$$
\text{hamming}(a, b) = \sum_{i=0}^{n-1} [a_i \neq b_i]
$$

Jaccard distance measures the dissimilarity of two sets.
For binary vectors, the intersection and union are computed via bitwise AND and OR with popcount:

$$
\text{jaccard}(a, b) = 1 - \frac{|A \cap B|}{|A \cup B|} = 1 - \frac{\text{popcount}(a \mathbin{\&} b)}{\text{popcount}(a \mathbin{|} b)}
$$

For word-level vectors (MinHash signatures), Jaccard similarity is the fraction of matching elements:

$$
\text{jaccard}(a, b) = 1 - \frac{\sum_{i=0}^{n-1} [a_i = b_i]}{n}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    return np.unpackbits(np.bitwise_xor(a, b)).sum()

def jaccard_bits(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.unpackbits(np.bitwise_and(a, b)).sum()
    union = np.unpackbits(np.bitwise_or(a, b)).sum()
    return 1 - intersection / union if union else 0

def jaccard_words(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.mean(a == b)
```

## Input & Output Types

| Input Type | Output Type | Description                                 |
| ---------- | ----------- | ------------------------------------------- |
| `u1`       | `u32`       | Binary Hamming distance, packed octets      |
| `u1`       | `f32`       | Binary Jaccard distance, packed octets      |
| `u8`       | `u32`       | Byte-level Hamming distance                 |
| `u16`      | `f32`       | Word-level Jaccard distance, 16-bit MinHash |
| `u32`      | `f32`       | Word-level Jaccard distance, 32-bit MinHash |

## Optimizations

### Popcount Without Harley-Seal Carry-Save Adders

The textbook way to amortize population counts is a Harley-Seal carry-save adder tree, which folds three inputs through a full-adder circuit before any popcount is issued:

```
ones  = a ^ b ^ c
twos  = (a & b) | (c & (a ^ b))
```

Chaining levels yields `fours` and `eights` accumulators, each weighted by a power of two, so popcount runs on the accumulators instead of on every input vector — roughly a third of the calls.
NumKong does not do this.
The cycles-per-byte measurements in `set.h` put the crossover above 1 KB per vector: below it, loop and reduction overhead outweighs the saved popcounts, and the largest binary input benchmarked here is 4096 bits, or 512 bytes.
Ice Lake and later expose `VPOPCNTQ` outright, and on Genoa it dual-issues on ports 0-1, so the popcount pressure that CSA trees relieve is largely absent on the hardware that could run them.

AVX2 has no vector population count at all, so `nk_hamming_u1_haswell` and `nk_jaccard_u1_haswell` walk the packed octets 8 bytes at a time and call the scalar `POPCNT` through `_mm_popcnt_u64`.
`nk_hamming_u1_neon` and `nk_jaccard_u1_neon` use the NEON `CNT` instruction instead, accumulating per-byte counts in a `uint8x16_t` for at most 31 iterations before widening, since 31 x 8 still fits in a byte.

### Native VPOPCNTQ on Ice Lake

`nk_hamming_u1_icelake`, `nk_jaccard_u1_icelake` use `VPOPCNTQ` on 512-bit vectors, which directly produces per-quadword population counts for 8 quadwords at once.
Inputs of up to 256 bytes are fully unrolled into one, two, three, or four masked register pairs with no loop at all, since binary metrics are hard to speed up at tiny lengths.
Longer inputs enter a loop that accumulates the per-quadword counts via `VPADDQ` and reduces once at the end.

### Jaccard via Precomputed Norms

The streaming helpers `nk_jaccard_u1x64_finalize_haswell` and `nk_jaccard_u1x512_finalize_icelake` exploit the identity $|A \cup B| = |A| + |B| - |A \cap B|$ to avoid computing both AND-popcount and OR-popcount in the inner loop.
The matching `update` step accumulates only the intersection popcount; `finalize` takes the query popcount and a vector of four target popcounts and recovers the union from them, halving the work in the critical path.
Both finalizers replace the division by a reciprocal estimate — `RCPPS` on Haswell, `VRCP14PS` on Ice Lake — refined with one Newton-Raphson step.

### Byte Hamming via Compare Masks

`nk_hamming_u8_haswell` compares 32 bytes at a time with `VPCMPEQB`, extracts the per-byte sign bits of the two 128-bit halves with `VPMOVMSKB`, inverts the masks, and counts the set bits with scalar `POPCNT`.
`nk_hamming_u8_icelake` does the same over 64 bytes with `VPCMPNEQB`, which writes the mismatch mask straight into a `__mmask64`, so no extract step is needed.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported where applicable as exact distance in the result representation; floating Jaccard rows are shown as mean ULP (units in last place).
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                   |                      256 |                     1024 |                     4096 |
| :----------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`   |                2.30 gb/s |                2.62 gb/s |                2.54 gb/s |
| `nk_jaccard_u1_serial`   |         1.35 gb/s, 0 ulp |         1.46 gb/s, 0 ulp |         1.50 gb/s, 0 ulp |
| `nk_hamming_u1_haswell`  |                9.63 gb/s |                25.2 gb/s |                56.2 gb/s |
| `nk_jaccard_u1_haswell`  |         5.24 gb/s, 0 ulp |         15.5 gb/s, 0 ulp |         27.0 gb/s, 0 ulp |
| `nk_hamming_u1_icelake`  |                11.2 gb/s |                38.2 gb/s |                56.1 gb/s |
| `nk_jaccard_u1_icelake`  |         6.46 gb/s, 0 ulp |         22.4 gb/s, 0 ulp |         33.3 gb/s, 0 ulp |
| __u8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`   |                15.0 gb/s |                14.9 gb/s |                14.8 gb/s |
| `nk_hamming_u8_haswell`  |                22.4 gb/s |                21.6 gb/s |                17.9 gb/s |
| `nk_hamming_u8_icelake`  |                55.2 gb/s |                37.7 gb/s |                24.3 gb/s |
| __u16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial`  |         27.8 gb/s, 0 ulp |         23.0 gb/s, 0 ulp |         19.2 gb/s, 0 ulp |
| `nk_jaccard_u16_haswell` |         22.2 gb/s, 0 ulp |         18.4 gb/s, 0 ulp |         13.7 gb/s, 0 ulp |
| `nk_jaccard_u16_icelake` |         54.2 gb/s, 0 ulp |         24.3 gb/s, 0 ulp |         20.9 gb/s, 0 ulp |
| __u32__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial`  |         33.1 gb/s, 0 ulp |         23.5 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| `nk_jaccard_u32_haswell` |         19.0 gb/s, 0 ulp |         16.9 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |
| `nk_jaccard_u32_icelake` |         33.0 gb/s, 0 ulp |         24.6 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_v128relaxed`  |               0.138 gb/s |               0.149 gb/s |               0.979 gb/s |
| `nk_jaccard_u1_v128relaxed`  |        0.153 gb/s, 0 ulp |        0.352 gb/s, 0 ulp |         2.50 gb/s, 0 ulp |
| __u8__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_v128relaxed`  |               0.370 gb/s |               0.400 gb/s |                2.19 gb/s |
| __u16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_v128relaxed` |         2.30 gb/s, 0 ulp |         2.34 gb/s, 0 ulp |        0.381 gb/s, 0 ulp |
| __u32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_v128relaxed` |        0.430 gb/s, 0 ulp |         2.46 gb/s, 0 ulp |         1.08 gb/s, 0 ulp |

### Apple M5

#### Native

| Kernel                  |                      256 |                     1024 |                     4096 |
| :---------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`  |                6.79 gb/s |                7.48 gb/s |                6.92 gb/s |
| `nk_jaccard_u1_serial`  |         4.36 gb/s, 0 ulp |         5.38 gb/s, 0 ulp |         5.45 gb/s, 0 ulp |
| `nk_hamming_u1_neon`    |                31.6 gb/s |                65.6 gb/s |                90.9 gb/s |
| `nk_jaccard_u1_neon`    |         28.4 gb/s, 0 ulp |         48.1 gb/s, 0 ulp |         51.0 gb/s, 0 ulp |
| __u8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`  |                27.8 gb/s |                30.1 gb/s |                31.2 gb/s |
| `nk_hamming_u8_neon`    |                96.9 gb/s |                79.5 gb/s |                56.3 gb/s |
| __u16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial` |         59.3 gb/s, 0 ulp |         69.4 gb/s, 0 ulp |         66.8 gb/s, 0 ulp |
| `nk_jaccard_u16_neon`   |         67.8 gb/s, 0 ulp |         61.6 gb/s, 0 ulp |         50.8 gb/s, 0 ulp |
| __u32__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial` |          105 gb/s, 0 ulp |          101 gb/s, 0 ulp |         89.1 gb/s, 0 ulp |
| `nk_jaccard_u32_neon`   |         89.3 gb/s, 0 ulp |         72.8 gb/s, 0 ulp |         68.2 gb/s, 0 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`       |                5.18 gb/s |                5.66 gb/s |                6.52 gb/s |
| `nk_jaccard_u1_serial`       |         1.74 gb/s, 0 ulp |         3.32 gb/s, 0 ulp |         3.61 gb/s, 0 ulp |
| `nk_hamming_u1_v128relaxed`  |                22.6 gb/s |                46.5 gb/s |                67.9 gb/s |
| `nk_jaccard_u1_v128relaxed`  |         16.1 gb/s, 0 ulp |         34.5 gb/s, 0 ulp |         50.8 gb/s, 0 ulp |
| __u8__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`       |                8.32 gb/s |                6.09 gb/s |                5.84 gb/s |
| `nk_hamming_u8_v128relaxed`  |                47.7 gb/s |                68.5 gb/s |                72.1 gb/s |
| __u16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial`      |         19.2 gb/s, 0 ulp |         12.4 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |
| `nk_jaccard_u16_v128relaxed` |         89.8 gb/s, 0 ulp |         74.0 gb/s, 0 ulp |         71.3 gb/s, 0 ulp |
| __u32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial`      |         91.6 gb/s, 0 ulp |         69.4 gb/s, 0 ulp |         68.4 gb/s, 0 ulp |
| `nk_jaccard_u32_v128relaxed` |         94.8 gb/s, 0 ulp |         76.2 gb/s, 0 ulp |         68.8 gb/s, 0 ulp |
