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
| :--------- | :---------- | :------------------------------------------ |
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
`nk_hamming_u1_neon` and `nk_jaccard_u1_neon` use the NEON `CNT` instruction instead, accumulating per-byte counts in a `uint8x16_t` for at most 31 iterations before widening, since 31 × 8 still fits in a byte.

### Native VPOPCNTQ on Ice Lake

`nk_hamming_u1_icelake`, `nk_jaccard_u1_icelake` use `VPOPCNTQ` on 512-bit vectors, which directly produces per-quadword population counts for 8 quadwords at once.
Inputs of up to 256 bytes are fully unrolled into one, two, three, or four masked register pairs with no loop at all, since binary metrics are hard to speed up at tiny lengths.
Longer inputs enter a loop that accumulates the per-quadword counts via `VPADDQ` and reduces once at the end.

### Jaccard via Precomputed Norms

The packed kernels in `sets.h` exploit the identity $|A \cup B| = |A| + |B| - |A \cap B|$ to avoid computing both AND-popcount and OR-popcount in the inner loop.
Their tiles accumulate only the intersection popcounts, and `nk_jaccard_f32x4_from_dot_haswell_` and its siblings recover the unions from the precomputed popcounts, halving the work in the critical path.

### Byte Hamming via Compare Masks

`nk_hamming_u8_haswell` compares 32 bytes at a time with `VPCMPEQB`, extracts the per-byte sign bits of the two 128-bit halves with `VPMOVMSKB`, inverts the masks, and counts the set bits with scalar `POPCNT`.
`nk_hamming_u8_icelake` does the same over 64 bytes with `VPCMPNEQB`, which writes the mismatch mask straight into a `__mmask64`, so no extract step is needed.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_DIMS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported where applicable as exact distance in the result representation; floating Jaccard rows are shown as mean ULP (units in last place).

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                   |                      256 |                     1024 |                     4096 |
| :----------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`   |                2.06 gb/s |                2.37 gb/s |                2.60 gb/s |
| `nk_jaccard_u1_serial`   |         1.23 gb/s, 0 ulp |         1.43 gb/s, 0 ulp |         1.48 gb/s, 0 ulp |
| `nk_hamming_u1_haswell`  |                13.2 gb/s |                16.1 gb/s |                14.9 gb/s |
| `nk_jaccard_u1_haswell`  |         10.1 gb/s, 0 ulp |         9.24 gb/s, 0 ulp |         9.36 gb/s, 0 ulp |
| `nk_hamming_u1_icelake`  |                10.6 gb/s |                33.0 gb/s |                45.6 gb/s |
| `nk_jaccard_u1_icelake`  |         7.41 gb/s, 0 ulp |         23.9 gb/s, 0 ulp |         33.2 gb/s, 0 ulp |
| __u8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`   |                2.21 gb/s |                2.37 gb/s |                2.31 gb/s |
| `nk_hamming_u8_haswell`  |                24.0 gb/s |                14.9 gb/s |                17.8 gb/s |
| `nk_hamming_u8_icelake`  |                50.4 gb/s |                33.3 gb/s |                21.9 gb/s |
| __u16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial`  |         4.29 gb/s, 0 ulp |         4.45 gb/s, 0 ulp |         4.44 gb/s, 0 ulp |
| `nk_jaccard_u16_haswell` |         22.7 gb/s, 0 ulp |         13.7 gb/s, 0 ulp |         18.5 gb/s, 0 ulp |
| `nk_jaccard_u16_icelake` |         51.2 gb/s, 0 ulp |         21.1 gb/s, 0 ulp |         22.2 gb/s, 0 ulp |
| __u32__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial`  |         7.05 gb/s, 0 ulp |         8.15 gb/s, 0 ulp |         11.5 gb/s, 0 ulp |
| `nk_jaccard_u32_haswell` |         14.1 gb/s, 0 ulp |         17.3 gb/s, 0 ulp |         18.4 gb/s, 0 ulp |
| `nk_jaccard_u32_icelake` |         24.7 gb/s, 0 ulp |         21.6 gb/s, 0 ulp |         22.5 gb/s, 0 ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                  |                      256 |                     1024 |                     4096 |
| :---------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`  |                2.06 gb/s |                2.49 gb/s |                2.98 gb/s |
| `nk_jaccard_u1_serial`  |                1.17 gb/s |                1.55 gb/s |                1.71 gb/s |
| `nk_hamming_u1_v128`    |                6.21 gb/s |                17.4 gb/s |                21.4 gb/s |
| `nk_jaccard_u1_v128`    |         3.79 gb/s, 0 ulp |         12.2 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| __u8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`  |                2.81 gb/s |                2.99 gb/s |                3.09 gb/s |
| `nk_hamming_u8_v128`    |                20.2 gb/s |                19.8 gb/s |                16.1 gb/s |
| __u16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial` |                5.30 gb/s |                5.58 gb/s |                5.96 gb/s |
| `nk_jaccard_u16_v128`   |         21.6 gb/s, 0 ulp |         15.9 gb/s, 0 ulp |         19.7 gb/s, 0 ulp |
| __u32__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial` |                9.20 gb/s |                10.9 gb/s |                9.35 gb/s |
| `nk_jaccard_u32_v128`   |         23.4 gb/s, 0 ulp |         25.4 gb/s, 0 ulp |         23.5 gb/s, 0 ulp |

### Apple M5

#### Native

| Kernel                  |                      256 |                     1024 |                     4096 |
| :---------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`  |                6.32 gb/s |                6.97 gb/s |                6.44 gb/s |
| `nk_jaccard_u1_serial`  |         4.06 gb/s, 0 ulp |         5.01 gb/s, 0 ulp |         5.08 gb/s, 0 ulp |
| `nk_hamming_u1_neon`    |                29.4 gb/s |                61.1 gb/s |                84.7 gb/s |
| `nk_jaccard_u1_neon`    |         26.4 gb/s, 0 ulp |         44.8 gb/s, 0 ulp |         47.5 gb/s, 0 ulp |
| __u8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`  |                25.9 gb/s |                28.0 gb/s |                29.1 gb/s |
| `nk_hamming_u8_neon`    |                90.2 gb/s |                74.0 gb/s |                52.4 gb/s |
| __u16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial` |         55.2 gb/s, 0 ulp |         64.6 gb/s, 0 ulp |         62.2 gb/s, 0 ulp |
| `nk_jaccard_u16_neon`   |         63.1 gb/s, 0 ulp |         57.4 gb/s, 0 ulp |         47.3 gb/s, 0 ulp |
| __u32__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial` |         97.8 gb/s, 0 ulp |         94.1 gb/s, 0 ulp |         83.0 gb/s, 0 ulp |
| `nk_jaccard_u32_neon`   |         83.2 gb/s, 0 ulp |         67.8 gb/s, 0 ulp |         63.5 gb/s, 0 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                  |                      256 |                     1024 |                     4096 |
| :---------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`  |                4.82 gb/s |                5.27 gb/s |                6.07 gb/s |
| `nk_jaccard_u1_serial`  |         1.62 gb/s, 0 ulp |         3.09 gb/s, 0 ulp |         3.36 gb/s, 0 ulp |
| `nk_hamming_u1_v128`    |                21.0 gb/s |                43.3 gb/s |                63.2 gb/s |
| `nk_jaccard_u1_v128`    |         15.0 gb/s, 0 ulp |         32.1 gb/s, 0 ulp |         47.3 gb/s, 0 ulp |
| __u8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`  |                7.75 gb/s |                5.67 gb/s |                5.44 gb/s |
| `nk_hamming_u8_v128`    |                44.4 gb/s |                63.8 gb/s |                67.1 gb/s |
| __u16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial` |         17.9 gb/s, 0 ulp |         11.5 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |
| `nk_jaccard_u16_v128`   |         83.6 gb/s, 0 ulp |         68.9 gb/s, 0 ulp |         66.4 gb/s, 0 ulp |
| __u32__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial` |         85.3 gb/s, 0 ulp |         64.6 gb/s, 0 ulp |         63.7 gb/s, 0 ulp |
| `nk_jaccard_u32_v128`   |         88.3 gb/s, 0 ulp |         71.0 gb/s, 0 ulp |         64.1 gb/s, 0 ulp |
