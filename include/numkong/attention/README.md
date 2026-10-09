# Ragged Scaled-Dot-Product Attention in NumKong

NumKong implements FlashAttention-style scaled-dot-product attention __(SDPA)__ over ragged batches with a pre-packed KV-cache, as the fused core of Transformer inference on CPUs.
Every backend shares one packing pair, `pack_size` to size the cache and `pack` to rearrange K/V into a backend-opaque layout, and one compute kernel over the same pack, `packed`, with `packed_gradients` for training.
Each takes a half-open `[tasks_begin, tasks_end)` window over its task grid for embarrassingly parallel execution, and windows run in any order.
Packing tasks are segments × key-value heads; the window starting at task 0 also writes the header and directory, which no other window reads.
Attention tasks are query tokens × heads, task `t · head_count + h` being head `h` of query token `t`, the index of its log-sum-exp, so a window may split a segment or a head group.
Bindings cut that grid into a few windows of equal cost per thread, a row costing the keys it sees, so a single decode token still spreads over its heads.

For a single segment and head, the operation is the classic softmax attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Ragged batches generalize this to independently sized segments sharing one packed cache, and decoupled `query_offsets` turn the same cache into a cross-attention or batched single-query pooling operator.
Reformulating as Python pseudocode:

```python
def attention_ragged(q, k, v, key_offsets, query_offsets, scale) -> Matrix:
    out = zeros(rows=query_offsets[-1], columns=q.columns)
    for s in range(len(key_offsets) - 1):
        kv = slice(key_offsets[s], key_offsets[s + 1])
        queries = slice(query_offsets[s], query_offsets[s + 1])
        for h in range(head_count):
            kv_head = h // (head_count // key_value_head_count)  # GQA / MQA sharing
            scores = q[queries, h] @ k[kv, kv_head].T * scale
            out[queries, h] = softmax(scores, axis=-1) @ v[kv, kv_head]
    return out
```

Each segment's queries align to the end of its keys: query row `r` of a segment with `q` queries and `k` keys sits at position `p = r + k − q` and sees keys `p − keys_before` through `p + keys_after`, each side unbounded at `NUMKONG_SIZE_MAX`.
Causal attention is `(NUMKONG_SIZE_MAX, 0)`, a sliding window of `w` keys is `(w − 1, 0)`, and bidirectional attention is `(NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX)`, so prefill, decode against a longer cache, and ragged chunked prefill with a different `k − q` per segment all run without an offset.
Rows that see no key produce zeros and a log-sum-exp of −∞, and so do segments without keys.
Masking clips each row's key range rather than writing −∞ scores, tiles outside the band are skipped outright, and tiles inside it run unmasked.

Attention `packed` is the twin of `dots_packed`: queries meet the packed keys and values as A meets the packed B.
A family gains a verb when its operands or outputs differ, and a parameter when only the visible region does.
So `dots_symmetric` is a verb, reading one operand and filling the upper triangle, the band `(0, NUMKONG_SIZE_MAX)` that A·Aᵀ mirrors, while the attention mask is the `keys_before, keys_after` band of `packed`, whose skipped tiles are forbidden rather than mirrored.

Internally every backend uses the streaming base-2 softmax: the scale folds $\log_2 e$, so the exponentials become `exp2` with one shared degree-4 polynomial after exact range reduction, and multi-panel sweeps carry a running maximum with the correction $O \leftarrow O \cdot 2^{m_{old} - m_{new}} + O_{panel}$.

## Input & Output Types

| Input Type  | Output Type | Description                                                                |
| :---------- | :---------- | :------------------------------------------------------------------------- |
| `bf16`      | `f32`       | 16-bit brain float; native AMX tiles and `VDPBF16PS` lanes                 |
| `f16`       | `f32`       | 16-bit IEEE half; F16 tensor cores and F16 weights from Ampere on          |
| `e4m3`      | `f32`       | 8-bit Float8; widened to the ISA's compute format at the pack boundary     |
| `i8`        | `f32`       | 8-bit signed integers; exact `i32` scores, probabilities quantized to `u8` |
| `nvfp4`     | `f32`       | E2M1 codes in blocks of 16 under UE4M3 scales and an F32 tensor scale      |
| `mxfp4`     | `f32`       | E2M1 codes in blocks of 32 under UE8M0 power-of-two scales                 |
| `mxfp6e2m3` | `f32`       | E2M3 codes in blocks of 32 under UE8M0 scales                              |
| `mxfp6e3m2` | `f32`       | E3M2 codes in blocks of 32 under UE8M0 scales                              |
| `mxfp8e4m3` | `f32`       | E4M3 codes in blocks of 32 under UE8M0 scales                              |
| `mxfp8e5m2` | `f32`       | E5M2 codes in blocks of 32 under UE8M0 scales                              |

Shape envelope: any `depth ≥ 1` (SIMD fast paths cover 1…256 with zero-padded channels; wider heads route to the width-agnostic serial kernel), arbitrary segment lengths including empty PAD segments, and any integer GQA ratio.
Quantization scales fold into the `scale` argument for `i8` queries and keys, or stay with the caller for values, so every dtype shares one signature.
Plain `e4m3` is unscaled, as in the `dots` family.
Block-scaled formats take their codes, block scales and NVFP4 tensor scale by reference, as in the `dots` family, with `depth` a whole number of blocks.

## Training

Every forward kernel optionally writes `log_sum_exp`, the natural log of each row's softmax normalizer, dense over query tokens and heads, and −∞ for rows that see no key.
The `packed_gradients` verb takes it back with the forward's output, the output gradient and the forward's band, recomputes the attention weights from it, and overwrites F32 gradients of the queries, keys and values:

$$
dV = P^\top dO, \quad dS = P \circ (dO\,V^\top - D), \quad dQ = \text{scale} \cdot dS\,K, \quad dK = \text{scale} \cdot dS^\top Q
$$

Here $D = \text{rowsum}(dO \circ O)$.
The backward's task grid is segments × key-value heads, each task owning one key-value head of one segment with every query head sharing it, so the gradients need no atomics and, outside the Blackwell single pass below, every sum runs in a fixed order.
Query gradient rows take their own `query_gradient_stride`, and key and value gradient rows past a segment's live keys, the gaps and tails before the next segment, are never written, so callers zero them when they matter.
The BF16 backward runs on tensor cores for heads of up to 256 dimensions: FlashAttention-2 on `mma.sync` from Ampere on, and `tcgen05.mma` accumulating in tensor memory on Blackwell.
Gradients accumulate 128 columns at a time, so heads past 128 dimensions recompute the scores for a second slice.
Block-scaled gradients are taken with respect to the decoded values.

### Two Backward Paths on Blackwell

The two-pass backward walks every 128-key block once for $dK$ and $dV$, then every 128-row chunk once more for $dQ$, so it recomputes $S$, $dP$ and $dS$ twice: 7 products per pair where 5 suffice.
The single pass computes them once per block and chunk, keeps $dK$ and $dV$ in tensor memory, writes the partial $dS\,K$ over the spent $dP$ columns, and adds it into `query_gradient` with TMA reduce-add of F32 boxes.
It runs for BF16 heads of 64 or 128 dimensions when the query head group divides 128 and every row and stride is 16-byte aligned; any other call takes the two-pass path.
It needs $dO$ as BF16 and $D$ once per row, so it allocates a workspace of `query_token_count` × `head_count` × (2 × `depth` + 4) bytes with `cudaMallocAsync` on the caller's stream and frees it with `cudaFreeAsync` after the kernels, which keeps graph capture working and never synchronizes.
If that allocation fails, the call falls back to the two-pass path.
A caller that synchronizes between calls lets the default memory pool return the workspace to the driver, so the next call maps it again for about a millisecond; raising that pool's `cudaMemPoolAttrReleaseThreshold` keeps it mapped.
Different blocks add their $dQ$ partials in whatever order they finish, so the low bits of $dQ$ may differ from run to run, as with any F32 accumulation of BF16 products; $dK$ and $dV$ stay deterministic.

## Optimizations

### Panel-Flash Sweep with L2-Resident Score Panels

All SIMD backends sweep KV in panels of 512 positions: scores for a query block land in a scratch panel, one row-major pass computes the running maximum, exponentials, and weight sum, and the weighted V accumulation drains once per panel with the correction FMA fused in.
This bounds the vector-unit work to one crossover per score, which is the dominant cost on matrix-unit ISAs where tile registers support no elementwise math.

### AMX 2×2 Tile Blocking with KV Reuse

`nk_attention_packed_bf16_sapphireamx` processes 32 query rows against 32-column K pair-tiles with all eight TMM registers: four accumulators, two Q tiles, two K tiles — exactly one tile load per `TDPBF16PS`.
Query blocks are chunked in groups of four so each KV panel streamed from L2 serves 128 query rows, which is what holds throughput flat from 1K to 16K context.
The `i8` variant swaps in `TDPBSSD` over quad-interleaved K tiles (depth 64 per step) and `TDPBUSD` for the u8-probability × i8-value product.

### Per-ISA Storage Formats

Packing converts dtypes only into the ISA's native compute format, mirroring the `dots` family: raw BF16 plus in-loop widening on Haswell and Skylake, `e4m3 → f16` at pack on Skylake so the hot loop widens with one `VCVTPH2PS`, `e4m3 → bf16` through the Ice Lake converters for Genoa's `VDPBF16PS` and the AMX tiles, and raw bytes on Haswell where every conversion is on the fly.
KV planes are the streamed operand, so at-rest bytes are memory bandwidth.

### SME Streaming Softmax with Lane-Parallel Bookkeeping

The Arm SME backend enters streaming mode once per call and never leaves it: scores accumulate as 2×2 widening MOPA outer products, drain through vertical ZA stores into a position-major panel with one query per lane, and the running maxima, corrections, weight sums, output rescaling, and normalization all run as plain lane-parallel vector operations with no horizontal reductions or scalar broadcasts.
Probabilities round to pair-interleaved BF16 MOPA operands in registers with one `TRN2` per position pair, and the Q staging plus the final output transpose reuse the ZA horizontal-write/vertical-read idiom from the `dots` packer.
Non-widening ZA16 tiles measure about twice the MOPA rate but lose ~12% relative accuracy on signed depth-256 reductions, so every reduction stays in F32 accumulators.

### Rebased Planes for Block-Scaled Formats

NVFP4 planes hold every element times its UE4M3 block scale, which F32 and even F16 represent exactly, and the pack header keeps both tensor scales.
MX scales span 254 binades, so each MX plane rebases its elements by the plane's largest finite block exponent less 31 and records that exponent in a small table past the planes.
A plane whose block exponents span more than 89 binades keeps its raw codes and scale codes instead, and scores against it take the exact dot of raw codes.
Scores sum each query block in F32 against the rebased key row and add the blocks into a wide sum, so query scales of any spread round once.
Outputs and query gradients accumulate under one plane base and apply its power of two once per element.

### Apple GPU Tiles and Split Decode

The `metal` baseline gives every query row a SIMD-group, while `apple9` and `apple10` attend 32-row tiles against 64-key panels, on SIMD-group matrices and on the `matmul2d` tensor operations of the Neural Accelerators.
Decoding up to four rows against 512 keys or more splits each row's keys into 16 partitions, and a merge kernel combines their maxima, sums and weighted values, so short outputs still fill the GPU.

### U8-Quantized Probabilities for INT8

The `i8` kernels quantize softmax weights as $\tilde{w} = \text{round}(255 \cdot 2^{s_2 - m_2})$ and normalize by $\sum \tilde{w}$, so the 255 cancels and no descale constant survives.
The maximum-scoring position always quantizes to exactly 255, making the weight sum provably non-zero.
Scores stay exact in `i32` integer arithmetic; only the probabilities round.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology) on one core pinned with `numactl --membind=0 taskset -c <core>`, counting `4 \cdot h \cdot n_q \cdot n_{kv} \cdot d` FLOPs.
Rows are kernels under the mask they measured, columns are square self-attention shapes at `depth = 128`, 32 query heads over 8 key-value heads; accuracy is the maximum absolute error against an `f64` reference over dtype-rounded inputs.
Bidirectional rows ran `keys_before = keys_after = NUMKONG_SIZE_MAX`, and causal rows `keys_after = 0`.
Cells marked `⋯` await measurement on the corresponding platform.

### Intel Sapphire Rapids

#### Native

| Kernel                                               |                 1024² |                 4096² |                16384² |
| :--------------------------------------------------- | --------------------: | --------------------: | --------------------: |
| __bf16__                                             | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`      |           0.6 gflop/s |                     ⋯ |                     ⋯ |
| bidirectional `nk_attention_packed_bf16_haswell`     |            36 gflop/s |            33 gflop/s |            25 gflop/s |
| bidirectional `nk_attention_packed_bf16_skylake`     |            48 gflop/s |            41 gflop/s |            28 gflop/s |
| bidirectional `nk_attention_packed_bf16_genoa`       |            54 gflop/s |            47 gflop/s |            30 gflop/s |
| bidirectional `nk_attention_packed_bf16_sapphireamx` |           845 gflop/s |           827 gflop/s |           766 gflop/s |
| __e4m3__                                             | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`      |           1.2 gflop/s |                     ⋯ |                     ⋯ |
| bidirectional `nk_attention_packed_e4m3_haswell`     |            11 gflop/s |            11 gflop/s |            11 gflop/s |
| bidirectional `nk_attention_packed_e4m3_skylake`     |            49 gflop/s |            41 gflop/s |            29 gflop/s |
| bidirectional `nk_attention_packed_e4m3_genoa`       |            57 gflop/s |            46 gflop/s |            30 gflop/s |
| bidirectional `nk_attention_packed_e4m3_sapphireamx` |           840 gflop/s |           841 gflop/s |           786 gflop/s |
| __i8__                                               | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`        |           1.2 gflop/s |                     ⋯ |                     ⋯ |
| bidirectional `nk_attention_packed_i8_haswell`       |           123 gflop/s |           125 gflop/s |           113 gflop/s |
| bidirectional `nk_attention_packed_i8_icelake`       |           185 gflop/s |           186 gflop/s |           171 gflop/s |
| bidirectional `nk_attention_packed_i8_sapphireamx`   |           941 gflop/s |           950 gflop/s |           931 gflop/s |

#### WASM

Measured with Wasmtime v24 (Cranelift backend).

| Kernel                                               |                 1024² |                 4096² |                16384² |
| :--------------------------------------------------- | --------------------: | --------------------: | --------------------: |
| __bf16__                                             | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`      |          0.60 gflop/s |                     ⋯ |                     ⋯ |
| bidirectional `nk_attention_packed_bf16_v128relaxed` |            10 gflop/s |           9.9 gflop/s |            11 gflop/s |
| __e4m3__                                             | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`      |          0.53 gflop/s |                     ⋯ |                     ⋯ |
| bidirectional `nk_attention_packed_e4m3_v128relaxed` |           3.3 gflop/s |           3.0 gflop/s |           3.7 gflop/s |
| __i8__                                               | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`        |           5.1 gflop/s |                     ⋯ |                     ⋯ |
| bidirectional `nk_attention_packed_i8_v128relaxed`   |          24.6 gflop/s |          24.1 gflop/s |          29.2 gflop/s |

### Intel Granite Rapids with RTX PRO 6000 Blackwell

#### Native

| Kernel                                                |                 1024² |                 4096² |                16384² |
| :---------------------------------------------------- | --------------------: | --------------------: | --------------------: |
| __bf16__                                              | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`       |           0.7 gflop/s |           0.7 gflop/s |                     ⋯ |
| causal `nk_attention_packed_bf16_serial`              |           1.4 gflop/s |           1.4 gflop/s |                     ⋯ |
| bidirectional `nk_attention_packed_bf16_haswell`      |            36 gflop/s |            28 gflop/s |            21 gflop/s |
| causal `nk_attention_packed_bf16_haswell`             |            70 gflop/s |            69 gflop/s |            44 gflop/s |
| bidirectional `nk_attention_packed_bf16_skylake`      |            47 gflop/s |            33 gflop/s |            25 gflop/s |
| causal `nk_attention_packed_bf16_skylake`             |            92 gflop/s |            71 gflop/s |            55 gflop/s |
| bidirectional `nk_attention_packed_bf16_genoa`        |            56 gflop/s |            41 gflop/s |            27 gflop/s |
| causal `nk_attention_packed_bf16_genoa`               |           105 gflop/s |            84 gflop/s |            57 gflop/s |
| bidirectional `nk_attention_packed_bf16_sapphireamx`  |           762 gflop/s |           751 gflop/s |           698 gflop/s |
| causal `nk_attention_packed_bf16_sapphireamx`         |         1,034 gflop/s |         1,406 gflop/s |         1,341 gflop/s |
| bidirectional `nk_attention_packed_bf16_ampere`       |                     ⋯ |       200,537 gflop/s |                     ⋯ |
| causal `nk_attention_packed_bf16_ampere`              |                     ⋯ |       297,474 gflop/s |                     ⋯ |
| __e4m3__                                              | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`       |           0.7 gflop/s |           0.6 gflop/s |                     ⋯ |
| causal `nk_attention_packed_e4m3_serial`              |           1.1 gflop/s |           1.3 gflop/s |                     ⋯ |
| bidirectional `nk_attention_packed_e4m3_haswell`      |            12 gflop/s |            12 gflop/s |            11 gflop/s |
| causal `nk_attention_packed_e4m3_haswell`             |            23 gflop/s |            23 gflop/s |            20 gflop/s |
| bidirectional `nk_attention_packed_e4m3_skylake`      |            50 gflop/s |            36 gflop/s |            28 gflop/s |
| causal `nk_attention_packed_e4m3_skylake`             |            98 gflop/s |            81 gflop/s |            55 gflop/s |
| bidirectional `nk_attention_packed_e4m3_genoa`        |            56 gflop/s |            41 gflop/s |            28 gflop/s |
| causal `nk_attention_packed_e4m3_genoa`               |           106 gflop/s |            89 gflop/s |            54 gflop/s |
| bidirectional `nk_attention_packed_e4m3_sapphireamx`  |           763 gflop/s |           733 gflop/s |           682 gflop/s |
| causal `nk_attention_packed_e4m3_sapphireamx`         |         1,065 gflop/s |         1,413 gflop/s |         1,343 gflop/s |
| bidirectional `nk_attention_packed_e4m3_ampere`       |                     ⋯ |       191,899 gflop/s |                     ⋯ |
| causal `nk_attention_packed_e4m3_ampere`              |                     ⋯ |       278,610 gflop/s |                     ⋯ |
| bidirectional `nk_attention_packed_e4m3_blackwellrtx` |                     ⋯ |       302,069 gflop/s |                     ⋯ |
| causal `nk_attention_packed_e4m3_blackwellrtx`        |                     ⋯ |       436,146 gflop/s |                     ⋯ |
| __i8__                                                | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`         |           5.8 gflop/s |           5.6 gflop/s |                     ⋯ |
| causal `nk_attention_packed_i8_serial`                |            11 gflop/s |            11 gflop/s |                     ⋯ |
| bidirectional `nk_attention_packed_i8_haswell`        |           149 gflop/s |           154 gflop/s |           141 gflop/s |
| causal `nk_attention_packed_i8_haswell`               |           267 gflop/s |           300 gflop/s |           266 gflop/s |
| bidirectional `nk_attention_packed_i8_icelake`        |           181 gflop/s |           190 gflop/s |           147 gflop/s |
| causal `nk_attention_packed_i8_icelake`               |           352 gflop/s |           308 gflop/s |           345 gflop/s |
| bidirectional `nk_attention_packed_i8_sapphireamx`    |           745 gflop/s |           821 gflop/s |           752 gflop/s |
| causal `nk_attention_packed_i8_sapphireamx`           |         1,136 gflop/s |         1,521 gflop/s |         1,516 gflop/s |
| bidirectional `nk_attention_packed_i8_ampere`         |                     ⋯ |       290,967 gflop/s |                     ⋯ |
| causal `nk_attention_packed_i8_ampere`                |                     ⋯ |       407,579 gflop/s |                     ⋯ |

### Intel Xeon 6 with B300

#### Native

| Kernel                                               |        1024² |        4096² |       16384² |
| :--------------------------------------------------- | -----------: | -----------: | -----------: |
| __bf16__                                             | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`      |  2.3 gflop/s |  2.1 gflop/s |            ⋯ |
| causal `nk_attention_packed_bf16_serial`             |  2.4 gflop/s |  2.2 gflop/s |            ⋯ |
| bidirectional `nk_attention_packed_bf16_haswell`     | 23.6 gflop/s | 22.5 gflop/s | 19.7 gflop/s |
| causal `nk_attention_packed_bf16_haswell`            | 23.1 gflop/s | 22.9 gflop/s | 19.7 gflop/s |
| bidirectional `nk_attention_packed_bf16_skylake`     | 29.1 gflop/s | 27.2 gflop/s | 23.2 gflop/s |
| causal `nk_attention_packed_bf16_skylake`            | 28.6 gflop/s | 28.4 gflop/s | 23.9 gflop/s |
| bidirectional `nk_attention_packed_bf16_genoa`       | 31.7 gflop/s | 30.4 gflop/s | 25.6 gflop/s |
| causal `nk_attention_packed_bf16_genoa`              | 32.2 gflop/s | 31.5 gflop/s | 26.0 gflop/s |
| bidirectional `nk_attention_packed_bf16_sapphireamx` |  418 gflop/s |  447 gflop/s |  450 gflop/s |
| causal `nk_attention_packed_bf16_sapphireamx`        |  306 gflop/s |  411 gflop/s |  437 gflop/s |
| __e4m3__                                             | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`      |  1.1 gflop/s |  1.1 gflop/s |            ⋯ |
| causal `nk_attention_packed_e4m3_serial`             |  1.2 gflop/s |  1.1 gflop/s |            ⋯ |
| bidirectional `nk_attention_packed_e4m3_haswell`     |  9.6 gflop/s |  9.7 gflop/s |  9.8 gflop/s |
| causal `nk_attention_packed_e4m3_haswell`            |  9.7 gflop/s |  9.9 gflop/s |  9.7 gflop/s |
| bidirectional `nk_attention_packed_e4m3_skylake`     | 29.9 gflop/s | 28.6 gflop/s | 23.9 gflop/s |
| causal `nk_attention_packed_e4m3_skylake`            | 29.6 gflop/s | 29.3 gflop/s | 24.3 gflop/s |
| bidirectional `nk_attention_packed_e4m3_genoa`       | 32.3 gflop/s | 29.8 gflop/s | 25.4 gflop/s |
| causal `nk_attention_packed_e4m3_genoa`              | 32.4 gflop/s | 31.7 gflop/s | 26.1 gflop/s |
| bidirectional `nk_attention_packed_e4m3_sapphireamx` |  419 gflop/s |  447 gflop/s |  451 gflop/s |
| causal `nk_attention_packed_e4m3_sapphireamx`        |  306 gflop/s |  415 gflop/s |  439 gflop/s |
| __i8__                                               | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`        |  5.0 gflop/s |  5.0 gflop/s |            ⋯ |
| causal `nk_attention_packed_i8_serial`               |  4.8 gflop/s |  4.9 gflop/s |            ⋯ |
| bidirectional `nk_attention_packed_i8_haswell`       |  130 gflop/s |  136 gflop/s |  140 gflop/s |
| causal `nk_attention_packed_i8_haswell`              |  122 gflop/s |  134 gflop/s |  139 gflop/s |
| bidirectional `nk_attention_packed_i8_icelake`       |  143 gflop/s |  142 gflop/s |  136 gflop/s |
| causal `nk_attention_packed_i8_icelake`              |  138 gflop/s |  142 gflop/s |  139 gflop/s |
| bidirectional `nk_attention_packed_i8_sapphireamx`   |  546 gflop/s |  616 gflop/s |  606 gflop/s |
| causal `nk_attention_packed_i8_sapphireamx`          |  425 gflop/s |  570 gflop/s |  601 gflop/s |

#### CUDA

Rows ran on one `1g.34gb` MIG slice of a B300 with 18 SMs, for 32 query heads sharing 8 key-value heads at depth 128 over 4096 keys, with 4096 queries in the first column and one in the second.
They count only the visible pairs, so causal and windowed rows compare directly with bidirectional ones.
Gradient rows time the backward alone at 10 · depth operations per pair.
The `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR` rows time cuDNN 9.27's fused forward over the same inputs; its backward descriptor has no engine on this GPU.

| Kernel                                                            |    4096 queries |       1 query |
| :---------------------------------------------------------------- | --------------: | ------------: |
| __bf16__                                                          | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_blackwell`                | 160,000 gflop/s | 1,743 gflop/s |
| causal `nk_attention_packed_bf16_blackwell`                       | 134,700 gflop/s | 1,701 gflop/s |
| window 1024 `nk_attention_packed_bf16_blackwell`                  |  93,000 gflop/s |   931 gflop/s |
| bidirectional `nk_attention_packed_bf16_blackwellultra`           | 173,000 gflop/s | 1,768 gflop/s |
| causal `nk_attention_packed_bf16_blackwellultra`                  | 144,200 gflop/s | 1,714 gflop/s |
| window 1024 `nk_attention_packed_bf16_blackwellultra`             |  95,040 gflop/s |   936 gflop/s |
| bidirectional `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`       | 240,300 gflop/s |   981 gflop/s |
| causal `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`              | 206,400 gflop/s |   982 gflop/s |
| window 1024 `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`         | 143,200 gflop/s |   881 gflop/s |
| bidirectional `nk_attention_packed_gradients_bf16_blackwell`      |  88,400 gflop/s |   479 gflop/s |
| causal `nk_attention_packed_gradients_bf16_blackwell`             |  75,500 gflop/s |   536 gflop/s |
| window 1024 `nk_attention_packed_gradients_bf16_blackwell`        |  63,090 gflop/s |   226 gflop/s |
| bidirectional `nk_attention_packed_gradients_bf16_blackwellultra` |  87,830 gflop/s |   582 gflop/s |
| causal `nk_attention_packed_gradients_bf16_blackwellultra`        |  72,480 gflop/s |   535 gflop/s |
| window 1024 `nk_attention_packed_gradients_bf16_blackwellultra`   |  50,740 gflop/s |   215 gflop/s |
| __f16__                                                           | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_f16_blackwell`                 | 154,100 gflop/s | 1,770 gflop/s |
| causal `nk_attention_packed_f16_blackwell`                        | 132,200 gflop/s | 1,697 gflop/s |
| window 1024 `nk_attention_packed_f16_blackwell`                   |  92,740 gflop/s |   927 gflop/s |
| bidirectional `nk_attention_packed_f16_blackwellultra`            | 160,400 gflop/s | 1,769 gflop/s |
| causal `nk_attention_packed_f16_blackwellultra`                   | 141,100 gflop/s | 1,699 gflop/s |
| window 1024 `nk_attention_packed_f16_blackwellultra`              |  94,720 gflop/s |   933 gflop/s |
| bidirectional `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`       | 240,400 gflop/s |   982 gflop/s |
| causal `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`              | 206,400 gflop/s |   981 gflop/s |
| window 1024 `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`         | 137,100 gflop/s |   883 gflop/s |
| __e4m3__                                                          | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_blackwell`                | 167,200 gflop/s | 1,864 gflop/s |
| causal `nk_attention_packed_e4m3_blackwell`                       | 136,400 gflop/s | 1,831 gflop/s |
| window 1024 `nk_attention_packed_e4m3_blackwell`                  |  99,720 gflop/s |   964 gflop/s |
| bidirectional `nk_attention_packed_e4m3_blackwellultra`           | 173,900 gflop/s | 1,864 gflop/s |
| causal `nk_attention_packed_e4m3_blackwellultra`                  | 143,700 gflop/s | 1,835 gflop/s |
| window 1024 `nk_attention_packed_e4m3_blackwellultra`             | 101,000 gflop/s |   973 gflop/s |
| bidirectional `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`       | 323,300 gflop/s |             ⋯ |
| causal `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`              | 243,700 gflop/s |             ⋯ |
| window 1024 `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`         | 166,100 gflop/s |             ⋯ |
| __i8__                                                            | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_blackwell`                  |  96,430 gflop/s | 1,359 gflop/s |
| causal `nk_attention_packed_i8_blackwell`                         |  78,090 gflop/s | 1,320 gflop/s |
| window 1024 `nk_attention_packed_i8_blackwell`                    |  53,710 gflop/s |   768 gflop/s |
| bidirectional `nk_attention_packed_i8_blackwellultra`             | 101,700 gflop/s | 1,358 gflop/s |
| causal `nk_attention_packed_i8_blackwellultra`                    |  82,320 gflop/s | 1,321 gflop/s |
| window 1024 `nk_attention_packed_i8_blackwellultra`               |  55,130 gflop/s |   767 gflop/s |

### AWS Graviton 4

#### Native

| Kernel                                             | 1024² | 4096² | 16384² |
| :------------------------------------------------- | ----: | ----: | -----: |
| __bf16__                                           | ░░░░░ | ░░░░░ | ░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`    |     ⋯ |     ⋯ |      ⋯ |
| bidirectional `nk_attention_packed_bf16_neonbfdot` |     ⋯ |     ⋯ |      ⋯ |
| __e4m3__                                           | ░░░░░ | ░░░░░ | ░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`    |     ⋯ |     ⋯ |      ⋯ |
| bidirectional `nk_attention_packed_e4m3_neonfhm`   |     ⋯ |     ⋯ |      ⋯ |
| __i8__                                             | ░░░░░ | ░░░░░ | ░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`      |     ⋯ |     ⋯ |      ⋯ |
| bidirectional `nk_attention_packed_i8_neonsdot`    |     ⋯ |     ⋯ |      ⋯ |

### Apple M5

#### Native

| Kernel                                                  |         1024² |         4096² |        16384² |
| :------------------------------------------------------ | ------------: | ------------: | ------------: |
| __bf16__                                                | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`         |   2.2 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_bf16_neon`           |    44 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_bf16_neonbfdot`      |    46 gflop/s |    47 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_bf16_sme`            |   780 gflop/s |   811 gflop/s |   790 gflop/s |
| bidirectional `nk_attention_packed_bf16_metal`          |   239 gflop/s |   226 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_bf16_apple9`         | 2,144 gflop/s | 1,713 gflop/s | 1,436 gflop/s |
| bidirectional `nk_attention_packed_bf16_apple10`        | 2,270 gflop/s | 1,841 gflop/s | 1,575 gflop/s |
| __f16__                                                 | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_f16_serial`          |   2.0 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_f16_neon`            |    44 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_f16_neonfhm`         |    50 gflop/s |    51 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_f16_sme`             |   784 gflop/s |   816 gflop/s |   699 gflop/s |
| bidirectional `nk_attention_packed_f16_metal`           |   246 gflop/s |   239 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_f16_apple9`          | 2,138 gflop/s | 1,704 gflop/s | 1,427 gflop/s |
| bidirectional `nk_attention_packed_f16_apple10`         | 2,279 gflop/s | 1,872 gflop/s | 1,606 gflop/s |
| __e4m3__                                                | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`         |   2.1 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_neon`           |    44 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_neonfhm`        |    51 gflop/s |    51 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_sme`            |   752 gflop/s |   802 gflop/s |   787 gflop/s |
| bidirectional `nk_attention_packed_e4m3_metal`          |   239 gflop/s |   260 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_apple9`         | 2,135 gflop/s | 1,876 gflop/s | 1,728 gflop/s |
| bidirectional `nk_attention_packed_e4m3_apple10`        | 2,295 gflop/s | 1,970 gflop/s | 1,870 gflop/s |
| __i8__                                                  | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`           |   7.1 gflop/s |   7.2 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_neon`             |   185 gflop/s |   190 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_neonsdot`         |   333 gflop/s |   353 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_sme`              |   829 gflop/s |   883 gflop/s |   797 gflop/s |
| bidirectional `nk_attention_packed_i8_metal`            |   219 gflop/s |   222 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_apple10`          | 1,610 gflop/s | 1,360 gflop/s | 1,330 gflop/s |
| __nvfp4__                                               | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_nvfp4_serial`        |   1.5 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_neon`          |    41 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_neonfhm`       |    51 gflop/s |    51 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_sme`           |   780 gflop/s |   803 gflop/s |   812 gflop/s |
| __mxfp4__                                               | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp4_serial`        |   3.2 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp4_neon`          |    40 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp4_neonbfdot`     |    45 gflop/s |    46 gflop/s |             ⋯ |
| __mxfp6e2m3__                                           | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp6e2m3_serial`    |   3.7 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e2m3_neon`      |    41 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e2m3_neonbfdot` |    46 gflop/s |    46 gflop/s |             ⋯ |
| __mxfp6e3m2__                                           | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp6e3m2_serial`    |   3.7 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e3m2_neon`      |    40 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e3m2_neonbfdot` |    43 gflop/s |    47 gflop/s |             ⋯ |
| __mxfp8e4m3__                                           | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp8e4m3_serial`    |   2.2 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e4m3_neon`      |    43 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e4m3_neonbfdot` |    45 gflop/s |    46 gflop/s |             ⋯ |
| __mxfp8e5m2__                                           | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp8e5m2_serial`    |   3.6 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e5m2_neon`      |    43 gflop/s |    45 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e5m2_neonbfdot` |    45 gflop/s |    47 gflop/s |             ⋯ |
