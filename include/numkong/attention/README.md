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
| `bf16`      | `f32`       | 16-bit brain float                                                         |
| `f16`       | `f32`       | 16-bit IEEE half                                                           |
| `e4m3`      | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits                                  |
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
The backward's task grid is segments × key-value heads, each task owning one key-value head of one segment with every query head sharing it, so the gradients need no atomics and every sum runs in a fixed order.
Query gradient rows take their own `query_gradient_stride`, and key and value gradient rows past a segment's live keys, the gaps and tails before the next segment, are never written, so callers zero them when they matter.
Block-scaled gradients are taken with respect to the decoded values.

The one exception is the Blackwell single pass for BF16, which adds $dQ$ partials in completion order, so the low bits of $dQ$ may differ between runs while $dK$ and $dV$ stay deterministic.
It allocates a workspace of `query_token_count` × `head_count` × (2 × `depth` + 4) bytes on the caller's stream and frees it after the kernels, which keeps graph capture working; a caller that synchronizes between calls lets the default memory pool unmap it, unless `cudaMemPoolAttrReleaseThreshold` is raised.

## Optimizations

### Panel-Flash Sweep with L2-Resident Score Panels

All CPU backends sweep KV in panels of 512 positions: scores for a query block land in a scratch panel, one row-major pass computes the running maximum, exponentials, and weight sum, and the weighted V accumulation drains once per panel with the correction FMA fused in.
This bounds the vector-unit work to one crossover per score, which is the dominant cost on matrix-unit ISAs where tile registers support no elementwise math.
Matrix units drain score tiles position-major, one query per lane, so the softmax bookkeeping stays lane-parallel with no horizontal reductions, and each KV panel serves several query blocks, which holds throughput flat as the context outgrows L2.

### Storage and Weight Formats

Packing converts K and V only into the format the ISA multiplies, as in the `dots` family, since at-rest bytes are memory bandwidth: FMA tiers keep BF16 and F16 raw and widen in the loop, and E4M3 becomes F16 or BF16 where a dot instruction consumes it.
Probabilities enter P · V in the same format as V on matrix, tile and widening instructions, F16 on NEON FHM, SME and Ampere, BF16 on AMX and SME, E4M3 on Blackwell, while FMA tiers keep them in F32 and integer tiers quantize them to U8.

### One Recipe for Block-Scaled Formats

An NVFP4 element times its UE4M3 block scale is an exact F16, so NVFP4 planes store those products and run each tier's F16 path, with both tensor scales in the pack header.
MX scales span 254 binades, so each MX plane rebases its elements by its largest finite block exponent less 31, exact in BF16, runs each tier's BF16 path, and records that exponent in a small table past the planes.
A plane whose block exponents span more than 89 binades keeps its raw codes and scale codes, and it and any query row past the window take serial's exact path outside the fast loop.
Scores sum each query block in F32 and add the blocks into a wide sum, so query scales of any spread round once, and outputs apply their plane's power of two once per element.

### Split Decode for Short Outputs

A decode call has fewer rows than the GPU has cores, so the GPUs split each row's keys and merge the partial maxima, sums and weighted values in a fixed order, keeping results identical between runs.
Blackwell pairs blocks in a cluster that merges through distributed shared memory when a call has at most half as many items as multiprocessors, and Metal cuts calls of at most four rows against 512 keys or more into 16 partitions.

### U8-Quantized Probabilities for INT8

The `i8` kernels quantize softmax weights as $\tilde{w} = \text{round}(255 \cdot 2^{s_2 - m_2})$ and normalize by $\sum \tilde{w}$, so the 255 cancels and no descale constant survives.
The maximum-scoring position always quantizes to exactly 255, making the weight sum provably non-zero.
Scores stay exact in `i32` integer arithmetic; only the probabilities round.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology) on one core pinned with `numactl --membind=0 taskset -c <core>`, counting $4 \cdot h \cdot n_q \cdot n_{kv} \cdot d$ FLOPs, or $10 \cdot h \cdot n_q \cdot n_{kv} \cdot d$ for gradient rows.
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
Gradient rows time the backward alone.
The `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR` rows time cuDNN 9.27's fused forward over the same inputs; its backward descriptor has no engine on this GPU.
Cells marked `✗` are shapes the library rejects: cuDNN returns `CUDNN_STATUS_NOT_SUPPORTED` for E4M3 decode.

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
| bidirectional `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`       | 323,300 gflop/s |             ✗ |
| causal `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`              | 243,700 gflop/s |             ✗ |
| window 1024 `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`         | 166,100 gflop/s |             ✗ |
| __i8__                                                            | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_blackwell`                  | 127,300 gflop/s | 1,621 gflop/s |
| causal `nk_attention_packed_i8_blackwell`                         |  97,240 gflop/s | 1,550 gflop/s |
| window 1024 `nk_attention_packed_i8_blackwell`                    |  62,120 gflop/s |   822 gflop/s |
| bidirectional `nk_attention_packed_i8_blackwellultra`             | 134,100 gflop/s | 1,622 gflop/s |
| causal `nk_attention_packed_i8_blackwellultra`                    | 100,900 gflop/s | 1,601 gflop/s |
| window 1024 `nk_attention_packed_i8_blackwellultra`               |  63,380 gflop/s |   836 gflop/s |

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

| Kernel                                                     |         1024² |         4096² |        16384² |
| :--------------------------------------------------------- | ------------: | ------------: | ------------: |
| __bf16__                                                   | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_bf16_serial`            |   2.2 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_bf16_neon`              |    44 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_bf16_neonbfdot`         |    46 gflop/s |    47 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_bf16_sme`               |   780 gflop/s |   811 gflop/s |   790 gflop/s |
| bidirectional `nk_attention_packed_bf16_metal`             |   239 gflop/s |   226 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_bf16_apple9`            | 2,144 gflop/s | 1,713 gflop/s | 1,436 gflop/s |
| bidirectional `nk_attention_packed_bf16_apple10`           | 2,270 gflop/s | 1,841 gflop/s | 1,575 gflop/s |
| bidirectional `nk_attention_packed_gradients_bf16_metal`   |   211 gflop/s |   207 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_gradients_bf16_apple9`  |   477 gflop/s |   462 gflop/s |   459 gflop/s |
| bidirectional `nk_attention_packed_gradients_bf16_apple10` |   475 gflop/s |   476 gflop/s |   468 gflop/s |
| __f16__                                                    | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_f16_serial`             |   2.0 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_f16_neon`               |    44 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_f16_neonfhm`            |    50 gflop/s |    51 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_f16_sme`                |   784 gflop/s |   816 gflop/s |   699 gflop/s |
| bidirectional `nk_attention_packed_f16_metal`              |   246 gflop/s |   239 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_f16_apple9`             | 2,138 gflop/s | 1,704 gflop/s | 1,427 gflop/s |
| bidirectional `nk_attention_packed_f16_apple10`            | 2,279 gflop/s | 1,872 gflop/s | 1,606 gflop/s |
| __e4m3__                                                   | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_e4m3_serial`            |   2.1 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_neon`              |    44 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_neonfhm`           |    51 gflop/s |    51 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_sme`               |   752 gflop/s |   802 gflop/s |   787 gflop/s |
| bidirectional `nk_attention_packed_e4m3_metal`             |   239 gflop/s |   260 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_e4m3_apple9`            | 2,135 gflop/s | 1,876 gflop/s | 1,728 gflop/s |
| bidirectional `nk_attention_packed_e4m3_apple10`           | 2,295 gflop/s | 1,970 gflop/s | 1,870 gflop/s |
| __i8__                                                     | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_i8_serial`              |   7.1 gflop/s |   7.2 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_neon`                |   185 gflop/s |   190 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_neonsdot`            |   333 gflop/s |   353 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_sme`                 |   829 gflop/s |   883 gflop/s |   797 gflop/s |
| bidirectional `nk_attention_packed_i8_metal`               |   219 gflop/s |   222 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_i8_apple10`             | 1,610 gflop/s | 1,360 gflop/s | 1,330 gflop/s |
| __nvfp4__                                                  | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_nvfp4_serial`           |   1.5 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_neon`             |    41 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_neonfhm`          |    51 gflop/s |    51 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_sme`              |   780 gflop/s |   803 gflop/s |   812 gflop/s |
| bidirectional `nk_attention_packed_nvfp4_metal`            |   239 gflop/s |   233 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_nvfp4_apple9`           | 2,010 gflop/s | 1,712 gflop/s | 1,484 gflop/s |
| bidirectional `nk_attention_packed_nvfp4_apple10`          | 2,199 gflop/s | 1,828 gflop/s | 1,574 gflop/s |
| __mxfp4__                                                  | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp4_serial`           |   3.2 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp4_neon`             |    40 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp4_neonbfdot`        |    45 gflop/s |    46 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp4_sme`              |   771 gflop/s |   812 gflop/s |   818 gflop/s |
| bidirectional `nk_attention_packed_mxfp4_metal`            |   233 gflop/s |   226 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp4_apple9`           | 1,938 gflop/s | 1,680 gflop/s | 1,554 gflop/s |
| bidirectional `nk_attention_packed_mxfp4_apple10`          | 2,050 gflop/s | 1,760 gflop/s | 1,655 gflop/s |
| __mxfp6e2m3__                                              | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp6e2m3_serial`       |   3.7 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e2m3_neon`         |    41 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e2m3_neonbfdot`    |    46 gflop/s |    46 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e2m3_sme`          |   776 gflop/s |   816 gflop/s |   823 gflop/s |
| bidirectional `nk_attention_packed_mxfp6e2m3_metal`        |   241 gflop/s |   238 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e2m3_apple9`       | 1,976 gflop/s | 1,710 gflop/s | 1,506 gflop/s |
| bidirectional `nk_attention_packed_mxfp6e2m3_apple10`      | 2,097 gflop/s | 1,819 gflop/s | 1,646 gflop/s |
| __mxfp6e3m2__                                              | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp6e3m2_serial`       |   3.7 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e3m2_neon`         |    40 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e3m2_neonbfdot`    |    43 gflop/s |    47 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e3m2_sme`          |   776 gflop/s |   814 gflop/s |   811 gflop/s |
| bidirectional `nk_attention_packed_mxfp6e3m2_metal`        |   245 gflop/s |   244 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp6e3m2_apple9`       | 1,981 gflop/s | 1,730 gflop/s | 1,476 gflop/s |
| bidirectional `nk_attention_packed_mxfp6e3m2_apple10`      | 2,087 gflop/s | 1,811 gflop/s | 1,629 gflop/s |
| __mxfp8e4m3__                                              | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp8e4m3_serial`       |   2.2 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e4m3_neon`         |    43 gflop/s |    44 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e4m3_neonbfdot`    |    45 gflop/s |    46 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e4m3_sme`          |   776 gflop/s |   815 gflop/s |   817 gflop/s |
| bidirectional `nk_attention_packed_mxfp8e4m3_metal`        |   244 gflop/s |   241 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e4m3_apple9`       | 1,982 gflop/s | 1,706 gflop/s | 1,491 gflop/s |
| bidirectional `nk_attention_packed_mxfp8e4m3_apple10`      | 2,095 gflop/s | 1,806 gflop/s | 1,582 gflop/s |
| __mxfp8e5m2__                                              | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ | ░░░░░░░░░░░░░ |
| bidirectional `nk_attention_packed_mxfp8e5m2_serial`       |   3.6 gflop/s |             ⋯ |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e5m2_neon`         |    43 gflop/s |    45 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e5m2_neonbfdot`    |    45 gflop/s |    47 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e5m2_sme`          |   774 gflop/s |   815 gflop/s |   820 gflop/s |
| bidirectional `nk_attention_packed_mxfp8e5m2_metal`        |   240 gflop/s |   238 gflop/s |             ⋯ |
| bidirectional `nk_attention_packed_mxfp8e5m2_apple9`       | 1,979 gflop/s | 1,723 gflop/s | 1,524 gflop/s |
| bidirectional `nk_attention_packed_mxfp8e5m2_apple10`      | 2,100 gflop/s | 1,828 gflop/s | 1,664 gflop/s |
