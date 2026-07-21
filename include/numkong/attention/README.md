# Ragged Scaled-Dot-Product Attention in NumKong

NumKong implements FlashAttention-style scaled-dot-product attention __(SDPA)__ over ragged batches with a pre-packed KV-cache, as the fused core of Transformer inference on CPUs.
Every backend shares one public triple — `pack_size` to size the cache, `pack` to rearrange K/V into a backend-opaque layout, and `packed` to compute attention against it — with `(begin, end)` half-open windows over the segment × head grid for embarrassingly parallel execution.

For a single segment and head, the operation is the classic softmax attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Ragged batches generalize this to independently sized segments sharing one packed cache, and decoupled `query_offsets` turn the same cache into a cross-attention or batched single-query pooling operator.
Reformulating as Python pseudocode:

```python
def attention_ragged(q, k, v, segment_offsets, query_offsets, scale) -> Matrix:
    out = zeros(rows=query_offsets[-1], cols=q.cols)
    for s in range(len(segment_offsets) - 1):
        kv = slice(segment_offsets[s], segment_offsets[s + 1])
        queries = slice(query_offsets[s], query_offsets[s + 1])
        for h in range(num_heads):
            kv_head = h // (num_heads // num_kv_heads)  # GQA / MQA sharing
            scores = q[queries, h] @ k[kv, kv_head].T * scale
            out[queries, h] = softmax(scores, axis=-1) @ v[kv, kv_head]
    return out
```

Internally every backend uses the streaming base-2 softmax: the scale folds $\log_2 e$, so the exponentials become `exp2` with one shared degree-4 polynomial after exact range reduction, and multi-panel sweeps carry a running maximum with the correction $O \leftarrow O \cdot 2^{m_{old} - m_{new}} + O_{panel}$.

## Input & Output Types

| Input Type | Output Type | Description                                                                |
| ---------- | ----------- | -------------------------------------------------------------------------- |
| `bf16`     | `f32`       | 16-bit brain float; native AMX tiles and `VDPBF16PS` lanes                 |
| `e4m3`     | `f32`       | 8-bit Float8; widened to the ISA's compute format at the pack boundary     |
| `i8`       | `f32`       | 8-bit signed integers; exact `i32` scores, probabilities quantized to `u8` |

Shape envelope: any `head_dim ≥ 1` (SIMD fast paths cover 1…256 with zero-padded channels; wider heads route to the width-agnostic serial tier), arbitrary segment lengths including empty PAD segments, and any integer GQA ratio.
Quantization scales fold into the `scale` argument for `i8` (queries and keys) or stay with the caller (values), so all three dtypes share one signature.

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

### U8-Quantized Probabilities for INT8

The `i8` kernels quantize softmax weights as $\tilde{w} = \text{round}(255 \cdot 2^{s_2 - m_2})$ and normalize by $\sum \tilde{w}$, so the 255 cancels and no descale constant survives.
The maximum-scoring position always quantizes to exactly 255, making the weight sum provably non-zero.
Scores stay exact in `i32` integer arithmetic; only the probabilities round.

## Performance

The tables below follow the house methodology: pinned single-core runs (`numactl --membind=0 taskset -c <core>`), `4 \cdot h \cdot n_q \cdot n_{kv} \cdot d` FLOP accounting, and separate passes for frequency-heavy AMX workloads.
Rows are kernels, columns are square self-attention shapes at `head_dim = 128`, 8 heads; accuracy is the maximum absolute error against an `f64` reference over dtype-rounded inputs.
Cells marked `⋯` await measurement on the corresponding platform.

### Intel Sapphire Rapids

#### Native

| Kernel                                 |                 1024² |                 4096² |                16384² |
| :------------------------------------- | --------------------: | --------------------: | --------------------: |
| __bf16__                               | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| `nk_attention_packed_bf16_serial`      |           0.6 gflop/s |                     ⋯ |                     ⋯ |
| `nk_attention_packed_bf16_haswell`     |            36 gflop/s |            33 gflop/s |            25 gflop/s |
| `nk_attention_packed_bf16_skylake`     |            48 gflop/s |            41 gflop/s |            28 gflop/s |
| `nk_attention_packed_bf16_genoa`       |            54 gflop/s |            47 gflop/s |            30 gflop/s |
| `nk_attention_packed_bf16_sapphireamx` |           845 gflop/s |           827 gflop/s |           766 gflop/s |
| __e4m3__                               | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| `nk_attention_packed_e4m3_serial`      |           1.2 gflop/s |                     ⋯ |                     ⋯ |
| `nk_attention_packed_e4m3_haswell`     |            11 gflop/s |            11 gflop/s |            11 gflop/s |
| `nk_attention_packed_e4m3_skylake`     |            49 gflop/s |            41 gflop/s |            29 gflop/s |
| `nk_attention_packed_e4m3_genoa`       |            57 gflop/s |            46 gflop/s |            30 gflop/s |
| `nk_attention_packed_e4m3_sapphireamx` |           840 gflop/s |           841 gflop/s |           786 gflop/s |
| __i8__                                 | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| `nk_attention_packed_i8_serial`        |           1.2 gflop/s |                     ⋯ |                     ⋯ |
| `nk_attention_packed_i8_haswell`       |           123 gflop/s |           125 gflop/s |           113 gflop/s |
| `nk_attention_packed_i8_icelake`       |           185 gflop/s |           186 gflop/s |           171 gflop/s |
| `nk_attention_packed_i8_sapphireamx`   |           941 gflop/s |           950 gflop/s |           931 gflop/s |

#### WASM

Measured with Wasmtime v24 (Cranelift backend).

| Kernel                                 |                 1024² |                 4096² |                16384² |
| :------------------------------------- | --------------------: | --------------------: | --------------------: |
| __bf16__                               | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| `nk_attention_packed_bf16_serial`      |          0.60 gflop/s |                     ⋯ |                     ⋯ |
| `nk_attention_packed_bf16_v128relaxed` |            10 gflop/s |           9.9 gflop/s |            11 gflop/s |
| __e4m3__                               | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| `nk_attention_packed_e4m3_serial`      |          0.53 gflop/s |                     ⋯ |                     ⋯ |
| `nk_attention_packed_e4m3_v128relaxed` |           3.3 gflop/s |           3.0 gflop/s |           3.7 gflop/s |
| __i8__                                 | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ |
| `nk_attention_packed_i8_serial`        |           5.1 gflop/s |                     ⋯ |                     ⋯ |
| `nk_attention_packed_i8_v128relaxed`   |          24.6 gflop/s |          24.1 gflop/s |          29.2 gflop/s |

### AWS Graviton 4

#### Native

| Kernel                               | 1024² | 4096² | 16384² |
| :----------------------------------- | ----: | ----: | -----: |
| __bf16__                             | ░░░░░ | ░░░░░ | ░░░░░░ |
| `nk_attention_packed_bf16_serial`    |     ⋯ |     ⋯ |      ⋯ |
| `nk_attention_packed_bf16_neonbfdot` |     ⋯ |     ⋯ |      ⋯ |
| __e4m3__                             | ░░░░░ | ░░░░░ | ░░░░░░ |
| `nk_attention_packed_e4m3_serial`    |     ⋯ |     ⋯ |      ⋯ |
| `nk_attention_packed_e4m3_neonfhm`   |     ⋯ |     ⋯ |      ⋯ |
| __i8__                               | ░░░░░ | ░░░░░ | ░░░░░░ |
| `nk_attention_packed_i8_serial`      |     ⋯ |     ⋯ |      ⋯ |
| `nk_attention_packed_i8_neonsdot`    |     ⋯ |     ⋯ |      ⋯ |

### Apple M5

#### Native

| Kernel                               |        1024² |        4096² |       16384² |
| :----------------------------------- | -----------: | -----------: | -----------: |
| __bf16__                             | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_attention_packed_bf16_serial`    |  2.8 gflop/s |            ⋯ |            ⋯ |
| `nk_attention_packed_bf16_neonbfdot` |   46 gflop/s |   47 gflop/s |   46 gflop/s |
| `nk_attention_packed_bf16_sme`       |  789 gflop/s |  800 gflop/s |  802 gflop/s |
| __e4m3__                             | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_attention_packed_e4m3_serial`    |  1.5 gflop/s |            ⋯ |            ⋯ |
| `nk_attention_packed_e4m3_neonfhm`   |   51 gflop/s |   51 gflop/s |   50 gflop/s |
| `nk_attention_packed_e4m3_sme`       |  722 gflop/s |  777 gflop/s |  797 gflop/s |
| __i8__                               | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_attention_packed_i8_serial`      |  133 gflop/s |            ⋯ |            ⋯ |
| `nk_attention_packed_i8_neonsdot`    |  310 gflop/s |  305 gflop/s |  285 gflop/s |
| `nk_attention_packed_i8_sme`         |  815 gflop/s |  825 gflop/s |  827 gflop/s |
