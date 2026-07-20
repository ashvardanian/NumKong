"""Tests for ragged scaled-dot-product attention: `attention_pack` and `attention_packed`.

The reference is a float64 NumPy softmax-attention over the same dtype-rounded inputs,
so tolerances only cover kernel arithmetic (BF16 tiles + F32 accumulation), not input rounding.
"""

import numpy as np
import pytest

import numkong as nk


ATTENTION_DTYPES = [
    ("bf16", 5e-3),
    ("e4m3", 5e-3),
]

SCENARIOS = {
    "ragged": [40, 90, 0, 17],
    "one-segment": [96],
    "tiny": [1, 2, 3],
}


def reference_attention(q_f64, k_f64, v_f64, offsets, lengths, num_heads, num_kv_heads, head_dim, scale):
    tokens = q_f64.shape[0]
    out = np.zeros((tokens, num_heads * head_dim), dtype=np.float64)
    gqa = num_heads // num_kv_heads
    for segment in range(len(lengths)):
        first, last = int(offsets[segment]), int(offsets[segment + 1])
        kv_len = int(lengths[segment])
        if first == last or kv_len == 0:
            continue
        for head in range(num_heads):
            kv_head = head // gqa
            queries = q_f64[first:last, head * head_dim : (head + 1) * head_dim]
            keys = k_f64[first : first + kv_len, kv_head * head_dim : (kv_head + 1) * head_dim]
            values = v_f64[first : first + kv_len, kv_head * head_dim : (kv_head + 1) * head_dim]
            scores = queries @ keys.T * scale
            probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            out[first:last, head * head_dim : (head + 1) * head_dim] = probabilities @ values
    return out


@pytest.mark.parametrize("dtype,tolerance", ATTENTION_DTYPES)
@pytest.mark.parametrize("scenario", SCENARIOS.keys())
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("threads", [1, 0])
def test_attention_packed(dtype, tolerance, scenario, head_dim, threads):
    lengths = SCENARIOS[scenario]
    offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    tokens, num_heads, num_kv_heads = int(offsets[-1]), 4, 2
    scale = 1.0 / np.sqrt(head_dim)
    np.random.seed(42)

    q_f32 = (np.random.randn(tokens, num_heads * head_dim) * 0.3).astype(np.float32)
    k_f32 = (np.random.randn(tokens, num_kv_heads * head_dim) * 0.3).astype(np.float32)
    v_f32 = (np.random.randn(tokens, num_kv_heads * head_dim) * 0.3).astype(np.float32)
    q, k, v = (nk.Tensor(x).astype(dtype) for x in (q_f32, k_f32, v_f32))

    kv = nk.attention_pack(k, v, segment_offsets=offsets, head_dim=head_dim, threads=threads)
    assert kv.segments == len(lengths)
    assert kv.heads == num_kv_heads
    assert kv.depth == head_dim
    assert kv.tokens == tokens

    out = nk.attention_packed(q, kv, query_offsets=offsets, threads=threads)
    result = np.from_dlpack(out)

    rounded = [np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v)]
    expected = reference_attention(*rounded, offsets, lengths, num_heads, num_kv_heads, head_dim, scale)
    np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype,tolerance", ATTENTION_DTYPES)
def test_attention_pool(dtype, tolerance):
    """One query per segment against the full segment KV — the batched pooling shape."""
    lengths = [33, 70, 5]
    kv_offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    pool_offsets = np.arange(len(lengths) + 1, dtype=np.uint32)
    tokens, num_heads, head_dim = int(kv_offsets[-1]), 4, 128
    scale = 1.0 / np.sqrt(head_dim)
    np.random.seed(7)

    q_f32 = (np.random.randn(len(lengths), num_heads * head_dim) * 0.3).astype(np.float32)
    k_f32 = (np.random.randn(tokens, num_heads * head_dim) * 0.3).astype(np.float32)
    v_f32 = (np.random.randn(tokens, num_heads * head_dim) * 0.3).astype(np.float32)
    q, k, v = (nk.Tensor(x).astype(dtype) for x in (q_f32, k_f32, v_f32))

    kv = nk.attention_pack(k, v, segment_offsets=kv_offsets, head_dim=head_dim, threads=1)
    out = nk.attention_packed(q, kv, query_offsets=pool_offsets, threads=1)
    result = np.from_dlpack(out)

    q_r, k_r, v_r = (np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v))
    for segment in range(len(lengths)):
        first, kv_len = int(kv_offsets[segment]), lengths[segment]
        for head in range(num_heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            scores = q_r[segment, sl] @ k_r[first : first + kv_len, sl].T * scale
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            expected = probabilities @ v_r[first : first + kv_len, sl]
            np.testing.assert_allclose(result[segment, sl], expected, atol=tolerance, rtol=tolerance)


def test_attention_i8():
    """I8 contract: exact integer scores, softmax weights quantized to u8, f32 outputs.

    The reference uses unquantized weights, so the tolerance is the u8 quantization
    noise floor (~1/255 of the value scale), not float rounding.
    """
    lengths = [40, 90, 0, 17]
    offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    tokens, num_heads, head_dim = int(offsets[-1]), 4, 128
    scale = 0.05 / np.sqrt(head_dim)
    np.random.seed(11)

    q = nk.Tensor(np.random.randint(-31, 32, (tokens, num_heads * head_dim)).astype(np.int8))
    k = nk.Tensor(np.random.randint(-31, 32, (tokens, num_heads * head_dim)).astype(np.int8))
    v = nk.Tensor(np.random.randint(-31, 32, (tokens, num_heads * head_dim)).astype(np.int8))

    kv = nk.attention_pack(k, v, segment_offsets=offsets, head_dim=head_dim, threads=0)
    out = nk.attention_packed(q, kv, query_offsets=offsets, scale=scale, threads=0)
    result = np.from_dlpack(out)

    rounded = [np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v)]
    expected = reference_attention(*rounded, offsets, lengths, num_heads, num_heads, head_dim, scale)
    value_scale = np.abs(expected).max()
    np.testing.assert_allclose(result, expected, atol=0.02 * value_scale)


def test_attention_validation():
    offsets = np.array([0, 4], dtype=np.uint32)
    matrix = nk.Tensor(np.zeros((4, 128), dtype=np.float32)).astype("bf16")
    kv = nk.attention_pack(matrix, matrix, segment_offsets=offsets, head_dim=128, threads=1)

    with pytest.raises(TypeError):
        nk.attention_packed(matrix, "not-packed", query_offsets=offsets)
    with pytest.raises(ValueError):  # wrong offsets length
        nk.attention_packed(matrix, kv, query_offsets=np.array([0, 2, 4], dtype=np.uint32))
    with pytest.raises(ValueError):  # offsets past the query token count
        nk.attention_packed(matrix, kv, query_offsets=np.array([0, 9], dtype=np.uint32))
    with pytest.raises(TypeError):  # missing head_dim for a 2-D input
        nk.attention_pack(matrix, matrix, segment_offsets=offsets)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
