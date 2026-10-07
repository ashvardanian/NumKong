"""Tests for ragged scaled-dot-product attention: `attention_pack`, `attention_packed` bands, and RoPE.

The reference is a float64 NumPy softmax-attention over the same dtype-rounded inputs and an
explicit visibility mask, so tolerances only cover kernel arithmetic, not input rounding.

File: test/attention.py
Author: Ash Vardanian
Date: July 7, 2026
"""

import numpy as np
import pytest
from base import (
    assert_allclose,
    make_nk,
    make_random,
    tolerances_for_dtype,
)

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

BANDS = [(None, None), (None, 0), (0, 0), (6, 0), (32, 0), (3, 5)]
"""(keys_before, keys_after) pairs; None is unbounded: bidirectional, causal, three sliding windows, two-sided."""


def visibility_mask(query_count, key_count, keys_before, keys_after):
    """Boolean `[query_count, key_count]` mask; queries align to the end of the keys.

    Row r sits at position p = r + key_count - query_count and sees key j iff
    p - keys_before <= j <= p + keys_after.
    """
    positions = np.arange(query_count)[:, None] + key_count - query_count
    keys = np.arange(key_count)[None, :]
    visible = np.ones((query_count, key_count), dtype=bool)
    if keys_before is not None:
        visible &= keys >= positions - keys_before
    if keys_after is not None:
        visible &= keys <= positions + keys_after
    return visible


def reference_attention(
    q_f64,
    k_f64,
    v_f64,
    query_offsets,
    key_offsets,
    lengths,
    head_count,
    key_value_head_count,
    depth,
    scale,
    keys_before=None,
    keys_after=None,
):
    out = np.zeros((q_f64.shape[0], head_count * depth), dtype=np.float64)
    gqa = head_count // key_value_head_count
    for segment in range(len(lengths)):
        first, last = int(query_offsets[segment]), int(query_offsets[segment + 1])
        key_first, kv_len = int(key_offsets[segment]), int(lengths[segment])
        visible = visibility_mask(last - first, kv_len, keys_before, keys_after)
        for head in range(head_count):
            kv_head = head // gqa
            queries = q_f64[first:last, head * depth : (head + 1) * depth]
            keys = k_f64[key_first : key_first + kv_len, kv_head * depth : (kv_head + 1) * depth]
            values = v_f64[key_first : key_first + kv_len, kv_head * depth : (kv_head + 1) * depth]
            scores = np.where(visible, queries @ keys.T * scale, -np.inf)
            row_max = scores.max(axis=1, keepdims=True, initial=-np.inf)
            shifted = np.where(visible, scores - np.where(np.isfinite(row_max), row_max, 0.0), 0.0)
            probabilities = np.where(visible, np.exp(shifted), 0.0)
            sums = probabilities.sum(axis=1, keepdims=True)
            probabilities = np.divide(probabilities, sums, out=np.zeros_like(probabilities), where=sums > 0)
            out[first:last, head * depth : (head + 1) * depth] = probabilities @ values
    return out


@pytest.mark.parametrize("dtype,tolerance", ATTENTION_DTYPES)
@pytest.mark.parametrize("scenario", SCENARIOS.keys())
@pytest.mark.parametrize("depth", [64, 128])
@pytest.mark.parametrize("threads", [1, 0])
def test_attention_packed(dtype, tolerance, scenario, depth, threads, np_rng: np.random.Generator):
    lengths = SCENARIOS[scenario]
    offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    tokens, head_count, key_value_head_count = int(offsets[-1]), 4, 2
    scale = 1.0 / np.sqrt(depth)

    q_f32 = (np_rng.standard_normal((tokens, head_count * depth)) * 0.3).astype(np.float32)
    k_f32 = (np_rng.standard_normal((tokens, key_value_head_count * depth)) * 0.3).astype(np.float32)
    v_f32 = (np_rng.standard_normal((tokens, key_value_head_count * depth)) * 0.3).astype(np.float32)
    q, k, v = (nk.Tensor(x).astype(dtype) for x in (q_f32, k_f32, v_f32))

    kv = nk.attention_pack(k, v, segment_offsets=offsets, depth=depth, threads=threads)
    assert kv.segments == len(lengths)
    assert kv.heads == key_value_head_count
    assert kv.depth == depth
    assert kv.tokens == tokens

    rounded = [np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v)]
    for keys_before, keys_after in BANDS:
        out = nk.attention_packed(
            q, kv, query_offsets=offsets, keys_before=keys_before, keys_after=keys_after, threads=threads
        )
        result = np.from_dlpack(out)
        expected = reference_attention(
            *rounded,
            offsets,
            offsets,
            lengths,
            head_count,
            key_value_head_count,
            depth,
            scale,
            keys_before,
            keys_after,
        )
        np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype,tolerance", ATTENTION_DTYPES)
@pytest.mark.parametrize("keys_before", [None, 4])
def test_attention_causal_decode(dtype, tolerance, keys_before, np_rng: np.random.Generator):
    """A few trailing queries per segment against a longer cache: queries align to the end of the keys."""
    length, query_count, segment_count, head_count, depth = 37, 3, 2, 4, 64
    key_offsets = np.arange(segment_count + 1, dtype=np.uint32) * length
    query_offsets = np.arange(segment_count + 1, dtype=np.uint32) * query_count
    lengths = [length] * segment_count
    scale = 1.0 / np.sqrt(depth)

    q_f32 = (np_rng.standard_normal((segment_count * query_count, head_count * depth)) * 0.3).astype(np.float32)
    k_f32 = (np_rng.standard_normal((segment_count * length, head_count * depth)) * 0.3).astype(np.float32)
    v_f32 = (np_rng.standard_normal((segment_count * length, head_count * depth)) * 0.3).astype(np.float32)
    q, k, v = (nk.Tensor(x).astype(dtype) for x in (q_f32, k_f32, v_f32))

    kv = nk.attention_pack(k, v, segment_offsets=key_offsets, depth=depth, threads=1)
    out = nk.attention_packed(q, kv, query_offsets=query_offsets, keys_before=keys_before, keys_after=0, threads=1)
    result = np.from_dlpack(out)

    rounded = [np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v)]
    expected = reference_attention(
        *rounded,
        query_offsets,
        key_offsets,
        lengths,
        head_count,
        head_count,
        depth,
        scale,
        keys_before,
        0,
    )
    np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype,tolerance", ATTENTION_DTYPES)
def test_attention_pool(dtype, tolerance, np_rng: np.random.Generator):
    """One query per segment against the full segment KV — the batched pooling shape."""
    lengths = [33, 70, 5]
    kv_offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    pool_offsets = np.arange(len(lengths) + 1, dtype=np.uint32)
    tokens, head_count, depth = int(kv_offsets[-1]), 4, 128
    scale = 1.0 / np.sqrt(depth)

    q_f32 = (np_rng.standard_normal((len(lengths), head_count * depth)) * 0.3).astype(np.float32)
    k_f32 = (np_rng.standard_normal((tokens, head_count * depth)) * 0.3).astype(np.float32)
    v_f32 = (np_rng.standard_normal((tokens, head_count * depth)) * 0.3).astype(np.float32)
    q, k, v = (nk.Tensor(x).astype(dtype) for x in (q_f32, k_f32, v_f32))

    kv = nk.attention_pack(k, v, segment_offsets=kv_offsets, depth=depth, threads=1)
    out = nk.attention_packed(q, kv, query_offsets=pool_offsets, threads=1)
    result = np.from_dlpack(out)

    q_r, k_r, v_r = (np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v))
    for segment in range(len(lengths)):
        first, kv_len = int(kv_offsets[segment]), lengths[segment]
        for head in range(head_count):
            sl = slice(head * depth, (head + 1) * depth)
            scores = q_r[segment, sl] @ k_r[first : first + kv_len, sl].T * scale
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            expected = probabilities @ v_r[first : first + kv_len, sl]
            np.testing.assert_allclose(result[segment, sl], expected, atol=tolerance, rtol=tolerance)


def test_attention_i8(np_rng: np.random.Generator):
    """I8 contract: exact integer scores, softmax weights quantized to u8, f32 outputs.

    The reference uses unquantized weights, so the tolerance is the u8 quantization
    noise floor (~1/255 of the value scale), not float rounding.
    """
    lengths = [40, 90, 0, 17]
    offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    tokens, head_count, depth = int(offsets[-1]), 4, 128
    scale = 0.05 / np.sqrt(depth)

    q = nk.Tensor(np_rng.integers(-31, 32, (tokens, head_count * depth)).astype(np.int8))
    k = nk.Tensor(np_rng.integers(-31, 32, (tokens, head_count * depth)).astype(np.int8))
    v = nk.Tensor(np_rng.integers(-31, 32, (tokens, head_count * depth)).astype(np.int8))

    kv = nk.attention_pack(k, v, segment_offsets=offsets, depth=depth, threads=0)
    rounded = [np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v)]
    for keys_before, keys_after in BANDS:
        out = nk.attention_packed(
            q, kv, query_offsets=offsets, scale=scale, keys_before=keys_before, keys_after=keys_after, threads=0
        )
        result = np.from_dlpack(out)
        expected = reference_attention(
            *rounded, offsets, offsets, lengths, head_count, head_count, depth, scale, keys_before, keys_after
        )
        value_scale = np.abs(expected).max()
        np.testing.assert_allclose(result, expected, atol=0.02 * value_scale)


def reference_attention_gradients(
    q_f64,
    k_f64,
    v_f64,
    output_gradient,
    query_offsets,
    key_offsets,
    head_count,
    key_value_head_count,
    depth,
    scale,
    band,
):
    """Float64 query, key and value gradients of banded softmax attention, straight from the definition."""
    query_gradient, key_gradient, value_gradient = np.zeros_like(q_f64), np.zeros_like(k_f64), np.zeros_like(v_f64)
    gqa = head_count // key_value_head_count
    for segment in range(len(key_offsets) - 1):
        first, last = int(query_offsets[segment]), int(query_offsets[segment + 1])
        key_first, key_last = int(key_offsets[segment]), int(key_offsets[segment + 1])
        visible = visibility_mask(last - first, key_last - key_first, *band)
        for head in range(head_count):
            query_columns = slice(head * depth, (head + 1) * depth)
            key_columns = slice(head // gqa * depth, (head // gqa + 1) * depth)
            queries = q_f64[first:last, query_columns]
            keys, values = k_f64[key_first:key_last, key_columns], v_f64[key_first:key_last, key_columns]
            scores = np.where(visible, queries @ keys.T * scale, -np.inf)
            row_max = scores.max(axis=1, keepdims=True, initial=-np.inf)
            weights = np.where(visible, np.exp(scores - np.where(np.isfinite(row_max), row_max, 0.0)), 0.0)
            sums = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)
            gradient = output_gradient[first:last, query_columns]
            row_dots = (gradient * (weights @ values)).sum(axis=1, keepdims=True)
            score_gradient = weights * (gradient @ values.T - row_dots) * scale
            query_gradient[first:last, query_columns] = score_gradient @ keys
            key_gradient[key_first:key_last, key_columns] += score_gradient.T @ queries
            value_gradient[key_first:key_last, key_columns] += weights.T @ gradient
    return query_gradient, key_gradient, value_gradient


@pytest.mark.parametrize("band", BANDS)
def test_attention_packed_gradients(band, np_rng: np.random.Generator):
    """BF16 gradients fed the serial forward's own output and log-sum-exp, over segments that offset
    their queries differently, one without keys, against a float64 reference."""
    lengths, query_counts = [40, 0, 17, 5], [40, 2, 1, 9]
    key_offsets = np.array([0, *np.cumsum(lengths)], dtype=np.uint32)
    query_offsets = np.array([0, *np.cumsum(query_counts)], dtype=np.uint32)
    query_tokens, key_tokens = int(query_offsets[-1]), int(key_offsets[-1])
    head_count, key_value_head_count, depth = 4, 2, 64
    scale = 1.0 / np.sqrt(depth)
    keys_before, keys_after = band

    q = nk.Tensor((np_rng.standard_normal((query_tokens, head_count * depth)) * 0.3).astype(np.float32)).astype("bf16")
    k, v = (
        nk.Tensor((np_rng.standard_normal((key_tokens, key_value_head_count * depth)) * 0.3).astype(np.float32)).astype(
            "bf16"
        )
        for _ in range(2)
    )
    output_gradient = (np_rng.standard_normal((query_tokens, head_count * depth)) * 0.3).astype(np.float32)
    kv = nk.attention_pack(k, v, segment_offsets=key_offsets, depth=depth, capabilities=nk.Capability.SERIAL)

    log_sum_exp = np.empty((query_tokens, head_count), dtype=np.float32)
    out = nk.attention_packed(
        q, kv, query_offsets=query_offsets, keys_before=keys_before, keys_after=keys_after, log_sum_exp=log_sum_exp
    )
    # The query gradient lands in a slice of a wider buffer, so its row stride exceeds the output's.
    query_gradient_buffer = nk.Tensor(np.full((query_tokens, head_count * depth + 4), np.nan, dtype=np.float32))
    gradients = nk.attention_packed_gradients(
        q,
        kv,
        query_offsets=query_offsets,
        key_offsets=key_offsets,
        output=out,
        output_gradient=output_gradient,
        log_sum_exp=log_sum_exp,
        keys_before=keys_before,
        keys_after=keys_after,
        query_gradient=query_gradient_buffer[:, : head_count * depth],
        threads=0,
    )
    rounded = [np.from_dlpack(t.astype("f32")).astype(np.float64) for t in (q, k, v)]
    expected = reference_attention_gradients(
        *rounded, output_gradient, query_offsets, key_offsets, head_count, key_value_head_count, depth, scale, band
    )
    for result, reference in zip(gradients, expected):
        np.testing.assert_allclose(np.from_dlpack(result), reference, atol=1e-4 * max(1.0, np.abs(reference).max()))
    assert np.isnan(np.from_dlpack(query_gradient_buffer)[:, head_count * depth :]).all()


def test_attention_validation():
    offsets = np.array([0, 4], dtype=np.uint32)
    matrix = nk.Tensor(np.zeros((4, 128), dtype=np.float32)).astype("bf16")
    kv = nk.attention_pack(matrix, matrix, segment_offsets=offsets, depth=128, threads=1)

    with pytest.raises(TypeError):
        nk.attention_packed(matrix, "not-packed", query_offsets=offsets)
    with pytest.raises(ValueError):  # wrong offsets length
        nk.attention_packed(matrix, kv, query_offsets=np.array([0, 2, 4], dtype=np.uint32))
    with pytest.raises(ValueError):  # offsets past the query token count
        nk.attention_packed(matrix, kv, query_offsets=np.array([0, 9], dtype=np.uint32), keys_after=0)
    with pytest.raises(TypeError):  # the old diagonal and window keywords are gone
        nk.attention_packed(matrix, kv, query_offsets=offsets, window=4)
    with pytest.raises(TypeError):  # missing depth for a 2-D input
        nk.attention_pack(matrix, matrix, segment_offsets=offsets)


def baseline_rope(x, cos, sin, head_count, depth):
    """NumPy float64 reference for NeoX split-half RoPE over the rounded input."""
    xf = np.asarray(x, dtype=np.float64)
    y = xf.copy()
    for r in range(xf.shape[0]):
        cosine, sine = cos[r].astype(np.float64), sin[r].astype(np.float64)
        for h in range(head_count):
            b, half_depth = h * depth, depth // 2
            low, high = xf[r, b : b + half_depth], xf[r, b + half_depth : b + depth]
            y[r, b : b + half_depth] = low * cosine - high * sine
            y[r, b + half_depth : b + depth] = low * sine + high * cosine
    return y


# (rows, head_count, depth)
@pytest.mark.parametrize("geom", [(4, 1, 8), (3, 2, 16), (5, 1, 128), (2, 3, 40)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float32", id="f32"),
        pytest.param("bf16", id="bf16"),
        pytest.param("e4m3", id="e4m3"),
    ],
)
def test_attention_rope(capabilities, geom, dtype, np_rng: np.random.Generator):
    """Test nk.attention_rope() out-of-place and in-place (out == x) against a float64 rotate-half reference."""
    rows, head_count, depth = geom
    width = head_count * depth
    angles = np_rng.standard_normal((rows, depth // 2)).astype(np.float32) * 0.5
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    x_raw, x_base = make_random((rows, width), dtype, np_rng)

    expected = baseline_rope(x_base, cos, sin, head_count, depth)
    if dtype != "float32":  # round the reference through the lossy output dtype
        expected = np.asarray(
            nk.Tensor(np.ascontiguousarray(expected.astype(np.float32))).astype(dtype).astype("float32")
        ).astype(np.float64)
    atol, rtol = tolerances_for_dtype(dtype)

    # Out-of-place: rotate x into a separate output buffer.
    nk_x = make_nk(x_raw, dtype)
    nk_y = make_nk(x_raw, dtype)
    nk.attention_rope(nk_x, cos, sin, head_count, depth, out=nk_y, capabilities=capabilities)
    y_out = np.asarray(nk_y if dtype == "float32" else nk_y.astype("float32"))
    assert_allclose(y_out, expected, atol=atol, rtol=rtol)

    # In-place: out defaults to x.
    nk_inplace = make_nk(x_raw, dtype)
    nk.attention_rope(nk_inplace, cos, sin, head_count, depth, capabilities=capabilities)
    y_inplace = np.asarray(nk_inplace if dtype == "float32" else nk_inplace.astype("float32"))
    assert_allclose(y_inplace, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
