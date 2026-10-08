/**
 *  @file python/attention.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Ragged attention API declarations for NumKong Python bindings.
 */
#ifndef NUMKONG_PYTHON_ATTENTION_H
#define NUMKONG_PYTHON_ATTENTION_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numkong/numkong.h>

/**
 *  @brief Pre-packed KV-cache for ragged scaled-dot-product attention.
 *
 *  Owns the backend-opaque packed blob inline as a flex-array, together with the geometry needed to
 *  validate query batches against it. Created via `nk.attention_pack()` and consumed by
 *  `nk.attention_packed()`.
 */
typedef struct AttentionPackedMatrix {
    PyObject_HEAD

    /** Input dtype (bf16 or e4m3). */
    nk_dtype_t dtype;

    /** Number of KV heads packed per token. */
    nk_size_t key_value_head_count;

    /** Channels per head. */
    nk_size_t depth;

    /** Number of ragged segments. */
    nk_size_t segment_count;

    /** Total KV tokens across all segments. */
    nk_size_t total_tokens;

    /** Size of the packed blob in bytes. */
    nk_size_t nbytes;

    /** The mask that packed it; later calls default to it, as only that mask reads the layout. */
    nk_capability_t capabilities;

    /** Variable-length packed data. */
    char start[];
} AttentionPackedMatrix;

/** AttentionPackedMatrix Python type object. */
extern PyTypeObject AttentionPackedMatrixType;

/** Pack ragged K/V token matrices into a backend-opaque KV-cache blob. */
PyObject *api_attention_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Ragged scaled-dot-product attention against a pre-packed KV-cache, under a band of keys. */
PyObject *api_attention_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Gradients of ragged attention with respect to its queries, keys and values. */
PyObject *api_attention_packed_gradients(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** NeoX split-half rotary position embedding, RoPE, separate aliasable output. */
PyObject *api_attention_rope(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_attention_pack[];
extern char const doc_attention_packed[];
extern char const doc_attention_packed_gradients[];
extern char const doc_attention_rope[];

#endif // NUMKONG_PYTHON_ATTENTION_H
