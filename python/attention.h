/**
 *  @brief Ragged attention API declarations for NumKong Python bindings.
 *  @file python/attention.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 */
#ifndef NK_PYTHON_ATTENTION_H
#define NK_PYTHON_ATTENTION_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numkong/numkong.h>

/**
 *  @brief Pre-packed KV-cache for ragged scaled-dot-product attention.
 *
 *  Owns the backend-opaque packed blob inline (flex-array), together with the
 *  geometry needed to validate query batches against it.  Created via
 *  `nk.attention_pack()` and consumed by `nk.attention_packed()`.
 */
typedef struct AttentionPackedKV {
    PyObject_HEAD
    /** Input dtype (bf16 or e4m3). */
    nk_dtype_t dtype;
    /** Number of KV heads packed per token. */
    nk_size_t num_kv_heads;
    /** Channels per head. */
    nk_size_t head_dim;
    /** Number of ragged segments. */
    nk_size_t segment_count;
    /** Total KV tokens across all segments. */
    nk_size_t total_tokens;
    /** Size of the packed blob in bytes. */
    nk_size_t nbytes;
    /** Variable-length packed data. */
    char start[];
} AttentionPackedKV;

/** @brief AttentionPackedKV Python type object. */
extern PyTypeObject AttentionPackedKVType;

/** @brief Pack ragged K/V token matrices into a backend-opaque KV-cache blob. */
PyObject *api_attention_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Ragged scaled-dot-product attention against a pre-packed KV-cache. */
PyObject *api_attention_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_attention_pack[];
extern char const doc_attention_packed[];

#endif // NK_PYTHON_ATTENTION_H
