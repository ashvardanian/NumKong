/**
 *  @brief Ragged attention operations for NumKong Python bindings.
 *  @file python/attention.c
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  This module owns:
 *  - `AttentionPackedMatrix`: opaque pre-packed ragged KV-cache.
 *  - Packing API: `attention_pack()`.
 *  - Compute API: `attention_bidirectional_packed()` and `attention_causal_packed()`.
 */
#include "attention.h"
#include "parallel.h" // `nk_parallel_for_tiles`
#include "tensor.h"

#include <numkong/attention.h>

#include <math.h>

/** @brief One segment-head window of a KV-cache pack, named as in `nk_attention_pack_*`. */
typedef struct attention_pack_task_t {
    nk_attention_pack_punned_t kernel;
    void const *keys;
    void const *values;
    nk_size_t key_value_head_count;
    nk_size_t depth;
    nk_u32_t const *segment_offsets;
    nk_u32_t const *segment_lengths;
    nk_size_t segment_count;
    nk_size_t key_stride_bytes;
    nk_size_t value_stride_bytes;
    void *key_value_packed;
} attention_pack_task_t;

static void attention_pack_tile_(nk_size_t tile_index, void *context) {
    attention_pack_task_t const *task = (attention_pack_task_t const *)context;
    // Window 0 initializes the blob's header and directory before the pool starts, so tile 0 is window 1.
    nk_size_t const window = tile_index + 1;
    task->kernel(task->keys, task->values, task->key_value_head_count, task->depth, task->segment_offsets,
                 task->segment_lengths, task->segment_count, task->key_stride_bytes, task->value_stride_bytes,
                 task->key_value_packed, window, window + 1);
}

/** @brief Arguments both attention kernels take, named as in `nk_attention_*_packed_*`. */
typedef struct attention_arguments_t {
    void const *queries;
    void const *key_value_packed;
    void *output;
    nk_size_t head_count;
    nk_size_t key_value_head_count;
    nk_size_t depth;
    nk_u32_t const *query_offsets;
    nk_size_t query_stride_bytes;
    nk_size_t output_stride_bytes;
    nk_f32_t scale;
} attention_arguments_t;

/** @brief One segment-head task of bidirectional attention. */
typedef struct attention_bidirectional_task_t {
    nk_attention_bidirectional_packed_punned_t kernel;
    attention_arguments_t arguments;
} attention_bidirectional_task_t;

/** @brief One segment-head task of causal attention, which adds the mask to the shared arguments. */
typedef struct attention_causal_task_t {
    nk_attention_causal_packed_punned_t kernel;
    attention_arguments_t arguments;
    nk_i64_t diagonal_offset;
    nk_size_t window;
} attention_causal_task_t;

static void attention_bidirectional_tile_(nk_size_t tile_index, void *context) {
    attention_bidirectional_task_t const *task = (attention_bidirectional_task_t const *)context;
    attention_arguments_t const *arguments = &task->arguments;
    task->kernel(arguments->queries, arguments->key_value_packed, arguments->output, arguments->head_count,
                 arguments->key_value_head_count, arguments->depth, arguments->query_offsets,
                 arguments->query_stride_bytes, arguments->output_stride_bytes, arguments->scale, tile_index, 1);
}

static void attention_causal_tile_(nk_size_t tile_index, void *context) {
    attention_causal_task_t const *task = (attention_causal_task_t const *)context;
    attention_arguments_t const *arguments = &task->arguments;
    task->kernel(arguments->queries, arguments->key_value_packed, arguments->output, arguments->head_count,
                 arguments->key_value_head_count, arguments->depth, arguments->query_offsets,
                 arguments->query_stride_bytes, arguments->output_stride_bytes, arguments->scale, task->diagonal_offset,
                 task->window, tile_index, 1);
}

static void AttentionPackedMatrix_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

static PyObject *AttentionPackedMatrix_repr(PyObject *self) {
    AttentionPackedMatrix *kv = (AttentionPackedMatrix *)self;
    return PyUnicode_FromFormat(
        "<AttentionPackedMatrix segments=%zu heads=%zu depth=%zu tokens=%zu dtype='%s' nbytes=%zu>",
        (size_t)kv->segment_count, (size_t)kv->heads, (size_t)kv->depth, (size_t)kv->total_tokens,
        nk_dtype_python_name(kv->dtype), (size_t)kv->nbytes);
}

static PyObject *AttentionPackedMatrix_get_segments(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->segment_count);
}

static PyObject *AttentionPackedMatrix_get_heads(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->heads);
}

static PyObject *AttentionPackedMatrix_get_depth(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->depth);
}

static PyObject *AttentionPackedMatrix_get_tokens(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->total_tokens);
}

static PyObject *AttentionPackedMatrix_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyUnicode_FromString(nk_dtype_python_name(((AttentionPackedMatrix *)self)->dtype));
}

static PyObject *AttentionPackedMatrix_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->nbytes);
}

static PyObject *AttentionPackedMatrix_get_shape(PyObject *self, void *closure) {
    nk_unused_(closure);
    AttentionPackedMatrix *mm = (AttentionPackedMatrix *)self;
    nk_attention_packed_shape_punned_t shape_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_packed_shape_k, mm->dtype, (nk_kernel_punned_t *)&shape_fn, &cap);
    if (!shape_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No packed_shape kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(mm->dtype));
        return NULL;
    }
    nk_size_t heads = 0, depth = 0, segments = 0;
    shape_fn(mm->start, &heads, &depth, &segments);
    return Py_BuildValue("(nnn)", (Py_ssize_t)heads, (Py_ssize_t)depth, (Py_ssize_t)segments);
}

static PyGetSetDef AttentionPackedMatrix_getset[] = {
    {"segments", AttentionPackedMatrix_get_segments, NULL, "Number of ragged segments", NULL},
    {"heads", AttentionPackedMatrix_get_heads, NULL, "Number of KV heads", NULL},
    {"depth", AttentionPackedMatrix_get_depth, NULL, "Channels per head", NULL},
    {"tokens", AttentionPackedMatrix_get_tokens, NULL, "Total KV tokens across segments", NULL},
    {"dtype", AttentionPackedMatrix_get_dtype, NULL, "Data type of the packed KV-cache", NULL},
    {"nbytes", AttentionPackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {"shape", AttentionPackedMatrix_get_shape, NULL,
     "Dimensions (heads, depth, segments) read from the packed buffer header", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject AttentionPackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.AttentionPackedMatrix",
    .tp_doc = "Opaque pre-packed ragged KV-cache for scaled-dot-product attention",
    .tp_basicsize = sizeof(AttentionPackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = AttentionPackedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = AttentionPackedMatrix_getset,
    .tp_repr = AttentionPackedMatrix_repr,
};

/** @brief Parses a 1-D contiguous `u32` buffer, releasing it on failure. */
static int attention_parse_u32_vector(PyObject *obj, char const *name, Py_buffer *buffer, nk_buffer_backing_t *backing,
                                      nk_u32_t const **data, nk_size_t *count) {
    if (!nk_get_buffer(obj, buffer, PyBUF_STRIDES | PyBUF_FORMAT, backing)) {
        PyErr_Format(PyExc_TypeError, "%s must support buffer protocol", name);
        return 0;
    }
    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(buffer);
    if (buffer->ndim != 1 || dtype != nk_u32_k || buffer->strides[0] != buffer->itemsize) {
        PyBuffer_Release(buffer);
        PyErr_Format(PyExc_ValueError, "%s must be a contiguous 1-D u32 array", name);
        return 0;
    }
    *data = (nk_u32_t const *)buffer->buf;
    *count = (nk_size_t)buffer->shape[0];
    return 1;
}

/** @brief Interprets a K/V/Q buffer as `[tokens, heads * depth]`, inferring the head split. */
static int attention_parse_token_matrix(Py_buffer const *buffer, char const *name, nk_size_t depth, nk_size_t *tokens,
                                        nk_size_t *heads, nk_size_t *row_stride) {
    if (buffer->ndim == 3) {
        if (buffer->strides[2] != buffer->itemsize || buffer->strides[1] != buffer->shape[2] * buffer->itemsize) {
            PyErr_Format(PyExc_ValueError, "%s heads must be contiguous in memory", name);
            return 0;
        }
        if (depth && (nk_size_t)buffer->shape[2] != depth) {
            PyErr_Format(PyExc_ValueError, "%s depth %zd does not match the packed KV-cache (%zu)", name,
                         buffer->shape[2], depth);
            return 0;
        }
        *tokens = (nk_size_t)buffer->shape[0];
        *heads = (nk_size_t)buffer->shape[1];
        *row_stride = (nk_size_t)buffer->strides[0];
        return 1;
    }
    if (buffer->ndim == 2) {
        if (buffer->strides[1] != buffer->itemsize) {
            PyErr_Format(PyExc_ValueError, "%s rows must be contiguous in memory", name);
            return 0;
        }
        if (!depth) {
            PyErr_Format(PyExc_TypeError, "%s is 2-D, so 'depth' must be provided to split the rows", name);
            return 0;
        }
        if ((nk_size_t)buffer->shape[1] % depth) {
            PyErr_Format(PyExc_ValueError, "%s row width %zd is not a multiple of depth %zu", name, buffer->shape[1],
                         depth);
            return 0;
        }
        *tokens = (nk_size_t)buffer->shape[0];
        *heads = (nk_size_t)buffer->shape[1] / depth;
        *row_stride = (nk_size_t)buffer->strides[0];
        return 1;
    }
    PyErr_Format(PyExc_ValueError, "%s must be a 2-D [tokens, heads*depth] or 3-D [tokens, heads, depth]", name);
    return 0;
}

char const doc_attention_pack[] =                                                     //
    "attention_pack(k, v, /, segment_offsets, segment_lengths=None, depth=None, "     //
    "threads=1) -> AttentionPackedMatrix\n\n"                                         //
    "Pack ragged K/V token matrices into a backend-opaque KV-cache blob.\n\n"         //
    "Parameters:\n"                                                                   //
    "    k, v (array_like): Token matrices, 2-D (tokens, heads*depth) or\n"           //
    "        3-D (tokens, heads, depth); bf16 or e4m3, rows may be strided\n"         //
    "        interior views of a fused QKV buffer.\n"                                 //
    "    segment_offsets (u32 array): Cumulative token offsets, length segments+1.\n" //
    "    segment_lengths (u32 array, optional): KV length per segment; defaults to\n" //
    "        adjacent offset differences (self-attention).\n"                         //
    "    depth (int, optional): Required when k/v are 2-D.\n"                         //
    "    threads (int): Threads for packing; 0 = all cores.\n\n"                      //
    "Returns:\n"                                                                      //
    "    AttentionPackedMatrix: Opaque packed KV-cache for attention_*_packed().\n\n" //
    "Signature:\n"                                                                    //
    "    >>> def attention_pack(k, v, /, segment_offsets, segment_lengths=None,\n"    //
    "    ...                    depth=None, threads=1) -> AttentionPackedMatrix: ...";

PyObject *api_attention_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    PyObject *k_obj = NULL, *v_obj = NULL, *offsets_obj = NULL, *lengths_obj = NULL;
    nk_size_t depth = 0, threads = 1;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 2 || nargs > 3 || nargs + nkw > 6) {
        PyErr_SetString(PyExc_TypeError, doc_attention_pack);
        return NULL;
    }
    k_obj = args[0], v_obj = args[1];
    if (nargs >= 3) offsets_obj = args[2];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "segment_offsets") == 0) offsets_obj = value;
        else if (PyUnicode_CompareWithASCIIString(name, "segment_lengths") == 0) lengths_obj = value;
        else if (PyUnicode_CompareWithASCIIString(name, "depth") == 0) {
            depth = (nk_size_t)PyLong_AsSize_t(value);
            if (depth == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "threads") == 0) {
            threads = (nk_size_t)PyLong_AsSize_t(value);
            if (threads == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "attention_pack() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }
    if (!offsets_obj) {
        PyErr_SetString(PyExc_TypeError, "attention_pack() requires 'segment_offsets'");
        return NULL;
    }

    Py_buffer k_buffer, v_buffer, offsets_buffer, lengths_buffer;
    nk_buffer_backing_t k_backing, v_backing, offsets_backing, lengths_backing;
    if (!nk_get_buffer(k_obj, &k_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &k_backing)) {
        PyErr_SetString(PyExc_TypeError, "k must support buffer protocol");
        return NULL;
    }
    if (!nk_get_buffer(v_obj, &v_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &v_backing)) {
        PyBuffer_Release(&k_buffer);
        PyErr_SetString(PyExc_TypeError, "v must support buffer protocol");
        return NULL;
    }

    AttentionPackedMatrix *packed = NULL;
    nk_u32_t *lengths_owned = NULL;
    int offsets_held = 0, lengths_held = 0;

    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&k_buffer);
    if (dtype == nk_dtype_unknown_k || nk_attention_output_dtype(dtype) == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported attention dtype '%s' (expected bf16 or e4m3)",
                     k_buffer.format ? k_buffer.format : "?");
        goto cleanup;
    }
    if (resolve_nk_dtype_in_py_buffer(&v_buffer) != dtype) {
        PyErr_SetString(PyExc_TypeError, "k and v must share the same dtype");
        goto cleanup;
    }

    nk_size_t k_tokens, v_tokens, heads, v_heads, k_stride, v_stride;
    if (!attention_parse_token_matrix(&k_buffer, "k", depth, &k_tokens, &heads, &k_stride)) goto cleanup;
    if (k_buffer.ndim == 3) depth = (nk_size_t)k_buffer.shape[2];
    if (!attention_parse_token_matrix(&v_buffer, "v", depth, &v_tokens, &v_heads, &v_stride)) goto cleanup;
    if (k_tokens != v_tokens || heads != v_heads) {
        PyErr_SetString(PyExc_ValueError, "k and v must have identical shapes");
        goto cleanup;
    }

    nk_u32_t const *segment_offsets = NULL, *segment_lengths = NULL;
    nk_size_t offsets_count = 0, lengths_count = 0;
    if (!attention_parse_u32_vector(offsets_obj, "segment_offsets", &offsets_buffer, &offsets_backing, &segment_offsets,
                                    &offsets_count))
        goto cleanup;
    offsets_held = 1;
    if (offsets_count < 2) {
        PyErr_SetString(PyExc_ValueError, "segment_offsets must have at least 2 entries");
        goto cleanup;
    }
    nk_size_t const segment_count = offsets_count - 1;
    if (lengths_obj) {
        if (!attention_parse_u32_vector(lengths_obj, "segment_lengths", &lengths_buffer, &lengths_backing,
                                        &segment_lengths, &lengths_count))
            goto cleanup;
        lengths_held = 1;
        if (lengths_count != segment_count) {
            PyErr_SetString(PyExc_ValueError, "segment_lengths must have one entry per segment");
            goto cleanup;
        }
    }
    else {
        lengths_owned = (nk_u32_t *)PyMem_Malloc(segment_count * sizeof(nk_u32_t));
        if (!lengths_owned) {
            PyErr_NoMemory();
            goto cleanup;
        }
        for (nk_size_t s = 0; s < segment_count; s++) lengths_owned[s] = segment_offsets[s + 1] - segment_offsets[s];
        segment_lengths = lengths_owned;
    }
    if ((nk_size_t)segment_offsets[segment_count] > k_tokens) {
        PyErr_SetString(PyExc_ValueError, "segment_offsets exceed the number of provided tokens");
        goto cleanup;
    }

    nk_attention_pack_size_punned_t size_fn = NULL;
    nk_attention_pack_punned_t pack_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_pack_size_k, dtype, (nk_kernel_punned_t *)&size_fn, &cap);
    if (size_fn && cap) nk_find_kernel_punned(nk_kernel_attention_pack_k, dtype, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!size_fn || !pack_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No attention pack kernels for dtype '%s'", nk_dtype_python_name(dtype));
        goto cleanup;
    }

    nk_size_t const packed_bytes = size_fn(heads, depth, segment_lengths, segment_count);
    packed = PyObject_NewVar(AttentionPackedMatrix, &AttentionPackedMatrixType, (Py_ssize_t)packed_bytes);
    if (!packed) {
        PyErr_NoMemory();
        goto cleanup;
    }
    packed->dtype = dtype;
    packed->heads = heads;
    packed->depth = depth;
    packed->segment_count = segment_count;
    packed->total_tokens = segment_offsets[segment_count];
    packed->nbytes = packed_bytes;

    {
        attention_pack_task_t task;
        task.kernel = pack_fn;
        task.keys = k_buffer.buf;
        task.values = v_buffer.buf;
        task.key_value_head_count = heads;
        task.depth = depth;
        task.segment_offsets = segment_offsets;
        task.segment_lengths = segment_lengths;
        task.segment_count = segment_count;
        task.key_stride_bytes = k_stride;
        task.value_stride_bytes = v_stride;
        task.key_value_packed = packed->start;
        nk_size_t const task_count = segment_count * heads;
        PyThreadState *save = PyEval_SaveThread();
        // The window covering task 0 initializes the blob's header and directory;
        // running it first keeps the parallel remainder read-only on that region.
        pack_fn(k_buffer.buf, v_buffer.buf, heads, depth, segment_offsets, segment_lengths, segment_count, k_stride,
                v_stride, packed->start, 0, 1);
        if (task_count > 1) nk_parallel_for_tiles(task_count - 1, threads, attention_pack_tile_, &task);
        PyEval_RestoreThread(save);
    }

cleanup:
    if (lengths_owned) PyMem_Free(lengths_owned);
    if (lengths_held) PyBuffer_Release(&lengths_buffer);
    if (offsets_held) PyBuffer_Release(&offsets_buffer);
    PyBuffer_Release(&v_buffer);
    PyBuffer_Release(&k_buffer);
    if (PyErr_Occurred()) {
        Py_XDECREF(packed);
        return NULL;
    }
    return (PyObject *)packed;
}

char const doc_attention_bidirectional_packed[] =                                               //
    "attention_bidirectional_packed(q, kv, /, query_offsets, out=None, scale=None, threads=1) " //
    "-> Tensor\n\n"                                                                             //
    "Ragged bidirectional scaled-dot-product attention against a pre-packed KV-cache.\n\n"      //
    "Parameters:\n"                                                                             //
    "    q (array_like): Query tokens, 2-D (tokens, heads*depth) or 3-D\n"                      //
    "        (tokens, heads, depth), same dtype as the packed KV-cache.\n"                      //
    "    kv (AttentionPackedMatrix): Packed KV-cache from attention_pack().\n"                  //
    "    query_offsets (u32 array): Cumulative query offsets, length segments+1;\n"             //
    "        arange(segments+1) turns the call into a batched single-query pool.\n"             //
    "    out (Tensor, optional): Pre-allocated f32 output of the same shape as q.\n"            //
    "    scale (float, optional): Score scale; default 1/sqrt(depth).\n"                        //
    "    threads (int): Threads over the segment*head task grid; 0 = all.\n\n"                  //
    "Returns:\n"                                                                                //
    "    Tensor: f32 outputs, rows covered by query_offsets are written.\n\n"                   //
    "Signature:\n"                                                                              //
    "    >>> def attention_bidirectional_packed(q, kv, /, query_offsets, out=None,\n"           //
    "    ...                                    scale=None, threads=1) -> Tensor: ...";

char const doc_attention_causal_packed[] =                                                      //
    "attention_causal_packed(q, kv, /, query_offsets, out=None, scale=None, "                   //
    "diagonal_offset=0, window=None, threads=1) -> Tensor\n\n"                                  //
    "Ragged causal, optionally sliding-window, attention against a pre-packed KV-cache.\n\n"    //
    "Query row r of a segment sits at position p = r + diagonal_offset and attends to\n"        //
    "keys max(0, p - window + 1) through min(p, length - 1); rows seeing no key are zeros.\n\n" //
    "Parameters:\n"                                                                             //
    "    q, kv, query_offsets, out, scale, threads: As in attention_bidirectional_packed().\n"  //
    "    diagonal_offset (int): Position of query row 0; length - queries for decode.\n"        //
    "    window (int, optional): Visible keys including the query; None is unbounded.\n\n"      //
    "Returns:\n"                                                                                //
    "    Tensor: f32 outputs, rows covered by query_offsets are written.\n\n"                   //
    "Signature:\n"                                                                              //
    "    >>> def attention_causal_packed(q, kv, /, query_offsets, out=None, scale=None,\n"      //
    "    ...                             diagonal_offset=0, window=None,\n"                     //
    "    ...                             threads=1) -> Tensor: ...";

/** @brief Python buffers and the output tensor behind an `attention_arguments_t`, held until its kernel returns. */
typedef struct attention_buffers_t {
    Py_buffer queries;
    nk_buffer_backing_t queries_backing;
    Py_buffer query_offsets;
    nk_buffer_backing_t query_offsets_backing;
    Tensor *output;
} attention_buffers_t;

/** @brief Validates the operands both attention modes share, releasing everything on failure;
 *  on success the caller passes `buffers` to `attention_buffers_release_`. */
static int attention_arguments_parse_(char const *name, PyObject *queries_object, PyObject *packed_object,
                                      PyObject *query_offsets_object, PyObject *output_object, PyObject *scale_object,
                                      attention_arguments_t *arguments, attention_buffers_t *buffers,
                                      AttentionPackedMatrix **packed_matrix) {
    if (!query_offsets_object) {
        PyErr_Format(PyExc_TypeError, "%s() requires 'query_offsets'", name);
        return 0;
    }
    if (!PyObject_TypeCheck(packed_object, &AttentionPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "kv must be an AttentionPackedMatrix from attention_pack()");
        return 0;
    }
    AttentionPackedMatrix *packed = (AttentionPackedMatrix *)packed_object;

    nk_f32_t scale = (nk_f32_t)(1.0 / sqrt((double)packed->depth));
    if (scale_object) {
        double const scale_f64 = PyFloat_AsDouble(scale_object);
        if (scale_f64 == -1.0 && PyErr_Occurred()) return 0;
        scale = (nk_f32_t)scale_f64;
    }

    if (!nk_get_buffer(queries_object, &buffers->queries, PyBUF_STRIDES | PyBUF_FORMAT, &buffers->queries_backing)) {
        PyErr_SetString(PyExc_TypeError, "q must support buffer protocol");
        return 0;
    }
    if (resolve_nk_dtype_in_py_buffer(&buffers->queries) != packed->dtype) {
        PyErr_Format(PyExc_TypeError, "q dtype must match the packed KV-cache ('%s')",
                     nk_dtype_python_name(packed->dtype));
        goto release_queries;
    }
    nk_size_t query_tokens, head_count, query_stride_bytes;
    if (!attention_parse_token_matrix(&buffers->queries, "q", packed->depth, &query_tokens, &head_count,
                                      &query_stride_bytes))
        goto release_queries;
    if (head_count % packed->heads) {
        PyErr_Format(PyExc_ValueError, "num_heads %zu is not a multiple of the packed heads %zu", (size_t)head_count,
                     (size_t)packed->heads);
        goto release_queries;
    }

    nk_u32_t const *query_offsets = NULL;
    nk_size_t query_offsets_count = 0;
    if (!attention_parse_u32_vector(query_offsets_object, "query_offsets", &buffers->query_offsets,
                                    &buffers->query_offsets_backing, &query_offsets, &query_offsets_count))
        goto release_queries;
    if (query_offsets_count != packed->segment_count + 1) {
        PyErr_Format(PyExc_ValueError, "query_offsets must have %zu entries (segments + 1)",
                     (size_t)packed->segment_count + 1);
        goto release_query_offsets;
    }
    if ((nk_size_t)query_offsets[packed->segment_count] > query_tokens) {
        PyErr_SetString(PyExc_ValueError, "query_offsets exceed the number of provided query tokens");
        goto release_query_offsets;
    }

    nk_size_t const row_values = head_count * packed->depth;
    if (output_object) {
        if (!PyObject_TypeCheck(output_object, &TensorType)) {
            PyErr_SetString(PyExc_TypeError, "out must be a numkong.Tensor");
            goto release_query_offsets;
        }
        Tensor *output = (Tensor *)output_object;
        if (output->dtype != nk_f32_k || output->rank != 2 || (nk_size_t)output->shape[0] < query_tokens ||
            (nk_size_t)output->shape[1] != row_values) {
            PyErr_SetString(PyExc_ValueError, "out must be an f32 Tensor of shape (tokens, heads*depth)");
            goto release_query_offsets;
        }
        Py_INCREF(output);
        buffers->output = output;
    }
    else {
        Py_ssize_t output_shape[2] = {(Py_ssize_t)query_tokens, (Py_ssize_t)row_values};
        buffers->output = Tensor_new(nk_f32_k, 2, output_shape);
        if (!buffers->output) goto release_query_offsets;
    }

    arguments->queries = buffers->queries.buf;
    arguments->key_value_packed = packed->start;
    arguments->output = buffers->output->data;
    arguments->head_count = head_count;
    arguments->key_value_head_count = packed->heads;
    arguments->depth = packed->depth;
    arguments->query_offsets = query_offsets;
    arguments->query_stride_bytes = query_stride_bytes;
    arguments->output_stride_bytes = row_values * sizeof(nk_f32_t);
    arguments->scale = scale;
    *packed_matrix = packed;
    return 1;

release_query_offsets:
    PyBuffer_Release(&buffers->query_offsets);
release_queries:
    PyBuffer_Release(&buffers->queries);
    return 0;
}

/** @brief Releases the input buffers and returns the output tensor, or NULL if a Python error is pending. */
static PyObject *attention_buffers_release_(attention_buffers_t *buffers) {
    PyBuffer_Release(&buffers->query_offsets);
    PyBuffer_Release(&buffers->queries);
    if (PyErr_Occurred()) {
        Py_DECREF(buffers->output);
        return NULL;
    }
    return (PyObject *)buffers->output;
}

PyObject *api_attention_bidirectional_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                             PyObject *kwnames) {
    nk_unused_(self);
    PyObject *query_offsets_object = NULL, *output_object = NULL, *scale_object = NULL;
    nk_size_t threads = 1;

    Py_ssize_t const keyword_count = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 2 || nargs > 3 || nargs + keyword_count > 6) {
        PyErr_SetString(PyExc_TypeError, doc_attention_bidirectional_packed);
        return NULL;
    }
    if (nargs >= 3) query_offsets_object = args[2];
    for (Py_ssize_t keyword_index = 0; keyword_index < keyword_count; keyword_index++) {
        PyObject *keyword = PyTuple_GET_ITEM(kwnames, keyword_index);
        PyObject *value = args[nargs + keyword_index];
        if (PyUnicode_CompareWithASCIIString(keyword, "query_offsets") == 0) query_offsets_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "out") == 0) output_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "scale") == 0) scale_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "threads") == 0) {
            threads = (nk_size_t)PyLong_AsSize_t(value);
            if (threads == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "attention_bidirectional_packed() got unexpected keyword argument '%S'",
                         keyword);
            return NULL;
        }
    }

    attention_bidirectional_task_t task;
    attention_buffers_t buffers;
    AttentionPackedMatrix *packed;
    if (!attention_arguments_parse_("attention_bidirectional_packed", args[0], args[1], query_offsets_object,
                                    output_object, scale_object, &task.arguments, &buffers, &packed))
        return NULL;

    task.kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_bidirectional_packed_k, packed->dtype, (nk_kernel_punned_t *)&task.kernel,
                          &capability);
    if (!task.kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No attention_bidirectional_packed kernel for dtype '%s'",
                     nk_dtype_python_name(packed->dtype));
        return attention_buffers_release_(&buffers);
    }
    PyThreadState *save = PyEval_SaveThread();
    nk_parallel_for_tiles(packed->segment_count * task.arguments.head_count, threads, attention_bidirectional_tile_,
                          &task);
    PyEval_RestoreThread(save);
    return attention_buffers_release_(&buffers);
}

PyObject *api_attention_causal_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *query_offsets_object = NULL, *output_object = NULL, *scale_object = NULL;
    nk_size_t threads = 1;
    attention_causal_task_t task;
    task.diagonal_offset = 0;
    task.window = NK_SIZE_MAX;

    Py_ssize_t const keyword_count = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 2 || nargs > 3 || nargs + keyword_count > 8) {
        PyErr_SetString(PyExc_TypeError, doc_attention_causal_packed);
        return NULL;
    }
    if (nargs >= 3) query_offsets_object = args[2];
    for (Py_ssize_t keyword_index = 0; keyword_index < keyword_count; keyword_index++) {
        PyObject *keyword = PyTuple_GET_ITEM(kwnames, keyword_index);
        PyObject *value = args[nargs + keyword_index];
        if (PyUnicode_CompareWithASCIIString(keyword, "query_offsets") == 0) query_offsets_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "out") == 0) output_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "scale") == 0) scale_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "threads") == 0) {
            threads = (nk_size_t)PyLong_AsSize_t(value);
            if (threads == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "diagonal_offset") == 0) {
            task.diagonal_offset = (nk_i64_t)PyLong_AsLongLong(value);
            if (task.diagonal_offset == -1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "window") == 0) {
            if (value != Py_None) {
                task.window = (nk_size_t)PyLong_AsSize_t(value);
                if (task.window == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
            }
        }
        else {
            PyErr_Format(PyExc_TypeError, "attention_causal_packed() got unexpected keyword argument '%S'", keyword);
            return NULL;
        }
    }

    attention_buffers_t buffers;
    AttentionPackedMatrix *packed;
    if (!attention_arguments_parse_("attention_causal_packed", args[0], args[1], query_offsets_object, output_object,
                                    scale_object, &task.arguments, &buffers, &packed))
        return NULL;

    task.kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_causal_packed_k, packed->dtype, (nk_kernel_punned_t *)&task.kernel,
                          &capability);
    if (!task.kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No attention_causal_packed kernel for dtype '%s'",
                     nk_dtype_python_name(packed->dtype));
        return attention_buffers_release_(&buffers);
    }
    PyThreadState *save = PyEval_SaveThread();
    nk_parallel_for_tiles(packed->segment_count * task.arguments.head_count, threads, attention_causal_tile_, &task);
    PyEval_RestoreThread(save);
    return attention_buffers_release_(&buffers);
}
