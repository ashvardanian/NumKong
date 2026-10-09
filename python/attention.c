/**
 *  @file python/attention.c
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Ragged attention operations for NumKong Python bindings.
 *
 *  This module owns:
 *  - @c AttentionPackedMatrix: the opaque pre-packed ragged KV-cache.
 *  - @c attention_pack: packs keys and values into it.
 *  - @c attention_packed: computes attention over it under a band of visible keys.
 */
#include "attention.h"
#include "parallel.h" // `nk_parallel_for_tiles`
#include "tensor.h"

#include <numkong/attention.h>

#include <math.h>

/** One window of the segments × key-value heads grid of a KV-cache pack, named as in
 *  `nk_attention_pack_*`. */
typedef struct attention_pack_task_t {
    nk_attention_pack_punned_t kernel;
    nk_size_t key_value_head_count;
    nk_size_t depth;
    nk_u32_t const *key_offsets;
    nk_u32_t const *key_lengths;
    nk_size_t segment_count;
    void const *keys;
    nk_size_t key_stride;
    void const *values;
    nk_size_t value_stride;
    void *key_value_packed;
    nk_stream_t stream;
} attention_pack_task_t;

static nk_status_t attention_pack_tile_(nk_size_t tile_index, void *context) {
    attention_pack_task_t const *task = (attention_pack_task_t const *)context;
    return task->kernel(task->key_value_head_count, task->depth, task->key_offsets, task->key_lengths,
                        task->segment_count, task->keys, task->key_stride, task->values, task->value_stride,
                        task->key_value_packed, tile_index, tile_index + 1, task->stream);
}

/** Arguments of the attention kernels, named as in `nk_attention_packed_*`; the gradients write
 *  the query gradient to @c output and read @c log_sum_exp. */
typedef struct attention_arguments_t {
    nk_size_t head_count;
    nk_size_t key_value_head_count;
    nk_size_t depth;
    nk_u32_t const *query_offsets;
    nk_size_t query_token_count;
    nk_f32_t scale;
    nk_size_t keys_before;
    nk_size_t keys_after;
    void const *queries;
    nk_size_t query_stride;
    void const *key_value_packed;
    void *output;
    nk_size_t output_stride;
    nk_f32_t *log_sum_exp;
    nk_stream_t stream;
} attention_arguments_t;

/** Most windows of the query tokens × heads grid one attention call is cut into. */
enum { attention_windows_limit_k = 4 * NUMKONG_PARALLEL_MAX_THREADS };

/** Attention cut into windows of the query tokens × heads grid, window @c w spanning the tasks
 *  from `window_bounds[w]` to `window_bounds[w + 1]`. */
typedef struct attention_task_t {
    nk_attention_packed_punned_t kernel;
    attention_arguments_t arguments;
    nk_size_t window_bounds[attention_windows_limit_k + 1];
} attention_task_t;

static nk_status_t attention_tile_(nk_size_t tile_index, void *context) {
    attention_task_t const *task = (attention_task_t const *)context;
    attention_arguments_t const *arguments = &task->arguments;
    return task->kernel(arguments->head_count, arguments->key_value_head_count, arguments->depth,
                        arguments->query_offsets, arguments->query_token_count, arguments->scale,
                        arguments->keys_before, arguments->keys_after, arguments->queries, arguments->query_stride,
                        arguments->key_value_packed, arguments->output, arguments->output_stride,
                        arguments->log_sum_exp, task->window_bounds[tile_index], task->window_bounds[tile_index + 1],
                        arguments->stream);
}

/** The cost of one head of query @p row of a segment of @p queries queries and @p keys keys: the
 *  keys it sees under @p band, plus one for writing the row. */
static nk_u64_t attention_row_cost_(nk_diagonal_band_t band, nk_size_t row, nk_size_t queries, nk_size_t keys) {
    nk_size_t key_begin, key_end;
    nk_diagonal_band_row_range_(band, (nk_i64_t)keys - (nk_i64_t)queries + (nk_i64_t)row, keys, &key_begin, &key_end);
    return key_end - key_begin + 1;
}

/** Cuts the query tokens × heads grid into @p window_count windows of about equal cost, cutting
 *  inside a token where a share ends, so a single decode token still spreads over its heads. */
static void attention_windows_(nk_u32_t const *query_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,
                               nk_size_t head_count, nk_diagonal_band_t band, nk_size_t window_count,
                               nk_size_t *window_bounds) {
    nk_u64_t total = 0;
    for (nk_size_t segment = 0; segment < segment_count; segment++) {
        nk_size_t const queries = query_offsets[segment + 1] - query_offsets[segment];
        for (nk_size_t row = 0; row < queries; row++)
            total += attention_row_cost_(band, row, queries, key_lengths[segment]) * head_count;
    }
    nk_u64_t done = 0;
    nk_size_t window = 1;
    window_bounds[0] = query_offsets[0] * head_count;
    for (nk_size_t segment = 0; segment < segment_count; segment++) {
        nk_size_t const queries = query_offsets[segment + 1] - query_offsets[segment];
        for (nk_size_t row = 0; row < queries; row++) {
            nk_u64_t const cost = attention_row_cost_(band, row, queries, key_lengths[segment]);
            for (; window < window_count; window++) {
                nk_u64_t const target = total * window / window_count;
                if (target > done + cost * head_count) break;
                nk_u64_t const heads = target > done ? nk_u64_divide_round_up_(target - done, cost) : 0;
                window_bounds[window] = (query_offsets[segment] + row) * head_count + (nk_size_t)heads;
            }
            done += cost * head_count;
        }
    }
    for (; window <= window_count; window++) window_bounds[window] = query_offsets[segment_count] * head_count;
}

/** One segment and key-value head task of the gradients, with the forward's output, the output
 *  gradient, and the key and value gradients it writes. */
typedef struct attention_gradients_task_t {
    nk_attention_packed_gradients_punned_t kernel;
    attention_arguments_t arguments;
    nk_f32_t const *output;
    nk_f32_t const *output_gradient;
    nk_f32_t *key_gradient;
    nk_f32_t *value_gradient;
    nk_size_t query_gradient_stride;
    nk_size_t key_value_gradient_stride;
} attention_gradients_task_t;

static nk_status_t attention_gradients_tile_(nk_size_t tile_index, void *context) {
    attention_gradients_task_t const *task = (attention_gradients_task_t const *)context;
    attention_arguments_t const *arguments = &task->arguments;
    return task->kernel(arguments->head_count, arguments->key_value_head_count, arguments->depth,
                        arguments->query_offsets, arguments->query_token_count, arguments->scale,
                        arguments->keys_before, arguments->keys_after, arguments->queries, arguments->query_stride,
                        arguments->key_value_packed, task->output, task->output_gradient, arguments->output_stride,
                        arguments->log_sum_exp, (nk_f32_t *)arguments->output, task->query_gradient_stride,
                        task->key_gradient, task->value_gradient, task->key_value_gradient_stride, tile_index,
                        tile_index + 1, arguments->stream);
}

static void AttentionPackedMatrix_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

static PyObject *AttentionPackedMatrix_repr(PyObject *self) {
    AttentionPackedMatrix *kv = (AttentionPackedMatrix *)self;
    return PyUnicode_FromFormat(
        "<AttentionPackedMatrix segments=%zu key_value_head_count=%zu depth=%zu tokens=%zu dtype='%s' nbytes=%zu>",
        (size_t)kv->segment_count, (size_t)kv->key_value_head_count, (size_t)kv->depth, (size_t)kv->total_tokens,
        nk_dtype_python_name(kv->dtype), (size_t)kv->nbytes);
}

static PyObject *AttentionPackedMatrix_get_segments(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->segment_count);
}

static PyObject *AttentionPackedMatrix_get_key_value_head_count(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((AttentionPackedMatrix *)self)->key_value_head_count);
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
    nk_find_kernel_punned(nk_kernel_attention_packed_shape_k, mm->dtype, mm->capabilities,
                          (nk_kernel_punned_t *)&shape_fn, &cap);
    if (!shape_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No packed_shape kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(mm->dtype));
        return NULL;
    }
    nk_size_t key_value_head_count = 0, depth = 0, segments = 0;
    if (!check_status(shape_fn(mm->start, &key_value_head_count, &depth, &segments, NULL))) return NULL;
    PyObject *heads_integer = PyLong_FromSsize_t((Py_ssize_t)key_value_head_count);
    PyObject *depth_integer = PyLong_FromSsize_t((Py_ssize_t)depth);
    PyObject *segments_integer = PyLong_FromSsize_t((Py_ssize_t)segments);
    PyObject *shape = heads_integer && depth_integer && segments_integer
                          ? PyTuple_Pack(3, heads_integer, depth_integer, segments_integer)
                          : NULL;
    Py_XDECREF(heads_integer);
    Py_XDECREF(depth_integer);
    Py_XDECREF(segments_integer);
    return shape;
}

static PyGetSetDef AttentionPackedMatrix_getset[] = {
    {"segments", AttentionPackedMatrix_get_segments, NULL, "Number of ragged segments", NULL},
    {"key_value_head_count", AttentionPackedMatrix_get_key_value_head_count, NULL, "Number of KV heads", NULL},
    {"depth", AttentionPackedMatrix_get_depth, NULL, "Channels per head", NULL},
    {"tokens", AttentionPackedMatrix_get_tokens, NULL, "Total KV tokens across segments", NULL},
    {"dtype", AttentionPackedMatrix_get_dtype, NULL, "Data type of the packed KV-cache", NULL},
    {"nbytes", AttentionPackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {"shape", AttentionPackedMatrix_get_shape, NULL,
     "Dimensions (key_value_head_count, depth, segments) read from the packed buffer header", NULL},
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

/** Parses a 1-D contiguous @c u32 buffer, releasing it on failure. */
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

/** Interprets a K/V/Q buffer as @b [tokens,heads×depth], inferring the head split. */
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

char const doc_attention_pack[] =                                                              //
    "attention_pack(k, v, /, key_offsets, key_lengths=None, depth=None, threads=1) ->\n"       //
    "AttentionPackedMatrix\n\n"                                                                //
    "Pack ragged K/V token matrices into a backend-opaque KV-cache blob.\n\n"                  //
    "Args:\n"                                                                                  //
    "    k, v (array_like): Token matrices, 2-D (tokens, heads*depth) or\n"                    //
    "        3-D (tokens, heads, depth); bf16, f16, e4m3 or i8, rows may be strided\n"         //
    "        interior views of a fused QKV buffer, and depth may be strided too, so\n"         //
    "        a depth-major buffer's transposed view packs without a copy.\n"                   //
    "    key_offsets (u32 array): Slot boundaries of the segments, length segments+1, never\n" //
    "        decreasing; segment s keeps its keys from row key_offsets[s].\n"                  //
    "    key_lengths (u32 array, optional): Keys held per segment, at most the slot width;\n"  //
    "        defaults to adjacent offset differences, as in self-attention.\n"                 //
    "    depth (int, optional): Required when k/v are 2-D.\n"                                  //
    "    threads (int): Threads for packing; 0 = all cores.\n\n"                               //
    "Returns:\n"                                                                               //
    "    AttentionPackedMatrix: Opaque packed KV-cache for attention_*_packed().\n\n"          //
    "Signature:\n"                                                                             //
    "    >>> def attention_pack(k, v, /, key_offsets, key_lengths=None,\n"                     //
    "    ...                    depth=None, threads=1) -> AttentionPackedMatrix: ...";

PyObject *api_attention_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    PyObject *k_obj = NULL, *v_obj = NULL, *offsets_obj = NULL, *lengths_obj = NULL;
    nk_size_t depth = 0, threads = 1;
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 2 || nargs > 3 || nargs + nkw > 8) {
        PyErr_SetString(PyExc_TypeError, doc_attention_pack);
        return NULL;
    }
    k_obj = args[0], v_obj = args[1];
    if (nargs >= 3) offsets_obj = args[2];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "key_offsets") == 0) offsets_obj = value;
        else if (PyUnicode_CompareWithASCIIString(name, "key_lengths") == 0) lengths_obj = value;
        else if (PyUnicode_CompareWithASCIIString(name, "depth") == 0) {
            depth = (nk_size_t)PyLong_AsSize_t(value);
            if (depth == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "threads") == 0) {
            threads = (nk_size_t)PyLong_AsSize_t(value);
            if (threads == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else if (!parse_dispatch_keyword(name, value, &capabilities, &stream)) return NULL;
    }
    if (!offsets_obj) {
        PyErr_SetString(PyExc_TypeError, "attention_pack() requires 'key_offsets'");
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
    int offsets_held = 0, lengths_held = 0;

    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&k_buffer);
    if (dtype == nk_dtype_unknown_k || nk_attention_output_dtype(dtype) == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported attention dtype '%s' (expected bf16, f16, e4m3 or i8)",
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

    nk_u32_t const *key_offsets = NULL, *key_lengths = NULL;
    nk_size_t offsets_count = 0, lengths_count = 0;
    if (!attention_parse_u32_vector(offsets_obj, "key_offsets", &offsets_buffer, &offsets_backing, &key_offsets,
                                    &offsets_count))
        goto cleanup;
    offsets_held = 1;
    if (offsets_count < 2) {
        PyErr_SetString(PyExc_ValueError, "key_offsets must have at least 2 entries");
        goto cleanup;
    }
    nk_size_t const segment_count = offsets_count - 1;
    if (lengths_obj) {
        if (!attention_parse_u32_vector(lengths_obj, "key_lengths", &lengths_buffer, &lengths_backing, &key_lengths,
                                        &lengths_count))
            goto cleanup;
        lengths_held = 1;
        if (lengths_count != segment_count) {
            PyErr_SetString(PyExc_ValueError, "key_lengths must have one entry per segment");
            goto cleanup;
        }
    }
    nk_size_t token_count = 0;
    for (nk_size_t segment = 0; segment < segment_count; ++segment) {
        nk_size_t const first = key_offsets[segment], slot_end = key_offsets[segment + 1];
        if (first > slot_end) {
            PyErr_SetString(PyExc_ValueError, "key_offsets must not decrease");
            goto cleanup;
        }
        nk_size_t const count = key_lengths ? key_lengths[segment] : slot_end - first;
        if (count > slot_end - first) {
            PyErr_SetString(PyExc_ValueError, "key_lengths must fit between adjacent key_offsets");
            goto cleanup;
        }
        token_count += count;
    }
    if ((nk_size_t)key_offsets[segment_count] > k_tokens) {
        PyErr_SetString(PyExc_ValueError, "key_offsets exceed the number of provided tokens");
        goto cleanup;
    }

    nk_attention_pack_size_punned_t size_fn = NULL;
    nk_attention_pack_punned_t pack_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_pack_size_k, dtype, capabilities, (nk_kernel_punned_t *)&size_fn, &cap);
    if (size_fn && cap)
        nk_find_kernel_punned(nk_kernel_attention_pack_k, dtype, capabilities, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!size_fn || !pack_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No attention pack kernels for dtype '%s'", nk_dtype_python_name(dtype));
        goto cleanup;
    }

    nk_size_t packed_bytes = 0;
    if (!check_status(size_fn(heads, depth, token_count, segment_count, &packed_bytes))) goto cleanup;
    packed = PyObject_NewVar(AttentionPackedMatrix, &AttentionPackedMatrixType, (Py_ssize_t)packed_bytes);
    if (!packed) {
        PyErr_NoMemory();
        goto cleanup;
    }
    packed->dtype = dtype;
    packed->key_value_head_count = heads;
    packed->depth = depth;
    packed->segment_count = segment_count;
    packed->total_tokens = token_count;
    packed->nbytes = packed_bytes;
    packed->capabilities = capabilities;

    {
        attention_pack_task_t task;
        task.kernel = pack_fn;
        task.keys = k_buffer.buf;
        task.values = v_buffer.buf;
        task.key_value_head_count = heads;
        task.depth = depth;
        task.key_offsets = key_offsets;
        task.key_lengths = key_lengths;
        task.segment_count = segment_count;
        task.key_stride = k_stride;
        task.value_stride = v_stride;
        task.key_value_packed = packed->start;
        task.stream = stream;
        PyThreadState *save = PyEval_SaveThread();
        nk_status_t const status = nk_parallel_for_tiles(segment_count * heads, threads, attention_pack_tile_, &task);
        PyEval_RestoreThread(save);
        check_status(status);
    }

cleanup:
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

char const doc_attention_packed[] =                                                                    //
    "attention_packed(q, kv, /, query_offsets, out=None, scale=None, keys_before=None,\n"              //
    "keys_after=None, log_sum_exp=None, threads=1) -> Tensor\n\n"                                      //
    "Ragged scaled-dot-product attention against a pre-packed KV-cache, under a band of keys.\n\n"     //
    "Each segment's queries align to the end of its keys: query row r of q queries over k keys sits\n" //
    "at position p = r + k - q and sees keys p - keys_before through p + keys_after. Causal\n"         //
    "attention is keys_after=0, a sliding window of w keys adds keys_before=w-1, and rows that see\n"  //
    "no key are zeros.\n\n"                                                                            //
    "Args:\n"                                                                                          //
    "    q (array_like): Query tokens, 2-D (tokens, heads*depth) or 3-D\n"                             //
    "        (tokens, heads, depth), same dtype as the packed KV-cache.\n"                             //
    "    kv (AttentionPackedMatrix): Packed KV-cache from attention_pack().\n"                         //
    "    query_offsets (u32 array): Cumulative query offsets, length segments+1;\n"                    //
    "        arange(segments+1) turns the call into a batched single-query pool.\n"                    //
    "    out (Tensor, optional): Pre-allocated f32 output of the same shape as q.\n"                   //
    "    scale (float, optional): Score scale; default 1/sqrt(depth).\n"                               //
    "    keys_before (int, optional): Keys visible before each query; None is unbounded.\n"            //
    "    keys_after (int, optional): Keys visible after each query; None is unbounded.\n"              //
    "    log_sum_exp (array_like, optional): Writable contiguous f32 (tokens, heads) that receives\n"  //
    "        each row's natural log-sum-exp, -inf for rows without keys, for the gradients.\n"         //
    "    threads (int): Threads sharing equal-cost windows of the token*head grid; 0 = all.\n\n"       //
    "Returns:\n"                                                                                       //
    "    Tensor: f32 outputs, rows covered by query_offsets are written.\n\n"                          //
    "Signature:\n"                                                                                     //
    "    >>> def attention_packed(q, kv, /, query_offsets, out=None, scale=None, keys_before=None,\n"  //
    "    ...                      keys_after=None, log_sum_exp=None, threads=1) -> Tensor: ...";

/** Python buffers and the output tensor behind an @c attention_arguments_t, held until its kernel
 *  returns. */
typedef struct attention_buffers_t {
    Py_buffer queries;
    nk_buffer_backing_t queries_backing;
    Py_buffer query_offsets;
    nk_buffer_backing_t query_offsets_backing;
    Py_buffer log_sum_exp;
    nk_buffer_backing_t log_sum_exp_backing;
    Tensor *output;
} attention_buffers_t;

/** Acquires a C-contiguous f32 buffer of exactly @p count values, releasing it on failure. */
static int attention_parse_f32_values_(PyObject *object, char const *name, nk_size_t count, int flags,
                                       Py_buffer *buffer, nk_buffer_backing_t *backing) {
    if (!nk_get_buffer(object, buffer, flags | PyBUF_STRIDES | PyBUF_FORMAT, backing)) {
        buffer->obj = NULL;
        PyErr_Format(PyExc_TypeError, "%s must support buffer protocol", name);
        return 0;
    }
    nk_size_t values = 1;
    Py_ssize_t stride = buffer->itemsize;
    int contiguous = 1;
    for (int axis = buffer->ndim - 1; axis >= 0; axis--) {
        if (buffer->shape[axis] > 1 && buffer->strides[axis] != stride) contiguous = 0;
        stride *= buffer->shape[axis];
        values *= (nk_size_t)buffer->shape[axis];
    }
    if (resolve_nk_dtype_in_py_buffer(buffer) != nk_f32_k || !contiguous || values != count) {
        PyBuffer_Release(buffer);
        PyErr_Format(PyExc_ValueError, "%s must be a contiguous f32 array of %zu values", name, (size_t)count);
        return 0;
    }
    return 1;
}

/**
 *  @brief Validates the operands both attention modes share, releasing everything on failure.
 *
 *  On success the caller passes @p buffers to @c attention_buffers_release_.
 */
static int attention_arguments_parse_(char const *name, PyObject *queries_object, PyObject *packed_object,
                                      PyObject *query_offsets_object, PyObject *output_object, PyObject *scale_object,
                                      attention_arguments_t *arguments, attention_buffers_t *buffers,
                                      AttentionPackedMatrix **packed_matrix) {
    buffers->log_sum_exp.obj = NULL;
    arguments->log_sum_exp = NULL;
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
    nk_size_t query_tokens, head_count, query_stride;
    if (!attention_parse_token_matrix(&buffers->queries, "q", packed->depth, &query_tokens, &head_count, &query_stride))
        goto release_queries;
    if (head_count % packed->key_value_head_count) {
        PyErr_Format(PyExc_ValueError, "head_count %zu is not a multiple of the packed heads %zu", (size_t)head_count,
                     (size_t)packed->key_value_head_count);
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
        if (!tensor_on_host(output)) goto release_query_offsets;
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
    arguments->key_value_head_count = packed->key_value_head_count;
    arguments->depth = packed->depth;
    arguments->query_offsets = query_offsets;
    arguments->query_token_count = query_offsets[packed->segment_count];
    arguments->query_stride = query_stride;
    arguments->output_stride = row_values * sizeof(nk_f32_t);
    arguments->scale = scale;
    *packed_matrix = packed;
    return 1;

release_query_offsets:
    PyBuffer_Release(&buffers->query_offsets);
release_queries:
    PyBuffer_Release(&buffers->queries);
    return 0;
}

/** Releases input buffers, returns the output tensor, or NULL on a pending Python error. */
static PyObject *attention_buffers_release_(attention_buffers_t *buffers) {
    PyBuffer_Release(&buffers->log_sum_exp);
    PyBuffer_Release(&buffers->query_offsets);
    PyBuffer_Release(&buffers->queries);
    if (PyErr_Occurred()) {
        Py_DECREF(buffers->output);
        return NULL;
    }
    return (PyObject *)buffers->output;
}

/** Reads an optional count of visible keys into @p keys, leaving it unbounded for None. */
static int attention_parse_keys_(PyObject *value, nk_size_t *keys) {
    if (value == Py_None) return 1;
    *keys = (nk_size_t)PyLong_AsSize_t(value);
    return !(*keys == (nk_size_t)-1 && PyErr_Occurred());
}

PyObject *api_attention_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *query_offsets_object = NULL, *output_object = NULL, *scale_object = NULL, *log_sum_exp_object = NULL;
    nk_size_t threads = 1, keys_before = NUMKONG_SIZE_MAX, keys_after = NUMKONG_SIZE_MAX;
    attention_task_t task;

    Py_ssize_t const keyword_count = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 2 || nargs > 3 || nargs + keyword_count > 11) {
        PyErr_SetString(PyExc_TypeError, doc_attention_packed);
        return NULL;
    }
    // A KV-cache is read with the mask that packed it, unless `capabilities=` says otherwise.
    nk_capability_t capabilities = PyObject_TypeCheck(args[1], &AttentionPackedMatrixType)
                                       ? ((AttentionPackedMatrix *)args[1])->capabilities
                                       : nk_cap_cpus_k;
    nk_stream_t stream = NULL;
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
        else if (PyUnicode_CompareWithASCIIString(keyword, "keys_before") == 0) {
            if (!attention_parse_keys_(value, &keys_before)) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "keys_after") == 0) {
            if (!attention_parse_keys_(value, &keys_after)) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "log_sum_exp") == 0) {
            if (value != Py_None) log_sum_exp_object = value;
        }
        else if (!parse_dispatch_keyword(keyword, value, &capabilities, &stream)) return NULL;
    }

    attention_buffers_t buffers;
    AttentionPackedMatrix *packed;
    if (!attention_arguments_parse_("attention_packed", args[0], args[1], query_offsets_object, output_object,
                                    scale_object, &task.arguments, &buffers, &packed))
        return NULL;
    task.arguments.keys_before = keys_before, task.arguments.keys_after = keys_after;
    task.arguments.stream = stream;
    if (log_sum_exp_object) {
        nk_size_t const rows = (nk_size_t)buffers.queries.shape[0] * task.arguments.head_count;
        if (!attention_parse_f32_values_(log_sum_exp_object, "log_sum_exp", rows, PyBUF_WRITABLE, &buffers.log_sum_exp,
                                         &buffers.log_sum_exp_backing))
            return attention_buffers_release_(&buffers);
        task.arguments.log_sum_exp = (nk_f32_t *)buffers.log_sum_exp.buf;
    }

    task.kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_packed_k, packed->dtype, capabilities, (nk_kernel_punned_t *)&task.kernel,
                          &capability);
    if (!task.kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No attention_packed kernel for dtype '%s'",
                     nk_dtype_python_name(packed->dtype));
        return attention_buffers_release_(&buffers);
    }
    if (threads == 0) threads = nk_parallel_concurrency();
    nk_u32_t const *query_offsets = task.arguments.query_offsets;
    nk_size_t const task_count = (query_offsets[packed->segment_count] - query_offsets[0]) * task.arguments.head_count;
    nk_size_t window_count = threads > 1 ? 4 * threads : 1;
    if (window_count > attention_windows_limit_k) window_count = attention_windows_limit_k;
    if (window_count > task_count) window_count = task_count;
    PyThreadState *save = PyEval_SaveThread();
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_u32_t const *pack_key_offsets = NULL, *key_lengths = NULL;
    nk_attention_packed_segments(packed->start, packed->segment_count, &pack_key_offsets, &key_lengths);
    nk_unused_(pack_key_offsets);
    attention_windows_(query_offsets, key_lengths, packed->segment_count, task.arguments.head_count, band, window_count,
                       task.window_bounds);
    nk_status_t const status = nk_parallel_for_tiles(window_count, threads, attention_tile_, &task);
    PyEval_RestoreThread(save);
    check_status(status);
    return attention_buffers_release_(&buffers);
}

char const doc_attention_packed_gradients[] =                                                        //
    "attention_packed_gradients(q, kv, /, query_offsets, output, output_gradient,\n"                 //
    "log_sum_exp, scale=None, keys_before=None, keys_after=None, query_gradient=None,\n"             //
    "threads=1) -> tuple\n\n"                                                                        //
    "Gradients of attention_packed() with respect to its queries, keys and values, bf16 only.\n\n"   //
    "Recomputes every attention weight from the forward's log-sum-exp under the forward's band.\n\n" //
    "Args:\n"                                                                                        //
    "    q, query_offsets, scale, keys_before, keys_after: As in the forward attention_packed().\n"  //
    "    kv (AttentionPackedMatrix): Packed by a gradient-capable capability, SERIAL on CPUs.\n"     //
    "    output (array_like): The forward's contiguous f32 output for these queries.\n"              //
    "    output_gradient (array_like): Contiguous f32 gradient of the loss, shaped like output.\n"   //
    "    log_sum_exp (array_like): The forward's contiguous f32 (tokens, heads) log-sum-exp.\n"      //
    "    query_gradient (Tensor, optional): Pre-allocated f32 (tokens, heads*depth) query\n"         //
    "        gradient, rows any multiple of 4 bytes apart, such as a slice of a wider buffer.\n"     //
    "    threads (int): Threads over the segment*kv_head task grid; 0 = all.\n\n"                    //
    "Returns:\n"                                                                                     //
    "    tuple[Tensor, Tensor, Tensor]: f32 query, key and value gradients, the query one shaped\n"  //
    "        like output and the others (rows across kv's key slots, kv.key_value_head_count*depth).\n\n" "Signature:\n" //
    "    >>> def attention_packed_gradients(q, kv, /, query_offsets, output,\n"          //
    "    ...                                output_gradient, log_sum_exp, scale=None,\n" //
    "    ...                                keys_before=None, keys_after=None,\n"        //
    "    ...                                query_gradient=None, threads=1) -> tuple: ...";

PyObject *api_attention_packed_gradients(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *query_offsets_object = NULL, *output_object = NULL, *output_gradient_object = NULL,
             *log_sum_exp_object = NULL, *scale_object = NULL, *query_gradient_object = NULL;
    nk_size_t threads = 1, keys_before = NUMKONG_SIZE_MAX, keys_after = NUMKONG_SIZE_MAX;
    attention_gradients_task_t task;

    Py_ssize_t const keyword_count = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 2 || keyword_count > 12) {
        PyErr_SetString(PyExc_TypeError, doc_attention_packed_gradients);
        return NULL;
    }
    // A KV-cache is read with the mask that packed it, unless `capabilities=` says otherwise.
    nk_capability_t capabilities = PyObject_TypeCheck(args[1], &AttentionPackedMatrixType)
                                       ? ((AttentionPackedMatrix *)args[1])->capabilities
                                       : nk_cap_cpus_k;
    nk_stream_t stream = NULL;
    for (Py_ssize_t keyword_index = 0; keyword_index < keyword_count; keyword_index++) {
        PyObject *keyword = PyTuple_GET_ITEM(kwnames, keyword_index);
        PyObject *value = args[nargs + keyword_index];
        if (PyUnicode_CompareWithASCIIString(keyword, "query_offsets") == 0) query_offsets_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "output") == 0) output_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "output_gradient") == 0) output_gradient_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "log_sum_exp") == 0) log_sum_exp_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "scale") == 0) scale_object = value;
        else if (PyUnicode_CompareWithASCIIString(keyword, "query_gradient") == 0) {
            if (value != Py_None) query_gradient_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "threads") == 0) {
            threads = (nk_size_t)PyLong_AsSize_t(value);
            if (threads == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "keys_before") == 0) {
            if (!attention_parse_keys_(value, &keys_before)) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(keyword, "keys_after") == 0) {
            if (!attention_parse_keys_(value, &keys_after)) return NULL;
        }
        else if (!parse_dispatch_keyword(keyword, value, &capabilities, &stream)) return NULL;
    }
    if (!output_object || !output_gradient_object || !log_sum_exp_object) {
        PyErr_SetString(PyExc_TypeError,
                        "attention_packed_gradients() requires 'output', 'output_gradient' and 'log_sum_exp'");
        return NULL;
    }

    // The output tensor the forward's parser checks or allocates is the query gradient
    attention_buffers_t buffers;
    AttentionPackedMatrix *packed;
    if (!attention_arguments_parse_("attention_packed_gradients", args[0], args[1], query_offsets_object,
                                    query_gradient_object, scale_object, &task.arguments, &buffers, &packed))
        return NULL;
    task.arguments.keys_before = keys_before, task.arguments.keys_after = keys_after;
    task.arguments.stream = stream;
    task.query_gradient_stride = (nk_size_t)buffers.output->strides[0];
    if (buffers.output->strides[1] != (Py_ssize_t)sizeof(nk_f32_t) || task.query_gradient_stride % sizeof(nk_f32_t)) {
        PyErr_SetString(PyExc_ValueError, "query_gradient rows must be contiguous and 4-byte aligned");
        return attention_buffers_release_(&buffers);
    }

    Py_buffer output_buffer, output_gradient_buffer;
    nk_buffer_backing_t output_backing, output_gradient_backing;
    output_buffer.obj = output_gradient_buffer.obj = NULL;
    Tensor *key_gradient = NULL, *value_gradient = NULL;
    PyObject *gradients = NULL;
    nk_size_t const query_rows = (nk_size_t)buffers.queries.shape[0] * task.arguments.head_count;
    if (!attention_parse_f32_values_(output_object, "output", query_rows * packed->depth, 0, &output_buffer,
                                     &output_backing) ||
        !attention_parse_f32_values_(output_gradient_object, "output_gradient", query_rows * packed->depth, 0,
                                     &output_gradient_buffer, &output_gradient_backing) ||
        !attention_parse_f32_values_(log_sum_exp_object, "log_sum_exp", query_rows, 0, &buffers.log_sum_exp,
                                     &buffers.log_sum_exp_backing))
        goto release;

    task.kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_packed_gradients_k, packed->dtype, capabilities,
                          (nk_kernel_punned_t *)&task.kernel, &capability);
    if (!task.kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No attention_packed_gradients kernel for dtype '%s'",
                     nk_dtype_python_name(packed->dtype));
        goto release;
    }
    // Rows past each segment's keys stay untouched by the kernel, so the gradients start at zero
    nk_u32_t const *pack_key_offsets = NULL, *pack_key_lengths = NULL;
    nk_attention_packed_segments(packed->start, packed->segment_count, &pack_key_offsets, &pack_key_lengths);
    nk_unused_(pack_key_lengths);
    nk_size_t const key_row_values = packed->key_value_head_count * packed->depth;
    Py_ssize_t const key_shape[2] = {(Py_ssize_t)pack_key_offsets[packed->segment_count], (Py_ssize_t)key_row_values};
    key_gradient = Tensor_new(nk_f32_k, 2, key_shape);
    value_gradient = key_gradient ? Tensor_new(nk_f32_k, 2, key_shape) : NULL;
    if (!value_gradient) goto release;
    memset(key_gradient->data, 0, (size_t)key_shape[0] * key_row_values * sizeof(nk_f32_t));
    memset(value_gradient->data, 0, (size_t)key_shape[0] * key_row_values * sizeof(nk_f32_t));
    task.output = (nk_f32_t const *)output_buffer.buf;
    task.output_gradient = (nk_f32_t const *)output_gradient_buffer.buf;
    task.arguments.log_sum_exp = (nk_f32_t *)buffers.log_sum_exp.buf;
    task.key_gradient = (nk_f32_t *)key_gradient->data;
    task.value_gradient = (nk_f32_t *)value_gradient->data;
    task.key_value_gradient_stride = key_row_values * sizeof(nk_f32_t);

    PyThreadState *save = PyEval_SaveThread();
    nk_status_t const status = nk_parallel_for_tiles(packed->segment_count * packed->key_value_head_count, threads,
                                                     attention_gradients_tile_, &task);
    PyEval_RestoreThread(save);
    check_status(status);
    if (!PyErr_Occurred())
        gradients = PyTuple_Pack(3, (PyObject *)buffers.output, (PyObject *)key_gradient, (PyObject *)value_gradient);

release:
    PyBuffer_Release(&output_gradient_buffer);
    PyBuffer_Release(&output_buffer);
    Py_XDECREF(key_gradient);
    Py_XDECREF(value_gradient);
    PyObject *query_gradient = attention_buffers_release_(&buffers);
    if (!query_gradient) return NULL;
    Py_DECREF(query_gradient);
    return gradients;
}

char const doc_attention_rope[] =                                                                        //
    "NeoX split-half rotary position embedding, RoPE.\n\n"                                               //
    "Rotates every channel pair (channel i against i+depth/2) of each head by the per-token angle\n"     //
    "grids. Bake position lookup and multi-axis, M-RoPE, assignment into the `[rows,depth/2]` cos/sin\n" //
    "grids so a single call rotates the whole head.\n\n"                                                 //
    "Args:\n"                                                                                            //
    "    x (Tensor): [rows,head_count×depth], float32/bfloat16/e4m3.\n"                                  //
    "    cos, sin (Tensor): `[rows, depth/2]` float32 angle grids, shared across heads.\n"               //
    "    head_count (int): Number of heads per token.\n"                                                 //
    "    depth (int): Even number of channels per head.\n"                                               //
    "    out (Tensor, optional): Output, same shape/dtype as x; may alias x. Defaults to x.\n\n"         //
    "Returns:\n"                                                                                         //
    "    None: The result is written into `out`, or into `x` in place.\n\n"                              //
    "Signature:\n"                                                                                       //
    "    >>> def attention_rope(x, cos, sin, head_count, depth, /, *, out) -> None: ...";

PyObject *api_attention_rope(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    nk_unused_(self);
    PyObject *x_object = NULL, *cos_object = NULL, *sin_object = NULL, *head_count_object = NULL, *depth_object = NULL;
    PyObject *out_object = NULL;
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;

    Py_buffer x_buffer, y_buffer, cos_buffer, sin_buffer;
    nk_buffer_backing_t x_backing, y_backing, cos_backing, sin_backing;
    memset(&x_buffer, 0, sizeof(Py_buffer));
    memset(&y_buffer, 0, sizeof(Py_buffer));
    memset(&cos_buffer, 0, sizeof(Py_buffer));
    memset(&sin_buffer, 0, sizeof(Py_buffer));
    int got_x = 0, got_y = 0, got_cos = 0, got_sin = 0;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 5 || args_count > 8) {
        PyErr_Format(PyExc_TypeError, "Function expects 5-8 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Only first 5 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }
    PyObject *positional[5] = {NULL, NULL, NULL, NULL, NULL};
    for (Py_ssize_t i = 0; i < positional_args_count; ++i) positional[i] = args[i];
    x_object = positional[0], cos_object = positional[1], sin_object = positional[2];
    head_count_object = positional[3], depth_object = positional[4];
    for (Py_ssize_t k = 0, p = positional_args_count; k < args_names_count; ++p, ++k) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, k);
        PyObject *const value = args[p];
        if (PyUnicode_CompareWithASCIIString(key, "head_count") == 0 && !head_count_object) head_count_object = value;
        else if (PyUnicode_CompareWithASCIIString(key, "depth") == 0 && !depth_object) depth_object = value;
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_object) out_object = value;
        else if (!parse_dispatch_keyword(key, value, &capabilities, &stream)) return NULL;
    }
    if (!x_object || !cos_object || !sin_object || !head_count_object || !depth_object) {
        PyErr_SetString(PyExc_TypeError, "attention_rope requires x, cos, sin, head_count, depth");
        return NULL;
    }

    long head_count = PyLong_AsLong(head_count_object), depth = PyLong_AsLong(depth_object);
    if (PyErr_Occurred()) return NULL;
    if (head_count <= 0 || depth <= 0 || depth % 2) {
        PyErr_SetString(PyExc_ValueError, "head_count and depth must be positive, and depth even");
        return NULL;
    }

    int const x_flags = out_object ? (PyBUF_STRIDES | PyBUF_FORMAT) : (PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT);
    if (!nk_get_buffer(x_object, &x_buffer, x_flags, &x_backing)) return NULL;
    got_x = 1;
    if (x_buffer.ndim < 1 || x_buffer.ndim > NUMKONG_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d unsupported", x_buffer.ndim);
        goto cleanup;
    }
    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&x_buffer);
    if (dtype != nk_f32_k && dtype != nk_bf16_k && dtype != nk_e4m3_k) {
        PyErr_Format(PyExc_TypeError, "attention_rope supports f32, bf16, e4m3; got '%s'", nk_dtype_python_name(dtype));
        goto cleanup;
    }
    int const rank = x_buffer.ndim;
    size_t const value_bytes = nk_dtype_bytes_per_value(dtype);
    if ((size_t)x_buffer.strides[rank - 1] != value_bytes) {
        PyErr_SetString(PyExc_ValueError, "attention_rope requires the last axis to be contiguous");
        goto cleanup;
    }
    if ((nk_size_t)x_buffer.shape[rank - 1] < (nk_size_t)(head_count * depth)) {
        PyErr_SetString(PyExc_ValueError, "last axis too small for head_count * depth");
        goto cleanup;
    }
    nk_size_t rows = 1;
    for (int d = 0; d < rank - 1; ++d) rows *= (nk_size_t)x_buffer.shape[d];
    for (int d = 0; d + 2 < rank; ++d) {
        if (x_buffer.strides[d] != x_buffer.shape[d + 1] * x_buffer.strides[d + 1]) {
            PyErr_SetString(PyExc_ValueError, "attention_rope requires C-contiguous leading axes for rank > 2");
            goto cleanup;
        }
    }
    nk_size_t const x_stride = rank >= 2 ? (nk_size_t)x_buffer.strides[rank - 2] : 0;

    void *y_data = x_buffer.buf;
    nk_size_t y_stride = x_stride;
    if (out_object) {
        if (!nk_get_buffer(out_object, &y_buffer, PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT, &y_backing))
            goto cleanup;
        got_y = 1;
        if (y_buffer.ndim != rank || resolve_nk_dtype_in_py_buffer(&y_buffer) != dtype) {
            PyErr_SetString(PyExc_ValueError, "out must have the same shape and dtype as x");
            goto cleanup;
        }
        for (int d = 0; d < rank; ++d)
            if (y_buffer.shape[d] != x_buffer.shape[d]) {
                PyErr_SetString(PyExc_ValueError, "out must have the same shape as x");
                goto cleanup;
            }
        if ((size_t)y_buffer.strides[rank - 1] != value_bytes) {
            PyErr_SetString(PyExc_ValueError, "out requires the last axis to be contiguous");
            goto cleanup;
        }
        y_data = y_buffer.buf;
        y_stride = rank >= 2 ? (nk_size_t)y_buffer.strides[rank - 2] : 0;
    }

    if (!nk_get_buffer(cos_object, &cos_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &cos_backing)) goto cleanup;
    got_cos = 1;
    if (!nk_get_buffer(sin_object, &sin_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &sin_backing)) goto cleanup;
    got_sin = 1;
    if (resolve_nk_dtype_in_py_buffer(&cos_buffer) != nk_f32_k ||
        resolve_nk_dtype_in_py_buffer(&sin_buffer) != nk_f32_k) {
        PyErr_SetString(PyExc_TypeError, "cos and sin must be float32");
        goto cleanup;
    }
    if ((nk_size_t)(cos_buffer.len / (Py_ssize_t)sizeof(nk_f32_t)) < rows * (nk_size_t)depth / 2 ||
        (nk_size_t)(sin_buffer.len / (Py_ssize_t)sizeof(nk_f32_t)) < rows * (nk_size_t)depth / 2) {
        PyErr_SetString(PyExc_ValueError, "cos and sin must each have at least rows * depth / 2 elements");
        goto cleanup;
    }

    nk_attention_rope_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_attention_rope_k, dtype, capabilities, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No attention_rope kernel for dtype '%s'", nk_dtype_python_name(dtype));
        goto cleanup;
    }

    {
        PyThreadState *gil = PyEval_SaveThread();
        nk_status_t const status = kernel(x_buffer.buf, (nk_f32_t const *)cos_buffer.buf,
                                          (nk_f32_t const *)sin_buffer.buf, y_data, rows, (nk_size_t)head_count,
                                          (nk_size_t)depth, x_stride, y_stride, stream);
        PyEval_RestoreThread(gil);
        if (!check_status(status)) goto cleanup;
    }
    if (got_x) PyBuffer_Release(&x_buffer);
    if (got_y) PyBuffer_Release(&y_buffer);
    if (got_cos) PyBuffer_Release(&cos_buffer);
    if (got_sin) PyBuffer_Release(&sin_buffer);
    Py_RETURN_NONE;
cleanup:
    if (got_x) PyBuffer_Release(&x_buffer);
    if (got_y) PyBuffer_Release(&y_buffer);
    if (got_cos) PyBuffer_Release(&cos_buffer);
    if (got_sin) PyBuffer_Release(&sin_buffer);
    return NULL;
}
