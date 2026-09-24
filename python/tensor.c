/**
 *  @file python/tensor.c
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *  @brief Tensor implementation for NumKong Python bindings.
 *
 *  This file implements the Tensor N-dimensional array type with NumPy-like interface and the
 *  TensorIter iterator.
 *
 *  Features:
 *  - Support for all NumKong dtypes: f32, f64, f16, bf16, i8, complex, etc.
 *  - Arbitrary strides for views and slices
 *  - Zero-copy views with reference counting
 *  - Python buffer protocol for interoperability
 *  - Arithmetic operators: +, -, *, @.
 *  - Reduction operations: sum, min, max, argmin, argmax.
 */
#include "tensor.h"
#include "matrix.h"
#include "dlpack_interop.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

int buffers_shapes_match(Py_buffer const *first, Py_buffer const *second) {
    if (first->ndim != second->ndim) {
        PyErr_SetString(PyExc_ValueError, "Input tensor ranks don't match");
        return 0;
    }
    for (int dimension = 0; dimension < first->ndim; ++dimension) {
        if (first->shape[dimension] != second->shape[dimension]) {
            PyErr_Format( //
                PyExc_ValueError,
                "Input tensor shapes don't match at dimension %d (%zd vs %zd). " //
                "NumKong does not support implicit shape broadcasting.",
                dimension, first->shape[dimension], second->shape[dimension]);
            return 0;
        }
    }
    return 1;
}

/**
 *  @brief Conservatively report whether two strided buffers share any byte of address range.
 *
 *  Accumulates the per-dimension extents into signed minimum and maximum byte offsets, so a
 *  reversed view with negative strides widens the range below its base pointer rather than above.
 */
static int py_buffers_may_overlap(Py_buffer const *first, Py_buffer const *second) {
    if (!first->len || !second->len) return 0;

    Py_ssize_t first_min_offset = 0, first_max_offset = 0;
    for (int dimension = 0; dimension < first->ndim; ++dimension) {
        Py_ssize_t const extent = (first->shape[dimension] - 1) * first->strides[dimension];
        if (extent < 0) first_min_offset += extent;
        else first_max_offset += extent;
    }
    Py_ssize_t second_min_offset = 0, second_max_offset = 0;
    for (int dimension = 0; dimension < second->ndim; ++dimension) {
        Py_ssize_t const extent = (second->shape[dimension] - 1) * second->strides[dimension];
        if (extent < 0) second_min_offset += extent;
        else second_max_offset += extent;
    }

    char const *first_begin = (char const *)first->buf + first_min_offset;
    char const *first_end = (char const *)first->buf + first_max_offset + first->itemsize;
    char const *second_begin = (char const *)second->buf + second_min_offset;
    char const *second_end = (char const *)second->buf + second_max_offset + second->itemsize;
    return first_begin < second_end && second_begin < first_end;
}

char *validate_out_py_buffer(Py_buffer const *out_buffer, Py_buffer const *input_buffer, nk_dtype_t expected_dtype) {
    nk_dtype_t const out_dtype = resolve_nk_dtype_in_py_buffer(out_buffer);
    if (out_dtype != expected_dtype) {
        PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected '%s'", nk_dtype_python_name(out_dtype),
                     nk_dtype_python_name(expected_dtype));
        return NULL;
    }
    if (!PyBuffer_IsContiguous(out_buffer, 'C')) {
        PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
        return NULL;
    }
    if (out_buffer->readonly) {
        PyErr_SetString(PyExc_ValueError, "out must be writable");
        return NULL;
    }
    if (py_buffers_may_overlap(input_buffer, out_buffer)) {
        PyErr_SetString(PyExc_ValueError, "out must not overlap the input");
        return NULL;
    }
    return (char *)out_buffer->buf;
}

size_t shared_contiguous_tail_dimensions(Py_buffer const *buffers[], size_t num_buffers, size_t num_dims) {
    size_t num_contiguous_dims = 0;
    for (size_t dimension = num_dims; dimension-- > 0;) {
        // Compute the expected stride for this dimension if it were packed
        int all_packed = 1;
        for (size_t buffer_idx = 0; buffer_idx < num_buffers; ++buffer_idx) {
            Py_ssize_t expected_stride = buffers[buffer_idx]->itemsize;
            for (size_t inner_dim = num_dims - 1; inner_dim > dimension; --inner_dim)
                expected_stride *= buffers[buffer_idx]->shape[inner_dim];
            if (buffers[buffer_idx]->strides[dimension] != expected_stride) {
                all_packed = 0;
                break;
            }
        }
        if (!all_packed) break;
        ++num_contiguous_dims;
    }
    return num_contiguous_dims;
}

void each_sum_recursive(                                           //
    nk_each_sum_punned_t kernel,                                   //
    char const *a_data, char const *b_data, char *result_data,     //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,          //
    Py_ssize_t const *b_strides, Py_ssize_t const *result_strides, //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    // Base case: all remaining dimensions are contiguous — one kernel call
    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, b_data, contiguous_elements, result_data);
        return;
    }

    // Iterate over the outermost non-contiguous dimension, then recurse
    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        Py_ssize_t const signed_position = (Py_ssize_t)position;
        each_sum_recursive(kernel,                                            //
                           a_data + signed_position * a_strides[0],           //
                           b_data + signed_position * b_strides[0],           //
                           result_data + signed_position * result_strides[0], //
                           shape + 1, a_strides + 1,                          //
                           b_strides + 1, result_strides + 1,                 //
                           remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_scale_recursive(                                           //
    nk_each_scale_punned_t kernel,                                   //
    char const *a_data, char *result_data,                           //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta, //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,            //
    Py_ssize_t const *result_strides,                                //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, contiguous_elements, alpha, beta, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        Py_ssize_t const signed_position = (Py_ssize_t)position;
        each_scale_recursive(kernel,                                            //
                             a_data + signed_position * a_strides[0],           //
                             result_data + signed_position * result_strides[0], //
                             alpha, beta,                                       //
                             shape + 1, a_strides + 1,                          //
                             result_strides + 1,                                //
                             remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_fma_recursive(                                                           //
    nk_each_fma_punned_t kernel,                                                   //
    char const *a_data, char const *b_data, char const *c_data, char *result_data, //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta,               //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,                          //
    Py_ssize_t const *b_strides, Py_ssize_t const *c_strides,                      //
    Py_ssize_t const *result_strides,                                              //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, b_data, c_data, contiguous_elements, alpha, beta, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        Py_ssize_t const signed_position = (Py_ssize_t)position;
        each_fma_recursive(kernel,                                            //
                           a_data + signed_position * a_strides[0],           //
                           b_data + signed_position * b_strides[0],           //
                           c_data + signed_position * c_strides[0],           //
                           result_data + signed_position * result_strides[0], //
                           alpha, beta,                                       //
                           shape + 1, a_strides + 1,                          //
                           b_strides + 1, c_strides + 1,                      //
                           result_strides + 1,                                //
                           remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_blend_recursive(                                           //
    nk_each_blend_punned_t kernel,                                   //
    char const *a_data, char const *b_data, char *result_data,       //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta, //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,            //
    Py_ssize_t const *b_strides, Py_ssize_t const *result_strides,   //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, b_data, contiguous_elements, alpha, beta, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        Py_ssize_t const signed_position = (Py_ssize_t)position;
        each_blend_recursive(kernel,                                            //
                             a_data + signed_position * a_strides[0],           //
                             b_data + signed_position * b_strides[0],           //
                             result_data + signed_position * result_strides[0], //
                             alpha, beta,                                       //
                             shape + 1, a_strides + 1,                          //
                             b_strides + 1, result_strides + 1,                 //
                             remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_unary_recursive(                                //
    nk_kernel_trig_punned_t kernel,                       //
    char const *a_data, char *result_data,                //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides, //
    Py_ssize_t const *result_strides,                     //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, contiguous_elements, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        Py_ssize_t const signed_position = (Py_ssize_t)position;
        each_unary_recursive(kernel,                                            //
                             a_data + signed_position * a_strides[0],           //
                             result_data + signed_position * result_strides[0], //
                             shape + 1, a_strides + 1,                          //
                             result_strides + 1,                                //
                             remaining_dims - 1, contiguous_tail_dims);
    }
}

/** Return a Python scalar from a tensor byte offset using nk_scalar_buffer_to_py_number. */
static PyObject *tensor_read_scalar(Tensor *tensor, size_t byte_offset) {
    size_t elem_size = nk_dtype_bytes_per_value(tensor->dtype);
    if (!elem_size) {
        PyErr_SetString(PyExc_TypeError, "unsupported dtype for indexing");
        return NULL;
    }
    nk_scalar_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    memcpy(&buf, tensor->data + byte_offset, elem_size);
    return nk_scalar_buffer_to_py_number(&buf, tensor->dtype);
}

size_t dimensions_to_values(nk_dtype_t dtype, size_t dimensions) {
    return dimensions / (size_t)nk_dimensions_per_value(dtype);
}

Py_ssize_t storage_extent(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape, size_t dimension) {
    if (dimension + 1 != rank) return shape[dimension];
    return (Py_ssize_t)dimensions_to_values(dtype, (size_t)shape[dimension]);
}

int validate_packed_dimensions(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape) {
    nk_size_t const dimensions_per_value = nk_dimensions_per_value(dtype);
    if (dimensions_per_value <= 1) return 1;
    if (rank == 0) {
        PyErr_Format(PyExc_ValueError, "packed dtype '%s' needs at least one dimension", nk_dtype_python_name(dtype));
        return 0;
    }
    if ((nk_size_t)shape[rank - 1] % dimensions_per_value == 0) return 1;
    PyErr_Format(PyExc_ValueError, "last dimension %zd must be a multiple of %zu for dtype '%s'", shape[rank - 1],
                 (size_t)dimensions_per_value, nk_dtype_python_name(dtype));
    return 0;
}

int strides_are_c_contiguous(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape, Py_ssize_t const *strides) {
    Py_ssize_t expected = (Py_ssize_t)nk_dtype_bytes_per_value(dtype);
    for (size_t dimension = rank; dimension-- > 0;) {
        if (strides[dimension] != expected) return 0;
        expected *= storage_extent(dtype, rank, shape, dimension);
    }
    return 1;
}

/** Check if a tensor is C-contiguous, row-major. */
static int tensor_is_c_contig(Tensor *tensor) {
    return strides_are_c_contiguous(tensor->dtype, tensor->rank, tensor->shape, tensor->strides);
}

/** Check if a tensor is Fortran-contiguous, column-major. */
static int tensor_is_f_contig(Tensor *tensor) {
    Py_ssize_t expected = (Py_ssize_t)nk_dtype_bytes_per_value(tensor->dtype);
    for (size_t dimension = 0; dimension < tensor->rank; dimension++) {
        if (tensor->strides[dimension] != expected) return 0;
        expected *= storage_extent(tensor->dtype, tensor->rank, tensor->shape, dimension);
    }
    return 1;
}

/** Read dimension @p lane of the packed value at @p byte_offset as a Python number. */
static PyObject *tensor_read_packed_scalar(Tensor *tensor, size_t byte_offset, size_t lane) {
    nk_f64_t lanes[NK_BITS_PER_BYTE];
    nk_cast(tensor->data + byte_offset, tensor->dtype, nk_dimensions_per_value(tensor->dtype), lanes, nk_f64_k);
    if (nk_dtype_family(tensor->dtype) == nk_dtype_family_float_k) return PyFloat_FromDouble(lanes[lane]);
    return PyLong_FromLongLong((long long)lanes[lane]);
}

/**
 *  @brief Bounded free-list of dead view headers, recycled to skip PyObject_New / tp_free churn.
 *
 *  Views, `parent != NULL`, borrowing the parent's storage, churn heavily: every slice, integer
 *  index, `.T`, @c flatten, and iteration step mints a fixed-size header torn down moments later.
 *  Rather than round-trip each header through the allocator, @c Tensor_dealloc parks a dead view on
 *  this intrusive list — the @c parent field doubles as the next-link — and the view constructors
 *  pop one back, reviving it with @c _Py_NewReference. Legal because Tensor is a non-GC,
 *  fixed-basicsize type. Owning tensors are never pooled: their `parent == NULL` sentinel would
 *  collide with the link, and their @c data buffer must be freed.
 */
#define NK_VIEW_FREELIST_CAP 64

static Tensor *g_view_freelist_head = NULL;
static size_t g_view_freelist_count = 0;

#ifdef Py_GIL_DISABLED
static PyMutex g_view_freelist_mutex = {0};
#define NK_VIEW_FREELIST_LOCK()   PyMutex_Lock(&g_view_freelist_mutex)
#define NK_VIEW_FREELIST_UNLOCK() PyMutex_Unlock(&g_view_freelist_mutex)
#else
#define NK_VIEW_FREELIST_LOCK()   ((void)0)
#define NK_VIEW_FREELIST_UNLOCK() ((void)0)
#endif

/** Park a dead view header for reuse. Returns 1 if pooled, 0 if the caller must free it. */
static int tensor_view_freelist_push(Tensor *view) {
    int pooled = 0;
    NK_VIEW_FREELIST_LOCK();
    if (g_view_freelist_count < NK_VIEW_FREELIST_CAP) {
        view->parent = (PyObject *)g_view_freelist_head; // reuse `parent` as the intrusive next-link
        g_view_freelist_head = view;
        g_view_freelist_count++;
        pooled = 1;
    }
    NK_VIEW_FREELIST_UNLOCK();
    return pooled;
}

/** Pop and revive a pooled view header, or NULL when the pool is empty. */
static Tensor *tensor_view_freelist_pop(void) {
    Tensor *view = NULL;
    NK_VIEW_FREELIST_LOCK();
    if (g_view_freelist_head) {
        view = g_view_freelist_head;
        g_view_freelist_head = (Tensor *)view->parent; // unlink
        g_view_freelist_count--;
    }
    NK_VIEW_FREELIST_UNLOCK();
    if (view) _Py_NewReference((PyObject *)view); // dead header → live, refcount 1
    return view;
}

/** Allocate a view header, preferring a recycled one over a fresh PyObject_New. */
static Tensor *tensor_view_header_new(void) {
    Tensor *view = tensor_view_freelist_pop();
    if (view) return view;
    return PyObject_New(Tensor, &TensorType);
}

/** Drain the view free-list at interpreter teardown, wired as the module's m_free. */
void nk_tensor_view_freelist_clear(void) {
    NK_VIEW_FREELIST_LOCK();
    Tensor *node = g_view_freelist_head;
    while (node) {
        Tensor *next = (Tensor *)node->parent;
        PyObject_Free(node);
        node = next;
    }
    g_view_freelist_head = NULL;
    g_view_freelist_count = 0;
    NK_VIEW_FREELIST_UNLOCK();
}

static void Tensor_dealloc(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->parent == NULL) {
        PyMem_Free(tensor->data); // owns the buffer; PyMem_Free(NULL) is a no-op
        Py_TYPE(self)->tp_free(self);
        return;
    }
    // View: drop the borrowed parent reference, then park the header for reuse.
    Py_DECREF(tensor->parent);
    if (tensor_view_freelist_push(tensor)) return;
    Py_TYPE(self)->tp_free(self);
}

Tensor *Tensor_new(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    if (!validate_packed_dimensions(dtype, rank, shape)) return NULL;

    size_t const item_size = nk_dtype_bytes_per_value(dtype);
    size_t total_items = 1;
    for (size_t i = 0; i < rank; i++) {
        if (shape[i] > 0 && total_items > SIZE_MAX / (size_t)shape[i]) {
            PyErr_SetString(PyExc_OverflowError, "Tensor shape too large");
            return NULL;
        }
        total_items *= (size_t)shape[i];
    }
    size_t const total_values = dimensions_to_values(dtype, total_items);
    if (item_size > 0 && total_values > SIZE_MAX / item_size) {
        PyErr_SetString(PyExc_OverflowError, "Tensor allocation too large");
        return NULL;
    }
    size_t const total_bytes = total_values * item_size;

    Tensor *tensor = PyObject_New(Tensor, &TensorType);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->rank = rank;
    tensor->parent = NULL;
    tensor->data = NULL;
    tensor->capacity = 0;
    tensor->exports = 0;

    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = (i < rank) ? shape[i] : 0;
        tensor->strides[i] = 0;
    }
    compute_contiguous_strides(rank, shape, dtype, tensor->strides);

    // Storage lives in a separate heap buffer so `reserve()` can grow it in place. A zero-element
    // tensor keeps `data == NULL`; `Tensor_dealloc` frees the buffer only when this owns it.
    if (total_bytes > 0) {
        tensor->data = (char *)PyMem_Malloc(total_bytes + NK_TENSOR_PADDING_);
        if (!tensor->data) {
            Py_DECREF(tensor);
            PyErr_NoMemory();
            return NULL;
        }
    }
    tensor->capacity = total_items;

    return tensor;
}

Tensor *Tensor_view(Tensor *parent, char *data_ptr, nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape,
                    Py_ssize_t const *strides) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "View rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    Tensor *view = tensor_view_header_new();
    if (!view) {
        PyErr_NoMemory();
        return NULL;
    }

    view->dtype = dtype;
    view->rank = rank;
    view->exports = 0;

    size_t numel = 1;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        view->shape[i] = (i < rank) ? shape[i] : 0;
        view->strides[i] = (i < rank) ? strides[i] : 0;
        if (i < rank) numel *= (size_t)shape[i];
    }
    view->capacity = numel; // a view owns no storage; capacity == numel (informational)

    view->parent = (PyObject *)parent;
    Py_INCREF(parent);
    view->data = data_ptr;

    return view;
}

/** Create a 0D scalar tensor. */
static Tensor *Tensor_scalar(nk_dtype_t dtype, void const *value) {
    size_t const item_size = nk_dtype_bytes_per_value(dtype);
    Tensor *tensor = PyObject_New(Tensor, &TensorType);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->rank = 0;
    tensor->parent = NULL;
    tensor->data = NULL;
    tensor->capacity = 1;
    tensor->exports = 0;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = 0;
        tensor->strides[i] = 0;
    }

    tensor->data = (char *)PyMem_Malloc(item_size + NK_TENSOR_PADDING_);
    if (!tensor->data) {
        Py_DECREF(tensor);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(tensor->data, value, item_size);

    return tensor;
}

/** Convert a 0D Tensor to a Python float. */
static PyObject *Tensor_float(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->rank != 0) {
        PyErr_SetString(PyExc_TypeError, "only 0-dimensional tensors can be converted to float");
        return NULL;
    }
    PyObject *scalar = tensor_read_scalar(tensor, 0);
    if (!scalar) return NULL;
    PyObject *result = PyNumber_Float(scalar);
    Py_DECREF(scalar);
    return result;
}

/** Convert a 0D Tensor to a Python int. */
static PyObject *Tensor_int(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->rank != 0) {
        PyErr_SetString(PyExc_TypeError, "only 0-dimensional tensors can be converted to int");
        return NULL;
    }
    PyObject *scalar = tensor_read_scalar(tensor, 0);
    if (!scalar) return NULL;
    PyObject *result = PyNumber_Long(scalar);
    Py_DECREF(scalar);
    return result;
}

static PyObject *Tensor_positive(PyObject *self) { return Tensor_copy(self, NULL, 0, NULL); }

/** Compute C-contiguous strides for a tensor shape. */
void compute_contiguous_strides(size_t rank, Py_ssize_t const *shape, nk_dtype_t dtype, Py_ssize_t *strides_out) {
    if (rank == 0) return;
    strides_out[rank - 1] = (Py_ssize_t)nk_dtype_bytes_per_value(dtype);
    for (size_t d = rank - 1; d > 0; --d) strides_out[d - 1] = strides_out[d] * storage_extent(dtype, rank, shape, d);
}

/**
 *  @brief Recursive stride walker shared by linearize_cast_into and cast_into_strided.
 *
 *  Both sides carry explicit strides, matching each_scale_recursive. Walks the outer dimensions
 *  that either side leaves unpacked; once only jointly contiguous tail dimensions remain, converts
 *  the whole slice with one memcpy or nk_cast.
 */
static void cast_strided_recursive(                                            //
    char const *src_data, nk_dtype_t src_dtype, Py_ssize_t const *src_strides, //
    char *dest_data, nk_dtype_t dest_dtype, Py_ssize_t const *dest_strides,    //
    size_t element_size, Py_ssize_t const *shape,                              //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    // Base case: both sides pack the remaining dimensions — one operation
    if (remaining_dims <= contiguous_tail_dims) {
        size_t slice_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension) slice_elements *= (size_t)shape[dimension];
        if (src_dtype == dest_dtype)
            memcpy(dest_data, src_data, dimensions_to_values(src_dtype, slice_elements) * element_size);
        else nk_cast(src_data, src_dtype, (nk_size_t)slice_elements, dest_data, dest_dtype);
        return;
    }

    // Recursive case: iterate the outermost dimension that is not jointly packed
    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        Py_ssize_t const signed_position = (Py_ssize_t)position;
        cast_strided_recursive(                                                      //
            src_data + signed_position * src_strides[0], src_dtype, src_strides + 1, //
            dest_data + signed_position * dest_strides[0], dest_dtype,               //
            dest_strides + 1, element_size, shape + 1,                               //
            remaining_dims - 1, contiguous_tail_dims);
    }
}

/** Count trailing dimensions that both stride arrays pack at their own element width. */
static size_t jointly_contiguous_tail_dimensions(size_t rank, Py_ssize_t const *shape, nk_dtype_t src_dtype,
                                                 Py_ssize_t const *src_strides, nk_dtype_t dest_dtype,
                                                 Py_ssize_t const *dest_strides) {
    size_t contiguous_tail_dims = 0;
    Py_ssize_t expected_src_stride = (Py_ssize_t)nk_dtype_bytes_per_value(src_dtype);
    Py_ssize_t expected_dest_stride = (Py_ssize_t)nk_dtype_bytes_per_value(dest_dtype);
    for (size_t dimension = rank; dimension-- > 0;) {
        if (src_strides[dimension] != expected_src_stride) break;
        if (dest_strides[dimension] != expected_dest_stride) break;
        expected_src_stride *= storage_extent(src_dtype, rank, shape, dimension);
        expected_dest_stride *= storage_extent(dest_dtype, rank, shape, dimension);
        contiguous_tail_dims++;
    }
    return contiguous_tail_dims;
}

void linearize_cast_into(char const *src_data, nk_dtype_t src_dtype, char *dest_data, nk_dtype_t dest_dtype,
                         size_t rank, Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements) {
    nk_unused_(total_elements);
    Py_ssize_t dense_dest_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides(rank, shape, dest_dtype, dense_dest_strides);
    size_t const contiguous_tail_dims = jointly_contiguous_tail_dimensions(rank, shape, src_dtype, strides, dest_dtype,
                                                                           dense_dest_strides);

    cast_strided_recursive(src_data, src_dtype, strides, dest_data, dest_dtype, dense_dest_strides,
                           nk_dtype_bytes_per_value(dest_dtype), shape, rank, contiguous_tail_dims);
}

void cast_into_strided(char const *src_data, nk_dtype_t src_dtype, char *dest_data, nk_dtype_t dest_dtype, size_t rank,
                       Py_ssize_t const *shape, Py_ssize_t const *dest_strides) {
    Py_ssize_t dense_src_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides(rank, shape, src_dtype, dense_src_strides);
    size_t const contiguous_tail_dims = jointly_contiguous_tail_dimensions(rank, shape, src_dtype, dense_src_strides,
                                                                           dest_dtype, dest_strides);

    cast_strided_recursive(src_data, src_dtype, dense_src_strides, dest_data, dest_dtype, dest_strides,
                           nk_dtype_bytes_per_value(dest_dtype), shape, rank, contiguous_tail_dims);
}

char *ensure_contiguous_buffer(char const *src_data, nk_dtype_t src_dtype, nk_dtype_t target_dtype, size_t rank,
                               Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements,
                               int *needs_free) {
    // Zero-copy: contiguous + same dtype
    if (src_dtype == target_dtype && strides_are_c_contiguous(src_dtype, rank, shape, strides)) {
        *needs_free = 0;
        return (char *)src_data;
    }

    // Single allocation, delegate
    size_t const output_bytes = dimensions_to_values(target_dtype, total_elements) *
                                nk_dtype_bytes_per_value(target_dtype);
    char *output = PyMem_Malloc(output_bytes + NK_TENSOR_PADDING_);
    if (!output) {
        PyErr_NoMemory();
        return NULL;
    }
    linearize_cast_into(src_data, src_dtype, output, target_dtype, rank, shape, strides, total_elements);
    *needs_free = 1;
    return output;
}

/** Shared helper for tensor-scalar elementwise operations via scale kernel. Computes: result =
 *  alpha * a + beta. */
static PyObject *tensor_elementwise_scalar(Tensor *a, double alpha_value, double beta_value) {
    nk_each_scale_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_scale_k, a->dtype, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_NotImplementedError, "scale not supported for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(a->dtype));
        return NULL;
    }

    Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
    if (!r) return NULL;

    size_t item_size = nk_dtype_bytes_per_value(a->dtype);
    Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides(a->rank, a->shape, a->dtype, r_strides);

    Py_buffer a_buf = {.ndim = (int)a->rank,
                       .itemsize = (Py_ssize_t)item_size,
                       .shape = a->shape,
                       .strides = a->strides};
    Py_buffer const *bufs[] = {&a_buf};
    size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 1, a->rank);

    nk_scalar_buffer_t alpha_buf, beta_buf;
    nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(a->dtype);
    alpha_buf.f64 = alpha_value, beta_buf.f64 = beta_value;
    nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
    nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);
    PyThreadState *gil = PyEval_SaveThread();
    each_scale_recursive(kernel, a->data, r->data, &alpha_buf, &beta_buf, a->shape, a->strides, r_strides, a->rank,
                         contiguous_tail);
    PyEval_RestoreThread(gil);
    return (PyObject *)r;
}

static PyObject *Tensor_negative(PyObject *self) { return tensor_elementwise_scalar((Tensor *)self, -1.0, 0.0); }

static PyObject *Tensor_add(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    // Tensor([1,2,3]) + Tensor([4,5,6]) → element-wise sum via sum kernel
    if (PyObject_TypeCheck(other, &TensorType)) {
        Tensor *b = (Tensor *)other;
        if (a->rank != b->rank || a->dtype != b->dtype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }

        nk_each_sum_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_sum_k, a->dtype, (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "add not supported for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = nk_dtype_bytes_per_value(a->dtype);
        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, a->dtype, r_strides);

        Py_buffer a_buf = {.ndim = (int)a->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = a->shape,
                           .strides = a->strides};
        Py_buffer b_buf = {.ndim = (int)b->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = b->shape,
                           .strides = b->strides};
        Py_buffer const *bufs[] = {&a_buf, &b_buf};
        size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 2, a->rank);

        PyThreadState *gil = PyEval_SaveThread();
        each_sum_recursive(kernel, a->data, b->data, r->data, a->shape, a->strides, b->strides, r_strides, a->rank,
                           contiguous_tail);
        PyEval_RestoreThread(gil);
        return (PyObject *)r;
    }

    // Tensor([1,2,3]) + 5.0 → broadcast scalar addition via scale kernel (α=1, β=scalar)
    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        return tensor_elementwise_scalar(a, 1.0, sc);
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Tensor_subtract(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    // Tensor([4,5,6]) - Tensor([1,2,3]) → element-wise difference via blend kernel (α=1, β=−1)
    if (PyObject_TypeCheck(other, &TensorType)) {
        Tensor *b = (Tensor *)other;
        if (a->rank != b->rank || a->dtype != b->dtype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }

        // Single-pass subtract via blend: result = 1 · a + (−1) · b
        nk_each_blend_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_blend_k, a->dtype, (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "subtract not supported for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = nk_dtype_bytes_per_value(a->dtype);
        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, a->dtype, r_strides);

        Py_buffer a_buf = {.ndim = (int)a->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = a->shape,
                           .strides = a->strides};
        Py_buffer b_buf = {.ndim = (int)b->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = b->shape,
                           .strides = b->strides};
        Py_buffer const *bufs[] = {&a_buf, &b_buf};
        size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 2, a->rank);

        nk_scalar_buffer_t alpha_buf, beta_buf;
        alpha_buf.f64 = 1.0, beta_buf.f64 = -1.0;
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(a->dtype);
        nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
        nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);
        PyThreadState *gil = PyEval_SaveThread();
        each_blend_recursive(kernel, a->data, b->data, r->data, &alpha_buf, &beta_buf, a->shape, a->strides, b->strides,
                             r_strides, a->rank, contiguous_tail);
        PyEval_RestoreThread(gil);
        return (PyObject *)r;
    }

    // Tensor([4,5,6]) - 1.0 → broadcast scalar subtraction via scale kernel (α=1, β=−scalar)
    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        return tensor_elementwise_scalar(a, 1.0, -sc);
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Tensor_multiply(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    // Tensor([1,2,3]) × Tensor([4,5,6]) → element-wise product via fma kernel (α=1, β=0)
    if (PyObject_TypeCheck(other, &TensorType)) {
        Tensor *b = (Tensor *)other;
        if (a->rank != b->rank || a->dtype != b->dtype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }

        nk_each_fma_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_fma_k, a->dtype, (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "multiply not supported for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = nk_dtype_bytes_per_value(a->dtype);
        size_t total_items = 1;
        for (size_t i = 0; i < a->rank; i++) total_items *= (size_t)a->shape[i];
        memset(r->data, 0, total_items * item_size); // prevent 0*NaN=NaN from uninitialized memory

        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, a->dtype, r_strides);

        Py_buffer a_buf = {.ndim = (int)a->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = a->shape,
                           .strides = a->strides};
        Py_buffer b_buf = {.ndim = (int)b->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = b->shape,
                           .strides = b->strides};
        Py_buffer const *bufs[] = {&a_buf, &b_buf};
        size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 2, a->rank);

        // fma(a, b, dummy, n, α=1, β=0) → 1 · a · b + 0 · dummy
        nk_scalar_buffer_t alpha_buf, beta_buf;
        alpha_buf.f64 = 1.0, beta_buf.f64 = 0.0;
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(a->dtype);
        nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
        nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);
        PyThreadState *gil = PyEval_SaveThread();
        each_fma_recursive(kernel, a->data, b->data, r->data, r->data, &alpha_buf, &beta_buf, a->shape, a->strides,
                           b->strides, r_strides, r_strides, a->rank, contiguous_tail);
        PyEval_RestoreThread(gil);
        return (PyObject *)r;
    }

    // Tensor([1,2,3]) × 5.0 → broadcast scalar multiply via scale kernel (α=scalar, β=0)
    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        return tensor_elementwise_scalar(a, sc, 0.0);
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyNumberMethods Tensor_as_number = {
    .nb_add = Tensor_add,
    .nb_subtract = Tensor_subtract,
    .nb_multiply = Tensor_multiply,
    .nb_matrix_multiply = Tensor_matmul,
    .nb_negative = Tensor_negative,
    .nb_positive = Tensor_positive,
    .nb_float = Tensor_float,
    .nb_int = Tensor_int,
};

static PyObject *Tensor_get_shape(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    PyObject *shape_tuple = PyTuple_New(tensor->rank);
    if (!shape_tuple) return NULL;
    for (size_t i = 0; i < tensor->rank; i++) {
        PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromSsize_t(tensor->shape[i]));
    }
    return shape_tuple;
}

static PyObject *Tensor_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    return PyUnicode_FromString(nk_dtype_python_name(tensor->dtype));
}

static PyObject *Tensor_get_ndim(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    return PyLong_FromSize_t(tensor->rank);
}

static PyObject *Tensor_get_size(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];
    return PyLong_FromSsize_t(total);
}

static PyObject *Tensor_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    size_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= (size_t)tensor->shape[i];
    return PyLong_FromSize_t(dimensions_to_values(tensor->dtype, total) * nk_dtype_bytes_per_value(tensor->dtype));
}

static PyObject *Tensor_get_strides(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    PyObject *strides_tuple = PyTuple_New(tensor->rank);
    if (!strides_tuple) return NULL;
    for (size_t i = 0; i < tensor->rank; i++) {
        PyTuple_SET_ITEM(strides_tuple, i, PyLong_FromSsize_t(tensor->strides[i]));
    }
    return strides_tuple;
}

static PyObject *Tensor_get_itemsize(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    return PyLong_FromSize_t(nk_dtype_bytes_per_value(tensor->dtype));
}

static PyObject *Tensor_get_T(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;

    if (tensor->rank < 2) {
        // 0D or 1D: transpose is a view of itself
        Py_INCREF(self);
        return self;
    }
    if (nk_dimensions_per_value(tensor->dtype) > 1) {
        PyErr_Format(PyExc_ValueError, "cannot transpose packed dtype '%s' away from its last axis",
                     nk_dtype_python_name(tensor->dtype));
        return NULL;
    }

    // Reverse shape and strides
    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
    for (size_t i = 0; i < tensor->rank; i++) {
        new_shape[i] = tensor->shape[tensor->rank - 1 - i];
        new_strides[i] = tensor->strides[tensor->rank - 1 - i];
    }

    Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
    return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, tensor->rank, new_shape, new_strides);
}

static PyObject *Tensor_get_array_interface(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;

    PyObject *dict = PyDict_New();
    if (!dict) return NULL;

    // The interface describes whole bytes, so a packed dtype reports its storage extents.
    PyObject *shape = PyTuple_New(tensor->rank);
    if (!shape) {
        Py_DECREF(dict);
        return NULL;
    }
    for (size_t i = 0; i < tensor->rank; i++)
        PyTuple_SET_ITEM(shape, i, PyLong_FromSsize_t(storage_extent(tensor->dtype, tensor->rank, tensor->shape, i)));
    PyDict_SetItemString(dict, "shape", shape);
    Py_DECREF(shape);

    char const *typestr = nk_dtype_to_numpy_typestr(tensor->dtype);
    PyObject *typestr_obj = PyUnicode_FromString(typestr);
    if (!typestr_obj) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "typestr", typestr_obj);
    Py_DECREF(typestr_obj);

    PyObject *data_ptr = PyLong_FromVoidPtr(tensor->data);
    if (!data_ptr) {
        Py_DECREF(dict);
        return NULL;
    }
    PyObject *data_tuple = PyTuple_Pack(2, data_ptr, Py_False);
    Py_DECREF(data_ptr);
    if (!data_tuple) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "data", data_tuple);
    Py_DECREF(data_tuple);

    PyObject *strides = Tensor_get_strides(self, NULL);
    if (!strides) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "strides", strides);
    Py_DECREF(strides);

    PyObject *version = PyLong_FromLong(3);
    if (!version) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "version", version);
    Py_DECREF(version);

    return dict;
}

static PyObject *Tensor_get_is_contiguous(PyObject *self, void *closure) {
    nk_unused_(closure);
    Tensor *tensor = (Tensor *)self;
    return PyBool_FromLong(tensor_is_c_contig(tensor));
}

static PyObject *Tensor_get_data_ptr(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromVoidPtr(((Tensor *)self)->data);
}

static PyObject *Tensor_get_capacity(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((Tensor *)self)->capacity);
}

/** Product of a shape's extents, the element count. */
static size_t shape_numel_(size_t rank, Py_ssize_t const *shape) {
    size_t numel = 1;
    for (size_t i = 0; i < rank; i++) numel *= (size_t)shape[i];
    return numel;
}

/** Overwrite a tensor's rank/shape/contiguous-strides in place. Caller guarantees capacity. */
static void tensor_set_contiguous_shape_(Tensor *tensor, size_t rank, Py_ssize_t const *shape) {
    tensor->rank = rank;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = (i < rank) ? shape[i] : 0;
        tensor->strides[i] = 0;
    }
    compute_contiguous_strides(rank, shape, tensor->dtype, tensor->strides);
}

/** Parse a `*ints` or single-shape-tuple argument list into an extents array. Returns 0 / -1. */
static int parse_shape_args_(PyObject *const *args, Py_ssize_t nargs, Py_ssize_t *out_shape, size_t *out_rank) {
    int const from_tuple = (nargs == 1 && PyTuple_Check(args[0]));
    size_t const rank = from_tuple ? (size_t)PyTuple_GET_SIZE(args[0]) : (size_t)nargs;
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "too many dimensions (%zu > %d)", rank, NK_TENSOR_MAX_RANK);
        return -1;
    }
    for (size_t i = 0; i < rank; i++) {
        PyObject *item = from_tuple ? PyTuple_GET_ITEM(args[0], (Py_ssize_t)i) : args[i];
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "shape dimensions must be integers");
            return -1;
        }
        Py_ssize_t const value = PyLong_AsSsize_t(item);
        if (value < 0) {
            PyErr_SetString(PyExc_ValueError, "negative dimensions not supported");
            return -1;
        }
        out_shape[i] = value;
    }
    *out_rank = rank;
    return 0;
}

char const doc_method_resize[] =                                                                          //
    "Reshape in place within the allocated capacity, without moving the data buffer.\n\n"                 //
    "Unlike reshape(), this mutates the tensor in place and keeps data_ptr stable. It fails if the new\n" //
    "element count exceeds capacity() (call reserve() to grow first), if the tensor is a view, or if\n"   //
    "any buffer is currently exported (memoryview / NumPy).\n\n"                                          //
    "Args:\n"                                                                                             //
    "    *shape: New dimensions, ints or a single shape tuple.\n\n"                                       //
    "Returns:\n"                                                                                          //
    "    Tensor: self.\n\n"                                                                               //
    "Signature:\n"                                                                                        //
    "    >>> def resize(self, *shape: int) -> 'Tensor': ...\n";

static PyObject *Tensor_resize(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->parent != NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot resize a view; only owning tensors are resizable");
        return NULL;
    }
    if (tensor->exports != 0) {
        PyErr_SetString(PyExc_BufferError, "cannot resize a tensor with exported buffers");
        return NULL;
    }
    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    size_t new_rank = 0;
    if (parse_shape_args_(args, nargs, new_shape, &new_rank) != 0) return NULL;

    if (!validate_packed_dimensions(tensor->dtype, new_rank, new_shape)) return NULL;
    size_t const new_total = shape_numel_(new_rank, new_shape);
    if (new_total > tensor->capacity) {
        PyErr_Format(PyExc_ValueError, "resize to %zu elements exceeds capacity %zu; use reserve() to grow first",
                     new_total, tensor->capacity);
        return NULL;
    }
    tensor_set_contiguous_shape_(tensor, new_rank, new_shape);
    return Py_NewRef(self);
}

char const doc_method_reserve[] =                                                                          //
    "Grow the allocated capacity to at least `capacity` elements, reallocating if needed.\n\n"             //
    "Preserves the current contents; a no-op when capacity() is already large enough. May move the data\n" //
    "buffer, so it invalidates outstanding views, memoryviews, NumPy arrays, and DLPack capsules. Fails\n" //
    "on a view or while buffers are exported.\n\n"                                                         //
    "Args:\n"                                                                                              //
    "    capacity: Minimum element capacity to guarantee.\n\n"                                             //
    "Returns:\n"                                                                                           //
    "    None.\n\n"                                                                                        //
    "Signature:\n"                                                                                         //
    "    >>> def reserve(self, capacity: int) -> None: ...\n";

static PyObject *Tensor_reserve(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->parent != NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot reserve on a view; only owning tensors own storage");
        return NULL;
    }
    if (tensor->exports != 0) {
        PyErr_SetString(PyExc_BufferError, "cannot reserve while buffers are exported");
        return NULL;
    }
    if (nargs != 1 || !PyLong_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "reserve() takes a single integer capacity");
        return NULL;
    }
    Py_ssize_t const requested = PyLong_AsSsize_t(args[0]);
    if (requested < 0) {
        PyErr_SetString(PyExc_ValueError, "capacity must be non-negative");
        return NULL;
    }
    if ((size_t)requested <= tensor->capacity) Py_RETURN_NONE; // already large enough

    Py_ssize_t const requested_shape[1] = {requested};
    if (!validate_packed_dimensions(tensor->dtype, 1, requested_shape)) return NULL;
    size_t const requested_bytes = dimensions_to_values(tensor->dtype, (size_t)requested) *
                                   nk_dtype_bytes_per_value(tensor->dtype);
    char *grown = (char *)PyMem_Realloc(tensor->data, requested_bytes + NK_TENSOR_PADDING_);
    if (!grown) {
        PyErr_NoMemory();
        return NULL;
    }
    tensor->data = grown;
    tensor->capacity = (size_t)requested;
    Py_RETURN_NONE;
}

char const doc_method_clear[] =                                                                     //
    "Reset to an empty shape, size 0, while keeping the allocated capacity.\n\n"                    //
    "The buffer can be refilled via resize() without reallocating. Fails on a view, or while any\n" //
    "buffers stay exported.\n\n"                                                                    //
    "Returns:\n"                                                                                    //
    "    None.\n\n"                                                                                 //
    "Signature:\n"                                                                                  //
    "    >>> def clear(self) -> None: ...\n";

static PyObject *Tensor_clear(PyObject *self, PyObject *ignored) {
    nk_unused_(ignored);
    Tensor *tensor = (Tensor *)self;
    if (tensor->parent != NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot clear a view");
        return NULL;
    }
    if (tensor->exports != 0) {
        PyErr_SetString(PyExc_BufferError, "cannot clear a tensor with exported buffers");
        return NULL;
    }
    tensor->rank = 1;
    tensor->shape[0] = 0;
    tensor->strides[0] = (Py_ssize_t)nk_dtype_bytes_per_value(tensor->dtype);
    for (size_t i = 1; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = 0;
        tensor->strides[i] = 0;
    }
    Py_RETURN_NONE;
}

static Tensor *Tensor_view_object(PyObject *owner, char *data_ptr, nk_dtype_t dtype, size_t rank,
                                  Py_ssize_t const *shape, Py_ssize_t const *strides) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "View rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    Tensor *view = tensor_view_header_new();
    if (!view) {
        PyErr_NoMemory();
        return NULL;
    }

    view->dtype = dtype;
    view->rank = rank;
    view->exports = 0;

    size_t numel = 1;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        view->shape[i] = (i < rank) ? shape[i] : 0;
        view->strides[i] = (i < rank) ? strides[i] : 0;
        if (i < rank) numel *= (size_t)shape[i];
    }
    view->capacity = numel; // a view owns no storage; capacity == numel (informational)

    view->parent = owner;
    Py_INCREF(owner);
    view->data = data_ptr;

    return view;
}

static PyGetSetDef Tensor_getset[] = {
    {"shape", Tensor_get_shape, NULL, "Shape of the array", NULL},
    {"dtype", Tensor_get_dtype, NULL, "Data type of the array", NULL},
    {"ndim", Tensor_get_ndim, NULL, "Number of dimensions", NULL},
    {"size", Tensor_get_size, NULL, "Total number of elements", NULL},
    {"nbytes", Tensor_get_nbytes, NULL, "Total bytes of data", NULL},
    {"strides", Tensor_get_strides, NULL, "Strides in bytes", NULL},
    {"itemsize", Tensor_get_itemsize, NULL, "Size of one element in bytes", NULL},
    {"T", Tensor_get_T, NULL, "Transposed view of the array", NULL},
    {"__array_interface__", Tensor_get_array_interface, NULL, "NumPy array interface", NULL},
    {"is_contiguous", Tensor_get_is_contiguous, NULL, "Whether the tensor is C-contiguous", NULL},
    {"data_ptr", Tensor_get_data_ptr, NULL, "Integer address of the data buffer", NULL},
    {"capacity", Tensor_get_capacity, NULL, "Allocated element capacity of the owned buffer (>= size)", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/**
 *  @brief Validate a user-supplied `out=` Tensor for an into-buffer op and return its data pointer.
 *
 *  Requires @c out to be a Tensor of exactly @p out_dtype, matching @p rank / @p shape, and
 *  C-contiguous — the layout @c linearize_cast_into writes. Sets a Python error and returns NULL on
 *  any mismatch. Shared by @c astype, @c copy, and @c flatten.
 */
static char *validate_out_buffer(PyObject *out_obj, nk_dtype_t out_dtype, size_t rank, Py_ssize_t const *shape) {
    if (!PyObject_TypeCheck(out_obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
        return NULL;
    }
    Tensor *out = (Tensor *)out_obj;
    if (out->dtype != out_dtype) {
        PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected '%s'",
                     nk_dtype_to_pybuffer_typestr(out->dtype), nk_dtype_to_pybuffer_typestr(out_dtype));
        return NULL;
    }
    if (out->rank != rank) {
        PyErr_Format(PyExc_ValueError, "out rank %zu does not match expected %zu", out->rank, rank);
        return NULL;
    }
    for (size_t i = 0; i < rank; i++)
        if (out->shape[i] != shape[i]) {
            PyErr_SetString(PyExc_ValueError, "out shape does not match expected shape");
            return NULL;
        }
    if (!tensor_is_c_contig(out)) {
        PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
        return NULL;
    }
    return out->data;
}

char const doc_method_copy[] =                                                  //
    "Return a deep copy of the tensor.\n\n"                                     //
    "Args:\n"                                                                   //
    "    out (Tensor, optional): Pre-allocated destination of the same dtype\n" //
    "        and shape, C-contiguous. When given, the copy writes into it\n"    //
    "        with no allocation and returns `out`.\n\n"                         //
    "Returns:\n"                                                                //
    "    Tensor: Independent copy, or `out` when provided.\n\n"                 //
    "Signature:\n"                                                              //
    "    >>> def copy(self, /, *, out=None): ...";

PyObject *Tensor_copy(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;

    PyObject *out_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 0) {
        PyErr_SetString(PyExc_TypeError, "copy(*, out=None) takes no positional arguments");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) out_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "copy() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elements *= (size_t)tensor->shape[i];

    if (out_obj && out_obj != Py_None) {
        char *out_data = validate_out_buffer(out_obj, tensor->dtype, tensor->rank, tensor->shape);
        if (!out_data) return NULL;
        linearize_cast_into(tensor->data, tensor->dtype, out_data, tensor->dtype, tensor->rank, tensor->shape,
                            tensor->strides, total_elements);
        Py_INCREF(out_obj);
        return out_obj;
    }

    Tensor *result = Tensor_new(tensor->dtype, tensor->rank, tensor->shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, tensor->dtype, tensor->rank, tensor->shape,
                        tensor->strides, total_elements);
    return (PyObject *)result;
}

char const doc_method_reshape[] =                                                       //
    "Return a tensor reshaped to the given dimensions.\n\n"                             //
    "Args:\n"                                                                           //
    "    *dims (int): New dimensions. Total element count must match the original.\n\n" //
    "Returns:\n"                                                                        //
    "    Tensor: Reshaped view, or a copy if the layout requires it.\n\n"               //
    "Signature:\n"                                                                      //
    "    >>> def reshape(self, *dims): ...";

PyObject *Tensor_reshape(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    Tensor *tensor = (Tensor *)self;

    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    size_t new_rank = 0;

    if (nargs == 1 && PyTuple_Check(args[0])) {
        PyObject *shape_tuple = args[0];
        new_rank = PyTuple_GET_SIZE(shape_tuple);
        if (new_rank > NK_TENSOR_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_TENSOR_MAX_RANK);
            return NULL;
        }
        for (size_t i = 0; i < new_rank; i++) {
            PyObject *item = PyTuple_GET_ITEM(shape_tuple, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "reshape: shape dimensions must be integers");
                return NULL;
            }
            new_shape[i] = PyLong_AsSsize_t(item);
            if (new_shape[i] < 0) {
                PyErr_SetString(PyExc_ValueError, "reshape: negative dimensions not supported");
                return NULL;
            }
        }
    }
    else {
        new_rank = (size_t)nargs;
        if (new_rank > NK_TENSOR_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_TENSOR_MAX_RANK);
            return NULL;
        }
        for (size_t i = 0; i < new_rank; i++) {
            PyObject *item = args[i];
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "reshape: shape dimensions must be integers");
                return NULL;
            }
            new_shape[i] = PyLong_AsSsize_t(item);
            if (new_shape[i] < 0) {
                PyErr_SetString(PyExc_ValueError, "reshape: negative dimensions not supported");
                return NULL;
            }
        }
    }

    Py_ssize_t new_total = 1;
    for (size_t i = 0; i < new_rank; i++) new_total *= new_shape[i];

    Py_ssize_t old_total = 1;
    for (size_t i = 0; i < tensor->rank; i++) old_total *= tensor->shape[i];

    if (new_total != old_total) {
        PyErr_Format(PyExc_ValueError, "reshape: cannot reshape tensor of size %zd into shape with size %zd", old_total,
                     new_total);
        return NULL;
    }

    if (!validate_packed_dimensions(tensor->dtype, new_rank, new_shape)) return NULL;

    if (tensor_is_c_contig(tensor)) {
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(new_rank, new_shape, tensor->dtype, new_strides);
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
        return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, new_rank, new_shape, new_strides);
    }

    // Non-contiguous: must copy
    Tensor *result = Tensor_new(tensor->dtype, new_rank, new_shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, tensor->dtype, tensor->rank, tensor->shape,
                        tensor->strides, (size_t)old_total);
    return (PyObject *)result;
}

char const doc_method_flatten[] =                                                          //
    "Return a flattened 1D view of the tensor.\n\n"                                        //
    "Args:\n"                                                                              //
    "    out (Tensor, optional): Pre-allocated 1D destination of the same dtype, length\n" //
    "        `size`, C-contiguous, filled with no allocation, never a view; `out` is\n"    //
    "        returned.\n\n"                                                                //
    "Returns:\n"                                                                           //
    "    Tensor: 1D view when contiguous and `out` is omitted; otherwise a 1D copy, or\n"  //
    "        the provided `out`.\n\n"                                                      //
    "Signature:\n"                                                                         //
    "    >>> def flatten(self, /, *, out=None): ...";

PyObject *Tensor_flatten(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;

    PyObject *out_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 0) {
        PyErr_SetString(PyExc_TypeError, "flatten(*, out=None) takes no positional arguments");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) out_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "flatten() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }

    Py_ssize_t total_elements = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elements *= tensor->shape[i];

    // Into a caller buffer: always materialize (copy), never return a view.
    if (out_obj && out_obj != Py_None) {
        Py_ssize_t flat_shape[1] = {total_elements};
        char *out_data = validate_out_buffer(out_obj, tensor->dtype, 1, flat_shape);
        if (!out_data) return NULL;
        linearize_cast_into(tensor->data, tensor->dtype, out_data, tensor->dtype, tensor->rank, tensor->shape,
                            tensor->strides, (size_t)total_elements);
        Py_INCREF(out_obj);
        return out_obj;
    }

    if (tensor_is_c_contig(tensor)) {
        Py_ssize_t flat_shape[1] = {total_elements};
        Py_ssize_t flat_strides[1] = {(Py_ssize_t)nk_dtype_bytes_per_value(tensor->dtype)};
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
        return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, 1, flat_shape, flat_strides);
    }

    // Non-contiguous: must copy then flatten
    Py_ssize_t flat_shape[1] = {total_elements};
    Tensor *result = Tensor_new(tensor->dtype, 1, flat_shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, tensor->dtype, tensor->rank, tensor->shape,
                        tensor->strides, (size_t)total_elements);
    return (PyObject *)result;
}

char const doc_method_squeeze[] =                                                           //
    "Remove dimensions of size 1.\n\n"                                                      //
    "Args:\n"                                                                               //
    "    *axes (int, optional): Specific axes to squeeze. If omitted, all size-1 axes.\n\n" //
    "Returns:\n"                                                                            //
    "    Tensor: View with the specified size-1 dimensions removed.\n\n"                    //
    "Signature:\n"                                                                          //
    "    >>> def squeeze(self, /, *axes): ...";

PyObject *Tensor_squeeze(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    Tensor *tensor = (Tensor *)self;

    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "squeeze() takes at most 1 argument");
        return NULL;
    }

    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
    size_t new_rank = 0;

    if (nargs == 1) {
        // Squeeze a specific axis
        Py_ssize_t axis = PyLong_AsSsize_t(args[0]);
        if (axis == -1 && PyErr_Occurred()) return NULL;
        if (axis < 0) axis += (Py_ssize_t)tensor->rank;
        if (axis < 0 || (size_t)axis >= tensor->rank) {
            PyErr_Format(PyExc_ValueError, "squeeze: axis %zd out of range for tensor with %zu dimensions", axis,
                         tensor->rank);
            return NULL;
        }
        if (tensor->shape[axis] != 1) {
            // Axis is not size 1, return a view of the same tensor
            Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
            return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, tensor->rank, tensor->shape,
                                           tensor->strides);
        }
        for (size_t i = 0; i < tensor->rank; i++) {
            if ((size_t)i != (size_t)axis) {
                new_shape[new_rank] = tensor->shape[i];
                new_strides[new_rank] = tensor->strides[i];
                new_rank++;
            }
        }
    }
    else {
        // Squeeze all size-1 dimensions
        for (size_t i = 0; i < tensor->rank; i++) {
            if (tensor->shape[i] != 1) {
                new_shape[new_rank] = tensor->shape[i];
                new_strides[new_rank] = tensor->strides[i];
                new_rank++;
            }
        }
    }

    if (new_rank == 0) {
        new_rank = 1;
        new_shape[0] = 1;
        new_strides[0] = (Py_ssize_t)nk_dtype_bytes_per_value(tensor->dtype);
    }

    Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
    return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, new_rank, new_shape, new_strides);
}

/** Add a partial sum value into a running accumulator, using the accumulator dtype. */
static void accum_add(void *accum, void const *partial, nk_dtype_t accum_dtype) {
    switch (accum_dtype) {
    case nk_f64_k: *(nk_f64_t *)accum += *(nk_f64_t const *)partial; break;
    case nk_f32_k: *(nk_f32_t *)accum += *(nk_f32_t const *)partial; break;
    case nk_i64_k:
        *(nk_i64_t *)accum = nk_i64_saturating_add(*(nk_i64_t const *)accum, *(nk_i64_t const *)partial);
        break;
    case nk_u64_k:
        *(nk_u64_t *)accum = nk_u64_saturating_add(*(nk_u64_t const *)accum, *(nk_u64_t const *)partial);
        break;
    default: break;
    }
}

/** Detect trailing dimensions that form a single arithmetic progression. Returns how many rightmost
 *  dims satisfy stride[i] == stride[i+1] * extent[i+1], where a packed @p dtype counts its
 *  innermost extent in storage values. When all dims collapse, the tensor forms one strided run. */
static size_t uniform_stride_tail_dims(nk_dtype_t dtype, Py_ssize_t const *shape, Py_ssize_t const *strides,
                                       size_t rank, size_t *out_count, Py_ssize_t *out_stride) {
    if (rank == 0) {
        *out_count = 1;
        *out_stride = 0;
        return 0;
    }
    size_t tail = 1;
    Py_ssize_t innermost = strides[rank - 1];
    Py_ssize_t expected = innermost;
    for (size_t i = rank - 1; i > 0; --i) {
        expected = expected * storage_extent(dtype, rank, shape, i);
        if (strides[i - 1] != expected) break;
        ++tail;
    }
    size_t count = 1;
    for (size_t i = rank - tail; i < rank; ++i) count *= (size_t)shape[i];
    *out_count = count;
    *out_stride = innermost;
    return tail;
}

/** Normalize a strided pointer for SIMD kernel consumption. For negative strides, flips the base
 *  pointer to the last element so the kernel walks forward. */
static char const *normalize_strided_base(char const *data, size_t count, Py_ssize_t stride,
                                          Py_ssize_t *out_stride_magnitude) {
    if (stride < 0) {
        *out_stride_magnitude = -stride;
        return data + (Py_ssize_t)(count - 1) * stride;
    }
    *out_stride_magnitude = stride;
    return data;
}

/**
 *  @brief Collapse trailing uniform-stride dims, writing the result into caller-provided arrays.
 *  @return The new rank after collapsing, @p remaining_dims - @p tail_dims + 1.
 */
static size_t build_collapsed_shape(Py_ssize_t const *shape, Py_ssize_t const *strides, size_t dim, size_t tail_dims,
                                    size_t collapsed_count, Py_ssize_t collapsed_stride, Py_ssize_t *out_shape,
                                    Py_ssize_t *out_strides, size_t remaining_dims) {
    size_t collapsed_rank = remaining_dims - tail_dims + 1;
    for (size_t i = 0; i < collapsed_rank - 1; i++) {
        out_shape[i] = shape[dim + i];
        out_strides[i] = strides[dim + i];
    }
    out_shape[collapsed_rank - 1] = (Py_ssize_t)collapsed_count;
    out_strides[collapsed_rank - 1] = collapsed_stride;
    return collapsed_rank;
}

/** Recursively reduce moments over an N-D tensor using a SIMD kernel. Re-analyzes remaining
 *  dimensions at each level to collapse uniform-stride tails. */
static void reduce_moments_recursive(                    //
    nk_reduce_moments_punned_t kernel, nk_dtype_t dtype, //
    nk_dtype_t sum_dtype, nk_dtype_t sumsq_dtype,        //
    char const *data, Py_ssize_t const *shape,           //
    Py_ssize_t const *strides, size_t rank, size_t dim,  //
    nk_scalar_buffer_t *sum_accum, nk_scalar_buffer_t *sumsq_accum) {

    size_t remaining = rank - dim;
    if (remaining >= 2) {
        size_t collapsed_count;
        Py_ssize_t collapsed_stride;
        size_t tail = uniform_stride_tail_dims(dtype, shape + dim, strides + dim, remaining, &collapsed_count,
                                               &collapsed_stride);
        if (tail == remaining) {
            Py_ssize_t stride_magnitude;
            char const *base = normalize_strided_base(data, collapsed_count, collapsed_stride, &stride_magnitude);
            nk_scalar_buffer_t partial_sum, partial_sumsq;
            memset(&partial_sum, 0, sizeof(partial_sum));
            memset(&partial_sumsq, 0, sizeof(partial_sumsq));
            kernel(base, (nk_size_t)collapsed_count, (nk_size_t)stride_magnitude, &partial_sum, &partial_sumsq);
            accum_add(sum_accum, &partial_sum, sum_dtype);
            accum_add(sumsq_accum, &partial_sumsq, sumsq_dtype);
            return;
        }
        if (tail >= 2) {
            Py_ssize_t collapsed_shape[NK_TENSOR_MAX_RANK], collapsed_strides[NK_TENSOR_MAX_RANK];
            size_t collapsed_rank = build_collapsed_shape(shape, strides, dim, tail, collapsed_count, collapsed_stride,
                                                          collapsed_shape, collapsed_strides, remaining);
            reduce_moments_recursive(kernel, dtype, sum_dtype, sumsq_dtype, data, collapsed_shape, collapsed_strides,
                                     collapsed_rank, 0, sum_accum, sumsq_accum);
            return;
        }
    }
    if (remaining == 1) {
        nk_scalar_buffer_t partial_sum, partial_sumsq;
        memset(&partial_sum, 0, sizeof(partial_sum));
        memset(&partial_sumsq, 0, sizeof(partial_sumsq));
        kernel(data, (nk_size_t)shape[dim], (nk_size_t)strides[dim], &partial_sum, &partial_sumsq);
        accum_add(sum_accum, &partial_sum, sum_dtype);
        accum_add(sumsq_accum, &partial_sumsq, sumsq_dtype);
    }
    else {
        size_t const n = (size_t)shape[dim];
        Py_ssize_t const stride = strides[dim];
        for (size_t i = 0; i < n; i++)
            reduce_moments_recursive(kernel, dtype, sum_dtype, sumsq_dtype, data + i * stride, shape, strides, rank,
                                     dim + 1, sum_accum, sumsq_accum);
    }
}

/** Reduce moments over an N-D tensor, returning typed scalar buffers. */
static int impl_reduce_moments(TensorView const *view, nk_scalar_buffer_t *sum_out, nk_dtype_t *sum_dtype_out,
                               nk_scalar_buffer_t *sumsq_out, nk_dtype_t *sumsq_dtype_out) {

    nk_reduce_moments_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_reduce_moments_k, view->dtype, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) return -1;

    nk_dtype_t sum_dtype = nk_reduce_moments_sum_dtype(view->dtype);
    nk_dtype_t sumsq_dtype = nk_reduce_moments_sumsq_dtype(view->dtype);

    nk_scalar_buffer_t sum_buf, sumsq_buf;
    memset(&sum_buf, 0, sizeof(sum_buf));
    memset(&sumsq_buf, 0, sizeof(sumsq_buf));

    // A 0-D tensor holds a single contiguous element: pass the element size as the stride so the
    // SIMD strided paths never see a zero stride (which would make their step computation loop forever).
    if (view->rank == 0) { kernel(view->data, 1, nk_dtype_bytes_per_value(view->dtype), &sum_buf, &sumsq_buf); }
    else {
        reduce_moments_recursive(kernel, view->dtype, sum_dtype, sumsq_dtype, view->data, view->shape, view->strides,
                                 view->rank, 0, &sum_buf, &sumsq_buf);
    }

    *sum_out = sum_buf;
    *sumsq_out = sumsq_buf;
    *sum_dtype_out = sum_dtype;
    *sumsq_dtype_out = sumsq_dtype;
    return 0;
}

/** Type-aware less-than comparison for scalar buffers. Uses native comparisons for standard types
 *  and sign-magnitude comparators for mini-floats. */
static int minmax_less_than(nk_scalar_buffer_t const *a, nk_scalar_buffer_t const *b, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return a->f64 < b->f64;
    case nk_f32_k: return a->f32 < b->f32;
    case nk_i64_k: return a->i64 < b->i64;
    case nk_u64_k: return a->u64 < b->u64;
    case nk_i32_k: return a->i32 < b->i32;
    case nk_u32_k: return a->u32 < b->u32;
    case nk_i16_k: return a->i16 < b->i16;
    case nk_u16_k: return a->u16 < b->u16;
    case nk_i8_k: return a->i8 < b->i8;
    case nk_u8_k: return a->u8 < b->u8;
    case nk_f16_k: return nk_f16_order(a->f16, b->f16) < 0;
    case nk_bf16_k: return nk_bf16_order(a->bf16, b->bf16) < 0;
    case nk_e4m3_k: return nk_e4m3_order(a->u8, b->u8) < 0;
    case nk_e5m2_k: return nk_e5m2_order(a->u8, b->u8) < 0;
    case nk_e2m3_k: return nk_e2m3_order(a->u8, b->u8) < 0;
    case nk_e3m2_k: return nk_e3m2_order(a->u8, b->u8) < 0;
    default: return 0;
    }
}

/** Update minmax accumulators with partial results from a SIMD kernel call. The SIMD kernel returns
 *  min/max values in the value dtype and indices relative to the slice. We compare against the
 *  running accumulators and update if better. */
static void minmax_update(nk_scalar_buffer_t *running_min, nk_size_t *running_min_idx, nk_scalar_buffer_t *running_max,
                          nk_size_t *running_max_idx, nk_scalar_buffer_t const *partial_min, nk_size_t partial_min_idx,
                          nk_scalar_buffer_t const *partial_max, nk_size_t partial_max_idx, nk_dtype_t value_dtype,
                          size_t flat_offset) {
    if (minmax_less_than(partial_min, running_min, value_dtype)) {
        memcpy(running_min, partial_min, sizeof(*running_min));
        *running_min_idx = flat_offset + partial_min_idx;
    }
    if (minmax_less_than(running_max, partial_max, value_dtype)) {
        memcpy(running_max, partial_max, sizeof(*running_max));
        *running_max_idx = flat_offset + partial_max_idx;
    }
}

/** Recursively reduce minmax over an N-D tensor using a SIMD kernel. Re-analyzes remaining
 *  dimensions at each level to collapse uniform-stride tails. */
static void reduce_minmax_recursive(                         //
    nk_reduce_minmax_punned_t kernel, nk_dtype_t dtype,      //
    nk_dtype_t value_dtype,                                  //
    char const *data, Py_ssize_t const *shape,               //
    Py_ssize_t const *strides, size_t rank, size_t dim,      //
    size_t flat_offset,                                      //
    nk_scalar_buffer_t *min_accum, nk_size_t *min_idx_accum, //
    nk_scalar_buffer_t *max_accum, nk_size_t *max_idx_accum) {

    size_t remaining = rank - dim;
    if (remaining >= 2) {
        size_t collapsed_count;
        Py_ssize_t collapsed_stride;
        size_t tail = uniform_stride_tail_dims(dtype, shape + dim, strides + dim, remaining, &collapsed_count,
                                               &collapsed_stride);
        if (tail == remaining) {
            Py_ssize_t stride_magnitude;
            char const *base = normalize_strided_base(data, collapsed_count, collapsed_stride, &stride_magnitude);
            nk_scalar_buffer_t partial_min, partial_max;
            memset(&partial_min, 0, sizeof(partial_min));
            memset(&partial_max, 0, sizeof(partial_max));
            nk_size_t partial_min_idx = 0, partial_max_idx = 0;
            kernel(base, (nk_size_t)collapsed_count, (nk_size_t)stride_magnitude, &partial_min, &partial_min_idx,
                   &partial_max, &partial_max_idx);
            if (collapsed_stride < 0) {
                partial_min_idx = (nk_size_t)(collapsed_count - 1) - partial_min_idx;
                partial_max_idx = (nk_size_t)(collapsed_count - 1) - partial_max_idx;
            }
            minmax_update(min_accum, min_idx_accum, max_accum, max_idx_accum, &partial_min, partial_min_idx,
                          &partial_max, partial_max_idx, value_dtype, flat_offset);
            return;
        }
        if (tail >= 2) {
            Py_ssize_t collapsed_shape[NK_TENSOR_MAX_RANK], collapsed_strides[NK_TENSOR_MAX_RANK];
            size_t collapsed_rank = build_collapsed_shape(shape, strides, dim, tail, collapsed_count, collapsed_stride,
                                                          collapsed_shape, collapsed_strides, remaining);
            reduce_minmax_recursive(kernel, dtype, value_dtype, data, collapsed_shape, collapsed_strides,
                                    collapsed_rank, 0, flat_offset, min_accum, min_idx_accum, max_accum, max_idx_accum);
            return;
        }
    }
    if (remaining == 1) {
        nk_scalar_buffer_t partial_min, partial_max;
        memset(&partial_min, 0, sizeof(partial_min));
        memset(&partial_max, 0, sizeof(partial_max));
        nk_size_t partial_min_idx = 0, partial_max_idx = 0;
        kernel(data, (nk_size_t)shape[dim], (nk_size_t)strides[dim], &partial_min, &partial_min_idx, &partial_max,
               &partial_max_idx);
        minmax_update(min_accum, min_idx_accum, max_accum, max_idx_accum, &partial_min, partial_min_idx, &partial_max,
                      partial_max_idx, value_dtype, flat_offset);
    }
    else {
        size_t const n = (size_t)shape[dim];
        Py_ssize_t const stride = strides[dim];
        size_t inner_size = 1;
        for (size_t d = dim + 1; d < rank; d++) inner_size *= (size_t)shape[d];
        for (size_t i = 0; i < n; i++)
            reduce_minmax_recursive(kernel, dtype, value_dtype, data + i * stride, shape, strides, rank, dim + 1,
                                    flat_offset + i * inner_size, min_accum, min_idx_accum, max_accum, max_idx_accum);
    }
}

/** Reduce minmax over an N-D tensor, returning typed scalar buffers. */
static int impl_reduce_minmax(TensorView const *view, nk_scalar_buffer_t *min_out, nk_dtype_t *min_dtype_out,
                              size_t *min_index_out, nk_scalar_buffer_t *max_out, nk_dtype_t *max_dtype_out,
                              size_t *max_index_out) {

    nk_reduce_minmax_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_reduce_minmax_k, view->dtype, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) return -1;

    nk_dtype_t value_dtype = nk_reduce_minmax_value_dtype(view->dtype);

    // For minmax, the SIMD kernel already initializes from the data it processes.
    // For multi-dimensional tensors, we need extreme initial values for the running accumulators.
    // Use the first element as initialization by calling the kernel on the first element,
    // then let the full recursive traversal find the real min/max.
    nk_scalar_buffer_t min_buf, max_buf;
    memset(&min_buf, 0, sizeof(min_buf));
    memset(&max_buf, 0, sizeof(max_buf));
    nk_size_t min_idx = 0, max_idx = 0;

    // An empty tensor has no element to prime the running min/max from; reading element [0] would be
    // an out-of-bounds access on the zero-length allocation. Return the zeroed result instead.
    nk_size_t element_count = 1;
    for (size_t d = 0; d < view->rank; ++d) element_count *= (nk_size_t)view->shape[d];
    if (element_count == 0) {
        *min_out = min_buf, *max_out = max_buf;
        *min_dtype_out = value_dtype, *max_dtype_out = value_dtype;
        *min_index_out = 0, *max_index_out = 0;
        return 0;
    }

    // See the moments path: a 0-D tensor is one contiguous element, so use the element size as the
    // stride rather than zero to keep the SIMD strided loops well-formed.
    if (view->rank == 0) {
        kernel(view->data, 1, nk_dtype_bytes_per_value(view->dtype), &min_buf, &min_idx, &max_buf, &max_idx);
    }
    else {
        size_t elem_size = nk_dtype_bytes_per_value(value_dtype);
        kernel(view->data, 1, (nk_size_t)elem_size, &min_buf, &min_idx, &max_buf, &max_idx);
        reduce_minmax_recursive(kernel, view->dtype, value_dtype, view->data, view->shape, view->strides, view->rank, 0,
                                0, &min_buf, &min_idx, &max_buf, &max_idx);
    }

    *min_out = min_buf;
    *max_out = max_buf;
    *min_dtype_out = value_dtype;
    *max_dtype_out = value_dtype;
    *min_index_out = (size_t)min_idx;
    *max_index_out = (size_t)max_idx;
    return 0;
}

char const doc_method_moments[] =                            //
    "Compute sum and sum-of-squares of all elements.\n\n"    //
    "Returns:\n"                                             //
    "    tuple: (sum, sum_of_squares) for all elements.\n\n" //
    "Signature:\n"                                           //
    "    >>> def moments(self, /): ...";

static PyObject *impl_moments_from_view(TensorView const *view) {
    nk_scalar_buffer_t sum_buf, sumsq_buf;
    nk_dtype_t sum_dtype, sumsq_dtype;
    if (impl_reduce_moments(view, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype) < 0)
        return PyErr_Format(PyExc_NotImplementedError, "moments not supported for dtype '%s'",
                            nk_dtype_to_pybuffer_typestr(view->dtype));
    PyObject *sum_obj = nk_scalar_buffer_to_py_number(&sum_buf, sum_dtype);
    if (!sum_obj) return NULL;
    PyObject *sumsq_obj = nk_scalar_buffer_to_py_number(&sumsq_buf, sumsq_dtype);
    if (!sumsq_obj) {
        Py_DECREF(sum_obj);
        return NULL;
    }
    PyObject *tuple = PyTuple_Pack(2, sum_obj, sumsq_obj);
    Py_DECREF(sum_obj);
    Py_DECREF(sumsq_obj);
    return tuple;
}

PyObject *Tensor_moments(PyObject *self, PyObject *args) {
    nk_unused_(args);
    Tensor *t = (Tensor *)self;
    TensorView view = {t->dtype, t->rank, t->shape, t->strides, t->data};
    return impl_moments_from_view(&view);
}

char const doc_method_minmax[] =                                                                //
    "Find minimum and maximum elements with their indices.\n\n"                                 //
    "Returns:\n"                                                                                //
    "    tuple: (min_val, min_index, max_val, max_index), or None if all elements are NaN.\n\n" //
    "Signature:\n"                                                                              //
    "    >>> def minmax(self, /): ...";

static PyObject *impl_minmax_from_view(TensorView const *view) {
    nk_scalar_buffer_t min_buf, max_buf;
    nk_dtype_t min_dtype, max_dtype;
    size_t min_index = 0, max_index = 0;
    if (impl_reduce_minmax(view, &min_buf, &min_dtype, &min_index, &max_buf, &max_dtype, &max_index) < 0)
        return PyErr_Format(PyExc_NotImplementedError, "minmax not supported for dtype '%s'",
                            nk_dtype_to_pybuffer_typestr(view->dtype));
    if (min_index == NK_SIZE_MAX) { Py_RETURN_NONE; }
    PyObject *min_obj = nk_scalar_buffer_to_py_number(&min_buf, min_dtype);
    if (!min_obj) return NULL;
    PyObject *min_idx_obj = PyLong_FromSsize_t((Py_ssize_t)min_index);
    if (!min_idx_obj) {
        Py_DECREF(min_obj);
        return NULL;
    }
    PyObject *max_obj = nk_scalar_buffer_to_py_number(&max_buf, max_dtype);
    if (!max_obj) {
        Py_DECREF(min_obj);
        Py_DECREF(min_idx_obj);
        return NULL;
    }
    PyObject *max_idx_obj = PyLong_FromSsize_t((Py_ssize_t)max_index);
    if (!max_idx_obj) {
        Py_DECREF(min_obj);
        Py_DECREF(min_idx_obj);
        Py_DECREF(max_obj);
        return NULL;
    }
    PyObject *tuple = PyTuple_Pack(4, min_obj, min_idx_obj, max_obj, max_idx_obj);
    Py_DECREF(min_obj);
    Py_DECREF(min_idx_obj);
    Py_DECREF(max_obj);
    Py_DECREF(max_idx_obj);
    return tuple;
}

PyObject *Tensor_minmax(PyObject *self, PyObject *args) {
    nk_unused_(args);
    Tensor *t = (Tensor *)self;
    TensorView view = {t->dtype, t->rank, t->shape, t->strides, t->data};
    return impl_minmax_from_view(&view);
}

typedef struct {
    Py_ssize_t axes[NK_TENSOR_MAX_RANK]; // sorted, normalized axes to reduce
    size_t n_axes;                       // 0 = reduce-all (axis=None), else 1..rank
    int keepdims;                        // 0 or 1
    Tensor *out;                         // NULL or user-provided
    nk_dtype_t dtype_override;           // nk_dtype_unknown_k means "use source dtype"
} reduce_args_t;

/** Parse a single axis value, int, tuple of ints, or None, into the axes array. Returns 1 if axes
 *  were set, 0 if None, -1 on error. */
static int parse_axis_value(PyObject *value, reduce_args_t *parsed) {
    if (value == Py_None) {
        parsed->n_axes = 0;
        return 0;
    }
    if (PyLong_Check(value)) {
        parsed->axes[0] = PyLong_AsSsize_t(value);
        if (parsed->axes[0] == -1 && PyErr_Occurred()) return -1;
        parsed->n_axes = 1;
        return 1;
    }
    if (PyTuple_Check(value)) {
        Py_ssize_t const len = PyTuple_GET_SIZE(value);
        if (len == 0) return (PyErr_SetString(PyExc_ValueError, "empty axis tuple"), -1);
        if ((size_t)len > NK_TENSOR_MAX_RANK) return (PyErr_SetString(PyExc_ValueError, "too many axes"), -1);
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject *item = PyTuple_GET_ITEM(value, i);
            if (!PyLong_Check(item))
                return (PyErr_SetString(PyExc_TypeError, "axis tuple elements must be integers"), -1);
            parsed->axes[i] = PyLong_AsSsize_t(item);
            if (parsed->axes[i] == -1 && PyErr_Occurred()) return -1;
        }
        parsed->n_axes = (size_t)len;
        return 1;
    }
    PyErr_SetString(PyExc_TypeError, "axis must be None, an integer, or a tuple of integers");
    return -1;
}

/** Normalize, sort, and validate the axes array against tensor rank. */
static int normalize_axes(reduce_args_t *parsed, size_t tensor_rank) {
    // Normalize negative indices and range-check
    for (size_t i = 0; i < parsed->n_axes; i++) {
        Py_ssize_t ax = parsed->axes[i];
        if (ax < 0) ax += (Py_ssize_t)tensor_rank;
        if (ax < 0 || (size_t)ax >= tensor_rank)
            return (PyErr_Format(PyExc_ValueError, "axis %zd out of range for rank %zu", parsed->axes[i], tensor_rank),
                    -1);
        parsed->axes[i] = ax;
    }
    // Insertion sort (n_axes <= 64)
    for (size_t i = 1; i < parsed->n_axes; i++) {
        Py_ssize_t key = parsed->axes[i];
        size_t j = i;
        while (j > 0 && parsed->axes[j - 1] > key) {
            parsed->axes[j] = parsed->axes[j - 1];
            j--;
        }
        parsed->axes[j] = key;
    }
    // Check for duplicates (adjacent after sort)
    for (size_t i = 1; i < parsed->n_axes; i++)
        if (parsed->axes[i] == parsed->axes[i - 1])
            return (PyErr_Format(PyExc_ValueError, "duplicate axis %zd", parsed->axes[i]), -1);
    return 0;
}

/** Parse (axis=None, keepdims=False, out=None) from FASTCALL kwargs. */
static int parse_reduce_kwargs(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, size_t tensor_rank,
                               reduce_args_t *parsed) {
    parsed->n_axes = 0;
    parsed->keepdims = 0;
    parsed->out = NULL;
    parsed->dtype_override = nk_dtype_unknown_k;
    int axis_set = 0;

    // axis is positional-or-keyword (position 0)
    if (nargs >= 1) {
        int rc = parse_axis_value(args[0], parsed);
        if (rc < 0) return -1;
        axis_set = rc;
    }
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "at most 1 positional argument");
        return -1;
    }

    Py_ssize_t const keyword_count = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < keyword_count; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "axis") == 0) {
            int rc = parse_axis_value(value, parsed);
            if (rc < 0) return -1;
            axis_set = rc;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "keepdims") == 0) {
            parsed->keepdims = PyObject_IsTrue(value);
            if (parsed->keepdims < 0) return -1;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "out") == 0) {
            if (value != Py_None) {
                if (!PyObject_TypeCheck(value, &TensorType))
                    return (PyErr_SetString(PyExc_TypeError, "out must be a Tensor"), -1);
                parsed->out = (Tensor *)value;
            }
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (value != Py_None) {
                parsed->dtype_override = py_object_to_nk_dtype(value);
                if (parsed->dtype_override == nk_dtype_unknown_k) return -1;
            }
        }
        else return (PyErr_Format(PyExc_TypeError, "unexpected keyword: %S", name), -1);
    }

    if (!axis_set) return 0;
    return normalize_axes(parsed, tensor_rank);
}

/**
 *  @brief Compute output shape for multi-axis reduction. Returns output rank.
 *  @param[in] axes Sorted array of axes to reduce, length @p n_axes.
 */
static size_t reduce_output_shape(Py_ssize_t const *in_shape, size_t in_rank, Py_ssize_t const *axes, size_t n_axes,
                                  int keepdims, Py_ssize_t *out_shape) {
    size_t j = 0, a = 0;
    for (size_t i = 0; i < in_rank; i++) {
        if (a < n_axes && (size_t)axes[a] == i) {
            a++;
            if (keepdims) out_shape[j++] = 1;
        }
        else { out_shape[j++] = in_shape[i]; }
    }
    return j;
}

/** Validate user-provided out tensor. */
static int validate_reduce_out(Tensor *out, Py_ssize_t const *expected_shape, size_t expected_rank,
                               nk_dtype_t expected_dtype) {
    if (out->dtype != expected_dtype)
        return (PyErr_Format(PyExc_TypeError, "out dtype mismatch: expected '%s'",
                             nk_dtype_to_pybuffer_typestr(expected_dtype)),
                -1);
    if (out->rank != expected_rank)
        return (PyErr_Format(PyExc_ValueError, "out rank mismatch: expected %zu", expected_rank), -1);
    for (size_t i = 0; i < expected_rank; i++)
        if (out->shape[i] != expected_shape[i]) return (PyErr_SetString(PyExc_ValueError, "out shape mismatch"), -1);
    if (!tensor_is_c_contig(out)) return (PyErr_SetString(PyExc_ValueError, "out must be C-contiguous"), -1);
    return 0;
}

/** Callback that processes one 1D slice and writes result to out. */
typedef void (*reduce_slice_fn_t)(TensorView const *slice, nk_scalar_buffer_t *out);

/**
 *  @brief Walk all positions except the reduced axes, calling @p on_slice for each sub-view.
 *  @param[in] axes Sorted array of axes to reduce, length @p n_axes.
 *  @param[in] next_ax Current scan position into the axes array.
 */
static void reduce_along_axes(char const *data, Py_ssize_t const *shape, Py_ssize_t const *strides, size_t rank,
                              size_t dim, Py_ssize_t const *axes, size_t n_axes, size_t next_ax, nk_dtype_t dtype,
                              char *out_data, size_t out_elem_size, reduce_slice_fn_t on_slice, size_t *out_index) {
    if (dim >= rank) {
        // Build sub-view over the reduced axes only
        Py_ssize_t sub_shape[NK_TENSOR_MAX_RANK];
        Py_ssize_t sub_strides[NK_TENSOR_MAX_RANK];
        for (size_t i = 0; i < n_axes; i++) {
            sub_shape[i] = shape[axes[i]];
            sub_strides[i] = strides[axes[i]];
        }
        TensorView slice = {dtype, n_axes, sub_shape, sub_strides, (char *)data};
        nk_scalar_buffer_t element = {0};
        on_slice(&slice, &element);
        memcpy(out_data + (*out_index) * out_elem_size, &element, out_elem_size);
        (*out_index)++;
        return;
    }
    if (next_ax < n_axes && (size_t)axes[next_ax] == dim) {
        // Skip reduced dimension — it will be part of the sub-view
        reduce_along_axes(data, shape, strides, rank, dim + 1, axes, n_axes, next_ax + 1, dtype, out_data,
                          out_elem_size, on_slice, out_index);
    }
    else {
        // Iterate over non-reduced dimension
        for (size_t i = 0; i < (size_t)shape[dim]; i++)
            reduce_along_axes(data + i * strides[dim], shape, strides, rank, dim + 1, axes, n_axes, next_ax, dtype,
                              out_data, out_elem_size, on_slice, out_index);
    }
}

/** Dispatch axis reduction: build output, walk slices, return result. For single-axis reductions on
 *  tensors where the non-axis dims are uniformly strided, uses a pointer-increment fast path
 *  instead of the recursive coordinate walker. */
static PyObject *reduce_axis_dispatch(TensorView const *view, reduce_args_t const *parsed, nk_dtype_t out_dtype,
                                      reduce_slice_fn_t on_slice) {
    // Packed kernels walk whole bytes of the last axis, so every slice must span it.
    if (nk_dimensions_per_value(view->dtype) > 1 && (size_t)parsed->axes[parsed->n_axes - 1] + 1 != view->rank) {
        PyErr_Format(PyExc_NotImplementedError, "packed dtype '%s' can only reduce along axes that include the last",
                     nk_dtype_python_name(view->dtype));
        return NULL;
    }
    Py_ssize_t out_shape[NK_TENSOR_MAX_RANK];
    size_t out_rank = reduce_output_shape(view->shape, view->rank, parsed->axes, parsed->n_axes, parsed->keepdims,
                                          out_shape);
    Tensor *result;
    if (parsed->out) {
        if (validate_reduce_out(parsed->out, out_shape, out_rank, out_dtype) < 0) return NULL;
        result = parsed->out;
    }
    else {
        result = Tensor_new(out_dtype, out_rank, out_shape);
        if (!result) return NULL;
    }

    // Fast path: single-axis reduction with contiguous non-axis dims → pointer-increment loop.
    if (parsed->n_axes == 1 && view->rank >= 2) {
        size_t axis = (size_t)parsed->axes[0];
        Py_ssize_t other_shapes[NK_TENSOR_MAX_RANK], other_strides[NK_TENSOR_MAX_RANK];
        size_t other_count = 0;
        for (size_t i = 0; i < view->rank; i++) {
            if (i != axis) {
                other_shapes[other_count] = view->shape[i];
                other_strides[other_count] = view->strides[i];
                other_count++;
            }
        }
        size_t collapsed_count;
        Py_ssize_t collapsed_stride;
        // The reduced axis includes any packed last axis, so the remaining ones count plain elements.
        size_t tail = uniform_stride_tail_dims(nk_dtype_unknown_k, other_shapes, other_strides, other_count,
                                               &collapsed_count, &collapsed_stride);
        if (tail == other_count) {
            Py_ssize_t lane_shape = view->shape[axis];
            Py_ssize_t lane_stride = view->strides[axis];
            size_t out_elem_size = nk_dtype_bytes_per_value(out_dtype);
            char const *ptr = view->data;
            for (size_t i = 0; i < collapsed_count; i++, ptr += collapsed_stride) {
                TensorView slice = {view->dtype, 1, &lane_shape, &lane_stride, (char *)ptr};
                nk_scalar_buffer_t element = {0};
                on_slice(&slice, &element);
                memcpy(result->data + i * out_elem_size, &element, out_elem_size);
            }
            if (parsed->out) Py_INCREF((PyObject *)parsed->out);
            return (PyObject *)result;
        }
    }

    size_t out_index = 0;
    reduce_along_axes(view->data, view->shape, view->strides, view->rank, 0, parsed->axes, parsed->n_axes, 0,
                      view->dtype, result->data, nk_dtype_bytes_per_value(out_dtype), on_slice, &out_index);
    if (parsed->out) Py_INCREF((PyObject *)parsed->out);
    return (PyObject *)result;
}

static void sum_slice(TensorView const *slice, nk_scalar_buffer_t *out) {
    nk_scalar_buffer_t sumsq_buf;
    nk_dtype_t sum_dtype, sumsq_dtype;
    impl_reduce_moments(slice, out, &sum_dtype, &sumsq_buf, &sumsq_dtype);
}

static void min_slice(TensorView const *slice, nk_scalar_buffer_t *out) {
    nk_scalar_buffer_t max_val;
    nk_dtype_t min_dtype, max_dtype;
    size_t min_idx, max_idx;
    impl_reduce_minmax(slice, out, &min_dtype, &min_idx, &max_val, &max_dtype, &max_idx);
}

static void max_slice(TensorView const *slice, nk_scalar_buffer_t *out) {
    nk_scalar_buffer_t min_val;
    nk_dtype_t min_dtype, max_dtype;
    size_t min_idx, max_idx;
    impl_reduce_minmax(slice, &min_val, &min_dtype, &min_idx, out, &max_dtype, &max_idx);
}

static void argmin_slice(TensorView const *slice, nk_scalar_buffer_t *out) {
    nk_scalar_buffer_t min_val, max_val;
    nk_dtype_t min_dtype, max_dtype;
    size_t min_idx, max_idx;
    impl_reduce_minmax(slice, &min_val, &min_dtype, &min_idx, &max_val, &max_dtype, &max_idx);
    out->i64 = (nk_i64_t)min_idx;
}

static void argmax_slice(TensorView const *slice, nk_scalar_buffer_t *out) {
    nk_scalar_buffer_t min_val, max_val;
    nk_dtype_t min_dtype, max_dtype;
    size_t min_idx, max_idx;
    impl_reduce_minmax(slice, &min_val, &min_dtype, &min_idx, &max_val, &max_dtype, &max_idx);
    out->i64 = (nk_i64_t)max_idx;
}

static void norm_slice(TensorView const *slice, nk_scalar_buffer_t *out) {
    nk_scalar_buffer_t sum_buf, sumsq_buf;
    nk_dtype_t sum_dtype, sumsq_dtype;
    impl_reduce_moments(slice, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype);
    nk_scalar_buffer_to_f64(&sumsq_buf, sumsq_dtype, &out->f64);
    out->f64 = nk_f64_sqrt(out->f64);
}

char const doc_method_sum[] =                                                                        //
    "Return the sum of all elements, or per-slice sums along one or more axes.\n\n"                  //
    "Args:\n"                                                                                        //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                    //
    "Returns:\n"                                                                                     //
    "    Scalar sum when axis is None.\n"                                                            //
    "    Tensor of per-slice sums when axis is given.\n\n"                                           //
    "Signature:\n"                                                                                   //
    "    >>> def sum(self, /, axis=None, *, keepdims=False, out=None): ...";

static PyObject *Tensor_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;
    TensorView view = {tensor->dtype, tensor->rank, tensor->shape, tensor->strides, tensor->data};
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args, nargs, kwnames, view.rank, &parsed) < 0) return NULL;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t sum_buf, sumsq_buf;
        nk_dtype_t sum_dtype, sumsq_dtype;
        if (impl_reduce_moments(&view, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype) < 0)
            return PyErr_Format(PyExc_NotImplementedError, "sum not supported for dtype '%s'",
                                nk_dtype_to_pybuffer_typestr(view.dtype));
        return nk_scalar_buffer_to_py_number(&sum_buf, sum_dtype);
    }
    return reduce_axis_dispatch(&view, &parsed, nk_reduce_moments_sum_dtype(view.dtype), sum_slice);
}

char const doc_method_norm[] =                                                                       //
    "Return the L2 norm, or per-slice norms along one or more axes.\n\n"                             //
    "Args:\n"                                                                                        //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                    //
    "Returns:\n"                                                                                     //
    "    Scalar L2 norm when axis is None.\n"                                                        //
    "    Tensor of per-slice norms when axis is given.\n\n"                                          //
    "Signature:\n"                                                                                   //
    "    >>> def norm(self, /, axis=None, *, keepdims=False, out=None): ...";

static PyObject *Tensor_norm(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;
    TensorView view = {tensor->dtype, tensor->rank, tensor->shape, tensor->strides, tensor->data};
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args, nargs, kwnames, view.rank, &parsed) < 0) return NULL;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t sum_buf, sumsq_buf;
        nk_dtype_t sum_dtype, sumsq_dtype;
        if (impl_reduce_moments(&view, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype) < 0)
            return PyErr_Format(PyExc_NotImplementedError, "norm not supported for dtype '%s'",
                                nk_dtype_to_pybuffer_typestr(view.dtype));
        nk_scalar_buffer_to_f64(&sumsq_buf, sumsq_dtype, &sumsq_buf.f64);
        return PyFloat_FromDouble(nk_f64_sqrt(sumsq_buf.f64));
    }
    return reduce_axis_dispatch(&view, &parsed, nk_f64_k, norm_slice);
}

char const doc_method_min[] =                                                                        //
    "Return the minimum element, or per-slice minimums along one or more axes.\n\n"                  //
    "Args:\n"                                                                                        //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                    //
    "Returns:\n"                                                                                     //
    "    Scalar minimum when axis is None; None if all elements are NaN.\n"                          //
    "    Tensor of per-slice minimums when axis is given.\n\n"                                       //
    "Signature:\n"                                                                                   //
    "    >>> def min(self, /, axis=None, *, keepdims=False, out=None): ...";

static PyObject *Tensor_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;
    TensorView view = {tensor->dtype, tensor->rank, tensor->shape, tensor->strides, tensor->data};
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args, nargs, kwnames, view.rank, &parsed) < 0) return NULL;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            return PyErr_Format(PyExc_NotImplementedError, "min not supported for dtype '%s'",
                                nk_dtype_to_pybuffer_typestr(view.dtype));
        if (min_idx == NK_SIZE_MAX) Py_RETURN_NONE;
        return nk_scalar_buffer_to_py_number(&min_buf, min_dtype);
    }
    return reduce_axis_dispatch(&view, &parsed, nk_reduce_minmax_value_dtype(view.dtype), min_slice);
}

char const doc_method_max[] =                                                                        //
    "Return the maximum element, or per-slice maximums along one or more axes.\n\n"                  //
    "Args:\n"                                                                                        //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                    //
    "Returns:\n"                                                                                     //
    "    Scalar maximum when axis is None; None if all elements are NaN.\n"                          //
    "    Tensor of per-slice maximums when axis is given.\n\n"                                       //
    "Signature:\n"                                                                                   //
    "    >>> def max(self, /, axis=None, *, keepdims=False, out=None): ...";

static PyObject *Tensor_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;
    TensorView view = {tensor->dtype, tensor->rank, tensor->shape, tensor->strides, tensor->data};
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args, nargs, kwnames, view.rank, &parsed) < 0) return NULL;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            return PyErr_Format(PyExc_NotImplementedError, "max not supported for dtype '%s'",
                                nk_dtype_to_pybuffer_typestr(view.dtype));
        if (max_idx == NK_SIZE_MAX) Py_RETURN_NONE;
        return nk_scalar_buffer_to_py_number(&max_buf, max_dtype);
    }
    return reduce_axis_dispatch(&view, &parsed, nk_reduce_minmax_value_dtype(view.dtype), max_slice);
}

char const doc_method_argmin[] =                                                                        //
    "Return the index of the minimum element, or per-slice indices along an axis.\n\n"                  //
    "Args:\n"                                                                                           //
    "    axis (int, optional): Axis to reduce along, defaulting to None, which reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"      //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                       //
    "Returns:\n"                                                                                        //
    "    Integer index when axis is None.\n"                                                            //
    "    Tensor of per-slice indices when axis is given.\n\n"                                           //
    "Signature:\n"                                                                                      //
    "    >>> def argmin(self, /, axis=None, *, keepdims=False, out=None): ...";

static PyObject *Tensor_argmin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;
    TensorView view = {tensor->dtype, tensor->rank, tensor->shape, tensor->strides, tensor->data};
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args, nargs, kwnames, view.rank, &parsed) < 0) return NULL;
    if (parsed.n_axes > 1) return (PyErr_SetString(PyExc_TypeError, "argmin does not support tuple axis"), NULL);
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            return PyErr_Format(PyExc_NotImplementedError, "argmin not supported for dtype '%s'",
                                nk_dtype_to_pybuffer_typestr(view.dtype));
        if (min_idx == NK_SIZE_MAX) Py_RETURN_NONE;
        return PyLong_FromSsize_t((Py_ssize_t)min_idx);
    }
    return reduce_axis_dispatch(&view, &parsed, nk_i64_k, argmin_slice);
}

char const doc_method_argmax[] =                                                                        //
    "Return the index of the maximum element, or per-slice indices along an axis.\n\n"                  //
    "Args:\n"                                                                                           //
    "    axis (int, optional): Axis to reduce along, defaulting to None, which reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"      //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                       //
    "Returns:\n"                                                                                        //
    "    Integer index when axis is None.\n"                                                            //
    "    Tensor of per-slice indices when axis is given.\n\n"                                           //
    "Signature:\n"                                                                                      //
    "    >>> def argmax(self, /, axis=None, *, keepdims=False, out=None): ...";

static PyObject *Tensor_argmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;
    TensorView view = {tensor->dtype, tensor->rank, tensor->shape, tensor->strides, tensor->data};
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args, nargs, kwnames, view.rank, &parsed) < 0) return NULL;
    if (parsed.n_axes > 1) return (PyErr_SetString(PyExc_TypeError, "argmax does not support tuple axis"), NULL);
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            return PyErr_Format(PyExc_NotImplementedError, "argmax not supported for dtype '%s'",
                                nk_dtype_to_pybuffer_typestr(view.dtype));
        if (max_idx == NK_SIZE_MAX) Py_RETURN_NONE;
        return PyLong_FromSsize_t((Py_ssize_t)max_idx);
    }
    return reduce_axis_dispatch(&view, &parsed, nk_i64_k, argmax_slice);
}

/**
 *  @brief Build a @c ScaledTensor by quantizing a dense tensor into @p target_dtype.
 *
 *  The dense source is linearized into a contiguous f32 staging buffer — quantization is on the
 *  last axis, the last dim must be a multiple of the format's block size. The element and scale
 *  buffers are sized via @c nk_block_scaled_elements_size and @c nk_block_scaled_scales_size,
 *  allocated as child Tensors, and filled by a single @c nk_cast_block_scaled call over the
 *  flattened element count — blocks never span rows because the last dim is block-aligned.
 *
 *  For NVFP4, `tensor_scale_dtype == f32`, the per-tensor scale is auto-derived by handing the
 *  kernel a zero-initialised @c to_tensor_scale and reading the result back into `.tensor_scale`.
 *  The MX family carries no per-tensor scale, `.tensor_scale is None`.
 */
static PyObject *Tensor_encode_block_scaled(Tensor *tensor, nk_dtype_t target_dtype) {
    nk_block_scaled_format_t to_format = nk_block_scaled_format_of_dtype(target_dtype);
    size_t block_size = to_format.block_size;

    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_ValueError, "cannot block-quantize a 0-dimensional tensor");
        return NULL;
    }

    Py_ssize_t last_dim = tensor->shape[tensor->rank - 1];
    if (block_size == 0 || (size_t)last_dim % block_size != 0) {
        PyErr_Format(PyExc_ValueError, "last dimension %zd must be a multiple of block size %zu for '%s'", last_dim,
                     block_size, nk_dtype_python_name(target_dtype));
        return NULL;
    }

    size_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= (size_t)tensor->shape[i];

    // Linearize the (possibly strided) source into a contiguous f32 staging buffer.
    int staging_free = 0;
    char *staging = ensure_contiguous_buffer(tensor->data, tensor->dtype, nk_f32_k, tensor->rank, tensor->shape,
                                             tensor->strides, total, &staging_free);
    if (!staging) return NULL;

    // Child tensor shapes: elements keep the dense shape, scales count blocks along the last axis.
    Py_ssize_t scales_shape[NK_TENSOR_MAX_RANK];
    for (size_t i = 0; i + 1 < tensor->rank; i++) scales_shape[i] = tensor->shape[i];
    scales_shape[tensor->rank - 1] = (Py_ssize_t)nk_block_scaled_scales_size((nk_size_t)last_dim, to_format);

    Tensor *elements = Tensor_new(to_format.element_dtype, tensor->rank, tensor->shape);
    Tensor *block_scales = Tensor_new(to_format.scale_dtype, tensor->rank, scales_shape);
    ScaledTensor *result = elements && block_scales ? PyObject_New(ScaledTensor, &ScaledTensorType) : NULL;
    if (!result) {
        Py_XDECREF(elements);
        Py_XDECREF(block_scales);
        if (staging_free) PyMem_Free(staging);
        return NULL;
    }

    int has_tensor_scale = (to_format.tensor_scale_dtype == nk_f32_k);
    nk_scalar_buffer_t to_tensor_scale;
    memset(&to_tensor_scale, 0, sizeof(to_tensor_scale));
    to_tensor_scale.f32 = 0.0f; // zero → kernel derives the per-tensor scale (NVFP4)

    nk_block_scaled_format_t from_format = nk_plain(nk_f32_k);
    PyThreadState *gil = PyEval_SaveThread();
    nk_cast_block_scaled(staging, NULL, NULL, &from_format,                                              //
                         elements->data, block_scales->data, has_tensor_scale ? &to_tensor_scale : NULL, //
                         &to_format, (nk_size_t)total);
    PyEval_RestoreThread(gil);

    if (staging_free) PyMem_Free(staging);

    result->elements = elements;
    result->block_scales = block_scales;
    result->dtype = target_dtype;
    result->block_size = block_size;
    result->tensor_scale = has_tensor_scale ? to_tensor_scale.f32 : 1.0f;
    result->has_tensor_scale = has_tensor_scale;
    return (PyObject *)result;
}

char const doc_method_astype[] =                                                             //
    "Cast the tensor to a different dtype.\n\n"                                              //
    "Args:\n"                                                                                //
    "    dtype (str): Target data type. Standard dtypes, such as 'float32' or 'bf16',\n"     //
    "        return a Tensor; block-scaled dtypes ('nvfp4', 'mxfp4', 'mxfp8_e4m3',\n"        //
    "        ...) quantize along the last axis and return a ScaledTensor.\n"                 //
    "    out (Tensor, optional): Pre-allocated destination. When given, conversion writes\n" //
    "        into it with no allocation; must match the target dtype and source shape.\n"    //
    "        Unsupported for block-scaled targets; returns `out`.\n\n"                       //
    "Returns:\n"                                                                             //
    "    Tensor or ScaledTensor: Converted result, or `out` when provided.\n\n"              //
    "Signature:\n"                                                                           //
    "    >>> def astype(self, dtype, /, *, out=None): ...";

PyObject *Tensor_astype(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    Tensor *tensor = (Tensor *)self;

    PyObject *dtype_arg = NULL;
    PyObject *out_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "astype(dtype, /, *, out=None) requires exactly one positional argument");
        return NULL;
    }
    dtype_arg = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) out_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "astype() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }

    // Parse target dtype from argument
    nk_dtype_t target_dtype = py_object_to_nk_dtype(dtype_arg);
    if (target_dtype == nk_dtype_unknown_k) return NULL;

    // Cast into a caller-provided buffer — the allocation-free path.
    if (out_obj && out_obj != Py_None) {
        if (nk_dtype_is_block_scaled(target_dtype)) {
            PyErr_SetString(PyExc_TypeError, "astype(out=...) does not support block-scaled target dtypes");
            return NULL;
        }
        char *out_data = validate_out_buffer(out_obj, target_dtype, tensor->rank, tensor->shape);
        if (!out_data) return NULL;
        Py_ssize_t out_total = 1;
        for (size_t i = 0; i < tensor->rank; i++) out_total *= tensor->shape[i];
        linearize_cast_into(tensor->data, tensor->dtype, out_data, target_dtype, tensor->rank, tensor->shape,
                            tensor->strides, (size_t)out_total);
        Py_INCREF(out_obj);
        return out_obj;
    }

    // Block-scaled target -> quantize into a ScaledTensor.
    if (nk_dtype_is_block_scaled(target_dtype)) return Tensor_encode_block_scaled(tensor, target_dtype);

    // Same dtype -> return copy
    if (target_dtype == tensor->dtype) return Tensor_copy(self, NULL, 0, NULL);

    // Compute total elements
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];

    // Allocate result tensor
    Tensor *result = Tensor_new(target_dtype, tensor->rank, tensor->shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, target_dtype, tensor->rank, tensor->shape,
                        tensor->strides, (size_t)total);
    return (PyObject *)result;
}

char const doc_astype[] =                                                                  //
    "Cast a buffer-protocol input to a NumKong dtype.\n\n"                                 //
    "Args:\n"                                                                              //
    "    a: Input buffer with arbitrary rank and strides.\n"                               //
    "    dtype: Target NumKong dtype name.\n"                                              //
    "    out: Optional writable C-contiguous output buffer with the exact input shape\n"   //
    "        and requested dtype. It must not overlap `a`.\n\n"                            //
    "Returns:\n"                                                                           //
    "    Tensor: A new tensor with the input shape and requested dtype, or `out` itself\n" //
    "        when provided.\n\n"                                                           //
    "Notes:\n"                                                                             //
    "    Unlike `Tensor(a).astype(dtype)`, the input is never staged through a Tensor.\n"  //
    "    Narrower floating formats round ties to even. Float-to-integer casts round\n"     //
    "    ties to even, saturate overflow and infinity, and map NaN to zero.\n\n"           //
    "Signature:\n"                                                                         //
    "    >>> def astype(a, dtype, /, *, out=None) -> Tensor: ...";

PyObject *api_astype(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "astype(a, dtype, /, *, out=None) requires exactly two positional arguments");
        return NULL;
    }

    PyObject *out_obj = NULL;
    Py_ssize_t const keyword_count = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < keyword_count; ++i) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) out_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "astype() got an unexpected keyword argument '%S'", name);
            return NULL;
        }
    }
    if (out_obj == Py_None) out_obj = NULL;

    PyObject *return_obj = NULL;
    Py_buffer input_buffer, out_buffer;
    nk_buffer_backing_t input_backing, out_backing;
    memset(&input_buffer, 0, sizeof(input_buffer));
    memset(&out_buffer, 0, sizeof(out_buffer));

    if (!nk_get_buffer(args[0], &input_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &input_backing)) return NULL;
    if (input_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", input_buffer.ndim,
                     NK_TENSOR_MAX_RANK);
        goto cleanup;
    }

    nk_dtype_t const input_dtype = resolve_nk_dtype_in_py_buffer(&input_buffer);
    if (input_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported input dtype '%s'", input_buffer.format);
        goto cleanup;
    }
    nk_dtype_t const output_dtype = py_object_to_nk_dtype(args[1]);
    if (output_dtype == nk_dtype_unknown_k) goto cleanup;

    char *result_data = NULL;
    if (!out_obj) {
        if (!nk_buffer_logical_shape(&input_buffer, input_dtype, &input_backing)) goto cleanup;
        Tensor *result = Tensor_new(output_dtype, (size_t)input_buffer.ndim, input_buffer.shape);
        if (!result) goto cleanup;
        return_obj = (PyObject *)result;
        result_data = result->data;
    }
    else {
        if (!nk_get_buffer(out_obj, &out_buffer, PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE, &out_backing))
            goto cleanup;
        result_data = validate_out_py_buffer(&out_buffer, &input_buffer, output_dtype);
        if (!result_data) goto cleanup;
        if (!nk_buffer_logical_shape(&input_buffer, input_dtype, &input_backing) ||
            !nk_buffer_logical_shape(&out_buffer, output_dtype, &out_backing) ||
            !buffers_shapes_match(&input_buffer, &out_buffer))
            goto cleanup;
        return_obj = out_obj;
        Py_INCREF(out_obj);
    }

    size_t total_elements = 1;
    for (int dim = 0; dim < input_buffer.ndim; ++dim) total_elements *= (size_t)input_buffer.shape[dim];

    PyThreadState *gil = PyEval_SaveThread();
    linearize_cast_into(input_buffer.buf, input_dtype, result_data, output_dtype, (size_t)input_buffer.ndim,
                        input_buffer.shape, input_buffer.strides, total_elements);
    PyEval_RestoreThread(gil);

cleanup:
    PyBuffer_Release(&input_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

char const doc_method___array__[] =                                                         //
    "Convert to a NumPy array.\n\n"                                                         //
    "Args:\n"                                                                               //
    "    dtype (str, optional): Desired NumPy dtype, defaulting to None, which preserves\n" //
    "        the original.\n"                                                               //
    "    copy (bool, optional): If True, force a copy. Default None.\n\n"                   //
    "Returns:\n"                                                                            //
    "    numpy.ndarray: Array sharing data when possible.\n\n"                              //
    "Signature:\n"                                                                          //
    "    >>> def __array__(self, /, dtype=None, *, copy=None): ...";

static PyObject *Tensor___array__(PyObject *self_obj, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *dtype_arg = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs > 1 || nargs + nkw > 2) {
        PyErr_SetString(PyExc_TypeError, "__array__(dtype=None, copy=None)");
        return NULL;
    }
    if (nargs >= 1) dtype_arg = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) dtype_arg = value;
        else if (PyUnicode_CompareWithASCIIString(name, "copy") == 0) { /* ignored */ }
        else {
            PyErr_Format(PyExc_TypeError, "__array__() unexpected keyword: %S", name);
            return NULL;
        }
    }

    Tensor *self = (Tensor *)self_obj;
    nk_dtype_conversion_info_t const *info = nk_dtype_conversion_info(self->dtype);
    if (!info) {
        PyErr_SetString(PyExc_TypeError, "Unknown dtype");
        return NULL;
    }
    // Reject exotic dtypes that NumPy can't represent natively
    char const *fmt = info->pybuffer_typestr;
    if (same_string(fmt, "e2m3") || same_string(fmt, "e3m2") ||       //
        same_string(fmt, "e4m3") || same_string(fmt, "e5m2") ||       //
        same_string(fmt, "bf16") || same_string(fmt, "bcomplex32") || //
        same_string(fmt, "Ze") || same_string(fmt, "i4") ||           //
        same_string(fmt, "u4") || same_string(fmt, "?")) {
        PyErr_Format(PyExc_TypeError,
                     "Cannot convert NumKong tensor of dtype '%s' to NumPy array. Use .astype('float32') first.",
                     info->name);
        return NULL;
    }

    // Cache numpy.asarray to avoid repeated import overhead
    static PyObject *cached_asarray = NULL;
    if (!cached_asarray) {
        PyObject *numpy = PyImport_ImportModule("numpy");
        if (!numpy) return NULL;
        cached_asarray = PyObject_GetAttrString(numpy, "asarray");
        Py_DECREF(numpy);
        if (!cached_asarray) return NULL;
    }

    PyObject *result;
    if (dtype_arg && dtype_arg != Py_None)
        result = PyObject_CallFunctionObjArgs(cached_asarray, self_obj, dtype_arg, NULL);
    else result = PyObject_CallOneArg(cached_asarray, self_obj);
    return result;
}

static PyMethodDef Tensor_methods[] = {
    {"copy", (PyCFunction)Tensor_copy, METH_FASTCALL | METH_KEYWORDS, doc_method_copy},
    {"reshape", (PyCFunction)Tensor_reshape, METH_FASTCALL, doc_method_reshape},
    {"moments", Tensor_moments, METH_NOARGS, doc_method_moments},
    {"minmax", Tensor_minmax, METH_NOARGS, doc_method_minmax},
    {"sum", (PyCFunction)Tensor_sum, METH_FASTCALL | METH_KEYWORDS, doc_method_sum},
    {"norm", (PyCFunction)Tensor_norm, METH_FASTCALL | METH_KEYWORDS, doc_method_norm},
    {"min", (PyCFunction)Tensor_min, METH_FASTCALL | METH_KEYWORDS, doc_method_min},
    {"max", (PyCFunction)Tensor_max, METH_FASTCALL | METH_KEYWORDS, doc_method_max},
    {"argmin", (PyCFunction)Tensor_argmin, METH_FASTCALL | METH_KEYWORDS, doc_method_argmin},
    {"argmax", (PyCFunction)Tensor_argmax, METH_FASTCALL | METH_KEYWORDS, doc_method_argmax},
    {"astype", (PyCFunction)Tensor_astype, METH_FASTCALL | METH_KEYWORDS, doc_method_astype},
    {"flatten", (PyCFunction)Tensor_flatten, METH_FASTCALL | METH_KEYWORDS, doc_method_flatten},
    {"squeeze", (PyCFunction)Tensor_squeeze, METH_FASTCALL, doc_method_squeeze},
    {"resize", (PyCFunction)Tensor_resize, METH_FASTCALL, doc_method_resize},
    {"reserve", (PyCFunction)Tensor_reserve, METH_FASTCALL, doc_method_reserve},
    {"clear", Tensor_clear, METH_NOARGS, doc_method_clear},
    {"__array__", (PyCFunction)Tensor___array__, METH_FASTCALL | METH_KEYWORDS, doc_method___array__},
    {"__dlpack__", (PyCFunction)Tensor_dlpack, METH_VARARGS | METH_KEYWORDS, doc_dlpack},
    {"__dlpack_device__", (PyCFunction)Tensor_dlpack_device, METH_NOARGS, doc_dlpack_device},
    {NULL, NULL, 0, NULL},
};

static int Tensor_getbuffer(PyObject *export_from, Py_buffer *view, int flags) {
    Tensor *tensor = (Tensor *)export_from;
    size_t const item_size = nk_dtype_bytes_per_value(tensor->dtype);

    int c_contig = tensor_is_c_contig(tensor);
    int f_contig = tensor_is_f_contig(tensor);

    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !c_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not C-contiguous");
        view->obj = NULL;
        return -1;
    }
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !f_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not Fortran-contiguous");
        view->obj = NULL;
        return -1;
    }
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS && !c_contig && !f_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not contiguous");
        view->obj = NULL;
        return -1;
    }

    // A buffer addresses whole bytes, so a packed dtype exports its storage extents.
    Py_ssize_t *storage_shape = NULL;
    if (nk_dimensions_per_value(tensor->dtype) > 1) {
        storage_shape = PyMem_Malloc(tensor->rank * sizeof(Py_ssize_t));
        if (!storage_shape) {
            PyErr_NoMemory();
            view->obj = NULL;
            return -1;
        }
        for (size_t i = 0; i < tensor->rank; i++)
            storage_shape[i] = storage_extent(tensor->dtype, tensor->rank, tensor->shape, i);
    }
    Py_ssize_t const *exported_shape = storage_shape ? storage_shape : tensor->shape;

    size_t total_items = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_items *= (size_t)exported_shape[i];

    view->buf = tensor->data;
    view->obj = (PyObject *)tensor;
    view->len = item_size * total_items;
    view->readonly = 0;
    view->itemsize = (Py_ssize_t)item_size;

    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        // Exotic types that numpy can't represent via PEP 3118: emit a valid
        // unsigned-integer format whose item size matches nk_dtype_bytes_per_value.
        // Internal callers needing the real dtype should check isinstance(obj, Tensor)
        // and read .dtype directly.
        switch (tensor->dtype) {
        case nk_bf16_k: view->format = "H"; break;  // 2 bytes → uint16
        case nk_e4m3_k: view->format = "B"; break;  // 1 byte  → uint8
        case nk_e5m2_k: view->format = "B"; break;  // 1 byte  → uint8
        case nk_e2m3_k: view->format = "B"; break;  // 1 byte  → uint8
        case nk_e3m2_k: view->format = "B"; break;  // 1 byte  → uint8
        case nk_i4_k: view->format = "B"; break;    // 1 byte  → uint8
        case nk_u4_k: view->format = "B"; break;    // 1 byte  → uint8
        case nk_u1_k: view->format = "B"; break;    // 1 byte  → uint8
        case nk_bf16c_k: view->format = "I"; break; // 4 bytes → uint32
        case nk_f16c_k: view->format = "I"; break;  // 4 bytes → uint32
        default: view->format = (char *)nk_dtype_to_pybuffer_typestr(tensor->dtype); break;
        }
    }
    else view->format = NULL;

    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = tensor->rank;
        view->shape = tensor->rank > 0 ? (Py_ssize_t *)exported_shape : NULL;
    }
    else {
        view->ndim = 0;
        view->shape = NULL;
    }

    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) view->strides = tensor->rank > 0 ? &tensor->strides[0] : NULL;
    else view->strides = NULL;

    view->suboffsets = NULL;
    view->internal = storage_shape;

    Py_INCREF(tensor);
    tensor->exports++; // block resize/reserve while this buffer is live (see Tensor_resize)
    return 0;
}

static void Tensor_releasebuffer(PyObject *export_from, Py_buffer *view) {
    PyMem_Free(view->internal);
    ((Tensor *)export_from)->exports--;
}

static PyBufferProcs Tensor_as_buffer = {
    .bf_getbuffer = Tensor_getbuffer,
    .bf_releasebuffer = Tensor_releasebuffer,
};

static Py_ssize_t Tensor_length(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of 0-dimensional tensor");
        return -1;
    }
    return tensor->shape[0];
}

static PySequenceMethods Tensor_as_sequence = {
    .sq_length = Tensor_length,
};

static int parse_slice(PyObject *slice, Py_ssize_t dim_size, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
                       Py_ssize_t *slice_len) {
    if (PySlice_Unpack(slice, start, stop, step) < 0) return -1;
    *slice_len = PySlice_AdjustIndices(dim_size, start, stop, *step);
    return 0;
}

/** Whether @p dimension of @p tensor packs several logical dimensions per storage value. */
static int tensor_axis_is_packed(Tensor const *tensor, size_t dimension) {
    return dimension + 1 == tensor->rank && nk_dimensions_per_value(tensor->dtype) > 1;
}

/**
 *  @brief Byte offset and stride of a slice along @p dimension.
 *
 *  A packed last axis addresses whole storage values only, so it needs step 1 and bounds that are
 *  multiples of the values per byte. Returns 0 with a Python error otherwise.
 */
static int tensor_axis_slice(Tensor const *tensor, size_t dimension, Py_ssize_t start, Py_ssize_t step,
                             Py_ssize_t slice_len, Py_ssize_t *byte_offset, Py_ssize_t *stride) {
    if (!tensor_axis_is_packed(tensor, dimension)) {
        *byte_offset = start * tensor->strides[dimension];
        *stride = tensor->strides[dimension] * step;
        return 1;
    }
    Py_ssize_t const dimensions_per_value = (Py_ssize_t)nk_dimensions_per_value(tensor->dtype);
    if ((step != 1 && slice_len > 1) || start % dimensions_per_value != 0 || slice_len % dimensions_per_value != 0) {
        PyErr_Format(PyExc_IndexError, "packed dtype '%s' slices its last axis in whole bytes of %zd with step 1",
                     nk_dtype_python_name(tensor->dtype), dimensions_per_value);
        return 0;
    }
    *byte_offset = (start / dimensions_per_value) * tensor->strides[dimension];
    *stride = tensor->strides[dimension];
    return 1;
}

/** Reads the element at @p byte_offset, position @p last_index along the last logical axis. */
static PyObject *tensor_read_element(Tensor *tensor, size_t byte_offset, Py_ssize_t last_index) {
    nk_size_t const dimensions_per_value = nk_dimensions_per_value(tensor->dtype);
    if (dimensions_per_value <= 1) return tensor_read_scalar(tensor, byte_offset);
    return tensor_read_packed_scalar(tensor, byte_offset, (size_t)last_index % dimensions_per_value);
}

/** Byte offset of logical position @p index along @p dimension. */
static Py_ssize_t tensor_axis_offset(Tensor const *tensor, size_t dimension, Py_ssize_t index) {
    if (!tensor_axis_is_packed(tensor, dimension)) return index * tensor->strides[dimension];
    return (index / (Py_ssize_t)nk_dimensions_per_value(tensor->dtype)) * tensor->strides[dimension];
}

static PyObject *Tensor_subscript(PyObject *self, PyObject *key) {
    Tensor *tensor = (Tensor *)self;

    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_IndexError, "0-dimensional tensor cannot be indexed");
        return NULL;
    }

    // Single slice
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slice_len;
        if (parse_slice(key, tensor->shape[0], &start, &stop, &step, &slice_len) < 0) return NULL;

        Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
        Py_ssize_t byte_offset;

        new_shape[0] = slice_len;
        if (!tensor_axis_slice(tensor, 0, start, step, slice_len, &byte_offset, &new_strides[0])) return NULL;

        for (size_t i = 1; i < tensor->rank; i++) {
            new_shape[i] = tensor->shape[i];
            new_strides[i] = tensor->strides[i];
        }

        char *view_data = tensor->data + byte_offset;
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;

        return (PyObject *)Tensor_view(root_parent, view_data, tensor->dtype, tensor->rank, new_shape, new_strides);
    }

    // Single integer
    if (PyLong_Check(key)) {
        Py_ssize_t idx = PyLong_AsSsize_t(key);
        if (idx == -1 && PyErr_Occurred()) return NULL;

        if (idx < 0) idx += tensor->shape[0];
        if (idx < 0 || idx >= tensor->shape[0]) {
            PyErr_SetString(PyExc_IndexError, "index out of bounds");
            return NULL;
        }

        if (tensor->rank == 1) return tensor_read_element(tensor, (size_t)tensor_axis_offset(tensor, 0, idx), idx);

        // Return a view with reduced rank
        char *view_data = tensor->data + idx * tensor->strides[0];
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;

        return (PyObject *)Tensor_view(root_parent, view_data, tensor->dtype, tensor->rank - 1, tensor->shape + 1,
                                       tensor->strides + 1);
    }

    // Tuple of indices/slices
    if (PyTuple_Check(key)) {
        Py_ssize_t ntuple = PyTuple_Size(key);
        if ((size_t)ntuple > tensor->rank) {
            PyErr_SetString(PyExc_IndexError, "too many indices for tensor");
            return NULL;
        }

        // Build the result view by processing each index element against its corresponding dimension
        Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
        char *view_data = tensor->data;
        size_t new_rank = 0;
        Py_ssize_t last_index = -1;

        // Track which source dimension each tuple element addresses
        size_t src_dim = 0;
        for (Py_ssize_t i = 0; i < ntuple; i++, src_dim++) {
            PyObject *idx = PyTuple_GetItem(key, i);

            if (PyLong_Check(idx)) {
                // Integer index: select one position, reduce rank
                Py_ssize_t index = PyLong_AsSsize_t(idx);
                if (index == -1 && PyErr_Occurred()) return NULL;
                if (index < 0) index += tensor->shape[src_dim];
                if (index < 0 || index >= tensor->shape[src_dim]) {
                    PyErr_SetString(PyExc_IndexError, "index out of bounds");
                    return NULL;
                }
                view_data += tensor_axis_offset(tensor, src_dim, index);
                if (tensor_axis_is_packed(tensor, src_dim)) last_index = index;
                // Dimension is consumed, rank reduced
            }
            else if (PySlice_Check(idx)) {
                // Slice: select a range, keep dimension
                Py_ssize_t start, stop, step, slice_len, byte_offset;
                if (parse_slice(idx, tensor->shape[src_dim], &start, &stop, &step, &slice_len) < 0) return NULL;
                if (!tensor_axis_slice(tensor, src_dim, start, step, slice_len, &byte_offset, &new_strides[new_rank]))
                    return NULL;
                view_data += byte_offset;
                new_shape[new_rank] = slice_len;
                new_rank++;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
                return NULL;
            }
        }

        // Copy remaining dimensions that weren't indexed
        for (size_t d = src_dim; d < tensor->rank; d++) {
            new_shape[new_rank] = tensor->shape[d];
            new_strides[new_rank] = tensor->strides[d];
            new_rank++;
        }

        // If all dims consumed by integer indices, return scalar
        if (new_rank == 0) return tensor_read_element(tensor, (size_t)(view_data - tensor->data), last_index);
        if (last_index >= 0) {
            PyErr_Format(PyExc_IndexError, "packed dtype '%s' cannot drop its last axis while keeping others",
                         nk_dtype_python_name(tensor->dtype));
            return NULL;
        }

        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
        return (PyObject *)Tensor_view(root_parent, view_data, tensor->dtype, new_rank, new_shape, new_strides);
    }

    PyErr_SetString(PyExc_TypeError, "indices must be integers, slices, or tuples");
    return NULL;
}

static PyMappingMethods Tensor_as_mapping = {
    .mp_length = Tensor_length,
    .mp_subscript = Tensor_subscript,
};

static PyObject *Tensor_repr(PyObject *self) {
    Tensor *tensor = (Tensor *)self;

    PyObject *shape_str = Tensor_get_shape(self, NULL);
    if (!shape_str) return NULL;

    PyObject *repr = PyUnicode_FromFormat("Tensor(shape=%R, dtype='%s')", shape_str,
                                          nk_dtype_python_name(tensor->dtype));
    Py_DECREF(shape_str);
    return repr;
}

static PyObject *Tensor_str(PyObject *self) {
    // Try numpy formatting for standard dtypes
    PyObject *np_arr = Tensor___array__(self, NULL, 0, NULL);
    if (np_arr) {
        PyObject *s = PyObject_Str(np_arr);
        Py_DECREF(np_arr);
        if (s) return s;
        PyErr_Clear();
    }
    PyErr_Clear();
    return Tensor_repr(self);
}

static PyObject *Tensor_richcompare(PyObject *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE) Py_RETURN_NOTIMPLEMENTED;
    if (!PyObject_TypeCheck(other, &TensorType)) Py_RETURN_NOTIMPLEMENTED;
    Tensor *a = (Tensor *)self, *b = (Tensor *)other;

    // Check dtype and rank match
    int equal = 1;
    if (a->rank != b->rank || a->dtype != b->dtype) { equal = 0; }
    else {
        for (size_t i = 0; i < a->rank; i++) {
            if (a->shape[i] != b->shape[i]) {
                equal = 0;
                break;
            }
        }
    }

    if (equal) {
        size_t total = 1;
        for (size_t i = 0; i < a->rank; i++) total *= (size_t)a->shape[i];
        size_t item_size = nk_dtype_bytes_per_value(a->dtype);
        int a_free = 0, b_free = 0;
        char *a_buffer = ensure_contiguous_buffer(a->data, a->dtype, a->dtype, a->rank, a->shape, a->strides, total,
                                                  &a_free);
        char *b_buffer = ensure_contiguous_buffer(b->data, b->dtype, b->dtype, b->rank, b->shape, b->strides, total,
                                                  &b_free);
        if (!a_buffer || !b_buffer) {
            if (a_free) PyMem_Free(a_buffer);
            if (b_free) PyMem_Free(b_buffer);
            return NULL;
        }
        equal = memcmp(a_buffer, b_buffer, dimensions_to_values(a->dtype, total) * item_size) == 0;
        if (a_free) PyMem_Free(a_buffer);
        if (b_free) PyMem_Free(b_buffer);
    }

    if (op == Py_EQ) { return PyBool_FromLong(equal); }
    else { return PyBool_FromLong(!equal); }
}

static PyObject *TensorIter_next(PyObject *self) {
    TensorIter *iter = (TensorIter *)self;
    Tensor *array = iter->array;

    if (iter->index >= array->shape[0]) {
        return NULL; // StopIteration
    }

    PyObject *item;
    if (array->rank == 1)
        item = tensor_read_element(array, (size_t)tensor_axis_offset(array, 0, iter->index), iter->index);
    else {
        char *view_data = array->data + iter->index * array->strides[0];
        Tensor *root_parent = array->parent ? (Tensor *)array->parent : array;
        item = (PyObject *)Tensor_view(root_parent, view_data, array->dtype, array->rank - 1, array->shape + 1,
                                       array->strides + 1);
    }

    iter->index++;
    return item;
}

static void TensorIter_dealloc(PyObject *self) {
    TensorIter *iter = (TensorIter *)self;
    Py_XDECREF(iter->array);
    Py_TYPE(self)->tp_free(self);
}

PyTypeObject TensorIterType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.TensorIter",
    .tp_basicsize = sizeof(TensorIter),
    .tp_dealloc = TensorIter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = TensorIter_next,
};

static PyObject *Tensor_iter(PyObject *self) {
    Tensor *array = (Tensor *)self;
    if (array->rank == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot iterate over 0-dimensional tensor");
        return NULL;
    }

    TensorIter *iter = PyObject_New(TensorIter, &TensorIterType);
    if (!iter) return NULL;

    iter->array = array;
    Py_INCREF(array);
    iter->index = 0;

    return (PyObject *)iter;
}

static PyObject *Tensor_tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    nk_unused_(type);
    static char const *kwlist[] = {"", "dtype", NULL};
    PyObject *source = NULL;
    PyObject *dtype_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", (char **)kwlist, &source, &dtype_obj)) return NULL;

    Py_buffer buf;
    nk_buffer_backing_t backing;
    if (!nk_get_buffer(source, &buf, PyBUF_FULL_RO, &backing)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError,
                            "Tensor() requires an object supporting the buffer protocol or __array_interface__");
        return NULL;
    }

    nk_dtype_t dtype;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) {
            PyBuffer_Release(&buf);
            return NULL;
        }
        if ((Py_ssize_t)nk_dtype_bytes_per_value(dtype) != buf.itemsize) {
            PyBuffer_Release(&buf);
            PyErr_Format(PyExc_ValueError, "dtype has itemsize %zu but buffer has itemsize %zd",
                         nk_dtype_bytes_per_value(dtype), buf.itemsize);
            return NULL;
        }
    }
    else {
        dtype = resolve_nk_dtype_in_py_buffer(&buf);
        if (dtype == nk_dtype_unknown_k) {
            char const *fmt = buf.format ? buf.format : "(null)";
            PyBuffer_Release(&buf);
            PyErr_Format(PyExc_TypeError, "Cannot determine dtype from buffer format '%s'", fmt);
            return NULL;
        }
    }

    // Whole-byte buffers carry storage extents; a packed dtype re-expresses them as logical dimensions.
    int const is_contiguous = PyBuffer_IsContiguous(&buf, 'C');
    if (!nk_buffer_logical_shape(&buf, dtype, &backing)) {
        PyBuffer_Release(&buf);
        return NULL;
    }
    Tensor *tensor = Tensor_new(dtype, (size_t)buf.ndim, buf.shape);
    if (!tensor) {
        PyBuffer_Release(&buf);
        return NULL;
    }

    if (is_contiguous) { memcpy(tensor->data, buf.buf, (size_t)buf.len); }
    else {
        // Non-C-contiguous input (Fortran-order, transposed, or strided): walk every axis' stride.
        // Same dtype on both sides — `linearize_cast_into` compacts the strided bytes without a cast.
        size_t total_elements = 1;
        for (int d = 0; d < buf.ndim; d++) total_elements *= (size_t)buf.shape[d];
        linearize_cast_into((char const *)buf.buf, dtype, tensor->data, dtype, (size_t)buf.ndim, buf.shape, buf.strides,
                            total_elements);
    }

    PyBuffer_Release(&buf);
    return (PyObject *)tensor;
}

PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.Tensor",
    .tp_doc = "N-dimensional tensor with full NumPy-like API, supporting NumKong's type system",
    .tp_basicsize = sizeof(Tensor),
    .tp_dealloc = Tensor_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Tensor_tp_new,
    .tp_as_buffer = &Tensor_as_buffer,
    .tp_as_number = &Tensor_as_number,
    .tp_as_sequence = &Tensor_as_sequence,
    .tp_as_mapping = &Tensor_as_mapping,
    .tp_getset = Tensor_getset,
    .tp_methods = Tensor_methods,
    .tp_repr = Tensor_repr,
    .tp_str = Tensor_str,
    .tp_richcompare = Tensor_richcompare,
    .tp_iter = Tensor_iter,
};

#pragma region Block Scaled Tensor

static void ScaledTensor_dealloc(PyObject *self) {
    ScaledTensor *scaled = (ScaledTensor *)self;
    Py_XDECREF(scaled->elements);
    Py_XDECREF(scaled->block_scales);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ScaledTensor_get_elements(PyObject *self, void *closure) {
    nk_unused_(closure);
    ScaledTensor *scaled = (ScaledTensor *)self;
    Py_INCREF(scaled->elements);
    return (PyObject *)scaled->elements;
}

static PyObject *ScaledTensor_get_block_scales(PyObject *self, void *closure) {
    nk_unused_(closure);
    ScaledTensor *scaled = (ScaledTensor *)self;
    Py_INCREF(scaled->block_scales);
    return (PyObject *)scaled->block_scales;
}

static PyObject *ScaledTensor_get_tensor_scale(PyObject *self, void *closure) {
    nk_unused_(closure);
    ScaledTensor *scaled = (ScaledTensor *)self;
    if (!scaled->has_tensor_scale) Py_RETURN_NONE;
    return PyFloat_FromDouble((double)scaled->tensor_scale);
}

static PyObject *ScaledTensor_get_block_size(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((ScaledTensor *)self)->block_size);
}

/** Canonical name for a composite block-scaled dtype, the parser-accepted spelling. */
static char const *scaled_dtype_name(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_nvfp4_k: return "nvfp4";
    case nk_mxfp4_k: return "mxfp4";
    case nk_mxfp6_e2m3_k: return "mxfp6_e2m3";
    case nk_mxfp6_e3m2_k: return "mxfp6_e3m2";
    case nk_mxfp8_e4m3_k: return "mxfp8_e4m3";
    case nk_mxfp8_e5m2_k: return "mxfp8_e5m2";
    case nk_mxint8_k: return "mxint8";
    default: return nk_dtype_python_name(dtype);
    }
}

static PyObject *ScaledTensor_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyUnicode_FromString(scaled_dtype_name(((ScaledTensor *)self)->dtype));
}

static PyObject *ScaledTensor_get_shape(PyObject *self, void *closure) {
    nk_unused_(closure);
    ScaledTensor *scaled = (ScaledTensor *)self;
    Tensor *block_scales = scaled->block_scales;
    PyObject *shape_tuple = PyTuple_New(block_scales->rank);
    if (!shape_tuple) return NULL;
    // Logical shape = scales shape with the last axis multiplied back out to elements.
    for (size_t i = 0; i < block_scales->rank; i++) {
        Py_ssize_t dim = block_scales->shape[i];
        if (i + 1 == block_scales->rank) dim *= (Py_ssize_t)scaled->block_size;
        PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromSsize_t(dim));
    }
    return shape_tuple;
}

static PyObject *ScaledTensor_get_capacity(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((ScaledTensor *)self)->elements->capacity);
}

char const doc_method_scaled_resize[] =                                                                 //
    "Resize the packed elements and per-block scales in lockstep, within capacity, without moving\n"    //
    "either buffer.\n\n"                                                                                //
    "Atomic: fails, unchanged, if either child would exceed capacity, if the last axis is not a\n"      //
    "multiple of block_size, or while any child buffer is exported. Takes the dense logical shape.\n\n" //
    "Args:\n"                                                                                           //
    "    *shape: New logical dimensions, ints or a single shape tuple.\n\n"                             //
    "Returns:\n"                                                                                        //
    "    ScaledTensor: self.\n\n"                                                                       //
    "Signature:\n"                                                                                      //
    "    >>> def resize(self, *shape: int) -> 'ScaledTensor': ...\n";

static PyObject *ScaledTensor_resize(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    ScaledTensor *scaled = (ScaledTensor *)self;
    Tensor *elements = scaled->elements;
    Tensor *block_scales = scaled->block_scales;
    if (elements->exports != 0 || block_scales->exports != 0) {
        PyErr_SetString(PyExc_BufferError, "cannot resize a scaled tensor with exported buffers");
        return NULL;
    }
    Py_ssize_t logical[NK_TENSOR_MAX_RANK];
    size_t rank = 0;
    if (parse_shape_args_(args, nargs, logical, &rank) != 0) return NULL;
    if (rank == 0) {
        PyErr_SetString(PyExc_ValueError, "block-scaled tensors require at least one dimension");
        return NULL;
    }
    size_t const last_dim = (size_t)logical[rank - 1];
    if (scaled->block_size == 0 || last_dim % scaled->block_size != 0) {
        PyErr_Format(PyExc_ValueError, "last axis (%zu) must be a multiple of block_size (%zu)", last_dim,
                     scaled->block_size);
        return NULL;
    }

    nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(scaled->dtype);
    Py_ssize_t scales_shape[NK_TENSOR_MAX_RANK];
    for (size_t i = 0; i + 1 < rank; i++) scales_shape[i] = logical[i];
    scales_shape[rank - 1] = (Py_ssize_t)nk_block_scaled_scales_size((nk_size_t)last_dim, format);

    // Pre-validate both capacities so neither child is mutated on failure (atomic).
    if (shape_numel_(rank, logical) > elements->capacity || shape_numel_(rank, scales_shape) > block_scales->capacity) {
        PyErr_SetString(PyExc_ValueError, "resize exceeds element or scale capacity; grow the source first");
        return NULL;
    }
    tensor_set_contiguous_shape_(elements, rank, logical);
    tensor_set_contiguous_shape_(block_scales, rank, scales_shape);
    return Py_NewRef(self);
}

static PyGetSetDef ScaledTensor_getset[] = {
    {"elements", ScaledTensor_get_elements, NULL, "Packed sub-byte element Tensor (DLPack-exportable)", NULL},
    {"block_scales", ScaledTensor_get_block_scales, NULL, "Per-block scale Tensor (DLPack-exportable)", NULL},
    {"tensor_scale", ScaledTensor_get_tensor_scale, NULL, "Per-tensor float multiplier (None for MX formats)", NULL},
    {"block_size", ScaledTensor_get_block_size, NULL, "Number of elements per block", NULL},
    {"dtype", ScaledTensor_get_dtype, NULL, "Composite block-scaled dtype name", NULL},
    {"shape", ScaledTensor_get_shape, NULL, "Logical (dense) shape of the quantized tensor", NULL},
    {"capacity", ScaledTensor_get_capacity, NULL, "Allocated element capacity of the packed elements buffer", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

char const doc_method_scaled_astype[] =                                                    //
    "Materialize the block-scaled tensor to a dense dtype.\n\n"                            //
    "Args:\n"                                                                              //
    "    dtype (str): Target dense dtype, for example 'float32', 'float16' or 'bf16'.\n\n" //
    "Returns:\n"                                                                           //
    "    Tensor: Dequantized dense tensor of the requested dtype.\n\n"                     //
    "Signature:\n"                                                                         //
    "    >>> def astype(self, dtype, /): ...";

/** Transcode a @c ScaledTensor into another block-scaled format, e.g. NVFP4 → MXFP8. */
static PyObject *ScaledTensor_transcode(ScaledTensor *scaled, nk_dtype_t target_dtype) {
    nk_block_scaled_format_t to_format = nk_block_scaled_format_of_dtype(target_dtype);
    size_t to_block_size = to_format.block_size;

    Tensor *src_elements = scaled->elements;
    Tensor *src_scales = scaled->block_scales;
    size_t rank = src_scales->rank;

    // Logical shape: leading dims from the scales tensor, last dim = blocks * source block_size.
    Py_ssize_t out_shape[NK_TENSOR_MAX_RANK];
    size_t total = 1;
    for (size_t i = 0; i < rank; i++) {
        Py_ssize_t dim = src_scales->shape[i];
        if (i + 1 == rank) dim *= (Py_ssize_t)scaled->block_size;
        out_shape[i] = dim;
        total *= (size_t)dim;
    }
    Py_ssize_t last_dim = rank ? out_shape[rank - 1] : 0;
    if (to_block_size == 0 || (size_t)last_dim % to_block_size != 0) {
        PyErr_Format(PyExc_ValueError, "last dimension %zd must be a multiple of block size %zu for '%s'", last_dim,
                     to_block_size, nk_dtype_python_name(target_dtype));
        return NULL;
    }

    // Compact the (possibly strided) source child tensors for the flat-buffer kernel.
    size_t elem_total = 1, scale_total = 1;
    for (size_t i = 0; i < rank; i++) {
        elem_total *= (size_t)src_elements->shape[i];
        scale_total *= (size_t)src_scales->shape[i];
    }
    int elem_free = 0, scale_free = 0;
    char *elem_buf = ensure_contiguous_buffer(src_elements->data, src_elements->dtype, src_elements->dtype, rank,
                                              src_elements->shape, src_elements->strides, elem_total, &elem_free);
    char *scale_buf = scale_total
                          ? ensure_contiguous_buffer(src_scales->data, src_scales->dtype, src_scales->dtype, rank,
                                                     src_scales->shape, src_scales->strides, scale_total, &scale_free)
                          : src_scales->data;
    if (!elem_buf || (scale_total && !scale_buf)) {
        if (elem_free) PyMem_Free(elem_buf);
        if (scale_free) PyMem_Free(scale_buf);
        return NULL;
    }

    // Destination child shapes: elements keep the dense shape, scales count blocks along the last axis.
    Py_ssize_t dst_scales_shape[NK_TENSOR_MAX_RANK];
    for (size_t i = 0; i + 1 < rank; i++) dst_scales_shape[i] = out_shape[i];
    dst_scales_shape[rank - 1] = (Py_ssize_t)nk_block_scaled_scales_size((nk_size_t)last_dim, to_format);

    Tensor *dst_elements = Tensor_new(to_format.element_dtype, rank, out_shape);
    Tensor *dst_scales = Tensor_new(to_format.scale_dtype, rank, dst_scales_shape);
    ScaledTensor *result = dst_elements && dst_scales ? PyObject_New(ScaledTensor, &ScaledTensorType) : NULL;
    if (!result) {
        Py_XDECREF(dst_elements);
        Py_XDECREF(dst_scales);
        if (elem_free) PyMem_Free(elem_buf);
        if (scale_free) PyMem_Free(scale_buf);
        return NULL;
    }

    nk_block_scaled_format_t from_format = nk_block_scaled_format_of_dtype(scaled->dtype);
    int to_has_tensor_scale = (to_format.tensor_scale_dtype == nk_f32_k);
    nk_scalar_buffer_t from_tensor_scale, to_tensor_scale;
    memset(&from_tensor_scale, 0, sizeof(from_tensor_scale));
    memset(&to_tensor_scale, 0, sizeof(to_tensor_scale));
    from_tensor_scale.f32 = scaled->tensor_scale;
    to_tensor_scale.f32 = 0.0f; // zero → kernel derives the destination per-tensor scale (NVFP4)

    PyThreadState *gil = PyEval_SaveThread();
    nk_cast_block_scaled(elem_buf, scale_buf, scaled->has_tensor_scale ? &from_tensor_scale : NULL, &from_format, //
                         dst_elements->data, dst_scales->data, to_has_tensor_scale ? &to_tensor_scale : NULL,     //
                         &to_format, (nk_size_t)total);
    PyEval_RestoreThread(gil);

    if (elem_free) PyMem_Free(elem_buf);
    if (scale_free) PyMem_Free(scale_buf);

    result->elements = dst_elements;
    result->block_scales = dst_scales;
    result->dtype = target_dtype;
    result->block_size = to_block_size;
    result->tensor_scale = to_has_tensor_scale ? to_tensor_scale.f32 : 1.0f;
    result->has_tensor_scale = to_has_tensor_scale;
    return (PyObject *)result;
}

/** Decode a @c ScaledTensor into a dense @c Tensor of @p target_dtype, or transcode to another
 *  block-scaled format when @p target_dtype is itself block-scaled. */
static PyObject *ScaledTensor_astype(PyObject *self, PyObject *dtype_arg) {
    ScaledTensor *scaled = (ScaledTensor *)self;

    nk_dtype_t target_dtype = py_object_to_nk_dtype(dtype_arg);
    if (target_dtype == nk_dtype_unknown_k) return NULL;
    if (nk_dtype_is_block_scaled(target_dtype)) return ScaledTensor_transcode(scaled, target_dtype);

    Tensor *elements = scaled->elements;
    Tensor *block_scales = scaled->block_scales;
    size_t rank = block_scales->rank;

    // Logical shape: leading dims from the scales tensor, last dim = blocks * block_size.
    Py_ssize_t out_shape[NK_TENSOR_MAX_RANK];
    size_t total = 1;
    for (size_t i = 0; i < rank; i++) {
        Py_ssize_t dim = block_scales->shape[i];
        if (i + 1 == rank) dim *= (Py_ssize_t)scaled->block_size;
        out_shape[i] = dim;
        total *= (size_t)dim;
    }

    // Decode into f32 first (the block-scaled kernel's plain side), then cast to the target.
    Tensor *f32_result = Tensor_new(nk_f32_k, rank, out_shape);
    if (!f32_result) return NULL;

    // The kernel reads packed element/scale bytes as flat contiguous buffers indexed by logical
    // position; a sliced view may be strided, so compact both child tensors first.
    size_t elem_total = 1, scale_total = 1;
    for (size_t i = 0; i < rank; i++) {
        elem_total *= (size_t)elements->shape[i];
        scale_total *= (size_t)block_scales->shape[i];
    }
    int elem_free = 0, scale_free = 0;
    char *elem_buf = ensure_contiguous_buffer(elements->data, elements->dtype, elements->dtype, rank, elements->shape,
                                              elements->strides, elem_total, &elem_free);
    char *scale_buf = scale_total ? ensure_contiguous_buffer(block_scales->data, block_scales->dtype,
                                                             block_scales->dtype, rank, block_scales->shape,
                                                             block_scales->strides, scale_total, &scale_free)
                                  : block_scales->data;
    if (!elem_buf || (scale_total && !scale_buf)) {
        if (elem_free) PyMem_Free(elem_buf);
        if (scale_free) PyMem_Free(scale_buf);
        Py_DECREF(f32_result);
        return NULL;
    }

    nk_block_scaled_format_t from_format = nk_block_scaled_format_of_dtype(scaled->dtype);
    nk_block_scaled_format_t to_format = nk_plain(nk_f32_k);
    nk_scalar_buffer_t from_tensor_scale;
    memset(&from_tensor_scale, 0, sizeof(from_tensor_scale));
    from_tensor_scale.f32 = scaled->tensor_scale;

    PyThreadState *gil = PyEval_SaveThread();
    nk_cast_block_scaled(elem_buf, scale_buf, scaled->has_tensor_scale ? &from_tensor_scale : NULL, &from_format, //
                         f32_result->data, NULL, NULL, &to_format, (nk_size_t)total);
    PyEval_RestoreThread(gil);

    if (elem_free) PyMem_Free(elem_buf);
    if (scale_free) PyMem_Free(scale_buf);

    if (target_dtype == nk_f32_k) return (PyObject *)f32_result;

    // Cast the dense f32 result to the requested dtype.
    Tensor *result = Tensor_new(target_dtype, rank, out_shape);
    if (!result) {
        Py_DECREF(f32_result);
        return NULL;
    }
    nk_cast(f32_result->data, nk_f32_k, (nk_size_t)total, result->data, target_dtype);
    Py_DECREF(f32_result);
    return (PyObject *)result;
}

/** Build a sliced @c ScaledTensor view sharing @p source's metadata. */
static PyObject *ScaledTensor_view_from(ScaledTensor *source, Tensor *elements_view, Tensor *block_scales_view) {
    ScaledTensor *view = PyObject_New(ScaledTensor, &ScaledTensorType);
    if (!view) {
        Py_XDECREF(elements_view);
        Py_XDECREF(block_scales_view);
        return NULL;
    }
    view->elements = elements_view;         // reference stolen
    view->block_scales = block_scales_view; // reference stolen
    view->dtype = source->dtype;
    view->block_size = source->block_size;
    view->tensor_scale = source->tensor_scale;
    view->has_tensor_scale = source->has_tensor_scale;
    return (PyObject *)view;
}

/**
 *  @brief `scaled[row]` / `scaled[:, start:stop]` — slice elements and block_scales in lockstep.
 *
 *  Integer / slice indices address logical, dense, positions. The last-axis slice must be
 *  block-aligned, start and stop multiples of @c block_size, so it maps cleanly onto both the
 *  packed element bytes and the per-block scale bytes. The result is a ScaledTensor view sharing
 *  storage with the source.
 */
static PyObject *ScaledTensor_subscript(PyObject *self, PyObject *key) {
    ScaledTensor *scaled = (ScaledTensor *)self;
    Tensor *elements = scaled->elements;
    Tensor *block_scales = scaled->block_scales;
    size_t rank = block_scales->rank;
    size_t block_size = scaled->block_size;

    // Normalize the key into a tuple of per-dimension operations.
    PyObject *items = NULL; // borrowed list of per-dim keys
    PyObject *owned_tuple = NULL;
    Py_ssize_t nkeys;
    if (PyTuple_Check(key)) {
        owned_tuple = key;
        Py_INCREF(owned_tuple);
        nkeys = PyTuple_GET_SIZE(owned_tuple);
        items = owned_tuple;
    }
    else {
        owned_tuple = PyTuple_Pack(1, key);
        if (!owned_tuple) return NULL;
        nkeys = 1;
        items = owned_tuple;
    }
    if ((size_t)nkeys > rank) {
        Py_DECREF(owned_tuple);
        PyErr_SetString(PyExc_IndexError, "too many indices for ScaledTensor");
        return NULL;
    }

    // Walk each addressed dimension, accumulating element-byte and scale-byte offsets and the
    // sliced shapes for the two child tensors. Dimensions beyond `nkeys` are kept whole.
    Py_ssize_t elem_shape[NK_TENSOR_MAX_RANK], elem_strides[NK_TENSOR_MAX_RANK];
    Py_ssize_t scale_shape[NK_TENSOR_MAX_RANK], scale_strides[NK_TENSOR_MAX_RANK];
    char *elem_data = elements->data;
    char *scale_data = block_scales->data;
    size_t out_rank = 0;

    for (size_t d = 0; d < rank; d++) {
        int is_last = (d + 1 == rank);
        PyObject *idx = (Py_ssize_t)d < nkeys ? PyTuple_GET_ITEM(items, (Py_ssize_t)d) : NULL;

        if (idx == NULL) {
            // Dimension not addressed by the key: keep whole.
            elem_shape[out_rank] = elements->shape[d];
            elem_strides[out_rank] = elements->strides[d];
            scale_shape[out_rank] = block_scales->shape[d];
            scale_strides[out_rank] = block_scales->strides[d];
            out_rank++;
            continue;
        }

        if (PyLong_Check(idx)) {
            Py_ssize_t logical = is_last ? (block_scales->shape[d] * (Py_ssize_t)block_size) : block_scales->shape[d];
            Py_ssize_t i = PyLong_AsSsize_t(idx);
            if (i == -1 && PyErr_Occurred()) {
                Py_DECREF(owned_tuple);
                return NULL;
            }
            if (i < 0) i += logical;
            if (i < 0 || i >= logical) {
                Py_DECREF(owned_tuple);
                PyErr_SetString(PyExc_IndexError, "index out of bounds");
                return NULL;
            }
            if (is_last) {
                Py_DECREF(owned_tuple);
                PyErr_SetString(PyExc_IndexError,
                                "scalar indexing of the quantized (last) axis is not supported; slice instead");
                return NULL;
            }
            // Reduce this dimension (integer index): advance both child pointers.
            elem_data += i * elements->strides[d];
            scale_data += i * block_scales->strides[d];
            continue;
        }

        if (PySlice_Check(idx)) {
            Py_ssize_t logical = is_last ? (block_scales->shape[d] * (Py_ssize_t)block_size) : block_scales->shape[d];
            Py_ssize_t start, stop, step, slice_len;
            if (parse_slice(idx, logical, &start, &stop, &step, &slice_len) < 0) {
                Py_DECREF(owned_tuple);
                return NULL;
            }
            if (step != 1) {
                Py_DECREF(owned_tuple);
                PyErr_SetString(PyExc_IndexError, "ScaledTensor slicing requires step == 1");
                return NULL;
            }
            if (is_last) {
                // Last (quantized) axis: slice must be block-aligned.
                if ((size_t)start % block_size != 0 || (size_t)slice_len % block_size != 0) {
                    Py_DECREF(owned_tuple);
                    PyErr_Format(PyExc_IndexError, "last-axis slice [%zd:%zd] must be aligned to block size %zu", start,
                                 start + slice_len, block_size);
                    return NULL;
                }
                size_t start_blocks = (size_t)start / block_size;
                size_t len_blocks = (size_t)slice_len / block_size;
                elem_data += (Py_ssize_t)dimensions_to_values(elements->dtype, (size_t)start) * elements->strides[d];
                elem_shape[out_rank] = slice_len;
                elem_strides[out_rank] = elements->strides[d];
                scale_data += (Py_ssize_t)start_blocks * block_scales->strides[d];
                scale_shape[out_rank] = (Py_ssize_t)len_blocks;
                scale_strides[out_rank] = block_scales->strides[d];
            }
            else {
                elem_data += start * elements->strides[d];
                elem_shape[out_rank] = slice_len;
                elem_strides[out_rank] = elements->strides[d];
                scale_data += start * block_scales->strides[d];
                scale_shape[out_rank] = slice_len;
                scale_strides[out_rank] = block_scales->strides[d];
            }
            out_rank++;
            continue;
        }

        Py_DECREF(owned_tuple);
        PyErr_SetString(PyExc_TypeError, "ScaledTensor indices must be integers or slices");
        return NULL;
    }
    Py_DECREF(owned_tuple);

    Tensor *elem_root = elements->parent ? (Tensor *)elements->parent : elements;
    Tensor *scale_root = block_scales->parent ? (Tensor *)block_scales->parent : block_scales;
    Tensor *elem_view = Tensor_view(elem_root, elem_data, elements->dtype, out_rank, elem_shape, elem_strides);
    if (!elem_view) return NULL;
    Tensor *scale_view = Tensor_view(scale_root, scale_data, block_scales->dtype, out_rank, scale_shape, scale_strides);
    if (!scale_view) {
        Py_DECREF(elem_view);
        return NULL;
    }
    return ScaledTensor_view_from(scaled, elem_view, scale_view);
}

static PyMappingMethods ScaledTensor_as_mapping = {
    .mp_subscript = ScaledTensor_subscript,
};

static PyObject *ScaledTensor_repr(PyObject *self) {
    ScaledTensor *scaled = (ScaledTensor *)self;
    PyObject *shape = ScaledTensor_get_shape(self, NULL);
    if (!shape) return NULL;
    PyObject *tensor_scale = ScaledTensor_get_tensor_scale(self, NULL); // float or None
    if (!tensor_scale) {
        Py_DECREF(shape);
        return NULL;
    }
    PyObject *repr = PyUnicode_FromFormat("ScaledTensor(shape=%R, dtype='%s', block_size=%zu, tensor_scale=%R)", shape,
                                          scaled_dtype_name(scaled->dtype), scaled->block_size, tensor_scale);
    Py_DECREF(shape);
    Py_DECREF(tensor_scale);
    return repr;
}

static PyMethodDef ScaledTensor_methods[] = {
    {"astype", ScaledTensor_astype, METH_O, doc_method_scaled_astype},
    {"resize", (PyCFunction)ScaledTensor_resize, METH_FASTCALL, doc_method_scaled_resize},
    {NULL, NULL, 0, NULL},
};

PyTypeObject ScaledTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.ScaledTensor",
    .tp_doc = "Block-scaled tensor (OCP MX family + NVIDIA NVFP4): packed elements, per-block scales, " //
              "optional per-tensor scale. Produced by Tensor.astype('nvfp4'/'mxfp4'/...).",
    .tp_basicsize = sizeof(ScaledTensor),
    .tp_dealloc = ScaledTensor_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_mapping = &ScaledTensor_as_mapping,
    .tp_getset = ScaledTensor_getset,
    .tp_methods = ScaledTensor_methods,
    .tp_repr = ScaledTensor_repr,
};

#pragma endregion Block Scaled Tensor

/**
 *  @brief Fill a fresh contiguous tensor with `first + index * step` in row-major order.
 *
 *  A packed dtype shares bytes between elements, so it converts an f64 staging row with @c nk_cast.
 *  Returns 0 and releases @p tensor on allocation failure.
 */
static int tensor_fill_affine(Tensor *tensor, nk_f64_t first, nk_f64_t step) {
    size_t const total = shape_numel_(tensor->rank, tensor->shape);
    if (nk_dimensions_per_value(tensor->dtype) > 1) {
        nk_f64_t *staging = PyMem_Malloc(total * sizeof(nk_f64_t) + NK_TENSOR_PADDING_);
        if (!staging) {
            Py_DECREF(tensor);
            PyErr_NoMemory();
            return 0;
        }
        for (size_t i = 0; i < total; i++) staging[i] = first + step * (nk_f64_t)i;
        nk_cast(staging, nk_f64_k, (nk_size_t)total, tensor->data, tensor->dtype);
        PyMem_Free(staging);
        return 1;
    }
    size_t const element_size = nk_dtype_bytes_per_value(tensor->dtype);
    nk_scalar_buffer_t value;
    value.f64 = first;
    nk_scalar_buffer_from_f64(&value.f64, &value, tensor->dtype);
    for (size_t i = 0; i < total; i++) {
        if (step != 0.0 && i != 0) {
            value.f64 = first + step * (nk_f64_t)i;
            nk_scalar_buffer_from_f64(&value.f64, &value, tensor->dtype);
        }
        memcpy(tensor->data + i * element_size, &value, element_size);
    }
    return 1;
}

static int parse_shape(PyObject *shape_obj, Py_ssize_t *shape, size_t *rank) {
    if (PyLong_Check(shape_obj)) {
        Py_ssize_t size = PyLong_AsSsize_t(shape_obj);
        if (size < 0) {
            PyErr_SetString(PyExc_ValueError, "Shape dimensions must be non-negative");
            return 0;
        }
        shape[0] = size;
        *rank = 1;
        return 1;
    }
    if (!PyTuple_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be an int or tuple of ints");
        return 0;
    }
    Py_ssize_t ndim = PyTuple_Size(shape_obj);
    if (ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Shape has %zd dimensions, max is %d", ndim, NK_TENSOR_MAX_RANK);
        return 0;
    }
    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject *dim = PyTuple_GetItem(shape_obj, i);
        if (!PyLong_Check(dim)) {
            PyErr_SetString(PyExc_TypeError, "Shape dimensions must be integers");
            return 0;
        }
        Py_ssize_t size = PyLong_AsSsize_t(dim);
        if (size < 0) {
            PyErr_SetString(PyExc_ValueError, "Shape dimensions must be non-negative");
            return 0;
        }
        shape[i] = size;
    }
    *rank = (size_t)ndim;
    return 1;
}

char const doc_from_pointer[] =                                                          //
    "Create a zero-copy Tensor view from a raw memory address.\n\n"                      //
    "Args:\n"                                                                            //
    "    address: Integer memory address of the data buffer.\n"                          //
    "    shape: Shape of the array, an int or a tuple of ints.\n"                        //
    "    dtype: Data type string, for example 'float32' or 'bf16'.\n"                    //
    "    strides: Optional byte strides, defaulting to C-contiguous.\n"                  //
    "    owner: Optional Python object to prevent garbage collection of the source.\n\n" //
    "Returns:\n"                                                                         //
    "    Tensor: Non-owning view into the given memory.\n\n"                             //
    "Signature:\n"                                                                       //
    "    >>> def from_pointer(address, shape, dtype, /, *, strides=None, owner=None): ...";

PyObject *api_from_pointer(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *address_obj = NULL, *shape_obj = NULL, *dtype_obj = NULL;
    PyObject *strides_obj = NULL, *owner_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    // Positional: address, shape, dtype. Keyword-only: strides, owner.
    if (nargs < 3 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "from_pointer(address, shape, dtype, *, strides=None, owner=None)");
        return NULL;
    }

    address_obj = args[0];
    shape_obj = args[1];
    dtype_obj = args[2];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        PyObject *val = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(key, "strides") == 0) strides_obj = val;
        else if (PyUnicode_CompareWithASCIIString(key, "owner") == 0) owner_obj = val;
        else {
            PyErr_Format(PyExc_TypeError, "from_pointer() unexpected keyword: %S", key);
            return NULL;
        }
    }

    // Parse address
    void *data = PyLong_AsVoidPtr(address_obj);
    if (!data && PyErr_Occurred()) return NULL;
    if (!data) {
        PyErr_SetString(PyExc_ValueError, "address must be a non-zero integer pointer");
        return NULL;
    }

    // Parse shape
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    // Parse dtype
    nk_dtype_t dtype = py_object_to_nk_dtype(dtype_obj);
    if (dtype == nk_dtype_unknown_k) return NULL;
    if (!validate_packed_dimensions(dtype, rank, shape)) return NULL;

    // Compute or parse strides
    Py_ssize_t strides[NK_TENSOR_MAX_RANK];
    if (strides_obj && strides_obj != Py_None) {
        if (!PyTuple_Check(strides_obj)) {
            PyErr_SetString(PyExc_TypeError, "strides must be a tuple of ints");
            return NULL;
        }
        if ((size_t)PyTuple_Size(strides_obj) != rank) {
            PyErr_SetString(PyExc_ValueError, "strides length must match shape length");
            return NULL;
        }
        for (size_t i = 0; i < rank; i++) {
            PyObject *s = PyTuple_GetItem(strides_obj, (Py_ssize_t)i);
            if (!PyLong_Check(s)) {
                PyErr_SetString(PyExc_TypeError, "strides must be integers");
                return NULL;
            }
            strides[i] = PyLong_AsSsize_t(s);
            if (strides[i] == -1 && PyErr_Occurred()) return NULL;
        }
    }
    else { compute_contiguous_strides(rank, shape, dtype, strides); }

    // Use a sentinel owner if none provided — a non-NULL parent makes Tensor_dealloc treat this
    // as a view and not free the caller's external data buffer.
    if (!owner_obj || owner_obj == Py_None) owner_obj = Py_None;

    return (PyObject *)Tensor_view_object(owner_obj, (char *)data, dtype, rank, shape, strides);
}

char const doc_empty[] =                                       //
    "Create an uninitialized Tensor with the given shape.\n\n" //
    "Args:\n"                                                  //
    "    shape: Shape of the array.\n"                         //
    "    dtype: Data type, defaulting to 'float32'.\n\n"       //
    "Returns:\n"                                               //
    "    Tensor: Uninitialized array.\n\n"                     //
    "Signature:\n"                                             //
    "    >>> def empty(shape, /, *, dtype='float32'): ...";

PyObject *api_empty(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 1 || nargs + nkw > 2 || nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "empty(shape, *, dtype='float32')");
        return NULL;
    }

    shape_obj = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "empty() unexpected keyword: %S", key);
            return NULL;
        }
    }

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    return (PyObject *)Tensor_new(dtype, rank, shape);
}

char const doc_zeros[] =                                 //
    "Create a Tensor filled with zeros.\n\n"             //
    "Args:\n"                                            //
    "    shape: Shape of the array.\n"                   //
    "    dtype: Data type, defaulting to 'float32'.\n\n" //
    "Returns:\n"                                         //
    "    Tensor: Array of zeros.\n\n"                    //
    "Signature:\n"                                       //
    "    >>> def zeros(shape, /, *, dtype='float32'): ...";

PyObject *api_zeros(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 1 || nargs + nkw > 2 || nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "zeros(shape, *, dtype='float32')");
        return NULL;
    }

    shape_obj = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "zeros() unexpected keyword: %S", key);
            return NULL;
        }
    }

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    size_t const total = shape_numel_(rank, shape);
    memset(result->data, 0, dimensions_to_values(dtype, total) * nk_dtype_bytes_per_value(dtype));

    return (PyObject *)result;
}

char const doc_ones[] =                                  //
    "Create a Tensor filled with ones.\n\n"              //
    "Args:\n"                                            //
    "    shape: Shape of the array.\n"                   //
    "    dtype: Data type, defaulting to 'float32'.\n\n" //
    "Returns:\n"                                         //
    "    Tensor: Array of ones.\n\n"                     //
    "Signature:\n"                                       //
    "    >>> def ones(shape, /, *, dtype='float32'): ...";

PyObject *api_ones(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 1 || nargs + nkw > 2 || nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "ones(shape, *, dtype='float32')");
        return NULL;
    }

    shape_obj = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "ones() unexpected keyword: %S", key);
            return NULL;
        }
    }

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    if (!tensor_fill_affine(result, 1.0, 0.0)) return NULL;
    return (PyObject *)result;
}

char const doc_full[] =                                  //
    "Create a Tensor filled with a given value.\n\n"     //
    "Args:\n"                                            //
    "    shape: Shape of the array.\n"                   //
    "    fill_value: Value to fill the array with.\n"    //
    "    dtype: Data type, defaulting to 'float32'.\n\n" //
    "Returns:\n"                                         //
    "    Tensor: Array filled with fill_value.\n\n"      //
    "Signature:\n"                                       //
    "    >>> def full(shape, fill_value, /, *, dtype='float32'): ...";

PyObject *api_full(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *shape_obj = NULL, *fill_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 2 || nargs + nkw > 3 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "full(shape, fill_value, *, dtype='float32')");
        return NULL;
    }

    shape_obj = args[0];
    fill_obj = args[1];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "full() unexpected keyword: %S", key);
            return NULL;
        }
    }

    nk_f64_t fill_value;
    if (!py_number_to_f64(fill_obj, &fill_value)) {
        PyErr_SetString(PyExc_TypeError, "fill_value must be a number");
        return NULL;
    }

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    if (!tensor_fill_affine(result, fill_value, 0.0)) return NULL;
    return (PyObject *)result;
}

char const doc_iota[] =                                             //
    "Create a Tensor with incrementing values.\n\n"                 //
    "Args:\n"                                                       //
    "    shape: Shape of the array.\n"                              //
    "    seed: Starting value, defaulting to 0.\n"                  //
    "    dtype: Data type, defaulting to 'float32'.\n\n"            //
    "Returns:\n"                                                    //
    "    Tensor: Array with elements seed, seed+1, seed+2, ...\n\n" //
    "Signature:\n"                                                  //
    "    >>> def iota(shape, seed=0, /, *, dtype='float32'): ...";

PyObject *api_iota(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    long long seed = 0;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 1 || nargs + nkw > 3 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "iota(shape, seed=0, *, dtype='float32')");
        return NULL;
    }

    shape_obj = args[0];
    if (nargs >= 2) seed = PyLong_AsLongLong(args[1]);
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) seed = PyLong_AsLongLong(args[nargs + i]);
        else if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "iota() unexpected keyword: %S", key);
            return NULL;
        }
    }
    if (PyErr_Occurred()) return NULL;

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    if (!tensor_fill_affine(result, (nk_f64_t)seed, 1.0)) return NULL;
    return (PyObject *)result;
}

char const doc_diagonal[] =                                            //
    "Create an [n,n] Tensor with seed on the diagonal.\n\n"            //
    "Args:\n"                                                          //
    "    n: Size of the square matrix.\n"                              //
    "    seed: Diagonal value, defaulting to 1.\n"                     //
    "    dtype: Data type, defaulting to 'float32'.\n\n"               //
    "Returns:\n"                                                       //
    "    Tensor: [n,n] matrix with seed on diagonal, 0 elsewhere.\n\n" //
    "Signature:\n"                                                     //
    "    >>> def diagonal(n, seed=1, /, *, dtype='float32'): ...";

PyObject *api_diagonal(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *dtype_obj = NULL;
    long long seed = 1;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 1 || nargs + nkw > 3 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "diagonal(n, seed=1, *, dtype='float32')");
        return NULL;
    }

    Py_ssize_t n = PyLong_AsSsize_t(args[0]);
    if (n == -1 && PyErr_Occurred()) return NULL;
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    if (nargs >= 2) seed = PyLong_AsLongLong(args[1]);
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) seed = PyLong_AsLongLong(args[nargs + i]);
        else if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "diagonal() unexpected keyword: %S", key);
            return NULL;
        }
    }
    if (PyErr_Occurred()) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    Py_ssize_t shape[2] = {n, n};
    Tensor *result = Tensor_new(dtype, 2, shape);
    if (!result) return NULL;

    size_t const total = (size_t)n * (size_t)n;
    if (nk_dimensions_per_value(dtype) > 1) {
        nk_f64_t *staging = PyMem_Calloc(total + 1, sizeof(nk_f64_t));
        if (!staging) {
            Py_DECREF(result);
            return PyErr_NoMemory();
        }
        for (Py_ssize_t i = 0; i < n; i++) staging[(size_t)i * ((size_t)n + 1)] = (nk_f64_t)seed;
        nk_cast(staging, nk_f64_k, (nk_size_t)total, result->data, dtype);
        PyMem_Free(staging);
        return (PyObject *)result;
    }

    size_t elem_size = nk_dtype_bytes_per_value(dtype);
    memset(result->data, 0, total * elem_size);

    {
        nk_scalar_buffer_t val;
        val.f64 = (nk_f64_t)seed;
        nk_scalar_buffer_from_f64(&val.f64, &val, dtype);
        for (Py_ssize_t i = 0; i < n; i++) {
            memcpy(result->data + (size_t)i * ((size_t)n + 1) * elem_size, &val, elem_size);
        }
    }

    return (PyObject *)result;
}

static inline uint64_t splitmix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

char const doc_hash[] =                                                      //
    "Create a Tensor filled with deterministic pseudo-random bits.\n\n"      //
    "Args:\n"                                                                //
    "    shape: Shape of the array.\n"                                       //
    "    seed: Hash seed, defaulting to 0.\n"                                //
    "    dtype: Data type, defaulting to 'float32'.\n\n"                     //
    "Returns:\n"                                                             //
    "    Tensor: Array with hashed bit patterns reinterpreted as dtype.\n\n" //
    "Signature:\n"                                                           //
    "    >>> def hash(shape, seed=0, /, *, dtype='float32'): ...";

PyObject *api_hash(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    long long seed = 0;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;

    if (nargs + nkw < 1 || nargs + nkw > 3 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "hash(shape, seed=0, *, dtype='float32')");
        return NULL;
    }

    shape_obj = args[0];
    if (nargs >= 2) seed = PyLong_AsLongLong(args[1]);
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) seed = PyLong_AsLongLong(args[nargs + i]);
        else if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "hash() unexpected keyword: %S", key);
            return NULL;
        }
    }
    if (PyErr_Occurred()) return NULL;

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];

    {
        size_t elem_size = nk_dtype_bytes_per_value(dtype);
        uint64_t seed_hash = splitmix64((uint64_t)seed ^ 0x9E3779B97F4A7C15ULL);
        for (size_t i = 0; i < total; i++) {
            uint64_t bits = splitmix64(seed_hash + (uint64_t)i);
            memcpy(result->data + i * elem_size, &bits, elem_size);
        }
    }

    return (PyObject *)result;
}

char const doc_reduce_moments[] =                                                       //
    "Compute the sum and sum-of-squares, the moments, of all elements in an array.\n\n" //
    "Args:\n"                                                                           //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"       //
    "    dtype (str, optional): Override the presumed input element type.\n\n"          //
    "Returns:\n"                                                                        //
    "    tuple: (sum, sum_of_squares) for all elements.\n\n"                            //
    "Signature:\n"                                                                      //
    "    >>> def moments(a, /, *, dtype=None): ...";

PyObject *api_moments(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 1) return (PyErr_SetString(PyExc_TypeError, "moments(a) takes exactly 1 positional argument"), NULL);

    nk_dtype_t dtype_override = nk_dtype_unknown_k;
    Py_ssize_t const kwcount = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < kwcount; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            PyObject *value = args[nargs + i];
            if (value != Py_None) {
                dtype_override = py_object_to_nk_dtype(value);
                if (dtype_override == nk_dtype_unknown_k) return NULL;
            }
        }
        else return (PyErr_Format(PyExc_TypeError, "unexpected keyword '%S'", name), NULL);
    }

    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &TensorType)) {
        Tensor *t = (Tensor *)a_obj;
        TensorView view = {t->dtype, t->rank, t->shape, t->strides, t->data};
        if (dtype_override != nk_dtype_unknown_k) view.dtype = dtype_override;
        return impl_moments_from_view(&view);
    }

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(a_obj, &buffer, &view, &backing, dtype_override)) return NULL;
    PyObject *result = impl_moments_from_view(&view);
    PyBuffer_Release(&buffer);
    return result;
}

char const doc_reduce_minmax[] =                                                                //
    "Find minimum and maximum elements with their indices in an array.\n\n"                     //
    "Args:\n"                                                                                   //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"               //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                  //
    "Returns:\n"                                                                                //
    "    tuple: (min_val, min_index, max_val, max_index), or None if all elements are NaN.\n\n" //
    "Signature:\n"                                                                              //
    "    >>> def minmax(a, /, *, dtype=None): ...";

PyObject *api_minmax(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 1) return (PyErr_SetString(PyExc_TypeError, "minmax(a) takes exactly 1 positional argument"), NULL);

    nk_dtype_t dtype_override = nk_dtype_unknown_k;
    Py_ssize_t const kwcount = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < kwcount; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            PyObject *value = args[nargs + i];
            if (value != Py_None) {
                dtype_override = py_object_to_nk_dtype(value);
                if (dtype_override == nk_dtype_unknown_k) return NULL;
            }
        }
        else return (PyErr_Format(PyExc_TypeError, "unexpected keyword '%S'", name), NULL);
    }

    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &TensorType)) {
        Tensor *t = (Tensor *)a_obj;
        TensorView view = {t->dtype, t->rank, t->shape, t->strides, t->data};
        if (dtype_override != nk_dtype_unknown_k) view.dtype = dtype_override;
        return impl_minmax_from_view(&view);
    }

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(a_obj, &buffer, &view, &backing, dtype_override)) return NULL;
    PyObject *result = impl_minmax_from_view(&view);
    PyBuffer_Release(&buffer);
    return result;
}

char const doc_reduce_sum[] =                                                                        //
    "Return the sum of all elements, or per-slice sums along one or more axes.\n\n"                  //
    "Args:\n"                                                                                        //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"                    //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                      //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                       //
    "Returns:\n"                                                                                     //
    "    Scalar sum when axis is None.\n"                                                            //
    "    Tensor of per-slice sums when axis is given.\n\n"                                           //
    "Signature:\n"                                                                                   //
    "    >>> def sum(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_norm[] =                                                                       //
    "Return the L2 norm, or per-slice norms along one or more axes.\n\n"                             //
    "Args:\n"                                                                                        //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"                    //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                      //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                       //
    "Returns:\n"                                                                                     //
    "    Scalar L2 norm when axis is None.\n"                                                        //
    "    Tensor of per-slice norms when axis is given.\n\n"                                          //
    "Signature:\n"                                                                                   //
    "    >>> def norm(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_min[] =                                                                        //
    "Return the minimum element, or per-slice minimums along one or more axes.\n\n"                  //
    "Args:\n"                                                                                        //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"                    //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                      //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                       //
    "Returns:\n"                                                                                     //
    "    Scalar minimum when axis is None; None if all elements are NaN.\n"                          //
    "    Tensor of per-slice minimums when axis is given.\n\n"                                       //
    "Signature:\n"                                                                                   //
    "    >>> def min(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_max[] =                                                                        //
    "Return the maximum element, or per-slice maximums along one or more axes.\n\n"                  //
    "Args:\n"                                                                                        //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"                    //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along, defaulting to None,\n" //
    "        which reduces all elements.\n"                                                          //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"   //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                      //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                       //
    "Returns:\n"                                                                                     //
    "    Scalar maximum when axis is None; None if all elements are NaN.\n"                          //
    "    Tensor of per-slice maximums when axis is given.\n\n"                                       //
    "Signature:\n"                                                                                   //
    "    >>> def max(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_argmin[] =                                                                        //
    "Return the index of the minimum element, or per-slice indices along an axis.\n\n"                  //
    "Args:\n"                                                                                           //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"                       //
    "    axis (int, optional): Axis to reduce along, defaulting to None, which reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"      //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                         //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                          //
    "Returns:\n"                                                                                        //
    "    Integer index when axis is None.\n"                                                            //
    "    Tensor of per-slice indices when axis is given.\n\n"                                           //
    "Signature:\n"                                                                                      //
    "    >>> def argmin(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_argmax[] =                                                                        //
    "Return the index of the maximum element, or per-slice indices along an axis.\n\n"                  //
    "Args:\n"                                                                                           //
    "    a: Input array, a Tensor, NumPy array, or any buffer-protocol object.\n"                       //
    "    axis (int, optional): Axis to reduce along, defaulting to None, which reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"      //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                         //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                          //
    "Returns:\n"                                                                                        //
    "    Integer index when axis is None.\n"                                                            //
    "    Tensor of per-slice indices when axis is given.\n\n"                                           //
    "Signature:\n"                                                                                      //
    "    >>> def argmax(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";

PyObject *api_sum(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 1) return (PyErr_SetString(PyExc_TypeError, "sum() requires at least 1 argument"), NULL);
    if (PyObject_TypeCheck(args[0], &TensorType)) return Tensor_sum(args[0], args + 1, nargs - 1, kwnames);

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(args[0], &buffer, &view, &backing, nk_dtype_unknown_k)) return NULL;
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args + 1, nargs - 1, kwnames, view.rank, &parsed) < 0) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    if (parsed.dtype_override != nk_dtype_unknown_k) view.dtype = parsed.dtype_override;

    PyObject *result;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t sum_buf, sumsq_buf;
        nk_dtype_t sum_dtype, sumsq_dtype;
        if (impl_reduce_moments(&view, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype) < 0)
            result = PyErr_Format(PyExc_NotImplementedError, "sum not supported for dtype '%s'",
                                  nk_dtype_to_pybuffer_typestr(view.dtype));
        else result = nk_scalar_buffer_to_py_number(&sum_buf, sum_dtype);
    }
    else result = reduce_axis_dispatch(&view, &parsed, nk_reduce_moments_sum_dtype(view.dtype), sum_slice);
    PyBuffer_Release(&buffer);
    return result;
}

PyObject *api_norm(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 1) return (PyErr_SetString(PyExc_TypeError, "norm() requires at least 1 argument"), NULL);
    if (PyObject_TypeCheck(args[0], &TensorType)) return Tensor_norm(args[0], args + 1, nargs - 1, kwnames);

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(args[0], &buffer, &view, &backing, nk_dtype_unknown_k)) return NULL;
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args + 1, nargs - 1, kwnames, view.rank, &parsed) < 0) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    if (parsed.dtype_override != nk_dtype_unknown_k) view.dtype = parsed.dtype_override;

    PyObject *result;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t sum_buf, sumsq_buf;
        nk_dtype_t sum_dtype, sumsq_dtype;
        if (impl_reduce_moments(&view, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype) < 0)
            result = PyErr_Format(PyExc_NotImplementedError, "norm not supported for dtype '%s'",
                                  nk_dtype_to_pybuffer_typestr(view.dtype));
        else {
            nk_scalar_buffer_to_f64(&sumsq_buf, sumsq_dtype, &sumsq_buf.f64);
            result = PyFloat_FromDouble(nk_f64_sqrt(sumsq_buf.f64));
        }
    }
    else result = reduce_axis_dispatch(&view, &parsed, nk_f64_k, norm_slice);
    PyBuffer_Release(&buffer);
    return result;
}

PyObject *api_min(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 1) return (PyErr_SetString(PyExc_TypeError, "min() requires at least 1 argument"), NULL);
    if (PyObject_TypeCheck(args[0], &TensorType)) return Tensor_min(args[0], args + 1, nargs - 1, kwnames);

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(args[0], &buffer, &view, &backing, nk_dtype_unknown_k)) return NULL;
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args + 1, nargs - 1, kwnames, view.rank, &parsed) < 0) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    if (parsed.dtype_override != nk_dtype_unknown_k) view.dtype = parsed.dtype_override;

    PyObject *result;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            result = PyErr_Format(PyExc_NotImplementedError, "min not supported for dtype '%s'",
                                  nk_dtype_to_pybuffer_typestr(view.dtype));
        else if (min_idx == NK_SIZE_MAX) {
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else result = nk_scalar_buffer_to_py_number(&min_buf, min_dtype);
    }
    else result = reduce_axis_dispatch(&view, &parsed, nk_reduce_minmax_value_dtype(view.dtype), min_slice);
    PyBuffer_Release(&buffer);
    return result;
}

PyObject *api_max(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 1) return (PyErr_SetString(PyExc_TypeError, "max() requires at least 1 argument"), NULL);
    if (PyObject_TypeCheck(args[0], &TensorType)) return Tensor_max(args[0], args + 1, nargs - 1, kwnames);

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(args[0], &buffer, &view, &backing, nk_dtype_unknown_k)) return NULL;
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args + 1, nargs - 1, kwnames, view.rank, &parsed) < 0) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    if (parsed.dtype_override != nk_dtype_unknown_k) view.dtype = parsed.dtype_override;

    PyObject *result;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            result = PyErr_Format(PyExc_NotImplementedError, "max not supported for dtype '%s'",
                                  nk_dtype_to_pybuffer_typestr(view.dtype));
        else if (max_idx == NK_SIZE_MAX) {
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else result = nk_scalar_buffer_to_py_number(&max_buf, max_dtype);
    }
    else result = reduce_axis_dispatch(&view, &parsed, nk_reduce_minmax_value_dtype(view.dtype), max_slice);
    PyBuffer_Release(&buffer);
    return result;
}

PyObject *api_argmin(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 1) return (PyErr_SetString(PyExc_TypeError, "argmin() requires at least 1 argument"), NULL);
    if (PyObject_TypeCheck(args[0], &TensorType)) return Tensor_argmin(args[0], args + 1, nargs - 1, kwnames);

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(args[0], &buffer, &view, &backing, nk_dtype_unknown_k)) return NULL;
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args + 1, nargs - 1, kwnames, view.rank, &parsed) < 0) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    if (parsed.n_axes > 1) {
        PyBuffer_Release(&buffer);
        return (PyErr_SetString(PyExc_TypeError, "argmin does not support tuple axis"), NULL);
    }
    if (parsed.dtype_override != nk_dtype_unknown_k) view.dtype = parsed.dtype_override;

    PyObject *result;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            result = PyErr_Format(PyExc_NotImplementedError, "argmin not supported for dtype '%s'",
                                  nk_dtype_to_pybuffer_typestr(view.dtype));
        else if (min_idx == NK_SIZE_MAX) {
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else result = PyLong_FromSsize_t((Py_ssize_t)min_idx);
    }
    else result = reduce_axis_dispatch(&view, &parsed, nk_i64_k, argmin_slice);
    PyBuffer_Release(&buffer);
    return result;
}

PyObject *api_argmax(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 1) return (PyErr_SetString(PyExc_TypeError, "argmax() requires at least 1 argument"), NULL);
    if (PyObject_TypeCheck(args[0], &TensorType)) return Tensor_argmax(args[0], args + 1, nargs - 1, kwnames);

    Py_buffer buffer;
    nk_buffer_backing_t backing;
    TensorView view;
    if (!parse_tensor_nd(args[0], &buffer, &view, &backing, nk_dtype_unknown_k)) return NULL;
    reduce_args_t parsed;
    if (parse_reduce_kwargs(args + 1, nargs - 1, kwnames, view.rank, &parsed) < 0) {
        PyBuffer_Release(&buffer);
        return NULL;
    }
    if (parsed.n_axes > 1) {
        PyBuffer_Release(&buffer);
        return (PyErr_SetString(PyExc_TypeError, "argmax does not support tuple axis"), NULL);
    }
    if (parsed.dtype_override != nk_dtype_unknown_k) view.dtype = parsed.dtype_override;

    PyObject *result;
    if (parsed.n_axes == 0) {
        nk_scalar_buffer_t min_buf, max_buf;
        nk_dtype_t min_dtype, max_dtype;
        size_t min_idx, max_idx;
        if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_idx, &max_buf, &max_dtype, &max_idx) < 0)
            result = PyErr_Format(PyExc_NotImplementedError, "argmax not supported for dtype '%s'",
                                  nk_dtype_to_pybuffer_typestr(view.dtype));
        else if (max_idx == NK_SIZE_MAX) {
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else result = PyLong_FromSsize_t((Py_ssize_t)max_idx);
    }
    else result = reduce_axis_dispatch(&view, &parsed, nk_i64_k, argmax_slice);
    PyBuffer_Release(&buffer);
    return result;
}

int elementwise_prepare_out(                                                    //
    PyObject *out_obj, Py_buffer *out_buffer, nk_buffer_backing_t *out_backing, //
    Py_buffer const **inputs, size_t num_inputs, nk_dtype_t dtype,              //
    char **result_data, Py_ssize_t *result_strides, int *contiguous_tail,       //
    PyObject **return_obj) {

    Py_buffer const *a = inputs[0];
    int const ndim = a->ndim;

    // Fresh allocation: output is fully contiguous, so the input operands bound the tail.
    if (!out_obj || out_obj == Py_None) {
        Tensor *result_tensor = Tensor_new(dtype, (size_t)ndim, a->shape);
        if (!result_tensor) return 0;
        *return_obj = (PyObject *)result_tensor;
        *result_data = result_tensor->data;
        compute_contiguous_strides((size_t)ndim, a->shape, dtype, result_strides);
        *contiguous_tail = (int)shared_contiguous_tail_dimensions(inputs, num_inputs, (size_t)ndim);
        return 1;
    }

    // In-place: the (possibly strided) out buffer joins the operand set that bounds the tail.
    if (!nk_get_buffer(out_obj, out_buffer, PyBUF_STRIDES | PyBUF_FORMAT, out_backing)) return 0;
    if (!buffers_shapes_match(a, out_buffer)) return 0;
    nk_dtype_t out_dtype = resolve_nk_dtype_in_py_buffer(out_buffer);
    if (out_dtype != dtype) {
        PyErr_Format(PyExc_TypeError, "out dtype '%s' must match the compute dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(out_dtype), nk_dtype_to_pybuffer_typestr(dtype));
        return 0;
    }
    *result_data = out_buffer->buf;
    for (int dim = 0; dim < ndim; ++dim) result_strides[dim] = out_buffer->strides[dim];

    Py_buffer const *operands[4]; // inputs (≤3) + out
    for (size_t i = 0; i < num_inputs; ++i) operands[i] = inputs[i];
    operands[num_inputs] = out_buffer;
    *contiguous_tail = (int)shared_contiguous_tail_dimensions(operands, num_inputs + 1, (size_t)ndim);

    *return_obj = Py_None;
    Py_INCREF(Py_None);
    return 1;
}
