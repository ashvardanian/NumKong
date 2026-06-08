/**
 *  @brief Tensor implementation for NumKong Python bindings.
 *  @file python/tensor.c
 *
 *  This file implements the Tensor N-dimensional array type with NumPy-like
 *  interface and the TensorIter iterator.
 *
 *  Features:
 *  - Support for all NumKong dtypes (f32, f64, f16, bf16, i8, complex, etc.)
 *  - Arbitrary strides for views and slices
 *  - Zero-copy views with reference counting
 *  - Python buffer protocol for interoperability
 *  - Arithmetic operators (+, -, *, @)
 *  - Reduction operations (sum, min, max, argmin, argmax)
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

static int tensor_is_c_contig(Tensor *tensor, size_t item_size);
static int tensor_is_f_contig(Tensor *tensor, size_t item_size);

/** @brief Return a Python scalar from a tensor byte offset using nk_scalar_buffer_to_py_number. */
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

/** @brief Check if a tensor is C-contiguous (row-major). */
static int tensor_is_c_contig(Tensor *tensor, size_t item_size) {
    if (tensor->rank == 0) return 1;
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = tensor->rank; i > 0; i--) {
        if (tensor->strides[i - 1] != expected) return 0;
        expected *= tensor->shape[i - 1];
    }
    return 1;
}

/** @brief Check if a tensor is Fortran-contiguous (column-major). */
static int tensor_is_f_contig(Tensor *tensor, size_t item_size) {
    if (tensor->rank == 0) return 1;
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = 0; i < tensor->rank; i++) {
        if (tensor->strides[i] != expected) return 0;
        expected *= tensor->shape[i];
    }
    return 1;
}

static void Tensor_dealloc(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    Py_XDECREF(tensor->parent);
    Py_TYPE(self)->tp_free(self);
}

Tensor *Tensor_new(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    size_t const item_size = nk_dtype_bytes_per_value(dtype);
    size_t total_items = 1;
    for (size_t i = 0; i < rank; i++) {
        if (shape[i] > 0 && total_items > SIZE_MAX / (size_t)shape[i]) {
            PyErr_SetString(PyExc_OverflowError, "Tensor shape too large");
            return NULL;
        }
        total_items *= (size_t)shape[i];
    }
    if (item_size > 0 && total_items > SIZE_MAX / item_size) {
        PyErr_SetString(PyExc_OverflowError, "Tensor allocation too large");
        return NULL;
    }
    size_t const total_bytes = total_items * item_size;

    Tensor *tensor = PyObject_NewVar(Tensor, &TensorType, total_bytes + NK_TENSOR_PADDING_);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->rank = rank;

    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = (i < rank) ? shape[i] : 0;
        tensor->strides[i] = 0;
    }

    if (rank > 0) {
        tensor->strides[rank - 1] = (Py_ssize_t)item_size;
        for (size_t i = rank - 1; i > 0; i--) tensor->strides[i - 1] = tensor->strides[i] * tensor->shape[i];
    }

    tensor->parent = NULL;
    tensor->data = tensor->start;

    return tensor;
}

Tensor *Tensor_view(Tensor *parent, char *data_ptr, nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape,
                    Py_ssize_t const *strides) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "View rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    Tensor *view = PyObject_NewVar(Tensor, &TensorType, 0);
    if (!view) {
        PyErr_NoMemory();
        return NULL;
    }

    view->dtype = dtype;
    view->rank = rank;

    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        view->shape[i] = (i < rank) ? shape[i] : 0;
        view->strides[i] = (i < rank) ? strides[i] : 0;
    }

    view->parent = (PyObject *)parent;
    Py_INCREF(parent);
    view->data = data_ptr;

    return view;
}

/** @brief Create a 0D scalar tensor. */
static Tensor *Tensor_scalar(nk_dtype_t dtype, void const *value) {
    size_t const item_size = nk_dtype_bytes_per_value(dtype);
    Tensor *tensor = PyObject_NewVar(Tensor, &TensorType, item_size);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->rank = 0;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = 0;
        tensor->strides[i] = 0;
    }

    tensor->parent = NULL;
    tensor->data = tensor->start;
    memcpy(tensor->data, value, item_size);

    return tensor;
}

/** @brief Convert a 0D Tensor to a Python float. */
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

/** @brief Convert a 0D Tensor to a Python int. */
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

static PyObject *Tensor_positive(PyObject *self) { return Tensor_copy(self, NULL); }

/** @brief Compute C-contiguous strides for a tensor shape. */
void compute_contiguous_strides(size_t rank, Py_ssize_t const *shape, size_t item_size, Py_ssize_t *strides_out) {
    if (rank == 0) return;
    strides_out[rank - 1] = (Py_ssize_t)item_size;
    for (size_t d = rank - 1; d > 0; --d) strides_out[d - 1] = strides_out[d] * shape[d];
}

/**
 *  @brief Recursive stride walker for linearize_cast_into.
 *
 *  Walks non-contiguous outer dimensions recursively. Once only contiguous tail
 *  dimensions remain, processes the entire contiguous slice with memcpy or nk_cast.
 */
static void linearize_cast_recursive(                                         //
    char const *src_data, nk_dtype_t src_dtype, char *dest_data,              //
    nk_dtype_t dest_dtype, size_t src_element_size, size_t dest_element_size, //
    Py_ssize_t const *shape, Py_ssize_t const *strides,                       //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    // Base case: all remaining dimensions are contiguous — one operation
    if (remaining_dims <= contiguous_tail_dims) {
        size_t slice_elements = 1;
        for (size_t dim = 0; dim < remaining_dims; ++dim) slice_elements *= (size_t)shape[dim];
        if (src_dtype == dest_dtype) memcpy(dest_data, src_data, slice_elements * src_element_size);
        else nk_cast(src_data, src_dtype, (nk_size_t)slice_elements, dest_data, dest_dtype);
        return;
    }

    // Recursive case: iterate outermost non-contiguous dimension
    size_t const dim_extent = (size_t)shape[0];
    // Compute the contiguous dest stride for this level
    size_t inner_elements = 1;
    for (size_t dim = 1; dim < remaining_dims; ++dim) inner_elements *= (size_t)shape[dim];
    size_t const dest_row_bytes = inner_elements * dest_element_size;

    for (size_t position = 0; position < dim_extent; ++position) {
        linearize_cast_recursive(                                          //
            src_data + (Py_ssize_t)position * strides[0], src_dtype,       //
            dest_data + (Py_ssize_t)position * (Py_ssize_t)dest_row_bytes, //
            dest_dtype, src_element_size, dest_element_size,               //
            shape + 1, strides + 1,                                        //
            remaining_dims - 1, contiguous_tail_dims);
    }
}

void linearize_cast_into(char const *src_data, nk_dtype_t src_dtype, char *dest_data, nk_dtype_t dest_dtype,
                         size_t rank, Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements) {
    nk_unused_(total_elements);
    size_t src_element_size = nk_dtype_bytes_per_value(src_dtype);
    size_t dest_element_size = nk_dtype_bytes_per_value(dest_dtype);

    // Count how many trailing dims are contiguous in src
    size_t contiguous_tail_dims = 0;
    Py_ssize_t expected_stride = (Py_ssize_t)src_element_size;
    for (size_t dim = rank; dim-- > 0;) {
        if (strides[dim] != expected_stride) break;
        expected_stride *= shape[dim];
        contiguous_tail_dims++;
    }

    linearize_cast_recursive(src_data, src_dtype, dest_data, dest_dtype, src_element_size, dest_element_size, shape,
                             strides, rank, contiguous_tail_dims);
}

char *ensure_contiguous_buffer(char const *src_data, nk_dtype_t src_dtype, nk_dtype_t target_dtype, size_t rank,
                               Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements,
                               int *needs_free) {
    size_t src_element_size = nk_dtype_bytes_per_value(src_dtype);
    size_t dest_element_size = nk_dtype_bytes_per_value(target_dtype);

    // Check full contiguity
    int is_contiguous = 1;
    Py_ssize_t expected_stride = (Py_ssize_t)src_element_size;
    for (size_t dim = rank; dim-- > 0;) {
        if (strides[dim] != expected_stride) {
            is_contiguous = 0;
            break;
        }
        expected_stride *= shape[dim];
    }

    // Zero-copy: contiguous + same dtype
    if (is_contiguous && src_dtype == target_dtype) {
        *needs_free = 0;
        return (char *)src_data;
    }

    // Single allocation, delegate
    char *output = PyMem_Malloc(total_elements * dest_element_size + NK_TENSOR_PADDING_);
    if (!output) {
        PyErr_NoMemory();
        return NULL;
    }
    linearize_cast_into(src_data, src_dtype, output, target_dtype, rank, shape, strides, total_elements);
    *needs_free = 1;
    return output;
}

/** @brief Shared helper for tensor-scalar elementwise operations via scale kernel.
 *  Computes: result = alpha * a + beta */
static PyObject *tensor_elementwise_scalar(Tensor *a, double alpha_value, double beta_value) {
    nk_each_scale_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_scale_k, a->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_NotImplementedError, "scale not supported for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(a->dtype));
        return NULL;
    }

    Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
    if (!r) return NULL;

    size_t item_size = nk_dtype_bytes_per_value(a->dtype);
    Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

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
        nk_find_kernel_punned(nk_kernel_each_sum_k, a->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "add not supported for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = nk_dtype_bytes_per_value(a->dtype);
        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

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

        // Single-pass subtract via blend: result = 1·a + (−1)·b
        nk_each_blend_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_blend_k, a->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel,
                              &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "subtract not supported for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = nk_dtype_bytes_per_value(a->dtype);
        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

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
        nk_find_kernel_punned(nk_kernel_each_fma_k, a->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
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
        compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

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

        // fma(a, b, dummy, n, α=1, β=0) → 1·a·b + 0·dummy
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
    return PyUnicode_FromString(nk_dtype_name(tensor->dtype));
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
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];
    return PyLong_FromSsize_t(total * (Py_ssize_t)nk_dtype_bytes_per_value(tensor->dtype));
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

    PyObject *shape = Tensor_get_shape(self, NULL);
    if (!shape) {
        Py_DECREF(dict);
        return NULL;
    }
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
    size_t item_size = nk_dtype_bytes_per_value(tensor->dtype);
    return PyBool_FromLong(tensor_is_c_contig(tensor, item_size));
}

static PyObject *Tensor_get_data_ptr(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromVoidPtr(((Tensor *)self)->data);
}

static Tensor *Tensor_view_object(PyObject *owner, char *data_ptr, nk_dtype_t dtype, size_t rank,
                                  Py_ssize_t const *shape, Py_ssize_t const *strides) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "View rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    Tensor *view = PyObject_NewVar(Tensor, &TensorType, 0);
    if (!view) {
        PyErr_NoMemory();
        return NULL;
    }

    view->dtype = dtype;
    view->rank = rank;

    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        view->shape[i] = (i < rank) ? shape[i] : 0;
        view->strides[i] = (i < rank) ? strides[i] : 0;
    }

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
    {NULL, NULL, NULL, NULL, NULL},
};

char const doc_method_copy[] =                                   //
    "Return a deep copy of the tensor.\n\n"                      //
    "Returns:\n"                                                 //
    "    Tensor: Independent copy with its own data buffer.\n\n" //
    "Signature:\n"                                               //
    "    >>> def copy(self, /): ...";

PyObject *Tensor_copy(PyObject *self, PyObject *args) {
    nk_unused_(args);
    Tensor *tensor = (Tensor *)self;

    size_t total_elements = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elements *= (size_t)tensor->shape[i];

    Tensor *result = Tensor_new(tensor->dtype, tensor->rank, tensor->shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, tensor->dtype, tensor->rank, tensor->shape,
                        tensor->strides, total_elements);
    return (PyObject *)result;
}

char const doc_method_reshape[] =                                                       //
    "Return a tensor reshaped to the given dimensions.\n\n"                             //
    "Parameters:\n"                                                                     //
    "    *dims (int): New dimensions. Total element count must match the original.\n\n" //
    "Returns:\n"                                                                        //
    "    Tensor: Reshaped view (or copy if the layout requires it).\n\n"                //
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

    size_t item_size = nk_dtype_bytes_per_value(tensor->dtype);

    if (tensor_is_c_contig(tensor, item_size)) {
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
        Py_ssize_t stride = item_size;
        for (size_t i = new_rank; i > 0; i--) {
            new_strides[i - 1] = stride;
            stride *= new_shape[i - 1];
        }
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

char const doc_method_flatten[] =                                 //
    "Return a flattened 1D view of the tensor.\n\n"               //
    "Returns:\n"                                                  //
    "    Tensor: 1D view if contiguous, otherwise a 1D copy.\n\n" //
    "Signature:\n"                                                //
    "    >>> def flatten(self, /): ...";

PyObject *Tensor_flatten(PyObject *self, PyObject *args) {
    nk_unused_(args);
    Tensor *tensor = (Tensor *)self;

    Py_ssize_t total_elements = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elements *= tensor->shape[i];

    size_t item_size = nk_dtype_bytes_per_value(tensor->dtype);

    if (tensor_is_c_contig(tensor, item_size)) {
        Py_ssize_t flat_shape[1] = {total_elements};
        Py_ssize_t flat_strides[1] = {(Py_ssize_t)item_size};
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
    "Parameters:\n"                                                                         //
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

/** @brief Add a partial sum value into a running accumulator, using the accumulator dtype. */
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

/** @brief Detect trailing dimensions that form a single arithmetic progression.
 *  Returns how many rightmost dims satisfy stride[i] == stride[i+1] * shape[i+1].
 *  When all dims collapse, the entire tensor is one strided sequence. */
static size_t uniform_stride_tail_dims(Py_ssize_t const *shape, Py_ssize_t const *strides, size_t rank,
                                       size_t *out_count, Py_ssize_t *out_stride) {
    if (rank == 0) {
        *out_count = 1;
        *out_stride = 0;
        return 0;
    }
    size_t tail = 1;
    Py_ssize_t innermost = strides[rank - 1];
    Py_ssize_t expected = innermost;
    for (size_t i = rank - 1; i > 0; --i) {
        expected = expected * (Py_ssize_t)shape[i];
        if (strides[i - 1] != expected) break;
        ++tail;
    }
    size_t count = 1;
    for (size_t i = rank - tail; i < rank; ++i) count *= (size_t)shape[i];
    *out_count = count;
    *out_stride = innermost;
    return tail;
}

/** @brief Normalize a strided pointer for SIMD kernel consumption.
 *  For negative strides, flips the base pointer to the last element so the kernel walks forward. */
static char const *normalize_strided_base(char const *data, size_t count, Py_ssize_t stride,
                                          Py_ssize_t *out_stride_magnitude) {
    if (stride < 0) {
        *out_stride_magnitude = -stride;
        return data + (Py_ssize_t)(count - 1) * stride;
    }
    *out_stride_magnitude = stride;
    return data;
}

/** @brief Collapse trailing uniform-stride dims into one, writing the result into caller-provided arrays.
 *  @return The new rank after collapsing (remaining_dims - tail_dims + 1). */
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

/** @brief Recursively reduce moments over an N-D tensor using a SIMD kernel.
 *  Re-analyzes remaining dimensions at each level to collapse uniform-stride tails. */
static void reduce_moments_recursive(                   //
    nk_kernel_reduce_moments_punned_t kernel,           //
    nk_dtype_t sum_dtype, nk_dtype_t sumsq_dtype,       //
    char const *data, Py_ssize_t const *shape,          //
    Py_ssize_t const *strides, size_t rank, size_t dim, //
    nk_scalar_buffer_t *sum_accum, nk_scalar_buffer_t *sumsq_accum) {

    size_t remaining = rank - dim;
    if (remaining >= 2) {
        size_t collapsed_count;
        Py_ssize_t collapsed_stride;
        size_t tail = uniform_stride_tail_dims(shape + dim, strides + dim, remaining, &collapsed_count,
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
            reduce_moments_recursive(kernel, sum_dtype, sumsq_dtype, data, collapsed_shape, collapsed_strides,
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
            reduce_moments_recursive(kernel, sum_dtype, sumsq_dtype, data + i * stride, shape, strides, rank, dim + 1,
                                     sum_accum, sumsq_accum);
    }
}

/** @brief Reduce moments over an N-D tensor, returning typed scalar buffers. */
static int impl_reduce_moments(TensorView const *view, nk_scalar_buffer_t *sum_out, nk_dtype_t *sum_dtype_out,
                               nk_scalar_buffer_t *sumsq_out, nk_dtype_t *sumsq_dtype_out) {

    nk_kernel_reduce_moments_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_reduce_moments_k, view->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel,
                          &cap);
    if (!kernel || !cap) return -1;

    nk_dtype_t sum_dtype = nk_reduce_moments_sum_dtype(view->dtype);
    nk_dtype_t sumsq_dtype = nk_reduce_moments_sumsq_dtype(view->dtype);

    nk_scalar_buffer_t sum_buf, sumsq_buf;
    memset(&sum_buf, 0, sizeof(sum_buf));
    memset(&sumsq_buf, 0, sizeof(sumsq_buf));

    if (view->rank == 0) { kernel(view->data, 1, 0, &sum_buf, &sumsq_buf); }
    else {
        reduce_moments_recursive(kernel, sum_dtype, sumsq_dtype, view->data, view->shape, view->strides, view->rank, 0,
                                 &sum_buf, &sumsq_buf);
    }

    *sum_out = sum_buf;
    *sumsq_out = sumsq_buf;
    *sum_dtype_out = sum_dtype;
    *sumsq_dtype_out = sumsq_dtype;
    return 0;
}

/** @brief Type-aware less-than comparison for scalar buffers.
 *  Uses native comparisons for standard types and sign-magnitude comparators for mini-floats. */
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

/** @brief Update minmax accumulators with partial results from a SIMD kernel call.
 *  The SIMD kernel returns min/max values in the value dtype and indices relative to the slice.
 *  We compare against the running accumulators and update if better. */
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

/** @brief Recursively reduce minmax over an N-D tensor using a SIMD kernel.
 *  Re-analyzes remaining dimensions at each level to collapse uniform-stride tails. */
static void reduce_minmax_recursive(                         //
    nk_kernel_reduce_minmax_punned_t kernel,                 //
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
        size_t tail = uniform_stride_tail_dims(shape + dim, strides + dim, remaining, &collapsed_count,
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
            reduce_minmax_recursive(kernel, value_dtype, data, collapsed_shape, collapsed_strides, collapsed_rank, 0,
                                    flat_offset, min_accum, min_idx_accum, max_accum, max_idx_accum);
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
            reduce_minmax_recursive(kernel, value_dtype, data + i * stride, shape, strides, rank, dim + 1,
                                    flat_offset + i * inner_size, min_accum, min_idx_accum, max_accum, max_idx_accum);
    }
}

/** @brief Reduce minmax over an N-D tensor, returning typed scalar buffers. */
static int impl_reduce_minmax(TensorView const *view, nk_scalar_buffer_t *min_out, nk_dtype_t *min_dtype_out,
                              size_t *min_index_out, nk_scalar_buffer_t *max_out, nk_dtype_t *max_dtype_out,
                              size_t *max_index_out) {

    nk_kernel_reduce_minmax_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_reduce_minmax_k, view->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel,
                          &cap);
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

    if (view->rank == 0) { kernel(view->data, 1, 0, &min_buf, &min_idx, &max_buf, &max_idx); }
    else {
        size_t elem_size = nk_dtype_bytes_per_value(value_dtype);
        kernel(view->data, 1, (nk_size_t)elem_size, &min_buf, &min_idx, &max_buf, &max_idx);
        reduce_minmax_recursive(kernel, value_dtype, view->data, view->shape, view->strides, view->rank, 0, 0, &min_buf,
                                &min_idx, &max_buf, &max_idx);
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

/** @brief Parse a single axis value (int, tuple of ints, or None) into the axes array.
 *  Returns 1 if axes were set, 0 if None, -1 on error. */
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

/** @brief Normalize, sort, and validate the axes array against tensor rank. */
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

/** @brief Parse (axis=None, keepdims=False, out=None) from FASTCALL kwargs. */
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

/** @brief Compute output shape for multi-axis reduction. Returns output rank.
 *  @param axes  sorted array of axes to reduce, length n_axes. */
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

/** @brief Validate user-provided out tensor. */
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
    if (!tensor_is_c_contig(out, nk_dtype_bytes_per_value(expected_dtype)))
        return (PyErr_SetString(PyExc_ValueError, "out must be C-contiguous"), -1);
    return 0;
}

/** @brief Callback that processes one 1D slice and writes result to out. */
typedef void (*reduce_slice_fn_t)(TensorView const *slice, nk_scalar_buffer_t *out);

/** @brief Walk all positions except the reduced axes, calling on_slice for each sub-view.
 *  @param axes     sorted array of axes to reduce, length n_axes.
 *  @param next_ax  current scan position into the axes array. */
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

/** @brief Dispatch axis reduction: build output, walk slices, return result.
 *  For single-axis reductions on tensors where the non-axis dims are uniformly strided,
 *  uses a pointer-increment fast path instead of the recursive coordinate walker. */
static PyObject *reduce_axis_dispatch(TensorView const *view, reduce_args_t const *parsed, nk_dtype_t out_dtype,
                                      reduce_slice_fn_t on_slice) {
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
        size_t tail = uniform_stride_tail_dims(other_shapes, other_strides, other_count, &collapsed_count,
                                               &collapsed_stride);
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

char const doc_method_sum[] =                                                                                         //
    "Return the sum of all elements, or per-slice sums along one or more axes.\n\n"                                   //
    "Parameters:\n"                                                                                                   //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                                     //
    "Returns:\n"                                                                                                      //
    "    Scalar sum when axis is None.\n"                                                                             //
    "    Tensor of per-slice sums when axis is given.\n\n"                                                            //
    "Signature:\n"                                                                                                    //
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

char const doc_method_norm[] =                                                                                        //
    "Return the L2 norm, or per-slice norms along one or more axes.\n\n"                                              //
    "Parameters:\n"                                                                                                   //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                                     //
    "Returns:\n"                                                                                                      //
    "    Scalar L2 norm when axis is None.\n"                                                                         //
    "    Tensor of per-slice norms when axis is given.\n\n"                                                           //
    "Signature:\n"                                                                                                    //
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

char const doc_method_min[] =                                                                                         //
    "Return the minimum element, or per-slice minimums along one or more axes.\n\n"                                   //
    "Parameters:\n"                                                                                                   //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                                     //
    "Returns:\n"                                                                                                      //
    "    Scalar minimum when axis is None (None if all NaN).\n"                                                       //
    "    Tensor of per-slice minimums when axis is given.\n\n"                                                        //
    "Signature:\n"                                                                                                    //
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

char const doc_method_max[] =                                                                                         //
    "Return the maximum element, or per-slice maximums along one or more axes.\n\n"                                   //
    "Parameters:\n"                                                                                                   //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                                     //
    "Returns:\n"                                                                                                      //
    "    Scalar maximum when axis is None (None if all NaN).\n"                                                       //
    "    Tensor of per-slice maximums when axis is given.\n\n"                                                        //
    "Signature:\n"                                                                                                    //
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

char const doc_method_argmin[] =                                                                   //
    "Return the index of the minimum element, or per-slice indices along an axis.\n\n"             //
    "Parameters:\n"                                                                                //
    "    axis (int, optional): Axis to reduce along. When None (default), reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n" //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                  //
    "Returns:\n"                                                                                   //
    "    Integer index when axis is None.\n"                                                       //
    "    Tensor of per-slice indices when axis is given.\n\n"                                      //
    "Signature:\n"                                                                                 //
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

char const doc_method_argmax[] =                                                                   //
    "Return the index of the maximum element, or per-slice indices along an axis.\n\n"             //
    "Parameters:\n"                                                                                //
    "    axis (int, optional): Axis to reduce along. When None (default), reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n" //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n\n"                  //
    "Returns:\n"                                                                                   //
    "    Integer index when axis is None.\n"                                                       //
    "    Tensor of per-slice indices when axis is given.\n\n"                                      //
    "Signature:\n"                                                                                 //
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

char const doc_method_astype[] =                                                 //
    "Cast the tensor to a different dtype.\n\n"                                  //
    "Parameters:\n"                                                              //
    "    dtype (str): Target data type (e.g. 'float32', 'bf16').\n\n"            //
    "Returns:\n"                                                                 //
    "    Tensor: New tensor with elements converted to the requested dtype.\n\n" //
    "Signature:\n"                                                               //
    "    >>> def astype(self, dtype, /): ...";

PyObject *Tensor_astype(PyObject *self, PyObject *dtype_arg) {
    Tensor *tensor = (Tensor *)self;

    // Parse target dtype from argument
    nk_dtype_t target_dtype = py_object_to_nk_dtype(dtype_arg);
    if (target_dtype == nk_dtype_unknown_k) return NULL;

    // Same dtype -> return copy
    if (target_dtype == tensor->dtype) return Tensor_copy(self, NULL);

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

char const doc_method___array__[] =                                                       //
    "Convert to a NumPy array.\n\n"                                                       //
    "Parameters:\n"                                                                       //
    "    dtype (str, optional): Desired NumPy dtype. Default None (preserve original).\n" //
    "    copy (bool, optional): If True, force a copy. Default None.\n\n"                 //
    "Returns:\n"                                                                          //
    "    numpy.ndarray: Array sharing data when possible.\n\n"                            //
    "Signature:\n"                                                                        //
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
    {"copy", Tensor_copy, METH_NOARGS, doc_method_copy},
    {"reshape", (PyCFunction)Tensor_reshape, METH_FASTCALL, doc_method_reshape},
    {"moments", Tensor_moments, METH_NOARGS, doc_method_moments},
    {"minmax", Tensor_minmax, METH_NOARGS, doc_method_minmax},
    {"sum", (PyCFunction)Tensor_sum, METH_FASTCALL | METH_KEYWORDS, doc_method_sum},
    {"norm", (PyCFunction)Tensor_norm, METH_FASTCALL | METH_KEYWORDS, doc_method_norm},
    {"min", (PyCFunction)Tensor_min, METH_FASTCALL | METH_KEYWORDS, doc_method_min},
    {"max", (PyCFunction)Tensor_max, METH_FASTCALL | METH_KEYWORDS, doc_method_max},
    {"argmin", (PyCFunction)Tensor_argmin, METH_FASTCALL | METH_KEYWORDS, doc_method_argmin},
    {"argmax", (PyCFunction)Tensor_argmax, METH_FASTCALL | METH_KEYWORDS, doc_method_argmax},
    {"astype", Tensor_astype, METH_O, doc_method_astype},
    {"flatten", Tensor_flatten, METH_NOARGS, doc_method_flatten},
    {"squeeze", (PyCFunction)Tensor_squeeze, METH_FASTCALL, doc_method_squeeze},
    {"__array__", (PyCFunction)Tensor___array__, METH_FASTCALL | METH_KEYWORDS, doc_method___array__},
    {"__dlpack__", (PyCFunction)Tensor_dlpack, METH_VARARGS | METH_KEYWORDS, doc_dlpack},
    {"__dlpack_device__", (PyCFunction)Tensor_dlpack_device, METH_NOARGS, doc_dlpack_device},
    {NULL, NULL, 0, NULL},
};

static int Tensor_getbuffer(PyObject *export_from, Py_buffer *view, int flags) {
    Tensor *tensor = (Tensor *)export_from;
    size_t const item_size = nk_dtype_bytes_per_value(tensor->dtype);

    int c_contig = tensor_is_c_contig(tensor, item_size);
    int f_contig = tensor_is_f_contig(tensor, item_size);

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

    size_t total_items = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_items *= (size_t)tensor->shape[i];

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
        view->shape = tensor->rank > 0 ? &tensor->shape[0] : NULL;
    }
    else {
        view->ndim = 0;
        view->shape = NULL;
    }

    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) view->strides = tensor->rank > 0 ? &tensor->strides[0] : NULL;
    else view->strides = NULL;

    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(tensor);
    return 0;
}

static void Tensor_releasebuffer(PyObject *export_from, Py_buffer *view) {
    nk_unused_(export_from);
    nk_unused_(view);
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

static PyObject *Tensor_subscript(PyObject *self, PyObject *key) {
    Tensor *tensor = (Tensor *)self;

    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_IndexError, "0-dimensional tensor cannot be indexed");
        return NULL;
    }

    size_t item_size = nk_dtype_bytes_per_value(tensor->dtype);

    // Single slice
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slice_len;
        if (parse_slice(key, tensor->shape[0], &start, &stop, &step, &slice_len) < 0) return NULL;

        Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];

        new_shape[0] = slice_len;
        new_strides[0] = tensor->strides[0] * step;

        for (size_t i = 1; i < tensor->rank; i++) {
            new_shape[i] = tensor->shape[i];
            new_strides[i] = tensor->strides[i];
        }

        char *view_data = tensor->data + start * tensor->strides[0];
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

        if (tensor->rank == 1) { return tensor_read_scalar(tensor, idx * tensor->strides[0]); }

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
                view_data += index * tensor->strides[src_dim];
                // Dimension is consumed, rank reduced
            }
            else if (PySlice_Check(idx)) {
                // Slice: select a range, keep dimension
                Py_ssize_t start, stop, step, slice_len;
                if (parse_slice(idx, tensor->shape[src_dim], &start, &stop, &step, &slice_len) < 0) return NULL;
                view_data += start * tensor->strides[src_dim];
                new_shape[new_rank] = slice_len;
                new_strides[new_rank] = tensor->strides[src_dim] * step;
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
        if (new_rank == 0) { return tensor_read_scalar(tensor, (size_t)(view_data - tensor->data)); }

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

    PyObject *repr = PyUnicode_FromFormat("Tensor(shape=%R, dtype='%s')", shape_str, nk_dtype_name(tensor->dtype));
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
        equal = memcmp(a_buffer, b_buffer, total * item_size) == 0;
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
    if (array->rank == 1) { item = tensor_read_scalar(array, iter->index * array->strides[0]); }
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

    Tensor *tensor = Tensor_new(dtype, (size_t)buf.ndim, buf.shape);
    if (!tensor) {
        PyBuffer_Release(&buf);
        return NULL;
    }

    if (PyBuffer_IsContiguous(&buf, 'C')) { memcpy(tensor->data, buf.buf, (size_t)buf.len); }
    else {
        size_t row_bytes = (size_t)buf.itemsize;
        for (int d = 1; d < buf.ndim; d++) row_bytes *= (size_t)buf.shape[d];
        Py_ssize_t dst_stride = (Py_ssize_t)row_bytes;
        Py_ssize_t src_stride = buf.strides[0];
        for (Py_ssize_t i = 0; i < buf.shape[0]; i++)
            memcpy(tensor->data + i * dst_stride, (char *)buf.buf + i * src_stride, row_bytes);
    }

    PyBuffer_Release(&buf);
    return (PyObject *)tensor;
}

PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.Tensor",
    .tp_doc = "N-dimensional tensor with full NumPy-like API, supporting NumKong's type system",
    .tp_basicsize = sizeof(Tensor),
    .tp_itemsize = sizeof(char),
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
    "Parameters:\n"                                                                      //
    "    address: Integer memory address of the data buffer.\n"                          //
    "    shape: Shape of the array (int or tuple of ints).\n"                            //
    "    dtype: Data type string (e.g. 'float32', 'bf16').\n"                            //
    "    strides: Optional byte strides (default: C-contiguous).\n"                      //
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
    else { compute_contiguous_strides(rank, shape, nk_dtype_bytes_per_value(dtype), strides); }

    // Use a sentinel owner if none provided — we need a non-NULL parent
    // so Tensor_dealloc doesn't try to free the inline data
    if (!owner_obj || owner_obj == Py_None) owner_obj = Py_None;

    return (PyObject *)Tensor_view_object(owner_obj, (char *)data, dtype, rank, shape, strides);
}

char const doc_empty[] =                                       //
    "Create an uninitialized Tensor with the given shape.\n\n" //
    "Parameters:\n"                                            //
    "    shape: Shape of the array.\n"                         //
    "    dtype: Data type (default 'float32').\n\n"            //
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

char const doc_zeros[] =                            //
    "Create a Tensor filled with zeros.\n\n"        //
    "Parameters:\n"                                 //
    "    shape: Shape of the array.\n"              //
    "    dtype: Data type (default 'float32').\n\n" //
    "Returns:\n"                                    //
    "    Tensor: Array of zeros.\n\n"               //
    "Signature:\n"                                  //
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

    size_t nbytes = nk_dtype_bytes_per_value(dtype);
    for (size_t i = 0; i < rank; i++) nbytes *= (size_t)shape[i];
    memset(result->data, 0, nbytes);

    return (PyObject *)result;
}

char const doc_ones[] =                             //
    "Create a Tensor filled with ones.\n\n"         //
    "Parameters:\n"                                 //
    "    shape: Shape of the array.\n"              //
    "    dtype: Data type (default 'float32').\n\n" //
    "Returns:\n"                                    //
    "    Tensor: Array of ones.\n\n"                //
    "Signature:\n"                                  //
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

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];

    {
        size_t elem_size = nk_dtype_bytes_per_value(dtype);
        nk_scalar_buffer_t one;
        one.f64 = 1.0;
        nk_scalar_buffer_from_f64(&one.f64, &one, dtype);
        for (size_t i = 0; i < total; i++) memcpy(result->data + i * elem_size, &one, elem_size);
    }

    return (PyObject *)result;
}

char const doc_full[] =                               //
    "Create a Tensor filled with a given value.\n\n"  //
    "Parameters:\n"                                   //
    "    shape: Shape of the array.\n"                //
    "    fill_value: Value to fill the array with.\n" //
    "    dtype: Data type (default 'float32').\n\n"   //
    "Returns:\n"                                      //
    "    Tensor: Array filled with fill_value.\n\n"   //
    "Signature:\n"                                    //
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

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];

    {
        size_t elem_size = nk_dtype_bytes_per_value(dtype);
        nk_scalar_buffer_t val;
        val.f64 = fill_value;
        nk_scalar_buffer_from_f64(&val.f64, &val, dtype);
        for (size_t i = 0; i < total; i++) memcpy(result->data + i * elem_size, &val, elem_size);
    }

    return (PyObject *)result;
}

char const doc_iota[] =                                             //
    "Create a Tensor with incrementing values.\n\n"                 //
    "Parameters:\n"                                                 //
    "    shape: Shape of the array.\n"                              //
    "    seed: Starting value (default 0).\n"                       //
    "    dtype: Data type (default 'float32').\n\n"                 //
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

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];

    {
        size_t elem_size = nk_dtype_bytes_per_value(dtype);
        nk_scalar_buffer_t val;
        for (size_t i = 0; i < total; i++) {
            val.f64 = (nk_f64_t)(seed + (nk_i64_t)i);
            nk_scalar_buffer_from_f64(&val.f64, &val, dtype);
            memcpy(result->data + i * elem_size, &val, elem_size);
        }
    }

    return (PyObject *)result;
}

char const doc_diagonal[] =                                            //
    "Create an n x n Tensor with seed on the diagonal.\n\n"            //
    "Parameters:\n"                                                    //
    "    n: Size of the square matrix.\n"                              //
    "    seed: Diagonal value (default 1).\n"                          //
    "    dtype: Data type (default 'float32').\n\n"                    //
    "Returns:\n"                                                       //
    "    Tensor: n x n matrix with seed on diagonal, 0 elsewhere.\n\n" //
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

    size_t elem_size = nk_dtype_bytes_per_value(dtype);
    size_t total_bytes = (size_t)n * (size_t)n * elem_size;
    memset(result->data, 0, total_bytes);

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
    "Parameters:\n"                                                          //
    "    shape: Shape of the array.\n"                                       //
    "    seed: Hash seed (default 0).\n"                                     //
    "    dtype: Data type (default 'float32').\n\n"                          //
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

char const doc_reduce_moments[] =                                                //
    "Compute sum and sum-of-squares (moments) of all elements in an array.\n\n"  //
    "Parameters:\n"                                                              //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n" //
    "    dtype (str, optional): Override the presumed input element type.\n\n"   //
    "Returns:\n"                                                                 //
    "    tuple: (sum, sum_of_squares) for all elements.\n\n"                     //
    "Signature:\n"                                                               //
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
    "Parameters:\n"                                                                             //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                //
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

char const doc_reduce_sum[] =                                                                                         //
    "Return the sum of all elements, or per-slice sums along one or more axes.\n\n"                                   //
    "Parameters:\n"                                                                                                   //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                                      //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                                       //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                                        //
    "Returns:\n"                                                                                                      //
    "    Scalar sum when axis is None.\n"                                                                             //
    "    Tensor of per-slice sums when axis is given.\n\n"                                                            //
    "Signature:\n"                                                                                                    //
    "    >>> def sum(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_norm[] =                                                                                        //
    "Return the L2 norm, or per-slice norms along one or more axes.\n\n"                                              //
    "Parameters:\n"                                                                                                   //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                                      //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                                       //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                                        //
    "Returns:\n"                                                                                                      //
    "    Scalar L2 norm when axis is None.\n"                                                                         //
    "    Tensor of per-slice norms when axis is given.\n\n"                                                           //
    "Signature:\n"                                                                                                    //
    "    >>> def norm(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_min[] =                                                                                         //
    "Return the minimum element, or per-slice minimums along one or more axes.\n\n"                                   //
    "Parameters:\n"                                                                                                   //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                                      //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                                       //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                                        //
    "Returns:\n"                                                                                                      //
    "    Scalar minimum when axis is None (None if all NaN).\n"                                                       //
    "    Tensor of per-slice minimums when axis is given.\n\n"                                                        //
    "Signature:\n"                                                                                                    //
    "    >>> def min(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_max[] =                                                                                         //
    "Return the maximum element, or per-slice maximums along one or more axes.\n\n"                                   //
    "Parameters:\n"                                                                                                   //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                                      //
    "    axis (int or tuple of ints, optional): Axis or axes to reduce along. None (default) reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n"                    //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                                       //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                                        //
    "Returns:\n"                                                                                                      //
    "    Scalar maximum when axis is None (None if all NaN).\n"                                                       //
    "    Tensor of per-slice maximums when axis is given.\n\n"                                                        //
    "Signature:\n"                                                                                                    //
    "    >>> def max(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_argmin[] =                                                                   //
    "Return the index of the minimum element, or per-slice indices along an axis.\n\n"             //
    "Parameters:\n"                                                                                //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                   //
    "    axis (int, optional): Axis to reduce along. When None (default), reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n" //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                    //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                     //
    "Returns:\n"                                                                                   //
    "    Integer index when axis is None.\n"                                                       //
    "    Tensor of per-slice indices when axis is given.\n\n"                                      //
    "Signature:\n"                                                                                 //
    "    >>> def argmin(a, /, axis=None, *, keepdims=False, out=None, dtype=None): ...";
char const doc_reduce_argmax[] =                                                                   //
    "Return the index of the maximum element, or per-slice indices along an axis.\n\n"             //
    "Parameters:\n"                                                                                //
    "    a: Input array (Tensor, NumPy array, or any buffer-protocol object).\n"                   //
    "    axis (int, optional): Axis to reduce along. When None (default), reduces all elements.\n" //
    "    keepdims (bool, optional): Keep the reduced axis as a size-1 dimension. Default False.\n" //
    "    out (Tensor, optional): Pre-allocated output tensor for the result.\n"                    //
    "    dtype (str, optional): Override the presumed input element type.\n\n"                     //
    "Returns:\n"                                                                                   //
    "    Integer index when axis is None.\n"                                                       //
    "    Tensor of per-slice indices when axis is given.\n\n"                                      //
    "Signature:\n"                                                                                 //
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
