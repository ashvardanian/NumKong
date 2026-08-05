/**
 *  @brief NumKong Tensor Type, constructors, reductions, and stride utilities for Python.
 *  @file python/tensor.h
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *
 *  Declares the Tensor N-dimensional array type, its iterator, factory functions
 *  (empty, zeros, ones, full), reduction operations (moments, minmax),
 *  stride-walking utilities (linearize_cast_into, ensure_contiguous_buffer,
 *  shared_contiguous_tail_dimensions), and recursive elementwise dispatch helpers
 *  (each_sum_recursive, each_scale_recursive, each_fma_recursive, each_blend_recursive).
 */
#ifndef NK_PYTHON_TENSOR_H
#define NK_PYTHON_TENSOR_H

#include "numkong.h"

/**
 *  Extra bytes appended to every SIMD-facing heap allocation (promoted buffers,
 *  cast-staging buffers, Tensor inline storage). MSVC versions before 19.30
 *  can mis-compile AVX-512 masked stores (`_mm256_mask_storeu_epi32` and friends),
 *  emitting a full-width store that writes past the logical buffer end. Adding
 *  one ZMM register (64 bytes) of padding absorbs any such spill. On GCC/Clang
 *  the masked instructions are correct, so zero padding suffices.
 */
#if defined(_MSC_VER)
#define NK_TENSOR_PADDING_ 64
#else
#define NK_TENSOR_PADDING_ 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief N-dimensional array type with NumPy-like interface.
 *
 *  Supports arbitrary strides for views and slices, reference counting
 *  for memory management, and NumKong's extended type system.
 *
 *  Memory layout:
 *  - If parent == NULL: owns data in variable-length `start[]` buffer
 *  - If parent != NULL: view into parent's memory, `data` points there
 */
typedef struct Tensor {
    PyObject_VAR_HEAD
    /** Logical dtype (f32, f64, bf16, etc.). */
    nk_dtype_t dtype;
    /** Number of dimensions (0 for scalar). */
    size_t rank;
    /** Extent along each dimension. */
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    /** Stride in bytes for each dimension. */
    Py_ssize_t strides[NK_TENSOR_MAX_RANK];
    /** Reference to parent (NULL if owns data). */
    PyObject *parent;
    /** Data pointer (start[] if owns, parent's if view). */
    char *data;
    /** Variable-length inline data storage. */
    char start[];
} Tensor;

/**
 *  @brief Iterator for Tensor.
 *
 *  Iterates over the first dimension of a Tensor, yielding views
 *  (for rank > 1) or scalars (for rank == 1).
 */
typedef struct TensorIter {
    PyObject_HEAD
    /** Tensor being iterated. */
    Tensor *array;
    /** Current position. */
    Py_ssize_t index;
} TensorIter;

/** @brief Tensor Python type object.  */
extern PyTypeObject TensorType;

/** @brief Tensor iterator Python type object.  */
extern PyTypeObject TensorIterType;

/**
 *  @brief Allocate a new Tensor with uninitialized data.
 *  @param[in] dtype Logical dtype for elements.
 *  @param[in] rank Number of dimensions.
 *  @param[in] shape Array of dimension sizes.
 *  @return New Tensor, or NULL on allocation failure.
 */
Tensor *Tensor_new(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape);

/**
 *  @brief Create a view into an existing Tensor.
 *
 *  The view shares memory with the parent and holds a reference to it.
 *
 *  @param[in] parent Parent Tensor (reference count incremented).
 *  @param[in] data Pointer to first element of view.
 *  @param[in] dtype Logical dtype (usually same as parent).
 *  @param[in] rank Number of dimensions.
 *  @param[in] shape Array of dimension sizes.
 *  @param[in] strides Array of byte strides.
 *  @return New Tensor view, or NULL on failure.
 */
Tensor *Tensor_view(Tensor *parent, char *data, nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape,
                    Py_ssize_t const *strides);

/** @brief Copy the tensor.  */
PyObject *Tensor_copy(PyObject *self, PyObject *args);

/** @brief Reshape the tensor (returns view if possible).  */
PyObject *Tensor_reshape(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

/** @brief Compute moments (sum, sum-of-squares). Returns (sum, sumsq) tuple.  */
PyObject *Tensor_moments(PyObject *self, PyObject *args);

/** @brief Find min and max with indices. Returns (min_val, min_idx, max_val, max_idx) tuple.  */
PyObject *Tensor_minmax(PyObject *self, PyObject *args);

/** @brief Cast tensor to a different dtype. Returns a new tensor.  */
PyObject *Tensor_astype(PyObject *self, PyObject *dtype_arg);

/**
 *  @brief Compute C-contiguous strides for a tensor shape.
 */
void compute_contiguous_strides(size_t rank, Py_ssize_t const *shape, size_t item_size, Py_ssize_t *strides_out);

/**
 *  @brief Linearize strided source data into a contiguous destination, with optional dtype cast.
 *
 *  Walks the source tensor's strides recursively. For each contiguous inner slice,
 *  calls nk_cast (or memcpy if same dtype) directly into @p dest_data. Zero allocations.
 */
void linearize_cast_into(char const *src_data, nk_dtype_t src_dtype, char *dest_data, nk_dtype_t dest_dtype,
                         size_t rank, Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements);

/**
 *  @brief Convert a dense source into an arbitrarily-strided destination — the mirror of
 *         linearize_cast_into, for writing results back into a caller's non-contiguous buffer.
 *
 *  Collapses to a single nk_cast when the destination is fully packed. Zero allocations.
 */
void cast_into_strided(char const *src_data, nk_dtype_t src_dtype, char *dest_data, nk_dtype_t dest_dtype, size_t rank,
                       Py_ssize_t const *shape, Py_ssize_t const *dest_strides);

/**
 *  @brief Produce a contiguous buffer in the target dtype from arbitrary-strided input.
 *  @return Contiguous data pointer, or NULL on error. Caller must PyMem_Free if *needs_free is set.
 */
char *ensure_contiguous_buffer(char const *src_data, nk_dtype_t src_dtype, nk_dtype_t target_dtype, size_t rank,
                               Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements,
                               int *needs_free);

/**
 *  @brief Validate that two buffers have identical shapes (element-wise compatibility).
 *  @return 1 if shapes match, 0 otherwise (with Python exception set).
 */
int buffers_shapes_match(Py_buffer const *first, Py_buffer const *second);

/**
 *  @brief Validate a caller-supplied `out` buffer for an operation that writes it densely.
 *
 *  Requires the exact input shape, the expected dtype, C-contiguity — the layout
 *  linearize_cast_into writes, as it derives destination offsets from shape alone —
 *  writability, and no address overlap with @p input_buffer.
 *
 *  @return The output base pointer, or NULL on error with a Python exception set.
 */
char *validate_out_py_buffer(Py_buffer const *out_buffer, Py_buffer const *input_buffer, nk_dtype_t expected_dtype);

/**
 *  @brief Compute the number of trailing contiguous dimensions shared across multiple buffers.
 */
size_t shared_contiguous_tail_dimensions(Py_buffer const *buffers[], size_t num_buffers, size_t num_dims);

/**
 *  @brief Recursively apply a binary elementwise sum kernel to N-D tensors.
 */
void each_sum_recursive(                                           //
    nk_each_sum_punned_t kernel,                                   //
    char const *a_data, char const *b_data, char *result_data,     //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,          //
    Py_ssize_t const *b_strides, Py_ssize_t const *result_strides, //
    size_t remaining_dims, size_t contiguous_tail_dims);

/**
 *  @brief Recursively apply a unary elementwise scale kernel to an N-D tensor.
 */
void each_scale_recursive(                                           //
    nk_each_scale_punned_t kernel,                                   //
    char const *a_data, char *result_data,                           //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta, //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,            //
    Py_ssize_t const *result_strides,                                //
    size_t remaining_dims, size_t contiguous_tail_dims);

/**
 *  @brief Recursively apply a ternary fused-multiply-add kernel to N-D tensors.
 */
void each_fma_recursive(                                                           //
    nk_each_fma_punned_t kernel,                                                   //
    char const *a_data, char const *b_data, char const *c_data, char *result_data, //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta,               //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,                          //
    Py_ssize_t const *b_strides, Py_ssize_t const *c_strides,                      //
    Py_ssize_t const *result_strides,                                              //
    size_t remaining_dims, size_t contiguous_tail_dims);

/**
 *  @brief Recursively apply a binary elementwise blend kernel to N-D tensors.
 */
void each_blend_recursive(                                           //
    nk_each_blend_punned_t kernel,                                   //
    char const *a_data, char const *b_data, char *result_data,       //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta, //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,            //
    Py_ssize_t const *b_strides, Py_ssize_t const *result_strides,   //
    size_t remaining_dims, size_t contiguous_tail_dims);

PyObject *api_from_pointer(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_empty(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_zeros(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_ones(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_full(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_iota(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_diagonal(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_hash(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_astype(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

PyObject *api_moments(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_minmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_norm(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_argmin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_argmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_from_pointer[];
extern char const doc_empty[];
extern char const doc_zeros[];
extern char const doc_ones[];
extern char const doc_full[];
extern char const doc_iota[];
extern char const doc_diagonal[];
extern char const doc_hash[];
extern char const doc_astype[];

extern char const doc_reduce_moments[];
extern char const doc_reduce_minmax[];
extern char const doc_reduce_sum[];
extern char const doc_reduce_norm[];
extern char const doc_reduce_min[];
extern char const doc_reduce_max[];
extern char const doc_reduce_argmin[];
extern char const doc_reduce_argmax[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_TENSOR_H
