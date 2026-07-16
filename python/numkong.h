/**
 *  @brief NumKong Python Bindings.
 *  @file python/numkong.h
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *
 *  This header provides common data types, buffer protocol helpers, and
 *  dtype conversion utilities used across the NumKong Python extension modules.
 */
#ifndef NK_PYTHON_NUMKONG_H
#define NK_PYTHON_NUMKONG_H

#include <math.h>
#include <string.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numkong/numkong.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Parsed 1D/2D tensor argument from Python buffer protocol.
 *
 *  This structure holds the essential information extracted from a Python
 *  object that supports the buffer protocol (NumPy arrays, PyTorch tensors, etc.)
 *  Only used for distance/metric APIs that expect 1D or 2D inputs.
 */
typedef struct MatrixOrVectorView {
    /** Pointer to the first element. */
    char *data;
    /** Vector size (1D) or column count (2D). */
    size_t cols;
    /** Number of vectors (1 for 1D, num rows for 2D). */
    size_t rows;
    /** Stride between rows in bytes (0 for 1D). */
    size_t row_stride;
    /** Number of dimensions (1 or 2). */
    int rank;
    /** Logical dtype. */
    nk_dtype_t dtype;
} MatrixOrVectorView;

/**
 *  @brief Lightweight view for stride-aware operations.
 *
 *  Used by impl_reduce_* functions to traverse N-dimensional tensors
 *  with arbitrary strides.
 */
typedef struct TensorView {
    /** Logical dtype. */
    nk_dtype_t dtype;
    /** Number of dimensions. */
    size_t rank;
    /** Shape array (borrowed pointer). */
    Py_ssize_t const *shape;
    /** Strides array in bytes (borrowed pointer). */
    Py_ssize_t const *strides;
    /** Data pointer. */
    char *data;
} TensorView;

/** @brief Metadata for a single dtype.  */
typedef struct {
    /** Logical dtype enum value. */
    nk_dtype_t dtype;
    /** Human-readable name (e.g., "float32"). */
    char const *name;
    /** Python buffer protocol format string. */
    char const *pybuffer_typestr;
    /** NumPy array interface typestr. */
    char const *numpy_typestr;
    /** Size in bytes per element. */
    size_t item_size;
} nk_dtype_conversion_info_t;

/**
 *  @brief Backing storage for shape/strides when synthesizing a Py_buffer from __array_interface__.
 */
typedef struct {
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    Py_ssize_t strides[NK_TENSOR_MAX_RANK];
} nk_buffer_backing_t;

/** @brief Global dtype metadata table. */
extern nk_dtype_conversion_info_t const nk_dtype_conversion_infos[];
extern size_t const nk_dtype_table_size;

/**
 *  @brief Look up metadata for a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Pointer to metadata, or NULL if not found.
 */
nk_dtype_conversion_info_t const *nk_dtype_conversion_info(nk_dtype_t dtype);

/**
 *  @brief Get byte size of a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Size in bytes, or 0 if unknown.
 */
size_t nk_dtype_bytes_per_value(nk_dtype_t dtype);

/**
 *  @brief Get human-readable name of a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Name string (e.g., "float32"), or "unknown" if not found.
 */
char const *nk_dtype_name(nk_dtype_t dtype);

/**
 *  @brief Get NumPy array interface typestr for a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Typestr (e.g., "<f4"), or "|V1" if not found.
 */
char const *nk_dtype_to_numpy_typestr(nk_dtype_t dtype);

/**
 *  @brief Get Python buffer protocol format string for a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Format string (e.g., "f"), or "unknown" if not found.
 */
char const *nk_dtype_to_pybuffer_typestr(nk_dtype_t dtype);

/** @brief Promotes two dtypes to a common type following NumPy-like rules. */
nk_dtype_t nk_dtype_promote(nk_dtype_t a, nk_dtype_t b);

/**
 *  @brief Convert Python-style dtype string to logical dtype.
 *
 *  Handles NumPy format strings, Python struct format characters, and
 *  NumKong-specific names like "bfloat16" and "e4m3".
 *
 *  @param[in] name Format string from buffer protocol or user input.
 *  @return Logical dtype, or nk_dtype_unknown_k if not recognized.
 *  @see https://docs.python.org/3/library/struct.html#format-characters
 *  @see https://numpy.org/doc/stable/reference/arrays.interface.html
 */
nk_dtype_t py_string_to_nk_dtype(char const *name, Py_ssize_t len);

/**
 *  @brief Convert a Python object (type object or string) to logical dtype.
 *
 *  Accepts NumKong scalar type objects (e.g., nk.bfloat16) for O(1) pointer
 *  comparison, or falls back to string parsing via py_string_to_nk_dtype().
 *
 *  @param[in] obj Python type object or string.
 *  @return Logical dtype, or nk_dtype_unknown_k if not recognized.
 */
nk_dtype_t py_object_to_nk_dtype(PyObject *obj);

/**
 *  @brief Resolve dtype from a Py_buffer, preferring the Tensor's dtype
 *  over the PEP 3118 format string (which may be a placeholder for exotic types).
 */
nk_dtype_t resolve_nk_dtype_in_py_buffer(Py_buffer const *buffer);

/**
 *  @brief Convert metric name string to kernel kind.
 *  @param[in] name Metric name (e.g., "l2", "dot", "angular").
 *  @param[in] len Length of the name string.
 *  @return Kernel kind, or nk_kernel_unknown_k if not recognized.
 */
nk_kernel_kind_t py_string_to_nk_kernel_kind(char const *name, Py_ssize_t len);

/**
 *  @brief Check string equality.
 *  @param[in] a First string.
 *  @param[in] b Second string.
 *  @return 1 if equal, 0 otherwise.
 */
int same_string(char const *a, char const *b);

/**
 *  @brief Check string equality with known lengths (memcmp-based).
 *  @param[in] input Input string.
 *  @param[in] input_len Length of input string.
 *  @param[in] literal Literal string to compare against.
 *  @param[in] literal_len Length of literal string.
 *  @return 1 if equal, 0 otherwise.
 */
int same_string_n(char const *input, Py_ssize_t input_len, char const *literal, Py_ssize_t literal_len);

/**
 *  @brief Check if a metric is commutative.
 *  @param[in] kind Kernel kind.
 *  @return 1 if commutative, 0 otherwise.
 */
int nk_kernel_is_commutative(nk_kernel_kind_t kind);

/**
 *  @brief Convert a scalar buffer to the appropriate Python number type.
 */
PyObject *nk_scalar_buffer_to_py_number(nk_scalar_buffer_t const *buf, nk_dtype_t dtype);

/**
 *  @brief Store a Python number (float, int, or complex) into a scalar buffer.
 *  @return 1 on success, 0 on error (with Python exception set).
 */
int py_number_to_nk_scalar_buffer(PyObject *obj, nk_scalar_buffer_t *buf, nk_dtype_t dtype);

/**
 *  @brief Write a scalar buffer result (including complex) to a numpy output array element.
 *  @return 1 on success, 0 on error.
 */
int nk_scalar_buffer_export(nk_scalar_buffer_t const *source, nk_dtype_t source_dtype, void *target,
                            nk_dtype_t target_dtype);

/**
 *  @brief Acquire a Py_buffer, falling back to __array_interface__ if needed.
 *
 *  Tries PyObject_GetBuffer first. If that fails, reads __array_interface__
 *  and synthesizes a Py_buffer with shape/strides pointing into @p backing.
 *  When the fallback path is taken, buffer->obj is NULL so PyBuffer_Release
 *  is a no-op.
 *
 *  @param[in]  obj     Python object.
 *  @param[out] buffer  Output Py_buffer.
 *  @param[in]  flags   PyBUF_* flags for PyObject_GetBuffer.
 *  @param[out] backing Storage for shape/strides (used only on fallback path).
 *  @return 1 on success, 0 on failure (with Python exception set).
 */
int nk_get_buffer(PyObject *obj, Py_buffer *buffer, int flags, nk_buffer_backing_t *backing);

/**
 *  @brief Parse a Python tensor object into MatrixOrVectorView.
 *
 *  Extracts buffer information from any Python object supporting the buffer
 *  protocol or __array_interface__. Validates that the tensor is 1D or 2D
 *  with contiguous rows.
 *
 *  @param[in] tensor Python object supporting buffer protocol.
 *  @param[out] buffer Output Py_buffer (caller must release with PyBuffer_Release).
 *  @param[out] parsed Output MatrixOrVectorView with extracted metadata.
 *  @param[out] backing Backing storage for shape/strides.
 *  @return 1 on success, 0 on failure (Python exception set).
 */
int parse_tensor(PyObject *tensor, Py_buffer *buffer, MatrixOrVectorView *parsed, nk_buffer_backing_t *backing,
                 nk_dtype_t dtype_hint);

/**
 *  @brief Build a TensorView from any buffer-protocol object.
 *
 *  N-dimensional sibling of parse_tensor (which is limited to 1D/2D).
 *  Caller must call PyBuffer_Release(buffer) when done with the view.
 *
 *  @param[in]  obj    Python object exposing buffer protocol or __array_interface__.
 *  @param[out] buffer Output Py_buffer (caller must release with PyBuffer_Release).
 *  @param[out] view   Output TensorView with borrowed pointers into buffer.
 *  @param[out] backing Backing storage for shape/strides (used by __array_interface__ fallback).
 *  @param[in]  dtype_hint Override dtype; nk_dtype_unknown_k to infer from buffer format.
 *  @return 1 on success, 0 on failure (Python exception set).
 */
int parse_tensor_nd(PyObject *obj, Py_buffer *buffer, TensorView *view, nk_buffer_backing_t *backing,
                    nk_dtype_t dtype_hint);

/**
 *  @brief Check if a Python object is a numeric scalar.
 *  @param[in] obj Python object.
 *  @return 1 if int or float, 0 otherwise.
 */
int py_object_is_scalar(PyObject *obj);

/**
 *  @brief Extract f64 value from a Python numeric scalar.
 *  @param[in] obj Python int or float.
 *  @param[out] value Output f64 value.
 *  @return 1 on success, 0 on failure.
 */
int py_number_to_f64(PyObject *obj, nk_f64_t *value);

PyObject *api_enable_capability(PyObject *self, PyObject *cap_name_obj);
PyObject *api_disable_capability(PyObject *self, PyObject *cap_name_obj);
PyObject *api_get_capabilities_detected(PyObject *self);
PyObject *api_get_capabilities_compiled(PyObject *self);
PyObject *api_get_capabilities_available(PyObject *self);
PyObject *api_get_capabilities_enabled(PyObject *self);

extern char const doc_enable_capability[];
extern char const doc_disable_capability[];
extern char const doc_get_capabilities_detected[];
extern char const doc_get_capabilities_compiled[];
extern char const doc_get_capabilities_available[];
extern char const doc_get_capabilities_enabled[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_NUMKONG_H
