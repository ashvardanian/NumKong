/**
 *  @brief Matrix multiplication and symmetric operations for NumKong Python bindings.
 *  @file python/matrix.h
 *  @author Ash Vardanian
 *  @date February 20, 2026
 *
 *  Declares the PackedMatrix type and API functions for packed/symmetric
 *  cross operations used by the Python module.
 */
#ifndef NK_PYTHON_MATRIX_H
#define NK_PYTHON_MATRIX_H

#include "numkong.h"

/*  Row-tile sizes for OpenMP parallelization of batch operations.
 *  Packed kernels use 2×2 tile blocking (32 rows), so 64 rows = 2 tile blocks per chunk.
 *  Symmetric kernels use 16-row tiles, so 32 rows = 2 tile blocks per chunk. */
#define NK_PARALLEL_PACKED_TILE    64
#define NK_PARALLEL_SYMMETRIC_TILE 32

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Pre-packed matrix optimized for matrix multiplication or set distances.
 *
 *  Stores matrix data in a hardware-optimized layout (e.g., for AMX, AVX-512).
 *  Created via `nk.dots_pack()` or `nk.hammings_pack()` and used with
 *  packed batch APIs (`nk.*_packed()`) or the `@` operator for dot products.
 */
typedef struct PackedMatrix {
    PyObject_HEAD
    /** Packed dtype (bf16, i8, f32, etc.). */
    nk_dtype_t dtype;
    /** Number of rows in original matrix (width). */
    nk_size_t width;
    /** Number of columns in original matrix (depth). */
    nk_size_t depth;
    /** Variable-length packed data. */
    char start[];
} PackedMatrix;

/** @brief PackedMatrix Python type object.  */
extern PyTypeObject PackedMatrixType;

/** @brief Pack a matrix into hardware-optimized layout for dot-product matmul. */
PyObject *api_dots_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Matrix multiplication with a pre-packed B matrix. */
PyObject *api_dots_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Pack a matrix into hardware-optimized layout for Hamming distance. */
PyObject *api_hammings_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Hamming distance computation with a pre-packed B matrix. */
PyObject *api_hammings_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Jaccard distance computation with a pre-packed B matrix. */
PyObject *api_jaccards_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Angular distance computation with a pre-packed B matrix. */
PyObject *api_angulars_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Euclidean distance computation with a pre-packed B matrix. */
PyObject *api_euclideans_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs dot products within a single matrix. */
PyObject *api_dots_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs Hamming distances within a single matrix. */
PyObject *api_hammings_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs Jaccard distances within a single matrix. */
PyObject *api_jaccards_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs angular distances within a single matrix. */
PyObject *api_angulars_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs Euclidean distances within a single matrix. */
PyObject *api_euclideans_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_dots_pack[];
extern char const doc_dots_packed[];
extern char const doc_hammings_pack[];
extern char const doc_hammings_packed[];
extern char const doc_jaccards_packed[];
extern char const doc_angulars_packed[];
extern char const doc_euclideans_packed[];
extern char const doc_dots_symmetric[];
extern char const doc_hammings_symmetric[];
extern char const doc_jaccards_symmetric[];
extern char const doc_angulars_symmetric[];
extern char const doc_euclideans_symmetric[];

/**
 *  @brief Tensor @ PackedMatrix operator implementation.
 *  Used by Tensor's nb_matrix_multiply slot.
 */
PyObject *Tensor_matmul(PyObject *self, PyObject *other);

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_MATRIX_H
