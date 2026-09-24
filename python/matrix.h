/**
 *  @file python/matrix.h
 *  @author Ash Vardanian
 *  @date February 20, 2026
 *  @brief Matrix multiplication and symmetric operations for NumKong Python bindings.
 *
 *  Declares the PackedMatrix type and API functions for packed/symmetric cross operations used by
 *  the Python module.
 */
#ifndef NK_PYTHON_MATRIX_H
#define NK_PYTHON_MATRIX_H

#include "numkong.h"

/** Below this many multiply-accumulates an auto-threaded op stays serial: waking the thread pool
 *  dominates small products. Explicit dots_packed(..., threads=N) / cdist(..., threads=N) paths are
 *  unaffected — they honor the caller's request as given; only auto-deciders and the `@` operator
 *  look at it. */
#define NK_PARALLEL_MIN_MACS ((nk_size_t)1 << 20)

/**
 *  @brief Shared parallelism policy: is a @p work_units-sized op worth threading across @p threads?
 *
 *  The single home for the "big enough to wake the pool?" decision used by auto-deciders like the
 *  `@` operator. Returns true only when more than one thread is available and the work clears the
 *  pool wake-up floor, while call sites passing an explicit `threads=N` honor it literally and
 *  never consult this.
 *
 *  @param[in] work_units Problem size in multiply-accumulates (or an equivalent work proxy).
 *  @param[in] threads Resolved thread count the pool would use.
 *  @return Non-zero when the op should run in parallel, zero to stay serial.
 */
static inline int nk_parallel_worthwhile(nk_size_t work_units, nk_size_t threads) {
    return threads > 1 && work_units >= NK_PARALLEL_MIN_MACS;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Pre-packed matrix optimized for matrix multiplication or set distances.
 *
 *  Stores matrix data in a hardware-optimized layout, e.g. for AMX, AVX-512. Created via
 *  `nk.dots_pack()` or `nk.hammings_pack()` and used with packed batch APIs, `nk.*_packed()`, or
 *  the `@` operator for dot products.
 */
typedef struct PackedMatrix {
    PyObject_HEAD

    /** Packed dtype, bf16, i8, f32, etc. */
    nk_dtype_t dtype;

    /** Number of rows in original matrix, the width. */
    nk_size_t width;

    /** Number of columns in original matrix, the depth. */
    nk_size_t depth;

    /** Variable-length packed data. */
    char start[];
} PackedMatrix;

/** PackedMatrix Python type object. */
extern PyTypeObject PackedMatrixType;

/** Pack a matrix into hardware-optimized layout for dot-product matmul. */
PyObject *api_dots_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Matrix multiplication with a pre-packed B matrix. */
PyObject *api_dots_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Pack a matrix into hardware-optimized layout for Hamming distance. */
PyObject *api_hammings_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Hamming distance computation with a pre-packed B matrix. */
PyObject *api_hammings_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Jaccard distance computation with a pre-packed B matrix. */
PyObject *api_jaccards_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Angular distance computation with a pre-packed B matrix. */
PyObject *api_angulars_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Euclidean distance computation with a pre-packed B matrix. */
PyObject *api_euclideans_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** All-pairs dot products within a single matrix. */
PyObject *api_dots_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** All-pairs Hamming distances within a single matrix. */
PyObject *api_hammings_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** All-pairs Jaccard distances within a single matrix. */
PyObject *api_jaccards_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** All-pairs angular distances within a single matrix. */
PyObject *api_angulars_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** All-pairs Euclidean distances within a single matrix. */
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

/** Tensor @ PackedMatrix operator implementation, used by Tensor's nb_matrix_multiply slot. */
PyObject *Tensor_matmul(PyObject *self, PyObject *other);

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_MATRIX_H
