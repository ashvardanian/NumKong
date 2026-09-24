/**
 *  @brief Register NumKong scalars as NumPy custom dtypes via runtime capsule API.
 *  @file python/numpy_interop.c
 *  @author Ash Vardanian
 *  @date March 14, 2026
 *
 *  Registers custom NumPy dtypes for bfloat16, float16, float8_e4m3, float8_e5m2,
 *  float6_e2m3, and float6_e3m2 without including any numpy headers. Instead, we
 *  define minimal ABI-compatible struct layouts and extract the C API function pointers
 *  at runtime from NumPy's capsule.
 *
 *  NumPy 2.x uses two different descriptor layouts:
 *  - PyArray_DescrProto (1.x-compatible layout) — input to RegisterDataType.
 *  - PyArray_Descr (2.x layout) — returned by DescrFromType, input to RegisterCastFunc.
 *  We define both to avoid struct layout mismatches that caused crashes on NumPy 2.4.
 */

#include "numpy_interop.h"
#include "types.h"

#include <string.h>
#include <stdint.h>

typedef Py_intptr_t npy_intp;
typedef Py_hash_t npy_hash_t;

/** NumPy type-number constants. */
enum {
    NK_NPY_BOOL = 0,
    NK_NPY_BYTE = 1,
    NK_NPY_UBYTE = 2,
    NK_NPY_SHORT = 3,
    NK_NPY_USHORT = 4,
    NK_NPY_INT = 5,
    NK_NPY_UINT = 6,
    NK_NPY_LONG = 7,
    NK_NPY_ULONG = 8,
    NK_NPY_LONGLONG = 9,
    NK_NPY_ULONGLONG = 10,
    NK_NPY_FLOAT = 11,
    NK_NPY_DOUBLE = 12,
    NK_NPY_LONGDOUBLE = 13,
    NK_NPY_CFLOAT = 14,
    NK_NPY_CDOUBLE = 15,
    NK_NPY_CLONGDOUBLE = 16,
    NK_NPY_OBJECT = 17,
    NK_NPY_STRING = 18,
    NK_NPY_UNICODE = 19,
    NK_NPY_VOID = 20,
    NK_NPY_NTYPES_ABI_COMPATIBLE = 21,
    NK_NPY_USERDEF = 256,
    NK_NPY_NOTYPE = -1,
    NK_NPY_SAFE_CASTING = 2,
    NK_NPY_NSORTS = 3,
};

/**
 *  @brief ABI-compatible layout for PyArray_ArrFuncs.
 *
 *  CRITICAL: The struct starts with `cast[21]` (NPY_NTYPES_ABI_COMPATIBLE = 21)
 *  function pointers. Omitting this shifts every subsequent field by 168 bytes
 *  on 64-bit, causing memory corruption.
 */
typedef struct {
    void (*cast[NK_NPY_NTYPES_ABI_COMPATIBLE])(void *, void *, npy_intp, void *, void *);
    PyObject *(*getitem)(void *, void *);
    int (*setitem)(PyObject *, void *, void *);
    void (*copyswapn)(void *, npy_intp, void *, npy_intp, npy_intp, int, void *);
    void (*copyswap)(void *, void *, int, void *);
    int (*compare)(void const *, void const *, void *);
    int (*argmax)(void *, npy_intp, npy_intp *, void *);
    void (*dotfunc)(void *, npy_intp, void *, npy_intp, void *, npy_intp, void *);
    int (*scanfunc)(FILE *, void *, void *, void *);
    int (*fromstr)(char *, void *, char **, void *);
    int (*nonzero)(void *, void *);
    int (*fill)(void *, npy_intp, void *);
    int (*fillwithscalar)(void *, npy_intp, void *, void *);
    int (*sort[NK_NPY_NSORTS])(void *, npy_intp, void *);
    int (*argsort[NK_NPY_NSORTS])(void *, npy_intp *, npy_intp, void *);
    PyObject *castdict;
    void *scalarkind;
    int **cancastscalarkindto;
    int *cancastto;
    void *_unused1;
    void *_unused2;
    void *_unused3;
    int (*argmin)(void *, npy_intp, npy_intp *, void *);
} nk_PyArray_ArrFuncs;

/**
 *  @brief PyArray_DescrProto layout (1.x-compatible) — input to RegisterDataType.
 *
 *  Uses `int elsize`, `int alignment`, includes `subarray`, `fields`, `names`,
 *  `f` (ArrFuncs pointer), `metadata`, `c_metadata`, `hash`.
 */
typedef struct {
    PyObject_HEAD PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    void *subarray;
    PyObject *fields;
    PyObject *names;
    nk_PyArray_ArrFuncs *f;
    PyObject *metadata;
    void *c_metadata;
    npy_hash_t hash;
} nk_PyArray_DescrProto;

/**
 *  @brief PyArray_Descr layout (2.x) — returned by DescrFromType, input to RegisterCastFunc.
 *
 *  Uses `npy_intp elsize`, `npy_intp alignment`, no `f` pointer. Much smaller.
 *  We only read `type_num` from this; remaining fields are opaque.
 */
typedef struct {
    PyObject_HEAD PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char _former_flags;
    int type_num;
} nk_PyArray_Descr;

typedef nk_PyArray_Descr *(*nk_DescrFromType_t)(int typenum);
typedef int (*nk_RegisterDataType_t)(nk_PyArray_DescrProto *descr);
typedef int (*nk_RegisterCastFunc_t)(nk_PyArray_Descr *descr, int totype,
                                     void (*castfunc)(void *, void *, npy_intp, void *, void *));
typedef int (*nk_RegisterCanCast_t)(nk_PyArray_Descr *descr, int totype, int scalar);

enum {
    NK_NPY_API_DescrFromType = 45,
    NK_NPY_API_RegisterDataType = 192,
    NK_NPY_API_RegisterCastFunc = 193,
    NK_NPY_API_RegisterCanCast = 194,
};

static void **nk_numpy_api = NULL;

static int nk_load_numpy_api(void) {
    if (nk_numpy_api) return 0;

    PyObject *mod = PyImport_ImportModule("numpy._core._multiarray_umath");
    if (!mod) {
        PyErr_Clear();
        return -1;
    }

    PyObject *capsule = PyObject_GetAttrString(mod, "_ARRAY_API");
    Py_DECREF(mod);
    if (!capsule) {
        PyErr_Clear();
        return -1;
    }

    nk_numpy_api = (void **)PyCapsule_GetPointer(capsule, NULL);
    Py_DECREF(capsule);
    if (!nk_numpy_api) {
        PyErr_Clear();
        return -1;
    }
    return 0;
}

static inline nk_PyArray_Descr *nk_descr_from_type(int typenum) {
    return ((nk_DescrFromType_t)nk_numpy_api[NK_NPY_API_DescrFromType])(typenum);
}
static inline int nk_register_data_type(nk_PyArray_DescrProto *descr) {
    return ((nk_RegisterDataType_t)nk_numpy_api[NK_NPY_API_RegisterDataType])(descr);
}
static inline int nk_register_cast_func(nk_PyArray_Descr *descr, int totype,
                                        void (*castfunc)(void *, void *, npy_intp, void *, void *)) {
    return ((nk_RegisterCastFunc_t)nk_numpy_api[NK_NPY_API_RegisterCastFunc])(descr, totype, castfunc);
}
static inline int nk_register_can_cast(nk_PyArray_Descr *descr, int totype, int scalar) {
    return ((nk_RegisterCanCast_t)nk_numpy_api[NK_NPY_API_RegisterCanCast])(descr, totype, scalar);
}

/** @brief DType conversion ops — one per custom dtype. */
typedef struct {
    size_t elem_size;
    void (*to_f32)(void const *src, nk_f32_t *dst);
    void (*from_f32)(nk_f32_t const *src, void *dst);
} nk_np_dtype_ops_t;

typedef void (*nk_to_f32_fn_t)(void const *, nk_f32_t *);
typedef void (*nk_from_f32_fn_t)(nk_f32_t const *, void *);

static nk_np_dtype_ops_t const nk_ops_bf16 = {sizeof(nk_bf16_t), (nk_to_f32_fn_t)nk_bf16_to_f32,
                                              (nk_from_f32_fn_t)nk_f32_to_bf16};
static nk_np_dtype_ops_t const nk_ops_f16 = {sizeof(nk_f16_t), (nk_to_f32_fn_t)nk_f16_to_f32,
                                             (nk_from_f32_fn_t)nk_f32_to_f16};
static nk_np_dtype_ops_t const nk_ops_e4m3 = {sizeof(nk_e4m3_t), (nk_to_f32_fn_t)nk_e4m3_to_f32,
                                              (nk_from_f32_fn_t)nk_f32_to_e4m3};
static nk_np_dtype_ops_t const nk_ops_e5m2 = {sizeof(nk_e5m2_t), (nk_to_f32_fn_t)nk_e5m2_to_f32,
                                              (nk_from_f32_fn_t)nk_f32_to_e5m2};
static nk_np_dtype_ops_t const nk_ops_e2m3 = {sizeof(nk_e2m3_t), (nk_to_f32_fn_t)nk_e2m3_to_f32,
                                              (nk_from_f32_fn_t)nk_f32_to_e2m3};
static nk_np_dtype_ops_t const nk_ops_e3m2 = {sizeof(nk_e3m2_t), (nk_to_f32_fn_t)nk_e3m2_to_f32,
                                              (nk_from_f32_fn_t)nk_f32_to_e3m2};

/** @brief Generic ArrFuncs implementations — parameterized by ops. */

static PyObject *nk_np_getitem_generic(void *data, nk_np_dtype_ops_t const *ops) {
    nk_f32_t f32;
    ops->to_f32(data, &f32);
    return PyFloat_FromDouble((double)f32);
}

static int nk_np_setitem_generic(PyObject *obj, void *data, nk_np_dtype_ops_t const *ops) {
    double val = PyFloat_AsDouble(obj);
    if (val == -1.0 && PyErr_Occurred()) return -1;
    nk_f32_t f32 = (nk_f32_t)val;
    ops->from_f32(&f32, data);
    return 0;
}

static void nk_np_copyswap_generic(void *dst, void *src, size_t elem_size) {
    if (src) memcpy(dst, src, elem_size);
}

static void nk_np_copyswapn_generic(void *dst, npy_intp dstride, void *src, npy_intp sstride, npy_intp n,
                                    size_t elem_size) {
    if (!src) return;
    if (dstride == (npy_intp)elem_size && sstride == (npy_intp)elem_size) { memcpy(dst, src, (size_t)n * elem_size); }
    else {
        char *d = (char *)dst, *s = (char *)src;
        for (npy_intp i = 0; i < n; i++, d += dstride, s += sstride) memcpy(d, s, elem_size);
    }
}

static int nk_np_nonzero_generic(void *data, nk_np_dtype_ops_t const *ops) {
    nk_f32_t f32;
    ops->to_f32(data, &f32);
    return f32 != 0.0f;
}

// Per-dtype trampolines: NumPy's ArrFuncs/cast signatures don't carry user-data,
// so each dtype needs thin wrappers that close over the ops struct.
// 5 ArrFuncs x 6 dtypes = 30, plus 4 cast directions x 6 dtypes = 24 using nk_cast.

// clang-format off
static PyObject *nk_np_getitem_bf16(void *d, void *a) { nk_unused_(a); return nk_np_getitem_generic(d, &nk_ops_bf16); }
static PyObject *nk_np_getitem_f16(void *d, void *a)  { nk_unused_(a); return nk_np_getitem_generic(d, &nk_ops_f16); }
static PyObject *nk_np_getitem_e4m3(void *d, void *a) { nk_unused_(a); return nk_np_getitem_generic(d, &nk_ops_e4m3); }
static PyObject *nk_np_getitem_e5m2(void *d, void *a) { nk_unused_(a); return nk_np_getitem_generic(d, &nk_ops_e5m2); }
static PyObject *nk_np_getitem_e2m3(void *d, void *a) { nk_unused_(a); return nk_np_getitem_generic(d, &nk_ops_e2m3); }
static PyObject *nk_np_getitem_e3m2(void *d, void *a) { nk_unused_(a); return nk_np_getitem_generic(d, &nk_ops_e3m2); }

static int nk_np_setitem_bf16(PyObject *o, void *d, void *a) { nk_unused_(a); return nk_np_setitem_generic(o, d, &nk_ops_bf16); }
static int nk_np_setitem_f16(PyObject *o, void *d, void *a)  { nk_unused_(a); return nk_np_setitem_generic(o, d, &nk_ops_f16); }
static int nk_np_setitem_e4m3(PyObject *o, void *d, void *a) { nk_unused_(a); return nk_np_setitem_generic(o, d, &nk_ops_e4m3); }
static int nk_np_setitem_e5m2(PyObject *o, void *d, void *a) { nk_unused_(a); return nk_np_setitem_generic(o, d, &nk_ops_e5m2); }
static int nk_np_setitem_e2m3(PyObject *o, void *d, void *a) { nk_unused_(a); return nk_np_setitem_generic(o, d, &nk_ops_e2m3); }
static int nk_np_setitem_e3m2(PyObject *o, void *d, void *a) { nk_unused_(a); return nk_np_setitem_generic(o, d, &nk_ops_e3m2); }

static void nk_np_copyswap_bf16(void *d, void *s, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswap_generic(d, s, nk_ops_bf16.elem_size); }
static void nk_np_copyswap_f16(void *d, void *s, int w, void *a)  { nk_unused_(w); nk_unused_(a); nk_np_copyswap_generic(d, s, nk_ops_f16.elem_size); }
static void nk_np_copyswap_e4m3(void *d, void *s, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswap_generic(d, s, nk_ops_e4m3.elem_size); }
static void nk_np_copyswap_e5m2(void *d, void *s, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswap_generic(d, s, nk_ops_e5m2.elem_size); }
static void nk_np_copyswap_e2m3(void *d, void *s, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswap_generic(d, s, nk_ops_e2m3.elem_size); }
static void nk_np_copyswap_e3m2(void *d, void *s, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswap_generic(d, s, nk_ops_e3m2.elem_size); }

static void nk_np_copyswapn_bf16(void *d, npy_intp ds, void *s, npy_intp ss, npy_intp n, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswapn_generic(d, ds, s, ss, n, nk_ops_bf16.elem_size); }
static void nk_np_copyswapn_f16(void *d, npy_intp ds, void *s, npy_intp ss, npy_intp n, int w, void *a)  { nk_unused_(w); nk_unused_(a); nk_np_copyswapn_generic(d, ds, s, ss, n, nk_ops_f16.elem_size); }
static void nk_np_copyswapn_e4m3(void *d, npy_intp ds, void *s, npy_intp ss, npy_intp n, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswapn_generic(d, ds, s, ss, n, nk_ops_e4m3.elem_size); }
static void nk_np_copyswapn_e5m2(void *d, npy_intp ds, void *s, npy_intp ss, npy_intp n, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswapn_generic(d, ds, s, ss, n, nk_ops_e5m2.elem_size); }
static void nk_np_copyswapn_e2m3(void *d, npy_intp ds, void *s, npy_intp ss, npy_intp n, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswapn_generic(d, ds, s, ss, n, nk_ops_e2m3.elem_size); }
static void nk_np_copyswapn_e3m2(void *d, npy_intp ds, void *s, npy_intp ss, npy_intp n, int w, void *a) { nk_unused_(w); nk_unused_(a); nk_np_copyswapn_generic(d, ds, s, ss, n, nk_ops_e3m2.elem_size); }

static int nk_np_nonzero_bf16(void *d, void *a) { nk_unused_(a); return nk_np_nonzero_generic(d, &nk_ops_bf16); }
static int nk_np_nonzero_f16(void *d, void *a)  { nk_unused_(a); return nk_np_nonzero_generic(d, &nk_ops_f16); }
static int nk_np_nonzero_e4m3(void *d, void *a) { nk_unused_(a); return nk_np_nonzero_generic(d, &nk_ops_e4m3); }
static int nk_np_nonzero_e5m2(void *d, void *a) { nk_unused_(a); return nk_np_nonzero_generic(d, &nk_ops_e5m2); }
static int nk_np_nonzero_e2m3(void *d, void *a) { nk_unused_(a); return nk_np_nonzero_generic(d, &nk_ops_e2m3); }
static int nk_np_nonzero_e3m2(void *d, void *a) { nk_unused_(a); return nk_np_nonzero_generic(d, &nk_ops_e3m2); }
// clang-format on

// clang-format off
static void nk_np_cast_bf16_to_f32(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_bf16_k, (nk_size_t)n, t, nk_f32_k); }
static void nk_np_cast_f16_to_f32(void *f, void *t, npy_intp n, void *fa, void *ta)  { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f16_k,  (nk_size_t)n, t, nk_f32_k); }
static void nk_np_cast_e4m3_to_f32(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e4m3_k, (nk_size_t)n, t, nk_f32_k); }
static void nk_np_cast_e5m2_to_f32(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e5m2_k, (nk_size_t)n, t, nk_f32_k); }
static void nk_np_cast_e2m3_to_f32(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e2m3_k, (nk_size_t)n, t, nk_f32_k); }
static void nk_np_cast_e3m2_to_f32(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e3m2_k, (nk_size_t)n, t, nk_f32_k); }

static void nk_np_cast_f32_to_bf16(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f32_k, (nk_size_t)n, t, nk_bf16_k); }
static void nk_np_cast_f32_to_f16(void *f, void *t, npy_intp n, void *fa, void *ta)  { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f32_k, (nk_size_t)n, t, nk_f16_k); }
static void nk_np_cast_f32_to_e4m3(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f32_k, (nk_size_t)n, t, nk_e4m3_k); }
static void nk_np_cast_f32_to_e5m2(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f32_k, (nk_size_t)n, t, nk_e5m2_k); }
static void nk_np_cast_f32_to_e2m3(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f32_k, (nk_size_t)n, t, nk_e2m3_k); }
static void nk_np_cast_f32_to_e3m2(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f32_k, (nk_size_t)n, t, nk_e3m2_k); }

static void nk_np_cast_bf16_to_f64(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_bf16_k, (nk_size_t)n, t, nk_f64_k); }
static void nk_np_cast_f16_to_f64(void *f, void *t, npy_intp n, void *fa, void *ta)  { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f16_k,  (nk_size_t)n, t, nk_f64_k); }
static void nk_np_cast_e4m3_to_f64(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e4m3_k, (nk_size_t)n, t, nk_f64_k); }
static void nk_np_cast_e5m2_to_f64(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e5m2_k, (nk_size_t)n, t, nk_f64_k); }
static void nk_np_cast_e2m3_to_f64(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e2m3_k, (nk_size_t)n, t, nk_f64_k); }
static void nk_np_cast_e3m2_to_f64(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_e3m2_k, (nk_size_t)n, t, nk_f64_k); }

static void nk_np_cast_f64_to_bf16(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f64_k, (nk_size_t)n, t, nk_bf16_k); }
static void nk_np_cast_f64_to_f16(void *f, void *t, npy_intp n, void *fa, void *ta)  { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f64_k, (nk_size_t)n, t, nk_f16_k); }
static void nk_np_cast_f64_to_e4m3(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f64_k, (nk_size_t)n, t, nk_e4m3_k); }
static void nk_np_cast_f64_to_e5m2(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f64_k, (nk_size_t)n, t, nk_e5m2_k); }
static void nk_np_cast_f64_to_e2m3(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f64_k, (nk_size_t)n, t, nk_e2m3_k); }
static void nk_np_cast_f64_to_e3m2(void *f, void *t, npy_intp n, void *fa, void *ta) { nk_unused_(fa); nk_unused_(ta); nk_cast(f, nk_f64_k, (nk_size_t)n, t, nk_e3m2_k); }
// clang-format on

/** @brief Initialize an ArrFuncs struct with the 5 required function pointers. */
static void nk_init_arrfuncs(nk_PyArray_ArrFuncs *af,                              //
                             PyObject *(*getitem)(void *, void *),                 //
                             int (*setitem)(PyObject *, void *, void *),           //
                             void (*copyswap)(void *, void *, int, void *),        //
                             void (*copyswapn)(void *, npy_intp, void *, npy_intp, //
                                               npy_intp, int, void *),             //
                             int (*nonzero)(void *, void *)) {
    memset(af, 0, sizeof(nk_PyArray_ArrFuncs));
    af->getitem = getitem;
    af->setitem = setitem;
    af->copyswap = copyswap;
    af->copyswapn = copyswapn;
    af->nonzero = nonzero;
}

/**
 *  @brief Initialize a DescrProto (1.x layout) for a custom dtype.
 *
 *  We borrow the ob_type (metaclass) from an existing NumPy descriptor
 *  so that NumPy recognizes our descriptor as a valid PyArray_Descr subclass.
 */
static void nk_init_proto(nk_PyArray_DescrProto *proto, nk_PyArray_ArrFuncs *af, PyTypeObject *scalar_type, char kind,
                          char type_char, int elsize) {
    memset(proto, 0, sizeof(nk_PyArray_DescrProto));

    nk_PyArray_Descr *f32_descr = nk_descr_from_type(NK_NPY_FLOAT);
    if (f32_descr) Py_SET_TYPE((PyObject *)proto, Py_TYPE((PyObject *)f32_descr));

    Py_SET_REFCNT((PyObject *)proto, 1);

    proto->typeobj = scalar_type;
    proto->kind = kind;
    proto->type = type_char;
    proto->byteorder = '=';
    proto->flags = 0;
    proto->type_num = 0;
    proto->elsize = elsize;
    proto->alignment = 1;
    proto->f = af;
    proto->hash = -1;
}

/** @brief Register cast functions for a custom dtype to/from float32 and float64. */
static int nk_register_casts(nk_PyArray_DescrProto *proto,                                    //
                             void (*cast_to_f32)(void *, void *, npy_intp, void *, void *),   //
                             void (*cast_from_f32)(void *, void *, npy_intp, void *, void *), //
                             void (*cast_to_f64)(void *, void *, npy_intp, void *, void *),   //
                             void (*cast_from_f64)(void *, void *, npy_intp, void *, void *)) {
    nk_PyArray_Descr *descr2x = nk_descr_from_type(proto->type_num);
    if (!descr2x) return -1;
    if (nk_register_cast_func(descr2x, NK_NPY_FLOAT, cast_to_f32) < 0) return -1;
    if (nk_register_cast_func(descr2x, NK_NPY_DOUBLE, cast_to_f64) < 0) return -1;
    nk_PyArray_Descr *f32d = nk_descr_from_type(NK_NPY_FLOAT);
    if (f32d && nk_register_cast_func(f32d, proto->type_num, cast_from_f32) < 0) return -1;
    nk_PyArray_Descr *f64d = nk_descr_from_type(NK_NPY_DOUBLE);
    if (f64d && nk_register_cast_func(f64d, proto->type_num, cast_from_f64) < 0) return -1;
    if (nk_register_can_cast(descr2x, NK_NPY_FLOAT, NK_NPY_SAFE_CASTING) < 0) return -1;
    if (nk_register_can_cast(descr2x, NK_NPY_DOUBLE, NK_NPY_SAFE_CASTING) < 0) return -1;
    return 0;
}

/** @brief Set the `.dtype` attribute on a scalar type to its registered NumPy descriptor. */
static int nk_set_dtype_attr(PyTypeObject *scalar_type, int type_num) {
    nk_PyArray_Descr *d = nk_descr_from_type(type_num);
    if (!d) return -1;
    PyObject *dict = scalar_type->tp_dict;
    if (!dict) {
        Py_DECREF((PyObject *)d);
        return -1;
    }
    int rc = PyDict_SetItemString(dict, "dtype", (PyObject *)d);
    Py_DECREF((PyObject *)d);
    PyType_Modified(scalar_type);
    return rc;
}

/** @brief Registration descriptor for one custom dtype. */
typedef struct {
    PyTypeObject *scalar_type;
    char kind, type_char;
    int elsize;
    PyObject *(*getitem)(void *, void *);
    int (*setitem)(PyObject *, void *, void *);
    void (*copyswap)(void *, void *, int, void *);
    void (*copyswapn)(void *, npy_intp, void *, npy_intp, npy_intp, int, void *);
    int (*nonzero)(void *, void *);
    void (*cast_to_f32)(void *, void *, npy_intp, void *, void *);
    void (*cast_from_f32)(void *, void *, npy_intp, void *, void *);
    void (*cast_to_f64)(void *, void *, npy_intp, void *, void *);
    void (*cast_from_f64)(void *, void *, npy_intp, void *, void *);
} nk_np_registration_t;

// clang-format off
static nk_np_registration_t const nk_np_registrations[] = {
    {&NkBFloat16Scalar_Type,  'V', 'k', 2, nk_np_getitem_bf16,  nk_np_setitem_bf16,  nk_np_copyswap_bf16,  nk_np_copyswapn_bf16,  nk_np_nonzero_bf16,  nk_np_cast_bf16_to_f32,  nk_np_cast_f32_to_bf16,  nk_np_cast_bf16_to_f64,  nk_np_cast_f64_to_bf16},
    {&NkFloat16Scalar_Type,   'V', 'j', 2, nk_np_getitem_f16,   nk_np_setitem_f16,   nk_np_copyswap_f16,   nk_np_copyswapn_f16,   nk_np_nonzero_f16,   nk_np_cast_f16_to_f32,   nk_np_cast_f32_to_f16,   nk_np_cast_f16_to_f64,   nk_np_cast_f64_to_f16},
    {&NkFloat8E4M3Scalar_Type,'V', 'w', 1, nk_np_getitem_e4m3,  nk_np_setitem_e4m3,  nk_np_copyswap_e4m3,  nk_np_copyswapn_e4m3,  nk_np_nonzero_e4m3,  nk_np_cast_e4m3_to_f32,  nk_np_cast_f32_to_e4m3,  nk_np_cast_e4m3_to_f64,  nk_np_cast_f64_to_e4m3},
    {&NkFloat8E5M2Scalar_Type,'V', 'x', 1, nk_np_getitem_e5m2,  nk_np_setitem_e5m2,  nk_np_copyswap_e5m2,  nk_np_copyswapn_e5m2,  nk_np_nonzero_e5m2,  nk_np_cast_e5m2_to_f32,  nk_np_cast_f32_to_e5m2,  nk_np_cast_e5m2_to_f64,  nk_np_cast_f64_to_e5m2},
    {&NkFloat6E2M3Scalar_Type,'V', 'y', 1, nk_np_getitem_e2m3,  nk_np_setitem_e2m3,  nk_np_copyswap_e2m3,  nk_np_copyswapn_e2m3,  nk_np_nonzero_e2m3,  nk_np_cast_e2m3_to_f32,  nk_np_cast_f32_to_e2m3,  nk_np_cast_e2m3_to_f64,  nk_np_cast_f64_to_e2m3},
    {&NkFloat6E3M2Scalar_Type,'V', 'z', 1, nk_np_getitem_e3m2,  nk_np_setitem_e3m2,  nk_np_copyswap_e3m2,  nk_np_copyswapn_e3m2,  nk_np_nonzero_e3m2,  nk_np_cast_e3m2_to_f32,  nk_np_cast_f32_to_e3m2,  nk_np_cast_e3m2_to_f64,  nk_np_cast_f64_to_e3m2},
};
// clang-format on

enum { nk_np_num_dtypes_k = sizeof(nk_np_registrations) / sizeof(nk_np_registrations[0]) };

static nk_PyArray_ArrFuncs nk_arrfuncs[nk_np_num_dtypes_k];
static nk_PyArray_DescrProto nk_protos[nk_np_num_dtypes_k];

int nk_register_numpy_dtypes(PyObject *module) {
    nk_unused_(module);

    if (nk_load_numpy_api() < 0) return 0;

    for (size_t i = 0; i < nk_np_num_dtypes_k; i++) {
        nk_np_registration_t const *reg = &nk_np_registrations[i];
        nk_init_arrfuncs(&nk_arrfuncs[i], reg->getitem, reg->setitem, reg->copyswap, reg->copyswapn, reg->nonzero);
        nk_init_proto(&nk_protos[i], &nk_arrfuncs[i], reg->scalar_type, reg->kind, reg->type_char, reg->elsize);
        if (nk_register_data_type(&nk_protos[i]) < 0) return -1;
        if (nk_register_casts(&nk_protos[i], reg->cast_to_f32, reg->cast_from_f32, reg->cast_to_f64,
                              reg->cast_from_f64) < 0)
            return -1;
        if (nk_set_dtype_attr(reg->scalar_type, nk_protos[i].type_num) < 0) return -1;
    }

    return 0;
}
