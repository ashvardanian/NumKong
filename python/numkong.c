/**
 *  @brief Pure CPython bindings for NumKong.
 *  @file python/numkong.c
 *  @author Ash Vardanian
 *  @date January 1, 2023
 *
 *  @section Latency, Quality, and Arguments Parsing
 *
 *  The complexity of implementing high-quality CPython bindings is often underestimated.
 *  Do not rely on high-level wrappers like PyBind11 and NanoBind, and avoid SWIG-like
 *  toolchains. Most of them use expensive dynamic data structures to map callbacks to
 *  object or module properties, rather than relying on the CPython API. They are too
 *  slow for low-latency operations such as checking container lengths, handling vectors,
 *  or processing strings.
 *
 *  Once you work directly with the CPython API, there is substantial boilerplate code
 *  to write, and it is common to use `PyArg_ParseTupleAndKeywords` or `PyArg_ParseTuple`.
 *  Those functions parse format specifier strings at runtime, which @b cannot be fast
 *  by design. Moreover, they do not support the Python "Fast Calling Convention". In
 *  a typical scenario, a function is defined with `METH_VARARGS | METH_KEYWORDS` and
 *  has a signature like:
 *
 *  @code {.c}
 *      static PyObject* cdist(
 *          PyObject * self,
 *          PyObject * positional_args_tuple,
 *          PyObject * named_args_dict) {
 *          PyObject * a_obj, b_obj, metric_obj, out_obj, dtype_obj, out_dtype_obj, threads_obj;
 *          static char* names[] = {"a", "b", "metric", "threads", "dtype", "out_dtype", NULL};
 *          if (!PyArg_ParseTupleAndKeywords(
 *              positional_args_tuple, named_args_dict, "OO|s$Kss", names,
 *              &a_obj, &b_obj, &metric_str, &threads, &dtype_str, &out_dtype_str))
 *              return NULL;
 *          ...
 *  @endcode
 *
 *  This `cdist` example takes 2 positional, 1 positional or named, and 3 named-only arguments.
 *  An alternative with `METH_FASTCALL` uses a function signature like:
 *
 *  @code {.c}
 *     static PyObject* cdist(
 *          PyObject * self,
 *          PyObject * const * args_c_array,    //! C array of `args_count` pointers
 *          Py_ssize_t const positional_args_count,   //! The `args_c_array` may be larger than this
 *          PyObject * args_names_tuple) {      //! May be smaller than `args_count`
 *          Py_ssize_t args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
 *          Py_ssize_t args_count = positional_args_count + args_names_count;
 *          ...
 *  @endcode
 *
 *  The positional elements are easy to access in that C array, but parsing the named arguments is tricky.
 *  There are cases where a call is ill-formed and provides more positional arguments than expected.
 *
 *  @code {.py}
 *      cdist(a, b, "cos", "dos"):               //! positional_args_count == 4, args_names_count == 0
 *      cdist(a, b, "cos", metric="dos"):        //! positional_args_count == 3, args_names_count == 1
 *      cdist(a, b, metric="cos", metric="dos"): //! positional_args_count == 2, args_names_count == 2
 *  @endcode
 *
 *  If the same argument is provided twice, a @b `TypeError` is raised.
 *  If the argument is not found, a @b `KeyError` is raised.
 *
 *  https://ashvardanian.com/posts/discount-on-keyword-arguments-in-python/
 *
 *  @section Buffer Protocol and NumPy Compatibility
 *
 *  Most modern machine learning frameworks struggle with buffer protocol compatibility.
 *  At best, they provide zero-copy NumPy views of the underlying data, which introduces an
 *  unnecessary dependency on NumPy, an allocation for the wrapper, and constraints on the
 *  supported numeric types. This is a limitation because PyTorch and TensorFlow have richer
 *  type systems than NumPy.
 *
 *  A PyTorch `Tensor` cannot be converted to a `memoryview` object.
 *  Converting a `bf16` TensorFlow `Tensor` to a `memoryview` raises:
 *
 *      ! ValueError: cannot include dtype 'E' in a buffer
 *
 *  Moreover, the CPython and NumPy documentation diverge on format specifiers for the `typestr`
 *  and `format` data type descriptor strings, which makes development error-prone. NumKong is
 *  @b one of the few packages that attempts to provide interoperability.
 *
 *  https://numpy.org/doc/stable/reference/arrays.interface.html
 *  https://pearu.github.io/array_interface_pytorch.html
 *  https://github.com/pytorch/pytorch/issues/54138
 *  https://github.com/pybind/pybind11/issues/1908
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numkong/numkong.h>

#include "numkong.h"
#include "tensor.h"
#include "matrix.h"
#include "types.h"
#include "distance.h"
#include "each.h"
#include "mesh.h"
#include "maxsim.h"
#include "numpy_interop.h"
#include "dlpack_interop.h"

nk_capability_t static_capabilities = 0;

nk_dtype_conversion_info_t const nk_dtype_conversion_infos[] = {
    {nk_f64_k, "float64", "d", "<f8", sizeof(nk_f64_t)},
    {nk_f32_k, "float32", "f", "<f4", sizeof(nk_f32_t)},
    {nk_f16_k, "float16", "e", "<f2", sizeof(nk_f16_t)},
    {nk_bf16_k, "bfloat16", "bf16", "<V2", sizeof(nk_bf16_t)},
    {nk_e4m3_k, "e4m3", "e4m3", "|V1", sizeof(nk_e4m3_t)},
    {nk_e5m2_k, "e5m2", "e5m2", "|V1", sizeof(nk_e5m2_t)},
    {nk_e2m3_k, "e2m3", "e2m3", "|V1", sizeof(nk_e2m3_t)},
    {nk_e3m2_k, "e3m2", "e3m2", "|V1", sizeof(nk_e3m2_t)},
    {nk_i4_k, "int4", "i4", "|V1", sizeof(nk_i4x2_t)},
    {nk_u4_k, "uint4", "u4", "|V1", sizeof(nk_u4x2_t)},
    {nk_f64c_k, "complex128", "Zd", "<c16", sizeof(nk_f64_t) * 2},
    {nk_f32c_k, "complex64", "Zf", "<c8", sizeof(nk_f32_t) * 2},
    {nk_f16c_k, "complex32", "Ze", "|V4", sizeof(nk_f16_t) * 2},
    {nk_bf16c_k, "bfloat16c", "bcomplex32", "|V4", sizeof(nk_bf16_t) * 2},
    {nk_u1_k, "uint1", "?", "|V1", sizeof(nk_u1x8_t)},
    {nk_i8_k, "int8", "b", "|i1", sizeof(nk_i8_t)},
    {nk_u8_k, "uint8", "B", "|u1", sizeof(nk_u8_t)},
    {nk_i16_k, "int16", "h", "<i2", sizeof(nk_i16_t)},
    {nk_u16_k, "uint16", "H", "<u2", sizeof(nk_u16_t)},
#if SIZEOF_LONG == 4
    // PEP 3118 format characters for integers depend on sizeof(long):
    //   ILP32 / LLP64 (Windows, i386, WASM): long = 4 bytes → 'l'/'L' = int32/uint32
    //   LP64 (Linux x86_64, macOS arm64):    long = 8 bytes → 'l'/'L' = int64/uint64
    // CPython's pyconfig.h provides SIZEOF_LONG on all platforms.
    // These must match `py_string_to_nk_dtype` so round-tripping through
    // the buffer protocol works correctly.
    {nk_i32_k, "int32", "l", "<i4", sizeof(nk_i32_t)},
    {nk_u32_k, "uint32", "L", "<u4", sizeof(nk_u32_t)},
    {nk_i64_k, "int64", "q", "<i8", sizeof(nk_i64_t)},
    {nk_u64_k, "uint64", "Q", "<u8", sizeof(nk_u64_t)},
#else
    {nk_i32_k, "int32", "i", "<i4", sizeof(nk_i32_t)},
    {nk_u32_k, "uint32", "I", "<u4", sizeof(nk_u32_t)},
    {nk_i64_k, "int64", "l", "<i8", sizeof(nk_i64_t)},
    {nk_u64_k, "uint64", "L", "<u8", sizeof(nk_u64_t)},
#endif
};

size_t const nk_dtype_table_size = sizeof(nk_dtype_conversion_infos) / sizeof(nk_dtype_conversion_infos[0]);

nk_dtype_conversion_info_t const *nk_dtype_conversion_info(nk_dtype_t dtype) {
    for (size_t i = 0; i < nk_dtype_table_size; i++) {
        if (nk_dtype_conversion_infos[i].dtype == dtype) return &nk_dtype_conversion_infos[i];
    }
    return NULL;
}

size_t nk_dtype_bytes_per_value(nk_dtype_t dtype) {
    nk_dtype_conversion_info_t const *info = nk_dtype_conversion_info(dtype);
    return info ? info->item_size : 0;
}

char const *nk_dtype_name(nk_dtype_t dtype) {
    nk_dtype_conversion_info_t const *info = nk_dtype_conversion_info(dtype);
    return info ? info->name : "unknown";
}

char const *nk_dtype_to_numpy_typestr(nk_dtype_t dtype) {
    nk_dtype_conversion_info_t const *info = nk_dtype_conversion_info(dtype);
    return info ? info->numpy_typestr : "|V1";
}

char const *nk_dtype_to_pybuffer_typestr(nk_dtype_t dtype) {
    nk_dtype_conversion_info_t const *info = nk_dtype_conversion_info(dtype);
    return info ? info->pybuffer_typestr : "unknown";
}

nk_dtype_t resolve_nk_dtype_in_py_buffer(Py_buffer const *buffer) {
    if (buffer->obj && PyObject_TypeCheck(buffer->obj, &TensorType)) return ((Tensor *)buffer->obj)->dtype;
    return buffer->format ? py_string_to_nk_dtype(buffer->format, (Py_ssize_t)strlen(buffer->format))
                          : nk_dtype_unknown_k;
}

/** @brief Per-component bit width: for complex types returns the width of one component. */
nk_size_t nk_dtype_component_bits_(nk_dtype_t dtype) {
    nk_size_t bits = nk_dtype_bits(dtype);
    if (nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) bits /= 2;
    return bits;
}

/** @brief Returns the signed integer dtype at the given bit width, or nk_dtype_unknown_k. */
nk_dtype_t nk_signed_int_at_bits_(nk_size_t bits) {
    switch (bits) {
    case 4: return nk_i4_k;
    case 8: return nk_i8_k;
    case 16: return nk_i16_k;
    case 32: return nk_i32_k;
    case 64: return nk_i64_k;
    default: return nk_dtype_unknown_k;
    }
}

/** @brief Returns the unsigned integer dtype at the given bit width, or nk_dtype_unknown_k. */
nk_dtype_t nk_unsigned_int_at_bits_(nk_size_t bits) {
    switch (bits) {
    case 4: return nk_u4_k;
    case 8: return nk_u8_k;
    case 16: return nk_u16_k;
    case 32: return nk_u32_k;
    case 64: return nk_u64_k;
    default: return nk_dtype_unknown_k;
    }
}

/** @brief Returns the next power-of-two bit width, doubling from the given width. */
nk_size_t nk_next_int_bits_(nk_size_t bits) {
    if (bits < 8) return 8;
    if (bits < 16) return 16;
    if (bits < 32) return 32;
    if (bits < 64) return 64;
    return 0; // overflow
}

nk_dtype_t nk_dtype_promote(nk_dtype_t a, nk_dtype_t b) {
    if (a == b) return a;

    nk_dtype_family_t family_a = nk_dtype_family(a), family_b = nk_dtype_family(b);
    nk_size_t bits_a = nk_dtype_component_bits_(a), bits_b = nk_dtype_component_bits_(b);
    if (bits_a == 0 || bits_b == 0) return nk_dtype_unknown_k;

    // Same family: return wider
    if (family_a == family_b) {
        if (family_a == nk_dtype_family_float_k) {
            // Exotic floats (e4m3, e5m2, bf16) mixed with standard floats → promote through f32
            if (bits_a <= 8 || bits_b <= 8 || a == nk_bf16_k || b == nk_bf16_k) {
                if (bits_a <= 16 && bits_b <= 16) return nk_f32_k;
            }
            return bits_a >= bits_b ? a : b;
        }
        if (family_a == nk_dtype_family_int_k) return nk_signed_int_at_bits_(bits_a >= bits_b ? bits_a : bits_b);
        if (family_a == nk_dtype_family_uint_k) return nk_unsigned_int_at_bits_(bits_a >= bits_b ? bits_a : bits_b);
        if (family_a == nk_dtype_family_complex_float_k) return bits_a >= bits_b ? a : b;
    }

    // Signed + unsigned → next wider signed if needed
    if ((family_a == nk_dtype_family_int_k && family_b == nk_dtype_family_uint_k) ||
        (family_a == nk_dtype_family_uint_k && family_b == nk_dtype_family_int_k)) {
        nk_size_t unsigned_bits = (family_a == nk_dtype_family_uint_k) ? bits_a : bits_b;
        nk_size_t signed_bits = (family_a == nk_dtype_family_int_k) ? bits_a : bits_b;
        if (unsigned_bits >= signed_bits) return nk_signed_int_at_bits_(nk_next_int_bits_(unsigned_bits));
        return nk_signed_int_at_bits_(signed_bits >= unsigned_bits ? signed_bits : unsigned_bits);
    }

    // Int + float → float wide enough
    if ((family_a == nk_dtype_family_float_k && family_b != nk_dtype_family_float_k) ||
        (family_b == nk_dtype_family_float_k && family_a != nk_dtype_family_float_k)) {
        nk_size_t int_bits = (family_a == nk_dtype_family_float_k) ? bits_b : bits_a;
        if (int_bits >= 32) return nk_f64_k;
        if (int_bits >= 8) return nk_f32_k;
        return nk_f32_k;
    }

    // Complex + real → complex with promoted component
    if (family_a == nk_dtype_family_complex_float_k || family_b == nk_dtype_family_complex_float_k) {
        nk_size_t max_bits = bits_a >= bits_b ? bits_a : bits_b;
        return max_bits >= 64 ? nk_f64c_k : nk_f32c_k;
    }

    return nk_dtype_unknown_k;
}

int same_string(char const *a, char const *b) { return strcmp(a, b) == 0; }

int same_string_n(char const *input, Py_ssize_t input_len, char const *literal, Py_ssize_t literal_len) {
    return input_len == literal_len && memcmp(input, literal, (size_t)input_len) == 0;
}

/** @brief Convenience macro: compare input of known length against a string literal. */
#define same_literal_(input, len, literal) same_string_n((input), (len), (literal), (Py_ssize_t)(sizeof(literal) - 1))

nk_dtype_t py_string_to_nk_dtype(char const *name, Py_ssize_t len) {
    switch (len) {
    case 1:
        switch (name[0]) {
        // Floating-point
        case 'f': return nk_f32_k;
        case 'e': return nk_f16_k;
        case 'd': return nk_f64_k;
        // Complex
        case 'F': return nk_f32c_k;
        case 'D': return nk_f64c_k;
        case 'E': return nk_f16c_k;
        // Boolean
        case '?': return nk_u1_k;
        // Signed integers
        case 'b': return nk_i8_k;
        case 'h': return nk_i16_k;
        // Unsigned integers
        case 'B': return nk_u8_k;
        case 'H': return nk_u16_k;
#if SIZEOF_LONG == 4
        case 'l': return nk_i32_k;
        case 'q': return nk_i64_k;
        case 'L': return nk_u32_k;
        case 'Q': return nk_u64_k;
#else
        case 'i': return nk_i32_k;
        case 'l': return nk_i64_k;
        case 'I': return nk_u32_k;
        case 'L': return nk_u64_k;
#endif
        }
        break;

    case 2:
        // Floating-point: "f2" → f16, "f4" → f32, "f8" → f64
        if (same_literal_(name, len, "f2")) return nk_f16_k;
        if (same_literal_(name, len, "f4")) return nk_f32_k;
        if (same_literal_(name, len, "f8")) return nk_f64_k;
        // Signed integers
        if (same_literal_(name, len, "i1")) return nk_i8_k;
        if (same_literal_(name, len, "i2")) return nk_i16_k;
#if SIZEOF_LONG == 4
        if (same_literal_(name, len, "i4")) return nk_i32_k;
        if (same_literal_(name, len, "i8")) return nk_i64_k;
#else
        if (same_literal_(name, len, "i4")) return nk_i32_k;
        if (same_literal_(name, len, "i8")) return nk_i64_k;
#endif
        // Unsigned integers
        if (same_literal_(name, len, "u1")) return nk_u8_k;
        if (same_literal_(name, len, "u2")) return nk_u16_k;
#if SIZEOF_LONG == 4
        if (same_literal_(name, len, "u4")) return nk_u32_k;
        if (same_literal_(name, len, "u8")) return nk_u64_k;
#else
        if (same_literal_(name, len, "u4")) return nk_u32_k;
        if (same_literal_(name, len, "u8")) return nk_u64_k;
#endif
        // Complex: "Zf" → f32c, "Zd" → f64c, "Ze" → f16c
        if (same_literal_(name, len, "Zf")) return nk_f32c_k;
        if (same_literal_(name, len, "Zd")) return nk_f64c_k;
        if (same_literal_(name, len, "Ze")) return nk_f16c_k;
        // Complex shorthand: "F2" → f16c, "F4" → f32c, "F8" → f64c
        if (same_literal_(name, len, "F2")) return nk_f16c_k;
        if (same_literal_(name, len, "F4")) return nk_f32c_k;
        if (same_literal_(name, len, "F8")) return nk_f64c_k;
        // Buffer protocol shorthand
        if (same_literal_(name, len, "<f")) return nk_f32_k;
        if (same_literal_(name, len, "<e")) return nk_f16_k;
        if (same_literal_(name, len, "<d")) return nk_f64_k;
        if (same_literal_(name, len, "<F")) return nk_f32c_k;
        if (same_literal_(name, len, "<D")) return nk_f64c_k;
        if (same_literal_(name, len, "<E")) return nk_f16c_k;
        if (same_literal_(name, len, "<b")) return nk_i8_k;
        if (same_literal_(name, len, "<B")) return nk_u8_k;
        if (same_literal_(name, len, "<h")) return nk_i16_k;
        if (same_literal_(name, len, "<H")) return nk_u16_k;
#if SIZEOF_LONG == 4
        if (same_literal_(name, len, "<l")) return nk_i32_k;
        if (same_literal_(name, len, "<q")) return nk_i64_k;
        if (same_literal_(name, len, "<L")) return nk_u32_k;
        if (same_literal_(name, len, "<Q")) return nk_u64_k;
#else
        if (same_literal_(name, len, "<i")) return nk_i32_k;
        if (same_literal_(name, len, "<l")) return nk_i64_k;
        if (same_literal_(name, len, "<I")) return nk_u32_k;
        if (same_literal_(name, len, "<L")) return nk_u64_k;
#endif
        break;

    case 3:
        // Floating-point: "f16", "f32", "f64"
        if (same_literal_(name, len, "f16")) return nk_f16_k;
        if (same_literal_(name, len, "f32")) return nk_f32_k;
        if (same_literal_(name, len, "f64")) return nk_f64_k;
        // NumPy array interface typestr: "<f2", "<f4", "<f8"
        if (same_literal_(name, len, "<f2")) return nk_f16_k;
        if (same_literal_(name, len, "<f4")) return nk_f32_k;
        if (same_literal_(name, len, "<f8")) return nk_f64_k;
        // Complex typestr: "<F2", "<F4", "<F8"
        if (same_literal_(name, len, "<F2")) return nk_f16c_k;
        if (same_literal_(name, len, "<F4")) return nk_f32c_k;
        if (same_literal_(name, len, "<F8")) return nk_f64c_k;
        // Signed integers: "<i1", "<i2", "|i1", "|i2"
        if (same_literal_(name, len, "<i1")) return nk_i8_k;
        if (same_literal_(name, len, "|i1")) return nk_i8_k;
        if (same_literal_(name, len, "<i2")) return nk_i16_k;
        if (same_literal_(name, len, "|i2")) return nk_i16_k;
#if SIZEOF_LONG == 4
        if (same_literal_(name, len, "<i4")) return nk_i32_k;
        if (same_literal_(name, len, "|i4")) return nk_i32_k;
        if (same_literal_(name, len, "<i8")) return nk_i64_k;
        if (same_literal_(name, len, "|i8")) return nk_i64_k;
#else
        if (same_literal_(name, len, "<i4")) return nk_i32_k;
        if (same_literal_(name, len, "|i4")) return nk_i32_k;
        if (same_literal_(name, len, "<i8")) return nk_i64_k;
        if (same_literal_(name, len, "|i8")) return nk_i64_k;
#endif
        // Unsigned integers: "<u1", "<u2", "|u1", "|u2"
        if (same_literal_(name, len, "<u1")) return nk_u8_k;
        if (same_literal_(name, len, "|u1")) return nk_u8_k;
        if (same_literal_(name, len, "<u2")) return nk_u16_k;
        if (same_literal_(name, len, "|u2")) return nk_u16_k;
#if SIZEOF_LONG == 4
        if (same_literal_(name, len, "<u4")) return nk_u32_k;
        if (same_literal_(name, len, "|u4")) return nk_u32_k;
        if (same_literal_(name, len, "<u8")) return nk_u64_k;
        if (same_literal_(name, len, "|u8")) return nk_u64_k;
#else
        if (same_literal_(name, len, "<u4")) return nk_u32_k;
        if (same_literal_(name, len, "|u4")) return nk_u32_k;
        if (same_literal_(name, len, "<u8")) return nk_u64_k;
        if (same_literal_(name, len, "|u8")) return nk_u64_k;
#endif
        break;

    case 4:
        // Floating-point: "bf16", "e4m3", "e5m2", "e2m3", "e3m2", "int4"
        if (same_literal_(name, len, "bf16")) return nk_bf16_k;
        if (same_literal_(name, len, "e4m3")) return nk_e4m3_k;
        if (same_literal_(name, len, "e5m2")) return nk_e5m2_k;
        if (same_literal_(name, len, "e2m3")) return nk_e2m3_k;
        if (same_literal_(name, len, "e3m2")) return nk_e3m2_k;
        // Sub-byte integers
        if (same_literal_(name, len, "int4")) return nk_i4_k;
        if (same_literal_(name, len, "int8")) return nk_i8_k;
        // Buffer protocol: "<c8" → f32c, "<c8" is 3 chars actually... no
        // Complex: "<c8"=3 already handled above
        break;

    case 5:
        if (same_literal_(name, len, "int16")) return nk_i16_k;
        if (same_literal_(name, len, "int32")) return nk_i32_k;
        if (same_literal_(name, len, "int64")) return nk_i64_k;
        if (same_literal_(name, len, "uint1")) return nk_u1_k;
        if (same_literal_(name, len, "uint4")) return nk_u4_k;
        if (same_literal_(name, len, "uint8")) return nk_u8_k;
        if (same_literal_(name, len, "bf16c")) return nk_bf16c_k;
        break;

    case 6:
        if (same_literal_(name, len, "uint16")) return nk_u16_k;
        if (same_literal_(name, len, "uint32")) return nk_u32_k;
        if (same_literal_(name, len, "uint64")) return nk_u64_k;
        break;

    case 7:
        if (same_literal_(name, len, "float16")) return nk_f16_k;
        if (same_literal_(name, len, "float32")) return nk_f32_k;
        if (same_literal_(name, len, "float64")) return nk_f64_k;
        break;

    case 8:
        if (same_literal_(name, len, "bfloat16")) return nk_bf16_k;
        break;

    case 9:
        if (same_literal_(name, len, "complex32")) return nk_f16c_k;
        if (same_literal_(name, len, "complex64")) return nk_f32c_k;
        if (same_literal_(name, len, "bfloat16c")) return nk_bf16c_k;
        break;

    case 10:
        if (same_literal_(name, len, "complex128")) return nk_f64c_k;
        if (same_literal_(name, len, "bcomplex32")) return nk_bf16c_k;
        break;

    case 11:
        if (same_literal_(name, len, "float8_e4m3")) return nk_e4m3_k;
        if (same_literal_(name, len, "float8_e5m2")) return nk_e5m2_k;
        if (same_literal_(name, len, "float6_e2m3")) return nk_e2m3_k;
        if (same_literal_(name, len, "float6_e3m2")) return nk_e3m2_k;
        break;

    case 13:
        if (same_literal_(name, len, "float8_e4m3fn")) return nk_e4m3_k;
        if (same_literal_(name, len, "float6_e2m3fn")) return nk_e2m3_k;
        if (same_literal_(name, len, "float6_e3m2fn")) return nk_e3m2_k;
        break;

    default: break;
    }
    return nk_dtype_unknown_k;
}

nk_dtype_t py_object_to_nk_dtype(PyObject *obj) {
    if (PyType_Check(obj)) {
        PyTypeObject *type = (PyTypeObject *)obj;
        if (type == &NkBFloat16Scalar_Type) return nk_bf16_k;
        if (type == &NkFloat16Scalar_Type) return nk_f16_k;
        if (type == &NkFloat8E4M3Scalar_Type) return nk_e4m3_k;
        if (type == &NkFloat8E5M2Scalar_Type) return nk_e5m2_k;
        if (type == &NkFloat6E2M3Scalar_Type) return nk_e2m3_k;
        if (type == &NkFloat6E3M2Scalar_Type) return nk_e3m2_k;
        PyErr_Format(PyExc_ValueError, "Unsupported dtype type: %s", type->tp_name);
        return nk_dtype_unknown_k;
    }
    if (PyUnicode_Check(obj)) {
        Py_ssize_t s_len = 0;
        char const *s = PyUnicode_AsUTF8AndSize(obj, &s_len);
        nk_dtype_t dtype = s ? py_string_to_nk_dtype(s, s_len) : nk_dtype_unknown_k;
        if (dtype == nk_dtype_unknown_k) PyErr_Format(PyExc_ValueError, "Unsupported dtype: '%s'", s ? s : "");
        return dtype;
    }
    PyErr_Format(PyExc_TypeError, "Expected a string or type for 'dtype', got %s", Py_TYPE(obj)->tp_name);
    return nk_dtype_unknown_k;
}

nk_kernel_kind_t py_string_to_nk_kernel_kind(char const *name, Py_ssize_t len) {
    switch (len) {
    case 3:
        if (same_literal_(name, len, "dot")) return nk_kernel_dot_k;
        if (same_literal_(name, len, "jsd")) return nk_kernel_jsd_k;
        if (same_literal_(name, len, "kld")) return nk_kernel_kld_k;
        if (same_literal_(name, len, "fma")) return nk_kernel_each_fma_k;
        break;
    case 4:
        if (same_literal_(name, len, "vdot")) return nk_kernel_vdot_k;
        break;
    case 5:
        if (same_literal_(name, len, "inner")) return nk_kernel_dot_k;
        if (same_literal_(name, len, "blend")) return nk_kernel_each_blend_k;
        break;
    case 7:
        if (same_literal_(name, len, "angular")) return nk_kernel_angular_k;
        if (same_literal_(name, len, "hamming")) return nk_kernel_hamming_k;
        if (same_literal_(name, len, "jaccard")) return nk_kernel_jaccard_k;
        break;
    case 8:
        if (same_literal_(name, len, "bilinear")) return nk_kernel_bilinear_k;
        break;
    case 9:
        if (same_literal_(name, len, "euclidean")) return nk_kernel_euclidean_k;
        break;
    case 11:
        if (same_literal_(name, len, "mahalanobis")) return nk_kernel_mahalanobis_k;
        if (same_literal_(name, len, "sqeuclidean")) return nk_kernel_sqeuclidean_k;
        break;
    case 13:
        if (same_literal_(name, len, "jensenshannon")) return nk_kernel_jsd_k;
        break;
    case 15:
        if (same_literal_(name, len, "kullbackleibler")) return nk_kernel_kld_k;
        break;
    }
    return nk_kernel_unknown_k;
}

PyObject *nk_scalar_buffer_to_py_number(nk_scalar_buffer_t const *buf, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return PyFloat_FromDouble(buf->f64);
    case nk_f32_k: return PyFloat_FromDouble((double)buf->f32);
    case nk_f16_k: {
        nk_f32_t f32_tmp;
        nk_f16_to_f32(&buf->f16, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_bf16_k: {
        nk_f32_t f32_tmp;
        nk_bf16_to_f32(&buf->bf16, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_f64c_k: return PyComplex_FromDoubles(buf->f64c.real, buf->f64c.imag);
    case nk_f32c_k: return PyComplex_FromDoubles((double)buf->f32c.real, (double)buf->f32c.imag);
    case nk_i64_k: return PyLong_FromLongLong(buf->i64);
    case nk_u64_k: return PyLong_FromUnsignedLongLong(buf->u64);
    case nk_i32_k: return PyLong_FromLong(buf->i32);
    case nk_u32_k: return PyLong_FromUnsignedLong(buf->u32);
    case nk_i16_k: return PyLong_FromLong(buf->i16);
    case nk_u16_k: return PyLong_FromUnsignedLong(buf->u16);
    case nk_i8_k: return PyLong_FromLong(buf->i8);
    case nk_u8_k: return PyLong_FromUnsignedLong(buf->u8);
    case nk_e4m3_k: {
        nk_f32_t f32_tmp;
        nk_e4m3_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_e5m2_k: {
        nk_f32_t f32_tmp;
        nk_e5m2_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_e2m3_k: {
        nk_f32_t f32_tmp;
        nk_e2m3_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_e3m2_k: {
        nk_f32_t f32_tmp;
        nk_e3m2_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    default: return PyFloat_FromDouble(0.0);
    }
}

int py_number_to_f64(PyObject *obj, nk_f64_t *value); // forward declaration

int py_number_to_nk_scalar_buffer(PyObject *obj, nk_scalar_buffer_t *buf, nk_dtype_t dtype) {
    memset(buf, 0, sizeof(*buf));
    if ((nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) && PyComplex_Check(obj)) {
        Py_complex py_complex = PyComplex_AsCComplex(obj);
        switch (dtype) {
        case nk_f64c_k:
            buf->f64c.real = py_complex.real;
            buf->f64c.imag = py_complex.imag;
            return 1;
        case nk_f32c_k:
            buf->f32c.real = (float)py_complex.real;
            buf->f32c.imag = (float)py_complex.imag;
            return 1;
        default: break;
        }
    }
    nk_f64_t value;
    if (!py_number_to_f64(obj, &value)) return 0;
    buf->f64 = value;
    nk_scalar_buffer_from_f64(&buf->f64, buf, dtype);
    return 1;
}

int nk_scalar_buffer_export(                                   //
    nk_scalar_buffer_t const *source, nk_dtype_t source_dtype, //
    void *target, nk_dtype_t target_dtype) {                   //
    nk_scalar_buffer_t converted;
    if (!nk_scalar_buffer_to_f64c(source, source_dtype, &converted.f64c)) return 0;
    if (!nk_scalar_buffer_from_f64c(&converted.f64c, &converted, target_dtype)) return 0;
    nk_size_t target_size = nk_dtype_bits(target_dtype) / NK_BITS_PER_BYTE;
    nk_copy_bytes_(target, &converted, target_size);
    return 1;
}

int nk_kernel_is_commutative(nk_kernel_kind_t kind) {
    switch (kind) {
    case nk_kernel_kld_k: return 0;
    case nk_kernel_vdot_k: return 0;
    case nk_kernel_bilinear_k: return 0;
    default: return 1;
    }
}

int py_object_is_scalar(PyObject *obj) {
    if (PyFloat_Check(obj) || PyLong_Check(obj)) return 1;
    // Check for NumPy scalar types (0D arrays or numpy.generic subclasses)
    if (PyNumber_Check(obj)) {
        // 0D numpy arrays and numpy scalars (e.g. np.int8(-11)) support the buffer protocol
        // but have ndim == 0. Check if the object has ndim attribute == 0.
        PyObject *ndim_obj = PyObject_GetAttrString(obj, "ndim");
        if (ndim_obj) {
            long ndim = PyLong_AsLong(ndim_obj);
            Py_DECREF(ndim_obj);
            if (ndim == 0) return 1;
        }
        else { PyErr_Clear(); }
    }
    return 0;
}

int py_number_to_f64(PyObject *obj, nk_f64_t *value) {
    if (PyFloat_Check(obj)) {
        *value = PyFloat_AsDouble(obj);
        return !PyErr_Occurred();
    }
    else if (PyLong_Check(obj)) {
        *value = PyLong_AsDouble(obj);
        return !PyErr_Occurred();
    }
    // Handle NumPy scalars and 0D arrays via PyNumber_Float
    PyObject *as_float = PyNumber_Float(obj);
    if (as_float) {
        *value = PyFloat_AsDouble(as_float);
        Py_DECREF(as_float);
        return !PyErr_Occurred();
    }
    PyErr_Clear();
    return 0;
}

/** @brief Fallback: synthesize a Py_buffer from __array_interface__. */
static int nk_get_buffer_via_array_interface(PyObject *obj, Py_buffer *buffer, nk_buffer_backing_t *backing) {
    PyObject *iface = PyObject_GetAttrString(obj, "__array_interface__");
    if (!iface) {
        PyErr_Clear();
        return 0;
    }
    if (!PyDict_Check(iface)) {
        Py_DECREF(iface);
        return 0;
    }

    // Extract data pointer from dict["data"][0].
    PyObject *data_tuple = PyDict_GetItemString(iface, "data");
    if (!data_tuple || !PyTuple_Check(data_tuple) || PyTuple_GET_SIZE(data_tuple) < 1) {
        Py_DECREF(iface);
        PyErr_SetString(PyExc_TypeError, "__array_interface__['data'] must be a tuple of (ptr, readonly)");
        return 0;
    }
    void *data_ptr = PyLong_AsVoidPtr(PyTuple_GET_ITEM(data_tuple, 0));
    if (!data_ptr && PyErr_Occurred()) {
        Py_DECREF(iface);
        return 0;
    }

    // The second element of the tuple, when present, marks the buffer as read-only.
    int readonly = 0;
    if (PyTuple_GET_SIZE(data_tuple) > 1) {
        readonly = PyObject_IsTrue(PyTuple_GET_ITEM(data_tuple, 1));
        if (readonly < 0) {
            Py_DECREF(iface);
            return 0;
        }
    }

    // Extract shape tuple.
    PyObject *shape_obj = PyDict_GetItemString(iface, "shape");
    if (!shape_obj || !PyTuple_Check(shape_obj)) {
        Py_DECREF(iface);
        PyErr_SetString(PyExc_TypeError, "__array_interface__['shape'] must be a tuple");
        return 0;
    }
    Py_ssize_t rank = PyTuple_GET_SIZE(shape_obj);
    if (rank < 1 || rank > NK_TENSOR_MAX_RANK) {
        Py_DECREF(iface);
        PyErr_Format(PyExc_ValueError, "Tensor rank %zd exceeds maximum supported rank %d", rank, NK_TENSOR_MAX_RANK);
        return 0;
    }
    for (Py_ssize_t i = 0; i < rank; i++) {
        backing->shape[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape_obj, i));
        if (backing->shape[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(iface);
            return 0;
        }
    }

    // Resolve dtype: try obj.dtype.name first, then dict["typestr"].
    nk_dtype_t dtype = nk_dtype_unknown_k;
    PyObject *dtype_attr = PyObject_GetAttrString(obj, "dtype");
    if (dtype_attr) {
        PyObject *name_attr = PyObject_GetAttrString(dtype_attr, "name");
        if (name_attr) {
            Py_ssize_t name_len = 0;
            char const *name_str = PyUnicode_AsUTF8AndSize(name_attr, &name_len);
            if (name_str) dtype = py_string_to_nk_dtype(name_str, name_len);
            Py_DECREF(name_attr);
        }
        else { PyErr_Clear(); }
        Py_DECREF(dtype_attr);
    }
    else { PyErr_Clear(); }

    if (dtype == nk_dtype_unknown_k) {
        PyObject *typestr_obj = PyDict_GetItemString(iface, "typestr");
        if (typestr_obj) {
            Py_ssize_t typestr_len = 0;
            char const *typestr = PyUnicode_AsUTF8AndSize(typestr_obj, &typestr_len);
            if (typestr) dtype = py_string_to_nk_dtype(typestr, typestr_len);
        }
    }
    if (dtype == nk_dtype_unknown_k) {
        Py_DECREF(iface);
        PyErr_SetString(PyExc_ValueError, "Cannot determine dtype from __array_interface__");
        return 0;
    }

    nk_dtype_conversion_info_t const *info = nk_dtype_conversion_info(dtype);
    if (!info) {
        Py_DECREF(iface);
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype from __array_interface__");
        return 0;
    }
    Py_ssize_t itemsize = (Py_ssize_t)info->item_size;

    // Extract strides (may be None for C-contiguous arrays).
    PyObject *strides_obj = PyDict_GetItemString(iface, "strides");
    if (strides_obj && PyTuple_Check(strides_obj) && PyTuple_GET_SIZE(strides_obj) == rank) {
        for (Py_ssize_t i = 0; i < rank; i++) {
            backing->strides[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(strides_obj, i));
            if (backing->strides[i] == -1 && PyErr_Occurred()) {
                Py_DECREF(iface);
                return 0;
            }
        }
    }
    else {
        // C-contiguous: compute strides from shape x itemsize.
        Py_ssize_t stride = itemsize;
        for (Py_ssize_t i = rank - 1; i >= 0; i--) {
            backing->strides[i] = stride;
            stride *= backing->shape[i];
        }
    }
    Py_DECREF(iface);

    // Populate the Py_buffer so callers can read shape/strides/itemsize uniformly.
    memset(buffer, 0, sizeof(*buffer));
    buffer->buf = data_ptr;
    buffer->readonly = readonly;
    buffer->itemsize = itemsize;
    buffer->format = (char *)info->pybuffer_typestr;
    buffer->ndim = (int)rank;
    buffer->len = itemsize;
    for (Py_ssize_t i = 0; i < rank; i++) buffer->len *= backing->shape[i];
    buffer->shape = backing->shape;
    buffer->strides = backing->strides;
    buffer->obj = NULL; // Makes PyBuffer_Release a no-op.
    return 1;
}

int nk_get_buffer(PyObject *obj, Py_buffer *buffer, int flags, nk_buffer_backing_t *backing) {
    if (PyObject_GetBuffer(obj, buffer, flags) == 0) return 1;
    PyErr_Clear();
    if (nk_get_buffer_via_array_interface(obj, buffer, backing)) {
        // The fallback bypasses the flag checks PyObject_GetBuffer would have made.
        if ((flags & PyBUF_WRITABLE) && buffer->readonly) {
            PyBuffer_Release(buffer);
            PyErr_SetString(PyExc_BufferError, "Object is not writable");
            return 0;
        }
        return 1;
    }
    if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "argument must support buffer protocol or __array_interface__");
    return 0;
}

int parse_tensor(PyObject *tensor, Py_buffer *buffer, MatrixOrVectorView *parsed, nk_buffer_backing_t *backing,
                 nk_dtype_t dtype_hint) {
    if (!nk_get_buffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT, backing)) return 0;

    if (buffer->ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", buffer->ndim,
                     NK_TENSOR_MAX_RANK);
        PyBuffer_Release(buffer);
        return 0;
    }

    parsed->data = buffer->buf;
    if (dtype_hint != nk_dtype_unknown_k) { parsed->dtype = dtype_hint; }
    else {
        parsed->dtype = resolve_nk_dtype_in_py_buffer(buffer);
        if (parsed->dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unsupported '%s' dtype specifier", buffer->format);
            PyBuffer_Release(buffer);
            return 0;
        }
    }

    parsed->rank = buffer->ndim;
    if (buffer->ndim == 1) {
        if (buffer->strides[0] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "Input vectors must be contiguous, check with `X.__array_interface__`");
            PyBuffer_Release(buffer);
            return 0;
        }
        parsed->cols = buffer->shape[0];
        parsed->rows = 1;
        parsed->row_stride = 0;
    }
    else if (buffer->ndim == 2) {
        if (buffer->strides[1] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "Input vectors must be contiguous, check with `X.__array_interface__`");
            PyBuffer_Release(buffer);
            return 0;
        }
        parsed->cols = buffer->shape[1];
        parsed->rows = buffer->shape[0];
        parsed->row_stride = buffer->strides[0];
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Input tensors must be 1D or 2D");
        PyBuffer_Release(buffer);
        return 0;
    }

    return 1;
}

int parse_tensor_nd(PyObject *obj, Py_buffer *buffer, TensorView *view, nk_buffer_backing_t *backing,
                    nk_dtype_t dtype_hint) {
    if (!nk_get_buffer(obj, buffer, PyBUF_STRIDES | PyBUF_FORMAT, backing)) return 0;
    if ((size_t)buffer->ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "rank %d exceeds maximum %d", buffer->ndim, NK_TENSOR_MAX_RANK);
        PyBuffer_Release(buffer);
        return 0;
    }
    view->data = buffer->buf;
    if (dtype_hint != nk_dtype_unknown_k) { view->dtype = dtype_hint; }
    else {
        view->dtype = resolve_nk_dtype_in_py_buffer(buffer);
        if (view->dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unsupported '%s' dtype specifier", buffer->format);
            PyBuffer_Release(buffer);
            return 0;
        }
    }
    view->rank = (size_t)buffer->ndim;
    view->shape = buffer->shape;
    view->strides = buffer->strides;
    return 1;
}

static struct {
    char const *name;
    nk_capability_t flag;
} const cap_table[] = {
    {"serial", nk_cap_serial_k},
    // ARM NEON
    {"neon", nk_cap_neon_k},
    {"neonhalf", nk_cap_neonhalf_k},
    {"neonfhm", nk_cap_neonfhm_k},
    {"neonbfdot", nk_cap_neonbfdot_k},
    {"neonsdot", nk_cap_neonsdot_k},
    {"neonfp8", nk_cap_neonfp8_k},
    // ARM SVE
    {"sve", nk_cap_sve_k},
    {"svehalf", nk_cap_svehalf_k},
    {"svebfdot", nk_cap_svebfdot_k},
    {"svesdot", nk_cap_svesdot_k},
    {"sve2", nk_cap_sve2_k},
    {"sve2p1", nk_cap_sve2p1_k},
    // ARM SME
    {"sme", nk_cap_sme_k},
    {"sme2", nk_cap_sme2_k},
    {"sme2p1", nk_cap_sme2p1_k},
    {"smef64", nk_cap_smef64_k},
    {"smehalf", nk_cap_smehalf_k},
    {"smebf16", nk_cap_smebf16_k},
    {"smebi32", nk_cap_smebi32_k},
    {"smelut2", nk_cap_smelut2_k},
    {"smefa64", nk_cap_smefa64_k},
    // x86
    {"haswell", nk_cap_haswell_k},
    {"alder", nk_cap_alder_k},
    {"sierra", nk_cap_sierra_k},
    {"skylake", nk_cap_skylake_k},
    {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},
    {"turin", nk_cap_turin_k},
    {"sapphire", nk_cap_sapphire_k},
    {"sapphireamx", nk_cap_sapphireamx_k},
    {"graniteamx", nk_cap_graniteamx_k},
    {"diamond", nk_cap_diamond_k},
    // RISC-V
    {"rvv", nk_cap_rvv_k},
    {"rvvhalf", nk_cap_rvvhalf_k},
    {"rvvbf16", nk_cap_rvvbf16_k},
    {"rvvbb", nk_cap_rvvbb_k},
    // LoongArch
    {"loongsonasx", nk_cap_loongsonasx_k},
    // Power
    {"powervsx", nk_cap_powervsx_k},
    // WASM
    {"v128relaxed", nk_cap_v128relaxed_k},
    {NULL},
};

char const doc_enable_capability[] =                                                         //
    "Enable a specific SIMD kernel family.\n\n"                                              //
    "Parameters:\n"                                                                          //
    "    capability (str): Name of the SIMD feature to enable (for example, 'haswell').\n\n" //
    "Signature:\n"                                                                           //
    "    >>> def enable_capability(capability): ...";

static int refresh_runtime_dispatch_after_capability_change(void) {
    if (!nk_configure_thread(static_capabilities)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to configure thread for updated capabilities");
        return 0;
    }
    nk_dispatch_table_update(static_capabilities);
    return 1;
}

PyObject *api_enable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    for (size_t i = 0; cap_table[i].name; ++i) {
        if (same_string(cap_name, cap_table[i].name)) {
            if (cap_table[i].flag == nk_cap_serial_k) {
                PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
                return NULL;
            }
            static_capabilities |= cap_table[i].flag;
            if (!refresh_runtime_dispatch_after_capability_change()) return NULL;
            Py_RETURN_NONE;
        }
    }

    PyErr_SetString(PyExc_ValueError, "Unknown capability");
    return NULL;
}

char const doc_disable_capability[] =                                                         //
    "Disable a specific SIMD kernel family.\n\n"                                              //
    "Parameters:\n"                                                                           //
    "    capability (str): Name of the SIMD feature to disable (for example, 'haswell').\n\n" //
    "Signature:\n"                                                                            //
    "    >>> def disable_capability(capability): ...";

PyObject *api_disable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    for (size_t i = 0; cap_table[i].name; ++i) {
        if (same_string(cap_name, cap_table[i].name)) {
            if (cap_table[i].flag == nk_cap_serial_k) {
                PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
                return NULL;
            }
            static_capabilities &= ~cap_table[i].flag;
            if (!refresh_runtime_dispatch_after_capability_change()) return NULL;
            Py_RETURN_NONE;
        }
    }

    PyErr_SetString(PyExc_ValueError, "Unknown capability");
    return NULL;
}

char const doc_get_capabilities[] =                                                               //
    "Get the current hardware SIMD capabilities as a dictionary of feature flags.\n\n"            //
    "The dictionary maps capability names to booleans. Available capabilities (beyond serial):\n" //
    "  x86 AVX2: haswell, alder, sierra.\n"                                                       //
    "  x86 AVX512: skylake, icelake, genoa, sapphire, turin, diamond.\n"                          //
    "  x86 AMX: sapphireamx, graniteamx.\n"                                                       //
    "  ARM NEON: neon, neonhalf, neonfhm, neonbfdot, neonsdot, neonfp8.\n"                        //
    "  ARM SVE: sve, svehalf, svebfdot, svesdot, sve2, sve2p1.\n"                                 //
    "  ARM SME: sme, sme2, sme2p1, smef64, smehalf, smebf16, smebi32, smelut2, smefa64.\n"        //
    "  RISC-V: rvv, rvvhalf, rvvbf16, rvvbb.\n"                                                   //
    "  LoongArch: loongsonasx.\n"                                                                 //
    "  Power: powervsx.\n"                                                                        //
    "  WASM: v128relaxed.\n";

PyObject *api_get_capabilities(PyObject *self) {
    nk_capability_t caps = static_capabilities;
    PyObject *cap_dict = PyDict_New();
    if (!cap_dict) return NULL;

    for (size_t i = 0; cap_table[i].name; ++i) {
        PyObject *val = PyBool_FromLong(caps & cap_table[i].flag);
        if (PyDict_SetItemString(cap_dict, cap_table[i].name, val) < 0) {
            Py_DECREF(val);
            Py_DECREF(cap_dict);
            return NULL;
        }
        Py_DECREF(val);
    }

    return cap_dict;
}

static PyMethodDef nk_methods[] = {
    // Introspecting library and hardware capabilities
    {"get_capabilities", (PyCFunction)api_get_capabilities, METH_NOARGS, doc_get_capabilities},
    {"enable_capability", (PyCFunction)api_enable_capability, METH_O, doc_enable_capability},
    {"disable_capability", (PyCFunction)api_disable_capability, METH_O, doc_disable_capability},

    // NumPy and SciPy compatible interfaces for dense vector representations
    // Each function can compute distances between:
    //  - A pair of vectors
    //  - A batch of vector pairs (two matrices of identical shape)
    //  - A matrix of vectors and a single vector
    {"euclidean", (PyCFunction)api_euclidean, METH_FASTCALL | METH_KEYWORDS, doc_euclidean},
    {"sqeuclidean", (PyCFunction)api_sqeuclidean, METH_FASTCALL | METH_KEYWORDS, doc_sqeuclidean},
    {"kld", (PyCFunction)api_kld, METH_FASTCALL | METH_KEYWORDS, doc_kld},
    {"jsd", (PyCFunction)api_jsd, METH_FASTCALL | METH_KEYWORDS, doc_jsd},
    {"angular", (PyCFunction)api_angular, METH_FASTCALL | METH_KEYWORDS, doc_angular},
    {"dot", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"vdot", (PyCFunction)api_vdot, METH_FASTCALL | METH_KEYWORDS, doc_vdot},
    {"hamming", (PyCFunction)api_hamming, METH_FASTCALL | METH_KEYWORDS, doc_hamming},
    {"jaccard", (PyCFunction)api_jaccard, METH_FASTCALL | METH_KEYWORDS, doc_jaccard},

    // Aliases
    {"inner", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"kullbackleibler", (PyCFunction)api_kld, METH_FASTCALL | METH_KEYWORDS, doc_kld},
    {"jensenshannon", (PyCFunction)api_jsd, METH_FASTCALL | METH_KEYWORDS, doc_jsd},

    // Conventional `cdist` interface for pairwise distances
    {"cdist", (PyCFunction)api_cdist, METH_FASTCALL | METH_KEYWORDS, doc_cdist},

    // Exposing underlying API for USearch `CompiledMetric`
    {"pointer_to_euclidean", (PyCFunction)api_euclidean_pointer, METH_O, doc_euclidean_pointer},
    {"pointer_to_sqeuclidean", (PyCFunction)api_sqeuclidean_pointer, METH_O, doc_sqeuclidean_pointer},
    {"pointer_to_angular", (PyCFunction)api_angular_pointer, METH_O, doc_angular_pointer},
    {"pointer_to_inner", (PyCFunction)api_dot_pointer, METH_O, doc_dot_pointer},
    {"pointer_to_dot", (PyCFunction)api_dot_pointer, METH_O, doc_dot_pointer},
    {"pointer_to_vdot", (PyCFunction)api_vdot_pointer, METH_O, doc_vdot_pointer},
    {"pointer_to_kullbackleibler", (PyCFunction)api_kld_pointer, METH_O, doc_kld_pointer},
    {"pointer_to_jensenshannon", (PyCFunction)api_jsd_pointer, METH_O, doc_jsd_pointer},

    // Set operations
    {"intersect", (PyCFunction)api_intersect, METH_FASTCALL, doc_intersect},
    {"sparse_dot", (PyCFunction)api_sparse_dot, METH_FASTCALL, doc_sparse_dot},

    // Symmetric pairwise operations
    {"dots_symmetric", (PyCFunction)api_dots_symmetric, METH_FASTCALL | METH_KEYWORDS, doc_dots_symmetric},
    {"hammings_symmetric", (PyCFunction)api_hammings_symmetric, METH_FASTCALL | METH_KEYWORDS, doc_hammings_symmetric},
    {"jaccards_symmetric", (PyCFunction)api_jaccards_symmetric, METH_FASTCALL | METH_KEYWORDS, doc_jaccards_symmetric},
    {"angulars_symmetric", (PyCFunction)api_angulars_symmetric, METH_FASTCALL | METH_KEYWORDS, doc_angulars_symmetric},
    {"euclideans_symmetric", (PyCFunction)api_euclideans_symmetric, METH_FASTCALL | METH_KEYWORDS,
     doc_euclideans_symmetric},

    // Curved spaces
    {"bilinear", (PyCFunction)api_bilinear, METH_FASTCALL | METH_KEYWORDS, doc_bilinear},
    {"mahalanobis", (PyCFunction)api_mahalanobis, METH_FASTCALL | METH_KEYWORDS, doc_mahalanobis},

    // Geospatial distances
    {"haversine", (PyCFunction)api_haversine, METH_FASTCALL | METH_KEYWORDS, doc_haversine},
    {"vincenty", (PyCFunction)api_vincenty, METH_FASTCALL | METH_KEYWORDS, doc_vincenty},

    // Tensor constructors
    {"from_pointer", (PyCFunction)api_from_pointer, METH_FASTCALL | METH_KEYWORDS, doc_from_pointer},
    {"from_dlpack", (PyCFunction)api_from_dlpack, METH_O, doc_from_dlpack},
    {"empty", (PyCFunction)api_empty, METH_FASTCALL | METH_KEYWORDS, doc_empty},
    {"zeros", (PyCFunction)api_zeros, METH_FASTCALL | METH_KEYWORDS, doc_zeros},
    {"ones", (PyCFunction)api_ones, METH_FASTCALL | METH_KEYWORDS, doc_ones},
    {"full", (PyCFunction)api_full, METH_FASTCALL | METH_KEYWORDS, doc_full},
    {"iota", (PyCFunction)api_iota, METH_FASTCALL | METH_KEYWORDS, doc_iota},
    {"diagonal", (PyCFunction)api_diagonal, METH_FASTCALL | METH_KEYWORDS, doc_diagonal},
    {"hash", (PyCFunction)api_hash, METH_FASTCALL | METH_KEYWORDS, doc_hash},
    {"astype", (PyCFunction)api_astype, METH_FASTCALL | METH_KEYWORDS, doc_astype},

    // Tensor reductions
    {"moments", (PyCFunction)api_moments, METH_FASTCALL | METH_KEYWORDS, doc_reduce_moments},
    {"minmax", (PyCFunction)api_minmax, METH_FASTCALL | METH_KEYWORDS, doc_reduce_minmax},
    {"sum", (PyCFunction)api_sum, METH_FASTCALL | METH_KEYWORDS, doc_reduce_sum},
    {"norm", (PyCFunction)api_norm, METH_FASTCALL | METH_KEYWORDS, doc_reduce_norm},
    {"min", (PyCFunction)api_min, METH_FASTCALL | METH_KEYWORDS, doc_reduce_min},
    {"max", (PyCFunction)api_max, METH_FASTCALL | METH_KEYWORDS, doc_reduce_max},
    {"argmin", (PyCFunction)api_argmin, METH_FASTCALL | METH_KEYWORDS, doc_reduce_argmin},
    {"argmax", (PyCFunction)api_argmax, METH_FASTCALL | METH_KEYWORDS, doc_reduce_argmax},

    // Vectorized operations
    {"fma", (PyCFunction)api_fma, METH_FASTCALL | METH_KEYWORDS, doc_fma},
    {"blend", (PyCFunction)api_blend, METH_FASTCALL | METH_KEYWORDS, doc_blend},
    {"scale", (PyCFunction)api_scale, METH_FASTCALL | METH_KEYWORDS, doc_scale},
    {"add", (PyCFunction)api_add, METH_FASTCALL | METH_KEYWORDS, doc_add},
    {"multiply", (PyCFunction)api_multiply, METH_FASTCALL | METH_KEYWORDS, doc_multiply},

    // Element-wise trigonometric functions
    {"sin", (PyCFunction)api_sin, METH_FASTCALL | METH_KEYWORDS, doc_sin},
    {"cos", (PyCFunction)api_cos, METH_FASTCALL | METH_KEYWORDS, doc_cos},
    {"atan", (PyCFunction)api_atan, METH_FASTCALL | METH_KEYWORDS, doc_atan},

    // Mesh alignment (point cloud registration)
    {"kabsch", (PyCFunction)api_kabsch, METH_FASTCALL | METH_KEYWORDS, doc_kabsch},
    {"umeyama", (PyCFunction)api_umeyama, METH_FASTCALL | METH_KEYWORDS, doc_umeyama},
    {"rmsd", (PyCFunction)api_rmsd, METH_FASTCALL | METH_KEYWORDS, doc_rmsd},

    // Matrix multiplication with pre-packed matrices
    {"dots_pack", (PyCFunction)api_dots_pack, METH_FASTCALL | METH_KEYWORDS, doc_dots_pack},
    {"dots_packed", (PyCFunction)api_dots_packed, METH_FASTCALL | METH_KEYWORDS, doc_dots_packed},
    {"hammings_pack", (PyCFunction)api_hammings_pack, METH_FASTCALL | METH_KEYWORDS, doc_hammings_pack},
    {"hammings_packed", (PyCFunction)api_hammings_packed, METH_FASTCALL | METH_KEYWORDS, doc_hammings_packed},
    {"jaccards_packed", (PyCFunction)api_jaccards_packed, METH_FASTCALL | METH_KEYWORDS, doc_jaccards_packed},
    {"angulars_packed", (PyCFunction)api_angulars_packed, METH_FASTCALL | METH_KEYWORDS, doc_angulars_packed},
    {"euclideans_packed", (PyCFunction)api_euclideans_packed, METH_FASTCALL | METH_KEYWORDS, doc_euclideans_packed},

    // MaxSim (ColBERT late-interaction)
    {"maxsim_pack", (PyCFunction)api_maxsim_pack, METH_FASTCALL | METH_KEYWORDS, doc_maxsim_pack},
    {"maxsim_packed", (PyCFunction)api_maxsim_packed, METH_FASTCALL | METH_KEYWORDS, doc_maxsim_packed},
    {"maxsim", (PyCFunction)api_maxsim, METH_FASTCALL | METH_KEYWORDS, doc_maxsim},

    // Sentinel
    {NULL, NULL, 0, NULL}};

static char const doc_module[] =                                                                    //
    "Portable mixed-precision BLAS-like vector math library for x86 and Arm.\n"                     //
    "\n"                                                                                            //
    "Performance Recommendations:\n"                                                                //
    " - Avoid converting to NumPy arrays. NumKong works with any tensor implementation\n"           //
    "   compatible with the Python buffer protocol, including PyTorch and TensorFlow.\n"            //
    " - In low-latency environments, provide the output array with the `out=` parameter\n"          //
    "   to avoid expensive memory allocations on the hot path.\n"                                   //
    " - On modern CPUs, when the application allows, prefer low-precision numeric types.\n"         //
    "   Whenever possible, use 'bf16' and 'f16' over 'f32'. Consider quantizing to 'i8'\n"          //
    "   and 'u8' for highest hardware compatibility and performance.\n"                             //
    " - If you only need relative proximity rather than absolute distance, prefer simpler\n"        //
    "   kernels such as squared Euclidean distance over Euclidean distance.\n"                      //
    " - Use row-major contiguous matrix representations. Strides between rows do not have\n"        //
    "   a significant impact on performance, but most modern HPC packages explicitly ban\n"         //
    "   non-contiguous rows where nearby cells within a row have multi-byte gaps.\n"                //
    " - The CPython runtime has noticeable overhead for function calls, so consider batching\n"     //
    "   kernel invocations. Many kernels compute 1-to-1 distances between vectors, as well as\n"    //
    "   1-to-N and N-to-N distances between batches of vectors packed into matrices.\n"             //
    "\n"                                                                                            //
    "Example:\n"                                                                                    //
    "    >>> import numkong\n"                                                                      //
    "    >>> numkong.euclidean(a, b)\n"                                                             //
    "\n"                                                                                            //
    "Mixed-precision 1-to-N example with numeric types missing in NumPy, but present in PyTorch:\n" //
    "    >>> import numkong\n"                                                                      //
    "    >>> import torch\n"                                                                        //
    "    >>> a = torch.randn(1536, dtype=torch.bfloat16)\n"                                         //
    "    >>> b = torch.randn((100, 1536), dtype=torch.bfloat16)\n"                                  //
    "    >>> c = torch.zeros(100, dtype=torch.float32)\n"                                           //
    "    >>> numkong.euclidean(a, b, dtype='bfloat16', out=c)\n";

static PyModuleDef nk_module = {
    PyModuleDef_HEAD_INIT, .m_name = "NumKong", .m_doc = doc_module, .m_size = -1, .m_methods = nk_methods,
};

PyMODINIT_FUNC PyInit__numkong(void) {
    PyObject *m;

    if (PyType_Ready(&TensorType) < 0) return NULL;
    if (PyType_Ready(&TensorIterType) < 0) return NULL;
    if (PyType_Ready(&PackedMatrixType) < 0) return NULL;
    if (PyType_Ready(&MaxSimPackedMatrixType) < 0) return NULL;
    if (PyType_Ready(&MeshAlignmentResultType) < 0) return NULL;

    m = PyModule_Create(&nk_module);
    if (m == NULL) return NULL;

    if (nk_dlpack_init(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    // Add version metadata
    {
        char version_str[64];
        snprintf(version_str, sizeof(version_str), "%d.%d.%d", NK_VERSION_MAJOR, NK_VERSION_MINOR, NK_VERSION_PATCH);
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Register Tensor type
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType) < 0) {
        Py_XDECREF(&TensorType);
        Py_XDECREF(m);
        return NULL;
    }

    // Register PackedMatrix type
    Py_INCREF(&PackedMatrixType);
    if (PyModule_AddObject(m, "PackedMatrix", (PyObject *)&PackedMatrixType) < 0) {
        Py_XDECREF(&PackedMatrixType);
        Py_XDECREF(m);
        return NULL;
    }

    // Register MaxSimPackedMatrix type
    Py_INCREF(&MaxSimPackedMatrixType);
    if (PyModule_AddObject(m, "MaxSimPackedMatrix", (PyObject *)&MaxSimPackedMatrixType) < 0) {
        Py_XDECREF(&MaxSimPackedMatrixType);
        Py_XDECREF(m);
        return NULL;
    }

    // Register MeshAlignmentResult type
    Py_INCREF(&MeshAlignmentResultType);
    if (PyModule_AddObject(m, "MeshAlignmentResult", (PyObject *)&MeshAlignmentResultType) < 0) {
        Py_XDECREF(&MeshAlignmentResultType);
        Py_XDECREF(m);
        return NULL;
    }

    static_capabilities = nk_capabilities();
    nk_configure_thread(static_capabilities);

    // Register scalar types (bfloat16, float8_e4m3, float8_e5m2)
    if (nk_register_scalar_types(m) < 0) {
        Py_XDECREF(m);
        return NULL;
    }

    // Register NumPy custom dtypes (non-fatal if NumPy is not installed).
    if (nk_register_numpy_dtypes(m) < 0) PyErr_Clear();

    return m;
}
