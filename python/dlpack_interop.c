/**
 *  @brief DLPack zero-copy interop for NumKong's Python extension.
 *  @file python/dlpack_interop.c
 *  @author Ash Vardanian
 *  @date April 17, 2026
 *
 *  @section What this provides
 *
 *  Two methods on `numkong.Tensor` (`__dlpack__`, `__dlpack_device__`) and the module-level `numkong.from_dlpack(obj)`
 *  constructor implement the cross-framework data-interchange protocol mandated by the Python Array API standard. Once
 *  loaded, a NumKong tensor can be consumed zero-copy by any framework that ships `from_dlpack`, and any framework's
 *  tensor that implements `__dlpack__` can be consumed zero-copy by NumKong.
 *
 *  @section Why DLPack and not the buffer protocol
 *
 *  NumKong already produces and consumes data via PEP 3118 (`Py_buffer`) and NumPy's `__array_interface__` — see the
 *  `@section Buffer Protocol and NumPy Compatibility` block in `python/numkong.c`. Those protocols carry two hard
 *  limits that DLPack solves:
 *
 *  1. @b Device-blind: `Py_buffer` cannot describe GPU/TPU memory. DLPack's `DLDevice` field carries
 *     `device_type`, so a PyTorch CUDA tensor is distinguishable from a CPU one and a NumKong consumer can
 *     refuse it instead of reading a bogus pointer.
 *  2. @b Dtype-blind for bf16/fp8/fp6: NumKong currently fakes these in `Tensor_getbuffer` (see
 *     `python/tensor.c` ~lines 2097-2107) by emitting `format = "H"` / `"B"`. A consumer reading the format
 *     string sees raw bytes, not the semantic dtype. DLPack 1.x carries `kDLBfloat`, `kDLFloat8_e4m3fn`,
 *     `kDLFloat8_e5m2`, and `kDLFloat6_*` codes — the dtype survives across the bridge.
 *
 *  @section Interop partners
 *
 *  Every project below implements the same protocol. Because NumKong now implements it too, NumKong tensors
 *  interchange zero-copy with all of them in either direction (CPU only at this stage; GPU revisited if NumKong ever
 *  ships GPU kernels). PR / commit references are the canonical landing point in each upstream:
 *
 *  - PyTorch — `torch.utils.dlpack` (`to_dlpack` / `from_dlpack`) since pytorch/pytorch#2933 (2017); Array API
 *    `__dlpack__` protocol added in pytorch/pytorch#57110 (2021); upgraded to DLPack 1.0 with `max_version` +
 *    versioned capsule support in pytorch/pytorch#145000 (2024).
 *  - NumPy — `ndarray.__dlpack__` and `np.from_dlpack` added in numpy/numpy#19083 (NumPy 1.22, 2021).
 *  - JAX — `jax.dlpack.from_dlpack` / `jax.dlpack.to_dlpack` (`jax/_src/dlpack.py` in jax-ml/jax).
 *  - CuPy — `cupy.fromDLPack` / `cupy.ndarray.toDLPack` introduced in cupy/cupy#1082; modern `cupy.from_dlpack`
 *    follows the Array API protocol.
 *  - TensorFlow — `tf.experimental.dlpack.{to,from}_dlpack` per RFC tensorflow/community#180; implementation in
 *    `tensorflow/c/eager/dlpack.cc`.
 *  - Apache Arrow / PyArrow — `Array.__dlpack__` and `__dlpack_device__` landed in apache/arrow#33984
 *    (Arrow 15.0.0, 2024).
 *  - ONNX Runtime — DLPack on `OrtValue`, enabled by default for inference in microsoft/onnxruntime#23110.
 *  - spaCy / Thinc — uses DLPack as the cross-framework array conversion path between PyTorch / TensorFlow / CuPy /
 *    NumPy (explosion/thinc#686).
 *  - TVM — original home of the protocol; native producer and consumer.
 *  - Apache MXNet — native DLPack support (`apache/mxnet/python/mxnet/dlpack.py`).
 *  - MLX (Apple silicon) — protocol support landing piecewise; tracking issue ml-explore/mlx#1159.
 *  - NNabla — `nnabla.utils.dlpack.{to,from}_dlpack`.
 *
 *  @section References
 *
 *  https://github.com/dmlc/dlpack
 *  https://data-apis.org/array-api/latest/design_topics/data_interchange.html
 *  https://github.com/data-apis/array-api/pull/106
 *  https://tvm.apache.org/2018/08/10/DLPack-Bridge
 *
 *  @section Implementation notes
 *
 *  - `Tensor.__dlpack__(...)` returns a `PyCapsule` wrapping either a `DLManagedTensor` (capsule name `"dltensor"`,
 *    legacy v0) or a `DLManagedTensorVersioned` (capsule name `"dltensor_versioned"`, v1+). A single
 *    `nk_dlpack_export_ctx_t` covers both paths because the v1 struct embeds the same `DLTensor` fields.
 *  - `numkong.from_dlpack(obj)` accepts either a capsule directly or any object exposing `__dlpack__`. On import, the
 *    capsule is renamed to `"used_dltensor[_versioned]"` per spec so the producer's destructor becomes a no-op;
 *    ownership transfers to a tiny `DLPackOwner` Python object whose `tp_dealloc` invokes the producer's deleter
 *    exactly once.
 *  - Strides: NumKong keeps strides in bytes; DLPack keeps them in elements. Conversion divides / multiplies by
 *    `nk_dtype_bytes_per_value`. DLPack 1.2+ requires non-NULL strides whenever ndim > 0 — exporter conforms.
 *  - FP6 byte-padded layout: NumKong stores `nk_e2m3_t` / `nk_e3m2_t` as one byte each (low 6 bits carry the value).
 *    DLPack expresses this with `{kDLFloat6_*, 6, 1}` plus the `IS_SUBBYTE_TYPE_PADDED` flag, which only exists on
 *    the versioned struct, so the exporter silently upgrades to v1 for FP6 even if the consumer didn't request it.
 *  - Sub-byte packed types (`u1`, `u4`, `i4`) are exchanged as byte containers (`{kDLUInt, 8, 1}` /
 *    `{kDLInt, 8, 1}`), matching NumKong's shape convention (`shape[i]` counts bytes). The semantic sub-byte nature
 *    is NumKong-specific metadata lost across the bridge — round trips preserve data but land as `u8` / `i8` without
 *    a manual astype.
 *
 *  @section Importer device acceptance
 *
 *  Any DLPack capsule whose `device.device_type` corresponds to a host-dereferenceable pointer is accepted. The
 *  exporter still always reports `kDLCPU`; this widening only affects what `numkong.from_dlpack` will consume.
 *
 *  @par Accepted — pointer is CPU-readable
 *  - `kDLCPU`         — plain host memory (default).
 *  - `kDLCUDAHost`    — `cudaMallocHost` pinned host memory; semantically equivalent to `kDLCPU`.
 *  - `kDLROCMHost`    — AMD ROCm pinned host equivalent.
 *  - `kDLCUDAManaged` — `cudaMallocManaged` unified memory. The first CPU access triggers a page-migration
 *                       round-trip from the GPU; correct but potentially expensive if the producer expected
 *                       the data to stay GPU-resident.
 *  - `kDLOneAPI`      — Intel oneAPI USM. Host and shared USM are CPU-readable; device-only USM is not, but
 *                       the device code doesn't distinguish — accepting all and faulting on bad pointers
 *                       matches what other consumers do.
 *  - `kDLMetal`       — Metal buffer. On Apple Silicon's unified-memory SoC the pointer is host-readable;
 *                       on Intel-Mac dGPU it isn't. The Apple Silicon case is the practical user (MLX).
 *
 *  @par Rejected — pure device memory
 *  - `kDLCUDA`, `kDLROCM`, `kDLOpenCL`, `kDLVulkan`, `kDLWebGPU`, `kDLHexagon`, `kDLMAIA`, `kDLTrn`,
 *    `kDLVPI`, `kDLExtDev` — error names the device code so the caller can debug.
 */
#include "dlpack_interop.h"
#include "dlpack_abi.h"

#include <stdint.h>
#include <string.h>

/** @brief Map a NumKong dtype to a DLPack `DLDataType`.
 *
 *  Returns `{code=0, bits=0, lanes=0}` for unsupported dtypes. The caller
 *  must check for `bits == 0` to detect failure.
 *
 *  When @p versioned is non-zero, byte-padded FP6 is emitted as
 *  `{kDLFloat6_*, 6, 1}` (caller sets the `IS_SUBBYTE_TYPE_PADDED` flag).
 *  When @p versioned is zero, FP6 is rejected (returns all-zero).
 */
static DLDataType nk_dtype_to_dl(nk_dtype_t dtype, int versioned) {
    DLDataType none = {0, 0, 0};
    switch (dtype) {
    case nk_f64_k: return (DLDataType) {kDLFloat, 64, 1};
    case nk_f32_k: return (DLDataType) {kDLFloat, 32, 1};
    case nk_f16_k: return (DLDataType) {kDLFloat, 16, 1};
    case nk_bf16_k: return (DLDataType) {kDLBfloat, 16, 1};
    case nk_e4m3_k: return (DLDataType) {kDLFloat8_e4m3fn, 8, 1};
    case nk_e5m2_k: return (DLDataType) {kDLFloat8_e5m2, 8, 1};
    case nk_e2m3_k: return versioned ? (DLDataType) {kDLFloat6_e2m3fn, 6, 1} : none;
    case nk_e3m2_k: return versioned ? (DLDataType) {kDLFloat6_e3m2fn, 6, 1} : none;
    // Block-scaled element & scale bytes. E2M1 is a 4-bit sub-byte element; like FP6 it is
    // byte-padded in our storage, so it needs a versioned capsule + the SUBBYTE_PADDED flag.
    // UE8M0 (MX pow-2 scale) is a whole byte. UE4M3 (NVFP4 scale) has no dedicated DLPack code,
    // so its byte is exported through the E4M3 view (it is an E4M3 byte with the sign forced 0).
    case nk_e2m1_k: return versioned ? (DLDataType) {kDLFloat4_e2m1fn, 4, 1} : none;
    case nk_ue8m0_k: return (DLDataType) {kDLFloat8_e8m0fnu, 8, 1};
    case nk_ue4m3_k: return (DLDataType) {kDLFloat8_e4m3fn, 8, 1};
    case nk_i8_k: return (DLDataType) {kDLInt, 8, 1};
    case nk_i16_k: return (DLDataType) {kDLInt, 16, 1};
    case nk_i32_k: return (DLDataType) {kDLInt, 32, 1};
    case nk_i64_k: return (DLDataType) {kDLInt, 64, 1};
    case nk_u8_k: return (DLDataType) {kDLUInt, 8, 1};
    case nk_u16_k: return (DLDataType) {kDLUInt, 16, 1};
    case nk_u32_k: return (DLDataType) {kDLUInt, 32, 1};
    case nk_u64_k: return (DLDataType) {kDLUInt, 64, 1};
    case nk_f64c_k: return (DLDataType) {kDLComplex, 128, 1};
    case nk_f32c_k: return (DLDataType) {kDLComplex, 64, 1};
    case nk_f16c_k: return (DLDataType) {kDLComplex, 32, 1};
    case nk_bf16c_k: return (DLDataType) {kDLComplex, 32, 1};
    // Byte-container view for NumKong's sub-byte packed types. Semantic
    // dtype (u1/u4/i4) is NumKong-internal and not representable in DLPack
    // without expanding the shape to logical element count.
    case nk_u1_k: return (DLDataType) {kDLUInt, 8, 1};
    case nk_u4_k: return (DLDataType) {kDLUInt, 8, 1};
    case nk_i4_k: return (DLDataType) {kDLInt, 8, 1};
    default: return none;
    }
}

/** @brief Map a DLPack `DLDataType` to a NumKong dtype.
 *
 *  Returns `nk_dtype_unknown_k` if the combination isn't supported.
 *  `flags` is the `DLManagedTensorVersioned.flags` field (0 for legacy
 *  capsules) — used to distinguish padded from packed FP6.
 */
static nk_dtype_t nk_dl_to_dtype(DLDataType dl, uint64_t flags) {
    if (dl.lanes != 1) return nk_dtype_unknown_k;
    switch (dl.code) {
    case kDLInt:
        switch (dl.bits) {
        case 8: return nk_i8_k;
        case 16: return nk_i16_k;
        case 32: return nk_i32_k;
        case 64: return nk_i64_k;
        default: return nk_dtype_unknown_k;
        }
    case kDLUInt:
        switch (dl.bits) {
        case 8: return nk_u8_k;
        case 16: return nk_u16_k;
        case 32: return nk_u32_k;
        case 64: return nk_u64_k;
        default: return nk_dtype_unknown_k;
        }
    case kDLFloat:
        switch (dl.bits) {
        case 16: return nk_f16_k;
        case 32: return nk_f32_k;
        case 64: return nk_f64_k;
        default: return nk_dtype_unknown_k;
        }
    case kDLBfloat: return dl.bits == 16 ? nk_bf16_k : nk_dtype_unknown_k;
    case kDLComplex:
        switch (dl.bits) {
        case 32: return nk_f16c_k; // ambiguous with bf16c, prefer f16c
        case 64: return nk_f32c_k;
        case 128: return nk_f64c_k;
        default: return nk_dtype_unknown_k;
        }
    case kDLFloat8_e4m3fn: return dl.bits == 8 ? nk_e4m3_k : nk_dtype_unknown_k;
    case kDLFloat8_e5m2: return dl.bits == 8 ? nk_e5m2_k : nk_dtype_unknown_k;
    case kDLFloat6_e2m3fn:
        // Only accept byte-padded FP6 (our storage layout).
        return (dl.bits == 6 && (flags & DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED)) ? nk_e2m3_k : nk_dtype_unknown_k;
    case kDLFloat6_e3m2fn:
        return (dl.bits == 6 && (flags & DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED)) ? nk_e3m2_k : nk_dtype_unknown_k;
    case kDLFloat4_e2m1fn:
        // Only accept byte-padded FP4 (our storage layout), matching the FP6 sub-byte handling.
        return (dl.bits == 4 && (flags & DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED)) ? nk_e2m1_k : nk_dtype_unknown_k;
    case kDLFloat8_e8m0fnu: return dl.bits == 8 ? nk_ue8m0_k : nk_dtype_unknown_k;
    default: return nk_dtype_unknown_k;
    }
}

/** @brief Manager context held by the DLPack capsule we produce.
 *
 *  Holds either a `DLManagedTensor` (legacy) or `DLManagedTensorVersioned`
 *  depending on which the consumer requested. Both paths keep a refcount
 *  on the source Tensor to pin its memory for the capsule's lifetime.
 */
typedef struct nk_dlpack_export_ctx_t {
    union {
        DLManagedTensor legacy;
        DLManagedTensorVersioned versioned;
    } managed;
    int is_versioned;
    PyObject *owner; // source Tensor, Py_INCREF'd
    int64_t shape[NK_TENSOR_MAX_RANK];
    int64_t strides[NK_TENSOR_MAX_RANK];
} nk_dlpack_export_ctx_t;

/** @brief Drop the owner refcount and free the export context.
 *
 *  Reused by both legacy and versioned deleters via the shared `manager_ctx`. The GIL must be held across @b both
 *  operations: `Py_XDECREF` is obvious, but `PyMem_Free` also requires the GIL because pymalloc dispatches through
 *  CPython state. Consumers (e.g. PyTorch's c10 storage finalizer) invoke this deleter from a path that doesn't hold
 *  the GIL by default, so we re-acquire it here. */
static void nk_dlpack_release_ctx(nk_dlpack_export_ctx_t *ctx) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_XDECREF(ctx->owner);
    PyMem_Free(ctx);
    PyGILState_Release(gstate);
}

static void nk_dlpack_legacy_deleter(DLManagedTensor *self) {
    nk_dlpack_release_ctx((nk_dlpack_export_ctx_t *)self->manager_ctx);
}

static void nk_dlpack_versioned_deleter(DLManagedTensorVersioned *self) {
    nk_dlpack_release_ctx((nk_dlpack_export_ctx_t *)self->manager_ctx);
}

/** @brief Called by CPython when the capsule itself is GC'd without being consumed.
 *
 *  If the consumer stole the capsule (renaming it to `"used_dltensor"` or
 *  `"used_dltensor_versioned"`) we never run — the consumer is responsible
 *  for calling the deleter. Otherwise we invoke the deleter ourselves.
 */
static void nk_dlpack_capsule_destructor(PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor *managed_legacy = (DLManagedTensor *)PyCapsule_GetPointer(capsule, "dltensor");
        if (managed_legacy && managed_legacy->deleter) managed_legacy->deleter(managed_legacy);
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        DLManagedTensorVersioned *managed_versioned = (DLManagedTensorVersioned *)PyCapsule_GetPointer(
            capsule, "dltensor_versioned");
        if (managed_versioned && managed_versioned->deleter) managed_versioned->deleter(managed_versioned);
    }
    // Renamed "used_*" capsules are owned by the consumer — do nothing.
}

static int nk_fill_dl_tensor(Tensor *tensor, DLTensor *out, nk_dlpack_export_ctx_t *ctx, int versioned) {
    DLDataType dl_dtype = nk_dtype_to_dl(tensor->dtype, versioned);
    if (dl_dtype.bits == 0) {
        if (!versioned && (tensor->dtype == nk_e2m3_k || tensor->dtype == nk_e3m2_k || tensor->dtype == nk_e2m1_k)) {
            PyErr_SetString( //
                PyExc_TypeError,
                "Sub-byte FP4/FP6 (e2m1/e2m3/e3m2) DLPack export requires max_version >= (1, 0) so " "the " "IS_"
                                                                                                            "SUBBYTE_"
                                                                                                            "TYPE_" "PA"
                                                                                                                    "DD"
                                                                                                                    "ED"
                                                                                                                    " f"
                                                                                                                    "la"
                                                                                                                    "g "
                                                                                                                    "ca"
                                                                                                                    "n " "be set");
        }
        else { PyErr_Format(PyExc_TypeError, "dtype %d has no DLPack mapping", (int)tensor->dtype); }
        return -1;
    }

    size_t const item_size = nk_dtype_bytes_per_value(tensor->dtype);
    if (item_size == 0) {
        PyErr_Format(PyExc_TypeError, "dtype %d has zero item size", (int)tensor->dtype);
        return -1;
    }

    for (size_t i = 0; i < tensor->rank; i++) {
        ctx->shape[i] = (int64_t)tensor->shape[i];
        if (tensor->strides[i] % (Py_ssize_t)item_size != 0) {
            PyErr_Format( //
                PyExc_BufferError,
                "stride %zd along dim %zu is not a multiple of item size %zu; cannot express in DLPack element strides",
                (Py_ssize_t)tensor->strides[i], i, item_size);
            return -1;
        }
        ctx->strides[i] = (int64_t)(tensor->strides[i] / (Py_ssize_t)item_size);
    }

    out->data = tensor->data;
    out->device = (DLDevice) {kDLCPU, 0};
    out->ndim = (int32_t)tensor->rank;
    out->dtype = dl_dtype;
    // DLPack 1.2+ requires shape/strides to be non-NULL whenever ndim != 0; both NULL is only valid for rank-0.
    out->shape = tensor->rank > 0 ? ctx->shape : NULL;
    out->strides = tensor->rank > 0 ? ctx->strides : NULL;
    out->byte_offset = 0;
    return 0;
}

/** @brief Build a capsule wrapping @p tensor. */
static PyObject *nk_build_capsule(Tensor *tensor, int versioned, uint64_t flags) {
    nk_dlpack_export_ctx_t *ctx = (nk_dlpack_export_ctx_t *)PyMem_Malloc(sizeof(*ctx));
    if (!ctx) return PyErr_NoMemory();
    memset(ctx, 0, sizeof(*ctx));
    ctx->is_versioned = versioned;

    DLTensor *dl_tensor = versioned ? &ctx->managed.versioned.dl_tensor : &ctx->managed.legacy.dl_tensor;
    if (nk_fill_dl_tensor(tensor, dl_tensor, ctx, versioned) != 0) {
        PyMem_Free(ctx);
        return NULL;
    }

    char const *capsule_name;
    if (versioned) {
        ctx->managed.versioned.version.major = DLPACK_MAJOR_VERSION;
        ctx->managed.versioned.version.minor = DLPACK_MINOR_VERSION;
        ctx->managed.versioned.manager_ctx = ctx;
        ctx->managed.versioned.deleter = nk_dlpack_versioned_deleter;
        ctx->managed.versioned.flags = flags;
        capsule_name = "dltensor_versioned";
    }
    else {
        ctx->managed.legacy.manager_ctx = ctx;
        ctx->managed.legacy.deleter = nk_dlpack_legacy_deleter;
        capsule_name = "dltensor";
    }

    ctx->owner = (PyObject *)tensor;
    Py_INCREF(tensor);

    void *managed_ptr = versioned ? (void *)&ctx->managed.versioned : (void *)&ctx->managed.legacy;
    PyObject *capsule = PyCapsule_New(managed_ptr, capsule_name, nk_dlpack_capsule_destructor);
    if (!capsule) {
        Py_DECREF(tensor);
        PyMem_Free(ctx);
        return NULL;
    }
    return capsule;
}

char const doc_dlpack[] =                                                                                      //
    "Return a DLPack capsule for zero-copy exchange with any framework that implements the Python Array API\n" //
    "DLPack protocol.\n\n"                                                                                     //
    "Parameters:\n"                                                                                            //
    "    stream: Ignored on CPU (accepted for Array API compatibility).\n"                                     //
    "    max_version: Optional (major, minor) tuple. If major >= 1, a versioned DLPack capsule is produced\n"  //
    "        ('dltensor_versioned'); otherwise a legacy one ('dltensor').\n"                                   //
    "    dl_device: Optional (device_type, device_id). Must be (kDLCPU=1, 0) if provided.\n"                   //
    "    copy: Optional bool. Only copy=False (or None) is supported; copy=True raises.\n\n"                   //
    "Notes:\n"                                                                                                 //
    "    FP6 dtypes (e2m3, e3m2) silently upgrade to versioned DLPack because they use the\n"                  //
    "    IS_SUBBYTE_TYPE_PADDED flag only available in v1+.\n\n"                                               //
    "Verified interop with:\n"                                                                                 //
    "    - PyTorch       (`torch.from_dlpack`,           PRs pytorch/pytorch#2933, #57110, #145000)\n"         //
    "    - NumPy         (`np.from_dlpack`,              PR numpy/numpy#19083, NumPy >= 1.22)\n"               //
    "    - JAX           (`jax.dlpack.from_dlpack`)\n"                                                         //
    "    - CuPy          (`cupy.from_dlpack`,            PR cupy/cupy#1082)\n"                                 //
    "    - TensorFlow    (`tf.experimental.dlpack.from_dlpack`, RFC tensorflow/community#180)\n"               //
    "    - PyArrow       (`Array.__dlpack__`,            PR apache/arrow#33984, Arrow >= 15.0.0)\n"            //
    "    - ONNX Runtime  (`OrtValue` DLPack,             PR microsoft/onnxruntime#23110)\n"                    //
    "    - spaCy / Thinc (cross-framework converter,     PR explosion/thinc#686)\n"                            //
    "    - TVM, Apache MXNet, MLX, NNabla\n\n"                                                                 //
    "Signature:\n"                                                                                             //
    "    >>> def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None): ...";

char const doc_dlpack_device[] =                                                                                 //
    "Return the DLPack device tuple for this tensor. Always (1, 0) (kDLCPU).\n\n"                                //
    "Part of the Python Array API DLPack protocol; consumers (PyTorch, NumPy, JAX, CuPy, TensorFlow, PyArrow,\n" //
    "etc.) call this before `__dlpack__` to know whether a copy or stream sync is needed before crossing the\n"  //
    "framework boundary.\n\n"                                                                                    //
    "Signature:\n"                                                                                               //
    "    >>> def __dlpack_device__(self, /): ...";

PyObject *Tensor_dlpack(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char const *kwlist[] = {"stream", "max_version", "dl_device", "copy", NULL};
    PyObject *stream = Py_None, *max_version = Py_None, *dl_device = Py_None, *copy = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$OOOO", (char **)kwlist, //
                                     &stream, &max_version, &dl_device, &copy))
        return NULL;
    nk_unused_(stream); // CPU-only: stream synchronization is a no-op.

    Tensor *tensor = (Tensor *)self;

    int const requested_copy = (copy != Py_None) ? PyObject_IsTrue(copy) : 0;
    if (requested_copy < 0) return NULL; // bool(copy) raised
    if (requested_copy) {
        PyErr_SetString(PyExc_BufferError, "NumKong DLPack exporter only supports zero-copy (copy=False)");
        return NULL;
    }

    if (dl_device != Py_None) {
        if (!PyTuple_Check(dl_device) || PyTuple_GET_SIZE(dl_device) != 2) {
            PyErr_SetString(PyExc_TypeError, "dl_device must be a (device_type, device_id) tuple");
            return NULL;
        }
        long requested_device_type = PyLong_AsLong(PyTuple_GET_ITEM(dl_device, 0));
        long requested_device_id = PyLong_AsLong(PyTuple_GET_ITEM(dl_device, 1));
        if (PyErr_Occurred()) return NULL;
        if (requested_device_type != kDLCPU || requested_device_id != 0) {
            PyErr_SetString(PyExc_BufferError, "NumKong DLPack exporter only supports (kDLCPU, 0)");
            return NULL;
        }
    }

    int want_versioned = 0;
    if (max_version != Py_None) {
        if (!PyTuple_Check(max_version) || PyTuple_GET_SIZE(max_version) < 1) {
            PyErr_SetString(PyExc_TypeError, "max_version must be a (major, minor) tuple");
            return NULL;
        }
        long major = PyLong_AsLong(PyTuple_GET_ITEM(max_version, 0));
        if (PyErr_Occurred()) return NULL;
        if (major >= 1) want_versioned = 1;
    }

    // FP6 requires the IS_SUBBYTE_TYPE_PADDED flag, only available on
    // versioned capsules. If the consumer didn't opt in to v1, upgrade
    // silently — better to exchange correctly than to refuse.
    int needs_versioned = (tensor->dtype == nk_e2m3_k || tensor->dtype == nk_e3m2_k || tensor->dtype == nk_e2m1_k);
    int versioned = want_versioned || needs_versioned;
    uint64_t flags = needs_versioned ? DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED : 0;

    return nk_build_capsule(tensor, versioned, flags);
}

PyObject *Tensor_dlpack_device(PyObject *self, PyObject *noargs) {
    nk_unused_(self);
    nk_unused_(noargs);
    return Py_BuildValue("(ii)", (int)kDLCPU, 0);
}

/** @brief Owner object that wraps an imported DLPack capsule and calls the
 *         producer's deleter on `tp_dealloc`. Never exposed to Python code. */
typedef struct {
    PyObject_HEAD void *managed; // DLManagedTensor* or DLManagedTensorVersioned*
    int is_versioned;
} nk_dlpack_owner_t;

static void nk_dlpack_owner_dealloc(PyObject *self) {
    nk_dlpack_owner_t *owner = (nk_dlpack_owner_t *)self;
    if (owner->managed) {
        if (owner->is_versioned) {
            DLManagedTensorVersioned *managed_versioned = (DLManagedTensorVersioned *)owner->managed;
            if (managed_versioned->deleter) managed_versioned->deleter(managed_versioned);
        }
        else {
            DLManagedTensor *managed_legacy = (DLManagedTensor *)owner->managed;
            if (managed_legacy->deleter) managed_legacy->deleter(managed_legacy);
        }
        owner->managed = NULL;
    }
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject DLPackOwnerType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong._DLPackOwner",
    .tp_doc = "Internal owner for imported DLPack capsules. Not constructible from Python.",
    .tp_basicsize = sizeof(nk_dlpack_owner_t),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = nk_dlpack_owner_dealloc,
};

int nk_dlpack_init(PyObject *module) {
    nk_unused_(module); // DLPackOwner is internal - not added to the module namespace.
    if (PyType_Ready(&DLPackOwnerType) < 0) return -1;
    return 0;
}

char const doc_from_dlpack[] =                                                                                      //
    "Consume a DLPack capsule (or any object implementing __dlpack__) as a NumKong Tensor.\n\n"                     //
    "Creates a zero-copy view; the underlying memory stays alive as long as the returned Tensor does. Implements\n" //
    "the consumer side of the Python Array API DLPack protocol (data-apis/array-api#106).\n\n"                      //
    "Parameters:\n"                                                                                                 //
    "    obj: A PyCapsule named 'dltensor' or 'dltensor_versioned', or any object with an __dlpack__ method.\n\n"   //
    "Returns:\n"                                                                                                    //
    "    Tensor: A NumKong view sharing memory with the producer.\n\n"                                              //
    "Accepts any DLPack device whose pointer is CPU-readable:\n"                                                    //
    "    - kDLCPU        — plain host memory.\n"                                                                    //
    "    - kDLCUDAHost   — cudaMallocHost pinned host memory.\n"                                                    //
    "    - kDLROCMHost   — AMD ROCm pinned host equivalent.\n"                                                      //
    "    - kDLCUDAManaged — cudaMallocManaged unified memory (first CPU touch migrates pages).\n"                   //
    "    - kDLOneAPI     — Intel oneAPI USM (host / shared variants only).\n"                                       //
    "    - kDLMetal      — Apple Silicon unified memory (used by MLX).\n"                                           //
    "Pure device memory (kDLCUDA, kDLROCM, kDLOpenCL, kDLVulkan, kDLWebGPU, kDLHexagon, kDLMAIA, kDLTrn) is\n"      //
    "rejected with a clear ValueError naming the device code.\n\n"                                                  //
    "Verified producers (zero-copy round-trip in tests):\n"                                                         //
    "    - PyTorch       `torch.Tensor.__dlpack__`            (pytorch/pytorch#57110, #145000)\n"                   //
    "    - NumPy         `ndarray.__dlpack__`                 (numpy/numpy#19083, NumPy >= 1.22)\n"                 //
    "    - JAX           `jax.Array.__dlpack__`\n"                                                                  //
    "    - CuPy          `cupy.ndarray.__dlpack__`            (cupy/cupy#1082)\n"                                   //
    "    - TensorFlow    `tf.experimental.dlpack.to_dlpack`   (RFC tensorflow/community#180)\n"                     //
    "    - PyArrow       `Array.__dlpack__`                   (apache/arrow#33984, Arrow >= 15.0.0)\n"              //
    "    - ONNX Runtime  `OrtValue` DLPack                    (microsoft/onnxruntime#23110)\n"                      //
    "    - spaCy / Thinc cross-framework converter            (explosion/thinc#686)\n"                              //
    "    - MLX           `mx.array.__dlpack__`                (Apple Silicon, ml-explore/mlx#1159)\n"               //
    "    - TVM, Apache MXNet, NNabla\n\n"                                                                           //
    "Signature:\n"                                                                                                  //
    "    >>> def from_dlpack(obj, /): ...";

PyObject *api_from_dlpack(PyObject *self, PyObject *obj) {
    nk_unused_(self);

    // Accept either a raw capsule or an object implementing __dlpack__.
    PyObject *capsule = NULL;
    int capsule_owned_by_us = 0;
    if (PyCapsule_CheckExact(obj)) {
        capsule = obj;
        Py_INCREF(capsule);
        capsule_owned_by_us = 1;
    }
    else {
        PyObject *dlpack_fn = PyObject_GetAttrString(obj, "__dlpack__");
        if (!dlpack_fn) {
            PyErr_SetString(PyExc_TypeError, "from_dlpack: argument is not a capsule and has no __dlpack__ method");
            return NULL;
        }
        // Pass max_version=(1, 0) so versioning-aware producers give us FP6 flags etc.
        PyObject *empty_args = PyTuple_New(0);
        PyObject *max_version = Py_BuildValue("(ii)", 1, 0);
        PyObject *kwargs = PyDict_New();
        if (empty_args && max_version && kwargs) {
            PyDict_SetItemString(kwargs, "max_version", max_version);
            capsule = PyObject_Call(dlpack_fn, empty_args, kwargs);
        }
        Py_XDECREF(kwargs);
        Py_XDECREF(max_version);
        if (!capsule && empty_args) {
            // Older producers reject unknown kwargs - retry bare.
            PyErr_Clear();
            capsule = PyObject_CallObject(dlpack_fn, NULL);
        }
        Py_XDECREF(empty_args);
        Py_DECREF(dlpack_fn);
        if (!capsule) return NULL;
        capsule_owned_by_us = 1;
    }

    int is_versioned = 0;
    DLTensor *dl_tensor = NULL;
    uint64_t flags = 0;

    if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        DLManagedTensorVersioned *managed_versioned = (DLManagedTensorVersioned *)PyCapsule_GetPointer(
            capsule, "dltensor_versioned");
        if (!managed_versioned) goto fail;
        if (managed_versioned->version.major != DLPACK_MAJOR_VERSION) {
            PyErr_Format(PyExc_ValueError, "Unsupported DLPack major version %u (expected %u)", //
                         managed_versioned->version.major, DLPACK_MAJOR_VERSION);
            goto fail;
        }
        dl_tensor = &managed_versioned->dl_tensor;
        flags = managed_versioned->flags;
        is_versioned = 1;
    }
    else if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor *managed_legacy = (DLManagedTensor *)PyCapsule_GetPointer(capsule, "dltensor");
        if (!managed_legacy) goto fail;
        dl_tensor = &managed_legacy->dl_tensor;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "from_dlpack: expected a capsule named 'dltensor' or 'dltensor_versioned'");
        goto fail;
    }

    // Accept any device whose pointer is dereferenceable from host code. The exporter still hands out plain CPU
    // memory only; this widening is a consumer-side concession so callers can pass pinned-host, unified, oneAPI
    // shared-USM, and Apple-Silicon Metal buffers without an explicit conversion step.
    switch (dl_tensor->device.device_type) {
    case kDLCPU:         // plain host memory
    case kDLCUDAHost:    // cudaMallocHost — pinned host memory
    case kDLROCMHost:    // AMD pinned host equivalent
    case kDLCUDAManaged: // cudaMallocManaged — first CPU touch migrates pages
    case kDLOneAPI:      // Intel USM (host / shared variants are CPU-readable; device-only USM would fault)
    case kDLMetal:       // Apple Silicon: unified memory; intel-Mac dGPU: would fault
        break;
    default:
        PyErr_Format( //
            PyExc_ValueError,
            "NumKong from_dlpack: device_type=%d is not CPU-accessible "    //
            "(only kDLCPU/CUDAHost/ROCMHost/CUDAManaged/OneAPI/Metal are)", //
            (int)dl_tensor->device.device_type);
        goto fail;
    }

    if (dl_tensor->ndim < 0 || (size_t)dl_tensor->ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "DLPack tensor rank %d out of range (max %d)", //
                     (int)dl_tensor->ndim, NK_TENSOR_MAX_RANK);
        goto fail;
    }

    nk_dtype_t dtype = nk_dl_to_dtype(dl_tensor->dtype, flags);
    if (dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "unsupported DLPack dtype (code=%u, bits=%u, lanes=%u)", //
                     (unsigned)dl_tensor->dtype.code, (unsigned)dl_tensor->dtype.bits,
                     (unsigned)dl_tensor->dtype.lanes);
        goto fail;
    }

    size_t const item_size = nk_dtype_bytes_per_value(dtype);

    // Validate the producer-supplied geometry before exposing a view (the rank is already bounded above):
    // reject negative extents and a NULL data pointer for a non-empty tensor. Failing here leaves the
    // capsule un-renamed, so its owner keeps ownership and there is no double-free.
    size_t dl_element_count = 1;
    for (int axis = 0; axis < dl_tensor->ndim; ++axis) {
        if (dl_tensor->shape[axis] < 0) {
            PyErr_Format(PyExc_ValueError, "DLPack tensor has a negative extent %lld on axis %d",
                         (long long)dl_tensor->shape[axis], axis);
            goto fail;
        }
        dl_element_count *= (size_t)dl_tensor->shape[axis];
    }
    if (dl_element_count > 0 && dl_tensor->data == NULL) {
        PyErr_SetString(PyExc_ValueError, "DLPack tensor has a NULL data pointer with a non-zero element count");
        goto fail;
    }
    // Reject a producer element-stride whose byte stride (stride x item_size) would overflow Py_ssize_t.
    // Strides may be legitimately negative (reversed axes), so the check is on the magnitude.
    if (dl_tensor->strides && item_size != 0) {
        for (int axis = 0; axis < dl_tensor->ndim; ++axis) {
            int64_t const element_stride = dl_tensor->strides[axis];
            uint64_t const stride_magnitude = element_stride < 0 ? -(uint64_t)element_stride : (uint64_t)element_stride;
            if (stride_magnitude > (uint64_t)PY_SSIZE_T_MAX / item_size) {
                PyErr_Format(PyExc_ValueError, "DLPack tensor byte stride overflows Py_ssize_t on axis %d", axis);
                goto fail;
            }
        }
    }

    // Translate the capsule into a NumKong view. The owner object holds the
    // DLManagedTensor(Versioned) and invokes its deleter on finalization.
    nk_dlpack_owner_t *owner = PyObject_New(nk_dlpack_owner_t, &DLPackOwnerType);
    if (!owner) goto fail;
    owner->is_versioned = is_versioned;
    // Rename the capsule so the producer's destructor becomes a no-op and ownership transfers to `owner`.
    // If the rename fails, leave the capsule's original name so the producer still frees it, and null
    // `owner->managed` so our dealloc does not free it a second time.
    if (is_versioned) {
        owner->managed = PyCapsule_GetPointer(capsule, "dltensor_versioned");
        if (PyCapsule_SetName(capsule, "used_dltensor_versioned") != 0) {
            owner->managed = NULL;
            Py_DECREF(owner);
            goto fail;
        }
    }
    else {
        owner->managed = PyCapsule_GetPointer(capsule, "dltensor");
        if (PyCapsule_SetName(capsule, "used_dltensor") != 0) {
            owner->managed = NULL;
            Py_DECREF(owner);
            goto fail;
        }
    }

    // Build the Tensor view directly (parallels Tensor_view_object in tensor.c,
    // which is static; duplicating the 10-line body avoids a cross-TU public).
    Tensor *view = PyObject_NewVar(Tensor, &TensorType, 0);
    if (!view) {
        Py_DECREF(owner);
        goto fail;
    }
    view->dtype = dtype;
    view->rank = (size_t)dl_tensor->ndim;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        view->shape[i] = 0;
        view->strides[i] = 0;
    }
    for (size_t i = 0; i < view->rank; i++) view->shape[i] = (Py_ssize_t)dl_tensor->shape[i];
    if (dl_tensor->strides) {
        for (size_t i = 0; i < view->rank; i++)
            view->strides[i] = (Py_ssize_t)dl_tensor->strides[i] * (Py_ssize_t)item_size;
    }
    else if (view->rank > 0) {
        // Compact row-major: compute strides in bytes.
        view->strides[view->rank - 1] = (Py_ssize_t)item_size;
        for (size_t i = view->rank - 1; i > 0; i--) view->strides[i - 1] = view->strides[i] * view->shape[i];
    }
    view->parent = (PyObject *)owner; // steals the owner's reference
    view->data = (char *)dl_tensor->data + dl_tensor->byte_offset;

    if (capsule_owned_by_us) Py_DECREF(capsule);
    return (PyObject *)view;

fail:
    if (capsule_owned_by_us) Py_DECREF(capsule);
    return NULL;
}
