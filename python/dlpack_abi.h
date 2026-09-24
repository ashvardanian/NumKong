/**
 *  @file python/dlpack_abi.h
 *  @author Ash Vardanian
 *  @date April 17, 2026
 *  @brief Minimal NumKong-authored declarations of the DLPack ABI we use.
 *
 *  This header declares only the subset of the DLPack 1.3 ABI that NumKong actually consumes or
 *  produces. We intentionally do not vendor upstream `dmlc/dlpack/include/dlpack/dlpack.h`: keeping
 *  the surface narrow means every newly-supported dtype, device, or flag is a deliberate,
 *  reviewable change rather than a side-effect of a third-party header sync.
 *
 *  Binary layout matches `dmlc/dlpack` v1.x. `DLPackVersion.major` is checked against
 *  @c DLPACK_MAJOR_VERSION at runtime in `python/dlpack_interop.c` so a future major bump is
 *  rejected with a clear error rather than read with the wrong layout. Minor bumps are additive —
 *  older minors stay readable.
 *
 *  NumPy follows the same approach, see `numpy/_core/src/multiarray/dlpack.c`: declare only the
 *  types you actually exchange, runtime-check the version.
 *
 *  @see DLPack repository: https://github.com/dmlc/dlpack
 *  @see DLPack v1.3 release: https://github.com/dmlc/dlpack/releases/tag/v1.3
 *  @see Array API data interchange: https://data-apis.org/array-api/latest/design_topics/data_interchange.html
 */
#ifndef NK_PYTHON_DLPACK_ABI_H
#define NK_PYTHON_DLPACK_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** DLPack ABI version this binding targets, v1.3, Jan 2026. */
#define DLPACK_MAJOR_VERSION 1
#define DLPACK_MINOR_VERSION 3

/** `DLManagedTensorVersioned.version` — runtime ABI handshake. */
typedef struct {
    uint32_t major;
    uint32_t minor;
} DLPackVersion;

/**
 *  @brief Device identifier carried by every DLPack tensor.
 *
 *  NumKong only handles @c kDLCPU, value 1. The non-CPU values are declared so the rejected-device
 *  error message in @c from_dlpack can name the caller's device type. Numeric values match upstream
 *  `dmlc/dlpack/include/dlpack/dlpack.h`.
 */
typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
    kDLMAIA = 17,
    kDLTrn = 18,
} DLDeviceType;

/** Device handle pairing the device kind with its index. Always `(kDLCPU, 0)` for NumKong-produced
 *  tensors; non-CPU values are accepted only to be rejected with a clear @c from_dlpack message. */
typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

/**
 *  @brief Type-code subset NumKong maps to its own dtypes.
 *
 *  Codes deliberately omitted, never emitted or accepted, are 3 for @c kDLOpaqueHandle, 6 for
 *  @c kDLBool, 7 for @c kDLFloat8_e3m4, 8 for @c kDLFloat8_e4m3, 9 for @c kDLFloat8_e4m3b11fnuz, 11
 *  for @c kDLFloat8_e4m3fnuz, and 13 for @c kDLFloat8_e5m2fnuz.
 *
 *  Adding any of those is a deliberate, reviewable change.
 *
 *  Codes 14 for @c kDLFloat8_e8m0fnu and 17 for @c kDLFloat4_e2m1fn carry the block-scaled element
 *  and scale bytes — MX UE8M0 scale, FP4 E2M1 element. We exchange the raw element buffer; the
 *  per-block scales and any per-tensor global travel separately, through the `block_scaled_*`
 *  module's own functions, never here.
 */
typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLFloat8_e4m3fn = 10,
    kDLFloat8_e5m2 = 12,
    kDLFloat8_e8m0fnu = 14,
    kDLFloat6_e2m3fn = 15,
    kDLFloat6_e3m2fn = 16,
    kDLFloat4_e2m1fn = 17,
} DLDataTypeCode;

/** Compact tensor element-type descriptor. The triple `(code, bits, lanes)` is enough for any
 *  consumer to compute element size as `(bits * lanes + 7) / 8` bytes — sub-byte types round up. */
typedef struct {

    /** One of @c DLDataTypeCode. @c uint8_t mirrors upstream's compact layout. */
    uint8_t code;

    /** Storage bits per element, e.g. 32 for f32, 6 for byte-padded fp6. */
    uint8_t bits;

    /** Vector lane count. NumKong only handles scalar, lanes = 1. */
    uint16_t lanes;
} DLDataType;

/**
 *  @brief The plain-data tensor view exchanged by both legacy and versioned wrappers.
 *
 *  NumKong's exporter always emits non-NULL @p strides when @p ndim > 0, matching DLPack 1.2's
 *  mandatory-stride rule. For @p ndim == 0 both @p shape and @p strides may be NULL.
 */
typedef struct {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} DLTensor;

/** Legacy pre-v1.0 wrapper, capsule name `"dltensor"`. Deprecated upstream, still widely used. */
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

/** Bit-flag values valid in `DLManagedTensorVersioned.flags`. */
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)

/** DLPack 1.1+, used for the byte-padded fp6 sub-byte type. */
#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2UL)

/** Current, v1.0+, wrapper. Capsule name: `"dltensor_versioned"`. Required for fp6 padded flag. */
typedef struct DLManagedTensorVersioned {
    DLPackVersion version;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensorVersioned *self);
    uint64_t flags;
    DLTensor dl_tensor;
} DLManagedTensorVersioned;

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_DLPACK_ABI_H
