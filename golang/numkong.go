// Package numkong provides SIMD-accelerated similarity measures and numeric kernels.
//
// # Operations
//
//   - Dot products: [DotF64], [DotF32], [DotI8], [DotU8]
//   - Angular distances: [AngularF64], [AngularF32], [AngularI8], [AngularU8]
//   - Euclidean distances: [EuclideanF64], [EuclideanF32], [EuclideanI8], [EuclideanU8]
//   - Squared Euclidean distances: [SqEuclideanF64], [SqEuclideanF32], [SqEuclideanI8], [SqEuclideanU8]
//   - Set similarities: [HammingU8], [HammingU1], [JaccardU1], [JaccardU16], [JaccardU32]
//   - Probability divergences: [KullbackLeiblerF64], [KullbackLeiblerF32], [JensenShannonF64], [JensenShannonF32]
//   - Geospatial distances: [HaversineF64], [HaversineF32], [VincentyF64], [VincentyF32]
//
// # Batch Operations
//
// Packed kernels take a right-hand side packed once into a [DotsPackedMatrix]:
//
//   - Packed dot products: [DotsPackedF64], [DotsPackedF32], [DotsPackedI8], [DotsPackedU8]
//   - Packed angular distances: [AngularsPackedF64], [AngularsPackedF32], [AngularsPackedI8], [AngularsPackedU8]
//   - Packed Euclidean distances: [EuclideansPackedF64], [EuclideansPackedF32], [EuclideansPackedI8], [EuclideansPackedU8]
//   - Packed binary distances: [HammingsPackedU1], [JaccardsPackedU1]
//   - Late interaction: [MaxSimF32] over a [MaxSimPackedMatrix]
//
// Symmetric kernels compare every pair within one set and write the upper triangle only:
//
//   - Symmetric dot products: [DotsSymmetricF64], [DotsSymmetricF32], [DotsSymmetricI8], [DotsSymmetricU8]
//   - Symmetric angular distances: [AngularsSymmetricF64], [AngularsSymmetricF32], [AngularsSymmetricI8], [AngularsSymmetricU8]
//   - Symmetric Euclidean distances: [EuclideansSymmetricF64], [EuclideansSymmetricF32], [EuclideansSymmetricI8], [EuclideansSymmetricU8]
//   - Symmetric binary distances: [HammingsSymmetricU1], [JaccardsSymmetricU1]
//
// # Output Types
//
// Outputs are widened to prevent overflow: float32 inputs produce float64, int8 inputs produce int32 or float32, and uint8 inputs produce uint32 or float32.
//
// # Binary Vectors
//
// Binary vectors pack 8 dimensions per byte, least significant bit first.
// [DimensionsPerValue] and [DimensionsToValues] convert a dimension count into stored values for a dtype name such as "u1".
//
// # Threads
//
// [ConfigureThread] pins the goroutine to an OS thread and configures its SIMD state.
// A [WorkerPool] keeps such threads alive, and the WithPool variants split a batch across them.
//
// # Errors
//
// Every function panics on invalid inputs such as mismatched lengths or short slices.
// Scalar functions return zero for empty inputs.
package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"
import (
	"runtime"
)

// nativeDType maps a [DotsPackedMatrix.DType] name such as "u1" onto the C enumeration.
func nativeDType(dtype string) C.nk_dtype_t {
	switch dtype {
	case "f64":
		return C.nk_f64_k
	case "f32":
		return C.nk_f32_k
	case "i8":
		return C.nk_i8_k
	case "u8":
		return C.nk_u8_k
	case "u1":
		return C.nk_u1_k
	}
	panic("unknown dtype " + dtype)
}

// DimensionsPerValue returns how many logical dimensions of dtype share one stored value, such as 8 for "u1".
// It panics on any name other than "f64", "f32", "i8", "u8" and "u1".
func DimensionsPerValue(dtype string) int {
	return int(C.nk_dimensions_per_value(nativeDType(dtype)))
}

// DimensionsToValues returns how many stored values hold dimensions logical dimensions of dtype.
// The dimensions must be a multiple of [DimensionsPerValue] for dtype.
func DimensionsToValues(dtype string, dimensions int) int {
	return dimensions / DimensionsPerValue(dtype)
}

// validateDimensions panics unless dimensions is a multiple of [DimensionsPerValue] for dtype.
func validateDimensions(dtype string, dimensions int) {
	if dimensions%DimensionsPerValue(dtype) != 0 {
		panic("dimension count must be a multiple of DimensionsPerValue(dtype)")
	}
}

// divideRoundUp divides rounding up, for tile and thread counts only.
func divideRoundUp(dividend, divisor int) int { return (dividend + divisor - 1) / divisor }

// CPU capability bit masks in chronological order (by first commercial silicon)
const (
	CapSerial      uint64 = 1 << 0  // Always: Fallback
	CapNeon        uint64 = 1 << 1  // 2013: ARM NEON
	CapHaswell     uint64 = 1 << 2  // 2013: Intel AVX2
	CapSkylake     uint64 = 1 << 3  // 2017: Intel AVX-512
	CapNeonHalf    uint64 = 1 << 4  // 2017: ARM NEON FP16
	CapNeonSdot    uint64 = 1 << 5  // 2017: ARM NEON i8 dot
	CapNeonFhm     uint64 = 1 << 6  // 2018: ARM NEON FP16 FML
	CapIcelake     uint64 = 1 << 7  // 2019: Intel AVX-512 VNNI
	CapGenoa       uint64 = 1 << 8  // 2020: AMD AVX-512 BF16
	CapNeonBfDot   uint64 = 1 << 9  // 2020: ARM NEON BF16
	CapSve         uint64 = 1 << 10 // 2020: ARM SVE
	CapSveHalf     uint64 = 1 << 11 // 2020: ARM SVE FP16
	CapSveSdot     uint64 = 1 << 12 // 2020: ARM SVE i8 dot
	CapAlder       uint64 = 1 << 13 // 2021: Intel AVX2+VNNI
	CapSveBfDot    uint64 = 1 << 14 // 2021: ARM SVE BF16
	CapSve2        uint64 = 1 << 15 // 2022: ARM SVE2
	CapV128Relaxed uint64 = 1 << 16 // 2022: WASM Relaxed SIMD
	CapSapphire    uint64 = 1 << 17 // 2023: Intel AVX-512 FP16
	CapSapphireAmx uint64 = 1 << 18 // 2023: Intel Sapphire AMX
	CapRvv         uint64 = 1 << 19 // 2023: RISC-V Vector
	CapRvvHalf     uint64 = 1 << 20 // 2023: RISC-V Zvfh
	CapRvvBf16     uint64 = 1 << 21 // 2023: RISC-V Zvfbfwma
	CapGraniteAmx  uint64 = 1 << 22 // 2024: Intel Granite AMX FP16
	CapTurin       uint64 = 1 << 23 // 2024: AMD Turin AVX-512 CD
	CapSme         uint64 = 1 << 24 // 2024: ARM SME
	CapSme2        uint64 = 1 << 25 // 2024: ARM SME2
	CapSmeF64      uint64 = 1 << 26 // 2024: ARM SME F64
	CapSmeFa64     uint64 = 1 << 27 // 2024: ARM SME FA64
	CapSve2p1      uint64 = 1 << 28 // 2025+: ARM SVE2.1
	CapSme2p1      uint64 = 1 << 29 // 2025+: ARM SME2.1
	CapSmeHalf     uint64 = 1 << 30 // 2025+: ARM SME F16F16
	CapSmeBf16     uint64 = 1 << 31 // 2025+: ARM SME B16B16
	CapSmeLut2     uint64 = 1 << 32 // 2025+: ARM SME LUTv2
	CapRvvBB       uint64 = 1 << 33 // RISC-V: Byte-Byte extensions
	CapSierra      uint64 = 1 << 34 // 2024: Intel AVXVNNIINT8
	CapSmeBi32     uint64 = 1 << 35 // 2025+: ARM SME BI32I32
	CapLoongsonAsx uint64 = 1 << 36 // LoongArch LASX 256-bit SIMD
	CapPowerVsx    uint64 = 1 << 37 // Power VSX 128-bit SIMD
	CapDiamond     uint64 = 1 << 38 // 2025+: Intel AVX10.2
	CapNeonFp8     uint64 = 1 << 39 // ARM NEON FP8
	CapDiamondAmx  uint64 = 1 << 40 // Intel Diamond Rapids AMX
	CapV128        uint64 = 1 << 41 // 2021: WASM SIMD128
)

// CapabilitiesDetected returns the bitmask of SIMD capabilities this CPU supports, whether or not their kernels were compiled in.
func CapabilitiesDetected() uint64 {
	return uint64(C.nk_capabilities_detected())
}

// CapabilitiesCompiled returns the bitmask of SIMD capabilities whose kernels were compiled in, whether or not this CPU supports them.
func CapabilitiesCompiled() uint64 {
	return uint64(C.nk_capabilities_compiled())
}

// CapabilitiesAvailable returns the bitmask of SIMD capabilities that can execute here, the intersection of [CapabilitiesDetected] and [CapabilitiesCompiled].
func CapabilitiesAvailable() uint64 {
	return uint64(C.nk_capabilities_available())
}

// CapabilitiesEnabled returns the bitmask of SIMD capabilities dispatch is currently restricted to, a subset of [CapabilitiesAvailable].
func CapabilitiesEnabled() uint64 {
	return uint64(C.nk_capabilities_enabled())
}

// CapabilitiesHas reports whether any bit of caps is in [CapabilitiesAvailable].
func CapabilitiesHas(caps uint64) bool {
	return CapabilitiesAvailable()&caps != 0
}

// CapabilitiesRestrict restricts dispatch to caps, clamped to [CapabilitiesAvailable].
// The serial fallback is always retained.
func CapabilitiesRestrict(caps uint64) {
	C.nk_capabilities_restrict(C.nk_capability_t(caps))
}

// CapabilitiesEnable adds caps to [CapabilitiesEnabled], ignoring any that are not available.
func CapabilitiesEnable(caps uint64) {
	C.nk_capabilities_enable(C.nk_capability_t(caps))
}

// CapabilitiesDisable removes caps from [CapabilitiesEnabled].
// The serial fallback cannot be removed.
func CapabilitiesDisable(caps uint64) {
	C.nk_capabilities_disable(C.nk_capability_t(caps))
}

// ConfigureThread pins the goroutine to an OS thread, configures its SIMD state for [CapabilitiesAvailable], and returns the unlock function.
// Call the returned function, typically via defer, once the SIMD work is done.
func ConfigureThread() func() {
	runtime.LockOSThread()
	C.nk_configure_thread(C.nk_capability_t(C.nk_capabilities_available()))
	return runtime.UnlockOSThread
}

// ConfigureThreadWith is [ConfigureThread] with an explicit capability mask caps.
func ConfigureThreadWith(caps uint64) func() {
	runtime.LockOSThread()
	C.nk_configure_thread(C.nk_capability_t(caps))
	return runtime.UnlockOSThread
}
