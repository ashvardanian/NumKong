// Package numkong provides SIMD-accelerated similarity measures and numeric kernels.
//
// Operations:
//   - Dot products: DotF64, DotF32, DotI8, DotU8
//   - Angular (cosine) distance: AngularF64, AngularF32, AngularI8, AngularU8
//   - Euclidean (L2) distance: EuclideanF64, EuclideanF32, EuclideanI8, EuclideanU8
//   - Squared Euclidean: SqEuclideanF64, SqEuclideanF32, SqEuclideanI8, SqEuclideanU8
//   - Set similarity: HammingU8, HammingU1, JaccardU1, JaccardU16, JaccardU32
//   - Probability: KullbackLeiblerF64/F32, JensenShannonF64/F32
//   - Geospatial: HaversineF64/F32, VincentyF64/F32
//
// Batch operations use PackedMatrix for type-safe pre-packed right-hand-side matrices:
//   - Packed dot products: DotsPackedF64/F32/I8/U8
//   - Packed angular/Euclidean: AngularsPackedF64/F32/I8/U8, EuclideansPackedF64/F32/I8/U8
//   - Packed binary: HammingsPackedU1, JaccardsPackedU1
//   - Symmetric self-similarity: DotsSymmetric*, AngularsSymmetric*, EuclideansSymmetric*
//   - MaxSim (ColBERT): MaxSimF32 with MaxSimPackedMatrix
//
// Output widening: f32 inputs produce f64 outputs; i8 inputs produce i32 or f32 outputs;
// u8 inputs produce u32 or f32 outputs. This prevents overflow in accumulation.
//
// Thread configuration: ConfigureThread pins the goroutine to an OS thread,
// configures SIMD state, and returns an unlock function for use with defer.
//
// Parallel batch operations: WorkerPool provides pre-pinned threads with
// pre-configured SIMD state. PackedMatrix *WithPool methods and
// *SymmetricWithPool free functions dispatch work to the pool.
//
// All functions panic on invalid inputs (length mismatches, insufficient slice capacity).
// Empty inputs return the zero value for scalar functions.
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
)

// CapabilitiesDetected returns a bitmask of the SIMD capabilities this CPU supports,
// whether or not their kernels were compiled in. See CapabilitiesAvailable for the set
// that can actually run here.
func CapabilitiesDetected() uint64 {
	return uint64(C.nk_capabilities_detected())
}

// CapabilitiesCompiled returns a bitmask of the SIMD capabilities whose kernels were
// compiled into this binary, whether or not this CPU supports them.
func CapabilitiesCompiled() uint64 {
	return uint64(C.nk_capabilities_compiled())
}

// CapabilitiesAvailable returns the bitmask of SIMD capabilities that can actually execute
// here: CapabilitiesDetected AND CapabilitiesCompiled. Prefer this over either axis alone,
// which over-reports.
func CapabilitiesAvailable() uint64 {
	return uint64(C.nk_capabilities_available())
}

// CapabilitiesEnabled returns the bitmask of SIMD capabilities dispatch is currently
// restricted to, a subset of CapabilitiesAvailable.
func CapabilitiesEnabled() uint64 {
	return uint64(C.nk_capabilities_enabled())
}

// CapabilitiesHas reports whether caps can actually execute here, i.e. whether it is in
// CapabilitiesAvailable. False both when this CPU lacks the feature and when its kernels
// were not compiled in.
func CapabilitiesHas(caps uint64) bool {
	return CapabilitiesAvailable()&caps != 0
}

// CapabilitiesRestrict restricts dispatch to caps, clamped to CapabilitiesAvailable.
// The serial fallback is always retained.
func CapabilitiesRestrict(caps uint64) {
	C.nk_capabilities_restrict(C.nk_capability_t(caps))
}

// CapabilitiesEnable adds caps to CapabilitiesEnabled. Anything not available is ignored.
func CapabilitiesEnable(caps uint64) {
	C.nk_capabilities_enable(C.nk_capability_t(caps))
}

// CapabilitiesDisable removes caps from CapabilitiesEnabled. The serial fallback cannot
// be removed.
func CapabilitiesDisable(caps uint64) {
	C.nk_capabilities_disable(C.nk_capability_t(caps))
}

// ConfigureThread pins the goroutine to an OS thread, configures SIMD state
// (AMX tile permissions on x86), and returns an unlock function.
// Call the returned function (typically via defer) to release the OS thread
// when SIMD work is done.
func ConfigureThread() func() {
	runtime.LockOSThread()
	C.nk_configure_thread(C.nk_capability_t(C.nk_capabilities_available()))
	return runtime.UnlockOSThread
}

// ConfigureThreadWith is like ConfigureThread but with an explicit capability mask.
func ConfigureThreadWith(caps uint64) func() {
	runtime.LockOSThread()
	C.nk_configure_thread(C.nk_capability_t(caps))
	return runtime.UnlockOSThread
}
