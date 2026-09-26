// SIMD capability bits, the masks dispatch picks kernels from, and the per-thread SIMD state.
//
// File: golang/capabilities.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"
import "runtime"

// Capability is one CPU capability tier, or a set of them.
type Capability uint64

// CPU capability bit masks, in chronological order of first commercial silicon
const (
	CapSerial      Capability = 1 << 0  // Always: Fallback
	CapNeon        Capability = 1 << 1  // 2013: ARM NEON
	CapHaswell     Capability = 1 << 2  // 2013: Intel AVX2
	CapSkylake     Capability = 1 << 3  // 2017: Intel AVX-512
	CapNeonHalf    Capability = 1 << 4  // 2017: ARM NEON FP16
	CapNeonSdot    Capability = 1 << 5  // 2017: ARM NEON i8 dot
	CapNeonFhm     Capability = 1 << 6  // 2018: ARM NEON FP16 FML
	CapIcelake     Capability = 1 << 7  // 2019: Intel AVX-512 VNNI
	CapGenoa       Capability = 1 << 8  // 2020: AMD AVX-512 BF16
	CapNeonBfDot   Capability = 1 << 9  // 2020: ARM NEON BF16
	CapSve         Capability = 1 << 10 // 2020: ARM SVE
	CapSveHalf     Capability = 1 << 11 // 2020: ARM SVE FP16
	CapSveSdot     Capability = 1 << 12 // 2020: ARM SVE i8 dot
	CapAlder       Capability = 1 << 13 // 2021: Intel AVX2+VNNI
	CapSveBfDot    Capability = 1 << 14 // 2021: ARM SVE BF16
	CapSve2        Capability = 1 << 15 // 2022: ARM SVE2
	CapV128Relaxed Capability = 1 << 16 // 2022: WASM Relaxed SIMD
	CapSapphire    Capability = 1 << 17 // 2023: Intel AVX-512 FP16
	CapSapphireAmx Capability = 1 << 18 // 2023: Intel Sapphire AMX
	CapRvv         Capability = 1 << 19 // 2023: RISC-V Vector
	CapRvvHalf     Capability = 1 << 20 // 2023: RISC-V Zvfh
	CapRvvBf16     Capability = 1 << 21 // 2023: RISC-V Zvfbfwma
	CapGraniteAmx  Capability = 1 << 22 // 2024: Intel Granite AMX FP16
	CapTurin       Capability = 1 << 23 // 2024: AMD Turin AVX-512 CD
	CapSme         Capability = 1 << 24 // 2024: ARM SME
	CapSme2        Capability = 1 << 25 // 2024: ARM SME2
	CapSmeF64      Capability = 1 << 26 // 2024: ARM SME F64
	CapSmeFa64     Capability = 1 << 27 // 2024: ARM SME FA64
	CapSve2p1      Capability = 1 << 28 // 2025+: ARM SVE2.1
	CapSme2p1      Capability = 1 << 29 // 2025+: ARM SME2.1
	CapSmeHalf     Capability = 1 << 30 // 2025+: ARM SME F16F16
	CapSmeBf16     Capability = 1 << 31 // 2025+: ARM SME B16B16
	CapSmeLut2     Capability = 1 << 32 // 2025+: ARM SME LUTv2
	CapRvvBB       Capability = 1 << 33 // RISC-V: Byte-Byte extensions
	CapSierra      Capability = 1 << 34 // 2024: Intel AVXVNNIINT8
	CapSmeBi32     Capability = 1 << 35 // 2025+: ARM SME BI32I32
	CapLoongsonAsx Capability = 1 << 36 // LoongArch LASX 256-bit SIMD
	CapPowerVsx    Capability = 1 << 37 // Power VSX 128-bit SIMD
	CapDiamond     Capability = 1 << 38 // 2025+: Intel AVX10.2
	CapNeonFp8     Capability = 1 << 39 // ARM NEON FP8
	CapDiamondAmx  Capability = 1 << 40 // Intel Diamond Rapids AMX
	CapV128        Capability = 1 << 41 // 2021: WASM SIMD128
)

// CapabilitiesDetected returns the capabilities this CPU supports, whether or not their kernels
// were compiled in.
func CapabilitiesDetected() Capability {
	return Capability(C.nk_cpu_capabilities_detected())
}

// CapabilitiesCompiled returns the capabilities whose kernels were compiled in, whether or not this
// CPU supports them.
func CapabilitiesCompiled() Capability {
	return Capability(C.nk_cpu_capabilities_compiled())
}

// CapabilitiesEnabled returns the capabilities dispatch uses: [CapabilitiesDetected] and
// [CapabilitiesCompiled] at once, unless narrowed by [CapabilitiesEnable]. Always has [CapSerial].
func CapabilitiesEnabled() Capability {
	return Capability(C.nk_cpu_capabilities_enabled())
}

// CapabilitiesEnable makes wanted the enabled set, clamped to [CapabilitiesDetected] and
// [CapabilitiesCompiled] and keeping [CapSerial], and returns the set that took effect.
func CapabilitiesEnable(wanted Capability) Capability {
	return Capability(C.nk_cpu_capabilities_enable(C.nk_capability_t(wanted)))
}

// ConfigureThread pins the goroutine to an OS thread, configures its SIMD state for capabilities,
// usually [CapabilitiesEnabled], and returns the unlock function. Call the returned function,
// typically via defer, once the SIMD work is done.
func ConfigureThread(capabilities Capability) func() {
	runtime.LockOSThread()
	C.nk_cpu_configure_thread(C.nk_capability_t(capabilities))
	return runtime.UnlockOSThread
}

// Has reports whether any bit of tier is in c.
func (c Capability) Has(tier Capability) bool { return c&tier != 0 }

// String names the tiers in c, comma-separated, like "serial,haswell".
func (c Capability) String() string {
	var names [C.NUMKONG_CAPABILITIES_NAME_CAPACITY]C.char
	length := C.nk_name_capabilities(C.nk_capability_t(c), &names[0], C.NUMKONG_CAPABILITIES_NAME_CAPACITY)
	return C.GoStringN(&names[0], C.int(length))
}
