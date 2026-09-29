// SIMD capability bits, the mask every kernel call passes, and the per-thread SIMD state.
//
// File: golang/capabilities.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"
import (
	"runtime"
	"strconv"
	"sync/atomic"
)

// Capability is one CPU capability, or a set of them.
type Capability uint64

// CPU capability bit masks, each capability group ascending by dispatch preference
const (
	CapSerial      Capability = C.nk_cap_serial_k      // Always: Fallback
	CapNeon        Capability = C.nk_cap_neon_k        // 2013: ARM NEON
	CapNeonHalf    Capability = C.nk_cap_neonhalf_k    // 2017: ARM NEON FP16
	CapNeonBfDot   Capability = C.nk_cap_neonbfdot_k   // 2020: ARM NEON BF16
	CapNeonFhm     Capability = C.nk_cap_neonfhm_k     // 2018: ARM NEON FP16 FML
	CapNeonSdot    Capability = C.nk_cap_neonsdot_k    // 2017: ARM NEON i8 dot
	CapNeonFp8     Capability = C.nk_cap_neonfp8_k     // ARM NEON FP8
	CapSve         Capability = C.nk_cap_sve_k         // 2020: ARM SVE
	CapSveHalf     Capability = C.nk_cap_svehalf_k     // 2020: ARM SVE FP16
	CapSveSdot     Capability = C.nk_cap_svesdot_k     // 2020: ARM SVE i8 dot
	CapSveBfDot    Capability = C.nk_cap_svebfdot_k    // 2021: ARM SVE BF16
	CapSve2        Capability = C.nk_cap_sve2_k        // 2022: ARM SVE2
	CapSme         Capability = C.nk_cap_sme_k         // 2024: ARM SME
	CapSmeF64      Capability = C.nk_cap_smef64_k      // 2024: ARM SME F64
	CapSmeBi32     Capability = C.nk_cap_smebi32_k     // 2025+: ARM SME BI32I32
	CapHaswell     Capability = C.nk_cap_haswell_k     // 2013: Intel AVX2
	CapAlder       Capability = C.nk_cap_alder_k       // 2021: Intel AVX2+VNNI
	CapSierra      Capability = C.nk_cap_sierra_k      // 2024: Intel AVXVNNIINT8
	CapSkylake     Capability = C.nk_cap_skylake_k     // 2017: Intel AVX-512
	CapIcelake     Capability = C.nk_cap_icelake_k     // 2019: Intel AVX-512 VNNI
	CapGenoa       Capability = C.nk_cap_genoa_k       // 2020: AMD AVX-512 BF16
	CapTurin       Capability = C.nk_cap_turin_k       // 2024: AMD Turin AVX-512 CD
	CapSapphire    Capability = C.nk_cap_sapphire_k    // 2023: Intel AVX-512 FP16
	CapDiamond     Capability = C.nk_cap_diamond_k     // 2025+: Intel AVX10.2
	CapSapphireAmx Capability = C.nk_cap_sapphireamx_k // 2023: Intel Sapphire AMX
	CapGraniteAmx  Capability = C.nk_cap_graniteamx_k  // 2024: Intel Granite AMX FP16
	CapDiamondAmx  Capability = C.nk_cap_diamondamx_k  // Intel Diamond Rapids AMX
	CapRvv         Capability = C.nk_cap_rvv_k         // 2023: RISC-V Vector
	CapRvvBf16     Capability = C.nk_cap_rvvbf16_k     // 2023: RISC-V Zvfbfwma
	CapRvvHalf     Capability = C.nk_cap_rvvhalf_k     // 2023: RISC-V Zvfh
	CapRvvBB       Capability = C.nk_cap_rvvbb_k       // RISC-V: Byte-Byte extensions
	CapV128        Capability = C.nk_cap_v128_k        // 2021: WASM SIMD128
	CapV128Relaxed Capability = C.nk_cap_v128relaxed_k // 2022: WASM Relaxed SIMD
	CapPowerVsx    Capability = C.nk_cap_powervsx_k    // Power VSX 128-bit SIMD
	CapLoongsonAsx Capability = C.nk_cap_loongsonasx_k // LoongArch LASX 256-bit SIMD
)

// enabled holds the mask every kernel call passes, zero until [CapabilitiesEnabled] first reads it.
var enabled atomic.Uint64

// CapabilitiesDetected returns the capabilities this CPU supports, whether or not their kernels
// were compiled in.
func CapabilitiesDetected() Capability {
	var capabilities C.nk_capability_t
	C.nk_cpu_capabilities_detected(&capabilities)
	return Capability(capabilities)
}

// CapabilitiesCompiled returns the capabilities whose kernels were compiled in, whether or not this
// CPU supports them.
func CapabilitiesCompiled() Capability {
	var capabilities C.nk_capability_t
	C.nk_cpu_capabilities_compiled(&capabilities)
	return Capability(capabilities)
}

// CapabilitiesEnabled returns the capabilities every kernel call passes: [CapabilitiesDetected] and
// [CapabilitiesCompiled] at once, unless narrowed by [CapabilitiesEnable]. Always has [CapSerial].
func CapabilitiesEnabled() Capability {
	if mask := enabled.Load(); mask != 0 {
		return Capability(mask)
	}
	enabled.CompareAndSwap(0, uint64(available()))
	return Capability(enabled.Load())
}

// CapabilitiesEnable makes wanted the enabled set, clamped to [CapabilitiesDetected] and
// [CapabilitiesCompiled] and keeping [CapSerial], and returns the set that took effect.
// Repack matrices packed before the call, since packed kernels refuse another capability's layout.
func CapabilitiesEnable(wanted Capability) Capability {
	mask := wanted&available() | CapSerial
	enabled.Store(uint64(mask))
	return mask
}

// available returns the capabilities this CPU supports and this binary contains.
func available() Capability {
	capabilities := C.nk_capability_t(CapSerial)
	C.nk_cpu_capabilities_enabled(&capabilities)
	return Capability(capabilities)
}

// capabilities returns [CapabilitiesEnabled] as the mask a kernel call takes.
func capabilities() C.nk_capability_t { return C.nk_capability_t(CapabilitiesEnabled()) }

// check panics when a kernel reports a failure, as the package does on invalid inputs.
func check(status C.nk_status_t) {
	if status != C.nk_success_k {
		panic("kernel failed with status " + strconv.Itoa(int(status)))
	}
}

// ConfigureThread pins the goroutine to an OS thread, configures its SIMD state for capabilities,
// usually [CapabilitiesEnabled], and returns the unlock function. Call the returned function,
// typically via defer, once the SIMD work is done.
func ConfigureThread(capabilities Capability) func() {
	runtime.LockOSThread()
	C.nk_cpu_configure_thread(C.nk_capability_t(capabilities))
	return runtime.UnlockOSThread
}

// Has reports whether any bit of capability is in c.
func (c Capability) Has(capability Capability) bool { return c&capability != 0 }

// String names the capabilities in c, comma-separated, like "serial,haswell".
func (c Capability) String() string {
	var names [C.NUMKONG_CAPABILITIES_NAME_CAPACITY]C.char
	length := C.nk_name_capabilities(C.nk_capability_t(c), &names[0], C.NUMKONG_CAPABILITIES_NAME_CAPACITY)
	return C.GoStringN(&names[0], C.int(length))
}
