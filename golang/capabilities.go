// Devices, their capability bits, the mask every kernel call passes, and the per-thread SIMD state.
//
// File: golang/capabilities.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"
import (
	"errors"
	"runtime"
	"sync/atomic"
)

// Capability is one capability of a CPU or a GPU, or a set of them.
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

	CapCuda         Capability = C.nk_cap_cuda_k         // Any CUDA device
	CapAmpere       Capability = C.nk_cap_ampere_k       // 2020: NVIDIA SM 8.0
	CapAda          Capability = C.nk_cap_ada_k          // 2022: NVIDIA SM 8.9
	CapHopper       Capability = C.nk_cap_hopper_k       // 2022: NVIDIA SM 9.x
	CapBlackwell    Capability = C.nk_cap_blackwell_k    // 2024: NVIDIA SM 10.x
	CapBlackwellRtx Capability = C.nk_cap_blackwellrtx_k // 2025: NVIDIA SM 12.x
	CapRocm         Capability = C.nk_cap_rocm_k         // Any ROCm device
	CapCdna4        Capability = C.nk_cap_cdna4_k        // 2025: AMD gfx950
	CapCdna5        Capability = C.nk_cap_cdna5_k        // AMD gfx1250
	CapMetal        Capability = C.nk_cap_metal_k        // Any Metal device
	CapApple9       Capability = C.nk_cap_apple9_k       // 2023: Apple GPU family 9
	CapApple10      Capability = C.nk_cap_apple10_k      // Apple GPU family 10

	CapCpus    Capability = C.nk_cap_cpus_k    // Every CPU capability
	CapDevices Capability = C.nk_cap_devices_k // Every GPU capability
	CapAny     Capability = ^Capability(0)     // Every capability
)

// DeviceKind is the runtime a device belongs to, as the `nk_<kind>_*` C functions name it.
type DeviceKind int

// Device kinds, one per family of `nk_<kind>_*` C functions.
const (
	DeviceCPU DeviceKind = iota
	DeviceCUDA
	DeviceROCm
	DeviceMetal
)

// Device is one device NumKong can run kernels on: the host CPU, or a GPU by its runtime's own
// ordinal, the one `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list.
type Device struct {
	Kind    DeviceKind
	Ordinal int
}

// enabled holds the mask every kernel call passes, zero until the CPU's
// [Device.CapabilitiesEnabled] first reads it.
var enabled atomic.Uint64

// CPU returns the host CPU, which every build has.
func CPU() Device { return Device{Kind: DeviceCPU} }

// CountDevices returns how many devices of kind the process sees: one CPU, or the GPUs its runtime
// counts, failing without one.
func CountDevices(kind DeviceKind) (int, error) {
	count := C.nk_size_t(1)
	var status C.nk_status_t = C.nk_success_k
	switch kind {
	case DeviceCPU:
	case DeviceCUDA:
		status = C.nk_cuda_count_devices(&count)
	case DeviceROCm:
		status = C.nk_rocm_count_devices(&count)
	case DeviceMetal:
		status = C.nk_metal_count_devices(&count)
	default:
		count, status = 0, C.nk_missing_gpu_k
	}
	return int(count), statusError(status)
}

// NewDevice returns device ordinal of kind, failing past the last one.
func NewDevice(kind DeviceKind, ordinal int) (Device, error) {
	count, err := CountDevices(kind)
	if err == nil && (ordinal < 0 || ordinal >= count) {
		err = statusError(C.nk_missing_gpu_k)
	}
	return Device{Kind: kind, Ordinal: ordinal}, err
}

// CapabilitiesDetected returns the capabilities d runs, whether or not they were compiled in.
func (d Device) CapabilitiesDetected() (Capability, error) {
	var capabilities C.nk_capability_t
	ordinal := C.nk_size_t(d.Ordinal)
	var status C.nk_status_t
	switch d.Kind {
	case DeviceCPU:
		status = C.nk_cpu_capabilities_detected(&capabilities)
	case DeviceCUDA:
		status = C.nk_cuda_capabilities_detected(ordinal, &capabilities)
	case DeviceROCm:
		status = C.nk_rocm_capabilities_detected(ordinal, &capabilities)
	case DeviceMetal:
		status = C.nk_metal_capabilities_detected(ordinal, &capabilities)
	default:
		status = C.nk_missing_gpu_k
	}
	return Capability(capabilities), statusError(status)
}

// CapabilitiesCompiled returns the capabilities whose kernels were compiled in for devices of d's
// kind, whether or not d runs them.
func (d Device) CapabilitiesCompiled() Capability {
	var capabilities C.nk_capability_t
	switch d.Kind {
	case DeviceCPU:
		C.nk_cpu_capabilities_compiled(&capabilities)
	case DeviceCUDA:
		C.nk_cuda_capabilities_compiled(&capabilities)
	case DeviceROCm:
		C.nk_rocm_capabilities_compiled(&capabilities)
	case DeviceMetal:
		C.nk_metal_capabilities_compiled(&capabilities)
	}
	return Capability(capabilities)
}

// CapabilitiesEnabled returns the mask for d's kernel calls: [Device.CapabilitiesDetected] and
// [Device.CapabilitiesCompiled] at once. On the CPU it is the mask every kernel call of this
// package passes, narrowed by [Device.CapabilitiesEnable], and always has [CapSerial].
func (d Device) CapabilitiesEnabled() (Capability, error) {
	var capabilities C.nk_capability_t
	ordinal := C.nk_size_t(d.Ordinal)
	var status C.nk_status_t
	switch d.Kind {
	case DeviceCPU:
		return Capability(cpuEnabled()), nil
	case DeviceCUDA:
		status = C.nk_cuda_capabilities_enabled(ordinal, &capabilities)
	case DeviceROCm:
		status = C.nk_rocm_capabilities_enabled(ordinal, &capabilities)
	case DeviceMetal:
		status = C.nk_metal_capabilities_enabled(ordinal, &capabilities)
	default:
		status = C.nk_missing_gpu_k
	}
	return Capability(capabilities), statusError(status)
}

// CapabilitiesEnable makes wanted the CPU's enabled set, clamped to what it detects and this binary
// compiled and keeping [CapSerial], and returns the set that took effect. GPUs keep no such set.
// Repack matrices packed before the call, since packed kernels refuse another capability's layout.
func (d Device) CapabilitiesEnable(wanted Capability) (Capability, error) {
	if d.Kind != DeviceCPU {
		return 0, statusError(C.nk_missing_kernel_k)
	}
	mask := wanted&available() | CapSerial
	enabled.Store(uint64(mask))
	return mask, nil
}

// ConfigureThread pins the goroutine to an OS thread, configures its SIMD state for capabilities,
// usually the CPU's [Device.CapabilitiesEnabled], and returns the unlock function. Call it,
// typically via defer, once the SIMD work is done. GPUs have no thread state to configure.
func (d Device) ConfigureThread(capabilities Capability) (func(), error) {
	if d.Kind != DeviceCPU {
		return func() {}, statusError(C.nk_missing_kernel_k)
	}
	runtime.LockOSThread()
	C.nk_cpu_configure_thread(C.nk_capability_t(capabilities))
	return runtime.UnlockOSThread, nil
}

// cpuEnabled returns the CPU's enabled mask, reading it on first use.
func cpuEnabled() uint64 {
	if mask := enabled.Load(); mask != 0 {
		return mask
	}
	enabled.CompareAndSwap(0, uint64(available()))
	return enabled.Load()
}

// available returns the capabilities this CPU supports and this binary contains.
func available() Capability {
	capabilities := C.nk_capability_t(CapSerial)
	C.nk_cpu_capabilities_enabled(&capabilities)
	return Capability(capabilities)
}

// capabilities returns the CPU's enabled mask as a kernel call takes it.
func capabilities() C.nk_capability_t { return C.nk_capability_t(cpuEnabled()) }

// statusError names a failed status, or returns nil on success.
func statusError(status C.nk_status_t) error {
	if status == C.nk_success_k {
		return nil
	}
	return errors.New(C.GoString(C.nk_status_name(status)))
}

// check panics when a kernel reports a failure, as the package does on invalid inputs.
func check(status C.nk_status_t) {
	if err := statusError(status); err != nil {
		panic(err)
	}
}

// Has reports whether any bit of capability is in c.
func (c Capability) Has(capability Capability) bool { return c&capability != 0 }

// String names the capabilities in c, comma-separated, like "serial,haswell".
func (c Capability) String() string {
	var names [C.NUMKONG_CAPABILITIES_NAME_CAPACITY]C.char
	length := C.nk_capabilities_name(C.nk_capability_t(c), &names[0], C.NUMKONG_CAPABILITIES_NAME_CAPACITY)
	return C.GoStringN(&names[0], C.int(length))
}
