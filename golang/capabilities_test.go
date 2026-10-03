// Capability names, checked against the C name table, and the mask kernels are called with.
//
// File: golang/capabilities_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestCapabilityNames(t *testing.T) {
	for capability, expected := range map[numkong.Capability]string{
		numkong.CapSerial:       "serial",
		numkong.CapNeon:         "neon",
		numkong.CapNeonHalf:     "neonhalf",
		numkong.CapNeonBfDot:    "neonbfdot",
		numkong.CapNeonFhm:      "neonfhm",
		numkong.CapNeonSdot:     "neonsdot",
		numkong.CapNeonFp8:      "neonfp8",
		numkong.CapSve:          "sve",
		numkong.CapSveHalf:      "svehalf",
		numkong.CapSveSdot:      "svesdot",
		numkong.CapSveBfDot:     "svebfdot",
		numkong.CapSve2:         "sve2",
		numkong.CapSme:          "sme",
		numkong.CapSmeF64:       "smef64",
		numkong.CapSmeBi32:      "smebi32",
		numkong.CapHaswell:      "haswell",
		numkong.CapAlder:        "alder",
		numkong.CapSierra:       "sierra",
		numkong.CapSkylake:      "skylake",
		numkong.CapIcelake:      "icelake",
		numkong.CapGenoa:        "genoa",
		numkong.CapTurin:        "turin",
		numkong.CapSapphire:     "sapphire",
		numkong.CapDiamond:      "diamond",
		numkong.CapSapphireAmx:  "sapphireamx",
		numkong.CapGraniteAmx:   "graniteamx",
		numkong.CapDiamondAmx:   "diamondamx",
		numkong.CapRvv:          "rvv",
		numkong.CapRvvBf16:      "rvvbf16",
		numkong.CapRvvHalf:      "rvvhalf",
		numkong.CapRvvBB:        "rvvbb",
		numkong.CapV128:         "v128",
		numkong.CapV128Relaxed:  "v128relaxed",
		numkong.CapPowerVsx:     "powervsx",
		numkong.CapLoongsonAsx:  "loongsonasx",
		numkong.CapCuda:         "cuda",
		numkong.CapAmpere:       "ampere",
		numkong.CapAda:          "ada",
		numkong.CapHopper:       "hopper",
		numkong.CapBlackwell:    "blackwell",
		numkong.CapBlackwellRtx: "blackwellrtx",
		numkong.CapRocm:         "rocm",
		numkong.CapCdna4:        "cdna4",
		numkong.CapCdna5:        "cdna5",
		numkong.CapMetal:        "metal",
		numkong.CapApple9:       "apple9",
		numkong.CapApple10:      "apple10",
	} {
		if got := capability.String(); got != expected {
			t.Errorf("Capability(%#x).String(): expected %q, got %q", uint64(capability), expected, got)
		}
	}
}

func TestCapabilitiesEnable(t *testing.T) {
	cpu := numkong.CPU()
	enabled, err := cpu.CapabilitiesEnabled()
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.CapabilitiesEnable(enabled)
	detected, _ := cpu.CapabilitiesDetected()
	if !enabled.Has(numkong.CapSerial) || enabled&^(detected|numkong.CapSerial) != 0 {
		t.Fatalf("CapabilitiesEnabled() = %v, expected serial plus detected capabilities", enabled)
	}
	if got, _ := cpu.CapabilitiesEnable(0); got != numkong.CapSerial {
		t.Fatalf("CapabilitiesEnable(0) = %v, expected serial alone", got)
	}
	if got, _ := cpu.CapabilitiesEnabled(); got != numkong.CapSerial {
		t.Fatalf("CapabilitiesEnabled() after narrowing = %v, expected serial alone", got)
	}
	if got := numkong.DotF32([]float32{1, 2, 3}, []float32{4, 5, 6}); got != 32 {
		t.Errorf("DotF32 on the serial capability: expected 32, got %v", got)
	}
	if got, _ := cpu.CapabilitiesEnable(enabled); got != enabled {
		t.Errorf("CapabilitiesEnable(%v) = %v, expected the original set back", enabled, got)
	}
}

func TestDevices(t *testing.T) {
	if numkong.CapCpus&numkong.CapGpus != 0 || numkong.CapCpus|numkong.CapGpus|numkong.CapAny != numkong.CapAny {
		t.Errorf("CapCpus %v and CapGpus %v overlap or escape CapAny", numkong.CapCpus, numkong.CapGpus)
	}
	if count, err := numkong.CountDevices(numkong.DeviceCPU); count != 1 || err != nil {
		t.Errorf("CountDevices(DeviceCPU) = %d, %v, expected one CPU", count, err)
	}
	for _, kind := range []numkong.DeviceKind{numkong.DeviceCPU, numkong.DeviceCUDA, numkong.DeviceROCm, numkong.DeviceMetal} {
		count, _ := numkong.CountDevices(kind)
		if _, err := numkong.NewDevice(kind, count); err == nil {
			t.Errorf("NewDevice(%d, %d) made a device past the last one", kind, count)
		}
		if count == 0 || kind == numkong.DeviceCPU {
			continue
		}
		gpu, err := numkong.NewDevice(kind, 0)
		if err != nil || gpu.CapabilitiesCompiled()&numkong.CapCpus != 0 {
			t.Errorf("NewDevice(%d, 0) = %v, compiling %v", kind, err, gpu.CapabilitiesCompiled())
		}
		if _, err := gpu.ConfigureThread(numkong.CapAny); err == nil {
			t.Errorf("a GPU of kind %d configured a CPU thread", kind)
		}
	}
}
