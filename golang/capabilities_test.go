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
		numkong.CapNEON:         "neon",
		numkong.CapNEONHalf:     "neonhalf",
		numkong.CapNEONBFDot:    "neonbfdot",
		numkong.CapNEONFHM:      "neonfhm",
		numkong.CapNEONSDot:     "neonsdot",
		numkong.CapNEONFP8:      "neonfp8",
		numkong.CapSVE:          "sve",
		numkong.CapSVEHalf:      "svehalf",
		numkong.CapSVESDot:      "svesdot",
		numkong.CapSVEBFDot:     "svebfdot",
		numkong.CapSVE2:         "sve2",
		numkong.CapSME:          "sme",
		numkong.CapSMEF64:       "smef64",
		numkong.CapSMEBi32:      "smebi32",
		numkong.CapHaswell:      "haswell",
		numkong.CapAlder:        "alder",
		numkong.CapSierra:       "sierra",
		numkong.CapSkylake:      "skylake",
		numkong.CapIceLake:      "icelake",
		numkong.CapGenoa:        "genoa",
		numkong.CapTurin:        "turin",
		numkong.CapSapphire:     "sapphire",
		numkong.CapDiamond:      "diamond",
		numkong.CapSapphireAMX:  "sapphireamx",
		numkong.CapGraniteAMX:   "graniteamx",
		numkong.CapDiamondAMX:   "diamondamx",
		numkong.CapRVV:          "rvv",
		numkong.CapRVVBF16:      "rvvbf16",
		numkong.CapRVVHalf:      "rvvhalf",
		numkong.CapRVVBB:        "rvvbb",
		numkong.CapV128:         "v128",
		numkong.CapV128Relaxed:  "v128relaxed",
		numkong.CapPowerVSX:     "powervsx",
		numkong.CapLoongsonASX:  "loongsonasx",
		numkong.CapCUDA:         "cuda",
		numkong.CapAmpere:       "ampere",
		numkong.CapAda:          "ada",
		numkong.CapHopper:       "hopper",
		numkong.CapBlackwell:    "blackwell",
		numkong.CapBlackwellRTX: "blackwellrtx",
		numkong.CapROCm:         "rocm",
		numkong.CapCDNA4:        "cdna4",
		numkong.CapCDNA5:        "cdna5",
		numkong.CapMetal:        "metal",
		numkong.CapApple9:       "apple9",
		numkong.CapApple10:      "apple10",
	} {
		if got := capability.String(); got != expected {
			t.Errorf("Capability(%#x).String(): expected %q, got %q", uint64(capability), expected, got)
		}
	}
}

func TestCapabilitiesEnabled(t *testing.T) {
	cpu := numkong.CPU()
	enabled, err := cpu.CapabilitiesEnabled()
	if err != nil {
		t.Fatal(err)
	}
	detected, _ := cpu.CapabilitiesDetected()
	if !enabled.Has(numkong.CapSerial) || enabled != detected&cpu.CapabilitiesCompiled() {
		t.Fatalf("CapabilitiesEnabled() = %v, expected serial plus detected and compiled capabilities", enabled)
	}
	if got := numkong.DotF32([]float32{1, 2, 3}, []float32{4, 5, 6}); got != 32 {
		t.Errorf("DotF32 on the enabled capabilities: expected 32, got %v", got)
	}
}

func TestDevices(t *testing.T) {
	if numkong.CapCPUs&numkong.CapGPUs != 0 || numkong.CapCPUs|numkong.CapGPUs|numkong.CapAny != numkong.CapAny {
		t.Errorf("CapCPUs %v and CapGPUs %v overlap or escape CapAny", numkong.CapCPUs, numkong.CapGPUs)
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
		if err != nil || gpu.CapabilitiesCompiled()&numkong.CapCPUs != 0 {
			t.Errorf("NewDevice(%d, 0) = %v, compiling %v", kind, err, gpu.CapabilitiesCompiled())
		}
		if _, err := gpu.ConfigureThread(numkong.CapAny); err == nil {
			t.Errorf("a GPU of kind %d configured a CPU thread", kind)
		}
	}
}
