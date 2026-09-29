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
		numkong.CapSerial:      "serial",
		numkong.CapNeon:        "neon",
		numkong.CapNeonHalf:    "neonhalf",
		numkong.CapNeonBfDot:   "neonbfdot",
		numkong.CapNeonFhm:     "neonfhm",
		numkong.CapNeonSdot:    "neonsdot",
		numkong.CapNeonFp8:     "neonfp8",
		numkong.CapSve:         "sve",
		numkong.CapSveHalf:     "svehalf",
		numkong.CapSveSdot:     "svesdot",
		numkong.CapSveBfDot:    "svebfdot",
		numkong.CapSve2:        "sve2",
		numkong.CapSme:         "sme",
		numkong.CapSmeF64:      "smef64",
		numkong.CapSmeBi32:     "smebi32",
		numkong.CapHaswell:     "haswell",
		numkong.CapAlder:       "alder",
		numkong.CapSierra:      "sierra",
		numkong.CapSkylake:     "skylake",
		numkong.CapIcelake:     "icelake",
		numkong.CapGenoa:       "genoa",
		numkong.CapTurin:       "turin",
		numkong.CapSapphire:    "sapphire",
		numkong.CapDiamond:     "diamond",
		numkong.CapSapphireAmx: "sapphireamx",
		numkong.CapGraniteAmx:  "graniteamx",
		numkong.CapDiamondAmx:  "diamondamx",
		numkong.CapRvv:         "rvv",
		numkong.CapRvvBf16:     "rvvbf16",
		numkong.CapRvvHalf:     "rvvhalf",
		numkong.CapRvvBB:       "rvvbb",
		numkong.CapV128:        "v128",
		numkong.CapV128Relaxed: "v128relaxed",
		numkong.CapPowerVsx:    "powervsx",
		numkong.CapLoongsonAsx: "loongsonasx",
	} {
		if got := capability.String(); got != expected {
			t.Errorf("Capability(%#x).String(): expected %q, got %q", uint64(capability), expected, got)
		}
	}
}

func TestCapabilitiesEnable(t *testing.T) {
	enabled := numkong.CapabilitiesEnabled()
	defer numkong.CapabilitiesEnable(enabled)
	if !enabled.Has(numkong.CapSerial) || enabled&^(numkong.CapabilitiesDetected()|numkong.CapSerial) != 0 {
		t.Fatalf("CapabilitiesEnabled() = %v, expected serial plus detected capabilities", enabled)
	}
	if got := numkong.CapabilitiesEnable(0); got != numkong.CapSerial {
		t.Fatalf("CapabilitiesEnable(0) = %v, expected serial alone", got)
	}
	if got := numkong.CapabilitiesEnabled(); got != numkong.CapSerial {
		t.Fatalf("CapabilitiesEnabled() after narrowing = %v, expected serial alone", got)
	}
	if got := numkong.DotF32([]float32{1, 2, 3}, []float32{4, 5, 6}); got != 32 {
		t.Errorf("DotF32 on the serial capability: expected 32, got %v", got)
	}
	if got := numkong.CapabilitiesEnable(enabled); got != enabled {
		t.Errorf("CapabilitiesEnable(%v) = %v, expected the original set back", enabled, got)
	}
}
