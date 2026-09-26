// Capability tier names, checked against the C name table.
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
		numkong.CapHaswell:     "haswell",
		numkong.CapSkylake:     "skylake",
		numkong.CapNeonHalf:    "neonhalf",
		numkong.CapNeonSdot:    "neonsdot",
		numkong.CapNeonFhm:     "neonfhm",
		numkong.CapIcelake:     "icelake",
		numkong.CapGenoa:       "genoa",
		numkong.CapNeonBfDot:   "neonbfdot",
		numkong.CapSve:         "sve",
		numkong.CapSveHalf:     "svehalf",
		numkong.CapSveSdot:     "svesdot",
		numkong.CapAlder:       "alder",
		numkong.CapSveBfDot:    "svebfdot",
		numkong.CapSve2:        "sve2",
		numkong.CapV128Relaxed: "v128relaxed",
		numkong.CapSapphire:    "sapphire",
		numkong.CapSapphireAmx: "sapphireamx",
		numkong.CapRvv:         "rvv",
		numkong.CapRvvHalf:     "rvvhalf",
		numkong.CapRvvBf16:     "rvvbf16",
		numkong.CapGraniteAmx:  "graniteamx",
		numkong.CapTurin:       "turin",
		numkong.CapSme:         "sme",
		numkong.CapSme2:        "sme2",
		numkong.CapSmeF64:      "smef64",
		numkong.CapSmeFa64:     "smefa64",
		numkong.CapSve2p1:      "sve2p1",
		numkong.CapSme2p1:      "sme2p1",
		numkong.CapSmeHalf:     "smehalf",
		numkong.CapSmeBf16:     "smebf16",
		numkong.CapSmeLut2:     "smelut2",
		numkong.CapRvvBB:       "rvvbb",
		numkong.CapSierra:      "sierra",
		numkong.CapSmeBi32:     "smebi32",
		numkong.CapLoongsonAsx: "loongsonasx",
		numkong.CapPowerVsx:    "powervsx",
		numkong.CapDiamond:     "diamond",
		numkong.CapNeonFp8:     "neonfp8",
		numkong.CapDiamondAmx:  "diamondamx",
		numkong.CapV128:        "v128",
	} {
		if got := capability.String(); got != expected {
			t.Errorf("Capability(%#x).String(): expected %q, got %q", uint64(capability), expected, got)
		}
	}
}
