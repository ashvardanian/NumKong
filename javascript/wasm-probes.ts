/**
 *  @file javascript/wasm-probes.ts
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief Probe modules for the WebAssembly capabilities, validated before a module is chosen.
 *
 *  `WebAssembly.validate` accepts each array exactly where the engine implements its capability;
 *  the SIMD ones are one function of type `() → v128`.
 */

/** The `WebAssembly` members this module calls; the package compiles without the DOM typings. */
declare const WebAssembly: { validate(bytes: Uint8Array): boolean; Memory: new (descriptor: object) => object };

/** SIMD128: two `v128.const` operands and one `f32x4.add`. */
export const simd128Probe = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic, version
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type section: () → v128
    0x03, 0x02, 0x01, 0x00, // function section
    0x0a, 0x2b, 0x01, 0x29, 0x00, // code section: one body of 41 bytes
    0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xfd, 0xe4, 0x01, // f32x4.add
    0x0b,
]);

/** Relaxed SIMD: three `v128.const` operands and one `f32x4.relaxed_madd`. */
export const relaxedProbe = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic, version
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type section: () → v128
    0x03, 0x02, 0x01, 0x00, // function section
    0x0a, 0x3d, 0x01, 0x3b, 0x00, // code section: one body of 59 bytes
    0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xfd, 0x85, 0x02, // f32x4.relaxed_madd
    0x0b,
]);

/** Memory64: one memory section declaring a 64-bit memory of zero pages. */
export const memory64Probe = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic, version
    0x05, 0x03, 0x01, 0x04, 0x00, // memory section: one 64-bit memory, no maximum, zero pages
]);

/** The SIMD capabilities a WebAssembly module is built for, from the fastest down. */
export type WasmSimdTier = 'v128relaxed' | 'v128' | 'serial';

/** The best capability this engine validates; `serial` where it validates neither probe. */
export function detectWasmSimdTier(): WasmSimdTier {
    if (WebAssembly.validate(relaxedProbe)) return 'v128relaxed';
    if (WebAssembly.validate(simd128Probe)) return 'v128';
    return 'serial';
}

/** Whether the engine runs an extended module: Memory64, Relaxed SIMD and memory shared across
 *  workers, which a browser grants only to a cross-origin isolated page. */
export function detectWasmExtended(): boolean {
    // Node shares memory without cross-origin isolation and leaves the global undefined.
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated !== false;
    if (!isolated || !WebAssembly.validate(memory64Probe) || !WebAssembly.validate(relaxedProbe)) return false;
    // Engines that validated Memory64 before its JavaScript API settled reject this descriptor.
    try {
        new WebAssembly.Memory({ address: 'i64', initial: 0n });
        return true;
    } catch {
        return false;
    }
}
