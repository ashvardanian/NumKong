/**
 *  @file javascript/emscripten-glue.d.ts
 *  @author Ash Vardanian
 *  @date October 31, 2026
 *  @brief Types for the Emscripten glue modules the WASM entries load by package subpath.
 */

declare module "numkong/wasm32-*" {
    const factory: () => Promise<any>;
    export default factory;
}

declare module "numkong/wasm64-*" {
    const factory: () => Promise<any>;
    export default factory;
}
