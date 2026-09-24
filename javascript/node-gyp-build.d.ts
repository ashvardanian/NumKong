/**
 *  @file javascript/node-gyp-build.d.ts
 *  @author Steven Roussey
 *  @date January 18, 2024
 *  @brief Type declarations for the untyped `node-gyp-build` loader.
 */

declare module "node-gyp-build" {
    function build(dir: string): any;
    export = build;
}
