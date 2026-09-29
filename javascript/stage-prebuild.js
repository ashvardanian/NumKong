/**
 *  @file javascript/stage-prebuild.js
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Copies the addon `cmake-js` produced to where the loaders find it.
 *
 *  `prebuilds/<platform>-<arch>/node.napi.node` is where `node-gyp-build` resolves it, and in a
 *  checkout the `@numkong/<platform>-<arch>` package the release publishes gets a copy too.
 *  Usage: `node javascript/stage-prebuild.js [build directory] [arch]`.
 */
const fs = require("fs");
const path = require("path");

const [, , buildDirectory = "build_node", architecture = process.arch] = process.argv;
const source = path.join(buildDirectory, "Release", "numkong.node");
const platform = `${process.platform}-${architecture}`;
const targets = [path.join("prebuilds", platform, "node.napi.node")];
const platformPackage = path.join("javascript", `@numkong-${platform}`);
if (fs.existsSync(platformPackage)) targets.push(path.join(platformPackage, "numkong.node"));

for (const target of targets) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.copyFileSync(source, target);
  console.log(`staged ${source} -> ${target}`);
}
