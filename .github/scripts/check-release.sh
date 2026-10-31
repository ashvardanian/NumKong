#!/usr/bin/env bash
# Checks that HEAD is the release commit for a tag and that every version site in its tree carries it.
#
#     .github/scripts/check-release.sh v7.8.3
set -euo pipefail
tag=$1
subject=$(git log -1 --format=%s)
if [ "$subject" != "Release: $tag" ]; then
    echo "check-release: $tag must sit on 'Release: $tag', but HEAD is '$subject'"
    exit 1
fi
version=$(printf '%s\n' "$tag" | sed -nE 's/^v([0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?)$/\1/p')
if [ -z "$version" ]; then
    echo "check-release: a release tag is exactly vX.Y.Z or vX.Y.Z-rc.N, got $tag"
    exit 1
fi
core=${version%%-*} stale=""
IFS=. read -r major minor patch <<< "$core"
[ "$(cat VERSION)" = "$version" ] || stale="$stale VERSION"
grep -qE "^version = \"${version//./\\.}\"" Cargo.toml || stale="$stale Cargo.toml"
grep -qxF "  \"version\": \"$version\"," package.json || stale="$stale package.json"
for directory in javascript/@numkong-*/; do
    platform=${directory#javascript/@numkong-} platform=${platform%/}
    grep -qxF "  \"version\": \"$version\"," "${directory}package.json" || stale="$stale ${directory}package.json"
    grep -qF "\"@numkong/$platform\": \"$version\"" package.json || stale="$stale package.json:@numkong/$platform"
done
# CMake's `project(VERSION)` and the C macros take numbers only, so a candidate keeps its core there.
grep -qE "^[[:space:]]+VERSION ${core//./\\.}$" CMakeLists.txt || stale="$stale CMakeLists.txt"
grep -qxF "version: $version" CITATION.cff || stale="$stale CITATION.cff"
for field in "MAJOR $major" "MINOR $minor" "PATCH $patch"; do
    grep -qxF "#define NK_VERSION_$field" include/numkong/capabilities.h || stale="$stale include/numkong/capabilities.h:${field% *}"
done
if [ -n "$stale" ]; then
    echo "check-release: these files don't carry $version:$stale"
    exit 1
fi
