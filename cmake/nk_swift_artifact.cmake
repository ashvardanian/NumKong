# Adds one build's `numkong_static` to the Swift artifact in NUMKONG_SWIFT_DIRECTORY, keeping what earlier builds
# added, so a release fills one artifact from one CMake build per target. Runs as `cmake -P`, from `numkong_swift`.
#
#   NUMKONG_SWIFT_ARCHIVE       the static library
#   NUMKONG_SWIFT_HEADERS       the public headers, with the module map
#   NUMKONG_SWIFT_DIRECTORY     where `CNumKong.xcframework` or `CNumKong.artifactbundle` lands
#   NUMKONG_SWIFT_SDK           on Apple platforms, the SDK the archive targets, like `iphonesimulator`: the
#                               XCFramework gets one library per SDK, fattened with `lipo`
#   NUMKONG_SWIFT_ARCHITECTURE  on Apple platforms, the archive's one architecture
#   NUMKONG_SWIFT_TRIPLES       elsewhere, the comma-separated Swift triples the archive serves
#   NUMKONG_SWIFT_VERSION       elsewhere, the library's version
get_filename_component(archive_name "${NUMKONG_SWIFT_ARCHIVE}" NAME)

if (NUMKONG_SWIFT_SDK)
    set(slices "${NUMKONG_SWIFT_DIRECTORY}/CNumKong.slices")
    file(MAKE_DIRECTORY "${slices}/${NUMKONG_SWIFT_SDK}/${NUMKONG_SWIFT_ARCHITECTURE}")
    file(COPY_FILE "${NUMKONG_SWIFT_ARCHIVE}"
         "${slices}/${NUMKONG_SWIFT_SDK}/${NUMKONG_SWIFT_ARCHITECTURE}/${archive_name}"
    )

    set(libraries)
    file(GLOB sdks LIST_DIRECTORIES true "${slices}/*")
    foreach (sdk IN LISTS sdks)
        file(GLOB architectures "${sdk}/*/${archive_name}")
        execute_process(
            COMMAND lipo -create ${architectures} -output "${sdk}/${archive_name}" COMMAND_ERROR_IS_FATAL ANY
        )
        list(APPEND libraries -library "${sdk}/${archive_name}" -headers "${NUMKONG_SWIFT_HEADERS}")
    endforeach ()
    file(REMOVE_RECURSE "${NUMKONG_SWIFT_DIRECTORY}/CNumKong.xcframework")
    execute_process(
        COMMAND xcodebuild -create-xcframework ${libraries} -output "${NUMKONG_SWIFT_DIRECTORY}/CNumKong.xcframework"
                COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET
    )
    return()
endif ()

# One directory per variant, named by its first triple, beside one `include/` every variant shares.
set(bundle "${NUMKONG_SWIFT_DIRECTORY}/CNumKong.artifactbundle")
string(REPLACE "," ";" triples "${NUMKONG_SWIFT_TRIPLES}")
list(GET triples 0 variant)
file(MAKE_DIRECTORY "${bundle}/${variant}")
file(COPY_FILE "${NUMKONG_SWIFT_ARCHIVE}" "${bundle}/${variant}/${archive_name}")
file(REMOVE_RECURSE "${bundle}/include")
file(COPY "${NUMKONG_SWIFT_HEADERS}/" DESTINATION "${bundle}/include")

set(entry [[{"path": "", "supportedTriples": [], "staticLibraryMetadata": {"headerPaths": ["include"],
    "moduleMapPath": "include/module.modulemap"}}]]
)
string(JSON entry SET "${entry}" path "\"${variant}/${archive_name}\"")
foreach (triple IN LISTS triples)
    string(JSON count LENGTH "${entry}" supportedTriples)
    string(JSON entry SET "${entry}" supportedTriples ${count} "\"${triple}\"")
endforeach ()

if (EXISTS "${bundle}/info.json")
    file(READ "${bundle}/info.json" info)
else ()
    set(info [[{"schemaVersion": "1.0", "artifacts": {"CNumKong": {"type": "staticLibrary", "variants": []}}}]])
endif ()
string(JSON info SET "${info}" artifacts CNumKong version "\"${NUMKONG_SWIFT_VERSION}\"")
# A rebuild of the same variant replaces it.
string(JSON count LENGTH "${info}" artifacts CNumKong variants)
while (count GREATER 0)
    math(EXPR count "${count} - 1")
    string(JSON path GET "${info}" artifacts CNumKong variants ${count} path)
    if (path STREQUAL "${variant}/${archive_name}")
        string(JSON info REMOVE "${info}" artifacts CNumKong variants ${count})
    endif ()
endwhile ()
string(JSON count LENGTH "${info}" artifacts CNumKong variants)
string(JSON info SET "${info}" artifacts CNumKong variants ${count} "${entry}")
file(WRITE "${bundle}/info.json" "${info}\n")
