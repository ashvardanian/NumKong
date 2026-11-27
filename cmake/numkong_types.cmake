# Guards on `numkong::types`: the target gains no define and no link beyond `numkong::abi`, every container header
# compiles alone, and a probe over all of them neither defines nor references a symbol the library defines. Run with
# `-P`, this file is the `numkong_types_exports` test itself.

if (CMAKE_SCRIPT_MODE_FILE)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${NUMKONG_BINARY_DIR}" --config "${NUMKONG_CONFIG}" --target
                numkong_types_exports numkong_static COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND "${NUMKONG_NM}" -g -j --defined-only "${NUMKONG_LIBRARY}"
        OUTPUT_FILE "${NUMKONG_BINARY_DIR}/numkong_types/library.symbols" COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND "${NUMKONG_NM}" -g -j "${NUMKONG_PROBE}" OUTPUT_FILE "${NUMKONG_BINARY_DIR}/numkong_types/probe.symbols"
                                                                     COMMAND_ERROR_IS_FATAL ANY
    )
    file(STRINGS "${NUMKONG_BINARY_DIR}/numkong_types/library.symbols" library_symbols_)
    file(STRINGS "${NUMKONG_BINARY_DIR}/numkong_types/probe.symbols" probe_symbols_)
    set(clashes_)
    foreach (symbol_ IN LISTS probe_symbols_)
        if (symbol_ IN_LIST library_symbols_)
            list(APPEND clashes_ "${symbol_}")
        endif ()
    endforeach ()
    if (clashes_)
        message(FATAL_ERROR "numkong::types headers define or reference library symbols: ${clashes_}")
    endif ()
    return()
endif ()

function (numkong_check_types_target)
    get_target_property(definitions_ numkong_types INTERFACE_COMPILE_DEFINITIONS)
    get_target_property(links_ numkong_types INTERFACE_LINK_LIBRARIES)
    if (definitions_ OR NOT links_ STREQUAL "numkong_abi")
        message(FATAL_ERROR "numkong::types defines nothing and links only numkong::abi, not `${definitions_}` and "
                            "`${links_}`"
        )
    endif ()
endfunction ()
cmake_language(DEFER CALL numkong_check_types_target)

if (NUMKONG_BUILD_TEST)
    file(GLOB numkong_types_headers_ RELATIVE "${PROJECT_SOURCE_DIR}/include"
         "${PROJECT_SOURCE_DIR}/include/numkong/*.hpp"
    )
    set(numkong_types_units_)
    set(numkong_types_probe_ "")
    foreach (header_ IN LISTS numkong_types_headers_)
        file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/numkong_types/${header_}.cpp" CONTENT "#include \"${header_}\"\n")
        list(APPEND numkong_types_units_ "${PROJECT_BINARY_DIR}/numkong_types/${header_}.cpp")
        string(APPEND numkong_types_probe_ "#include \"${header_}\"\n")
    endforeach ()
    file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/numkong_types/exports.cpp" CONTENT "${numkong_types_probe_}")

    # Built by the tests rather than by default, so a failure reads as a failed guard.
    add_library(numkong_types_standalone OBJECT EXCLUDE_FROM_ALL ${numkong_types_units_})
    target_link_libraries(numkong_types_standalone PRIVATE numkong_types numkong_warnings)
    target_compile_features(numkong_types_standalone PRIVATE cxx_std_20)
    add_test(NAME numkong_types_standalone COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --config
                                                   $<CONFIG> --target numkong_types_standalone
    )

    # Every inline function is emitted, so the references in its body reach the symbol table.
    add_library(numkong_types_exports STATIC EXCLUDE_FROM_ALL "${PROJECT_BINARY_DIR}/numkong_types/exports.cpp")
    target_link_libraries(numkong_types_exports PRIVATE numkong_types)
    target_compile_features(numkong_types_exports PRIVATE cxx_std_20)
    target_compile_options(
        numkong_types_exports PRIVATE "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-O0;-femit-all-decls>"
                                      "$<$<CXX_COMPILER_ID:GNU>:-O0;-fkeep-inline-functions>"
    )
    # Both build in the one tree, which two concurrent builds would race over.
    set_tests_properties(numkong_types_standalone PROPERTIES RESOURCE_LOCK numkong_build_tree)
    if (CMAKE_NM)
        add_test(
            NAME numkong_types_exports
            COMMAND
                "${CMAKE_COMMAND}" -D "NUMKONG_BINARY_DIR=${PROJECT_BINARY_DIR}" -D "NUMKONG_CONFIG=$<CONFIG>" -D
                "NUMKONG_NM=${CMAKE_NM}" -D "NUMKONG_LIBRARY=$<TARGET_FILE:numkong_static>" -D
                "NUMKONG_PROBE=$<TARGET_FILE:numkong_types_exports>" -P "${CMAKE_CURRENT_LIST_FILE}"
        )
        set_tests_properties(numkong_types_exports PROPERTIES RESOURCE_LOCK numkong_build_tree)
    endif ()
endif ()
