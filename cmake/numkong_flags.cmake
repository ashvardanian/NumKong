# Sanitizers, and the warning flags NumKong's own targets take through `numkong_warnings`.

# Sanitizers instrument every target this directory builds, the fetched {fmt} included, but never the probes.
if (NUMKONG_BUILD_SANITIZERS STREQUAL "address" AND MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/fsanitize=address>")
elseif (NUMKONG_BUILD_SANITIZERS STREQUAL "address")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address,undefined>")
    add_link_options("$<$<LINK_LANGUAGE:C,CXX>:-fsanitize=address,undefined>")
    # GCC misplaces the ASan frame of functions taking an SVE local's address
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        add_compile_options(-fno-stack-protector -fno-sanitize-address-use-after-scope)
    endif ()
elseif (NUMKONG_BUILD_SANITIZERS STREQUAL "thread" AND NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=thread>")
    add_link_options("$<$<LINK_LANGUAGE:C,CXX>:-fsanitize=thread>")
elseif (NOT NUMKONG_BUILD_SANITIZERS STREQUAL "off")
    message(FATAL_ERROR "NUMKONG_BUILD_SANITIZERS is off, address or thread, the last with Clang or GCC, "
                        "not '${NUMKONG_BUILD_SANITIZERS}'"
    )
endif ()

# Warnings apply to NumKong's own targets only, so fetched dependencies build with their own flags. Not for CUDA:
# nvcc hands the host compiler line markers that `-pedantic` flags on every line.
add_library(numkong_warnings INTERFACE)
if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
    target_compile_options(
        numkong_warnings
        INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-pedantic;-Wundef>"
                  "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wdocumentation>"
    )
    if (NUMKONG_WERROR)
        target_compile_options(numkong_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
    endif ()
elseif (MSVC)
    # The conformance switches MSVC leaves off by default, which every other supported compiler behaves as.
    target_compile_options(
        numkong_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor;/utf-8;/bigobj>"
                                   "$<$<COMPILE_LANGUAGE:CXX>:/permissive-;/Zc:__cplusplus>"
    )
    if (NUMKONG_WERROR)
        target_compile_options(numkong_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|IntelLLVM)$")
    add_compile_options(-ferror-limit=1 -Wno-tautological-constant-compare)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fmax-errors=1 -Wno-tautological-constant-compare -Wno-psabi)
endif ()
