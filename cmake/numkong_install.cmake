# Install rules, the CMake package files, and the Swift artifact.

if (NUMKONG_INSTALL)
    install(
        TARGETS ${numkong_libraries_}
        EXPORT numkongTargets
        ARCHIVE
        BUNDLE
        FRAMEWORK
        LIBRARY
        OBJECTS
        PRIVATE_HEADER
        PUBLIC_HEADER
        RESOURCE
        RUNTIME
    )
endif ()

# The static library as SwiftPM links it, beside the headers and their module map: an XCFramework on Apple platforms,
# and elsewhere an SE-0482 artifact bundle. Each build adds its target to the one in NUMKONG_SWIFT_DIRECTORY, so a
# release fills one artifact from a build per target.
if (NUMKONG_BUILD_SWIFT)
    set(NUMKONG_SWIFT_DIRECTORY "${PROJECT_BINARY_DIR}" CACHE PATH "Where the Swift artifact lands")
    if (APPLE)
        # The SDK names the XCFramework library an archive joins, and Mac Catalyst builds with the macOS one.
        get_filename_component(numkong_swift_sysroot_name_ "${CMAKE_OSX_SYSROOT}" NAME)
        string(TOLOWER "${numkong_swift_sysroot_name_}" numkong_swift_sysroot_lowercase_)
        string(REGEX REPLACE "[0-9.]*\\.sdk$" "" numkong_swift_sdk_ "${numkong_swift_sysroot_lowercase_}")
        if (CMAKE_C_FLAGS MATCHES "-macabi")
            set(numkong_swift_sdk_ maccatalyst)
        elseif (NOT numkong_swift_sdk_)
            set(numkong_swift_sdk_ macosx)
        endif ()
        if (CMAKE_OSX_ARCHITECTURES)
            set(numkong_swift_architecture_ "${CMAKE_OSX_ARCHITECTURES}")
        else ()
            set(numkong_swift_architecture_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(numkong_swift_target_ -D "NUMKONG_SWIFT_SDK=${numkong_swift_sdk_}" -D
                                  "NUMKONG_SWIFT_ARCHITECTURE=${numkong_swift_architecture_}"
        )
    else ()
        # SwiftPM matches a variant by the exact triple it builds for, spelled as Swift spells it.
        if (numkong_arch_x8664_)
            set(numkong_swift_arch_ x86_64)
        elseif (numkong_arch_arm64_)
            set(numkong_swift_arch_ aarch64)
        elseif (numkong_arch_arm32_)
            set(numkong_swift_arch_ armv7)
        else ()
            set(numkong_swift_arch_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(numkong_swift_triple_)
        if (numkong_os_linux_)
            set(numkong_swift_triple_ "${numkong_swift_arch_}-unknown-linux-gnu")
        elseif (numkong_os_android_)
            set(numkong_swift_triple_ "${numkong_swift_arch_}-unknown-linux-android${ANDROID_NATIVE_API_LEVEL}")
        elseif (numkong_os_windows_)
            set(numkong_swift_triple_ "${numkong_swift_arch_}-unknown-windows-msvc")
        elseif (numkong_os_wasi_)
            set(numkong_swift_triple_ "wasm32-unknown-wasip1")
        endif ()
        set(NUMKONG_SWIFT_TRIPLES "${numkong_swift_triple_}" CACHE STRING
                                                                   "Swift target triples the artifact bundle serves"
        )
        if (NOT NUMKONG_SWIFT_TRIPLES)
            message(FATAL_ERROR "NUMKONG_BUILD_SWIFT needs NUMKONG_SWIFT_TRIPLES on ${CMAKE_SYSTEM_NAME}")
        endif ()
        set(numkong_swift_target_ -D "NUMKONG_SWIFT_TRIPLES=$<JOIN:${NUMKONG_SWIFT_TRIPLES},,>" -D
                                  "NUMKONG_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    add_custom_target(
        numkong_swift ALL
        COMMAND
            "${CMAKE_COMMAND}" -D NUMKONG_SWIFT_MODULE=NumKongC -D "NUMKONG_SWIFT_ARCHIVE=$<TARGET_FILE:numkong_static>"
            -D "NUMKONG_SWIFT_HEADERS=${PROJECT_SOURCE_DIR}/include" -D
            "NUMKONG_SWIFT_DIRECTORY=${NUMKONG_SWIFT_DIRECTORY}" ${numkong_swift_target_} -P
            "${PROJECT_SOURCE_DIR}/cmake/numkong_swift_artifact.cmake"
        DEPENDS numkong_static
        VERBATIM
    )
endif ()

# Install & export — gated, so a parent project's `make install` ships NumKong only on request, and
# `find_package(numkong)` works after installation, mirroring ForkUnion's packaging.
if (NUMKONG_INSTALL)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY c/ DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${PROJECT_NAME}/src)
    install(TARGETS numkong_header EXPORT numkongTargets)
    install(EXPORT numkongTargets FILE numkongTargets.cmake NAMESPACE numkong::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numkong
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/numkongConfigVersion.cmake" VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/numkongConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/numkongConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numkong
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/numkongConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/numkongConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numkong
    )
endif ()
