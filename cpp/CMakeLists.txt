# The C++ SDK over the C interface: `cpp/numkong/*.hpp` hold the vocabulary of types, vectors, tensors and matrices and
# the wrappers over the C headers `numkong::abi` carries, so a consumer linking `numkong::cpp` gets the static library
# and dispatches on its compiled kernels.
add_library(numkong_cpp INTERFACE)
add_library(numkong::cpp ALIAS numkong_cpp)
set_target_properties(numkong_cpp PROPERTIES EXPORT_NAME cpp)
target_sources(numkong_cpp INTERFACE "$<BUILD_INTERFACE:${numkong_cpp_sources_}>")
target_include_directories(
    numkong_cpp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(numkong_cpp INTERFACE numkong::abi numkong::static)
target_compile_features(numkong_cpp INTERFACE cxx_std_20)

# The smoke test of the SDK, against the compiled kernels a `numkong::cpp` consumer links, never the header-only engine.
if (NUMKONG_BUILD_TEST)
    add_executable(numkong_test_cpp test.cpp)
    target_link_libraries(numkong_test_cpp PRIVATE numkong::cpp numkong_warnings)
    add_test(NAME numkong_test_cpp COMMAND numkong_test_cpp)
endif ()
