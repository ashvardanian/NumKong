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
