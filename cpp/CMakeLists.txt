# The C++ SDK over the C interface: `cpp/numkong/*.hpp` include the C headers `numkong::abi` carries and the vocabulary
# of `include/numkong/*.hpp`, so a consumer linking `numkong::cpp` gets the static library, whose public interface
# carries that vocabulary, and dispatches on its compiled kernels.
add_library(numkong_cpp INTERFACE)
add_library(numkong::cpp ALIAS numkong_cpp)
set_target_properties(numkong_cpp PROPERTIES EXPORT_NAME cpp)
target_sources(numkong_cpp INTERFACE "$<BUILD_INTERFACE:${numkong_cpp_sources_}>")
target_include_directories(
    numkong_cpp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(numkong_cpp INTERFACE numkong::abi numkong::static)
target_compile_features(numkong_cpp INTERFACE cxx_std_20)
