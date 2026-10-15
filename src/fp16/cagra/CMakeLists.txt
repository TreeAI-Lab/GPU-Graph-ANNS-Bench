# CAGRA resident search with float or half vectors. Uses the CMake package of cuVS, which carries the include
# directories and compile options that RAPIDS needs.
cmake_minimum_required(VERSION 3.26)
project(cagra_fp16 LANGUAGES CXX CUDA)
find_package(cuvs REQUIRED)
add_executable(cagra_resident_fp16 cagra_resident_fp16.cu)
# In RAPIDS 25.02 the RMM logger implementation is the separate target rmm::rmm_logger_impl, which the
# executable has to link.
target_link_libraries(cagra_resident_fp16 PRIVATE cuvs::cuvs rmm::rmm_logger_impl)
set_target_properties(cagra_resident_fp16 PROPERTIES CUDA_ARCHITECTURES 80 CXX_STANDARD 17 CUDA_STANDARD 17
                      BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib")
target_compile_options(cagra_resident_fp16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
