# cagra_build_search against the cuVS CMake package (libcuvs 25.02 of the cagra environment).
# Configure with CMAKE_PREFIX_PATH pointing to that environment, see ../../build_cagra.sh.
cmake_minimum_required(VERSION 3.26)
project(equal_degree_cagra LANGUAGES CXX CUDA)
find_package(cuvs REQUIRED)
add_executable(cagra_build_search cagra_build_search.cu)
# In RAPIDS 25.02 the RMM logger implementation is the separate target rmm::rmm_logger_impl.
target_link_libraries(cagra_build_search PRIVATE cuvs::cuvs rmm::rmm_logger_impl)
set_target_properties(cagra_build_search PROPERTIES CUDA_ARCHITECTURES 80 CXX_STANDARD 17 CUDA_STANDARD 17
                      BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib")
target_compile_options(cagra_build_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
