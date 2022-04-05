cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( GCNv2_model_tvm )

# Set C++ standard
# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)


# Set path to find OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# Set tvm related
# TVM related settings
set(TVM_ROOT "/home/xiaosx/work/tvm")

include_directories(
    "${TVM_ROOT}/include"
    "${TVM_ROOT}/3rdparty/dmlc-core/include"
    "${TVM_ROOT}/3rdparty/dlpack/include"
)

add_definitions("-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>")

# Run TVM
add_executable(run_tvm run_tvm.cpp )
target_link_libraries(run_tvm tvm_runtime )
target_link_libraries(run_tvm "${OpenCV_LIBS}")
set_property(TARGET run_tvm PROPERTY CXX_STANDARD 14)

# Benchmark TVM
add_executable(benchmark_tvm benchmark_tvm.cpp)
target_link_libraries(benchmark_tvm tvm_runtime )
target_link_libraries(benchmark_tvm "${OpenCV_LIBS}")
set_property(TARGET benchmark_tvm PROPERTY CXX_STANDARD 14)