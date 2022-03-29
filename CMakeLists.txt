cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( GCNv2 )

# Version number
set(GCNv2_MAJOR_VERSION 1)
set(GCNv2_MINOR_VERSION 0)
set(GCNv2_PATCH_VERSION 0)
set(GCNv2_VERSION ${GCNv2_MAJOR_VERSION}.${GCNv2_MINOR_VERSION}.${GCNv2_PATCH_VERSION})

# Set C++ standard
# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++14 -fPIC")

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/brisk)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

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
# Build Release
# set(CMAKE_BUILD_TYPE Release)

if(APPLE)
  # The clang compiler (on osx) is somehow much more strict
  # than the compilers on ubuntu and so this does not seem
  # possible on OSX just yet.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++14 -fPIC")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -march=native -Wall -std=c++14 -fPIC")
endif()

# Arm / X86 optimize flag
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()

# Set path to find Torch
# 1. set enviroment variable
# export CMAKE_PREFIX_PATH=/absolute/path/to/unzipped_libtorch
# 
# 2. cmake command line input
# cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
# 
# All below methods require already download pytorch with conda / pip
# Run conda activate before running `cmake ..`
# 
# 3. set enviroment variable
# export CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
#
# 4. set Torch_Dir (need to run conda activate py3.8 first)
# set(Torch_DIR "/home/xiaosx/anaconda3/envs/py3.8/lib/python3.8/site-packages/torch/share/cmake/Torch/")
#
# 5. set CMAKE_PREFIX_PATH (need to run conda activate py3.8 first)
execute_process(
  COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set path to find OpenCV
find_package( OpenCV REQUIRED )
include_directories( BEFORE ${OpenCV_INCLUDE_DIRS} )

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET benchmark_torchscript
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:benchmark_torchscript>)
endif (MSVC)

# GCNv2
include_directories(include)
add_library(${PROJECT_NAME} STATIC 
                            src/gcn.cc )
            
# Add link
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${Torch_LIBS})

# installation if required
install(TARGETS ${PROJECT_NAME} EXPORT gcnv2Targets ARCHIVE DESTINATION ${INSTALL_LIB_DIR})
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.h")
