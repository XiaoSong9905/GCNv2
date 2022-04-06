cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project name
project(gcnv2)

# Set version
set(GCNV2_MAJOR_VERSION 1)
set(GCNV2_MINOR_VERSION 0)
set(GCNV2_PATCH_VERSION 0)
set(GCNV2_VERSION ${GCNV2_MAJOR_VERSION}.${GCNV2_MINOR_VERSION}.${GCNV2_PATCH_VERSION})

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
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/gcnv2)
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

# make sure we use Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
else()
  message(STATUS "Building with build type '${CMAKE_BUILD_TYPE}'")
endif()

# require OpenCV
find_package( OpenCV REQUIRED )
include_directories(BEFORE PUBLIC ${OpenCV_INCLUDE_DIRS} )

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
# export CMAKE_PREFIX_PATH=/absolute/path/to/unzipped_libtorch:$CMAKE_PREFIX_PATH
# 
# 2. cmake command line input
# cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch:$CMAKE_PREFIX_PATH ..
# 
# All below methods require already download pytorch with conda / pip
# Run conda activate before running `cmake ..`
# 
# 3. set enviroment variable
# export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
#
# 4. set Torch_Dir (need to run conda activate py3.8 first)
# set(Torch_DIR "/home/xiaosx/anaconda3/envs/py3.8/lib/python3.8/site-packages/torch/share/cmake/Torch/")
#
# 5. set CMAKE_PREFIX_PATH (need to run conda activate py3.8 first)

# If you want to use python3's pytorch, uncomment below section
# execute_process(
#   COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
#   OUTPUT_VARIABLE TorchPath )
# list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

# If you prefer to use your own pytorch version, use methods 1.2 mentioned above

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(BEFORE PUBLIC ${TORCH_INCLUDE_DIRS})

if (${TORCH_FOUND})
  message(STATUS "Found Torch dir : ${TORCH_INCLUDE_DIRS}")
  message(STATUS "Found Torch obj : ${TORCH_LIBRARIES}")
endif()

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

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  add_definitions(-mfpu=neon -DHAVE_OPENCV -Wall -pedantic -std=c++0x -fPIC)
else()
  message(STATUS "Assuming SSE instructions available.")
  add_definitions(-mssse3 -DHAVE_OPENCV -Wall -pedantic -std=c++0x -fPIC)
endif()

# set up the output tree
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# build GCNV2
include_directories(include)
add_library(${PROJECT_NAME} STATIC src/gcnv2.cc)


# and link it
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS} 
  ${TORCH_LIBRARIES} )

# installation if required
install(TARGETS ${PROJECT_NAME} 
  EXPORT gcnv2Targets
  ARCHIVE DESTINATION ${INSTALL_LIB_DIR})

install(DIRECTORY include/ DESTINATION 
  ${INSTALL_INCLUDE_DIR} COMPONENT 
  dev FILES_MATCHING PATTERN "*.h")

# installation is invoked in the individual modules...
export (TARGETS 
    gcnv2
    FILE "${PROJECT_BINARY_DIR}/gcnv2Targets.cmake")
export ( PACKAGE gcnv2 )

# Create the gcnv2Config.cmake and gcnv2ConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
set(CONF_INCLUDE_DIRS 
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}"
)

configure_file(cmake/gcnv2Config.cmake.in
  "${PROJECT_BINARY_DIR}/gcnv2Config.cmake" @ONLY)

# ... for the install tree
set(CONF_INCLUDE_DIRS "\${GCNV2_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/gcnv2Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gcnv2Config.cmake" @ONLY)

# ... for both
configure_file(cmake/gcnv2ConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/gcnv2ConfigVersion.cmake" @ONLY)
 
# Install the gcnv2Config.cmake and gcnv2ConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gcnv2Config.cmake"
  "${PROJECT_BINARY_DIR}/gcnv2ConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}")

# Install the export set for use with the install-tree
install(EXPORT gcnv2Targets DESTINATION
  "${INSTALL_CMAKE_DIR}")