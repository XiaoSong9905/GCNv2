cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( gcnv2_example )

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

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

# Set path to find Torch
# 1. set enviroment variable (can also be pytorch installed with python)
# export CMAKE_PREFIX_PATH=/absolute/path/to/unzipped_libtorch:$CMAKE_PREFIX_PATH
# 
# Example, my conda enviroment is named `py3.7`
# export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/home/xiaosx/anaconda3/envs/py3.7/lib/python3.7/site-packages/torch/share/cmake/
# export CMAKE_PREFIX_PATH=/home/xiaosx/anaconda3/envs/py3.7/lib/python3.7/site-packages/torch/share/cmake/
# 
# 2. cmake command line input
# cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch:$CMAKE_PREFIX_PATH ..
# cmake -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/home/xiaosx/anaconda3/envs/py3.7/lib/python3.7/site-packages/torch/share/cmake/ ..
# 
# All below methods require already download pytorch with conda / pip
# Run conda activate before running `cmake ..`
# 
# 3. set enviroment variable (only work for later version of pytorch)
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
include_directories(BEFORE ${TORCH_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS})

find_package( gcnv2 REQUIRED )
include_directories( ${GCNV2_INCLUDE_DIRS} )
message( STATUS "Found GCNv2 : ${GCNV2_INCLUDE_DIRS}")

add_executable( demo demo.cc  )
target_link_libraries( demo ${OpenCV_LIBS} gcnv2)
