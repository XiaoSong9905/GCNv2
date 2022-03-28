cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Set project name
project( GCNv2_model )

# Set C++ standard
# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O0")

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
include_directories( ${OpenCV_INCLUDE_DIRS}) 

# Run TorchScript
add_executable(run_torchscript run_torchscript.cpp)
target_link_libraries(run_torchscript "${TORCH_LIBRARIES}")
target_link_libraries(run_torchscript "${OpenCV_LIBS}")
set_property(TARGET run_torchscript PROPERTY CXX_STANDARD 14)

# Benchmark TorchScript
add_executable(benchmark_torchscript benchmark_torchscript.cpp)
target_link_libraries(benchmark_torchscript "${TORCH_LIBRARIES}")
target_link_libraries(benchmark_torchscript "${OpenCV_LIBS}")
set_property(TARGET benchmark_torchscript PROPERTY CXX_STANDARD 14)

# Benchmark ONNX


# Benchmark TVM


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