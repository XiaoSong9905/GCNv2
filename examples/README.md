# Example of using GCNv2 in your own project

This folder is used to test if GCNv2 is installed & run correctely

### Usage
```shell
export GCNV2_TORCH_MODEL_PATH=<PATH/TO/TORCHSCRIPT/MODEL>
mkdir build && cd build
cmake ..
make -j 10

./demo ../image1.png ../image2.png
```