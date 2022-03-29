
## Description

This folder contain script & code related to train actual GCNv2 model. 

The model is then converted to onnx format to be used by TVM

## File structure
```shell
arch.py                           # model architecture
train.py                          # train model and save (model structure & state dict) to PyTorch.pth file
convert_pytorch_to_onnx.py        # convert PyTorch.pth model to ONNX.onnx model,
convert_pytorch_to_torchscript.py # convert PyTorch.pth model to TorchScript.pt model
convert_onnx_to_tvm_by_tvmc.py    # convert ONNX.onnx model to TVM model through TVMC (simple wrapper). 
                                  # This should be enough for our project
convert_onnx_to_tvm_by_pyapi      # convert ONNX.onnx model to TVM model through Python API. 
                                  # This is more advanced conversion with multiple hyperparameter setting. Optional
benchmark*.cpp                    # benchmark different model representation's performence
```

## Usage
```shell
# train model
python3 arch.py

# convert model to onnx
python3 convert_to_onnx.py

# convert model to torchscript
python3 convert_to_torchscript.py

# build benchmark code
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 10

# Run benchmark code
./benchmark_torchscript ../GCNv2.pt

```

## TODO
1. try pytorch quantilization techniques
2. try tvm quantilization techniques
3. benchmark different method performence
4. benchmark onnx runtime. The onnx doc is to bad