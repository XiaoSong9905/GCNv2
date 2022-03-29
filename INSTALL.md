
## Install cpp related dependency
1. OpenCV 3.4 [github](https://github.com/opencv/opencv) via CMake. 
    * the 3.4 version is used as compatable with [OpenARK](https://github.com/XiaoSong9905/OpenARK)
    * `git clone` the repo and `git checkout 3.4` to compile on correct branch
2. TVM [github](https://github.com/apache/tvm) via CMake 
    * Default CMake install both the TVM compiler and TVM runtime. 
    * If you only want to build the GCNv2 OpenCV package, then you only need to install TVM runtime
    * If you want to convert (compile) onnx/pytorch model to tvm model, then you need to install both TVM runtime and compiler
3. LibTorch [pytorch.org](https://pytorch.org/cppdocs/installing.html), cpp interface of pytorch
4. ONNX runtime [github](https://github.com/microsoft/onnxruntime.git) and build from source, [doc](https://onnxruntime.ai/docs/build/inferencing.html)
    * `./build.sh --config RelWithDebInfo --build_shared_lib --parallel --config Release --build_wheel`

## Install python related dependency
1. create conda enviroment with python=3.8
2. `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
3. `pip3 install onnx onnxruntime`


## Install other dependency
1. netron `snap install netron` used to visualize model arch
    * you can use netron to open .onnx file


## Install optional dependency
1.  htop `sudo apt install htop`