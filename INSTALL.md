
## Step by step
```shell
# create conda env with python 3.7
conda create --name py3.7 python=3.7
conda activate py3.7

# remove any previous nvidia related package
sudo apt-get purge nvidia*

# add graphoc drive ppa
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

# search for potential driver
ubuntu-drivers devices

# We'll be install nvidia-driver 470 ( for pytorch version compatable )
# For which cuda driver, cuda toolkits match, check https://docs.nvidia.com/deploy/cuda-compatibility/index.html
sudo apt-get install nvidia-driver-470

# Install pytorch (1.1.0) from scratch
pip install pyyaml==5.4.1
git clone --recursive -b v1.0.1 https://github.com/pytorch/pytorch
cd pytorch && mkdir build && cd build
python3 ../tools/build_libtorch.py

# install cuda toolkit 10.0 (deprecated, no use)
# conda install cudatoolkit=10.0 -c pytorch

# Install pytroch (1.1.0) from conda (deprecated, no use)
# conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# export the cuda toolkits enviroment (deprecated, no use)
# export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/home/xiaosx/anaconda3/envs/py3.7/lib/python3.7/site-packages/torch/share/cmake/

# Install cuDNN 7.6.5 (deprecated, no use)
# conda install cudnn=7.6.5

# export the cudnn env (deprecated, no use)
# note: below my conda env is called py3.7
# export CUDNN_LIBRARY=/home/xiaosx/anaconda3/envs/py3.7/lib/libcudnn.so
# export CUDNN_INCLUDE_DIR=/home/xiaosx/anaconda3/envs/py3.7/include/
```


## Install cpp related dependency
1. OpenCV 3.4 [github](https://github.com/opencv/opencv) via CMake. 
    * the 3.4 version is used as compatable with [OpenARK](https://github.com/XiaoSong9905/OpenARK)
    * `git clone` the repo and `git checkout 3.4` to compile on correct branch


## Install optional dependency
```shell
# netron : model arch visualization tool
snap install netron

# htop : check cpu usage
sudo apt install htop
```