
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

# install cuda toolkit 10.0 ( for pytorch version compatable )
# Install instruction can be found here https://developer.nvidia.com/cuda-10.0-download-archive
# If you install the wrong cudatookits version, follow https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one to remove previous version
# TODO may considered install pytorch in other way
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# export the cuda toolkits enviroment
export CMAKE_PREFIX_PATH=/home/xiaosx/anaconda3/envs/py3.7/lib/python3.7/site-packages/torch/share/cmake/

# Install cuDNN 7.6.5 (for compatable with cuda toolkits)
# https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux
# TODO may considered install cudnn in other way
conda install cudnn=7.6.5

# export the cudnn env
# note: below my conda env is called py3.7
export CUDNN_LIBRARY=/home/xiaosx/anaconda3/envs/py3.7/lib/libcudnn.so
export CUDNN_INCLUDE_DIR=/home/xiaosx/anaconda3/envs/py3.7/include/
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