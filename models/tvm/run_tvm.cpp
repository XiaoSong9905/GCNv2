//
// Load and run TVM model in cpp
// Refernece 
// 1. https://bbs.cvmart.net/articles/4357 
// 2. https://github.com/dmlc/nnvm/blob/master/docs/how_to/deploy.md
// 
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <opencv2/opencv.hpp>

int main( int argc, const char* argv[])
{
    if (argc != 3) 
    {
        printf( "usage: ./run_tvm <TVM_MODEL_PATH> <IMAGE_PATH>\n" );
        return -1;
    }

    // Load tvm model
    printf("..Try to load tvm model %s\n", argv[1]); fflush(stdout);


    // Set input
    printf("..Read in input\n"); fflush(stdout);
    int img_width = 640;
    int img_height = 480;
    cv::Mat image = cv::imread( argv[2], cv::IMREAD_GRAYSCALE );
    image.convertTo(image, CV_32FC1, 1.f / 255.f , 0);
    cv::resize(image, image, cv::Size(img_width, img_height));

    // Convert input to tvm format
    printf("..Convert to tvm data\n"); fflush(stdout);


    // Run model
    printf("..Run model\n"); fflush(stdout);


    // Extract output
    printf("..Extract output\n"); fflush( stdout );

}



void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/test_relay_add.so");
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      static_cast<float*>(x->data)[i * 2 + j] = i * 2 + j;
    }
  }
  // set the right input
  set_input("x", x);
  // run the code
  run();
  // get the output
  get_output(0, y);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      ICHECK_EQ(static_cast<float*>(y->data)[i * 2 + j], i * 2 + j + 1);
    }
  }
}
