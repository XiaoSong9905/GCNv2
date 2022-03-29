//
// Load and run ONNX model in cpp 
// Reference: 
// 1. https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/
// 2. https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
//
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <chrono> // timming
#include <vector>
#include <string>

int main(int argc, const char* argv[]) 
{
    if (argc != 2) 
    {
        printf( "usage: ./run_onnx <ONNX_MODEL_FILENAME>\n" );
        return -1;
    }

    // Load ONNX module

    // Set input
    printf("..Read in input\n"); fflush(stdout);
    int img_width = 640;
    int img_height = 480;
    cv::Mat image = cv::imread( "../sather_gate.jpg", cv::IMREAD_GRAYSCALE );
    image.convertTo(image, CV_32FC1, 1.f / 255.f , 0);
    cv::resize(image, image, cv::Size(img_width, img_height));

    // Convert to onnx input format
    printf("..Convert to onnx data\n"); fflush(stdout);

    // Run model
    printf("..Run model\n"); fflush(stdout);

    // Extract output
    printf("..Extract output\n"); fflush( stdout );

}