//
// Load and run TorchScript model in cpp 
// Reference: https://pytorch.org/tutorials/advanced/cpp_export.html#step-3-loading-your-script-module-in-c
// 
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <chrono> // timming
#include <vector>
#include <string>

int main(int argc, const char* argv[]) 
{
    if (argc != 2) 
    {
        printf( "usage: ./run_torchscript <TORCHSCRIPT_MODEL_FILENAME>\n" );
        return -1;
    }

    // Load torchscript module
    torch::jit::script::Module model;
    try 
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        printf("..Try to load %s\n", argv[1]); fflush(stdout);
        model = torch::jit::load(argv[1]);
    }
    catch ( std::exception& e )
    {
        printf("Error loading model. %s\n", e.what() );
        return -1;
    }

    torch::Device device( torch::DeviceType::CPU );

    // Set input
    printf("..Read in input\n"); fflush(stdout);
    int img_width = 640;
    int img_height = 480;
    cv::Mat image = cv::imread( "../sather_gate.jpg", cv::IMREAD_GRAYSCALE );
    image.convertTo(image, CV_32FC1, 1.f / 255.f , 0);
    cv::resize(image, image, cv::Size(img_width, img_height));

    // Convert to torchscript format
    printf("..Convert to torchscript data\n"); fflush(stdout);
    std::vector<int64_t> dims = {1, img_height, img_width, 1};
    auto img_var = torch::from_blob( image.data, dims, torch::kFloat32).to(device);
    img_var = img_var.permute({0,3,1,2});

    // Run model
    printf("..Run model\n"); fflush(stdout);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var);
    auto output = model.forward(inputs).toTuple();

    // Extract output
    printf("..Extract output\n"); fflush( stdout );
    torch::Tensor pts  = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
    torch::Tensor desc = output->elements()[1].toTensor().to(torch::kCPU).squeeze();
}