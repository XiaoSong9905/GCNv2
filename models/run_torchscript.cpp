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
        printf( "usage: ./run_torchscript <PATH_TO_TORCHSCRIPT_FILE>\n" );
        return -1;
    }

    // Load torchscript module
    torch::jit::script::Module module;
    try 
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        printf("Try to load %s\n", argv[1]);
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) 
    {
        printf( "error loading the model" );
        throw std::runtime_error("Unable to load torchscript file\n");
    }

    // Set device
    torch::Device device( torch::DeviceType::CPU );

    // Set input
    int img_height = 640;
    int img_width = 480;
    std::vector<torch::jit::IValue> inputs_torch;
    inputs_torch.push_back( torch::ones({1, 1, img_height, img_width}));

    // Run model
    auto outputs_torch = module.forward(inputs_torch).toTuple();

    auto pts  = outputs_torch->elements()[0].toTensor().to(torch::kCPU).squeeze();
    auto desc = outputs_torch->elements()[1].toTensor().to(torch::kCPU).squeeze();

    // Convert output to cv::Mat to ensure model acrually run
    cv::Mat pts_mat(cv::Size(3, pts.size(0)), CV_32FC1, pts.data_ptr<float>());
    cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data_ptr<unsigned char>());

    printf("pts_mat size %d %d\n", pts_mat.size().height, pts_mat.size().width );
    printf("desc_mat size %d %d\n", desc_mat.size().height, desc_mat.size().width );
}