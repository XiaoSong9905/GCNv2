//
// Load and run TorchScript model in cpp 
// Reference: https://pytorch.org/tutorials/advanced/cpp_export.html#step-3-loading-your-script-torch_model-in-c
// 
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <chrono> // timming
#include <vector>
#include <string>

int main(int argc, const char* argv[]) 
{
    if (argc != 3) 
    {
        printf( "usage: ./run_torchscript <TORCHSCRIPT_MODEL_PATH> <IMAGE_PATH>\n" );
        return -1;
    }

    // Load torchscript torch_model
    printf("..Load torchscript model\n");
    torch::jit::script::Module torch_model;
    try
    {
        // Deserialize the Module from a file using torch::jit::load().
        torch_model = torch::jit::load(argv[1]);
    }
    catch ( std::exception& e )
    {
        printf("Error loading model. %s\n", e.what() );
        return -1;
    }

    // Initialize random data in cv::Mat format
    printf("..Read input image\n");
    int img_height = 640;
    int img_width = 480;
    cv::Mat input_opencv = cv::imread( argv[2], cv::IMREAD_GRAYSCALE );
    input_opencv.convertTo( input_opencv, CV_32FC1, 1.0f / 255.0f, 0 );
    cv::resize( input_opencv, input_opencv, cv::Size( img_width, img_height ) );

    // Set device
    torch::Device device( torch::DeviceType::CPU );

    /* Timming */
    // We timming both convert cv::Mat to corresbonding data format and time to run the model
    int num_iter = 100;

    printf("..Benchmark for %d iterations\n", num_iter );
    auto start_time = std::chrono::steady_clock::now();

    for ( int i = 0; i < num_iter; ++i )
    {
        // Convert cv::Mat input to torchscript input
        std::vector<int64_t> dims = {1, img_height, img_width, 1};
        auto input_torch = torch::from_blob( input_opencv.data, dims, torch::kFloat32 ).to(device);
        input_torch = input_torch.permute({0,3,1,2});

        std::vector<torch::jit::IValue> inputs_torch;
        inputs_torch.push_back( input_torch );

        // Run model
        auto outputs_torch = torch_model.forward(inputs_torch).toTuple();

        torch::Tensor pts  = outputs_torch->elements()[0].toTensor().to(torch::kCPU).squeeze();
        torch::Tensor desc = outputs_torch->elements()[1].toTensor().to(torch::kCPU).squeeze();

        // TODO: add convert back to cv::mat
    }

    auto end_time = std::chrono::steady_clock::now();

    // Print out timing information
    std::chrono::duration<double> duration_time = end_time - start_time;
    double duratin_time_in_sec = duration_time.count();
    duratin_time_in_sec /= num_iter;
    printf("..TorchScript make %lf second to run each model inference\n", duratin_time_in_sec );
    printf("..TorchScript model can run %d fps\n", int( 1.0 / duratin_time_in_sec ) );
}