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
        printf( "usage: ./benchmark_torchscript <PATH_TO_TORCHSCRIPT_FILE>\n" );
        return -1;
    }

    // Load torchscript module
    torch::jit::script::Module module;
    try 
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) 
    {
        printf( "error loading the model" );
        return -1;
    }

    // Initialize random data in cv::Mat format
    cv::Mat input_opencv = cv::imread( "../sather_gate.jpg", cv::IMREAD_GRAYSCALE );
    int img_height = 640;
    int img_width = 480;
    cv::resize( input_opencv, input_opencv, cv::Size( img_width, img_height ) );
    input_opencv.convertTo( input_opencv, CV_32FC1, 1.0f / 255.0f, 0 );

    int num_iter = 1000;
    std::vector<cv::Mat> inputs_opencv( num_iter );
    for ( int i = 0; i < num_iter; i++ ) // every iteration
    {
        inputs_opencv.emplace_back( input_opencv.clone() );
    }

    // Set device
    torch::Device device( torch::DeviceType::CPU );

    /* Timming */
    // We timming both convert cv::Mat to corresbonding data format and time to run the model
    auto start_time = std::chrono::steady_clock::now();

    for ( int i = 0; i < num_iter; ++i )
    {
        // Convert cv::Mat input to torchscript input
        std::vector<int64_t> dims = {1, img_height, img_width, 1};
        auto input_torch = torch::from_blob( inputs_opencv[ i ].data, dims, torch::kFloat32 ).to(device);
        input_torch = input_torch.permute({0,3,1,2});

        std::vector<torch::jit::IValue> inputs_torch;
        inputs_torch.push_back( input_torch );

        // Run model
        auto outputs_torch = module.forward(inputs_torch).toTuple();

        auto pts  = outputs_torch->elements()[0].toTensor().to(torch::kCPU).squeeze();
        auto desc = outputs_torch->elements()[1].toTensor().to(torch::kCPU).squeeze();

        cv::Mat pts_mat(cv::Size(3, pts.size(0)), CV_32FC1, pts.data_ptr<float>());
        cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data_ptr<unsigned char>());
    }

    auto end_time = std::chrono::steady_clock::now();

    // Print out timing information
    std::chrono::duration<double> duration_time = end_time - start_time;
    double duratin_time_in_sec = duration_time.count();
    duratin_time_in_sec /= num_iter;
    printf("TorchScript make %lf second to run each model inference\n", duratin_time_in_sec );
    printf("TorchScript model can run %d fps\n", int( 1.0 / duratin_time_in_sec ) );
}