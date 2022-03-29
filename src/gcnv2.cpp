#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <gcnv2/gcnv2.h>

namespace gcnv2
{ 

GCNv2DetectorDescriptor::GCNv2DetectorDescriptor( \
    RUNTIME_TYPE runtime_type, int img_height, int img_width, bool use_cuda ) : \
    runtime_type( runtime_type ), \
    img_height( img_height ), \
    img_width( img_width ), \
    use_cuda( use_cuda ) 
{
    // Load model
    switch ( runtime_type )
    {
    case RUNTIME_TYPE::TORCH:
        initTorch( );
        break;
    case RUNTIME_TYPE::ONNX:
        initONNX( );
        break;
    case RUNTIME_TYPE::TVM:
        initTVM( );
        break;
    default:
        throw std::runtime_error("Not supported runtime type\n");
    }

}

// TODO: add ONNX, TVM constructor 

GCNv2DetectorDescriptor::~GCNv2DetectorDescriptor() {}


void GCNv2FeatureDetector::initTorch( )
{
    // Set export GCNV2_TORCH_MODEL_PATH="/path/to/model.pt"
    std::string model_filename( std::getenv("GCNV2_TORCH_MODEL_PATH") );
    
    // Try load model
    try
    {
        // Deserialize the Module from a file using torch::jit::load().
        torch_model = std::make_unique<torch::jit::script::Module>( torch::jit::load( model_filename ) );
    }
    catch ( std::exception& e )
    {
        printf("Error loading model. [%s]\n", e.what() );
        exit(-1);
    }

    if ( use_cuda )
    {
        torch_model->to( torch::kCUDA );
        torch_device = torch::DeviceType::CUDA;
    }
    else
    {
        torch_device = torch::DeviceType::CPU;
    }
}


void GCNv2FeatureDetector::initONNX( )
{
    // Set export GCNV2_ONNX_MODEL_PATH="/path/to/model.pt"
    std::string model_filename( std::genenv("GCNV2_ONNX_MODEL_PATH") );
}


void GCNv2FeatureDetector::initTVM( )
{
    // Set export GCNV2_ONNX_MODEL_PATH="/path/to/model.pt"
    std::string model_filename( std::genenv("GCNV2_ONNX_MODEL_PATH") );
}


void GCNv2FeatureDetector::detectAndCompute( cv::InputArray _image, \
                                             cv::InputArray _mask, \
                                             std::vector<cv::KeyPoint>& _keypoints, \
                                             cv::OutputArray& _descriptors, \
                                             bool _useProvidedKeypoints )
{
    // Currently don't support provided keypoint
    if ( _useProvidedKeypoints || _keypoints.size() )
    {
        throw std::runtime_error("GCNv2FeatureDetector::detectAndCompute do not support provided keypoint\n" );
    }

    if ( !_mask.empty() )
    {
        throw std::runtime_error("GCNv2FeatureDetector::detectAndCompute do not support customize mask\n");
    }

    // No input image, return
    if ( _image.empty() )
    {
        return;
    }

    // Convert image to gray scale
    cv::Mat _gray_image_fp32 = _image.getMat();
    if ( _gray_image_fp32.type() != CV_8UC1 || _gray_image_fp32.type() != CV_32FC1 )
    {
        cv::cvtColor( _gray_image_fp32, _gray_image_fp32, cv::COLOR_BGR2GRAY );
    }

    // Convert image to fp32
    if ( _gray_image_fp32.type() == CV_8UC1 )
    {
        _gray_image_fp32.convertTo( _gray_image_fp32, CV_32FC1, 1.0f / 255.0f , 0 );
    }

    // Resize image
    // Model only accept fixed size
    if ( _gray_image_fp32.size().height != img_height || _gray_image_fp32.size().width != img_width )
    {
        cv::resize( _gray_image_fp32, _gray_image_fp32, cv::Size( img_width, img_height ) );
    }

    // Run DL model and extracto keypoints
    switch ( runtime_type )
    {
    case RUNTIME_TYPE::TORCH:
        detectAndComputeTorch( _gray_image_fp32, _keypoints, _descriptors );
        break;
    case RUNTIME_TYPE::ONNX:
        detectAndComputeONNX( _gray_image_fp32, _keypoints, _descriptors );
        break;
    case RUNTIME_TYPE::TVM:
        detectAndComputeTVM( _gray_image_fp32, _keypoints, _descriptors );
        break;
    }
}


void GCNv2DetectorDescriptor::detectAndComputeTorch( cv::Mat& _gray_image_fp32, \
                                                     std::vector<cv::KeyPoint>& _keypoints, \
                                                     cv::OutputArray& _descriptors )
{
    // Convert OpenCV data to torch compatable data type
    static std::vector<int64_t> dims = {1, img_height, img_width, 1};
    auto input_torch = torch::from_blob( _gray_image_fp32.data, dims, torch::kFloat32 ).to( torch_device );
    input_torch = input_torch.permute({0,3,1,2});
    std::vector<torch::jit::IValue> inputs_torch;
    inputs_torch.push_back( input_torch );

    // Run model
    auto outputs_torch = torch_model.forward(inputs_torch).toTuple();

    // Extract output
    torch::Tensor pts  = outputs_torch->elements()[0].toTensor().to(torch::kCPU).squeeze();
    torch::Tensor desc = outputs_torch->elements()[1].toTensor().to(torch::kCPU).squeeze();

    // TODO: Afroz please help add the postprocessing here
}


void GCNv2DetectorDescriptor::detectAndComputeONNX( cv::Mat& _gray_image_fp32, \
                                                     std::vector<cv::KeyPoint>& _keypoints, \
                                                     cv::OutputArray& _descriptors )
{
    throw std::runtime_error("detectAndComputeONNX not implement yet\n");
    // TODO: implement this
}


void GCNv2DetectorDescriptor::detectAndComputeTVM( cv::Mat& _gray_image_fp32, \
                                                     std::vector<cv::KeyPoint>& _keypoints, \
                                                     cv::OutputArray& _descriptors )
{
    throw std::runtime_error("detectAndComputeTVM not implement yet\n");
    // TODO: implement this
}


std::string GCNv2DetectorDescriptor::getDefaultName() const
{
    return (cv::FeatureDetector::getDefaultName() + ".GCNv2DetectorDescriptor");
}


} // namespace GCNv2